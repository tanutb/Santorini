/**
 * The trained PPO networks, evaluated in plain JavaScript (no ML runtime).
 * Mirrors server/rl/model.py exactly: the MLP (format 1) and the residual CNN (format 2).
 * Weights come from scripts/export_models.py; tests/policy.test.js checks outputs against PyTorch.
 */
import { ACTION_COUNT, OBSERVATION_SIZE } from './engine.js';

const CELLS = 25;
/** For each 3x3 kernel tap, the (output, input) square pairs that stay on the 5x5 board. */
const TAPS = [];
for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
  const out = [], input = [];
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) {
    const sy = y + ky - 1, sx = x + kx - 1;
    if (sy >= 0 && sy < 5 && sx >= 0 && sx < 5) { out.push(y * 5 + x); input.push(sy * 5 + sx); }
  }
  TAPS.push({ out: Int32Array.from(out), input: Int32Array.from(input) });
}

function conv3x3(input, inChannels, weight, bias, outChannels) {
  const output = new Float32Array(outChannels * CELLS);
  for (let o = 0; o < outChannels; o++) {
    const base = o * CELLS;
    output.fill(bias[o], base, base + CELLS);
    for (let i = 0; i < inChannels; i++) {
      const source = i * CELLS, kernel = (o * inChannels + i) * 9;
      for (let t = 0; t < 9; t++) {
        const w = weight[kernel + t];
        if (w === 0) continue;
        const { out, input: from } = TAPS[t];
        for (let k = 0; k < out.length; k++) output[base + out[k]] += w * input[source + from[k]];
      }
    }
  }
  return output;
}

function conv1x1(input, inChannels, weight, bias, outChannels) {
  const output = new Float32Array(outChannels * CELLS);
  for (let o = 0; o < outChannels; o++) {
    const base = o * CELLS;
    output.fill(bias[o], base, base + CELLS);
    for (let i = 0; i < inChannels; i++) {
      const w = weight[o * inChannels + i], source = i * CELLS;
      for (let c = 0; c < CELLS; c++) output[base + c] += w * input[source + c];
    }
  }
  return output;
}

function linear(input, weight, bias, outFeatures) {
  const inFeatures = input.length, output = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o];
    const row = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) sum += weight[row + i] * input[i];
    output[o] = sum;
  }
  return output;
}

const relu = values => { for (let i = 0; i < values.length; i++) if (values[i] < 0) values[i] = 0; return values; };
const tanh = values => { for (let i = 0; i < values.length; i++) values[i] = Math.tanh(values[i]); return values; };

/** Softmax over legal actions only; illegal actions get exactly zero probability. */
export function maskedSoftmax(logits, mask) {
  let max = -Infinity;
  for (let a = 0; a < ACTION_COUNT; a++) if (mask[a] && logits[a] > max) max = logits[a];
  if (max === -Infinity) throw new Error('A policy decision requires at least one legal action.');
  const probabilities = new Float32Array(ACTION_COUNT);
  let total = 0;
  for (let a = 0; a < ACTION_COUNT; a++) if (mask[a]) { probabilities[a] = Math.exp(logits[a] - max); total += probabilities[a]; }
  for (let a = 0; a < ACTION_COUNT; a++) probabilities[a] /= total;
  return probabilities;
}

export class Policy {
  /** `buffer` holds float32 tensors laid out as listed in the manifest entry. */
  constructor(entry, buffer) {
    this.spec = entry.spec;
    const data = new Float32Array(buffer);
    this.tensors = {};
    for (const { name, shape, offset } of entry.tensors) this.tensors[name] = data.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    if (!['mlp', 'conv', 'separate'].includes(this.spec.architecture)) throw new Error('Unsupported model: ' + this.spec.architecture);
  }

  /** The 3x3 stem and residual blocks; `prefix` selects the network ('' shared, 'pi.' policy, 'v.' value). */
  trunk(planes, prefix, channels, blocks, record) {
    const t = this.tensors;
    let x = record('stem', relu(conv3x3(planes, 15, t[prefix + 'stem.0.weight'], t[prefix + 'stem.0.bias'], channels)), channels);
    for (let block = 0; block < blocks; block++) {
      const inner = relu(conv3x3(x, channels, t[`${prefix}tower.${block}.first.weight`], t[`${prefix}tower.${block}.first.bias`], channels));
      const residual = conv3x3(inner, channels, t[`${prefix}tower.${block}.second.weight`], t[`${prefix}tower.${block}.second.bias`], channels);
      for (let i = 0; i < residual.length; i++) residual[i] += x[i];
      x = record('block ' + (block + 1), relu(residual), channels);
    }
    return x;
  }

  /** The value head on a trunk's features: a 1x1 convolution to 4 planes, then 100 -> 128 -> 1. */
  valueHead(features, prefix, channels, record) {
    const t = this.tensors;
    const head = record('value planes', relu(conv1x1(features, channels, t[prefix + 'value.0.weight'], t[prefix + 'value.0.bias'], 4)), 4);
    const hidden = relu(linear(head, t[prefix + 'value.3.weight'], t[prefix + 'value.3.bias'], 128));
    return linear(hidden, t[prefix + 'value.5.weight'], t[prefix + 'value.5.bias'], 1)[0];
  }

  /** `trace`, if given, collects each layer's output ({ name, data, channels }) for visualisation. */
  logitsAndValue(observation, trace = null) {
    const t = this.tensors, spec = this.spec;
    const record = (name, data, channels) => { trace?.push({ name, data, channels }); return data; };
    if (spec.architecture === 'mlp') {
      const hidden = spec.hidden;
      const encoded = tanh(linear(tanh(linear(observation, t['trunk.0.weight'], t['trunk.0.bias'], hidden)), t['trunk.2.weight'], t['trunk.2.bias'], hidden));
      return { logits: linear(encoded, t['actor.weight'], t['actor.bias'], ACTION_COUNT), value: linear(encoded, t['critic.weight'], t['critic.bias'], 1)[0] };
    }
    // Planes: 9 board planes (heights, workers), 5 broadcast flags, 1 constant edge plane.
    const planes = new Float32Array(15 * CELLS);
    planes.set(observation.subarray(0, 225), 0);
    for (let f = 0; f < 5; f++) planes.fill(observation[225 + f], (9 + f) * CELLS, (10 + f) * CELLS);
    planes.fill(1, 14 * CELLS, 15 * CELLS);
    record('input', planes, 15);
    if (spec.architecture === 'conv') {
      const x = this.trunk(planes, '', spec.channels, spec.blocks, record);
      const logits = record('policy', conv1x1(x, spec.channels, t['policy.weight'], t['policy.bias'], 3), 3);
      return { logits, value: this.valueHead(x, '', spec.channels, record) };
    }
    // Separate networks: the policy pi and the value network v each have their own trunk.
    const x = this.trunk(planes, 'pi.', spec.channels, spec.blocks, record);
    const logits = record('policy', conv1x1(x, spec.channels, t['pi.policy.weight'], t['pi.policy.bias'], 3), 3);
    const valueRecord = (name, data, channels) => record(name === 'value planes' ? name : 'value ' + name, data, channels);
    const features = this.trunk(planes, 'v.', spec.value_channels, spec.value_blocks, valueRecord);
    return { logits, value: this.valueHead(features, 'v.', spec.value_channels, valueRecord) };
  }

  /** Masked action probabilities and the critic's value for the player to act. */
  evaluate(observation, mask, trace = null) {
    if (observation.length !== OBSERVATION_SIZE) throw new Error('Unexpected observation size.');
    const { logits, value } = this.logitsAndValue(observation, trace);
    return { probabilities: maskedSoftmax(logits, mask), value };
  }
}

/** Sample like the server's agent (`random` in [0, 1)), or take the most likely action. */
export function pick(probabilities, random = Math.random()) {
  let best = 0;
  for (let a = 1; a < probabilities.length; a++) if (probabilities[a] > probabilities[best]) best = a;
  if (random === null) return best;
  let cumulative = 0;
  for (let a = 0; a < probabilities.length; a++) {
    cumulative += probabilities[a];
    if (probabilities[a] > 0 && random < cumulative) return a;
  }
  return best;
}
