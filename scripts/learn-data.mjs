/**
 * Build web/learn-data.js for the learning page from a training log and the exported networks.
 *   node scripts/learn-data.mjs checkpoints/spinup/training.jsonl [--medium-update N]
 * Writes SUMMARY (totals and network sizes), COLUMNS/ROWS (the log averaged over blocks of updates)
 * and EXAMPLE (a training game: the Hard network as the learner, coral, against Medium).
 * Run after web/models has been exported (scripts/export_models.py).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Game } from '../web/engine.js';
import { Policy, pick } from '../web/policy.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const logPath = args.find(arg => !arg.startsWith('--') && !/^\d+$/.test(arg)) || 'checkpoints/spinup/training.jsonl';
const mediumIndex = args.indexOf('--medium-update');
const mediumUpdate = mediumIndex >= 0 ? Number(args[mediumIndex + 1]) : null;

// The log, keeping the last row for each update (a resumed run may repeat some).
const byUpdate = new Map();
for (const line of readFileSync(path.resolve(root, logPath), 'utf8').split('\n')) if (line.trim()) {
  const row = JSON.parse(line);
  byUpdate.set(row.updates, row);
}
// The networks, from the exported manifest; the served Hard agent decides where the story ends.
const manifest = JSON.parse(readFileSync(path.join(root, 'web/models/manifest.json'), 'utf8'));
const served = manifest.levels.find(level => level.id === 'full').metadata;
const updates = [...byUpdate.keys()].filter(u => !served.updates || u <= served.updates).sort((a, b) => a - b), last = byUpdate.get(updates.at(-1));
const block = Math.max(1, Math.round(updates.length / 120));
const kinds = ['lookahead', 'tactical', 'random', 'snapshot', 'fixed'], means = ['entropy', 'value_loss', 'kl', 'policy_steps', 'clip_fraction', 'step_limit', 'value_steps'];
const COLUMNS = ['u', 'steps', ...kinds, ...means];
const ROWS = [];
for (let start = 0; start < updates.length; start += block) {
  const rows = updates.slice(start, start + block).map(u => byUpdate.get(u)), end = rows.at(-1);
  const row = [end.updates, end.steps];
  for (const kind of kinds) {
    const [wins, games] = rows.reduce(([w, g], r) => { const [rw, rg] = r.win_rates[kind] || [0, 0]; return [w + rw, g + rg]; }, [0, 0]);
    row.push(games ? Number((wins / games).toFixed(4)) : null);
  }
  for (const key of means) row.push(Number((rows.reduce((sum, r) => sum + (r[key] ?? 0), 0) / rows.length).toFixed(4)));
  ROWS.push(row);
}

const load = id => {
  const entry = manifest.levels.find(level => level.id === id), bytes = readFileSync(path.join(root, 'web/models', entry.file));
  return { entry, policy: new Policy(entry, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)) };
};
const hard = load('full'), medium = load('developing');
const count = prefix => hard.entry.tensors.filter(t => t.name.startsWith(prefix)).reduce((sum, t) => sum + t.shape.reduce((a, b) => a * b, 1), 0);
const separate = hard.entry.spec.architecture === 'separate';
// Totals are the served checkpoint's own, so the page agrees with the game's About dialog.
const SUMMARY = { games: served.episodes ?? last.episodes, decisions: served.steps ?? last.steps, updates: served.updates ?? last.updates, block, mediumUpdate,
  policyWeights: separate ? count('pi.') : hard.entry.parameters, valueWeights: separate ? count('v.') : null, algorithm: last.algorithm };

// An example game with a clear story: a coral win where the value network's estimate dips and recovers.
let seed = 1;
const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
function play() {
  const game = new Game(), actions = [], decisions = [];
  while (game.winner === null && game.revision < 160) {
    const mover = game.current, { probabilities, value } = (mover === 0 ? hard : medium).policy.evaluate(game.observation(), game.mask());
    const action = pick(probabilities, random());
    if (mover === 0) decisions.push({ at: actions.length, value: Number(value.toFixed(4)),
      probs: game.legalActions().map(a => [a, Number(probabilities[a].toFixed(4))]).sort((a, b) => b[1] - a[1]) });
    actions.push(action);
    game.apply(action);
  }
  return { actions, decisions, winner: game.winner, reason: game.reason };
}
function advantages(values, won, gamma = 0.99, lambda = 0.97) {
  const adv = [];
  let next = 0;
  for (let t = values.length - 1; t >= 0; t--) {
    const end = t === values.length - 1, delta = (end ? (won ? 1 : -1) : 0) + (end ? 0 : gamma * values[t + 1]) - values[t];
    next = delta + (end ? 0 : gamma * lambda * next);
    adv[t] = next;
  }
  return adv;
}
let EXAMPLE = null, fallback = null;
for (let attempt = 0; attempt < 600 && !EXAMPLE; attempt++) {
  const game = play(), n = game.decisions.length;
  if (game.winner !== 0 || n < 12 || n > 24) continue;
  fallback ??= game;
  const values = game.decisions.map(d => d.value), adv = advantages(values, true);
  const uncertain = game.decisions.filter(d => d.probs[0][1] < 0.7).length;
  if (Math.max(...values) - Math.min(...values) > 0.5 && adv.filter(a => a < -0.05).length >= 2 && uncertain >= 3) EXAMPLE = game;
}
EXAMPLE ??= fallback;

const text = `/** Data for the learning page, generated by scripts/learn-data.mjs from ${logPath.replace(/\\/g, '/')}
 * and the exported networks in web/models. Do not edit by hand. */

/** Totals of the training run and the size of the networks. */
export const SUMMARY = ${JSON.stringify(SUMMARY)};

/** The training log averaged over blocks of ${block} updates. Win rates are the learner's share of games won. */
export const COLUMNS = ${JSON.stringify(COLUMNS)};
export const ROWS = [
${ROWS.map(row => JSON.stringify(row)).join(',\n')}
];

/** A training game: Hard as the learner (coral) against Medium, both sampling their moves. For each learner
 * decision: its index in actions, the value network's estimate and the probability of every legal action. */
export const EXAMPLE = ${JSON.stringify(EXAMPLE)};
`;
writeFileSync(path.join(root, 'web/learn-data.js'), text);
console.log(`web/learn-data.js: ${ROWS.length} rows of ${block} updates, ${SUMMARY.updates} updates, example with ${EXAMPLE.decisions.length} decisions.`);
