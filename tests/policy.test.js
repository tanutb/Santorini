import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Game } from '../web/engine.js';
import { Policy, pick, maskedSoftmax } from '../web/policy.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('../web/models/manifest.json', import.meta.url), 'utf8'));
const load = entry => {
  const bytes = readFileSync(new URL('../web/models/' + entry.file, import.meta.url));
  return new Policy(entry, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};

for (const entry of manifest.levels) {
  test(`${entry.name} network matches PyTorch on recorded positions`, () => {
    const policy = load(entry);
    let worst = 0, agree = 0;
    fixture.policies.positions.forEach((position, index) => {
      const game = Game.fromDict(position), expected = fixture.policies.outputs[entry.id][index];
      const { probabilities, value } = policy.evaluate(game.observation(), game.mask());
      for (let a = 0; a < 75; a++) worst = Math.max(worst, Math.abs(probabilities[a] - expected.probabilities[a]));
      assert.ok(Math.abs(value - expected.value) < 1e-4, `value ${value} vs ${expected.value}`);
      const best = expected.probabilities.indexOf(Math.max(...expected.probabilities));
      if (pick(probabilities, null) === best) agree++;
      for (let a = 0; a < 75; a++) if (!game.mask()[a]) assert.equal(probabilities[a], 0);
    });
    assert.ok(worst < 1e-4, `largest probability difference ${worst}`);
    assert.equal(agree, fixture.policies.positions.length, 'the top choice always matches');
  });
}

test('sampling follows the probabilities and never picks an illegal action', () => {
  const probabilities = maskedSoftmax(Float32Array.from({ length: 75 }, (_, a) => a === 3 ? 2 : 0), Uint8Array.from({ length: 75 }, (_, a) => a < 5));
  assert.equal(pick(probabilities, null), 3);
  assert.equal(pick(probabilities, 0), 0);
  assert.equal(pick(probabilities, .999999), 4);
  assert.throws(() => maskedSoftmax(new Float32Array(75), new Uint8Array(75)));
});
