import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Game } from '../web/engine.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'));

test('the browser rules engine replays every Python game identically', () => {
  let steps = 0;
  for (const { steps: recorded } of fixture.games) {
    const game = new Game();
    for (const step of recorded) {
      assert.deepEqual(game.legalActions(), step.legal, `legal actions at revision ${game.revision}`);
      assert.equal(Array.from(game.observation(), value => (value ? '1' : '0')).join(''), step.observation, `observation at revision ${game.revision}`);
      assert.equal(game.apply(step.action), step.message);
      assert.deepEqual(game.toDict(), step.after);
      steps++;
    }
  }
  assert.ok(steps > 2000 && fixture.games.some(g => g.steps.at(-1).after.reason === 'Reached the third level'));
});

test('illegal actions are rejected without changing the game', () => {
  const game = new Game();
  game.apply(0);
  const before = game.toDict();
  for (const action of [0, -1, 75, 1.5, '3', null]) assert.throws(() => game.apply(action));
  assert.deepEqual(game.toDict(), before);
  assert.equal(Game.fromDict(before).toDict().revision, 1);
});
