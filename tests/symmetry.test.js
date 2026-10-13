import test from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../web/engine.js';
import { FRAMES, orient, unorient } from '../web/symmetry.js';

test('the eight frames are distinct permutations of the board that keep neighbours together', () => {
  assert.equal(new Set(FRAMES.map(map => map.join())).size, 8);
  for (const map of FRAMES) {
    assert.deepEqual([...map].sort((a, b) => a - b), Array.from({ length: 25 }, (_, i) => i));
    assert.equal(map[12], 12, 'the centre stays put');
  }
});

test('legal actions seen through a frame map back to exactly the legal actions on the real board', () => {
  const game = new Game();
  for (const action of [12, 7, 6, 18]) game.apply(action);
  game.heights[13] = 1; game.heights[8] = 2; game.heights[11] = 3; game.heights[2] = 4;
  for (const phase of ['move', 'build']) {
    const position = Game.fromDict(game.toDict());
    if (phase === 'build') position.apply(position.legalActions()[0]);
    const legal = position.legalActions().sort((a, b) => a - b);
    for (let frame = 0; frame < FRAMES.length; frame++) {
      const seen = Game.fromDict(orient(position.toDict(), frame));
      assert.deepEqual(seen.legalActions().map(action => unorient(action, frame)).sort((a, b) => a - b), legal, `${phase}, frame ${frame}`);
    }
  }
});

test('a missing or unknown frame leaves the board as it is', () => {
  const state = new Game({ workers: [[3, 9], [-1, -1]] }).toDict();
  for (const frame of [undefined, 0, 9, -1, 'x']) {
    assert.deepEqual(orient(state, frame), state);
    assert.equal(unorient(42, frame), 42);
  }
});
