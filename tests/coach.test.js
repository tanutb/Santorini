import test from 'node:test';
import assert from 'node:assert/strict';
import { NEIGHBORS, analyze, coordinate, describeTransition, detectChange, winningSteps } from '../web/coach.js';

const empty = () => ({ heights: Array(25).fill(0), workers: [[0, 4], [20, 24]], current: 0, phase: 'move', selected: -1, winner: null, reason: '', revision: 10 });

test('neighbours match the 8-connected 5x5 grid', () => {
  assert.deepEqual(NEIGHBORS[0], [1, 5, 6]);
  assert.equal(NEIGHBORS[12].length, 8);
  assert.equal(coordinate(0), 'A1');
  assert.equal(coordinate(24), 'E5');
});

test('winning steps need a level-two worker beside a free level-three square', () => {
  const game = empty();
  game.heights[0] = 2; game.heights[1] = 3; game.heights[5] = 3;
  game.workers[1][0] = 5;
  assert.deepEqual(winningSteps(game, 0), [1]);
  game.heights[0] = 1;
  assert.deepEqual(winningSteps(game, 0), []);
});

test('analysis reports wins, threats and risky builds for the player to act', () => {
  const game = empty();
  game.heights[24] = 2; game.heights[23] = 3;
  game.heights[0] = 2; game.heights[1] = 3;
  const move = analyze(game, 0);
  assert.deepEqual(move.wins, [1]);
  assert.deepEqual(move.threats, [23]);
  const build = { ...empty(), phase: 'build', selected: 0 };
  build.heights[24] = 2; build.heights[18] = 2;
  assert.deepEqual(analyze(build, 0, [18, 1]).risky, [18], 'raising 18 to level three next to a level-two rival is risky');
  assert.deepEqual(analyze({ ...empty(), phase: 'place' }, 0), { wins: [], threats: [], risky: [] });
});

test('transitions are detected and described in plain words', () => {
  const before = empty(), after = structuredClone(before);
  after.workers[0][0] = 6; after.heights[6] = 1; after.revision = 11; before.heights[6] = 1;
  assert.deepEqual(detectChange(before, after), { type: 'move', player: 0, worker: 0, from: 0, cell: 6 });
  assert.deepEqual(describeTransition(before, after, ['You', 'PPO agent']),
    { kind: 'move', title: 'You moved to B2', detail: 'From A1, climbing to level 1.' });
  const built = structuredClone(after); built.heights[7] = 4; built.revision = 12;
  after.heights[7] = 3;
  assert.equal(describeTransition(after, built, ['You', 'PPO agent']).kind, 'dome');
  assert.equal(describeTransition(before, built, ['You', 'PPO agent']), null, 'skipped revisions are not narrated');
  const won = structuredClone(after); won.winner = 1; won.reason = 'Opponent resigned'; won.revision = 12;
  assert.deepEqual(describeTransition(after, won, ['You', 'PPO agent']), { kind: 'win', title: 'PPO agent wins', detail: 'You resigned.' });
});
