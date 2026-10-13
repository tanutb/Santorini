import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Game } from '../web/engine.js';
import { rankTurns, tacticalMask } from '../web/tactics.js';
import { orient } from '../web/symmetry.js';

const allowed = (game, ...search) => Array.from(tacticalMask(game, ...search), (value, action) => value ? action : -1).filter(a => a >= 0);
const copy = game => Game.fromDict(game.toDict());
function threatened() {
  const game = new Game({ workers: [[12, 1], [23, 24]], phase: 'move' });
  game.heights[24] = 2;
  game.heights[18] = 3;
  return game;
}
/** A reported game: the opponent ladders A1 and B1 from the corner while the agent races E4 and D5 a tempo behind. */
function ladder() {
  const game = new Game({ workers: [[24, 0], [17, 23]], current: 1, phase: 'move', turn: 8 });
  Object.assign(game.heights, { 0: 2, 1: 2, 19: 2, 23: 1 });
  return game;
}
function replyWins(game) {
  return game.legalActions().some(action => {
    const next = copy(game), actor = next.current;
    next.apply(action);
    return next.winner === actor;
  });
}

test('every allowed move preserves a block and every allowed build executes one in all frames', () => {
  for (let frame = 0; frame < 8; frame++) {
    const game = Game.fromDict(orient(threatened().toDict(), frame));
    const before = game.toDict();
    assert.ok(allowed(game).length);
    assert.ok(allowed(game).length < game.legalActions().length);
    for (const action of allowed(game)) {
      const moved = copy(game);
      moved.apply(action);
      for (const build of allowed(moved)) {
        const built = copy(moved);
        built.apply(build);
        assert.ok(built.winner === game.current || !replyWins(built));
      }
    }
    assert.deepEqual(game.toDict(), before, 'search must not mutate the live game');
  }
});

test('takes an immediate win even when the opponent also threatens to win', () => {
  const game = threatened();
  game.heights[12] = 2;
  game.heights[7] = 3;
  assert.deepEqual(allowed(game), [32, 43]);
});

test('domes the threatened square and avoids building a new winning square', () => {
  const game = threatened();
  game.apply(38); // Worker one moves from 12 to 13, beside the threat on 18.
  assert.deepEqual(allowed(game), [18]);
  game.heights[18] = 2;
  assert.ok(!allowed(game).includes(18), 'do not create a level-three target for the opponent');
  assert.ok(allowed(game).length);
});

test('unavoidable double threats keep legal actions available', () => {
  const game = threatened();
  game.apply(38);
  game.heights[19] = 3;
  assert.deepEqual(allowed(game), game.legalActions());
  assert.ok(game.legalActions().every(action => {
    const next = copy(game);
    next.apply(action);
    return replyWins(next);
  }));
});

test('sends a worker to stop a corner ladder two turns before it wins', () => {
  // Racing on (D5 to E4) loses: A1 reaches level three next turn and nothing can dome it in time.
  assert.ok(allowed(ladder(), 1).includes(69));
  assert.deepEqual(allowed(ladder(), 2), [36, 37]);
  assert.deepEqual(allowed(ladder()), [36, 37]); // C4 to B3 or C3, one step from doming A1.
});

test('plays a double threat the opponent cannot answer', () => {
  const game = new Game({ workers: [[8, 20], [11, 22]], current: 1, phase: 'move' });
  Object.assign(game.heights, { 11: 1, 12: 2, 13: 2, 7: 3 });
  const { turns } = rankTurns(game);
  assert.equal(Math.max(...turns.map(turn => turn.score)), 98, 'a win on the next turn');
  assert.deepEqual(allowed(game), [37]);
  game.apply(37);
  assert.deepEqual(allowed(game), [13], 'D3 to level three beside C2: two threats, one dome');
});

test('a move keeps its best build when the node budget cuts the search short', () => {
  for (const budget of [500, 5000, 50000]) {
    for (let frame = 0; frame < 8; frame++) {
      const game = Game.fromDict(orient(ladder().toDict(), frame));
      const { turns } = rankTurns(game, 4, budget), best = Math.max(...turns.map(turn => turn.score));
      for (const move of allowed(game, 4, budget)) {
        const moved = copy(game);
        moved.apply(move);
        const builds = allowed(moved, 4, budget);
        assert.ok(builds.length);
        for (const build of builds) assert.ok(turns.some(turn => turn.move === move && turn.build === build && turn.score === best));
      }
    }
  }
});

test('placement and finished games preserve their legal masks', () => {
  const game = new Game();
  assert.deepEqual(tacticalMask(game), game.mask());
  game.winner = 0;
  assert.deepEqual(tacticalMask(game), game.mask());
});

test('real network move and hint requests both use the tactical mask', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async url => {
    const local = new URL(url);
    local.search = '';
    const bytes = await readFile(local);
    return { ok: true, json: async () => JSON.parse(bytes.toString()),
      arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
  };
  try {
    const { move, hint } = await import('../web/agent-core.js');
    for (let frame = 0; frame < 8; frame++) {
      const game = threatened();
      const decision = await move(game.toDict(), 'full', frame);
      assert.ok(allowed(game).includes(decision.action));
      game.apply(decision.action);
      assert.equal((await move(game.toDict(), 'full', frame)).action, 18);
      assert.equal((await hint(game.toDict())).action, 18);
    }
    for (let frame = 0; frame < 8; frame++) {
      // The ladder continues A1 to B1, building A1 to level three; the agent must dome it.
      const game = ladder();
      while (game.current === 1) game.apply((await move(game.toDict(), 'full', frame)).action);
      game.apply(51);
      game.apply(0);
      while (game.current === 1) game.apply((await move(game.toDict(), 'full', frame)).action);
      assert.equal(game.heights[0], 4);
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});
