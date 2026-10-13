import test from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../web/engine.js';
import { LocalRoom } from '../web/local-game.js';

const memory = () => {
  const values = new Map();
  return { get: key => values.get(key) ?? null, set: (key, value) => values.set(key, structuredClone(value)), remove: key => values.delete(key) };
};
/** Plays the first legal action at once, so a test sees each agent decision. */
const firstLegal = { move: async state => ({ action: Game.fromDict(state).legalActions()[0], revision: state.revision }) };
const waitFor = room => new Promise(resolve => { room.onChange = resolve; });

test('the agent places first when the player chooses to move second', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const room = LocalRoom.create({ mode: 'ai', name: 'You', level: 'full', seat: 1 }, { agent: firstLegal, storage: memory(), onChange() {} });
  const snapshot = room.snapshot();
  assert.equal(snapshot.seat, 1);
  assert.deepEqual(snapshot.players, ['Agent', 'You']);
  assert.throws(() => room.act(0, snapshot.game.revision), /agent’s turn/);
  for (let placed = 0; placed < 2; placed++) {
    const changed = waitFor(room);
    room.scheduleAgent();
    t.mock.timers.tick(600);
    await changed;
  }
  const game = room.snapshot().game;
  assert.equal(game.current, 1);
  assert.equal(game.workers[0].filter(cell => cell >= 0).length, 2);
  room.act(Game.fromDict(game).legalActions()[0], game.revision);
  assert.equal(room.resign().game.winner, 0, 'resigning from the second seat hands the win to the agent');
  room.dispose();
});

test('the player moves first by default, and older saved games keep that seat', () => {
  const storage = memory();
  const room = LocalRoom.create({ mode: 'ai', name: 'You', level: 'full' }, { agent: firstLegal, storage, onChange() {} });
  assert.equal(room.snapshot().seat, 0);
  assert.deepEqual(room.snapshot().players, ['You', 'Agent']);
  const saved = storage.get('santorini.local');
  delete saved.seat;
  storage.set('santorini.local', saved);
  assert.equal(LocalRoom.restore({ agent: firstLegal, storage, onChange() {} }).snapshot().seat, 0);
  assert.equal(LocalRoom.create({ mode: 'local', name: 'A', rival: 'B', seat: 1 }, { agent: firstLegal, storage, onChange() {} }).snapshot().seat, 0,
    'pass and play has no agent seat');
  room.dispose();
});
