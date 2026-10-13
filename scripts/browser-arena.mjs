/**
 * Seat-balanced matches for the agent as the browser plays it: a network, the forced-win search, and a random frame per game.
 *   node scripts/browser-arena.mjs full:4 climber [games=200] [seed=7] [threads=8]
 * A player is `climber` (a port of server/rl/opponents.py) or `<level>:<depth>`: a network from web/models
 * and the opponent turns searched (0 is the network alone, 1 blocks only climbs one move away).
 */
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { Game } from '../web/engine.js';
import { Policy, pick } from '../web/policy.js';
import { tacticalMask } from '../web/tactics.js';
import { orient, unorient } from '../web/symmetry.js';

const MODELS = new URL('../web/models/', import.meta.url);
const CENTER = Array.from({ length: 25 }, (_, cell) => Math.abs(Math.floor(cell / 5) - 2) + Math.abs(cell % 5 - 2));

function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The item with the largest key, comparing keys element by element like Python tuples. */
function largest(items, key) {
  let best = items[0], bestKey = key(best);
  for (const item of items.slice(1)) {
    const next = key(item), i = next.findIndex((value, k) => value !== bestKey[k]);
    if (i >= 0 && next[i] > bestKey[i]) [best, bestKey] = [item, next];
  }
  return best;
}

/** Races one worker up adjacent towers from a corner and never defends (climber_action in server/rl/opponents.py). */
function climber(game, rng) {
  const legal = game.legalActions(), mine = game.workers[game.current], heights = game.heights;
  if (game.phase === 'place') {
    const corners = [0, 4, 20, 24].filter(cell => legal.includes(cell)), pool = corners.length ? corners : legal;
    if (mine[0] < 0) return pool[Math.floor(rng() * pool.length)];
    return largest(pool, cell => [Math.abs(Math.floor(cell / 5) - Math.floor(mine[0] / 5)) + Math.abs(cell % 5 - mine[0] % 5), rng()]);
  }
  if (game.phase === 'build') {
    const height = heights[mine[game.selected]];
    return largest(legal, cell => [+(heights[cell] + 1 <= Math.min(height + 1, 3)), heights[cell] < 3 ? heights[cell] : -1, CENTER[cell], rng()]);
  }
  const win = legal.find(action => heights[mine[Math.floor((action - 25) / 25)]] === 2 && heights[(action - 25) % 25] === 3);
  if (win !== undefined) return win;
  const focused = legal.filter(action => action < 50);
  return largest(focused.length ? focused : legal, action => [heights[(action - 25) % 25], CENTER[(action - 25) % 25], rng()]);
}

async function playGames({ players, indices, seed }) {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', MODELS), 'utf8')), policies = new Map();
  async function policy(level) {
    if (!policies.has(level)) {
      const entry = manifest.levels.find(item => item.id === level);
      if (!entry) throw new Error('Unknown level: ' + level);
      const bytes = await readFile(new URL(entry.file, MODELS));
      policies.set(level, new Policy(entry, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
    }
    return policies.get(level);
  }
  const rows = [];
  for (const index of indices) {
    const rng = random(seed * 1_000_003 + index), seat = index % 2;
    const specs = { [seat]: players[0], [1 - seat]: players[1] }, frames = [Math.floor(rng() * 8), Math.floor(rng() * 8)];
    const game = new Game();
    let decisions = 0, time = 0, slowest = 0;
    while (game.winner === null && game.revision < 400) {
      const mover = game.current, spec = specs[mover];
      if (spec === 'climber') { game.apply(climber(game, rng)); continue; }
      const [level, depth] = spec.split(':'), start = performance.now();
      const seen = Game.fromDict(orient(game.toDict(), frames[mover]));
      const mask = Number(depth) > 0 ? tacticalMask(seen, Number(depth)) : seen.mask();
      const { probabilities } = (await policy(level)).evaluate(seen.observation(), mask);
      game.apply(unorient(pick(probabilities, rng()), frames[mover]));
      if (mover === seat) {
        const elapsed = performance.now() - start;
        decisions++;
        time += elapsed;
        slowest = Math.max(slowest, elapsed);
      }
    }
    rows.push({ seat, winner: game.winner, decisions, time, slowest });
  }
  return rows;
}

if (isMainThread) {
  const [first, second, games = '200', seed = '7', threads = '8'] = process.argv.slice(2);
  const count = Number(games), workers = Math.min(Number(threads), count);
  if (!first || !second || count < 2 || count % 2) throw new Error('Usage: node scripts/browser-arena.mjs <player> <player> [even games] [seed] [threads]');
  const rows = (await Promise.all(Array.from({ length: workers }, (_, k) => new Promise((resolve, reject) => {
    const indices = Array.from({ length: count }, (_, i) => i).filter(i => i % workers === k);
    const worker = new Worker(new URL(import.meta.url), { workerData: { players: [first, second], indices, seed: Number(seed) } });
    worker.on('message', resolve);
    worker.on('error', reject);
  })))).flat();
  const seats = [0, 1].map(seat => `${rows.filter(row => row.seat === seat && row.winner === seat).length}/${rows.filter(row => row.seat === seat).length}`);
  const wins = rows.filter(row => row.winner === row.seat).length, decisions = rows.reduce((sum, row) => sum + row.decisions, 0);
  console.log(JSON.stringify({ players: [first, second], games: count, wins, losses: rows.filter(row => row.winner === 1 - row.seat).length,
    win_rate: wins / count, moving_first: seats[0], moving_second: seats[1],
    mean_decision_ms: decisions ? +(rows.reduce((sum, row) => sum + row.time, 0) / decisions).toFixed(1) : null,
    slowest_decision_ms: +Math.max(...rows.map(row => row.slowest)).toFixed(0) }));
} else {
  parentPort.postMessage(await playGames(workerData));
}
