/**
 * An exact forced-win search over complete turns (a move and its build) that filters the network's choices.
 *
 * Every turn the agent could take is ranked by searching all replies: a turn that forces a win ranks
 * highest, the fastest first; otherwise a turn ranks by how many of the opponent's turns it survives
 * against best play. The network then chooses among the best-ranked turns, so it keeps its own plan but
 * cannot ignore a climb it could still stop, such as a tower race it is a tempo behind in.
 * The search deepens one opponent turn at a time and keeps the deepest pass that fits its node budget,
 * so every decision takes a bounded time.
 */
import { ACTION_COUNT, NEIGHBORS } from './engine.js';

/** Opponent turns searched at each difficulty; Easy only blocks a climb that is one move away. */
export const SEARCH_DEPTH = { early: 1, developing: 2, full: 4 };
export const NODE_BUDGET = 3_000_000;
const WIN = 100;
const OUT_OF_BUDGET = Symbol('out of budget');
let nodes = 0, limit = 0;

// The search state: heights, and workers as [player 0's two, player 1's two]. Each turn is undone after it is searched.
const occupied = (w, cell) => w[0] === cell || w[1] === cell || w[2] === cell || w[3] === cell;
const canStep = (h, w, from, to) => h[to] < 4 && h[to] <= h[from] + 1 && !occupied(w, to);

function canClimb(h, w, p) {
  for (let i = 2 * p; i < 2 * p + 2; i++) {
    if (h[w[i]] !== 2) continue;
    for (const to of NEIGHBORS[w[i]]) if (h[to] === 3 && !occupied(w, to)) return true;
  }
  return false;
}

/** A worker can always build on the square it left, so a player is stuck only when no worker can move. */
function stuck(h, w, p) {
  for (let i = 2 * p; i < 2 * p + 2; i++) for (const to of NEIGHBORS[w[i]]) if (canStep(h, w, w[i], to)) return false;
  return true;
}

/** One turn removes at most two squares (one moved onto, one built on), so three reachable squares mean no stalemate. */
function mayBeStalemated(h, w, p) {
  let first = -1, second = -1;
  for (let i = 2 * p; i < 2 * p + 2; i++) for (const to of NEIGHBORS[w[i]]) {
    if (to === first || to === second || !canStep(h, w, w[i], to)) continue;
    if (first < 0) first = to; else if (second < 0) second = to; else return false;
  }
  return true;
}

/** Plays each complete turn of p that does not climb to win, until `visit` returns true. */
function someTurn(h, w, p, visit) {
  for (let i = 2 * p; i < 2 * p + 2; i++) {
    const from = w[i];
    for (const to of NEIGHBORS[from]) {
      if (!canStep(h, w, from, to)) continue;
      w[i] = to;
      for (const build of NEIGHBORS[to]) {
        if (h[build] >= 4 || occupied(w, build)) continue;
        h[build]++;
        const found = visit();
        h[build]--;
        if (found) { w[i] = from; return true; }
      }
      w[i] = from;
    }
  }
  return false;
}

/** Can p, to move, force a win within `depth` of its own turns? */
function wins(h, w, p, depth) {
  if (++nodes > limit) throw OUT_OF_BUDGET;
  if (canClimb(h, w, p)) return true;
  if (depth === 1 && !mayBeStalemated(h, w, 1 - p)) return false;
  return someTurn(h, w, p, () => loses(h, w, 1 - p, depth));
}

/** After the other player's turn, must q, to move, lose within that player's next `depth - 1` turns? */
function loses(h, w, q, depth) {
  if (++nodes > limit) throw OUT_OF_BUDGET;
  if (stuck(h, w, q)) return true;
  if (depth === 1 || canClimb(h, w, q)) return false;
  const p = 1 - q;
  return !someTurn(h, w, q, () => stuck(h, w, p) || !wins(h, w, p, depth - 1));
}

/**
 * Every complete turn from a move or build phase, scored: WIN - n for a forced win in n turns
 * (this one included), otherwise the number of opponent turns it survives, up to the depth searched.
 */
export function rankTurns(game, depth = SEARCH_DEPTH.full, budget = NODE_BUDGET) {
  const me = game.current, you = 1 - me;
  const h = Int8Array.from(game.heights), w = Int8Array.from(game.workers.flat());
  const turns = [];
  if (game.phase === 'move') {
    for (const move of game.legalActions()) {
      const index = 2 * me + Math.floor((move - 25) / 25), to = (move - 25) % 25;
      if (h[w[index]] === 2 && h[to] === 3) { turns.push({ move, build: -1, index, to, score: WIN - 1 }); continue; }
      const moved = Int8Array.from(w);
      moved[index] = to;
      for (const build of NEIGHBORS[to]) if (h[build] < 4 && !occupied(moved, build)) turns.push({ move, build, index, to, score: null });
    }
  } else if (game.phase === 'build') {
    const index = 2 * me + game.selected;
    for (const build of game.legalActions()) turns.push({ move: -1, build, index, to: w[index], score: null });
  }
  let open = turns.filter(turn => turn.score === null), searched = 0;
  nodes = 0;
  limit = budget;
  for (let d = 1; d <= depth && open.length; d++) {
    const scores = new Map();
    try {
      for (const turn of open) {
        const from = w[turn.index];
        w[turn.index] = turn.to;
        h[turn.build]++;
        const score = loses(h, w, you, d) ? WIN - d : wins(h, w, you, d) ? d - 1 : null;
        h[turn.build]--;
        w[turn.index] = from;
        if (score !== null) scores.set(turn, score);
      }
    } catch (error) {
      if (error !== OUT_OF_BUDGET) throw error;
      break;
    }
    for (const [turn, score] of scores) turn.score = score;
    open = open.filter(turn => turn.score === null);
    searched = d;
  }
  for (const turn of open) turn.score = searched;
  return { turns, depth: searched };
}

/**
 * The legal actions that keep the best-ranked turn available: a move that has one of those builds,
 * or one of those builds. When every turn loses at once, all legal actions stay allowed.
 */
export function tacticalMask(game, depth = SEARCH_DEPTH.full, budget = NODE_BUDGET) {
  const mask = new Uint8Array(ACTION_COUNT);
  const legal = game.winner === null ? game.legalActions() : [];
  if (game.phase === 'place' || !legal.length) {
    for (const action of legal) mask[action] = 1;
    return mask;
  }
  const { turns } = rankTurns(game, depth, budget);
  const best = Math.max(...turns.map(turn => turn.score));
  for (const turn of turns) if (turn.score === best) mask[game.phase === 'move' ? turn.move : turn.build] = 1;
  return mask;
}
