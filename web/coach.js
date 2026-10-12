/** Rule-derived facts for feedback. Mirrors server/game.py; the server stays authoritative. */
export const NEIGHBORS = Array.from({ length: 25 }, (_, i) => Array.from({ length: 25 }, (_, j) => j)
  .filter(j => j !== i && Math.abs(Math.floor(i / 5) - Math.floor(j / 5)) <= 1 && Math.abs(i % 5 - j % 5) <= 1));

export const coordinate = cell => String.fromCharCode(65 + cell % 5) + (Math.floor(cell / 5) + 1);
const occupiedCells = game => new Set(game.workers.flat().filter(cell => cell >= 0));

/** Cells a player could step onto from level two to win, ignoring whose turn it is. */
export function winningSteps(game, player) {
  const occupied = occupiedCells(game), cells = new Set();
  for (const origin of game.workers[player]) {
    if (origin < 0 || game.heights[origin] !== 2) continue;
    for (const target of NEIGHBORS[origin]) if (game.heights[target] === 3 && !occupied.has(target)) cells.add(target);
  }
  return [...cells].sort((a, b) => a - b);
}

/**
 * What the player in `seat` should know before acting.
 * wins: squares they can win on right now (move phase only).
 * threats: squares the opponent could win on next turn.
 * risky: build targets that would create a new winning step for the opponent.
 */
export function analyze(game, seat, buildTargets = []) {
  if (!game || game.winner !== null || game.phase === 'place') return { wins: [], threats: [], risky: [] };
  const opponent = 1 - seat;
  const wins = game.phase === 'move' && game.current === seat ? winningSteps(game, seat) : [];
  const threats = winningSteps(game, opponent);
  const risky = [];
  if (game.phase === 'build' && game.current === seat) {
    const occupied = occupiedCells(game);
    for (const cell of buildTargets) {
      if (game.heights[cell] !== 2 || occupied.has(cell)) continue;
      if (game.workers[opponent].some(worker => worker >= 0 && game.heights[worker] === 2 && NEIGHBORS[worker].includes(cell))) risky.push(cell);
    }
  }
  return { wins, threats, risky };
}

/** The single change between consecutive revisions, used for animation and announcements. */
export function detectChange(before, after) {
  if (!before || !after) return null;
  for (let player = 0; player < 2; player++) for (let worker = 0; worker < 2; worker++) {
    const from = before.workers[player][worker], cell = after.workers[player][worker];
    if (from !== cell && cell >= 0) return { type: from < 0 ? 'place' : 'move', player, worker, from, cell };
  }
  for (let cell = 0; cell < 25; cell++) {
    if (after.heights[cell] > before.heights[cell]) {
      return { type: after.heights[cell] === 4 ? 'dome' : 'build', player: before.current, cell, level: after.heights[cell] };
    }
  }
  return null;
}

/** A short, factual sentence about the last action. */
export function describeTransition(before, after, players) {
  if (!before || !after || after.revision !== before.revision + 1) return null;
  const name = player => players[player] || 'Player ' + (player + 1);
  if (after.winner !== null && after.winner !== before.winner) {
    const resigned = after.reason === 'Opponent resigned';
    return { kind: 'win', title: name(after.winner) + ' wins', detail: resigned ? name(1 - after.winner) + ' resigned.' : after.reason + '.' };
  }
  const change = detectChange(before, after);
  if (!change) return null;
  if (change.type === 'place') return { kind: 'place', title: name(change.player) + ' placed a worker', detail: 'On ' + coordinate(change.cell) + '.' };
  if (change.type === 'move') {
    const from = before.heights[change.from], to = after.heights[change.cell];
    const level = height => (height ? 'level ' + height : 'the ground');
    const how = to > from ? 'climbing to ' : to < from ? 'stepping down to ' : 'staying on ';
    return { kind: 'move', title: name(change.player) + ' moved to ' + coordinate(change.cell), detail: 'From ' + coordinate(change.from) + ', ' + how + level(to) + '.' };
  }
  if (change.type === 'dome') return { kind: 'dome', title: name(change.player) + ' capped a tower', detail: coordinate(change.cell) + ' is closed for good.' };
  return { kind: 'build', title: name(change.player) + ' built level ' + change.level, detail: 'On ' + coordinate(change.cell) + '.' };
}
