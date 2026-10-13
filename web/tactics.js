/** Exact one-turn protection against an opponent's winning climb. */
import { ACTION_COUNT, Game } from './engine.js';

function winningClimbs(game) {
  if (game.winner !== null || game.phase !== 'move') return [];
  return game.legalActions().filter(action => {
    const worker = Math.floor((action - 25) / 25), target = (action - 25) % 25;
    return game.heights[game.workers[game.current][worker]] === 2 && game.heights[target] === 3;
  });
}

function safeBuilds(game) {
  const actor = game.current;
  return game.legalActions().filter(action => {
    const next = Game.fromDict(game.toDict());
    next.apply(action);
    return next.winner === actor || (next.winner === null && winningClimbs(next).length === 0);
  });
}

/** Preserve a safe build when choosing a move, then enforce it in the build phase.
 * If every continuation allows a winning climb, retain legal actions: no block exists.
 * This guarantees avoidable immediate climbs are blocked, not perfect multi-turn play.
 */
export function tacticalMask(game) {
  const legal = game.legalActions();
  let allowed = legal;
  if (game.phase === 'move' && legal.length) {
    const wins = winningClimbs(game);
    if (wins.length) allowed = wins;
    else {
      const safe = legal.filter(action => {
        const next = Game.fromDict(game.toDict());
        next.apply(action);
        return safeBuilds(next).length > 0;
      });
      if (safe.length) allowed = safe;
    }
  } else if (game.phase === 'build' && legal.length) {
    const safe = safeBuilds(game);
    if (safe.length) allowed = safe;
  }
  const mask = new Uint8Array(ACTION_COUNT);
  for (const action of allowed) mask[action] = 1;
  return mask;
}
