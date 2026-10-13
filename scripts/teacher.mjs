/**
 * The browser's forced-win search (web/tactics.js), served to training over standard input and output.
 * Each input line is {"depth", "budget", "games": [game dicts]}; each output line is {"masks": [...]},
 * one entry per game: the actions the search allows, or null where it allows every legal action.
 * server/rl/teacher.py starts one of these per rollout worker.
 */
import { createInterface } from 'node:readline';
import { Game } from '../web/engine.js';
import { tacticalMask } from '../web/tactics.js';

function allowed(state, depth, budget) {
  const game = Game.fromDict(state);
  if (game.winner !== null || game.phase === 'place') return null;
  const mask = tacticalMask(game, depth, budget), legal = game.legalActions();
  const actions = legal.filter(action => mask[action]);
  return actions.length === legal.length ? null : actions;
}

for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  if (!line.trim()) continue;
  const { depth, budget, games } = JSON.parse(line);
  process.stdout.write(JSON.stringify({ masks: games.map(state => allowed(state, depth, budget)) }) + '\n');
}
