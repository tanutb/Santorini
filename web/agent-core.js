/** Loads the exported networks and answers move and hint requests. Runs in a worker or, as a fallback, on the page. */
import { Game } from './engine.js';
import { Policy, pick } from './policy.js';
import { tacticalMask } from './tactics.js';
import { explainAction } from './coach.js';
import { orient, unorient } from './symmetry.js';

const MODELS = new URL('./models/', import.meta.url);
let manifest = null;
const policies = new Map();

export async function loadManifest() {
  if (!manifest) {
    const response = await fetch(new URL('manifest.json', MODELS), { cache: 'no-cache' });
    if (!response.ok) throw new Error('The agent’s model list could not be downloaded.');
    manifest = await response.json();
  }
  return manifest;
}

async function policyFor(level) {
  const { levels, default: fallback } = await loadManifest();
  const entry = levels.find(item => item.id === level) || levels.find(item => item.id === fallback);
  if (!policies.has(entry.id)) {
    policies.set(entry.id, (async () => {
      const url = new URL(entry.file, MODELS);
      if (entry.sha256) url.searchParams.set('v', entry.sha256);
      const response = await fetch(url);
      if (!response.ok) throw new Error('The agent’s network could not be downloaded.');
      return new Policy(entry, await response.arrayBuffer());
    })());
    policies.get(entry.id).catch(() => policies.delete(entry.id));
  }
  return policies.get(entry.id);
}

/** The agent's move, sampled from its policy while it reads the board through the game's frame (see symmetry.js). */
export async function move(state, level, frame = 0) {
  const game = Game.fromDict(orient(state, frame)), policy = await policyFor(level);
  const { probabilities } = policy.evaluate(game.observation(), tacticalMask(game));
  return { action: unorient(pick(probabilities), frame), revision: game.revision };
}

/** The strongest network's top tactically allowed choice and its conditional probability. */
export async function hint(state) {
  const game = Game.fromDict(state), policy = await policyFor('full');
  const { probabilities } = policy.evaluate(game.observation(), tacticalMask(game));
  const action = pick(probabilities, null);
  return { action, revision: game.revision, confidence: probabilities[action], ...explainAction(game, action) };
}

export async function warm(level) { await policyFor(level); return true; }
