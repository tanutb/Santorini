/** Web Worker wrapper so the agent's thinking never blocks the board's animation. */
import * as core from './agent-core.js';

const handlers = { move: ({ game, level, frame }) => core.move(game, level, frame), hint: ({ game }) => core.hint(game), warm: ({ level }) => core.warm(level), manifest: () => core.loadManifest() };

self.addEventListener('message', async ({ data }) => {
  try {
    self.postMessage({ id: data.id, result: await handlers[data.type](data) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
});
