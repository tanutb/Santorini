/** Talks to the agent worker; falls back to running the agent on the page if module workers are unavailable. */
export class AgentClient {
  constructor() {
    this.pending = new Map();
    this.next = 1;
    try {
      this.worker = new Worker(new URL('./agent-worker.js', import.meta.url), { type: 'module' });
      this.worker.addEventListener('message', ({ data }) => {
        const request = this.pending.get(data.id);
        if (!request) return;
        this.pending.delete(data.id);
        if (data.error) request.reject(new Error(data.error)); else request.resolve(data.result);
      });
      this.worker.addEventListener('error', () => this.fallBack());
    } catch {
      this.fallBack();
    }
  }

  fallBack() {
    this.worker?.terminate();
    this.worker = null;
    const waiting = [...this.pending.values()];
    this.pending.clear();
    for (const request of waiting) this.inline(request.type, request.payload).then(request.resolve, request.reject);
  }

  async inline(type, payload) {
    this.core ??= import('./agent-core.js');
    const core = await this.core;
    return { move: () => core.move(payload.game, payload.level, payload.frame), hint: () => core.hint(payload.game),
      warm: () => core.warm(payload.level), manifest: () => core.loadManifest() }[type]();
  }

  request(type, payload = {}) {
    if (!this.worker) return this.inline(type, payload);
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, { resolve, reject, type, payload });
      this.worker.postMessage({ id, type, ...payload });
    });
  }

  move(game, level, frame) { return this.request('move', { game, level, frame }); }
  hint(game) { return this.request('hint', { game }); }
  warm(level) { return this.request('warm', { level }).catch(() => false); }
  manifest() { return this.request('manifest'); }
}
