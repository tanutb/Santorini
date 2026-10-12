/** A tiny tween scheduler. The renderer only runs frames while a tween is active. */
export const easing = {
  linear: t => t,
  outCubic: t => 1 - (1 - t) ** 3,
  inOutCubic: t => (t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
  outBack: t => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2,
  outBounce(t) {
    const n = 7.5625, d = 2.75;
    if (t < 1 / d) return n * t * t;
    if (t < 2 / d) return n * (t -= 1.5 / d) * t + .75;
    if (t < 2.5 / d) return n * (t -= 2.25 / d) * t + .9375;
    return n * (t -= 2.625 / d) * t + .984375;
  },
};

export class Tweens {
  constructor(clock = () => performance.now()) {
    this.clock = clock;
    this.active = new Set();
  }

  /** Start a tween; `update` receives eased progress in [0, 1]. Returns a cancel function. */
  add({ duration, delay = 0, ease = easing.outCubic, update, done }) {
    const tween = { start: this.clock() + delay, duration: Math.max(1, duration), ease, update, done };
    this.active.add(tween);
    return () => this.active.delete(tween);
  }

  /** Advance every tween to `now`; returns true while any tween still needs frames. */
  step(now = this.clock()) {
    for (const tween of [...this.active]) {
      if (now < tween.start) continue;
      const progress = Math.min(1, (now - tween.start) / tween.duration);
      tween.update?.(tween.ease(progress), progress);
      if (progress === 1) {
        this.active.delete(tween);
        tween.done?.();
      }
    }
    return this.active.size > 0;
  }

  /** Jump every tween to its end state, e.g. when the user prefers reduced motion. */
  finish() {
    for (const tween of [...this.active]) {
      this.active.delete(tween);
      tween.update?.(1, 1);
      tween.done?.();
    }
  }

  get busy() { return this.active.size > 0; }
}
