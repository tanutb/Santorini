/**
 * Classic Santorini rules, a line-for-line port of server/game.py.
 * Used for games that run entirely in the browser (against the agent, pass and play).
 * tests/engine.test.js replays Python-generated games to keep the two identical.
 */
export const ACTION_COUNT = 75;
export const OBSERVATION_SIZE = 230;
export const NEIGHBORS = Array.from({ length: 25 }, (_, i) => Array.from({ length: 25 }, (_, j) => j)
  .filter(j => j !== i && Math.abs(Math.floor(i / 5) - Math.floor(j / 5)) <= 1 && Math.abs(i % 5 - j % 5) <= 1));
export const coordinate = cell => String.fromCharCode(65 + cell % 5) + (Math.floor(cell / 5) + 1);

export class Game {
  constructor({ heights = Array(25).fill(0), workers = [[-1, -1], [-1, -1]], current = 0, phase = 'place', selected = -1,
    winner = null, reason = '', turn = 1, revision = 0 } = {}) {
    Object.assign(this, { heights: [...heights], workers: workers.map(pair => [...pair]), current, phase, selected, winner, reason, turn, revision });
  }

  static fromDict(data) { return new Game(data); }

  toDict() {
    const { heights, workers, current, phase, selected, winner, reason, turn, revision } = this;
    return { heights: [...heights], workers: workers.map(pair => [...pair]), current, phase, selected, winner, reason, turn, revision };
  }

  occupied() { return new Set(this.workers.flat().filter(cell => cell >= 0)); }

  /** Only actions that allow the required turn to complete. */
  legalActions() {
    if (this.winner !== null) return [];
    const occupied = this.occupied();
    if (this.phase === 'place') return Array.from({ length: 25 }, (_, i) => i).filter(i => !occupied.has(i));
    if (this.phase === 'build') {
      const origin = this.workers[this.current][this.selected];
      return NEIGHBORS[origin].filter(i => !occupied.has(i) && this.heights[i] < 4);
    }
    const result = [];
    this.workers[this.current].forEach((origin, worker) => {
      for (const target of NEIGHBORS[origin]) {
        if (occupied.has(target) || this.heights[target] === 4) continue;
        if (this.heights[target] > this.heights[origin] + 1) continue;
        const wins = this.heights[origin] === 2 && this.heights[target] === 3;
        const next = new Set(occupied);
        next.delete(origin);
        next.add(target);
        const canBuild = NEIGHBORS[target].some(i => !next.has(i) && this.heights[i] < 4);
        if (wins || canBuild) result.push(25 + worker * 25 + target);
      }
    });
    return result;
  }

  /** Validate and apply one phase action; returns the history entry. */
  apply(action) {
    if (!Number.isInteger(action) || !this.legalActions().includes(action)) throw new Error('That action is not legal in the current position.');
    const actor = this.current;
    let message;
    if (this.phase === 'place') {
      const worker = this.workers[actor].indexOf(-1);
      this.workers[actor][worker] = action;
      message = `Placed worker ${worker + 1} on ${coordinate(action)}`;
      if (!this.workers[actor].includes(-1)) {
        if (actor === 0) this.current = 1;
        else { this.current = 0; this.phase = 'move'; }
      }
    } else if (this.phase === 'move') {
      const worker = Math.floor((action - 25) / 25), target = (action - 25) % 25;
      const origin = this.workers[actor][worker];
      this.workers[actor][worker] = target;
      this.selected = worker;
      message = `Moved ${coordinate(origin)} to ${coordinate(target)}`;
      if (this.heights[origin] === 2 && this.heights[target] === 3) { this.winner = actor; this.reason = 'Reached the third level'; }
      else this.phase = 'build';
    } else {
      this.heights[action] += 1;
      const kind = this.heights[action] === 4 ? 'a dome' : `level ${this.heights[action]}`;
      message = `Built ${kind} on ${coordinate(action)}`;
      this.current = 1 - actor;
      this.phase = 'move';
      this.selected = -1;
      this.turn += 1;
    }
    this.revision += 1;
    if (this.winner === null && this.phase === 'move' && !this.legalActions().length) {
      this.winner = 1 - this.current;
      this.reason = 'Opponent has no complete legal turn';
    }
    return message;
  }

  /** The 230-value observation from the player-to-act perspective, as in training. */
  observation() {
    const features = new Float32Array(OBSERVATION_SIZE);
    for (let level = 0; level < 5; level++) for (let cell = 0; cell < 25; cell++) features[level * 25 + cell] = this.heights[cell] === level ? 1 : 0;
    let offset = 125;
    for (const player of [this.current, 1 - this.current]) for (const cell of this.workers[player]) {
      if (cell >= 0) features[offset + cell] = 1;
      offset += 25;
    }
    ['place', 'move', 'build'].forEach((phase, i) => { features[225 + i] = this.phase === phase ? 1 : 0; });
    for (let worker = 0; worker < 2; worker++) features[228 + worker] = this.selected === worker ? 1 : 0;
    return features;
  }

  mask() {
    const mask = new Uint8Array(ACTION_COUNT);
    for (const action of this.legalActions()) mask[action] = 1;
    return mask;
  }
}
