/**
 * Games against the agent and pass-and-play games, run entirely in this browser.
 * Produces the same room snapshot shape as the server, so the interface treats both alike,
 * and saves itself after every action so a refresh resumes the game.
 */
import { Game } from './engine.js';
import { randomFrame } from './symmetry.js';

const KEY = 'santorini.local';
const AGENT_DELAY = 600;
const NAMES = { ai: 'Agent', local: 'Gold player' };

export class LocalRoom {
  constructor(state, { agent, onChange, storage }) {
    this.state = state;
    this.agent = agent;
    this.onChange = onChange;
    this.storage = storage;
    this.timer = 0;
    this.thinking = false;
    this.error = null;
  }

  /** Against the agent, `seat` 1 lets the agent place and move first. */
  static create({ mode, name, rival, level, seat = 0 }, services) {
    seat = mode === 'ai' && seat === 1 ? 1 : 0;
    const players = [name, rival || NAMES[mode]];
    const state = { version: 1, mode, seat, players: seat ? players.reverse() : players, game: new Game().toDict(), history: [],
      level: mode === 'ai' ? level : null, frame: randomFrame() };
    const room = new LocalRoom(state, services);
    room.save();
    return room;
  }

  /** The saved game from a previous visit, if it is still readable. */
  static restore(services) {
    const state = services.storage.get(KEY);
    if (!state || state.version !== 1 || !['ai', 'local'].includes(state.mode)) return null;
    try { Game.fromDict(state.game).legalActions(); } catch { return null; }
    return new LocalRoom(state, services);
  }

  get code() { return 'LOCAL'; }

  /** The human's seat; games saved before turn order was a choice have the human first. */
  get seat() { return this.state.seat === 1 ? 1 : 0; }

  snapshot() {
    const game = Game.fromDict(this.state.game);
    return { code: this.code, local: true, mode: this.state.mode, seat: this.seat, players: [...this.state.players], connected: [true, true],
      game: game.toDict(), legal: game.legalActions(), history: this.state.history.slice(-100), rematch: [],
      ai_error: this.error, level: this.state.level, ready: true };
  }

  save() { this.storage.set(KEY, this.state); }

  forget() {
    this.dispose();
    this.storage.remove(KEY);
  }

  dispose() {
    clearTimeout(this.timer);
    this.disposed = true;
  }

  apply(action) {
    const game = Game.fromDict(this.state.game), actor = game.current;
    const text = game.apply(action);
    this.state.game = game.toDict();
    this.state.history.push({ player: actor, text, revision: game.revision });
    this.save();
  }

  /** A human action, checked the same way the server checks it. */
  act(action, revision) {
    const game = this.state.game;
    if (revision !== game.revision) throw new Error('The board changed. Choose again.');
    if (game.winner !== null) throw new Error('This game has finished.');
    if (this.state.mode === 'ai' && game.current !== this.seat) throw new Error('It is the agent’s turn.');
    this.apply(action);
    this.scheduleAgent();
    return this.snapshot();
  }

  resign() {
    const game = this.state.game;
    if (game.winner !== null) throw new Error('This game has already finished.');
    const resigning = this.state.mode === 'local' ? game.current : this.seat;
    Object.assign(game, { winner: 1 - resigning, reason: 'Opponent resigned', revision: game.revision + 1 });
    clearTimeout(this.timer);
    this.save();
    return this.snapshot();
  }

  rematch() {
    if (this.state.game.winner === null) throw new Error('Finish the current game before starting another.');
    // Revisions keep increasing so late agent replies from the old game are ignored.
    this.state.game = new Game({ revision: this.state.game.revision + 1 }).toDict();
    this.state.history = [];
    this.state.frame = randomFrame();
    this.error = null;
    this.save();
    this.scheduleAgent();
    return this.snapshot();
  }

  async hint(revision) {
    if (revision !== this.state.game.revision) throw new Error('The position changed. Ask again for the current board.');
    const suggestion = await this.agent.hint(this.state.game);
    if (suggestion.revision !== this.state.game.revision) throw new Error('The position changed while the agent was thinking.');
    return suggestion;
  }

  /** Play the agent's move and build, pausing so each animation reads clearly. */
  scheduleAgent() {
    clearTimeout(this.timer);
    const game = this.state.game;
    if (this.disposed || this.state.mode !== 'ai' || game.winner !== null || game.current === this.seat || this.thinking) return;
    this.timer = setTimeout(async () => {
      const revision = this.state.game.revision;
      this.thinking = true;
      try {
        const { action } = await this.agent.move(this.state.game, this.state.level, this.state.frame);
        if (this.disposed || this.state.game.revision !== revision) return;
        this.error = null;
        this.apply(action);
      } catch (error) {
        this.error = 'The agent could not load (' + error.message + '). Check your connection, then click the board to try again.';
      } finally {
        this.thinking = false;
      }
      if (this.disposed) return;
      this.onChange(this.snapshot());
      if (!this.error) this.scheduleAgent();
    }, AGENT_DELAY);
  }

  retry() {
    this.error = null;
    this.onChange(this.snapshot());
    this.scheduleAgent();
  }
}
