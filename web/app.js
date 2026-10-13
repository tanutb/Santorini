import { demoGame } from './board.js';
import { createBoard } from './board-factory.js';
import { Feedback } from './feedback.js';
import { NEIGHBORS, analyze, coordinate, describeTransition } from './coach.js';
import { AgentClient } from './agent-client.js';
import { LocalRoom } from './local-game.js';
import { API_KEY, SERVER_URL } from './config.js';

const $ = id => document.getElementById(id);
const feedback = new Feedback();
const storage = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Storage can be disabled; play continues. */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* Storage can be disabled. */ } },
};
const MODE_LABEL = { ai: 'Against the agent', online: 'Online game', local: 'Pass and play' };
const START_LABEL = { ai: 'Play the agent', online: 'Create a room', local: 'Start the game' };
const COLOR = ['Coral', 'Gold'];
// Players choose a difficulty; the manifest's names describe training history and stay in the About dialog.
const LEVEL_LABEL = { early: 'Easy', developing: 'Medium', full: 'Hard' };
const LEVEL_ROLE = { early: 'Beginner', developing: 'Intermediate', full: 'Expert' };
const ONLINE_NOTE = 'Send them a room code';
const coarse = matchMedia('(pointer: coarse)');

let room = null, token = '', mode = 'ai', selectedWorker = -1, pending = false, hover = -1;
let socket = null, reconnectTimer = null, heartbeat = null, polling = null, retry = 0;
let toastTimer = 0, noticeTimer = 0, feedbackTimer = 0, confirmResolve = null;
let hint = null, hintPending = false, lastHistoryKey = '';
let levels = [], level = null;
// Against the agent, the seat the player takes: 0 moves first, 1 lets the agent start.
let seat = storage.get('santorini.seat') === 1 ? 1 : 0;
// Agent and pass-and-play games run in this browser; only online games use the game server.
let local = null, serverOnline = null, serverWaking = false;
const agent = new AgentClient();
const server = SERVER_URL === null ? null : SERVER_URL.replace(/\/+$/, '');

const board = createBoard($('board'), {
  onPick: pickCell,
  onHover: cell => { hover = cell; describeHover(); },
  onNotice: showNotice,
  onMode: kind => {
    $('loading-note').hidden = true;
    if (kind === 'canvas') $('hover-label').textContent = 'Showing the lightweight board because 3D graphics are unavailable.';
  },
});

// ------------------------------------------------------------------ helpers

function toast(message, error = false) {
  clearTimeout(toastTimer);
  if (error) feedback.play('error');
  $('toast').textContent = message;
  $('toast').classList.toggle('error', error);
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 6500 : 3800);
}

function showNotice(message) {
  clearTimeout(noticeTimer);
  $('board-notice').textContent = message;
  $('board-notice').hidden = false;
  noticeTimer = setTimeout(() => { $('board-notice').hidden = true; }, Math.max(1600, message.length * 70));
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(server + '/api' + path, {
      ...options, signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
        ...(token ? { Authorization: 'Bearer ' + token } : {}), ...options.headers },
    });
    const body = await response.json();
    if (!response.ok) {
      const error = new Error(typeof body.detail === 'string' ? body.detail : 'The server rejected that request. Check your input and try again.');
      error.status = response.status;
      throw error;
    }
    return body;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) throw new Error('Cannot reach the game server for online play. Check your connection and try again.');
    throw error;
  } finally { clearTimeout(timeout); }
}

const isMine = player => room && (room.mode === 'local' || room.seat === player);
const nameOf = player => {
  if (!room) return player ? 'Opponent' : 'You';
  if (room.mode !== 'local' && player === room.seat) return 'You';
  return room.players[player] || 'Your friend';
};
const coached = () => room && room.mode !== 'online';

function canAct() {
  return Boolean(room && room.ready && room.game.winner === null && !pending && isMine(room.game.current));
}

function movableWorkers() {
  if (!canAct() || room.game.phase !== 'move') return [];
  return [0, 1].filter(worker => room.legal.some(action => Math.floor((action - 25) / 25) === worker));
}

/** The selected worker, or the only worker that can move. */
function activeWorker() {
  if (selectedWorker >= 0) return selectedWorker;
  const movable = movableWorkers();
  return movable.length === 1 ? movable[0] : -1;
}

function targets() {
  if (!canAct()) return [];
  if (room.game.phase !== 'move') return room.legal;
  const worker = activeWorker();
  if (worker < 0) return [];
  return room.legal.filter(action => Math.floor((action - 25) / 25) === worker).map(action => (action - 25) % 25);
}

function selectedCell() {
  const game = room?.game;
  if (!game || !canAct()) return -1;
  if (game.phase === 'build') return game.workers[game.current][game.selected];
  const worker = activeWorker();
  return worker >= 0 ? game.workers[game.current][worker] : -1;
}

/** Rule-based warnings for the player about to act; only in coached (agent and pass-and-play) games. */
function alerts() {
  if (!coached() || !canAct()) return { wins: [], threats: [], risky: [] };
  return analyze(room.game, room.game.current, room.game.phase === 'build' ? targets() : []);
}

/** Explain why a square is not playable, using the rules rather than a generic message. */
function whyNot(cell) {
  const game = room.game, occupied = game.workers.flat().includes(cell);
  if (game.phase === 'place') return 'That square is taken. Choose an empty one.';
  if (game.phase === 'build') {
    const origin = game.workers[game.current][game.selected];
    if (!NEIGHBORS[origin].includes(cell)) return 'Build next to the worker you just moved.';
    if (occupied) return 'You cannot build on a square with a worker.';
    if (game.heights[cell] === 4) return 'That tower already has a dome.';
    return 'Choose one of the teal squares.';
  }
  const worker = activeWorker();
  if (worker < 0) return 'Choose one of your workers first.';
  const origin = game.workers[game.current][worker];
  if (!NEIGHBORS[origin].includes(cell)) return 'Workers move one square at a time, diagonals included.';
  if (occupied) return 'That square is taken.';
  if (game.heights[cell] === 4) return 'Domes cannot be entered.';
  if (game.heights[cell] > game.heights[origin] + 1) return 'Too high. Workers climb one level at a time.';
  return 'From there your worker could not build, so that move is not allowed.';
}

// ------------------------------------------------------------------ rendering

function acceptRoom(next) {
  if (!room || next.code !== room.code || next.game.revision < room.game.revision) return;
  if (next.game.revision !== room.game.revision) {
    const before = room.game;
    selectedWorker = -1;
    hint = null;
    $('hint-result').hidden = true;
    const change = describeTransition(before, next.game, next.players);
    if (change && change.kind !== 'win') feedback.play(change.kind);
    if (next.game.winner === null && next.mode !== 'local' && next.game.current === room.seat && before.current !== room.seat) feedback.play('turn');
    if (next.game.winner !== null && before.winner === null) feedback.play(next.mode === 'local' || next.game.winner === room.seat ? 'win' : 'lose');
  }
  room = next;
  render();
}

function render() {
  const playing = Boolean(room);
  // A background tab shows when an opponent has finished their turn.
  document.title = playing && room.mode !== 'local' && canAct() ? '(Your move) Santorini' : 'Santorini';
  document.body.classList.toggle('in-game', playing);
  $('lobby').hidden = playing;
  $('room-panel').hidden = !playing;
  $('history-section').hidden = !playing;
  $('practice-panel').hidden = !coached();
  if (!playing) { renderLobby(); return; }
  const game = room.game;
  $('room-mode').textContent = MODE_LABEL[room.mode];
  $('active-code').textContent = room.code;
  $('room-code-box').hidden = Boolean(local);
  $('connection-status').hidden = Boolean(local);
  $('copy-button').hidden = room.mode !== 'online';
  $('room-help').textContent = local
    ? (room.mode === 'ai' ? levelLabel(room.level) + ' difficulty. ' : '') + 'Saved on this device, so a refresh picks up where you left off.'
    : room.ready ? 'Your seat is saved in this browser. Refreshing reconnects you.' : 'Send the room code or copy the invitation link. The game starts when your friend joins.';
  $('replay-button').disabled = !room.history.length;
  renderPlayers();
  renderStatus();
  renderCoach();
  renderJournal();
  const decor = { targets: targets(), selected: selectedCell(), movable: selectedWorker < 0 && activeWorker() < 0 ? movableWorkers().map(w => game.workers[game.current][w]) : [], hint, ...alerts() };
  board.setState(game, decor);
  renderGrid(decor);
  describeHover();
}

function renderLobby() {
  board.setState(demoGame(), { targets: [] });
  renderStrength();
  renderOrder();
  // No server configured (static site) and an unreachable server read differently; both keep the agent playable.
  const waking = serverWaking && server !== null, offline = serverOnline === false && !waking;
  const online = document.querySelector('[data-mode=online]');
  // Without a server the option is unavailable; a waking or unreachable server stays selectable.
  online.setAttribute('aria-disabled', String(offline && server === null));
  const state = waking ? 'waking' : offline && server !== null ? 'unreachable' : '';
  if (state) online.dataset.state = state; else delete online.dataset.state;
  online.querySelector('.mode-note').textContent = waking ? 'Waking up the game server, about a minute'
    : !offline ? ONLINE_NOTE : server === null ? 'Needs a game server, which this site does not have' : 'The game server is not responding. Select to try again';
  $('join-form').hidden = offline || waking || mode !== 'online';
  // Agent games need no names; online games show yours to your friend; pass and play names both seats.
  $('names').hidden = mode === 'ai';
  $('second-name-field').hidden = mode !== 'local';
  $('player-name-label').textContent = mode === 'local' ? 'Coral player' : 'Your name';
  $('player-name').placeholder = mode === 'local' ? 'Coral player' : 'Shown to your friend';
  $('start-button').textContent = START_LABEL[mode];
  $('replay-button').disabled = true;
  $('result-card').hidden = true;
  renderGrid({ targets: [], selected: -1 });
}

const levelLabel = id => LEVEL_LABEL[id] || levels.find(entry => entry.id === id)?.name || 'Hard';

/** A three-level tower with the difficulty's levels built and the rest drawn faintly. */
function towerGlyph(storeys) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 36 36');
  svg.setAttribute('aria-hidden', 'true');
  for (let level = 0; level < 3; level++) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    const width = 24 - level * 5;
    Object.entries({ x: 18 - width / 2, y: 25 - level * 9, width, height: 8, rx: 1.2 }).forEach(([key, value]) => rect.setAttribute(key, value));
    if (level >= storeys) rect.setAttribute('class', 'unbuilt');
    svg.append(rect);
  }
  const ground = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  ground.setAttribute('d', 'M4 33.5h28');
  svg.append(ground);
  return svg;
}

/** Earlier checkpoints of the training run, offered as difficulty levels. */
function renderStrength() {
  const options = levels.filter(entry => entry.available);
  $('strength').hidden = mode !== 'ai' || options.length < 2;
  if ($('strength').hidden) return;
  const container = $('strength-options');
  if (container.children.length !== options.length) {
    container.replaceChildren(...options.map((entry, index) => {
      const button = document.createElement('button'), label = document.createElement('span');
      button.type = 'button';
      button.className = 'level';
      button.setAttribute('role', 'radio');
      button.dataset.level = entry.id;
      label.textContent = levelLabel(entry.id);
      button.append(towerGlyph(Math.min(3, index + 1)), label, Object.assign(document.createElement('small'), { textContent: LEVEL_ROLE[entry.id] || '' }));
      button.addEventListener('click', () => { level = entry.id; storage.set('santorini.level', level); renderStrength(); });
      return button;
    }));
  }
  for (const button of container.children) {
    const checked = button.dataset.level === level;
    button.setAttribute('aria-checked', String(checked));
    button.tabIndex = checked ? 0 : -1;
  }
  const chosen = options.find(entry => entry.id === level);
  $('strength-note').textContent = chosen ? levelNote(chosen) : '';
}

/** Who moves first against the agent; moving first is an advantage in Santorini. */
function renderOrder() {
  $('order').hidden = mode !== 'ai';
  for (const button of document.querySelectorAll('[data-seat]')) {
    const checked = Number(button.dataset.seat) === seat;
    button.setAttribute('aria-checked', String(checked));
    button.tabIndex = checked ? 0 : -1;
  }
  $('order-note').textContent = seat ? 'The agent places its workers first. You play gold.' : 'You place your workers first and play coral.';
}

/** How hard a level is, in words a player can use, from its measured results. */
function levelNote(entry) {
  const pct = value => Math.round(value * 100) + '%';
  const bot = entry.win_rates?.lookahead, hard = levels.find(level => level.id === 'full');
  const sentences = [];
  if (entry.id === 'early') {
    sentences.push('For learning the rules: it misses threats you can punish.');
    if (bot !== undefined) sentences.push(`It wins only ${pct(bot)} of games against a simple bot that looks one move ahead.`);
  } else if (entry.id === 'developing') {
    sentences.push('A real test: it looks two turns ahead, so it stops most tower races.');
    if (bot !== undefined) sentences.push(`It beats the one-move bot ${pct(bot)} of the time${hard?.versus?.developing !== undefined ? `, but Hard wins ${pct(hard.versus.developing)} of games against it` : ''}.`);
  } else {
    sentences.push(`The strongest agent, with ${Math.round(entry.steps / 1e5) / 10} million decisions of practice.`);
    const medium = entry.versus?.developing;
    if (bot !== undefined) sentences.push(`It beats the one-move bot ${pct(bot)} of the time${medium !== undefined ? ` and Medium ${pct(medium)} of the time` : ''}.`);
    sentences.push('Before each move it searches four turns ahead for wins it can force and losses it must stop.');
  }
  return sentences.join(' ');
}

/** Strengths and results ship with the exported networks (web/models/manifest.json). */
async function loadLevels() {
  try {
    const manifest = await agent.manifest();
    levels = manifest.levels.map(entry => ({ id: entry.id, name: entry.name, available: true, default: entry.id === manifest.default,
      steps: entry.metadata.steps, versus: entry.versus || {},
      win_rates: Object.fromEntries(Object.entries(entry.evaluation || {}).map(([name, result]) => [name, result.win_rate])) }));
    const saved = storage.get('santorini.level');
    const available = levels.filter(entry => entry.available);
    level = available.some(entry => entry.id === saved) ? saved : (available.find(entry => entry.default) || available.at(-1))?.id ?? null;
  } catch { levels = []; }
  if (!room) renderStrength();
}

/** How a player appears in the turn strip and the move list. */
function displayName(player) {
  if (room.mode === 'ai') return player === room.seat ? 'You' : 'Agent';
  if (room.mode === 'online' && player === room.seat) return 'You';
  return room.players[player] || (room.mode === 'online' ? 'Your friend' : COLOR[player] + ' player');
}

function renderPlayers() {
  const game = room.game;
  for (const i of [0, 1]) {
    const active = game.winner === null && room.ready && game.current === i;
    let state = COLOR[i];
    if (!room.players[i]) state = 'Waiting to join';
    else if (game.winner !== null) state = game.winner === i ? 'Winner' : COLOR[i];
    else if (room.mode === 'online' && !room.connected[i]) state = 'Disconnected, seat saved';
    else if (active && room.mode === 'ai' && i !== room.seat) state = room.ai_error ? 'Stopped' : game.phase === 'place' ? 'Placing workers' : 'Thinking…';
    else if (active) state = game.phase === 'place' ? 'Placing workers' : isMine(i) && room.mode !== 'local' ? 'Your turn' : 'To play';
    else if (room.mode === 'ai' && i !== room.seat && room.level) state = levelLabel(room.level);
    $('player-name-' + i).textContent = room.players[i] ? displayName(i) : 'Open seat';
    $('player-state-' + i).textContent = state;
    $('player-' + i).classList.toggle('is-active', active || game.winner === i);
    $('player-' + i).classList.toggle('thinking', active && room.mode === 'ai' && i !== room.seat && !room.ai_error);
  }
}

function setStatus(title, description = '') {
  $('turn-title').textContent = title;
  $('turn-description').textContent = description;
}

/** Consecutive history entries by one player: a pair of placements, or a move and its build. */
function turnsOf(history) {
  const turns = [];
  for (const entry of history) {
    const last = turns.at(-1);
    if (last && last.player === entry.player) last.entries.push(entry); else turns.push({ player: entry.player, entries: [entry] });
  }
  return turns;
}

function summarize(entries) {
  const placed = entries.map(entry => /^Placed worker \d on ([A-E][1-5])$/.exec(entry.text));
  if (placed.every(Boolean)) return placed.length === 1 ? 'placed a worker on ' + placed[0][1] : 'placed workers on ' + placed.map(match => match[1]).join(' and ');
  return entries.map(entry => entry.text.charAt(0).toLowerCase() + entry.text.slice(1)).join(', then ');
}

/** What the other player did last, so nobody has to read the move list to catch up. */
function lastTurnText() {
  const previous = turnsOf(room.history).reverse().find(turn => turn.player !== room.game.current);
  if (!previous) return '';
  const text = displayName(previous.player) + ' ' + summarize(previous.entries) + '.';
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function renderStatus() {
  const game = room.game, current = game.current;
  $('result-card').hidden = game.winner === null;
  if (!room.ready) {
    setStatus('Waiting for your friend', 'Share room code ' + room.code + '. The game starts when they join.');
    return;
  }
  if (game.winner !== null) {
    const title = displayName(game.winner) === 'You' ? 'You win' : displayName(game.winner) + ' wins';
    const detail = game.reason === 'Opponent resigned' ? displayName(1 - game.winner) + ' resigned.' : game.reason + '.';
    setStatus(title, detail);
    $('result-title').textContent = title;
    $('result-detail').textContent = detail + (room.rematch.length && !room.rematch.includes(room.seat) ? ' Your opponent wants a rematch.' : '');
    $('result-rematch').disabled = pending || room.rematch.includes(room.seat);
    $('result-rematch').textContent = room.rematch.includes(room.seat) ? 'Waiting for your friend' : 'Play again';
    return;
  }
  if (room.ai_error) { setStatus('The agent needs attention', room.ai_error); return; }
  const last = lastTurnText();
  if (!isMine(current)) {
    if (room.mode === 'ai') setStatus(game.phase === 'place' ? 'The agent is placing its workers' : 'The agent is thinking', last);
    else setStatus('Waiting for ' + displayName(current), last || 'Their move appears here as soon as they make it.');
    return;
  }
  if (pending) { setStatus('Sending your move…', last); return; }
  const who = room.mode === 'local' ? displayName(current) + ', ' : '';
  const say = text => who ? who + text.charAt(0).toLowerCase() + text.slice(1) : text;
  if (game.phase === 'place') {
    const count = game.workers[current].filter(cell => cell >= 0).length;
    setStatus(say(count ? 'Place your second worker' : 'Place your first worker'), last || 'Any empty square. Central squares leave more room to move.');
  } else if (game.phase === 'move') {
    const wins = coached() ? alerts().wins : [];
    if (wins.length) setStatus(say('You can win this move'), 'Step up onto ' + wins.map(coordinate).join(' or ') + '.');
    else if (activeWorker() < 0) setStatus(say('Choose a worker to move'), last);
    else setStatus(say('Move to a highlighted square'), last || 'Climb up one level, or step down any number.');
  } else {
    setStatus(say('Build next to your worker'), 'Add a level, or cap a third level with a dome.');
  }
}

function renderCoach() {
  if (!coached()) return;
  const game = room.game;
  const list = $('alerts');
  list.replaceChildren();
  const playing = room.ready && game.winner === null;
  $('hint-button').disabled = !playing || !canAct() || hintPending;
  $('hint-button').querySelector('span').textContent = hintPending ? 'Asking the agent…' : 'Suggest a move';
  if (!playing) return;
  // Only real, rule-derived warnings: no filler tips.
  const { wins, threats, risky } = alerts(), mover = game.current;
  const cells = values => values.map(coordinate).join(', ');
  const add = (text, kind) => { const li = document.createElement('li'); li.textContent = text; if (kind) li.className = kind; list.append(li); };
  const you = room.mode === 'local' ? displayName(mover) : 'You', rival = displayName(1 - mover);
  if (wins.length) add(you + ' can win now by stepping onto ' + cells(wins) + '.', 'win');
  if (threats.length) add(rival + ' could win next turn on ' + cells(threats) + '. A dome there blocks it.');
  if (risky.length) add('Building on ' + cells(risky) + ' would give ' + rival + ' a winning step.');
}

function renderJournal() {
  const game = room.game, turns = turnsOf(room.history);
  $('history-count').textContent = turns.length ? turns.length + (turns.length === 1 ? ' turn' : ' turns') : '';
  const key = room.code + ':' + game.revision + ':' + room.history.length + ':' + room.mode;
  if (key !== lastHistoryKey) {
    lastHistoryKey = key;
    const list = $('history-list');
    list.replaceChildren();
    turns.forEach((turn, index) => {
      const li = document.createElement('li'), number = document.createElement('span'), line = document.createElement('span'), who = document.createElement('b');
      number.className = 'turn-no';
      number.textContent = index + 1;
      who.textContent = displayName(turn.player);
      line.append(who, ' ' + summarize(turn.entries));
      li.className = 'p' + turn.player;
      li.append(number, line);
      list.prepend(li);
    });
    if (!turns.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'No moves yet. Coral places first.';
      list.append(li);
    }
  }
  $('resign-button').hidden = game.winner !== null || !room.ready;
  $('rematch-button').hidden = game.winner === null;
  $('rematch-button').disabled = pending || room.rematch.includes(room.seat);
  $('rematch-button').textContent = room.rematch.includes(room.seat) ? 'Waiting for your friend' : 'Play again';
}

function renderGrid(decor) {
  const grid = $('cell-grid');
  if (!grid.children.length) for (let i = 0; i < 25; i++) {
    const button = document.createElement('button');
    button.type = 'button';
    button.addEventListener('click', () => pickCell(i));
    grid.append(button);
  }
  const game = room ? room.game : demoGame(), legal = new Set(decor.targets || []);
  [...grid.children].forEach((button, cell) => {
    const owner = game.workers[0].includes(cell) ? 0 : game.workers[1].includes(cell) ? 1 : -1;
    const height = game.heights[cell];
    button.textContent = coordinate(cell) + ' ' + (height === 4 ? 'dome' : 'L' + height);
    button.className = [legal.has(cell) ? 'legal' : '', owner >= 0 ? 'p' + owner : '', decor.selected === cell ? 'selected' : ''].join(' ').trim();
    button.setAttribute('aria-label', coordinate(cell) + ', ' + (height === 4 ? 'dome' : 'level ' + height)
      + (owner >= 0 ? ', ' + COLOR[owner].toLowerCase() + ' worker' : '') + (legal.has(cell) ? ', playable' : ''));
    button.setAttribute('aria-pressed', String(decor.selected === cell));
  });
}

function describeHover() {
  const label = $('hover-label');
  if (hover < 0) {
    // Until the first action, remind new players that the island turns.
    label.textContent = room && room.history.length ? '' : coarse.matches ? 'Swipe sideways to turn the island' : 'Drag to turn the island';
    return;
  }
  const game = room ? room.game : demoGame(), height = game.heights[hover];
  const owner = game.workers[0].includes(hover) ? 0 : game.workers[1].includes(hover) ? 1 : -1;
  let text = coordinate(hover) + ', ' + (height === 4 ? 'dome' : height ? 'level ' + height : 'ground');
  if (owner >= 0) text += ', ' + COLOR[owner].toLowerCase() + ' worker';
  if (room && targets().includes(hover)) {
    if (game.phase === 'place') text += '. Place a worker here.';
    else if (game.phase === 'build') text += height === 3 ? '. Cap with a dome.' : '. Build level ' + (height + 1) + '.';
    else {
      const origin = game.workers[game.current][activeWorker()];
      text += height === 3 && game.heights[origin] === 2 ? '. Step up here to win.' : height > game.heights[origin] ? '. Climb here.' : '. Move here.';
    }
  }
  label.textContent = text;
}

// ------------------------------------------------------------------ actions

async function pickCell(cell) {
  feedback.unlock().catch(() => {});
  if (!room) { showNotice('Start a game to play'); return; }
  if (room.game.winner !== null) { showNotice('This game is over'); return; }
  if (!canAct()) {
    if (local && room.ai_error) { local.retry(); return; }
    if (!pending) showNotice(room.ready ? (room.mode === 'ai' ? 'The agent is still playing' : 'Wait for your turn') : 'Waiting for your friend to join');
    return;
  }
  const game = room.game;
  let action = cell;
  if (game.phase === 'move') {
    const movable = movableWorkers(), own = game.workers[game.current].indexOf(cell);
    if (own >= 0) {
      if (!movable.includes(own)) { toast('That worker is boxed in. Move your other worker.', true); return; }
      selectedWorker = selectedWorker === own && movable.length > 1 ? -1 : own;
      hint = hint && hint.worker === selectedWorker ? hint : null;
      feedback.play('select');
      render();
      return;
    }
    let worker = activeWorker();
    if (worker < 0) {
      const reachable = movable.filter(w => room.legal.includes(25 + w * 25 + cell));
      if (reachable.length !== 1) { toast(reachable.length ? 'Both workers can reach that square. Choose which one moves.' : 'Choose one of your workers first.'); return; }
      worker = reachable[0];
    }
    action = 25 + worker * 25 + cell;
  }
  if (!room.legal.includes(action)) { toast(whyNot(cell), true); return; }
  if (local) {
    try { acceptRoom(local.act(action, game.revision)); } catch (error) { toast(error.message, true); }
    return;
  }
  const code = room.code;
  pending = true;
  render();
  try { acceptRoom(await api('/rooms/' + code + '/actions', { method: 'POST', body: JSON.stringify({ action, revision: game.revision }) })); }
  catch (error) { toast(error.message, true); await refreshRoom(); }
  finally { pending = false; render(); }
}

function setConnection(text, offline = false) {
  $('connection-status').textContent = text;
  $('connection-status').classList.toggle('offline', offline);
}

function disconnect() {
  clearTimeout(reconnectTimer);
  clearInterval(heartbeat);
  clearInterval(polling);
  if (socket) { socket.onclose = null; socket.close(); socket = null; }
}

function connect() {
  disconnect();
  if (!room || local) return;
  const code = room.code;
  setConnection('Connecting');
  const base = server ? new URL(server) : location;
  const connection = new WebSocket((base.protocol === 'https:' ? 'wss://' : 'ws://') + base.host + '/ws/' + code);
  socket = connection;
  connection.onopen = () => {
    connection.send(JSON.stringify(API_KEY ? { token, key: API_KEY } : { token }));
    heartbeat = setInterval(() => { if (connection.readyState === WebSocket.OPEN) connection.send('ping'); }, 25000);
  };
  connection.onmessage = event => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === 'state' && room?.code === code) {
        retry = 0;
        clearInterval(polling);
        setConnection('Connected');
        acceptRoom(message.room);
      }
    } catch { toast('A server update could not be read. Reconnecting.', true); connection.close(); }
  };
  connection.onclose = event => {
    if (socket !== connection || !room || room.code !== code) return;
    clearInterval(heartbeat);
    if (event.code === 1008) { setConnection('Session ended', true); toast('This seat is no longer valid. Go back to the lobby to start a new game.', true); return; }
    setConnection('Reconnecting', true);
    clearInterval(polling);
    polling = setInterval(refreshRoom, 5000);
    reconnectTimer = setTimeout(connect, Math.min(1000 * 2 ** retry++, 12000));
  };
  connection.onerror = () => setConnection('Reconnecting', true);
}

async function refreshRoom() {
  if (!room || local) return;
  const code = room.code;
  try { const next = await api('/rooms/' + code); if (room?.code === code) acceptRoom(next); }
  catch { setConnection('Offline', true); }
}

/** Start or resume a game that runs in this browser. */
function openLocal(instance) {
  disconnect();
  local?.dispose();
  local = instance;
  token = '';
  room = instance.snapshot();
  selectedWorker = -1;
  hint = null;
  lastHistoryKey = '';
  $('hint-result').hidden = true;
  storage.remove('santorini.active');
  history.replaceState(null, '', location.pathname);
  if (focused()) setDrawer(false, false);
  render();
  local.scheduleAgent();
}

function saveSession(data) {
  local?.forget();
  local = null;
  token = data.token;
  room = data.room;
  selectedWorker = -1;
  hint = null;
  lastHistoryKey = '';
  $('hint-result').hidden = true;
  storage.set('santorini.session.' + room.code, token);
  storage.set('santorini.active', room.code);
  storage.set('santorini.name', $('player-name').value.trim());
  history.replaceState(null, '', location.pathname + '?room=' + room.code);
  retry = 0;
  if (focused()) setDrawer(false, false);
  render();
  connect();
}

async function startGame() {
  feedback.unlock().catch(() => {});
  if (pending) return;
  const typed = $('player-name').value.trim(), second = $('second-name').value.trim();
  storage.set('santorini.name', typed);
  if (mode === 'local') storage.set('santorini.name2', second);
  if (mode !== 'online') {
    if (mode === 'ai') agent.warm(level);
    const names = mode === 'ai' ? { name: 'You' } : { name: typed || 'Coral player', rival: second || 'Gold player' };
    openLocal(LocalRoom.create({ mode, ...names, level, seat }, { agent, storage, onChange: acceptRoom }));
    if (window.innerWidth <= 1080) $('board-frame').scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  if (serverOnline === false) { toast('Online play needs the game server, and this site does not have one. Play the agent or pass and play instead.', true); return; }
  pending = true;
  $('start-button').disabled = true;
  try {
    const body = { mode, name: typed || 'Coral player' };
    const data = await api('/rooms', { method: 'POST', body: JSON.stringify(body) });
    saveSession(data);
    if (window.innerWidth <= 1080) $('board-frame').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { toast(error.message, true); }
  finally { pending = false; $('start-button').disabled = false; render(); }
}

async function joinGame(event) {
  event.preventDefault();
  const code = $('room-code').value.trim().toUpperCase();
  if (!/^[A-Z0-9]{6}$/.test(code)) { toast('Room codes have six letters and numbers.', true); return; }
  if (serverOnline === false) { toast('Joining a room needs the game server, and this site does not have one.', true); return; }
  const button = event.submitter;
  button.disabled = true;
  try {
    const saved = storage.get('santorini.session.' + code);
    if (saved) saveSession({ room: await api('/rooms/' + code, { headers: { Authorization: 'Bearer ' + saved } }), token: saved });
    else saveSession(await api('/rooms/' + code + '/join', { method: 'POST', body: JSON.stringify({ name: $('player-name').value.trim() || 'Gold player' }) }));
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; render(); }
}

function confirmAction(title, text, label) {
  $('confirm-title').textContent = title;
  $('confirm-text').textContent = text;
  $('confirm-accept').textContent = label;
  $('confirm-dialog').showModal();
  return new Promise(resolve => { confirmResolve = resolve; });
}
function closeConfirmation(result) {
  if ($('confirm-dialog').open) $('confirm-dialog').close();
  const resolve = confirmResolve;
  confirmResolve = null;
  resolve?.(result);
}

async function leaveGame() {
  const unfinished = room?.game.winner === null && room.ready && room.history.length;
  if (unfinished && !(await confirmAction('Leave this game?', local ? 'Games on this device end when you leave.'
    : 'It stays saved. Enter room code ' + room.code + ' in this browser to return to your seat.', 'Back to lobby'))) return;
  local?.forget();
  local = null;
  disconnect();
  room = null;
  token = '';
  selectedWorker = -1;
  hint = null;
  storage.remove('santorini.active');
  history.replaceState(null, '', location.pathname);
  render();
  if (focused()) setDrawer(true, false);
}

async function resignGame() {
  if (!room || !(await confirmAction('Resign this game?', 'Your opponent wins. You can both play again afterwards.', 'Resign'))) return;
  try { acceptRoom(local ? local.resign() : await api('/rooms/' + room.code + '/resign', { method: 'POST' })); }
  catch (error) { toast(error.message, true); }
}

async function rematch() {
  if (!room || pending) return;
  pending = true;
  render();
  try { acceptRoom(local ? local.rematch() : await api('/rooms/' + room.code + '/rematch', { method: 'POST' })); }
  catch (error) { toast(error.message, true); }
  finally { pending = false; render(); }
}

async function copyInvite() {
  if (!room) return;
  const link = location.origin + location.pathname + '?room=' + room.code;
  try { await navigator.clipboard.writeText(link); toast('Invitation link copied. Send it to your friend.'); }
  catch { toast('Copying is blocked here. Share room code ' + room.code + ' instead.'); }
}

async function requestHint() {
  if (!canAct() || hintPending || !coached()) return;
  const code = room.code, revision = room.game.revision;
  hintPending = true;
  render();
  try {
    const suggestion = local ? await local.hint(revision) : await api('/rooms/' + code + '/hint', { method: 'POST', body: JSON.stringify({ revision }) });
    if (!room || room.code !== code || room.game.revision !== suggestion.revision) return;
    if (room.game.phase === 'move' && suggestion.worker !== null) selectedWorker = suggestion.worker;
    const from = room.game.phase === 'move' ? room.game.workers[room.game.current][suggestion.worker] : -1;
    hint = { cell: suggestion.cell, from, worker: suggestion.worker };
    $('hint-title').textContent = suggestion.title;
    const share = Math.round((suggestion.confidence || 0) * 100);
    $('hint-detail').textContent = suggestion.detail + (share ? ' Here the agent itself would play this ' + (share >= 99 ? 'almost every time.' : share + '% of the time.') : '');
    $('hint-result').hidden = false;
    feedback.play('select');
    // On narrow screens the coach sits below the board: bring the board back so the marked move is visible.
    if (window.innerWidth <= 1080) {
      $('board-frame').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
      showNotice('Hint: ' + suggestion.title.charAt(0).toLowerCase() + suggestion.title.slice(1));
    }
  } catch (error) { toast(error.message, true); }
  finally { hintPending = false; render(); }
}

// ------------------------------------------------------------------ full screen

const fullscreenApi = Boolean(document.documentElement.requestFullscreen && document.fullscreenEnabled);
const focused = () => document.body.classList.contains('focus');

/** The game panel slides over the island in full screen. */
function setDrawer(open, restoreFocus = true) {
  document.body.classList.toggle('drawer-open', open);
  $('panel-button').setAttribute('aria-expanded', String(open));
  if (open) $('drawer-close').focus();
  else if (restoreFocus && focused()) $('panel-button').focus();
}

function setFocusLayout(on) {
  document.body.classList.toggle('focus', on);
  $('fullscreen-button').setAttribute('aria-pressed', String(on));
  $('fullscreen-button').setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
  $('fullscreen-button').title = on ? 'Exit full screen (F)' : 'Full screen (F)';
  $('fullscreen-button').querySelector('use').setAttribute('href', on ? '#i-collapse' : '#i-expand');
  // In the lobby there is nothing to play yet, so show the options straight away.
  setDrawer(on && !room, false);
}

/** Real full screen where the browser allows it; otherwise (iPhone) the same layout inside the page. */
async function toggleFullscreen() {
  const on = !focused();
  if (fullscreenApi) {
    try {
      if (on) await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
      else if (document.fullscreenElement) await document.exitFullscreen();
    } catch { /* Refused (for example inside a frame): the in-page layout still applies. */ }
  }
  setFocusLayout(on);
}

async function showAgent() {
  $('agent-dialog').showModal();
  $('agent-stats').textContent = 'Loading the agent’s record…';
  $('evaluation-results').replaceChildren();
  try {
    const manifest = await agent.manifest();
    const strongest = manifest.levels.find(entry => entry.id === 'full') || manifest.levels.at(-1);
    const info = { metadata: strongest.metadata, evaluation: strongest.evaluation && { results: strongest.evaluation }, levels: manifest.levels.map(entry => ({
      id: entry.id, name: entry.name, available: true, steps: entry.metadata.steps, win_rates: { lookahead: entry.evaluation?.lookahead?.win_rate } })) };
    $('agent-stats').replaceChildren();
    const meta = info.metadata;
    for (const [value, label] of [[meta.episodes, 'training games'], [meta.steps, 'decisions learned from'], [meta.updates, 'PPO updates']]) {
      const stat = document.createElement('div'), number = document.createElement('b'), caption = document.createElement('span');
      number.textContent = Number(value).toLocaleString();
      caption.textContent = label;
      stat.append(number, caption);
      $('agent-stats').append(stat);
    }
    const ladder = (info.levels || []).filter(entry => entry.available);
    $('level-section').hidden = ladder.length < 2;
    $('level-results').replaceChildren(...ladder.map(entry => {
      const row = document.createElement('div'), label = document.createElement('span'), value = document.createElement('strong'), note = document.createElement('small');
      row.className = 'result-row';
      label.textContent = levelLabel(entry.id) + (LEVEL_ROLE[entry.id] ? ' (' + LEVEL_ROLE[entry.id].toLowerCase() + ')' : '');
      value.textContent = entry.win_rates?.lookahead === undefined ? 'Not evaluated' : Math.round(entry.win_rates.lookahead * 100) + '% vs lookahead';
      note.textContent = Number(entry.steps).toLocaleString() + ' training decisions';
      row.append(label, value, note);
      return row;
    }));
    if (!info.evaluation) { $('evaluation-results').textContent = 'This checkpoint has not been evaluated yet.'; return; }
    const names = { random: 'Random mover', tactical: 'Simple tactician', lookahead: 'One-move lookahead' };
    for (const [name, result] of Object.entries(info.evaluation.results)) {
      const row = document.createElement('div'), label = document.createElement('span'), value = document.createElement('strong');
      const bar = document.createElement('div'), fill = document.createElement('span'), note = document.createElement('small');
      row.className = 'result-row';
      bar.className = 'result-bar';
      label.textContent = names[name] || name;
      value.textContent = Math.round(result.win_rate * 100) + '% wins';
      fill.style.width = result.win_rate * 100 + '%';
      bar.append(fill);
      note.textContent = result.games + ' games' + (result.confidence_95 ? ', 95% interval ' + Math.round(result.confidence_95[0] * 100) + ' to ' + Math.round(result.confidence_95[1] * 100) + '%' : '');
      row.append(label, value, bar, note);
      $('evaluation-results').append(row);
    }
  } catch (error) { $('agent-stats').textContent = error.message; }
}

function updateSoundButton() {
  $('sound-toggle').setAttribute('aria-pressed', String(feedback.enabled));
  $('sound-toggle').querySelector('use').setAttribute('href', feedback.enabled ? '#i-sound' : '#i-mute');
  $('sound-toggle').querySelector('span').textContent = feedback.enabled ? 'Sound on' : 'Sound off';
}

// ------------------------------------------------------------------ wiring

function chooseMode(next) {
  mode = next;
  for (const choice of document.querySelectorAll('[data-mode]')) {
    choice.setAttribute('aria-checked', String(choice.dataset.mode === mode));
    choice.tabIndex = choice.dataset.mode === mode ? 0 : -1;
  }
  render();
}

for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => {
  if (button.getAttribute('aria-disabled') === 'true') {
    toast('Online play needs a game server, and this site does not have one. The agent and pass and play work here.');
    return;
  }
  if (button.dataset.state === 'waking') {
    toast('The game server is waking up. This takes about a minute, and online play turns on by itself.');
    return;
  }
  if (button.dataset.state === 'unreachable') {
    showNotice('Checking the game server…');
    checkServer({ wake: true }).then(ok => {
      if (ok) chooseMode('online');
      else toast('The game server is still not responding. You can play the agent or pass and play meanwhile.');
    });
    return;
  }
  chooseMode(button.dataset.mode);
});

for (const button of document.querySelectorAll('[data-seat]')) button.addEventListener('click', () => {
  seat = Number(button.dataset.seat);
  storage.set('santorini.seat', seat);
  renderOrder();
});

/** Radio groups: arrow keys move the choice, as with native radio buttons. */
for (const group of document.querySelectorAll('[role=radiogroup]')) group.addEventListener('keydown', event => {
  const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
  if (!step) return;
  const options = [...group.querySelectorAll('[role=radio]')].filter(option => option.getAttribute('aria-disabled') !== 'true');
  const index = options.indexOf(document.activeElement);
  if (index < 0) return;
  event.preventDefault();
  const next = options[(index + step + options.length) % options.length];
  next.click();
  next.focus();
});
$('start-button').addEventListener('click', startGame);
$('join-form').addEventListener('submit', joinGame);
$('player-name').addEventListener('input', () => { if (!room) render(); });
$('leave-button').addEventListener('click', leaveGame);
$('result-lobby').addEventListener('click', leaveGame);
$('copy-button').addEventListener('click', copyInvite);
$('resign-button').addEventListener('click', resignGame);
$('rematch-button').addEventListener('click', rematch);
$('result-rematch').addEventListener('click', rematch);
$('rotate-left').addEventListener('click', () => board.rotate(-1));
$('rotate-right').addEventListener('click', () => board.rotate(1));
$('reset-view').addEventListener('click', () => board.resetView());
$('zoom-in').addEventListener('click', () => board.zoomBy(.12));
$('zoom-out').addEventListener('click', () => board.zoomBy(-.12));
$('hint-button').addEventListener('click', requestHint);
$('replay-button').addEventListener('click', () => board.replayLast());
$('grid-toggle').addEventListener('click', () => {
  const panel = $('accessible-board');
  panel.hidden = !panel.hidden;
  $('grid-toggle').setAttribute('aria-expanded', String(!panel.hidden));
});
$('fullscreen-button').addEventListener('click', toggleFullscreen);
$('panel-button').addEventListener('click', () => setDrawer(!document.body.classList.contains('drawer-open')));
$('drawer-close').addEventListener('click', () => setDrawer(false));
// Leaving full screen with Esc or the browser's own controls restores the normal page.
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && focused()) setFocusLayout(false); });
$('rules-button').addEventListener('click', () => $('rules-dialog').showModal());
$('agent-button').addEventListener('click', showAgent);
for (const button of document.querySelectorAll('.dialog-close, .dialog-done')) button.addEventListener('click', () => button.closest('dialog').close());
$('confirm-accept').addEventListener('click', () => closeConfirmation(true));
$('confirm-cancel').addEventListener('click', () => closeConfirmation(false));
$('confirm-dialog').addEventListener('cancel', () => closeConfirmation(false));
$('sound-toggle').addEventListener('click', async () => {
  feedback.enabled = !feedback.enabled;
  storage.set('santorini.sound', feedback.enabled);
  updateSoundButton();
  try { await feedback.unlock(); feedback.play('select'); }
  catch (error) { feedback.enabled = false; updateSoundButton(); toast(error.message, true); }
});
$('quality-select').addEventListener('change', () => {
  board.setQuality($('quality-select').value);
  storage.set('santorini.quality', $('quality-select').value);
});
window.addEventListener('online', () => { if (local && room?.ai_error) local.retry(); else if (room) { refreshRoom(); connect(); } });
document.addEventListener('visibilitychange', () => { if (!document.hidden && room) refreshRoom(); });
document.addEventListener('keydown', event => {
  if (event.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
  if ((event.key === 'f' || event.key === 'F') && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); toggleFullscreen(); return; }
  if (event.key !== 'Escape') return;
  if (document.body.classList.contains('drawer-open')) { setDrawer(false); return; }
  selectedWorker = -1;
  hint = null;
  $('hint-result').hidden = true;
  if (room) render();
});

/** Read-only hook for automated browser checks. Holds no credentials. */
Object.defineProperty(window, 'santorini', { value: Object.freeze({ board, get room() { return room ? structuredClone(room) : null; } }) });

/** Online play is offered only when the game server answers. */
async function probe(timeout) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(server + '/api/health', { signal: controller.signal, cache: 'no-store' });
    return response.ok && (await response.json()).online === true;
  } catch { return false; } finally { clearTimeout(timer); }
}

/** With `wake`, a server that does not answer at once gets about 90 seconds: free hosts such as
 * Render put idle servers to sleep, and the first request takes about a minute to wake them. */
async function checkServer({ wake = false } = {}) {
  if (server === null) { serverOnline = false; if (!room) render(); return false; }
  serverOnline = await probe(5000);
  if (!serverOnline && wake) {
    serverWaking = true;
    if (!room) render();
    const until = Date.now() + 90000;
    while (!serverOnline && Date.now() < until) {
      await new Promise(resolve => setTimeout(resolve, 4000));
      serverOnline = await probe(15000);
    }
    serverWaking = false;
  }
  if (!room) render();
  return serverOnline;
}

async function initialize() {
  $('player-name').value = storage.get('santorini.name') || '';
  $('second-name').value = storage.get('santorini.name2') || '';
  feedback.enabled = storage.get('santorini.sound') === true;
  updateSoundButton();
  const quality = storage.get('santorini.quality');
  $('quality-select').value = ['auto', 'high', 'economy'].includes(quality) ? quality : 'auto';
  board.setQuality($('quality-select').value);
  const requested = new URLSearchParams(location.search).get('room')?.toUpperCase();
  chooseMode(mode);
  loadLevels();
  const health = checkServer({ wake: true });
  const resumed = !requested && LocalRoom.restore({ agent, storage, onChange: acceptRoom });
  if (resumed) { openLocal(resumed); return; }
  const code = requested || storage.get('santorini.active');
  if (!code || !/^[A-Z0-9]{6}$/.test(code)) return;
  if (!(await health)) {
    toast(server === null ? 'This invitation is for an online game, which needs the game server. This site does not have one.'
      : 'The game server is not responding, so this invitation cannot open yet. Try again in a minute.', true);
    return;
  }
  const savedToken = storage.get('santorini.session.' + code);
  if (!savedToken) {
    $('room-code').value = code;
    toast('You have an invitation to room ' + code + '. Enter your name, then select Join.');
    return;
  }
  try {
    token = savedToken;
    saveSession({ room: await api('/rooms/' + code), token: savedToken });
  } catch (error) {
    token = '';
    $('room-code').value = code;
    toast(error.message, true);
    if (error.status === 404 || error.status === 401) { storage.remove('santorini.session.' + code); storage.remove('santorini.active'); }
  }
}

initialize();
