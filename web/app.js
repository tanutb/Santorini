import { demoGame } from './board.js';
import { createBoard } from './board-factory.js';
import { Feedback } from './feedback.js';
import { NEIGHBORS, analyze, coordinate, describeTransition } from './coach.js';

const $ = id => document.getElementById(id);
const feedback = new Feedback();
const storage = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Storage can be disabled; play continues. */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* Storage can be disabled. */ } },
};
const MODE_LABEL = { ai: 'Against the agent', online: 'Online game', local: 'Pass and play' };
const COLOR = ['Coral', 'Gold'];

let room = null, token = '', mode = 'ai', selectedWorker = -1, pending = false, hover = -1;
let socket = null, reconnectTimer = null, heartbeat = null, polling = null, retry = 0;
let toastTimer = 0, noticeTimer = 0, feedbackTimer = 0, confirmResolve = null;
let hint = null, hintPending = false, lastHistoryKey = '';
let levels = [], level = null;

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
    const response = await fetch('/api' + path, {
      ...options, signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: 'Bearer ' + token } : {}), ...options.headers },
    });
    const body = await response.json();
    if (!response.ok) {
      const error = new Error(typeof body.detail === 'string' ? body.detail : 'The server rejected that request. Check your input and try again.');
      error.status = response.status;
      throw error;
    }
    return body;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) throw new Error('Cannot reach the game server. Check your connection and try again.');
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
    showActionFeedback(describeTransition(before, next.game, next.players.map((_, i) => room.mode === 'local' ? next.players[i] : (i === room.seat ? 'You' : next.players[i]))), before.current);
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
  $('copy-button').hidden = room.mode !== 'online';
  $('room-help').textContent = room.mode === 'online'
    ? (room.ready ? 'Your seat is saved in this browser. Refreshing reconnects you.' : 'Send the room code or copy the invitation link. The game starts when your friend joins.')
    : 'Saved automatically. Return in this browser to continue.';
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
  for (const i of [0, 1]) $('player-' + i).classList.remove('is-active');
  $('player-name-0').textContent = $('player-name').value.trim() || 'You';
  $('player-name-1').textContent = { ai: 'PPO agent', online: 'Your friend', local: 'Second player' }[mode];
  $('player-state-0').textContent = 'Coral workers, moves first';
  $('player-state-1').textContent = 'Gold workers';
  setStatus('Choose how to play', 'Pick an opponent in the panel. Drag to turn the island.', -1);
  setPhase(null);
  $('result-card').hidden = true;
  renderGrid({ targets: [], selected: -1 });
}

const levelName = id => levels.find(entry => entry.id === id)?.name || 'Full strength';

/** Earlier checkpoints of the training run, described by what they were measured to do. */
function renderStrength() {
  const options = levels.filter(entry => entry.available);
  $('strength').hidden = mode !== 'ai' || options.length < 2;
  if ($('strength').hidden) return;
  const container = $('strength-options');
  if (container.children.length !== options.length) {
    container.replaceChildren(...options.map(entry => {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('role', 'radio');
      button.dataset.level = entry.id;
      button.textContent = entry.name;
      button.addEventListener('click', () => { level = entry.id; storage.set('santorini.level', level); renderStrength(); });
      return button;
    }));
  }
  for (const button of container.children) button.setAttribute('aria-checked', String(button.dataset.level === level));
  const chosen = options.find(entry => entry.id === level);
  const rate = chosen?.win_rates?.lookahead;
  $('strength-note').textContent = chosen
    ? 'Trained on ' + Math.round(chosen.steps / 1e5) / 10 + ' million decisions' + (rate === undefined ? '.' : '. Beats the one-move lookahead player in ' + Math.round(rate * 100) + '% of games.')
    : '';
}

async function loadLevels() {
  try {
    levels = (await api('/agent')).levels || [];
    const saved = storage.get('santorini.level');
    const available = levels.filter(entry => entry.available);
    level = available.some(entry => entry.id === saved) ? saved : (available.find(entry => entry.default) || available.at(-1))?.id ?? null;
  } catch { levels = []; }
  if (!room) renderStrength();
}

function renderPlayers() {
  const game = room.game;
  for (const i of [0, 1]) {
    let state = COLOR[i] + ' workers';
    if (!room.players[i]) state = 'Waiting to join';
    else if (game.winner !== null) state = game.winner === i ? 'Winner' : COLOR[i] + ' workers';
    else if (room.mode === 'online' && !room.connected[i]) state = 'Disconnected; their seat is saved';
    else if (room.ready && game.current === i) state = room.mode === 'ai' && i === 1 ? 'Thinking' : game.phase === 'place' ? 'Placing workers' : 'To move';
    else if (room.mode === 'ai' && i === 1 && room.level) state = COLOR[1] + ' workers, ' + levelName(room.level).toLowerCase();
    $('player-name-' + i).textContent = room.players[i] ? (room.mode !== 'local' && i === room.seat ? room.players[i] + ' (you)' : room.players[i]) : 'Open seat';
    $('player-state-' + i).textContent = state;
    $('player-' + i).classList.toggle('is-active', game.winner === null && room.ready && game.current === i);
  }
}

function setStatus(title, description, player, thinking = false, win = false) {
  $('turn-title').textContent = title;
  $('turn-description').textContent = description;
  const piece = $('status-piece');
  if (player >= 0) piece.dataset.player = player; else delete piece.dataset.player;
  piece.classList.toggle('thinking', thinking);
  $('status-card').classList.toggle('win', win);
}

function setPhase(phase) {
  const order = ['place', 'move', 'build'];
  for (const step of $('phase-steps').children) {
    const index = order.indexOf(step.dataset.phase), current = order.indexOf(phase);
    step.classList.toggle('active', index === current);
    step.classList.toggle('done', phase !== null && (phase === 'place' ? false : index === 0 || (phase === 'build' && index === 1)));
    if (index === current) step.setAttribute('aria-current', 'step'); else step.removeAttribute('aria-current');
  }
  $('phase-steps').hidden = phase === null;
}

function renderStatus() {
  const game = room.game, current = game.current, name = nameOf(current);
  const owner = room.mode === 'local' ? room.players[current] + ': ' : '';
  $('result-card').hidden = game.winner === null;
  if (!room.ready) {
    setStatus('Waiting for your friend', 'Share room code ' + room.code + '. The game starts when they join.', 0);
    setPhase(null);
    return;
  }
  if (game.winner !== null) {
    const winner = nameOf(game.winner);
    const title = room.mode === 'local' ? winner + ' wins' : game.winner === room.seat ? 'You win' : winner + ' wins';
    const detail = game.reason === 'Opponent resigned' ? nameOf(1 - game.winner) + ' resigned.' : game.reason + '.';
    setStatus(title, detail, game.winner, false, true);
    setPhase(null);
    $('result-title').textContent = title;
    $('result-detail').textContent = detail + (room.rematch.length && !room.rematch.includes(room.seat) ? ' Your opponent wants a rematch.' : '');
    $('result-rematch').disabled = pending || room.rematch.includes(room.seat);
    $('result-rematch').textContent = room.rematch.includes(room.seat) ? 'Waiting for your friend' : 'Play again';
    return;
  }
  setPhase(game.phase);
  if (room.ai_error) { setStatus('The agent needs attention', room.ai_error, 1); return; }
  if (!isMine(current)) {
    const thinking = room.mode === 'ai';
    setStatus(thinking ? 'The agent is thinking' : name + '’s turn', thinking ? 'Its move and build will animate on the island.' : 'Their move will appear here as soon as they make it.', current, thinking);
    return;
  }
  if (pending) { setStatus('Sending your move', 'One moment.', current); return; }
  if (game.phase === 'place') {
    const count = game.workers[current].filter(cell => cell >= 0).length;
    setStatus(owner + (count ? 'Place your second worker' : 'Place your first worker'), 'Pick any empty square. Central squares leave more room to move.', current);
  } else if (game.phase === 'move') {
    if (activeWorker() < 0) setStatus(owner + 'Choose a worker to move', 'Rings mark the workers that can move.', current);
    else setStatus(owner + 'Move to a teal square', 'Climb up one level, or step down any number.', current);
  } else {
    setStatus(owner + 'Build next to your worker', 'Add a level, or cap a third level with a dome.', current);
  }
}

function renderCoach() {
  if (!coached()) return;
  const game = room.game, insight = room.insight;
  // Scored at the start of each turn; it stays until the next turn begins.
  const showInsight = insight && insight.revision >= game.revision - 1 && room.ready && game.phase !== 'place' && game.winner === null;
  $('insight').hidden = !showInsight;
  if (showInsight) {
    // Only clear leads are called: in calibration, mid-range estimates were too noisy to quote as numbers.
    const names = room.mode === 'ai' ? ['You', 'The agent'] : room.players;
    const lead = insight.p0 >= .75 ? 0 : insight.p0 <= .25 ? 1 : -1;
    $('insight-value').textContent = lead < 0 ? 'Roughly even' : names[lead] + (lead === 0 && room.mode === 'ai' ? ' are ahead' : ' is ahead');
    $('insight-note').textContent = 'From the agent’s value network, calibrated on its own games. It only calls clear leads.';
  }
  const list = $('alerts');
  list.replaceChildren();
  if (!room.ready || game.winner !== null) return;
  const { wins, threats, risky } = alerts(), mover = game.current;
  const cells = values => values.map(coordinate).join(', ');
  const add = (text, kind) => { const li = document.createElement('li'); li.textContent = text; if (kind) li.className = kind; list.append(li); };
  const you = room.mode === 'local' ? room.players[mover] : 'You', rival = nameOf(1 - mover);
  if (wins.length) add(you + ' can win now by stepping onto ' + cells(wins) + '.', 'win');
  if (threats.length) add(rival + ' could win next turn on ' + cells(threats) + '. A dome there blocks it.');
  if (risky.length) add('Building on ' + cells(risky) + ' would give ' + rival + ' a winning step.');
  if (!list.children.length && canAct()) {
    add(game.phase === 'place' ? 'Workers near the centre have more squares to move and build on.'
      : game.phase === 'move' ? 'A worker on level two next to a level-three tower threatens to win.'
        : 'Domes close a square for good. Use them to cut off a climb.', 'tip');
  }
  $('hint-button').disabled = !canAct() || hintPending || !room.ready;
  $('hint-button').querySelector('span').textContent = hintPending ? 'Asking the agent…' : 'Suggest a move';
  $('replay-button').disabled = !room.history.length;
}

function renderJournal() {
  const game = room.game;
  $('history-count').textContent = room.history.length ? room.history.length + (room.history.length === 1 ? ' action' : ' actions') : '';
  const key = room.code + ':' + game.revision + ':' + room.history.length;
  if (key !== lastHistoryKey) {
    lastHistoryKey = key;
    const list = $('history-list');
    list.replaceChildren();
    for (const entry of room.history.slice(-40).reverse()) {
      const li = document.createElement('li');
      li.textContent = (room.mode === 'local' ? room.players[entry.player] : nameOf(entry.player)) + ': ' + entry.text.charAt(0).toLowerCase() + entry.text.slice(1);
      li.className = 'p' + entry.player;
      list.append(li);
    }
    if (!room.history.length) {
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
    label.textContent = matchMedia('(pointer: coarse)').matches ? 'Swipe sideways to turn the island.'
      : room ? 'Drag to turn the island. Ctrl + scroll zooms.' : 'Drag to turn the island.';
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

function showActionFeedback(change, player) {
  if (!change) return;
  feedback.play(change.kind === 'win' ? 'select' : change.kind);
  clearTimeout(feedbackTimer);
  $('action-title').textContent = change.title;
  $('action-detail').textContent = change.detail;
  $('action-icon').dataset.player = player;
  $('action-feedback').hidden = change.kind === 'win';
  feedbackTimer = setTimeout(() => { $('action-feedback').hidden = true; }, 3200);
}

// ------------------------------------------------------------------ actions

async function pickCell(cell) {
  feedback.unlock().catch(() => {});
  if (!room) { showNotice('Start a game to play'); return; }
  if (room.game.winner !== null) { showNotice('This game is over'); return; }
  if (!canAct()) {
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
  if (!room) return;
  const code = room.code;
  setConnection('Connecting');
  const connection = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/' + code);
  socket = connection;
  connection.onopen = () => {
    connection.send(JSON.stringify({ token }));
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
  if (!room) return;
  const code = room.code;
  try { const next = await api('/rooms/' + code); if (room?.code === code) acceptRoom(next); }
  catch { setConnection('Offline', true); }
}

function saveSession(data) {
  token = data.token;
  room = data.room;
  selectedWorker = -1;
  hint = null;
  lastHistoryKey = '';
  $('hint-result').hidden = true;
  $('action-feedback').hidden = true;
  storage.set('santorini.session.' + room.code, token);
  storage.set('santorini.active', room.code);
  storage.set('santorini.name', $('player-name').value.trim());
  history.replaceState(null, '', '/?room=' + room.code);
  retry = 0;
  render();
  connect();
}

async function startGame() {
  feedback.unlock().catch(() => {});
  if (pending) return;
  pending = true;
  $('start-button').disabled = true;
  try {
    const body = { mode, name: $('player-name').value.trim() || 'Island builder', ...(mode === 'ai' && level ? { level } : {}) };
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
  const button = event.submitter;
  button.disabled = true;
  try {
    const saved = storage.get('santorini.session.' + code);
    if (saved) saveSession({ room: await api('/rooms/' + code, { headers: { Authorization: 'Bearer ' + saved } }), token: saved });
    else saveSession(await api('/rooms/' + code + '/join', { method: 'POST', body: JSON.stringify({ name: $('player-name').value.trim() || 'Guest' }) }));
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
  if (room?.game.winner === null && room.ready && !(await confirmAction('Leave this game?', 'It stays saved. Enter room code ' + room.code + ' in this browser to return to your seat.', 'Back to lobby'))) return;
  disconnect();
  room = null;
  token = '';
  selectedWorker = -1;
  hint = null;
  storage.remove('santorini.active');
  history.replaceState(null, '', '/');
  $('action-feedback').hidden = true;
  render();
}

async function resignGame() {
  if (!room || !(await confirmAction('Resign this game?', 'Your opponent wins. You can both play again afterwards.', 'Resign'))) return;
  try { acceptRoom(await api('/rooms/' + room.code + '/resign', { method: 'POST' })); }
  catch (error) { toast(error.message, true); }
}

async function rematch() {
  if (!room || pending) return;
  pending = true;
  render();
  try { acceptRoom(await api('/rooms/' + room.code + '/rematch', { method: 'POST' })); }
  catch (error) { toast(error.message, true); }
  finally { pending = false; render(); }
}

async function copyInvite() {
  if (!room) return;
  const link = location.origin + '/?room=' + room.code;
  try { await navigator.clipboard.writeText(link); toast('Invitation link copied. Send it to your friend.'); }
  catch { toast('Copying is blocked here. Share room code ' + room.code + ' instead.'); }
}

async function requestHint() {
  if (!canAct() || hintPending || !coached()) return;
  const code = room.code, revision = room.game.revision;
  hintPending = true;
  render();
  try {
    const suggestion = await api('/rooms/' + code + '/hint', { method: 'POST', body: JSON.stringify({ revision }) });
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

async function showAgent() {
  $('agent-dialog').showModal();
  $('agent-stats').textContent = 'Loading the agent’s record…';
  $('evaluation-results').replaceChildren();
  try {
    const info = await api('/agent');
    $('agent-stats').replaceChildren();
    if (!info.available) { $('agent-stats').textContent = 'No trained checkpoint is loaded. See the README to train one.'; return; }
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
      label.textContent = entry.name;
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
  $('sound-toggle').title = feedback.enabled ? 'Sound on' : 'Sound off';
}

// ------------------------------------------------------------------ wiring

for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => {
  mode = button.dataset.mode;
  for (const choice of document.querySelectorAll('[data-mode]')) choice.setAttribute('aria-checked', String(choice === button));
  render();
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
window.addEventListener('online', () => { if (room) { refreshRoom(); connect(); } });
document.addEventListener('visibilitychange', () => { if (!document.hidden && room) refreshRoom(); });
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape' || event.target.matches('input, select, textarea') || document.querySelector('dialog[open]')) return;
  selectedWorker = -1;
  hint = null;
  $('hint-result').hidden = true;
  if (room) render();
});

/** Read-only hook for automated browser checks. Holds no credentials. */
Object.defineProperty(window, 'santorini', { value: Object.freeze({ board, get room() { return room ? structuredClone(room) : null; } }) });

async function initialize() {
  $('player-name').value = storage.get('santorini.name') || '';
  feedback.enabled = storage.get('santorini.sound') === true;
  updateSoundButton();
  const quality = storage.get('santorini.quality');
  $('quality-select').value = ['auto', 'high', 'economy'].includes(quality) ? quality : 'auto';
  board.setQuality($('quality-select').value);
  const requested = new URLSearchParams(location.search).get('room')?.toUpperCase();
  const code = requested || storage.get('santorini.active');
  render();
  loadLevels();
  if (!code || !/^[A-Z0-9]{6}$/.test(code)) return;
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
