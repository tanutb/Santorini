/** Real HTTP/WebSocket integration test. Start the server first. */
import assert from 'node:assert/strict';
const base = process.env.SANTORINI_URL || process.argv[2] || 'http://127.0.0.1:8000';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function request(path, method = 'GET', body, token) {
  const response = await fetch(base + path, { method,
    headers: { 'Content-Type': 'application/json', ...(token ? {Authorization: 'Bearer ' + token} : {}) },
    body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(15000) });
  const data = await response.json();
  assert.ok(response.ok, response.status + ': ' + JSON.stringify(data));
  return data;
}
async function connect(code, token) {
  const socket = new WebSocket(base.replace(/^http/, 'ws') + '/ws/' + code);
  const states = [];
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('WebSocket initial state timed out')), 10000);
    socket.addEventListener('open', () => socket.send(JSON.stringify({ token })));
    socket.addEventListener('error', reject);
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.type === 'state') { states.push(message.room); clearTimeout(timeout); resolve(); }
    });
  });
  return { socket, states };
}
const health = await request('/api/health'); assert.equal(health.ok, true);
const html = await fetch(base + '/').then(response => response.text());
assert.ok(html.includes('<title>Santorini</title>'));
for (const asset of ['styles.css', 'app.js', 'board.js', 'board-factory.js', 'three-board.js', 'island-models.js', 'coach.js',
  'orbit.js', 'tween.js', 'feedback.js', 'vendor/three.js', 'fonts/commissioner-latin.woff2', 'fonts/marcellus-latin.woff2', 'icon.svg']) {
  const response = await fetch(base + '/static/' + asset); assert.equal(response.status, 200, asset);
}
const owner = await request('/api/rooms', 'POST', { mode: 'online', name: 'Smoke Alice' });
const code = owner.room.code;
const guest = await request('/api/rooms/' + code + '/join', 'POST', { name: 'Smoke Bob' });
const first = await connect(code, owner.token), second = await connect(code, guest.token);
let room = await request('/api/rooms/' + code, 'GET', undefined, owner.token);
const tokens = [owner.token, guest.token];
let decisions = 0;
try {
  while (room.game.winner === null && decisions < 250) {
    const action = room.legal[(decisions * 7 + 3) % room.legal.length];
    room = await request('/api/rooms/' + code + '/actions', 'POST', {action, revision: room.game.revision}, tokens[room.game.current]);
    decisions++;
  }
  assert.notEqual(room.game.winner, null);
  await pause(200);
  assert.equal(first.states.at(-1).game.revision, room.game.revision);
  assert.equal(second.states.at(-1).game.revision, room.game.revision);
  console.log('Online match completed:', decisions, 'actions; both WebSocket clients synchronized.');
} finally { first.socket.close(); second.socket.close(); }
const reconnect = await connect(code, owner.token);
assert.equal(reconnect.states.at(-1).game.winner, room.game.winner);
reconnect.socket.close();
console.log('Seat reconnect and final board restoration passed.');
const agent = await request('/api/agent');
assert.equal(agent.available, true, 'A trained checkpoint must be loaded for this test.');
const aiSession = await request('/api/rooms', 'POST', {mode:'ai', name:'Smoke human'});
room = aiSession.room; decisions = 0;
const deadline = Date.now() + 120000;
while (room.game.winner === null && Date.now() < deadline) {
  if (room.game.current === 0) {
    const action = room.legal[(decisions * 3 + 2) % room.legal.length];
    room = await request('/api/rooms/' + room.code + '/actions', 'POST', {action, revision:room.game.revision}, aiSession.token);
    decisions++;
  } else {
    await pause(150);
    room = await request('/api/rooms/' + room.code, 'GET', undefined, aiSession.token);
    assert.equal(room.ai_error, null);
  }
}
assert.notEqual(room.game.winner, null, 'AI game should finish within two minutes.');
assert.ok(room.history.some(entry => entry.player === 1));
console.log('PPO game completed:', room.game.revision, 'actions; checkpoint', agent.metadata.steps, 'training steps.');
