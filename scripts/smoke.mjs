/**
 * Real HTTP/WebSocket check of a running game server (local or deployed). Start the server first.
 *   node scripts/smoke.mjs https://your-server.example.com      (SANTORINI_API_KEY=... if the server requires a key)
 */
import assert from 'node:assert/strict';

const base = (process.env.SANTORINI_URL || process.argv[2] || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const key = process.env.SANTORINI_API_KEY || '';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function request(path, method = 'GET', body, token) {
  const response = await fetch(base + path, { method,
    headers: { 'Content-Type': 'application/json', ...(key ? { 'X-API-Key': key } : {}), ...(token ? { Authorization: 'Bearer ' + token } : {}) },
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
    socket.addEventListener('open', () => socket.send(JSON.stringify(key ? { token, key } : { token })));
    socket.addEventListener('error', reject);
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.type === 'state') { states.push(message.room); clearTimeout(timeout); resolve(); }
    });
  });
  return { socket, states };
}

const health = await request('/api/health');
assert.equal(health.ok, true);
assert.equal(health.online, true, 'this server must host online rooms');
const page = await fetch(base + '/');
assert.ok((await page.text()).includes('<title>Santorini</title>'));
assert.match(page.headers.get('content-security-policy') || '', /script-src 'self'/);
assert.equal(page.headers.get('x-content-type-options'), 'nosniff');
for (const asset of ['styles.css', 'app.js', 'engine.js', 'policy.js', 'agent-worker.js', 'local-game.js', 'config.js',
  'three-board.js', 'vendor/three.js', 'models/manifest.json', 'models/full.bin', 'fonts/commissioner-latin.woff2']) {
  const response = await fetch(base + '/static/' + asset);
  assert.equal(response.status, 200, asset);
}
console.log('Website, security headers and the in-browser agent files are served.');

const owner = await request('/api/rooms', 'POST', { name: 'Smoke Alice' });
const code = owner.room.code;
const guest = await request('/api/rooms/' + code + '/join', 'POST', { name: 'Smoke Bob' });
const first = await connect(code, owner.token), second = await connect(code, guest.token);
let room = await request('/api/rooms/' + code, 'GET', undefined, owner.token);
const tokens = [owner.token, guest.token];
let decisions = 0;
try {
  while (room.game.winner === null && decisions < 250) {
    const action = room.legal[(decisions * 7 + 3) % room.legal.length];
    room = await request('/api/rooms/' + code + '/actions', 'POST', { action, revision: room.game.revision }, tokens[room.game.current]);
    decisions++;
  }
  assert.notEqual(room.game.winner, null);
  await pause(300);
  assert.equal(first.states.at(-1).game.revision, room.game.revision);
  assert.equal(second.states.at(-1).game.revision, room.game.revision);
  console.log('Online match completed:', decisions, 'actions; both WebSocket clients synchronized.');
} finally { first.socket.close(); second.socket.close(); }
const reconnect = await connect(code, owner.token);
assert.equal(reconnect.states.at(-1).game.winner, room.game.winner);
reconnect.socket.close();
console.log('Seat reconnect and final board restoration passed.');
