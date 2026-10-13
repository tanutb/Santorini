/**
 * Build the static site into dist/ (index.html plus /static/...), the layout the Python server
 * also uses. Agent and pass-and-play games run fully in the browser, so dist/ can be hosted
 * anywhere static, such as GitHub Pages. Set SANTORINI_SERVER_URL to point online games at a game server.
 */
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const web = path.join(root, 'web'), dist = path.join(root, 'dist');
const required = ['index.html', 'app.js', 'board.js', 'board-factory.js', 'three-board.js', 'island-models.js', 'engine.js', 'policy.js',
  'agent-core.js', 'tactics.js', 'agent-worker.js', 'agent-client.js', 'local-game.js', 'symmetry.js', 'learn.html', 'learn.css', 'learn.js', 'learn-data.js', 'coach.js', 'config.js', 'styles.css', 'icon.svg',
  'vendor/three.js', 'models/manifest.json'];
for (const file of required) {
  if (!(await readFile(path.join(web, file), 'utf8')).trim()) throw new Error('Empty web asset: ' + file);
}
const manifest = JSON.parse(await readFile(path.join(web, 'models/manifest.json'), 'utf8'));
for (const level of manifest.levels) {
  const expected = level.tensors.reduce((sum, tensor) => sum + tensor.shape.reduce((a, b) => a * b, 1), 0) * 4;
  const { size } = await stat(path.join(web, 'models', level.file));
  if (size !== expected) throw new Error(`models/${level.file} has ${size} bytes; the manifest expects ${expected}. Re-run scripts/export_models.py.`);
}
await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
await cp(web, path.join(dist, 'static'), { recursive: true });
await cp(path.join(web, 'index.html'), path.join(dist, 'index.html'));
await cp(path.join(web, 'learn.html'), path.join(dist, 'learn.html'));
// A static build has no game server of its own: online games need SANTORINI_SERVER_URL.
const serverUrl = process.env.SANTORINI_SERVER_URL?.trim();
if (serverUrl && !/^https?:\/\/[^\s'"]+$/.test(serverUrl)) throw new Error('SANTORINI_SERVER_URL must be an http(s) URL.');
const apiKey = process.env.SANTORINI_API_KEY?.trim() || '';
await writeFile(path.join(dist, 'static/config.js'),
  `export const SERVER_URL = ${serverUrl ? JSON.stringify(serverUrl.replace(/\/+$/, '')) : 'null'};\nexport const API_KEY = ${JSON.stringify(apiKey)};\n`);
let bytes = 0;
const walk = async dir => { for (const entry of await readdir(dir, { withFileTypes: true })) {
  const full = path.join(dir, entry.name);
  if (entry.isDirectory()) await walk(full); else bytes += (await stat(full)).size;
} };
await walk(dist);
console.log(`Built dist/ (${(bytes / 1048576).toFixed(1)} MB). Online games: ${serverUrl || 'off (set SANTORINI_SERVER_URL to enable)'}.`);
