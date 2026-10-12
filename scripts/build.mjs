import { cp, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
for (const file of ['index.html', 'app.js', 'board.js', 'board-factory.js', 'three-board.js', 'island-models.js', 'styles.css', 'icon.svg', 'vendor/three.js']) {
  const contents = await readFile(path.join(root, 'web', file), 'utf8');
  if (!contents.trim()) throw new Error('Empty web asset: ' + file);
}
await mkdir(path.join(root, 'dist'), { recursive: true });
await cp(path.join(root, 'web'), path.join(root, 'dist', 'static'), { recursive: true });
await cp(path.join(root, 'web/index.html'), path.join(root, 'dist/index.html'));
console.log('Built dist/. The website requires the FastAPI server for gameplay.');
