/** Syntax-check every browser module and script without executing it. */
import { readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const files = [];
for (const dir of ['web', 'scripts', 'tests']) {
  for (const name of await readdir(path.join(root, dir))) if (/\.(m?js)$/.test(name)) files.push(path.join(dir, name));
}
for (const file of files) execFileSync(process.execPath, ['--check', path.join(root, file)], { stdio: 'inherit' });
console.log(`Checked ${files.length} files.`);
