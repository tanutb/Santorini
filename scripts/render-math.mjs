/**
 * Render the equations in web/learn.html as native MathML, so the page needs no maths library
 * and works under the server's strict Content-Security-Policy.
 *
 * Write new formulas as $$...$$ (display) or \(...\) (inline), or edit the TeX kept in an existing
 * formula's <annotation>, then run:
 *   npm install --no-save temml@0.10.34
 *   node scripts/render-math.mjs
 * Temml's inline style attributes become classes in web/learn.css (between the math-styles markers),
 * because the page's CSP forbids inline styles.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pagePath = path.join(root, 'web/learn.html'), cssPath = path.join(root, 'web/learn.css');
let temml;
try { const loaded = createRequire(import.meta.url)(process.env.TEMML_MODULE || 'temml'); temml = loaded.renderToString ? loaded : loaded.default; }
catch { console.error('Temml is not installed. Run: npm install --no-save temml@0.10.34'); process.exit(1); }

const unescape = text => text.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, '&');
let count = 0;
const render = (tex, displayMode) => {
  count++;
  try { return temml.renderToString(tex.trim(), { displayMode, annotate: true, throwOnError: true, trust: context => context.command === '\\class' }); }
  catch (error) { throw new Error('Cannot render ' + JSON.stringify(tex.trim()) + ': ' + error.message); }
};

let html = await readFile(pagePath, 'utf8');
html = html.replace(/<math\b([^>]*)>[\s\S]*?<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>[\s\S]*?<\/math>/g,
  (_, attrs, tex) => render(unescape(tex), /display="block"/.test(attrs)));
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => render(tex, true));
html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => render(tex, false));

const styles = new Map();
html = html.replace(/<(m[a-z]+|math)(\s[^>]*?)?>/g, (tag, name, attrs = '') => {
  const style = /\sstyle="([^"]*)"/.exec(attrs);
  if (!style) return tag;
  if (!styles.has(style[1])) styles.set(style[1], 'ms' + (styles.size + 1));
  const name2 = styles.get(style[1]);
  attrs = attrs.replace(style[0], '');
  attrs = /\sclass="/.test(attrs) ? attrs.replace(/\sclass="([^"]*)"/, (m, c) => ` class="${c} ${name2}"`) : attrs + ` class="${name2}"`;
  return `<${name}${attrs}>`;
});
if (/\sstyle="/.test(html)) throw new Error('An inline style attribute is left in the page.');
await writeFile(pagePath, html);
const rules = [...styles].map(([style, name]) => `.${name} { ${style.replace(/;?\s*$/, ';')} }`).join('\n');
const sheet = await readFile(cssPath, 'utf8');
await writeFile(cssPath, sheet.replace(/\/\* math-styles:start[\s\S]*\/\* math-styles:end \*\//,
  `/* math-styles:start (generated with the page's MathML) */\n${rules}\n/* math-styles:end */`));
console.log(`Rendered ${count} formulas; ${styles.size} style classes.`);
