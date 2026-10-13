/**
 * Interactive figures for the "How the agent learns" page.
 * The live figures run the real exported networks (web/policy.js) on the browser rules engine;
 * the rest come from the training log and an example game in web/learn-data.js.
 */
import { Game, coordinate } from './engine.js';
import { Policy, maskedSoftmax, pick } from './policy.js';
import { COLUMNS, ROWS, EXAMPLE, SUMMARY } from './learn-data.js';

const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const svgNode = (tag, attrs = {}, parent = null) => {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent?.append(node);
  return node;
};
const html = (tag, className = '', text = '') => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== '') node.textContent = text;
  return node;
};
const bold = text => html('b', '', String(text));
const tone = (className, text) => html('span', className, text);
const minus = text => String(text).replace(/^-/, '−');
const signed = (value, digits = 2) => minus((value >= 0 ? '+' : '') + value.toFixed(digits));
const percent = (value, digits = 0) => (value * 100).toFixed(digits) + '%';
const millions = value => (value >= 1e6 ? (value / 1e6).toFixed(value >= 1e7 ? 0 : 1) + 'M' : value.toLocaleString('en'));
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const wait = ms => new Promise(resolve => setTimeout(resolve, calm ? 0 : ms));
const column = name => COLUMNS.indexOf(name);

const COLOR = { policy: '#17706f', value: '#1f4fa3', advantage: '#c27a12', limit: '#6a4fb3', negative: '#d4583e', ink: '#12324f', soft: '#8796a2' };
const PIECE = ['#d4583e', '#e9b93f'];
const NAME = ['Coral', 'Gold'];
const GAMMA = 0.99, LAMBDA = 0.97;

/** A mid-game position from the agent's self-play: gold to move. */
const SAMPLE = { heights: [0, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  workers: [[11, 4], [0, 13]], current: 1, phase: 'move', selected: -1, winner: null, reason: '', turn: 10, revision: 22 };

// ------------------------------------------------------------------ models

const MODELS = new URL('./models/', import.meta.url);
let manifestPromise = null;
const policies = new Map();
const manifest = () => (manifestPromise ??= fetch(new URL('manifest.json', MODELS)).then(response => response.json()));
function loadPolicy(id) {
  if (!policies.has(id)) {
    policies.set(id, manifest().then(async ({ levels }) => {
      const entry = levels.find(level => level.id === id);
      const response = await fetch(new URL(entry.file, MODELS));
      if (!response.ok) throw new Error('download failed');
      return new Policy(entry, await response.arrayBuffer());
    }));
    policies.get(id).catch(() => policies.delete(id));
  }
  return policies.get(id);
}

/** Run `start` once the element is close to the screen. */
function whenNear(element, start) {
  if (!('IntersectionObserver' in window)) { start(); return; }
  const watcher = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) { watcher.disconnect(); start(); }
  }, { rootMargin: '400px' });
  watcher.observe(element);
}

// ------------------------------------------------------------------ boards

const CELL = 52, X0 = 26, Y0 = 6;
const centre = cell => [X0 + (cell % 5) * CELL + CELL / 2, Y0 + Math.floor(cell / 5) * CELL + CELL / 2];
const LEVEL_FILL = ['#eef3e8', '#dde7f1', '#c3d4e8', '#9fb8d6', '#c9d6e6'];

/** Draw a 5x5 board into an SVG (viewBox 0 0 300 300); returns a group for overlays. */
function drawBoard(svg, game, { onCell = null } = {}) {
  svg.replaceChildren();
  for (let cell = 0; cell < 25; cell++) {
    const x = X0 + (cell % 5) * CELL, y = Y0 + Math.floor(cell / 5) * CELL, height = game.heights[cell];
    const rect = svgNode('rect', { x, y, width: CELL, height: CELL, rx: 5, class: 'cell', fill: LEVEL_FILL[height] }, svg);
    if (onCell) {
      Object.entries({ tabindex: 0, role: 'button', 'aria-label': coordinate(cell) }).forEach(([key, value]) => rect.setAttribute(key, value));
      rect.addEventListener('click', () => onCell(cell));
      rect.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onCell(cell); } });
    }
    if (height === 4) svgNode('path', { d: `M${x + 12} ${y + 34}a14 14 0 0 1 28 0z`, fill: COLOR.value }, svg);
    else if (height > 0) svgNode('text', { x: x + 7, y: y + 15, fill: COLOR.ink, 'font-size': 12 }, svg).textContent = height;
  }
  game.workers.forEach((pair, player) => pair.forEach((cell, index) => {
    if (cell < 0) return;
    const [cx, cy] = centre(cell);
    svgNode('circle', { cx, cy: cy + 3, r: 14, fill: PIECE[player], stroke: '#fff', 'stroke-width': 2.5 }, svg);
    svgNode('text', { x: cx, y: cy + 7, 'text-anchor': 'middle', fill: player ? '#4f3d06' : '#fff', 'font-size': 11 }, svg).textContent = index + 1;
  }));
  for (let i = 0; i < 5; i++) {
    svgNode('text', { x: X0 + i * CELL + CELL / 2, y: Y0 + 5 * CELL + 18, 'text-anchor': 'middle', class: 'label' }, svg).textContent = 'ABCDE'[i];
    svgNode('text', { x: X0 - 12, y: Y0 + i * CELL + CELL / 2 + 4, 'text-anchor': 'middle', class: 'label' }, svg).textContent = i + 1;
  }
  return svgNode('g', {}, svg);
}

/** Mark an action on a full-size board: an arrow for a move, a ring for a placement or build. */
function markAction(group, game, action, color = COLOR.policy) {
  if (action === null || action === undefined) return;
  if (game.phase === 'move' && action >= 25) {
    const [x1, y1] = centre(game.workers[game.current][Math.floor((action - 25) / 25)]), [x2, y2] = centre((action - 25) % 25);
    svgNode('line', { x1, y1, x2, y2, stroke: color, 'stroke-width': 5, 'stroke-linecap': 'round', opacity: 0.85 }, group);
    svgNode('circle', { cx: x2, cy: y2, r: 21, fill: 'none', stroke: color, 'stroke-width': 4 }, group);
  } else {
    const [cx, cy] = centre(action);
    svgNode('circle', { cx, cy, r: 21, fill: 'none', stroke: color, 'stroke-width': 4 }, group);
  }
}

/** A thumbnail board (viewBox 0 0 50 50) with one action marked. */
function miniBoard(game, action) {
  const svg = svgNode('svg', { viewBox: '0 0 50 50', class: 'mini-board', 'aria-hidden': 'true' });
  for (let cell = 0; cell < 25; cell++) {
    svgNode('rect', { x: (cell % 5) * 10 + 0.5, y: Math.floor(cell / 5) * 10 + 0.5, width: 9, height: 9, rx: 1.5, fill: LEVEL_FILL[game.heights[cell]] }, svg);
  }
  const at = cell => [(cell % 5) * 10 + 5, Math.floor(cell / 5) * 10 + 5];
  game.workers.forEach((pair, player) => pair.forEach(cell => {
    if (cell >= 0) { const [cx, cy] = at(cell); svgNode('circle', { cx, cy, r: 3.3, fill: PIECE[player] }, svg); }
  }));
  const target = action >= 25 ? (action - 25) % 25 : action, [tx, ty] = at(target);
  if (game.phase === 'move' && action >= 25) {
    const [fx, fy] = at(game.workers[game.current][Math.floor((action - 25) / 25)]);
    svgNode('line', { x1: fx, y1: fy, x2: tx, y2: ty, stroke: COLOR.policy, 'stroke-width': 1.8, 'stroke-linecap': 'round' }, svg);
  }
  svgNode('circle', { cx: tx, cy: ty, r: 4.2, fill: 'none', stroke: COLOR.policy, 'stroke-width': 1.6 }, svg);
  return svg;
}

function describe(game, action) {
  if (game.phase === 'place') return 'Place ' + coordinate(action);
  if (game.phase === 'build') return (game.heights[action] === 3 ? 'Dome ' : 'Build ') + coordinate(action);
  return coordinate(game.workers[game.current][Math.floor((action - 25) / 25)]) + ' → ' + coordinate((action - 25) % 25);
}

// ------------------------------------------------------------------ the example game

/** Positions before each action, and the learner's decisions with their stored numbers. */
function prepare(source) {
  const states = [], game = new Game();
  for (const action of source.actions) { states.push(game.toDict()); game.apply(action); }
  states.push(game.toDict());
  const decisions = source.decisions.map(decision => {
    const state = states[decision.at], action = source.actions[decision.at];
    const probability = decision.probs.find(([a]) => a === action)?.[1] ?? 0;
    return { ...decision, state, action, probability, label: describe(Game.fromDict(state), action) };
  });
  return { ...source, states, decisions, won: source.winner === 0 };
}

/** TD errors, GAE advantages and rewards-to-go for the learner's decisions (the game ends after the last one). */
function score(values, won, gamma = GAMMA, lambda = LAMBDA) {
  const deltas = [], adv = [], targets = [], result = won ? 1 : -1;
  let next = 0;
  for (let t = values.length - 1; t >= 0; t--) {
    const end = t === values.length - 1;
    deltas[t] = (end ? result : 0) + (end ? 0 : gamma * values[t + 1]) - values[t];
    next = deltas[t] + (end ? 0 : gamma * lambda * next);
    adv[t] = next;
    targets[t] = result * gamma ** (values.length - 1 - t);
  }
  return { deltas, adv, targets };
}

let example = prepare(EXAMPLE);
const exampleListeners = new Set();
function setExample(next) { example = prepare(next); exampleListeners.forEach(listener => listener()); }
const exampleValues = () => example.decisions.map(d => d.value);

/** A decision worth showing: the largest advantage among moves the policy was unsure about. */
function featured() {
  const { adv } = score(exampleValues(), example.won);
  let best = -1;
  example.decisions.forEach((decision, index) => {
    if (decision.probability < 0.75 && decision.probs.length > 1 && (best < 0 || Math.abs(adv[index]) > Math.abs(adv[best]))) best = index;
  });
  return best >= 0 ? best : adv.indexOf(Math.max(...adv));
}

/** Play a fresh training game in the browser: Hard (learner, coral) against Medium. */
async function playExample() {
  const [learner, opponent] = await Promise.all([loadPolicy('full'), loadPolicy('developing')]);
  for (;;) {
    const game = new Game(), actions = [], decisions = [];
    while (game.winner === null && game.revision < 160) {
      const { probabilities, value } = (game.current === 0 ? learner : opponent).evaluate(game.observation(), game.mask());
      const action = pick(probabilities);
      if (game.current === 0) decisions.push({ at: actions.length, value, probs: game.legalActions().map(a => [a, probabilities[a]]).sort((a, b) => b[1] - a[1]) });
      actions.push(action);
      game.apply(action);
    }
    if (game.winner !== null && decisions.length > 3) return { actions, decisions, winner: game.winner, reason: game.reason };
  }
}

// ------------------------------------------------------------------ contents and totals

function contents() {
  const links = [...document.querySelectorAll('.toc a')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href')));
  const narrow = matchMedia('(max-width: 960px)');
  if (narrow.matches) $('toc-details').open = false;
  for (const link of links) link.addEventListener('click', () => { if (narrow.matches) $('toc-details').open = false; });
  let queued = false;
  const mark = () => {
    queued = false;
    let current = 0;
    sections.forEach((section, index) => { if (section.getBoundingClientRect().top < innerHeight * 0.3) current = index; });
    links.forEach((link, index) => link.classList.toggle('current', index === current));
  };
  addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(mark); } }, { passive: true });
  mark();
  $('fact-games').textContent = millions(SUMMARY.games);
  $('fact-decisions').textContent = millions(SUMMARY.decisions);
  $('fact-weights').textContent = SUMMARY.policyWeights.toLocaleString('en');
  $('fact-updates').textContent = SUMMARY.updates.toLocaleString('en');
}

// ------------------------------------------------------------------ the training loop

const STAGES = ['Play', 'Record', 'Score', 'Improve', 'Fit', 'Repeat'];

function pipeline() {
  const tabs = [...document.querySelectorAll('#pipeline [data-stage]')];
  const visual = $('stage-visual'), text = $('stage-text');
  let stage = 0, selected = null, token = 0, auto = false;

  const setText = (title, ...parts) => {
    const body = html('p');
    body.append(...parts);
    text.replaceChildren(html('h4', '', title), body);
  };
  const facts = () => {
    const values = exampleValues(), { adv, targets } = score(values, example.won);
    return { values, adv, targets, low: values.indexOf(Math.min(...values)), best: adv.indexOf(Math.max(...adv)) };
  };
  const focusDecision = () => (selected !== null && selected < example.decisions.length ? selected : featured());

  async function show(next) {
    stage = next;
    const mine = ++token;
    tabs.forEach((tab, index) => {
      tab.setAttribute('aria-selected', String(index === stage));
      tab.tabIndex = index === stage ? 0 : -1;
      tab.parentElement.classList.toggle('done', index < stage);
    });
    $('pipe-prev').disabled = stage === 0;
    $('pipe-next').textContent = stage === STAGES.length - 1 ? 'Start again' : 'Next step';
    visual.parentElement.className = 'stage-body ' + (stage === 0 ? 'split' : 'wide');
    await [playStage, recordStage, scoreStage, improveStage, fitStage, repeatStage][stage](() => mine === token);
  }

  async function playStage(alive) {
    const svg = svgNode('svg', { class: 'board-svg', viewBox: '0 0 300 300', role: 'img', 'aria-label': 'The example game, replayed' });
    const track = html('div', 'timeline'), caption = html('span', 'counter');
    track.setAttribute('aria-label', 'Actions in the game; coral marks are the learner\'s decisions');
    const learnerAt = new Set(example.decisions.map(d => d.at));
    const marks = example.actions.map((_, index) => {
      const mark = html('button', 'tick ' + (learnerAt.has(index) ? 'learner' : 'opponent'));
      mark.type = 'button';
      mark.setAttribute('aria-label', 'Show the board after action ' + (index + 1));
      mark.addEventListener('click', () => { token++; draw(index + 1); });
      track.append(mark);
      return mark;
    });
    visual.replaceChildren(svg, track);
    const draw = count => {
      const before = Game.fromDict(example.states[Math.max(0, count - 1)]);
      const overlay = drawBoard(svg, Game.fromDict(example.states[count]));
      if (count > 0) markAction(overlay, before, example.actions[count - 1], before.current === 0 ? COLOR.negative : '#b8860b');
      marks.forEach((mark, index) => mark.classList.toggle('past', index < count));
      caption.textContent = count ? `Action ${count} of ${example.actions.length}: ${NAME[before.current]}, ${describe(before, example.actions[count - 1])}.` : 'The game starts.';
    };
    setText('1. Play: collect trajectories',
      'Coral, the learner, plays with the current policy network and ', html('em', '', 'samples'), ' its moves. Gold is the opponent, here the Medium agent. ',
      `Coral's ${example.decisions.length} decisions are what it learns from; gold's moves are part of the environment. `,
      `${example.won ? 'Coral wins' : 'Gold wins'} on action ${example.actions.length}. The trainer plays 128 games like this at once.`, html('br'), caption);
    draw(0);
    for (let count = 1; count <= example.actions.length; count++) {
      await wait(190);
      if (!alive()) return;
      draw(count);
    }
  }

  async function recordStage() {
    const tape = html('ol', 'tape');
    tape.setAttribute('aria-label', 'The recorded decisions');
    example.decisions.forEach((decision, index) => {
      const chip = html('li', 'chip-decision'), button = html('button');
      button.type = 'button';
      button.setAttribute('aria-pressed', String(index === focusDecision()));
      button.setAttribute('aria-label', `Decision ${index}: ${decision.label}, probability ${percent(decision.probability)}, value ${decision.value.toFixed(2)}`);
      const bar = html('span', 'p-bar'), fill = html('span');
      fill.style.width = percent(decision.probability, 1);
      bar.append(fill);
      button.append(html('span', 't', 't = ' + index), miniBoard(Game.fromDict(decision.state), decision.action), html('span', 'act', decision.label), bar,
        html('span', 'p', 'π ' + percent(decision.probability)), html('span', 'v', 'V ' + decision.value.toFixed(2)));
      button.addEventListener('click', () => { selected = index; tape.querySelectorAll('button').forEach((other, i) => other.setAttribute('aria-pressed', String(i === index))); });
      chip.append(button);
      chip.style.setProperty('--i', index);
      tape.append(chip);
    });
    visual.replaceChildren(tape);
    setText('2. Record each decision',
      `For each of coral's ${example.decisions.length} decisions the trainer keeps the position, the legal moves, the action, `,
      tone('c-policy', 'its probability π'), ' under the policy that played, ', tone('c-value', 'the value V'),
      ` and the reward: 0 until the end, then ${example.won ? '+1 for the win' : '−1 for the loss'}. Select a decision to follow it through the next steps.`);
    await wait(example.decisions.length * 60 + 600);
  }

  async function scoreStage(alive) {
    const svg = svgNode('svg', { class: 'chart', viewBox: '0 0 640 250', role: 'img', 'aria-label': 'Values and advantages for each decision' });
    visual.replaceChildren(svg);
    const { values, adv, low, best } = facts();
    const chart = advantageChart(svg, values, adv, { height: 250, onPick: index => { selected = index; chart.draw(adv.length, focusDecision()); } });
    setText('3. Score: rewards-to-go and advantages',
      'Working backwards from the result, GAE turns the ', tone('c-value', 'value network\'s estimates'), ' into ', tone('c-adv', 'advantages'), '. ',
      `It expected ${values[0].toFixed(2)} at the start and ${values[low].toFixed(2)} at its lowest, decision ${low}. `,
      `The biggest advantage is decision ${best}, ${signed(adv[best])}. The rewards-to-go, the value network's targets, are computed at the same time.`);
    chart.draw(0, focusDecision());
    for (let shown = 1; shown <= adv.length; shown++) {
      await wait(85);
      if (!alive()) return;
      chart.draw(shown, focusDecision());
    }
  }

  async function improveStage(alive) {
    const index = focusDecision(), container = html('div', 'nudge');
    visual.replaceChildren(container);
    const nudge = createNudge(container);
    nudge.set(index);
    setText('4. Improve the policy', 'Running gradient ascent on the clipped objective…');
    const result = await nudge.run(alive);
    if (!alive() || !result) return;
    setText('4. Improve the policy', ...nudgeStory(index, result),
      ' The real step does this through every weight of the policy network, in minibatches of 2,048 of the batch\'s 16,384 decisions, stopping when the KL passes 0.015.');
  }

  async function fitStage(alive) {
    const svg = svgNode('svg', { class: 'chart', viewBox: '0 0 640 250', role: 'img', 'aria-label': 'Values moving towards their rewards-to-go targets' });
    visual.replaceChildren(svg);
    const { values, targets } = facts(), fitted = [...values];
    setText('5. Fit the value network',
      'Then, separately, the value network is fitted to the ', bold('rewards-to-go'), ': ', tone('c-value', 'each value'),
      ' moves towards its target by gradient descent on the squared error: 32 steps, 4 passes over the batch in minibatches. Here each value is treated as its own parameter; the real network shares its weights across all positions, so it moves towards the targets on average.');
    for (let step = 0; step <= 32; step += 2) {
      drawFit(svg, values, fitted, targets, step);
      await wait(90);
      if (!alive()) return;
      for (let repeat = 0; repeat < 2; repeat++) for (let t = 0; t < fitted.length; t++) fitted[t] -= 0.025 * 2 * (fitted[t] - targets[t]);
    }
  }

  async function repeatStage() {
    const svg = svgNode('svg', { class: 'chart', viewBox: '0 0 640 220', role: 'img', 'aria-label': 'Win rate against the lookahead program across training' });
    const stats = html('div', 'insights small');
    const final = ROWS.at(-1)[column('lookahead')];
    for (const [value, label] of [[SUMMARY.updates.toLocaleString('en'), 'loops: PPO updates'], [millions(SUMMARY.decisions), 'decisions learned from'], [percent(final, 1), 'wins against lookahead at the end']]) {
      const item = html('div');
      item.append(bold(value), html('span', '', label));
      stats.append(item);
    }
    visual.replaceChildren(svg, stats);
    const width = 640, left = 40, right = 12, top = 12, bottom = 186, maxUpdate = ROWS.at(-1)[0];
    const x = update => left + (update / maxUpdate) * (width - left - right), y = v => bottom - v * (bottom - top);
    for (const v of [0, 0.5, 1]) {
      svgNode('line', { x1: left, x2: width - right, y1: y(v), y2: y(v), class: v ? 'axis' : 'zero' }, svg);
      svgNode('text', { x: left - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = percent(v);
    }
    for (const update of [0, Math.round(maxUpdate / 2), maxUpdate]) svgNode('text', { x: x(update), y: bottom + 20, 'text-anchor': 'middle' }, svg).textContent = update.toLocaleString('en');
    const points = ROWS.filter(row => row[column('lookahead')] !== null).map(row => x(row[0]).toFixed(1) + ' ' + y(row[column('lookahead')]).toFixed(1));
    const path = svgNode('path', { d: 'M' + x(0) + ' ' + y(0) + 'L' + points.join('L'), fill: 'none', stroke: COLOR.value, 'stroke-width': 3, class: 'draw-in' }, svg);
    path.style.setProperty('--length', Math.ceil(path.getTotalLength()));
    svgNode('text', { x: x(maxUpdate), y: y(1) - 4, 'text-anchor': 'end', fill: COLOR.value }, svg).textContent = 'wins against the lookahead program';
    const half = ROWS.find(row => row[column('lookahead')] >= 0.5);
    setText('6. Repeat',
      'The improved networks play the next batch of games, and the loop starts again. Each pass is one PPO update. ',
      half ? `The agent first won half its games against the lookahead program around update ${half[0]}; by the end, ${percent(final, 1)}.` : '');
    await wait(2600);
  }

  const stopAuto = () => { auto = false; $('pipe-auto').setAttribute('aria-pressed', 'false'); $('pipe-auto').textContent = 'Play the whole loop'; };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => { stopAuto(); show(index); });
    tab.addEventListener('keydown', event => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = (stage + step + tabs.length) % tabs.length;
      stopAuto();
      show(next);
      tabs[next].focus();
    });
  });
  $('pipe-prev').addEventListener('click', () => { stopAuto(); if (stage > 0) show(stage - 1); });
  $('pipe-next').addEventListener('click', () => { stopAuto(); show(stage === STAGES.length - 1 ? 0 : stage + 1); });
  $('pipe-auto').addEventListener('click', async () => {
    if (auto) { stopAuto(); token++; return; }
    auto = true;
    $('pipe-auto').setAttribute('aria-pressed', 'true');
    $('pipe-auto').textContent = 'Stop';
    for (let next = 0; next < STAGES.length && auto; next++) {
      await show(next);
      if (auto) await wait(1400);
    }
    stopAuto();
  });
  $('pipe-new').addEventListener('click', async () => {
    const button = $('pipe-new');
    button.disabled = true;
    button.textContent = 'Loading the networks…';
    try {
      setExample(await playExample());
      selected = null;
      stopAuto();
      show(0);
    } catch {
      setText('The networks could not be downloaded', 'Check your connection and try again.');
    } finally {
      button.disabled = false;
      button.textContent = 'Play another game';
    }
  });
  show(0);
}

/** Critic values (line) and advantages (bars) per decision. Returns { draw(shown, focus) }. */
function advantageChart(svg, values, adv, { height = 300, onPick = null } = {}) {
  const width = 640, left = 42, right = 14, top = 16, bottom = height - 34;
  const low = Math.min(-0.5, ...adv, ...values) - 0.1, high = Math.max(1.1, ...adv, ...values) + 0.05;
  const n = values.length, slot = (width - left - right) / n;
  const x = t => left + slot * (t + 0.5), y = v => bottom - ((v - low) / (high - low)) * (bottom - top);
  return {
    draw(shown, focus = -1) {
      svg.replaceChildren();
      for (const v of [-1, -0.5, 0, 0.5, 1]) {
        if (v < low || v > high) continue;
        svgNode('line', { x1: left, x2: width - right, y1: y(v), y2: y(v), class: v ? 'axis' : 'zero' }, svg);
        svgNode('text', { x: left - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = minus(v);
      }
      for (let t = 0; t < n; t += n > 12 ? 5 : 1) svgNode('text', { x: x(t), y: bottom + 18, 'text-anchor': 'middle' }, svg).textContent = t;
      svgNode('text', { x: width - right, y: bottom + 30, 'text-anchor': 'end' }, svg).textContent = 'decision t';
      if (focus >= 0) svgNode('rect', { x: x(focus) - slot / 2, y: top, width: slot, height: bottom - top, fill: '#eaf0fa' }, svg);
      for (let t = n - shown; t < n; t++) {
        const y0 = y(0), y1 = y(adv[t]);
        svgNode('rect', { x: x(t) - slot * 0.3, y: Math.min(y0, y1), width: slot * 0.6, height: Math.max(1, Math.abs(y1 - y0)), rx: 2,
          fill: adv[t] >= 0 ? COLOR.advantage : COLOR.negative }, svg);
      }
      svgNode('path', { d: values.map((v, t) => (t ? 'L' : 'M') + x(t).toFixed(1) + ' ' + y(v).toFixed(1)).join(''), fill: 'none', stroke: COLOR.value, 'stroke-width': 2.5 }, svg);
      values.forEach((v, t) => svgNode('circle', { cx: x(t), cy: y(v), r: 3.5, fill: '#fff', stroke: COLOR.value, 'stroke-width': 2 }, svg));
      if (shown < n && shown > 0) svgNode('line', { x1: x(n - shown) - slot / 2, x2: x(n - shown) - slot / 2, y1: top, y2: bottom, stroke: COLOR.advantage, 'stroke-dasharray': '4 3' }, svg);
      if (onPick) for (let t = 0; t < n; t++) {
        const hit = svgNode('rect', { x: x(t) - slot / 2, y: top, width: slot, height: bottom - top, fill: 'transparent', class: 'hit', tabindex: 0, role: 'button',
          'aria-label': `Decision ${t}: value ${values[t].toFixed(2)}, advantage ${signed(adv[t])}` }, svg);
        hit.addEventListener('click', () => onPick(t));
        hit.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onPick(t); } });
      }
    },
  };
}

/** Values before (faint), during the fit (solid) and their rewards-to-go targets (dots). */
function drawFit(svg, before, fitted, targets, step) {
  const width = 640, height = 250, left = 42, right = 14, top = 16, bottom = height - 34, n = before.length, slot = (width - left - right) / n;
  const low = Math.min(-1, ...before, ...targets) - 0.05, high = Math.max(1, ...before, ...targets) + 0.05;
  const x = t => left + slot * (t + 0.5), y = v => bottom - ((v - low) / (high - low)) * (bottom - top);
  svg.replaceChildren();
  for (const v of [-1, 0, 1]) {
    svgNode('line', { x1: left, x2: width - right, y1: y(v), y2: y(v), class: v ? 'axis' : 'zero' }, svg);
    svgNode('text', { x: left - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg).textContent = minus(v);
  }
  for (let t = 0; t < n; t += n > 12 ? 5 : 1) svgNode('text', { x: x(t), y: bottom + 18, 'text-anchor': 'middle' }, svg).textContent = t;
  svgNode('text', { x: width - right, y: bottom + 30, 'text-anchor': 'end' }, svg).textContent = `decision t, value step ${step} of 32`;
  const line = (values, attrs) => svgNode('path', { d: values.map((v, t) => (t ? 'L' : 'M') + x(t).toFixed(1) + ' ' + y(v).toFixed(1)).join(''), fill: 'none', ...attrs }, svg);
  line(before, { stroke: COLOR.value, 'stroke-width': 2, opacity: 0.3, 'stroke-dasharray': '5 4' });
  targets.forEach((v, t) => svgNode('rect', { x: x(t) - 4, y: y(v) - 4, width: 8, height: 8, fill: COLOR.ink, transform: `rotate(45 ${x(t)} ${y(v)})` }, svg));
  line(fitted, { stroke: COLOR.value, 'stroke-width': 3 });
  fitted.forEach((v, t) => svgNode('circle', { cx: x(t), cy: y(v), r: 3.5, fill: '#fff', stroke: COLOR.value, 'stroke-width': 2 }, svg));
}

// ------------------------------------------------------------------ the clip at work on one decision

/**
 * Gradient ascent on the clipped objective with one decision's legal scores as the only parameters:
 * d/dz [rho * A] = A * rho * (onehot(a) - pi). Stops when the clip makes the gradient zero.
 */
function createNudge(container) {
  const board = svgNode('svg', { class: 'board-svg small', viewBox: '0 0 300 300', role: 'img', 'aria-label': 'The decision being improved' });
  const bars = html('ol', 'nudge-bars'), plot = svgNode('svg', { class: 'chart ratio-plot', viewBox: '0 0 440 156', role: 'img', 'aria-label': 'The ratio rho over the update steps' });
  const side = html('div', 'nudge-side');
  side.append(bars, plot);
  container.replaceChildren(board, side);
  let decision = null;
  const epsilon = 0.2;

  function rows(probabilities) {
    const chosen = decision.action, top = decision.probs.map(([a]) => a).slice(0, 4);
    if (!top.includes(chosen)) top[top.length - 1] = chosen;
    bars.replaceChildren(...top.map(action => {
      const item = html('li', action === chosen ? 'chosen' : ''), track = html('span', 'track'), old = html('span', 'old'), now = html('span', 'now');
      const before = decision.probs.find(([a]) => a === action)[1], after = probabilities.get(action);
      old.style.width = percent(before, 1);
      now.style.width = percent(after, 1);
      track.append(now, old);
      item.append(html('span', 'name', describe(Game.fromDict(decision.state), action) + (action === chosen ? ' (played)' : '')), track,
        html('span', 'pct', percent(before) + ' → ' + percent(after)));
      return item;
    }));
  }

  function drawPlot(history) {
    plot.replaceChildren();
    const width = 440, left = 46, right = 12, top = 22, bottom = 128, steps = Math.max(12, history.length - 1);
    const x = s => left + 8 + (s / steps) * (width - left - right - 8), y = r => bottom - ((Math.max(0.7, Math.min(1.3, r)) - 0.7) / 0.6) * (bottom - top);
    svgNode('rect', { x: left, y: y(1 + epsilon), width: width - left - right, height: y(1 - epsilon) - y(1 + epsilon), fill: '#efeafa' }, plot);
    for (const r of [1 - epsilon, 1, 1 + epsilon]) {
      svgNode('line', { x1: left, x2: width - right, y1: y(r), y2: y(r), stroke: r === 1 ? '#b9c7cf' : COLOR.limit, 'stroke-dasharray': r === 1 ? '' : '4 3' }, plot);
      svgNode('text', { x: left - 8, y: y(r) + 4, 'text-anchor': 'end' }, plot).textContent = r.toFixed(1);
    }
    svgNode('text', { x: width - right, y: bottom + 22, 'text-anchor': 'end' }, plot).textContent = 'gradient steps';
    svgNode('text', { x: left, y: top - 8 }, plot).textContent = 'ratio ρ of the played move';
    svgNode('path', { d: history.map((r, s) => (s ? 'L' : 'M') + x(s).toFixed(1) + ' ' + y(r).toFixed(1)).join(''), fill: 'none', stroke: COLOR.policy, 'stroke-width': 2.5 }, plot);
    svgNode('circle', { cx: x(history.length - 1), cy: y(history.at(-1)), r: 4.5, fill: '#fff', stroke: COLOR.policy, 'stroke-width': 2.5 }, plot);
  }

  return {
    set(index) {
      decision = { ...example.decisions[index], advantage: score(exampleValues(), example.won).adv[index] };
      const game = Game.fromDict(decision.state);
      markAction(drawBoard(board, game), game, decision.action);
      rows(new Map(decision.probs));
      drawPlot([1]);
    },
    get decision() { return decision; },
    async run(alive = () => true) {
      const legal = decision.probs.map(([a]) => a), old = new Map(decision.probs), chosen = decision.action, A = decision.advantage;
      const z = new Map(legal.map(a => [a, Math.log(Math.max(old.get(a), 1e-9))]));
      const softmax = () => {
        const peak = Math.max(...z.values()), p = new Map();
        let total = 0;
        for (const [a, v] of z) { p.set(a, Math.exp(v - peak)); total += p.get(a); }
        for (const a of legal) p.set(a, p.get(a) / total);
        return p;
      };
      let probabilities = softmax(), clipped = false;
      const history = [1];
      for (let step = 0; step < 40; step++) {
        const rho = probabilities.get(chosen) / old.get(chosen);
        clipped = (A > 0 && rho > 1 + epsilon) || (A < 0 && rho < 1 - epsilon);
        if (clipped || Math.abs(A) < 1e-6) break;
        for (const a of legal) z.set(a, z.get(a) + 0.06 * A * rho * ((a === chosen ? 1 : 0) - probabilities.get(a)));
        probabilities = softmax();
        history.push(probabilities.get(chosen) / old.get(chosen));
        if (!calm) { rows(probabilities); drawPlot(history); await wait(55); if (!alive()) return null; }
      }
      rows(probabilities);
      drawPlot(history);
      return { before: old.get(chosen), after: probabilities.get(chosen), rho: history.at(-1), clipped, steps: history.length - 1, advantage: A };
    },
  };
}

function nudgeStory(index, result) {
  const { before, after, rho, clipped, steps, advantage } = result;
  const parts = [`Decision ${index} has advantage `, tone('c-adv', signed(advantage)),
    advantage >= 0 ? ', so the played move becomes more likely: ' : ', so the played move becomes less likely: ', tone('c-policy', `${percent(before, 1)} → ${percent(after, 1)}`), '. '];
  if (clipped) parts.push(`After ${steps} steps the ratio ρ reached `, tone('c-limit', rho.toFixed(2)), ', outside [0.8, 1.2], so the clip makes the gradient zero and the push stops.');
  else if (advantage > 0 && before > 0.8) parts.push('It was already so likely that it cannot rise by 20%, so the clip never comes into play.');
  else parts.push('The advantage is small, so after 40 steps the ratio has only reached ', tone('c-limit', rho.toFixed(2)), '.');
  return parts;
}

// ------------------------------------------------------------------ observation planes

const PLANE_NAMES = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Dome', 'My worker 1', 'My worker 2', 'Their worker 1', 'Their worker 2',
  'Place phase', 'Move phase', 'Build phase', 'Worker 1 moved', 'Worker 2 moved', 'Edge: all ones'];
/** Short labels that fit under a plane on one line. */
const PLANE_LABELS = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Dome', 'Mine 1', 'Mine 2', 'Theirs 1', 'Theirs 2',
  'Place', 'Move', 'Build', 'Moved 1', 'Moved 2', 'Edge'];
const PLANE_COLOR = { height: '#1f4fa3', mine: '#e9b93f', theirs: '#d4583e', flag: '#17706f', edge: '#8fa3b3' };
const PLANE_GROUP = index => index < 5 ? 'height' : index < 7 ? 'mine' : index < 9 ? 'theirs' : index < 14 ? 'flag' : 'edge';

function planes() {
  const game = Game.fromDict(SAMPLE), observation = game.observation();
  const overlay = drawBoard($('planes-board'), game);
  $('planes-who').textContent = 'Gold to move. Hover or focus a plane to see its squares on the board.';
  const value = (plane, cell) => (plane < 9 ? observation[plane * 25 + cell] : plane < 14 ? observation[225 + plane - 9] : 1);
  const highlight = plane => {
    overlay.replaceChildren();
    if (plane === null) return;
    for (let cell = 0; cell < 25; cell++) if (value(plane, cell)) {
      svgNode('rect', { x: X0 + (cell % 5) * CELL + 3, y: Y0 + Math.floor(cell / 5) * CELL + 3, width: CELL - 6, height: CELL - 6, rx: 6, fill: 'none', stroke: COLOR.ink, 'stroke-width': 3.5 }, overlay);
    }
  };
  for (let plane = 0; plane < 15; plane++) {
    const wrap = html('button', 'plane group-' + PLANE_GROUP(plane)), mini = html('span', 'mini');
    wrap.type = 'button';
    mini.setAttribute('aria-hidden', 'true');
    for (let cell = 0; cell < 25; cell++) mini.append(html('i', value(plane, cell) ? 'on' : ''));
    wrap.setAttribute('aria-label', PLANE_NAMES[plane]);
    wrap.title = PLANE_NAMES[plane];
    wrap.append(mini, html('span', 'plane-name', PLANE_LABELS[plane]));
    for (const event of ['pointerenter', 'focus']) wrap.addEventListener(event, () => highlight(plane));
    for (const event of ['pointerleave', 'blur']) wrap.addEventListener(event, () => highlight(null));
    $('planes-grid').append(wrap);
  }
  const vector = $('planes-vector');
  for (const [name, start, end] of [['heights', 0, 125], ['workers', 125, 225], ['flags', 225, 230]]) {
    vector.append(name + ': ');
    for (let i = start; i < end; i++) {
      vector.append(observation[i] ? bold('1') : '0');
      if ((i - start) % 25 === 24 || i === end - 1) vector.append(' ');
    }
  }
}

// ------------------------------------------------------------------ the networks, layer by layer

/** The position shared by the network explorer and the move figure. */
const position = { game: Game.fromDict(SAMPLE), listeners: new Set(), set(game) { this.game = game; this.listeners.forEach(listener => listener()); } };
let analysisKey = '', analysis = null;
function analyse(policy, game) {
  const key = JSON.stringify(game.toDict());
  if (key !== analysisKey) {
    const trace = [], mask = game.mask();
    const { logits, value } = policy.logitsAndValue(game.observation(), trace);
    analysis = { trace, logits, value, mask, probabilities: maskedSoftmax(logits, mask) };
    analysisKey = key;
  }
  return analysis;
}

const mixColor = (from, to, t) => '#' + [0, 2, 4].map(i => Math.round(parseInt(from.slice(1 + i, 3 + i), 16) * (1 - t) + parseInt(to.slice(1 + i, 3 + i), 16) * t).toString(16).padStart(2, '0')).join('');

/** One 5x5 heatmap: `color(value, cell)` gives each square's colour. */
function heatmap(values, color, label, onFocus) {
  const map = html('button', 'heat');
  map.type = 'button';
  map.setAttribute('aria-label', label);
  for (let cell = 0; cell < 25; cell++) {
    const square = html('i');
    square.style.backgroundColor = color(values[cell], cell);
    map.append(square);
  }
  for (const event of ['pointerenter', 'focus', 'click']) map.addEventListener(event, onFocus);
  return map;
}

function explorer() {
  const flow = $('flow'), heads = $('heads-flow'), readout = $('explorer-readout');
  let policy = null;
  const say = text => { readout.textContent = text; };
  const column2 = (title, shape, maps, className = '') => {
    const node = html('div', 'layer-col ' + className), head = html('div', 'layer-head'), grid = html('div', 'maps');
    head.append(bold(title), html('small', '', shape));
    grid.append(...maps);
    node.append(head, grid);
    return node;
  };
  const headRow = (title, shape, maps, step, output, toneName) => {
    const row = html('div', 'head-row ' + toneName), label = html('div', 'layer-head'), grid = html('div', 'maps');
    label.append(bold(title), html('small', '', shape));
    grid.append(...maps);
    row.append(label, grid, html('span', 'step', step), output);
    return row;
  };

  function render() {
    if (!policy) return;
    const game = position.game, { trace, logits, value, mask, probabilities } = analyse(policy, game);
    const layer = name => trace.find(entry => entry.name === name);
    const input = layer('input');
    const columns = [column2('Input', '15 × 5 × 5', Array.from({ length: 15 }, (_, c) => heatmap(input.data.subarray(c * 25, c * 25 + 25),
      v => (v ? PLANE_COLOR[PLANE_GROUP(c)] : '#ffffff'), PLANE_NAMES[c], () => say(`Input plane ${c + 1}: ${PLANE_NAMES[c]}.`))), 'input')];
    for (const name of ['stem', 'block 1', 'block 2', 'block 3', 'block 4']) {
      const entry = layer(name);
      if (!entry) continue;
      const { data, channels } = entry;
      let peak = 0;
      const strongest = Array.from({ length: channels }, (_, c) => {
        let sum = 0;
        for (let i = 0; i < 25; i++) { sum += data[c * 25 + i]; peak = Math.max(peak, data[c * 25 + i]); }
        return [c, sum];
      }).sort((a, b) => b[1] - a[1]).slice(0, 6);
      const title = name === 'stem' ? 'Stem' : name.replace('block', 'Block');
      columns.push(column2(title, `${channels} × 5 × 5`, strongest.map(([c]) => {
        const values = data.subarray(c * 25, c * 25 + 25);
        let best = 0;
        for (let i = 1; i < 25; i++) if (values[i] > values[best]) best = i;
        return heatmap(values, v => mixColor('#ffffff', COLOR.policy, peak ? Math.min(1, v / peak) : 0), `${title}, channel ${c + 1}`,
          () => say(`${title}, channel ${c + 1} of ${channels}: strongest on ${coordinate(best)} (${values[best].toFixed(2)}). Darker squares respond more.`));
      })));
    }
    flow.replaceChildren(...columns);

    let scale = 0;
    for (let a = 0; a < 75; a++) if (mask[a]) scale = Math.max(scale, Math.abs(logits[a]));
    const families = ['Place or build', 'Move worker 1', 'Move worker 2'];
    const policyMaps = families.map((family, f) => {
      const wrap = html('div', 'family');
      wrap.append(heatmap(logits.subarray(f * 25, f * 25 + 25), (v, cell) => {
        if (!mask[f * 25 + cell]) return '#eef1f2';
        const t = Math.min(1, Math.abs(v) / (scale || 1));
        return v >= 0 ? mixColor('#ffffff', COLOR.policy, t) : mixColor('#ffffff', COLOR.negative, t);
      }, `Policy scores: ${family}`, () => say(`${family}: one score per square. Grey squares are illegal now and get masked out; teal scores are high, coral low.`)), html('small', '', ['Place, build', 'Worker 1', 'Worker 2'][f]));
      return wrap;
    });
    const probabilityBoard = svgNode('svg', { class: 'board-svg tiny', viewBox: '0 0 300 300', role: 'img', 'aria-label': 'Move probabilities' });
    probabilityOverlay(drawBoard(probabilityBoard, game), game, probabilities);
    const valueBox = html('div', 'value-out'), gauge = html('div', 'gauge'), marker = html('i');
    marker.style.left = Math.min(100, Math.max(0, (value + 1) * 50)) + '%';
    gauge.append(marker);
    valueBox.append(bold(signed(value)), html('small', '', 'value for the player to move'), gauge);
    const planesOut = layer('value planes'), valuePeak = Math.max(...planesOut.data, 1e-6);
    const separate = policy.spec.architecture === 'separate';
    const valueMaps = Array.from({ length: 4 }, (_, c) => heatmap(planesOut.data.subarray(c * 25, c * 25 + 25), v => mixColor('#ffffff', COLOR.value, v / valuePeak),
      `Value plane ${c + 1}`, () => say(separate
        ? `The value network has its own trunk (${policy.spec.value_channels} channels, ${policy.spec.value_blocks} blocks). Its head squeezes that into 4 planes, then two dense layers turn those 100 numbers into one value.`
        : 'The value head squeezes the trunk\'s channels into 4 planes, then two dense layers turn those 100 numbers into one value.')));
    heads.replaceChildren(
      headRow(separate ? 'Policy head' : 'Policy head', 'scores 3 × 5 × 5 = 75', policyMaps, 'mask, softmax', probabilityBoard, 'policy'),
      headRow(separate ? 'Value network' : 'Value head', separate ? `own trunk, then 4 × 5 × 5 → 128 → 1` : '4 × 5 × 5 → 128 → 1', valueMaps, 'dense layers', valueBox, 'value'));
    say(`${NAME[game.current]} to ${game.phase === 'place' ? 'place a worker' : game.phase}. Hover any small grid to see what it shows.`);
  }

  $('explorer-run').addEventListener('click', async () => {
    const parts = [...flow.children, ...heads.children];
    parts.forEach(part => part.classList.add('dim'));
    for (const part of parts) { await wait(170); part.classList.remove('dim'); }
  });
  position.listeners.add(render);
  whenNear($('explorer'), () => loadPolicy('full').then(loaded => {
    policy = loaded;
    $('explorer-run').disabled = false;
    render();
  }).catch(() => say('The network could not be downloaded. Check your connection and reload the page.')));
}

/** Lines and rings for the moves the agent is considering, scaled by probability. */
function probabilityOverlay(overlay, game, probabilities) {
  const atCell = new Map();
  for (const action of game.legalActions()) {
    const p = probabilities[action], target = game.phase === 'move' ? (action - 25) % 25 : action;
    atCell.set(target, (atCell.get(target) || 0) + p);
    if (game.phase === 'move' && p >= 0.01) {
      const [x1, y1] = centre(game.workers[game.current][Math.floor((action - 25) / 25)]), [x2, y2] = centre(target);
      svgNode('line', { x1, y1, x2, y2, stroke: COLOR.policy, 'stroke-width': 1.5 + 9 * p, 'stroke-linecap': 'round', opacity: 0.3 + 0.7 * p }, overlay);
    }
  }
  for (const [cell, p] of atCell) {
    if (p < 0.01) continue;
    const [cx, cy] = centre(cell);
    svgNode('circle', { cx, cy, r: 21, fill: 'none', stroke: COLOR.policy, 'stroke-width': 1.5 + 7 * p, opacity: 0.35 + 0.65 * p }, overlay);
    if (p >= 0.05) svgNode('text', { x: cx, y: cy + 4, 'text-anchor': 'middle', fill: '#0f5453', 'font-size': 12 }, overlay).textContent = percent(p);
  }
  if (game.phase === 'build') {
    const [cx, cy] = centre(game.workers[game.current][game.selected]);
    svgNode('circle', { cx, cy: cy + 3, r: 18, fill: 'none', stroke: COLOR.ink, 'stroke-width': 2 }, overlay);
  }
}

// ------------------------------------------------------------------ receptive field

const STOPS = [[0, 'the input alone'], [1, 'the stem'], [3, 'the stem and block 1'], [5, 'the stem and blocks 1–2'], [7, 'the stem and blocks 1–3'], [9, 'the stem and all four blocks']];

function receptive() {
  let chosen = 0;
  const input = $('receptive'), empty = new Game().toDict();
  const draw = () => {
    const [layers, name] = STOPS[Number(input.value)];
    const reach = cell => Math.max(Math.abs((cell % 5) - (chosen % 5)), Math.abs(Math.floor(cell / 5) - Math.floor(chosen / 5))) <= layers;
    const svg = $('receptive-board');
    drawBoard(svg, empty, { onCell: cell => { chosen = cell; draw(); svg.querySelectorAll('rect')[cell].focus(); } });
    [...svg.querySelectorAll('rect')].forEach((rect, cell) => {
      rect.setAttribute('fill', cell === chosen ? COLOR.value : reach(cell) ? '#bfe3d8' : '#eef3e8');
      rect.setAttribute('aria-pressed', String(cell === chosen));
    });
    const seen = Array.from({ length: 25 }, (_, cell) => cell).filter(reach).length;
    $('receptive-out').textContent = layers + (layers === 1 ? ' convolution' : ' convolutions');
    $('receptive-readout').textContent = `After ${name}, the features at ${coordinate(chosen)} depend on ${seen} of the 25 squares.` + (seen === 25 ? ' The whole board.' : '');
  };
  input.addEventListener('input', draw);
  draw();
}

// ------------------------------------------------------------------ choosing a move

function live() {
  let policy = null, busy = false, last = '';
  const buttons = ['live-step', 'live-reset', 'live-new'].map($);

  function render(highlight = null) {
    if (!policy) return;
    const game = position.game, overlay = drawBoard($('live-board'), game);
    if (game.winner !== null) {
      $('live-who').textContent = `${NAME[game.winner]} wins: ${game.reason.toLowerCase()}. Start a new game or go back to the example.`;
      $('live-bars').replaceChildren();
      $('live-step').disabled = true;
      return;
    }
    $('live-step').disabled = busy;
    const { logits, value, probabilities } = analyse(policy, game);
    probabilityOverlay(overlay, game, probabilities);
    const legal = game.legalActions().sort((a, b) => probabilities[b] - probabilities[a]);
    const shown = legal.slice(0, 6);
    if (highlight !== null && !shown.includes(highlight)) shown.push(highlight);
    let scale = 0;
    for (const a of legal) scale = Math.max(scale, Math.abs(logits[a]));
    $('live-bars').replaceChildren(...shown.map(action => {
      const row = html('li', action === highlight ? 'lit' : ''), scoreCell = html('span', 'score'), fill = html('span', logits[action] >= 0 ? 'pos' : 'neg');
      const share = Math.abs(logits[action]) / (scale || 1) * 50;
      fill.style.width = share + '%';
      fill.style.left = (logits[action] >= 0 ? 50 : 50 - share) + '%';
      scoreCell.append(fill, html('output', '', minus(logits[action].toFixed(1))));
      const prob = html('span', 'prob'), bar = html('span');
      bar.style.width = percent(probabilities[action], 1);
      prob.append(bar, html('output', '', percent(probabilities[action], 1)));
      row.append(html('span', 'name', describe(game, action)), scoreCell, prob);
      return row;
    }));
    const entropy = -legal.reduce((sum, a) => sum + (probabilities[a] > 0 ? probabilities[a] * Math.log(probabilities[a]) : 0), 0);
    $('live-who').textContent = `${last}${NAME[game.current]} to ${game.phase === 'place' ? 'place a worker' : game.phase}: ${legal.length} legal of 75 actions.`;
    $('live-value').textContent = signed(value);
    $('live-entropy').textContent = entropy.toFixed(2);
    $('live-choices').textContent = Math.exp(entropy).toFixed(1);
  }

  $('live-step').addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    const game = position.game, { probabilities } = analyse(policy, game);
    const action = pick(probabilities);
    // A short roulette over the listed moves, landing on the sampled one.
    const order = game.legalActions().sort((a, b) => probabilities[b] - probabilities[a]).slice(0, 6);
    if (!calm) for (let spin = 0; spin < 9; spin++) { render(order[spin % order.length]); await wait(45 + spin * 14); }
    render(action);
    await wait(650);
    const next = Game.fromDict(game.toDict()), mover = next.current, text = next.apply(action);
    last = `${NAME[mover]} ${text.charAt(0).toLowerCase() + text.slice(1)} (it had ${percent(probabilities[action])}). `;
    busy = false;
    position.set(next);
  });
  $('live-reset').addEventListener('click', () => { last = ''; position.set(Game.fromDict(SAMPLE)); });
  $('live-new').addEventListener('click', () => { last = ''; position.set(new Game()); });
  position.listeners.add(() => render());
  drawBoard($('live-board'), position.game);
  whenNear($('live-widget'), () => loadPolicy('full').then(loaded => {
    policy = loaded;
    buttons.forEach(button => { button.disabled = false; });
    render();
  }).catch(() => { $('live-who').textContent = 'The network could not be downloaded. Check your connection and reload the page.'; }));
}

// ------------------------------------------------------------------ opponents

function mix() {
  const parts = [['random', 'Random', 10], ['tactical', 'Tactical', 15], ['lookahead', 'Lookahead', 35], ['selves', 'Itself', 40]];
  $('mix').replaceChildren(...parts.map(([key, name, share]) => {
    const part = html('span', 'mix-part ' + key);
    part.style.flexGrow = share;
    part.append(bold(share + '%'), html('small', '', name));
    return part;
  }));
}

// ------------------------------------------------------------------ advantages

function gae() {
  let won = example.won, focus = -1, sweeping = false, chart = null;
  const svg = $('gae-chart');
  const compute = () => {
    const values = exampleValues(), gamma = Number($('gae-gamma').value), lambda = Number($('gae-lambda').value);
    return { values, gamma, lambda, ...score(values, won, gamma, lambda) };
  };
  const explain = (t, data) => {
    const { values, deltas, adv, gamma, lambda } = data, last = t === values.length - 1;
    $('gae-readout').replaceChildren(bold(`Decision ${t}`), ` (${example.decisions[t].label}): value `, tone('c-value', values[t].toFixed(2)),
      ', surprise δ = ', last ? `${won ? '+1' : '−1'} − ${values[t].toFixed(2)}` : `${gamma.toFixed(2)} × ${values[t + 1].toFixed(2)} − ${values[t].toFixed(2)}`,
      ' = ', tone('c-adv', signed(deltas[t])), ', advantage Â = δ', last ? '' : ` + ${(gamma * lambda).toFixed(3)} × Â(next)`, ' = ', tone('c-adv', signed(adv[t])), '.');
  };
  const draw = (shown = null) => {
    const data = compute();
    $('gae-gamma-out').textContent = data.gamma.toFixed(2);
    $('gae-lambda-out').textContent = data.lambda.toFixed(2);
    chart = advantageChart(svg, data.values, data.adv, { onPick: t => { focus = t; draw(); explain(t, compute()); } });
    chart.draw(shown ?? data.adv.length, focus);
    if (focus < 0) {
      const best = data.adv.indexOf(Math.max(...data.adv)), worst = data.adv.indexOf(Math.min(...data.adv));
      $('gae-readout').replaceChildren(`Largest advantage: decision ${best}, `, tone('c-adv', signed(data.adv[best])), `. Smallest: decision ${worst}, `,
        tone('c-adv', signed(data.adv[worst])), '. Select a decision for its numbers.');
    }
    return data;
  };
  const caption = () => {
    const values = exampleValues(), low = values.indexOf(Math.min(...values));
    $('gae-caption').textContent = `Coral's ${values.length} decisions from the training loop above. The value network expected ${values[0].toFixed(2)} at the start and ${values[low].toFixed(2)} at decision ${low}; `
      + 'where its estimate fell after a decision, that decision gets a negative advantage, and where it rose, a positive one. Select a decision for its numbers.';
  };
  for (const [id, result] of [['gae-win', true], ['gae-loss', false]]) $(id).addEventListener('click', () => {
    won = result;
    $('gae-win').setAttribute('aria-pressed', String(won));
    $('gae-loss').setAttribute('aria-pressed', String(!won));
    draw();
  });
  $('gae-gamma').addEventListener('input', () => draw());
  $('gae-lambda').addEventListener('input', () => draw());
  $('gae-sweep').addEventListener('click', async () => {
    if (sweeping) return;
    sweeping = true;
    const data = compute();
    for (let shown = 1; shown <= data.adv.length; shown++) {
      focus = data.adv.length - shown;
      chart.draw(shown, focus);
      explain(focus, data);
      await wait(420);
    }
    sweeping = false;
  });
  exampleListeners.add(() => { won = example.won; focus = -1; $('gae-win').setAttribute('aria-pressed', String(won)); $('gae-loss').setAttribute('aria-pressed', String(!won)); caption(); draw(); });
  caption();
  draw();
}

// ------------------------------------------------------------------ discount

function discount() {
  const chart = $('discount-chart'), input = $('gamma');
  const draw = () => {
    const gamma = Number(input.value), width = 640, left = 34, bottom = 150, n = 61, step = (width - left - 6) / n;
    $('gamma-out').textContent = gamma.toFixed(3).replace(/0$/, '');
    chart.replaceChildren();
    for (const value of [0, 0.5, 1]) {
      const y = bottom - value * 130;
      svgNode('line', { x1: left, x2: width, y1: y, y2: y, class: value ? 'axis' : 'zero' }, chart);
      svgNode('text', { x: left - 6, y: y + 4, 'text-anchor': 'end' }, chart).textContent = value;
    }
    for (let k = 0; k < n; k++) {
      const value = gamma ** k;
      svgNode('rect', { x: left + k * step + 1, y: bottom - value * 130, width: Math.max(1, step - 2), height: value * 130, rx: 1.5, fill: k % 10 ? '#9fb8d6' : COLOR.value }, chart);
      if (k % 10 === 0) svgNode('text', { x: left + k * step + step / 2, y: bottom + 18, 'text-anchor': 'middle' }, chart).textContent = k;
    }
    svgNode('text', { x: width - 4, y: bottom + 36, 'text-anchor': 'end' }, chart).textContent = 'decisions ahead, k';
    $('discount-readout').replaceChildren('A win 10 decisions away is worth ', bold((gamma ** 10).toFixed(3)), ' now; 40 away, ', bold((gamma ** 40).toFixed(3)), '.');
  };
  input.addEventListener('input', draw);
  draw();
}

// ------------------------------------------------------------------ the clipped objective

function clip() {
  let sign = 1;
  const chart = $('clip-chart');
  const draw = () => {
    const rho = Number($('clip-rho').value), epsilon = Number($('clip-eps').value);
    $('clip-rho-out').textContent = rho.toFixed(2);
    $('clip-eps-out').textContent = epsilon.toFixed(2);
    const width = 640, left = 46, right = 14, top = 14, bottom = 244;
    const [low, high] = sign > 0 ? [-0.2, 2.2] : [-2.2, 0.2];
    const x = r => left + (r / 2) * (width - left - right), y = v => bottom - ((v - low) / (high - low)) * (bottom - top);
    const clamp = r => Math.min(Math.max(r, 1 - epsilon), 1 + epsilon);
    const objective = r => Math.min(r * sign, clamp(r) * sign);
    chart.replaceChildren();
    const [flatFrom, flatTo] = sign > 0 ? [1 + epsilon, 2] : [0, 1 - epsilon];
    svgNode('rect', { x: x(flatFrom), y: top, width: x(flatTo) - x(flatFrom), height: bottom - top, fill: '#efeafa' }, chart);
    svgNode('text', { x: (x(flatFrom) + x(flatTo)) / 2, y: sign > 0 ? top + 40 : bottom - 40, 'text-anchor': 'middle', fill: COLOR.limit, 'font-weight': 600 }, chart).textContent = 'clipped: no gradient';
    for (const v of sign > 0 ? [0, 1, 2] : [-2, -1, 0]) {
      svgNode('line', { x1: left, x2: width - right, y1: y(v), y2: y(v), class: v ? 'axis' : 'zero' }, chart);
      svgNode('text', { x: left - 8, y: y(v) + 4, 'text-anchor': 'end' }, chart).textContent = minus(v);
    }
    for (const r of [0, 0.5, 1, 1.5, 2]) svgNode('text', { x: x(r), y: bottom + 20, 'text-anchor': 'middle' }, chart).textContent = r;
    svgNode('text', { x: width - right, y: bottom + 40, 'text-anchor': 'end' }, chart).textContent = 'ratio ρ';
    for (const [r, label] of [[1 - epsilon, '1 − ε'], [1 + epsilon, '1 + ε']]) {
      svgNode('line', { x1: x(r), x2: x(r), y1: top, y2: bottom, stroke: COLOR.limit, 'stroke-dasharray': '4 4' }, chart);
      svgNode('text', { x: x(r), y: top + 12, 'text-anchor': 'middle', fill: COLOR.limit }, chart).textContent = label;
    }
    const path = f => Array.from({ length: 201 }, (_, i) => i / 100).map((r, i) => (i ? 'L' : 'M') + x(r).toFixed(1) + ' ' + y(f(r)).toFixed(1)).join('');
    svgNode('path', { d: path(r => r * sign), fill: 'none', stroke: COLOR.soft, 'stroke-width': 2, 'stroke-dasharray': '6 5' }, chart);
    svgNode('path', { d: path(objective), fill: 'none', stroke: COLOR.policy, 'stroke-width': 3.5, 'stroke-linejoin': 'round' }, chart);
    svgNode('circle', { cx: x(rho), cy: y(objective(rho)), r: 7, fill: '#fff', stroke: COLOR.policy, 'stroke-width': 3 }, chart);
    const flat = sign > 0 ? rho > 1 + epsilon : rho < 1 - epsilon;
    $('clip-readout').replaceChildren('Unclipped ', bold(signed(rho * sign)), ', clipped ', bold(signed(clamp(rho) * sign)), '; PPO keeps the smaller, ', bold(signed(objective(rho))), '. ',
      flat ? `The action is already ${Math.round(Math.abs(rho - 1) * 100)}% ${rho >= 1 ? 'more' : 'less'} likely than when it was played, past the ${Math.round(epsilon * 100)}% limit, so it adds no gradient.`
        : sign > 0 ? 'The gradient still pushes this action to become more likely.' : 'The gradient still pushes this action to become less likely.');
  };
  for (const [id, value] of [['clip-good', 1], ['clip-bad', -1]]) $(id).addEventListener('click', () => {
    sign = value;
    $('clip-good').setAttribute('aria-pressed', String(sign > 0));
    $('clip-bad').setAttribute('aria-pressed', String(sign < 0));
    draw();
  });
  $('clip-rho').addEventListener('input', draw);
  $('clip-eps').addEventListener('input', draw);
  draw();
}

function nudgeWidget() {
  const nudge = createNudge($('nudge')), input = $('nudge-pick');
  const choose = () => {
    const index = Math.min(Number(input.value), example.decisions.length - 1);
    nudge.set(index);
    $('nudge-out').textContent = `${index}: ${nudge.decision.label}, Â ${signed(nudge.decision.advantage)}`;
    $('nudge-readout').textContent = 'Run the update to watch this decision\'s probability move.';
  };
  input.addEventListener('input', choose);
  $('nudge-run').addEventListener('click', async () => {
    const button = $('nudge-run');
    button.disabled = true;
    choose();
    const result = await nudge.run();
    $('nudge-readout').replaceChildren(...nudgeStory(Number(input.value), result));
    button.disabled = false;
  });
  const reset = () => { input.max = example.decisions.length - 1; input.value = featured(); choose(); };
  exampleListeners.add(reset);
  reset();
}

// ------------------------------------------------------------------ inside the real updates

function steps() {
  const input = $('steps-pick'), grid = $('steps'), readout = $('steps-readout');
  if (column('policy_steps') < 0) { $('steps-widget').hidden = true; return; }
  input.max = ROWS.length - 1;
  input.value = Math.max(0, ROWS.length - 1);
  const lines = [['Policy steps', 'policy'], ['Value steps', 'value']].map(([name, kind]) => {
    const row = html('div', 'step-line ' + kind), cells = html('div', 'cells');
    row.append(html('span', 'epoch-name', name), cells);
    grid.append(row);
    return cells;
  });
  // Rows from before the minibatch schedule have no step counts: they used Spinning Up's 80 + 80.
  const limitOf = row => Math.round((column('step_limit') >= 0 && row[column('step_limit')]) || 80);
  const valueOf = row => Math.round((column('value_steps') >= 0 && row[column('value_steps')]) || 80);
  const fill = (cells, count, used, kind) => {
    cells.replaceChildren(...Array.from({ length: count }, (_, i) => {
      const cell = html('i');
      cell.className = kind === 'value' ? 'done value' : i < used ? 'done' : i === used && used < count ? 'stop' : 'skip';
      return cell;
    }));
    cells.classList.toggle('wide', count > 40);
  };
  const draw = () => {
    const row = ROWS[Number(input.value)], limit = limitOf(row), value = valueOf(row);
    const used = Math.min(limit, Math.round(row[column('policy_steps')])), first = row[0] - SUMMARY.block + 1;
    $('steps-out').textContent = SUMMARY.block > 1 ? `${Math.max(1, first).toLocaleString('en')}–${row[0].toLocaleString('en')}` : row[0].toLocaleString('en');
    fill(lines[0], limit, used, 'policy');
    fill(lines[1], value, value, 'value');
    const schedule = limit >= 80 ? 'full-batch steps (Spinning Up\'s default)' : 'minibatch steps of 2,048 decisions';
    readout.replaceChildren(SUMMARY.block > 1 ? 'On average over these updates: ' : '', bold(used + ' of ' + limit + ' policy steps'),
      used < limit ? ' before the KL estimate passed 0.015 (it was ' : ' without passing the KL limit (final KL ', tone('c-limit', row[column('kl')].toFixed(4)), '), then ',
      bold(value + ' value steps'), `, all ${schedule}.`);
  };
  input.addEventListener('input', draw);
  draw();
}

// ------------------------------------------------------------------ training curves, results and evaluation

const SERIES = [['lookahead', 'Lookahead', COLOR.value], ['tactical', 'Tactical', COLOR.policy], ['random', 'Random', COLOR.soft],
  ['fixed', 'Earlier agents', COLOR.negative], ['snapshot', 'Its own snapshots', COLOR.advantage]];

function results() {
  const first = ROWS[0], final = ROWS.at(-1), half = ROWS.find(row => row[column('lookahead')] >= 0.5);
  const insights = [];
  if (half) insights.push([half[0].toLocaleString('en'), 'updates to beat the lookahead program in half its games']);
  insights.push([`${first[column('entropy')].toFixed(1)} → ${final[column('entropy')].toFixed(2)}`, `entropy: from choosing between ~${Math.round(Math.exp(first[column('entropy')]))} moves to ~${Math.exp(final[column('entropy')]).toFixed(1)}`]);
  insights.push([`${first[column('value_loss')].toFixed(2)} → ${final[column('value_loss')].toFixed(2)}`, 'value network error before each fit']);
  $('insights').replaceChildren(...insights.map(([value, label]) => { const item = html('div'); item.append(bold(value), html('span', '', label)); return item; }));
  $('results-takeaway').textContent = `${SUMMARY.updates.toLocaleString('en')} updates, ${millions(SUMMARY.decisions)} decisions and ${millions(SUMMARY.games)} games, `
    + `ending at ${percent(final[column('lookahead')], 1)} wins against the lookahead program during training.`;
  if (SUMMARY.mediumUpdate) $('curves-caption').append(` Medium is this run's snapshot from update ${SUMMARY.mediumUpdate}.`);
}

function curves() {
  const width = 640, left = 46, right = 46, top = 12, maxUpdate = ROWS.at(-1)[0];
  const x = update => left + (update / maxUpdate) * (width - left - right);
  const line = (svg, key, y, color, dash = '') => {
    let d = '';
    ROWS.forEach(row => { const v = row[column(key)]; if (v !== null && v !== undefined) d += (d ? 'L' : 'M') + x(row[0]).toFixed(1) + ' ' + y(v).toFixed(1); });
    svgNode('path', { d, fill: 'none', stroke: color, 'stroke-width': 2.2, 'stroke-linejoin': 'round', ...(dash ? { 'stroke-dasharray': dash } : {}) }, svg);
  };
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => Math.round(maxUpdate * f / 10) * 10);
  const xAxis = (svg, bottom) => {
    for (const update of ticks) svgNode('text', { x: x(update), y: bottom + 18, 'text-anchor': 'middle' }, svg).textContent = update.toLocaleString('en');
    svgNode('text', { x: width - right, y: bottom + 34, 'text-anchor': 'end' }, svg).textContent = 'PPO update';
  };
  const key = (name, color) => { const item = html('span', '', name); item.style.setProperty('--swatch', color); return item; };

  const wins = $('curve-wins'), winBottom = 220, winY = v => winBottom - v * (winBottom - top);
  for (const v of [0, 0.5, 1]) {
    svgNode('line', { x1: left, x2: width - right, y1: winY(v), y2: winY(v), class: v ? 'axis' : 'zero' }, wins);
    svgNode('text', { x: left - 8, y: winY(v) + 4, 'text-anchor': 'end' }, wins).textContent = percent(v);
  }
  if (SUMMARY.mediumUpdate) {
    svgNode('line', { x1: x(SUMMARY.mediumUpdate), x2: x(SUMMARY.mediumUpdate), y1: top, y2: winBottom, stroke: COLOR.ink, 'stroke-dasharray': '3 4' }, wins);
    svgNode('text', { x: x(SUMMARY.mediumUpdate) + 6, y: winY(0.3) }, wins).textContent = 'Medium saved here';
  }
  for (const [name, , color] of SERIES) line(wins, name, winY, color);
  xAxis(wins, winBottom);
  $('curve-legend').replaceChildren(...SERIES.map(([, name, color]) => key(name, color)));

  const loss = $('curve-loss'), lossBottom = 160;
  const entropyMax = Math.max(2, Math.ceil(Math.max(...ROWS.map(row => row[column('entropy')])))), valueMax = Math.max(0.2, Math.ceil(Math.max(...ROWS.map(row => row[column('value_loss')])) * 10) / 10);
  const entropyY = v => lossBottom - (v / entropyMax) * (lossBottom - top), valueY = v => lossBottom - (v / valueMax) * (lossBottom - top);
  for (const f of [0, 0.5, 1]) {
    svgNode('line', { x1: left, x2: width - right, y1: entropyY(f * entropyMax), y2: entropyY(f * entropyMax), class: f ? 'axis' : 'zero' }, loss);
    svgNode('text', { x: left - 8, y: entropyY(f * entropyMax) + 4, 'text-anchor': 'end' }, loss).textContent = (f * entropyMax).toFixed(f ? 1 : 0);
    svgNode('text', { x: width - right + 8, y: entropyY(f * entropyMax) + 4 }, loss).textContent = (f * valueMax).toFixed(2);
  }
  line(loss, 'entropy', entropyY, COLOR.policy);
  line(loss, 'value_loss', valueY, COLOR.value, '6 4');
  xAxis(loss, lossBottom);
  $('loss-legend').replaceChildren(key('Policy entropy, left scale', COLOR.policy), key('Value error, right scale (dashed)', COLOR.value));

  const rules = [wins, loss].map((svg, index) => svgNode('line', { y1: top, y2: index ? lossBottom : winBottom, stroke: COLOR.ink, 'stroke-width': 1, opacity: 0 }, svg));
  const read = row => {
    const title = html('p', 'snapshot-title');
    const extra = column('policy_steps') >= 0 ? `, ${Math.round(row[column('policy_steps')])} policy steps per update` : '';
    title.append(bold(`Update ${row[0].toLocaleString('en')}`), ` (${(row[1] / 1e6).toFixed(1)} million decisions): entropy ${row[column('entropy')].toFixed(2)}, value error ${row[column('value_loss')].toFixed(3)}${extra}`);
    const list = html('ol', 'snapshot-bars');
    for (const [name, label, color] of SERIES) {
      const v = row[column(name)];
      if (v === null || v === undefined) continue;
      const item = html('li'), track = html('span', 'track'), fill = html('span');
      fill.style.width = percent(v, 1);
      fill.style.backgroundColor = color;
      track.append(fill);
      item.append(html('span', 'name', label), track, html('output', '', percent(v)));
      list.append(item);
    }
    $('curve-snapshot').replaceChildren(title, list);
    rules.forEach(rule => { rule.setAttribute('x1', x(row[0])); rule.setAttribute('x2', x(row[0])); rule.setAttribute('opacity', 0.5); });
  };
  for (const svg of [wins, loss]) {
    svg.addEventListener('pointermove', event => {
      const box = svg.getBoundingClientRect(), update = ((event.clientX - box.left) / box.width * width - left) / (width - left - right) * maxUpdate;
      read(ROWS.reduce((best, row) => (Math.abs(row[0] - update) < Math.abs(best[0] - update) ? row : best)));
    });
  }
  read(ROWS[Math.min(ROWS.length - 1, Math.round(ROWS.length * 0.1))]);
}

async function evaluation() {
  const box = $('eval');
  try {
    const { levels } = await manifest();
    const label = { full: 'Hard', developing: 'Medium', early: 'Easy' }, colors = { full: COLOR.value, developing: COLOR.policy, early: COLOR.soft };
    const opponents = [['random', 'Random'], ['tactical', 'Tactical'], ['lookahead', 'One-move lookahead']];
    box.replaceChildren(...opponents.map(([key, name]) => {
      const group = html('div', 'eval-group');
      group.append(html('p', 'eval-name', name));
      for (const id of ['full', 'developing', 'early']) {
        const rate = levels.find(entry => entry.id === id)?.evaluation?.[key]?.win_rate;
        if (rate === undefined) continue;
        const row = html('div', 'eval-row'), track = html('span', 'track'), fill = html('span');
        fill.style.width = percent(rate, 1);
        fill.style.backgroundColor = colors[id];
        track.append(fill);
        row.append(html('span', 'name', label[id]), track, html('output', '', percent(rate, 1)));
        group.append(row);
      }
      return group;
    }));
  } catch { box.textContent = 'The evaluation results could not be loaded.'; }
}

contents();
pipeline();
planes();
explorer();
receptive();
live();
mix();
gae();
discount();
clip();
nudgeWidget();
steps();
results();
curves();
evaluation();
