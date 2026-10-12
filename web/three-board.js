/**
 * Three.js island renderer.
 * Frames are drawn on demand: after a state change, during tweens, camera inertia and (High quality only)
 * a throttled ambient loop. Shadows are re-rendered only when pieces move.
 */
import {
  BoxGeometry, BufferGeometry, Color, DirectionalLight, DynamicDrawUsage, Float32BufferAttribute,
  Group, HemisphereLight, InstancedMesh, Mesh, MeshBasicMaterial, MeshStandardMaterial, NeutralToneMapping,
  OctahedronGeometry, PCFShadowMap, PMREMGenerator, PerspectiveCamera, Points, PointsMaterial,
  QuadraticBezierCurve3, Raycaster, RoomEnvironment, RoundedBoxGeometry, SRGBColorSpace, Scene, TubeGeometry,
  Vector2, Vector3, WebGLRenderer,
} from './vendor/three.js';
import {
  LEVEL_TOP, PALETTE, cellX, cellZ, createIsland, createLabels, createPieces, markerGeometry, ringGeometry,
  setInstance, surface,
} from './island-models.js';
import { Tweens, easing } from './tween.js';
import { DEFAULT_VIEW, PHI_LIMITS, ZOOM_LIMITS, clamp, fitDistance, orbitPosition, stepCell } from './orbit.js';
import { detectChange } from './coach.js';
import { demoGame } from './board.js';

const FOV = 30;
const TARGET = new Vector3(0, .32, 0);
const QUALITY = {
  high: { ratio: 2, shadows: 2048, ambient: true },
  auto: { ratio: 1.5, shadows: 1024, ambient: false },
  economy: { ratio: 1, shadows: 0, ambient: false },
};
const WORKER_HEIGHT = .56;
const clone = game => JSON.parse(JSON.stringify(game));
const sameBoard = (a, b) => a.heights.join() === b.heights.join() && a.workers.flat().join() === b.workers.flat().join();
const basic = (color, opacity = 1) => new MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false });

export class ThreeBoard {
  constructor(canvas, { onPick = () => {}, onHover = () => {}, onFailure = () => {}, onNotice = () => {} } = {}) {
    this.canvas = canvas;
    this.callbacks = { onPick, onHover, onFailure, onNotice };
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.type = PCFShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    this.renderer.setClearColor(0x000000, 0);
    this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    this.tweens = new Tweens();
    this.view = { ...DEFAULT_VIEW };
    this.velocity = 0;
    this.quality = 'auto';
    this.ratioCap = QUALITY.auto.ratio;
    this.frameTimes = [];
    this.frameCount = 0;
    this.hover = -1;
    this.cursor = -1;
    this.decor = { targets: [], selected: -1, movable: [], hint: null, wins: [], threats: [], risky: [] };
    this.queue = [];
    this.anim = null;
    this.markerScale = 1;
    this.buildScene();
    this.game = demoGame();
    this.latest = this.game;
    this.queued = this.game;
    this.layoutPieces();
    this.bindEvents();
    this.resize();
    document.fonts?.ready.then(() => { this.labels.refresh(); this.requestRender(); });
  }

  buildScene() {
    const scene = this.scene = new Scene();
    this.camera = new PerspectiveCamera(FOV, 1, .1, 120);
    scene.add(new HemisphereLight('#e6f2ff', '#b9a382', 1.05));
    const sun = this.sun = new DirectionalLight('#fff0d8', 3.1);
    sun.position.set(-7.5, 8, 2.2);
    sun.castShadow = true;
    Object.assign(sun.shadow.camera, { left: -5.6, right: 5.6, top: 5.6, bottom: -5.6, near: .5, far: 30 });
    sun.shadow.bias = -.0004;
    sun.shadow.normalBias = .02;
    sun.shadow.radius = 3;
    scene.add(sun, sun.target);
    const pmrem = new PMREMGenerator(this.renderer);
    this.environment = pmrem.fromScene(new RoomEnvironment(), .04).texture;
    pmrem.dispose();
    scene.environment = this.environment;
    scene.environmentIntensity = .25;

    this.island = createIsland();
    scene.add(this.island);
    this.labels = createLabels();
    scene.add(this.labels);
    const pieces = this.pieces = createPieces();

    this.tiles = new InstancedMesh(new RoundedBoxGeometry(.92, .1, .92, 2, .025).translate(0, .05, 0),
      new MeshStandardMaterial({ color: '#ffffff', roughness: .86 }), 25);
    this.tiles.receiveShadow = true;
    for (let cell = 0; cell < 25; cell++) setInstance(this.tiles, cell, cellX(cell), 0, cellZ(cell));
    this.storeys = pieces.storeys.map(geometry => this.instanced(geometry, pieces.material, 25, true));
    this.domes = this.instanced(pieces.dome, pieces.material, 25, true);
    this.workers = [0, 1].map(player => [0, 1].map(() => {
      const mesh = new Mesh(pieces.workers[player], pieces.workerMaterials[player]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      return mesh;
    }));
    scene.add(this.tiles, ...this.storeys, this.domes, ...this.workers.flat());

    this.targetMarkers = this.instanced(markerGeometry(), basic('#ffffff', .95), 25);
    this.moverRings = this.instanced(ringGeometry(.235, .285), basic('#ffffff', .9), 4);
    this.alertRings = this.instanced(ringGeometry(.3, .365, 6), basic('#ffffff', .95), 25);
    this.selectedRing = new Mesh(ringGeometry(.23, .31), basic(PALETTE.teal));
    this.lastRing = new Mesh(ringGeometry(.425, .45, 64), basic('#b8892c', .85));
    this.hintGroup = new Group();
    this.hintRing = new Mesh(ringGeometry(.27, .335), basic(PALETTE.cobalt));
    this.hintGem = new Mesh(new OctahedronGeometry(.12), new MeshStandardMaterial({ color: PALETTE.cobalt, emissive: '#1b3f86', roughness: .3 }));
    this.hintFrom = new Mesh(ringGeometry(.24, .3), basic(PALETTE.cobalt, .85));
    this.hintGroup.add(this.hintRing, this.hintGem, this.hintFrom);
    const ghost = color => new MeshStandardMaterial({ color, transparent: true, opacity: .5, depthWrite: false, roughness: .6 });
    this.ghostWorkers = [0, 1].map(player => new Mesh(pieces.workers[player], ghost([PALETTE.coral, PALETTE.gold][player])));
    const ghostStone = new MeshStandardMaterial({ vertexColors: true, transparent: true, opacity: .42, depthWrite: false, roughness: .7, emissive: '#1f8a8a', emissiveIntensity: .45 });
    this.ghostPieces = [...pieces.storeys, pieces.dome].map(geometry => new Mesh(geometry, ghostStone));
    this.path = new Mesh(new BufferGeometry(), basic(PALETTE.teal, .75));
    scene.add(this.targetMarkers, this.moverRings, this.alertRings, this.selectedRing, this.lastRing, this.hintGroup,
      ...this.ghostWorkers, ...this.ghostPieces, this.path);

    const capacity = 220;
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(DynamicDrawUsage));
    geometry.setAttribute('color', new Float32BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(DynamicDrawUsage));
    this.particles = new Points(geometry, new PointsMaterial({ size: .075, vertexColors: true, transparent: true, depthWrite: false }));
    this.particles.frustumCulled = false;
    this.particles.visible = false;
    scene.add(this.particles);

    const unit = new BoxGeometry(1, 1, 1);
    this.pickBoxes = Array.from({ length: 50 }, (_, index) => {
      const box = new Mesh(unit);
      box.userData.cell = index % 25;
      return box;
    });
    this.raycaster = new Raycaster();
  }

  instanced(geometry, material, count, shadows = false) {
    const mesh = new InstancedMesh(geometry, material, count);
    mesh.count = 0;
    mesh.castShadow = shadows;
    mesh.receiveShadow = shadows;
    mesh.frustumCulled = false;
    return mesh;
  }

  // ---------------------------------------------------------------- state

  /** Show an authoritative position. Consecutive revisions animate; anything else snaps. */
  setState(game, decor = {}) {
    this.decor = { targets: [], selected: -1, movable: [], hint: null, wins: [], threats: [], risky: [], ...decor };
    const previous = this.queued;
    this.latest = game;
    if (game !== previous && game.revision === previous.revision + 1 && (!sameBoard(game, previous) || game.winner !== previous.winner)) {
      this.queue.push({ before: clone(previous), after: clone(game) });
      this.queued = clone(game);
      this.nextTransition();
    } else if (game.revision !== previous.revision || !sameBoard(game, previous)) {
      this.queue = [];
      this.finishAnimation();
      this.queued = clone(game);
      this.game = clone(game);
      this.lastChange = null;
      this.layoutPieces();
    }
    this.refreshDecor(true);
  }

  nextTransition() {
    if (this.anim || !this.queue.length) return;
    // Catch up instantly when several updates arrive together (e.g. after a reconnect).
    const skip = this.queue.length > 2 || this.reducedMotion.matches;
    const { before, after } = this.queue.shift();
    this.game = after;
    const change = detectChange(before, after);
    if (change) this.lastChange = { before, after, change };
    // Queued transitions start after setState returned, so markers must follow the displayed board here.
    this.refreshDecor();
    if (skip || !change) {
      this.layoutPieces();
      if (after.winner !== null && before.winner === null && !skip) this.celebrate(after);
      this.nextTransition();
      return;
    }
    this.play(before, after, change, 1, () => {
      if (after.winner !== null && before.winner === null) this.celebrate(after);
    });
  }

  play(before, after, change, speed = 1, onDone) {
    const anim = this.anim = { change, offset: 0, squash: 1, worker: null };
    const finish = () => {
      if (this.anim !== anim) return;
      this.anim = null;
      this.layoutPieces();
      onDone?.();
      this.nextTransition();
    };
    if (change.type === 'move' || change.type === 'place') {
      const from = change.from >= 0 ? [cellX(change.from), surface(before.heights[change.from]), cellZ(change.from)] : null;
      const to = [cellX(change.cell), surface(after.heights[change.cell]), cellZ(change.cell)];
      const lift = from ? .32 + Math.max(0, to[1] - from[1]) : 0;
      this.tweens.add({
        duration: (from ? 520 : 460) * speed, ease: easing.linear,
        update: (_, t) => {
          if (from) {
            const e = easing.inOutCubic(t);
            anim.worker = { ...change, x: from[0] + (to[0] - from[0]) * e, y: from[1] + (to[1] - from[1]) * e + Math.sin(Math.PI * t) * lift, z: from[2] + (to[2] - from[2]) * e, squash: 1 };
          } else {
            anim.worker = { ...change, x: to[0], y: to[1] + (1 - easing.outBounce(t)) * 1.1, z: to[2], squash: 1 };
          }
          this.layoutPieces();
        },
        done: () => this.tweens.add({
          duration: 160 * speed, ease: easing.linear,
          update: (_, t) => { anim.worker = { ...change, x: to[0], y: to[1], z: to[2], squash: 1 - .12 * Math.sin(Math.PI * t) }; this.layoutPieces(); },
          done: finish,
        }),
      });
    } else {
      const top = surface(after.heights[change.cell] - 1);
      this.tweens.add({
        duration: 480 * speed, ease: easing.linear,
        update: (_, t) => { anim.offset = (1 - easing.outBounce(t)) * .95; this.layoutPieces(); },
        done: finish,
      });
      this.tweens.add({ delay: 210 * speed, duration: 10, update: () => {}, done: () => this.dust(cellX(change.cell), top, cellZ(change.cell), change.type === 'dome') });
    }
    this.requestRender();
  }

  finishAnimation() {
    this.tweens.finish();
    this.anim = null;
  }

  replayLast() {
    if (!this.lastChange || this.anim || this.reducedMotion.matches) return;
    const { before, after, change } = this.lastChange;
    if (!sameBoard(after, this.game)) return;
    this.play(before, after, change, 1.25);
  }

  /** Place every storey, dome and worker for `this.game`, honouring the active animation. */
  layoutPieces() {
    const game = this.game, anim = this.anim, counts = [0, 0, 0];
    let domes = 0;
    this.pickTargets = this.pickBoxes.slice(0, 25);
    for (let cell = 0; cell < 25; cell++) {
      const height = game.heights[cell], x = cellX(cell), z = cellZ(cell);
      for (let level = 0; level < Math.min(height, 3); level++) {
        const lifted = anim && anim.change.cell === cell && anim.change.level === level + 1 && anim.change.type === 'build';
        setInstance(this.storeys[level], counts[level]++, x, LEVEL_TOP[level] + (lifted ? anim.offset : 0), z);
      }
      if (height === 4) {
        const lifted = anim && anim.change.cell === cell && anim.change.type === 'dome';
        setInstance(this.domes, domes++, x, LEVEL_TOP[3] + (lifted ? anim.offset : 0), z);
      }
      // A full-width column for the square and its tower, plus a slim box for a standing worker,
      // so a worker never hides clicks on the square behind it.
      const top = surface(height) + (height === 4 ? .3 : .02), column = this.pickBoxes[cell], figure = this.pickBoxes[cell + 25];
      column.scale.set(.96, top, .96);
      column.position.set(x, top / 2, z);
      column.updateMatrixWorld();
      const occupied = game.workers.some(pair => pair.includes(cell));
      if (occupied) {
        figure.scale.set(.45, WORKER_HEIGHT, .45);
        figure.position.set(x, surface(height) + WORKER_HEIGHT / 2, z);
        figure.updateMatrixWorld();
        this.pickTargets.push(figure);
      }
    }
    this.storeys.forEach((mesh, level) => { mesh.count = counts[level]; mesh.instanceMatrix.needsUpdate = true; });
    this.domes.count = domes;
    this.domes.instanceMatrix.needsUpdate = true;
    for (let player = 0; player < 2; player++) for (let index = 0; index < 2; index++) {
      const mesh = this.workers[player][index], cell = game.workers[player][index];
      mesh.visible = cell >= 0;
      if (cell < 0) continue;
      const moving = anim?.worker && anim.worker.player === player && anim.worker.worker === index ? anim.worker : null;
      mesh.position.set(moving ? moving.x : cellX(cell), moving ? moving.y : surface(game.heights[cell]), moving ? moving.z : cellZ(cell));
      mesh.scale.set(1, moving ? moving.squash : 1, 1);
      mesh.rotation.y = moving?.spin ?? 0;
    }
    this.shadowsDirty = true;
    this.requestRender();
  }

  // ---------------------------------------------------------------- decorations

  refreshDecor(animateMarkers = false) {
    const { targets, selected, movable, hint, wins, threats, risky } = this.decor;
    const game = this.latest, heights = this.game.heights;
    const targetSet = new Set(targets), riskySet = new Set(risky);
    const focus = this.hover >= 0 ? this.hover : this.cursor;
    const tile = new Color();
    for (let cell = 0; cell < 25; cell++) {
      tile.set((cell % 5 + Math.floor(cell / 5)) % 2 ? PALETTE.tileB : PALETTE.tileA);
      if (targetSet.has(cell)) tile.lerp(new Color('#bfe4d6'), .75);
      if (cell === focus) tile.lerp(new Color(targetSet.has(cell) ? '#9fdcc8' : '#f7f8ef'), .7);
      this.tiles.setColorAt(cell, tile);
    }
    this.tiles.instanceColor.needsUpdate = true;

    const markerKey = targets.join();
    if (animateMarkers && markerKey !== this.markerKey && targets.length && !this.reducedMotion.matches) {
      this.markerScale = .3;
      this.tweens.add({ duration: 260, ease: easing.outBack, update: e => { this.markerScale = .3 + .7 * e; this.placeMarkers(); } });
    }
    this.markerKey = markerKey;
    this.placeMarkers();

    this.moverRings.count = 0;
    for (const cell of movable) {
      setInstance(this.moverRings, this.moverRings.count++, cellX(cell), surface(heights[cell]) + .008, cellZ(cell));
      this.moverRings.setColorAt(this.moverRings.count - 1, new Color(cell === focus ? '#ffffff' : '#e9fbf5'));
    }
    this.moverRings.instanceMatrix.needsUpdate = true;
    if (this.moverRings.instanceColor) this.moverRings.instanceColor.needsUpdate = true;

    this.alertRings.count = 0;
    for (const [cells, color] of [[threats, PALETTE.amber], [wins, PALETTE.gold]]) for (const cell of cells) {
      setInstance(this.alertRings, this.alertRings.count, cellX(cell), surface(heights[cell]) + .01, cellZ(cell), 1, Math.PI / 6);
      this.alertRings.setColorAt(this.alertRings.count++, new Color(color));
    }
    this.alertRings.instanceMatrix.needsUpdate = true;
    if (this.alertRings.instanceColor) this.alertRings.instanceColor.needsUpdate = true;

    this.place(this.selectedRing, selected, .012);
    this.lastRing.visible = false;
    if (this.lastChange && sameBoard(this.lastChange.after, this.game)) this.place(this.lastRing, this.lastChange.change.cell, .004);

    this.hintGroup.visible = Boolean(hint);
    if (hint) {
      this.place(this.hintRing, hint.cell, .014);
      this.hintGem.position.set(cellX(hint.cell), surface(heights[hint.cell]) + .62, cellZ(hint.cell));
      this.hintGem.visible = true;
      this.place(this.hintFrom, hint.from ?? -1, .014);
    }
    this.preview(game, focus, targetSet, selected, riskySet);
    this.requestRender();
  }

  placeMarkers() {
    const { targets, risky } = this.decor, riskySet = new Set(risky), heights = this.game.heights;
    this.targetMarkers.count = targets.length;
    targets.forEach((cell, i) => {
      setInstance(this.targetMarkers, i, cellX(cell), surface(heights[cell]) + .006, cellZ(cell), this.markerScale);
      this.targetMarkers.setColorAt(i, new Color(riskySet.has(cell) ? PALETTE.amber : PALETTE.teal));
    });
    this.targetMarkers.instanceMatrix.needsUpdate = true;
    if (this.targetMarkers.instanceColor) this.targetMarkers.instanceColor.needsUpdate = true;
    this.requestRender();
  }

  place(mesh, cell, lift) {
    mesh.visible = cell >= 0;
    if (cell >= 0) mesh.position.set(cellX(cell), surface(this.game.heights[cell]) + lift, cellZ(cell));
  }

  /** Ghost of the piece that would appear, plus the path a worker would take. */
  preview(game, focus, targets, selected, risky) {
    for (const mesh of [...this.ghostWorkers, ...this.ghostPieces, this.path]) mesh.visible = false;
    if (focus < 0 || !targets.has(focus) || !game || game.winner !== null) return;
    const x = cellX(focus), z = cellZ(focus), height = this.game.heights[focus];
    if (game.phase === 'build') {
      const ghost = this.ghostPieces[height];
      ghost.visible = true;
      ghost.position.set(x, LEVEL_TOP[height] + .015, z);
      ghost.material.emissive.set(risky.has(focus) ? PALETTE.amber : PALETTE.teal);
      return;
    }
    const ghost = this.ghostWorkers[game.current];
    ghost.visible = true;
    ghost.position.set(x, surface(height), z);
    if (game.phase === 'move' && selected >= 0) {
      const start = new Vector3(cellX(selected), surface(this.game.heights[selected]) + .3, cellZ(selected));
      const end = new Vector3(x, surface(height) + .3, z);
      const middle = start.clone().lerp(end, .5);
      middle.y = Math.max(start.y, end.y) + .45;
      this.path.geometry.dispose();
      this.path.geometry = new TubeGeometry(new QuadraticBezierCurve3(start, middle, end), 24, .018, 6, false);
      this.path.visible = true;
    }
  }

  setHover(cell) {
    if (cell === this.hover) return;
    this.hover = cell;
    this.callbacks.onHover(cell);
    this.canvas.style.cursor = cell >= 0 ? 'pointer' : 'grab';
    this.refreshDecor();
  }

  get selected() { return this.decor.selected; }

  // ---------------------------------------------------------------- effects

  dust(x, y, z, dome) {
    const count = 26, position = this.particles.geometry.attributes.position, color = this.particles.geometry.attributes.color;
    const seeds = Array.from({ length: count }, (_, i) => [i / count * Math.PI * 2 + Math.random() * .3, .5 + Math.random() * .6, Math.random()]);
    const tint = new Color(dome ? '#cfe0f5' : '#f3efe4');
    for (let i = 0; i < count; i++) color.setXYZ(i, tint.r, tint.g, tint.b);
    color.needsUpdate = true;
    this.particles.geometry.setDrawRange(0, count);
    this.particles.material.size = .06;
    this.particles.visible = true;
    this.tweens.add({
      duration: 520, ease: easing.outCubic,
      update: e => {
        seeds.forEach(([angle, speed, rise], i) => position.setXYZ(i, x + Math.cos(angle) * .55 * speed * e, y + .04 + rise * .18 * e, z + Math.sin(angle) * .55 * speed * e));
        position.needsUpdate = true;
        this.particles.material.opacity = .85 * (1 - e);
      },
      done: () => { this.particles.visible = false; },
    });
  }

  celebrate(game) {
    if (this.reducedMotion.matches) return;
    const cell = game.workers[game.winner].find(c => c >= 0 && game.heights[c] === 3) ?? game.workers[game.winner][0];
    if (cell < 0) return;
    const index = game.workers[game.winner].indexOf(cell);
    const x = cellX(cell), base = surface(game.heights[cell]), z = cellZ(cell);
    const anim = this.anim = { change: { type: 'win', cell }, offset: 0, worker: null };
    this.tweens.add({
      duration: 900, ease: easing.linear,
      update: (_, t) => { anim.worker = { player: game.winner, worker: index, x, y: base + Math.sin(Math.PI * t) * .45, z, squash: 1, spin: easing.inOutCubic(t) * Math.PI * 2 }; this.layoutPieces(); },
      done: () => { if (this.anim === anim) this.anim = null; this.layoutPieces(); this.nextTransition(); },
    });
    const count = 200, position = this.particles.geometry.attributes.position, color = this.particles.geometry.attributes.color;
    const palette = [PALETTE.coral, PALETTE.gold, PALETTE.cobalt, '#ffffff', PALETTE.teal].map(c => new Color(c));
    const seeds = Array.from({ length: count }, () => {
      const angle = Math.random() * Math.PI * 2, spread = .6 + Math.random() * 1.7;
      return [Math.cos(angle) * spread, 2.2 + Math.random() * 2.2, Math.sin(angle) * spread, palette[Math.floor(Math.random() * palette.length)]];
    });
    seeds.forEach(([, , , tint], i) => color.setXYZ(i, tint.r, tint.g, tint.b));
    color.needsUpdate = true;
    this.particles.geometry.setDrawRange(0, count);
    this.particles.material.size = .085;
    this.particles.visible = true;
    this.tweens.add({
      duration: 2200, ease: easing.linear,
      update: (_, t) => {
        const s = t * 2.2;
        seeds.forEach(([vx, vy, vz], i) => position.setXYZ(i, x + vx * s, base + .6 + vy * s - 2.6 * s * s, z + vz * s));
        position.needsUpdate = true;
        this.particles.material.opacity = t < .75 ? 1 : (1 - t) / .25;
      },
      done: () => { this.particles.visible = false; },
    });
    this.requestRender();
  }

  // ---------------------------------------------------------------- camera

  rotate(direction) { this.tweenView({ theta: this.view.theta + direction * Math.PI / 2 }); }
  zoomBy(delta) { this.tweenView({ zoom: clamp(this.view.zoom - delta, ...ZOOM_LIMITS) }, 260); }
  resetView() {
    const turn = Math.PI * 2, theta = this.view.theta - Math.round((this.view.theta - DEFAULT_VIEW.theta) / turn) * turn;
    this.view.theta = theta;
    this.tweenView({ ...DEFAULT_VIEW });
  }

  tweenView(goal, duration = 480) {
    this.velocity = 0;
    const start = { ...this.view };
    if (this.reducedMotion.matches) { Object.assign(this.view, goal); this.requestRender(); return; }
    this.viewTween?.();
    this.viewTween = this.tweens.add({
      duration, ease: easing.inOutCubic,
      update: e => { for (const key of Object.keys(goal)) this.view[key] = start[key] + (goal[key] - start[key]) * e; },
    });
    this.requestRender();
  }

  updateCamera() {
    // Portrait frames carry the status card over the far rows, so aim a little higher to push the board down.
    const aspect = this.camera.aspect, target = TARGET.clone();
    target.y += aspect < 1 ? .55 : .15;
    const distance = fitDistance(aspect < 1 ? 3.55 : 4.05, FOV, aspect) * this.view.zoom;
    this.camera.position.set(...orbitPosition(target.toArray(), distance, this.view.theta, this.view.phi));
    this.camera.lookAt(target);
  }

  /** A viewport point where a click selects `cell`, preferring its centre; null if fully hidden. */
  cellScreenPosition(cell) {
    this.updateCamera();
    this.camera.updateMatrixWorld();
    const box = this.canvas.getBoundingClientRect(), y = surface(this.game.heights[cell]);
    const lift = this.game.workers.some(pair => pair.includes(cell)) ? .3 : .01;
    const offsets = [[0, 0]];
    for (const r of [.2, .34]) for (let i = 0; i < 8; i++) offsets.push([Math.cos(i * Math.PI / 4) * r, Math.sin(i * Math.PI / 4) * r]);
    for (const [dx, dz] of offsets) {
      const point = new Vector3(cellX(cell) + dx, y + (dx || dz ? .01 : lift), cellZ(cell) + dz).project(this.camera);
      const screen = { x: box.left + (point.x + 1) / 2 * box.width, y: box.top + (1 - point.y) / 2 * box.height };
      if (this.pick(screen.x, screen.y) === cell) return screen;
    }
    return null;
  }

  pick(clientX, clientY) {
    const box = this.canvas.getBoundingClientRect();
    const pointer = new Vector2((clientX - box.left) / box.width * 2 - 1, -((clientY - box.top) / box.height) * 2 + 1);
    this.updateCamera();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickTargets, false).map(hit => hit.object.userData.cell);
    // Prefer a playable square further along the ray, so towers never block a legal click behind them.
    const playable = new Set([...this.decor.targets, ...this.decor.movable, this.decor.selected]);
    return hits.find(cell => playable.has(cell)) ?? hits[0] ?? -1;
  }

  // ---------------------------------------------------------------- input

  bindEvents() {
    this.abort = new AbortController();
    const listen = (target, type, handler, options = {}) => target.addEventListener(type, handler, { ...options, signal: this.abort.signal });
    const canvas = this.canvas, pointers = new Map();
    let gesture = null, lastNotice = 0;
    canvas.tabIndex = 0;
    canvas.style.cursor = 'grab';

    listen(canvas, 'pointerdown', event => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      this.velocity = 0;
      if (pointers.size === 1) gesture = { id: event.pointerId, x: event.clientX, y: event.clientY, time: performance.now(), moved: false, view: { ...this.view }, samples: [] };
      else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        gesture = { pinch: Math.hypot(a.x - b.x, a.y - b.y), mid: (a.y + b.y) / 2, view: { ...this.view }, moved: true };
      }
    });
    listen(canvas, 'pointermove', event => {
      if (pointers.has(event.pointerId)) pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (gesture?.pinch && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        this.view.zoom = clamp(gesture.view.zoom * gesture.pinch / Math.max(20, Math.hypot(a.x - b.x, a.y - b.y)), ...ZOOM_LIMITS);
        this.view.phi = clamp(gesture.view.phi - ((a.y + b.y) / 2 - gesture.mid) * .006, ...PHI_LIMITS);
        this.requestRender();
        return;
      }
      if (gesture && gesture.id === event.pointerId) {
        const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
        if (!gesture.moved && Math.hypot(dx, event.pointerType === 'touch' ? 0 : dy) > 6) {
          gesture.moved = true;
          canvas.setPointerCapture?.(event.pointerId);
          canvas.style.cursor = 'grabbing';
          this.viewTween?.();
        }
        if (gesture.moved) {
          this.view.theta = gesture.view.theta - dx * .0065;
          if (event.pointerType !== 'touch') this.view.phi = clamp(gesture.view.phi - dy * .005, ...PHI_LIMITS);
          gesture.samples.push([performance.now(), this.view.theta]);
          if (gesture.samples.length > 5) gesture.samples.shift();
          this.requestRender();
          return;
        }
      }
      if (event.pointerType !== 'touch' && !gesture) this.queueHover(event.clientX, event.clientY);
    });
    const release = event => {
      const ended = gesture;
      pointers.delete(event.pointerId);
      if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (pointers.size) return;
      gesture = null;
      canvas.style.cursor = this.hover >= 0 ? 'pointer' : 'grab';
      if (!ended || event.type === 'pointercancel') return;
      if (!ended.moved && !ended.pinch && performance.now() - ended.time < 700) {
        const cell = this.pick(event.clientX, event.clientY);
        if (cell >= 0) { this.cursor = -1; this.callbacks.onPick(cell); }
        return;
      }
      const samples = ended.samples || [];
      if (samples.length > 1 && !this.reducedMotion.matches) {
        const [t0, a0] = samples[0], [t1, a1] = samples.at(-1);
        if (performance.now() - t1 < 80 && t1 > t0) { this.velocity = (a1 - a0) / (t1 - t0); this.requestRender(); }
      }
    };
    listen(canvas, 'pointerup', release);
    listen(canvas, 'pointercancel', release);
    listen(canvas, 'pointerleave', event => { if (event.pointerType !== 'touch' && !gesture) this.setHover(-1); });
    listen(canvas, 'wheel', event => {
      if (!event.ctrlKey && !event.metaKey) {
        if (performance.now() - lastNotice > 5000) { lastNotice = performance.now(); this.callbacks.onNotice('Hold Ctrl and scroll to zoom'); }
        return;
      }
      event.preventDefault();
      this.view.zoom = clamp(this.view.zoom * Math.exp(event.deltaY * .0025), ...ZOOM_LIMITS);
      this.requestRender();
    }, { passive: false });
    listen(canvas, 'keydown', event => {
      const key = event.key;
      if (key.startsWith('Arrow')) {
        event.preventDefault();
        this.cursor = stepCell(this.cursor >= 0 ? this.cursor : this.hover, key, this.view.theta);
        this.hover = -1;
        this.callbacks.onHover(this.cursor);
        this.refreshDecor();
      } else if ((key === 'Enter' || key === ' ') && this.cursor >= 0) {
        event.preventDefault();
        this.callbacks.onPick(this.cursor);
      } else if (key === 'q' || key === 'Q' || key === '[') this.rotate(-1);
      else if (key === 'e' || key === 'E' || key === ']') this.rotate(1);
      else if (key === '+' || key === '=') this.zoomBy(.1);
      else if (key === '-' || key === '_') this.zoomBy(-.1);
      else if (key === '0') this.resetView();
    });
    listen(canvas, 'blur', () => { if (this.cursor >= 0) { this.cursor = -1; this.callbacks.onHover(-1); this.refreshDecor(); } });
    listen(canvas, 'webglcontextlost', event => { event.preventDefault(); this.dispose(); this.callbacks.onFailure(); });
    listen(document, 'visibilitychange', () => this.requestRender());
    listen(this.reducedMotion, 'change', () => { if (this.reducedMotion.matches) this.finishAnimation(); this.requestRender(); });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.visibility = new IntersectionObserver(([entry]) => { this.onScreen = entry.isIntersecting; this.requestRender(); });
    this.visibility.observe(canvas);
  }

  queueHover(x, y) {
    this.hoverPoint = [x, y];
    if (this.hoverFrame) return;
    this.hoverFrame = requestAnimationFrame(() => {
      this.hoverFrame = 0;
      this.setHover(this.pick(...this.hoverPoint));
    });
  }

  // ---------------------------------------------------------------- frames

  setQuality(quality) {
    this.quality = QUALITY[quality] ? quality : 'auto';
    const preset = QUALITY[this.quality];
    this.ratioCap = preset.ratio;
    const enabled = preset.shadows > 0;
    if (this.renderer.shadowMap.enabled !== enabled) this.scene.traverse(object => { if (object.material) object.material.needsUpdate = true; });
    this.renderer.shadowMap.enabled = enabled;
    if (enabled && this.sun.shadow.mapSize.x !== preset.shadows) {
      this.sun.shadow.mapSize.set(preset.shadows, preset.shadows);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
    this.scene.environment = this.quality === 'economy' ? null : this.environment;
    this.shadowsDirty = true;
    this.resize();
  }

  resize() {
    const { width, height } = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.ratioCap));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (this.frame || this.disposed) return;
    this.frame = requestAnimationFrame(time => this.tick(time));
  }

  tick(time) {
    this.frame = 0;
    if (this.disposed || document.hidden) return;
    const dt = this.lastTick ? Math.min(64, time - this.lastTick) : 16;
    this.lastTick = time;
    let active = this.tweens.step(time);
    if (this.velocity) {
      this.view.theta += this.velocity * dt;
      this.velocity *= Math.exp(-dt / 170);
      if (Math.abs(this.velocity) < 2e-5) this.velocity = 0;
      active = true;
    }
    const ambient = QUALITY[this.quality].ambient && this.onScreen !== false && !this.reducedMotion.matches;
    if (ambient) {
      this.island.windmill.rotation.z += dt * .0006;
      this.hintGem.rotation.y += dt * .002;
    }
    this.updateCamera();
    if (this.shadowsDirty && this.renderer.shadowMap.enabled) {
      this.renderer.shadowMap.needsUpdate = true;
      this.shadowsDirty = false;
    }
    const began = performance.now();
    this.renderer.render(this.scene, this.camera);
    this.frameCount++;
    this.lastCpuMs = performance.now() - began;
    if (active) this.adapt(dt);
    if (active) this.requestRender();
    else if (ambient) { this.lastTick = 0; setTimeout(() => this.requestRender(), 33); }
    else this.lastTick = 0;
  }

  /** In Auto quality, step down resolution, then shadows, when animation frames are consistently slow. */
  adapt(dt) {
    if (this.quality !== 'auto') return;
    this.frameTimes.push(dt);
    if (this.frameTimes.length < 40) return;
    const average = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    this.frameTimes = [];
    if (average < 30) return;
    if (this.ratioCap > 1) { this.ratioCap = 1; this.resize(); }
    else if (this.renderer.shadowMap.enabled) { this.renderer.shadowMap.enabled = false; this.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; }); }
  }

  get diagnostics() {
    return { renderer: 'three', frames: this.frameCount, cpuMs: this.lastCpuMs || 0, pixelRatio: this.renderer.getPixelRatio(),
      shadows: this.renderer.shadowMap.enabled, calls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.abort?.abort();
    this.resizeObserver?.disconnect();
    this.visibility?.disconnect();
    this.scene.traverse(object => {
      object.geometry?.dispose();
      for (const material of [].concat(object.material || [])) { material.map?.dispose(); material.dispose(); }
    });
    this.environment?.dispose();
    this.renderer.dispose();
  }
}
