/** Geometry and materials for the island. Built once; the board only moves instances. */
import {
  BoxGeometry, BufferAttribute, CanvasTexture, CatmullRomCurve3, CircleGeometry, Color, ConeGeometry, CylinderGeometry,
  DoubleSide, ExtrudeGeometry, Group, IcosahedronGeometry, InstancedMesh, LatheGeometry, Mesh, MeshBasicMaterial,
  MeshStandardMaterial, Object3D, PlaneGeometry, RingGeometry, RoundedBoxGeometry, SRGBColorSpace, Shape, ShapeGeometry,
  SphereGeometry, TorusGeometry, Vector2, Vector3, mergeGeometries,
} from './vendor/three.js';

export const PALETTE = {
  stone: '#f4f2ec', trim: '#ffffff', window: '#2d5e94', cobalt: '#2453a6', tileA: '#e1e9cf', tileB: '#d0dcbb',
  plinth: '#d9d1bc', coral: '#d4583e', gold: '#efc04a', teal: '#1f8a8a', amber: '#e8742c', ink: '#10304d',
};
/** Top surface height of a tile and of each building level. */
export const LEVEL_TOP = [.1, .5, .86, 1.17];
const LEVELS = [{ width: .84, height: .4 }, { width: .7, height: .36 }, { width: .56, height: .31 }];
export const cellX = cell => cell % 5 - 2;
export const cellZ = cell => Math.floor(cell / 5) - 2;
export const surface = level => LEVEL_TOP[Math.min(level, 3)];

function painted(geometry, hex) {
  const color = new Color(hex), count = geometry.attributes.position.count, data = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) data.set([color.r, color.g, color.b], i * 3);
  geometry.setAttribute('color', new BufferAttribute(data, 3));
  return geometry;
}
const plain = geometry => (geometry.index ? geometry.toNonIndexed() : geometry);
const merge = parts => mergeGeometries(parts.map(plain), false);

/** A whitewashed storey with a slightly wider cap and blue shutters on each face. */
function storeyGeometry(level) {
  const { width, height } = LEVELS[level];
  const parts = [
    painted(new RoundedBoxGeometry(width, height - .035, width, 2, .035).translate(0, (height - .035) / 2, 0), PALETTE.stone),
    painted(new RoundedBoxGeometry(width + .05, .035, width + .05, 1, .012).translate(0, height - .0175, 0), PALETTE.trim),
  ];
  if (level < 2) {
    const size = level ? [.11, .13] : [.13, .17];
    for (let side = 0; side < 4; side++) {
      const shutter = new BoxGeometry(size[0], size[1], .02).translate(0, height * .46, width / 2 + .004);
      shutter.rotateY(side * Math.PI / 2);
      parts.push(painted(shutter, PALETTE.window));
    }
  }
  return merge(parts);
}

function domeGeometry() {
  return merge([
    painted(new CylinderGeometry(.27, .28, .05, 32).translate(0, .025, 0), PALETTE.trim),
    painted(new SphereGeometry(.25, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2).translate(0, .05, 0), PALETTE.cobalt),
    painted(new SphereGeometry(.035, 12, 8).translate(0, .3, 0), PALETTE.trim),
  ]);
}

/** A turned game piece. The gold player's pieces carry a brim so the two sides differ by shape too. */
export function workerGeometry(player) {
  const profile = [[0, 0], [.2, 0], [.21, .025], [.19, .05], [.14, .07], [.12, .1], [.115, .16], [.1, .24], [.085, .31],
    [.11, .335], [.115, .35], [.09, .37], [.07, .385]];
  for (let i = 0; i <= 10; i++) {
    const angle = -1.15 + i / 10 * (Math.PI / 2 + 1.15);
    profile.push([Math.max(0, Math.cos(angle) * .088), .463 + Math.sin(angle) * .088]);
  }
  const parts = [new LatheGeometry(profile.map(([x, y]) => new Vector2(x, y)), 28)];
  if (player === 1) parts.push(new TorusGeometry(.1, .018, 8, 28).rotateX(Math.PI / 2).translate(0, .43, 0));
  const geometry = merge(parts);
  geometry.computeVertexNormals();
  return geometry;
}

const stoneMaterial = () => new MeshStandardMaterial({ vertexColors: true, roughness: .78, metalness: 0 });

export function createPieces() {
  return {
    storeys: LEVELS.map((_, level) => storeyGeometry(level)),
    dome: domeGeometry(),
    material: stoneMaterial(),
    workers: [0, 1].map(workerGeometry),
    workerMaterials: [PALETTE.coral, PALETTE.gold].map(color => new MeshStandardMaterial({ color, roughness: .42, metalness: .02 })),
  };
}

/** Coastline radius at a bearing; the board plinth (corner radius ~4.1) always fits inside. */
const coast = angle => 4.85 + .32 * Math.sin(3 * angle + .5) + .18 * Math.sin(5 * angle + 1.3) + .1 * Math.sin(9 * angle);

function outline(scale = 1, points = 96) {
  const curve = new CatmullRomCurve3(Array.from({ length: 18 }, (_, i) => {
    const angle = i / 18 * Math.PI * 2;
    return new Vector3(Math.cos(angle) * coast(angle) * scale, -Math.sin(angle) * coast(angle) * scale, 0);
  }), true);
  return new Shape(curve.getSpacedPoints(points).slice(0, -1).map(p => new Vector2(p.x, p.y)));
}

/** Extruded terrain coloured in caldera strata from its height. */
function terrain(scale, top, depth, strata) {
  const geometry = new ExtrudeGeometry(outline(scale), { depth, steps: 7, bevelEnabled: true, bevelThickness: .1, bevelSize: .14, bevelSegments: 3, curveSegments: 4 });
  geometry.rotateX(-Math.PI / 2);
  geometry.computeBoundingBox();
  geometry.translate(0, top - geometry.boundingBox.max.y, 0);
  const position = geometry.attributes.position, normal = geometry.attributes.normal;
  const data = new Float32Array(position.count * 3), color = new Color();
  for (let i = 0; i < position.count; i++) {
    const y = position.getY(i), x = position.getX(i), z = position.getZ(i);
    if (normal.getY(i) > .7) color.set(strata.top);
    else {
      const t = (top - y + .04 * Math.sin(x * 2.1 + z * 1.7)) / (depth + .2);
      const band = strata.bands[Math.min(strata.bands.length - 1, Math.max(0, Math.floor(t * strata.bands.length)))];
      color.set(band);
    }
    data.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute('color', new BufferAttribute(data, 3));
  return new Mesh(geometry, new MeshStandardMaterial({ vertexColors: true, roughness: .95, flatShading: true }));
}

function seaTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 1024;
  const g = canvas.getContext('2d');
  const gradient = g.createRadialGradient(512, 512, 0, 512, 512, 512);
  [[0, '#7fd8d2'], [.1, '#4bb8c4'], [.22, '#2f93b5'], [.48, '#246f9f'], [.78, '#5d97b8'], [1, '#d8e9ee']]
    .forEach(([stop, color]) => gradient.addColorStop(stop, color));
  g.fillStyle = gradient;
  g.fillRect(0, 0, 1024, 1024);
  // Sun glints: a deterministic scatter so every visit sees the same sea.
  let seed = 7;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  g.fillStyle = 'rgba(255,255,255,.22)';
  for (let i = 0; i < 900; i++) {
    const angle = random() * Math.PI * 2, radius = 70 + random() * 360;
    g.fillRect(512 + Math.cos(angle) * radius, 512 + Math.sin(angle) * radius, 1 + random() * 3, 1);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function labelTexture(text) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const g = canvas.getContext('2d');
  g.fillStyle = 'rgba(16,48,77,.62)';
  g.font = '600 72px Commissioner, "Segoe UI", sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, 64, 68);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

/** Column letters along the near edge and row numbers along the left edge. */
export function createLabels() {
  const group = new Group(), geometry = new PlaneGeometry(.28, .28).rotateX(-Math.PI / 2);
  const add = (text, x, z) => {
    const mesh = new Mesh(geometry, new MeshBasicMaterial({ map: labelTexture(text), transparent: true, depthWrite: false }));
    mesh.position.set(x, .062, z);
    group.add(mesh);
  };
  for (let i = 0; i < 5; i++) {
    add(String.fromCharCode(65 + i), i - 2, 2.68);
    add(String(i + 1), -2.68, i - 2);
  }
  group.refresh = () => group.children.forEach((mesh, i) => {
    const text = i % 2 ? String(Math.floor(i / 2) + 1) : String.fromCharCode(65 + Math.floor(i / 2));
    mesh.material.map.dispose();
    mesh.material.map = labelTexture(text);
  });
  return group;
}

function scatter(geometry, material, transforms, shadows = true) {
  const mesh = new InstancedMesh(geometry, material, transforms.length), object = new Object3D();
  transforms.forEach(([x, y, z, sx, sy, sz, rotation = 0], i) => {
    object.position.set(x, y, z);
    object.scale.set(sx, sy, sz);
    object.rotation.set(0, rotation, 0);
    object.updateMatrix();
    mesh.setMatrixAt(i, object.matrix);
  });
  mesh.castShadow = shadows;
  mesh.receiveShadow = shadows;
  return mesh;
}

/** Fixed scenery: sea, caldera island, plinth, shrubs, a small village and a windmill. */
export function createIsland() {
  const group = new Group();
  const sea = new Mesh(new CircleGeometry(70, 96).rotateX(-Math.PI / 2), new MeshBasicMaterial({ map: seaTexture() }));
  sea.position.y = -.66;
  group.add(sea);
  for (const [scale, color, opacity, y] of [[1.2, '#9fe3dc', .55, -.645], [1.075, '#ffffff', .8, -.64]]) {
    const band = new Mesh(new ShapeGeometry(outline(scale, 160)).rotateX(-Math.PI / 2),
      new MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    band.position.y = y;
    group.add(band);
  }
  const lower = terrain(1.02, -.42, .32, { top: '#ece4cf', bands: ['#b8875f', '#7d5546', '#4f4545'] });
  const upper = terrain(.93, -.05, .5, { top: '#e4ddc6', bands: ['#e6dac0', '#c99568', '#a2593f', '#6d4a42'] });
  for (const mesh of [lower, upper]) { mesh.receiveShadow = true; mesh.castShadow = true; group.add(mesh); }

  const plinth = new Mesh(new RoundedBoxGeometry(5.75, .2, 5.75, 3, .08).translate(0, -.04, 0),
    new MeshStandardMaterial({ color: PALETTE.plinth, roughness: .9 }));
  plinth.receiveShadow = true;
  plinth.castShadow = true;
  group.add(plinth);

  const shrubs = [], cypresses = [];
  let seed = 11;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 60 && shrubs.length < 22; i++) {
    const angle = random() * Math.PI * 2;
    const edge = 2.95 / Math.max(Math.abs(Math.cos(angle)), Math.abs(Math.sin(angle)));
    const radius = edge + .3 + random() * Math.max(0, coast(angle) * .93 - edge - .55);
    if (radius > coast(angle) * .93 - .25 || (Math.sin(angle) < -.55 && Math.abs(Math.cos(angle)) < .6)) continue;
    const front = Math.sin(angle) > .45, size = (front ? .07 : .1) + random() * (front ? .05 : .08);
    shrubs.push([Math.cos(angle) * radius, -.03 + size * .45, Math.sin(angle) * radius, size, size * .75, size, random() * 3]);
    if (random() < .22 && Math.sin(angle) < .2) cypresses.push([Math.cos(angle) * radius + .2, .3, Math.sin(angle) * radius, 1, 1, 1]);
  }
  group.add(scatter(new IcosahedronGeometry(1, 1), new MeshStandardMaterial({ color: '#6f8a4a', roughness: .9, flatShading: true }), shrubs));
  group.add(scatter(new ConeGeometry(.11, .7, 7), new MeshStandardMaterial({ color: '#4f6b3e', roughness: .9, flatShading: true }), cypresses));

  // A small village behind the board (away from the default camera) keeps the play area unobstructed.
  const village = new Group(), whitewash = stoneMaterial();
  const houses = [[-.9, -3.55, .34, .28, .3], [-.45, -3.75, .26, .22, .26], [.15, -3.6, .3, .32, .28], [.7, -3.8, .24, .2, .24], [1.15, -3.5, .28, .26, .3]];
  for (const [x, z, w, h, d] of houses) {
    const house = new Mesh(merge([
      painted(new RoundedBoxGeometry(w, h, d, 1, .02).translate(0, h / 2, 0), PALETTE.stone),
      painted(new BoxGeometry(.06, .1, .01).translate(0, .05, d / 2 + .003), PALETTE.window),
    ]), whitewash);
    house.position.set(x, -.06, z);
    house.castShadow = house.receiveShadow = true;
    village.add(house);
  }
  const chapel = new Mesh(domeGeometry(), whitewash);
  chapel.scale.setScalar(.55);
  chapel.position.set(.15, .26, -3.6);
  chapel.castShadow = true;
  village.add(chapel);
  group.add(village);

  const mill = new Group();
  const tower = new Mesh(merge([
    painted(new CylinderGeometry(.16, .2, .62, 16).translate(0, .31, 0), PALETTE.stone),
    painted(new ConeGeometry(.19, .2, 16).translate(0, .72, 0), '#8a6a52'),
  ]), whitewash);
  tower.castShadow = true;
  mill.add(tower);
  const sails = new Group();
  for (let i = 0; i < 4; i++) {
    const sail = new Mesh(new BoxGeometry(.035, .42, .01).translate(0, .21, 0), new MeshStandardMaterial({ color: '#f7f3e8', roughness: .8, side: DoubleSide }));
    sail.rotation.z = i * Math.PI / 2 + .3;
    sail.castShadow = true;
    sails.add(sail);
  }
  sails.position.set(0, .58, .2);
  mill.add(sails);
  mill.position.set(-3.85, -.06, .45);
  mill.rotation.y = -Math.PI / 2.4;
  group.add(mill);
  group.windmill = sails;
  return group;
}

/** Flat markers that sit on the surface of a square. */
export function ringGeometry(inner, outer, segments = 48) {
  return new RingGeometry(inner, outer, segments).rotateX(-Math.PI / 2);
}

export function markerGeometry() {
  return mergeGeometries([plain(ringGeometry(.15, .205)), plain(new CircleGeometry(.055, 24).rotateX(-Math.PI / 2))], false);
}

export function setInstance(mesh, index, x, y, z, scale = 1, rotation = 0) {
  setInstance.object ??= new Object3D();
  const object = setInstance.object;
  object.position.set(x, y, z);
  object.scale.setScalar(scale);
  object.rotation.set(0, rotation, 0);
  object.updateMatrix();
  mesh.setMatrixAt(index, object.matrix);
}

