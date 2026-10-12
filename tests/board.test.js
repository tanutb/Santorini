import test from 'node:test';
import assert from 'node:assert/strict';
import { IslandBoard, demoGame } from '../web/board.js';

globalThis.window = { devicePixelRatio: 1 };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };

function makeBoard(width = 700, height = 490) {
  const gradient = { addColorStop() {} };
  const context = new Proxy({}, { get(target, key) {
    if (key === 'createLinearGradient' || key === 'createRadialGradient') return () => gradient;
    return target[key] ?? (() => {});
  } });
  const canvas = { style: {}, parentElement: {}, getContext: () => context,
    getBoundingClientRect: () => ({ width, height, left: 0, top: 0 }), addEventListener() {} };
  return new IslandBoard(canvas, () => {}, () => {});
}

test('all 25 empty squares are pickable in every camera orientation', () => {
  const board = makeBoard();
  board.setState({ heights: Array(25).fill(0), workers: [[-1,-1],[-1,-1]] });
  for (let orientation = 0; orientation < 4; orientation++) {
    for (let cell = 0; cell < 25; cell++) {
      const [x,y] = board.project(cell % 5 - 2, .266, Math.floor(cell / 5) - 2);
      assert.equal(board.pick({clientX:x,clientY:y}), cell);
    }
    board.rotate(1);
  }
});

test('towers and workers produce finite visible geometry on desktop and mobile', () => {
  for (const [width,height] of [[700,490],[330,360],[1100,550]]) {
    const board = makeBoard(width,height);
    board.setState(demoGame(), [4,5,10], 8, [8,15]);
    assert.ok(board.surfaces.length > 100);
    for (const surface of board.surfaces) {
      assert.ok(surface.cell >= 0 && surface.cell < 25);
      for (const [x,y] of surface.points) {
        assert.ok(Number.isFinite(x) && Number.isFinite(y));
        assert.ok(x >= 0 && x <= width, 'geometry stays inside canvas horizontally');
        assert.ok(y >= 0 && y <= height, 'geometry stays inside canvas vertically');
      }
    }
    assert.equal(board.pick({clientX:-10,clientY:-10}), -1);
  }
});

test('camera reset restores the initial projection', () => {
  const board = makeBoard();
  const initial = board.project(2,1,1);
  board.rotate(1); assert.notDeepEqual(board.project(2,1,1),initial);
  board.resetView(); assert.deepEqual(board.project(2,1,1),initial);
});
