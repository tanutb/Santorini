import test from 'node:test';
import assert from 'node:assert/strict';
import { fitDistance, orbitPosition, quarter, stepCell } from '../web/orbit.js';

test('orbit places the camera on a sphere around the target', () => {
  const [x, y, z] = orbitPosition([0, 0, 0], 10, 0, Math.PI / 2);
  assert.ok(Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9 && Math.abs(z - 10) < 1e-9);
  const p = orbitPosition([1, 2, 3], 5, .7, .9);
  assert.ok(Math.abs(Math.hypot(p[0] - 1, p[1] - 2, p[2] - 3) - 5) < 1e-9);
});

test('portrait viewports need a more distant camera', () => {
  assert.ok(fitDistance(4, 30, .6) > fitDistance(4, 30, 1.6));
  assert.equal(fitDistance(4, 30, 2), fitDistance(4, 30, 3), 'wide screens are limited by the vertical field of view');
});

test('arrow keys follow the screen whichever way the island is turned', () => {
  assert.equal(stepCell(-1, 'ArrowUp', 0), 12, 'the cursor starts in the centre');
  assert.equal(stepCell(12, 'ArrowUp', 0), 7);
  assert.equal(stepCell(12, 'ArrowRight', 0), 13);
  assert.equal(stepCell(12, 'ArrowUp', Math.PI / 2), 11, 'looking along -x, up moves toward column A');
  assert.equal(stepCell(12, 'ArrowRight', Math.PI / 2), 7);
  assert.equal(stepCell(12, 'ArrowUp', Math.PI), 17);
  assert.equal(stepCell(0, 'ArrowLeft', 0), 0, 'the cursor stays on the board');
  assert.equal(quarter(-Math.PI / 2), 3);
});
