/** Camera maths shared by the renderer and tests: spherical orbit, framing and keyboard steps. */
export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const DEFAULT_VIEW = Object.freeze({ theta: .3, phi: .82, zoom: 1 });
export const PHI_LIMITS = Object.freeze([.3, 1.22]);
export const ZOOM_LIMITS = Object.freeze([.62, 1.3]);

/** Camera position for azimuth `theta` (around +y, 0 = looking from +z) and polar `phi` (from vertical). */
export function orbitPosition(target, distance, theta, phi) {
  return [
    target[0] + distance * Math.sin(phi) * Math.sin(theta),
    target[1] + distance * Math.cos(phi),
    target[2] + distance * Math.sin(phi) * Math.cos(theta),
  ];
}

/** Distance that fits a sphere of `radius` inside a perspective frustum of vertical `fov` degrees. */
export function fitDistance(radius, fov, aspect) {
  const vertical = fov * Math.PI / 360;
  const horizontal = Math.atan(Math.tan(vertical) * aspect);
  return radius / Math.sin(Math.min(vertical, horizontal));
}

/** Snap an azimuth to the nearest quarter turn, returned as 0..3. */
export function quarter(theta) {
  return ((Math.round(theta / (Math.PI / 2)) % 4) + 4) % 4;
}

const SCREEN_STEPS = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };

/** Move a keyboard cursor so arrow keys follow the screen, whatever way the island is turned. */
export function stepCell(cell, key, theta) {
  const screen = SCREEN_STEPS[key];
  if (!screen) return cell;
  if (cell < 0) return 12;
  const turns = quarter(theta);
  let [dx, dz] = screen;
  for (let i = 0; i < turns; i++) [dx, dz] = [dz, -dx];
  const x = clamp(cell % 5 + dx, 0, 4), z = clamp(Math.floor(cell / 5) + dz, 0, 4);
  return z * 5 + x;
}
