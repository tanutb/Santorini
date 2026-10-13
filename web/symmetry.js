/**
 * The 5x5 island looks the same after any of its eight rotations and reflections.
 * The trained agent is not symmetric: left alone it opens on the same squares every game.
 * Each game picks one of these frames; the agent reads the board through it and its answer is
 * mapped back. Its strength is unchanged and its openings vary across all eight orientations.
 */
const TRANSFORMS = [
  (x, y) => [x, y], (x, y) => [4 - y, x], (x, y) => [4 - x, 4 - y], (x, y) => [y, 4 - x],
  (x, y) => [4 - x, y], (x, y) => [x, 4 - y], (x, y) => [y, x], (x, y) => [4 - y, 4 - x],
];

/** For each frame, where every square lands. */
export const FRAMES = TRANSFORMS.map(transform => Array.from({ length: 25 }, (_, cell) => {
  const [x, y] = transform(cell % 5, Math.floor(cell / 5));
  return y * 5 + x;
}));
const INVERSE = FRAMES.map(map => { const back = Array(25); map.forEach((to, from) => { back[to] = from; }); return back; });

export const randomFrame = () => Math.floor(Math.random() * FRAMES.length);

const frameOf = frame => (Number.isInteger(frame) && frame > 0 && frame < FRAMES.length ? frame : 0);

/** A game state as seen through `frame`. */
export function orient(state, frame) {
  const map = FRAMES[frameOf(frame)], heights = Array(25);
  state.heights.forEach((height, cell) => { heights[map[cell]] = height; });
  return { ...state, heights, workers: state.workers.map(pair => pair.map(cell => (cell >= 0 ? map[cell] : cell))) };
}

/** An action chosen in `frame`, mapped back onto the real board. Workers keep their numbers. */
export function unorient(action, frame) {
  const back = INVERSE[frameOf(frame)];
  if (action < 25) return back[action];
  const worker = Math.floor((action - 25) / 25);
  return 25 + worker * 25 + back[(action - 25) % 25];
}
