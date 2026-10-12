# Santorini design

Classic two-player Santorini on a 3D island, with invite-code online play, pass and play, and a custom PyTorch PPO-Clip opponent.

## Architecture
- **Server:** FastAPI owns every state transition. SQLite stores rooms, hashed seat tokens and history; WebSockets broadcast the authoritative state. One pure Python rules engine (`server/game.py`) serves play, training and evaluation.
- **Browser:** buildless ES modules served from `web/`. Three.js is vendored as a tree-shaken bundle (`web/vendor/three.js`, about 150 KB gzipped) and loaded lazily. The UI works while it downloads. If WebGL is unavailable or the context is lost, a Canvas board takes over with the same state.

## Rendering
- **Static scene:** sea, caldera island, plinth, village, windmill, labels.
- **Board:** 25 instanced tiles, and three instanced storey meshes plus an instanced dome mesh.
- **Workers:** four lathe-turned meshes. The gold side has a brim, so the two sides differ by shape as well as colour.
- **Cost:** roughly 60 draw calls and 50k triangles for a full board.
- **On demand:** frames are drawn only after a change, during tweens or camera inertia. Idle frames measured zero. High quality adds a throttled ambient loop that turns the windmill.
- **Shadows:** re-rendered only when pieces move, never while orbiting.
- **Quality presets:** pixel-ratio cap and shadow-map size. In Auto, the renderer steps resolution and then shadows down when animation frames run slow.
- **Animation:** each revision animates in order: a hop arc for moves, a drop for placement, and a bounce-drop with dust for builds. A win spins the winning worker and releases confetti. Several updates at once (for example after a reconnect) snap instead of replaying, and reduced motion removes all tweens.
- **Picking:** raycasts against invisible per-square columns and slim worker volumes. If the first hit is not playable but a playable square lies behind it, that square wins, so towers never block a legal click. A ground square directly behind a level-three tower that is itself playable can be fully hidden; turn the island, or use the keyboard cursor or the square list.

## Interaction
- **Camera:** drag to orbit. On touch, a horizontal swipe orbits and vertical swipes scroll the page. Pinch and Ctrl + scroll zoom; plain scroll keeps scrolling the page, with a hint. Buttons and the Q/E keys turn the island.
- **Keyboard:** arrow keys move a cursor relative to the screen, whatever the camera angle; Enter plays.
- **Selection:** a worker is selected automatically when it is the only one that can move. Clicking a square only one worker can reach moves that worker.
- **Illegal clicks:** they are explained from the rules, for example "Too high. Workers climb one level at a time."
- **Previews:** a ghost worker and arc path for moves, a translucent storey for builds. Risky builds are tinted amber.
- **Coach (agent and pass-and-play games only):**
  - immediate wins (gold star rings);
  - opponent winning steps (amber hexagons);
  - builds that would hand the opponent a winning step;
  - a PPO hint showing how often the agent itself plays that move;
  - replay of the last action;
  - a calibrated position read that only calls clear leads. It is shown only when the calibration beats a constant guess by 10%; the served model's does not, so the read is hidden.

  Online games get no assistance, so neither player has an advantage.
- **Feedback:** each action is announced on the board and in an ARIA live region. Synthesised sound cues are off by default. A result card appears on the board at the end of a game.

## Presentation
- **Palette:** cool whitewash `#f5f7f6`, Aegean ink `#10304d`, dome cobalt `#2453a6`, sea teal for playable squares, and amber for danger. Coral `#d4583e` and gold `#efc04a` workers differ in lightness, so colour-blind players can tell them apart. The caldera reds appear only in the island's cliffs.
- **Type:** Marcellus (inscriptional display) and Commissioner (humanist UI sans), both self-hosted.
- **Layout:** the board fills the viewport beside one 360 px sidebar. On narrow screens the board comes first and the panels stack below.

## Agent strengths
Lighter opponents are earlier agents from this project, listed in `checkpoints/levels.json`: the first MLP agent (Early) and update 100 of the conv run (Developing). Each is labelled with its training size and measured win rate. Hints and the position read always use the strongest model.

## Hosting
One process and one worker: live broadcasts and per-room locks are in memory, while state persists in SQLite. Deploy with Docker on a host that supports WebSockets.
