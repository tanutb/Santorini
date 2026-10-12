# Santorini

Classic Santorini on a 3D island. Play a friend online, pass and play on one screen, or take on an opponent trained with an original **PyTorch PPO-Clip** implementation.

## Start

On this machine the Python environment and trained agent are already in place. From PowerShell:

~~~powershell
.\start.ps1            # http://localhost:8000
.\start.ps1 -Port 8001 # any free port
~~~

On a fresh checkout (Python 3.11 or 3.12):

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\start.ps1
~~~

On macOS/Linux, use `.venv/bin/python -m server`.

The website is plain ES modules served by the Python server; there is no frontend build step. Three.js is vendored as a minified bundle in `web/vendor/three.js`. Node 22+ is needed only for the checks below and for regenerating that bundle.

## Play

- **Place** both workers. Coral moves first.
- **Move** one worker to a neighbouring square, diagonals included. You may climb up one level or drop down any number. Occupied squares and domes are off limits.
- **Build** next to the worker you moved: one more level, or a dome on top of level three.
- **Win** by stepping up onto level three, or when your opponent has no move they can complete.

On the board:

- Teal rings show where you can go.
- Hovering shows a ghost of the move, with its path, or of the build.
- Clicking an illegal square tells you why it is illegal.
- With one movable worker, it is selected for you.

Controls:

- **Turn:** drag the island, swipe sideways on touch, or press Q/E.
- **Zoom:** Ctrl + scroll, pinch, or the + and − buttons.
- **Keyboard:** focus the board, then use the arrow keys and Enter. The square-list button gives an accessible grid of buttons.

In games against the agent and in pass and play, the **Coach** panel offers:

- Alerts for a win available now, a square where the opponent could win next turn, and a build that would hand them a winning step.
- **Suggest a move**: the agent's top choice, and how often it would play that move itself.
- **Replay last action**.
- A position read, shown only when the critic is calibrated well enough (see below).

Online games have no coaching.

**Online play:** start a game, then share the room code or the copied invitation link. Each seat gets a private credential saved in its browser, so refreshing or reconnecting returns you to your seat. Rematches need both players.

**Graphics:**

- **Auto** caps resolution and steps it down, then switches off shadows, if animation frames run slow.
- **High** uses full resolution and adds a turning windmill.
- **Battery saver** turns shadows off.

The board draws only when something changes. Without WebGL, a lightweight 2D board takes over.

## The agent

The agent is a neural network trained with our own PPO-Clip code; no RL library is used. Illegal moves are masked out, and every move the agent makes comes from the network. If the checkpoint cannot be loaded, agent games are unavailable; the server never quietly swaps in a scripted opponent.

The served model is a residual convolutional network trained on 38 million decisions (2.7 million games) in parallel self-play. Its opponents were random, tactical and one-move-lookahead programs, frozen copies of itself, and the two earlier agents. **Agent strength** offers earlier agents from the same project as lighter opponents.

| 400 seat-balanced games each | Random | Simple tactician | One-move lookahead | Head-to-head |
| --- | --- | --- | --- | --- |
| **Full strength** (served, 38.0M decisions) | 100% | 99% | 100% | 99.8% vs the previous agent, 100% vs the first |
| Developing (update 100, 1.6M decisions) | 99.8% | 96.8% | 99.8% | Full strength beats it 78.5% |
| Early (the first agent, MLP, 0.45M decisions) | 96.5% | 73.5% | 49.3% | |

The served model was chosen from twelve archived snapshots by round-robin: it won 83% of its games against the other eleven on average, and at least 68% against each one. Moves and hints take about 1 ms on a CPU.

These results are against fixed programs, not a rating against people.

The position read is the critic's view of the board, calibrated on self-play. For the served model the calibration beat a constant guess by only 3% (Brier score), so the read is hidden. It appears automatically for a model whose calibration clears 10%. See [the PPO notes](docs/ppo.md) for the networks, objective, opponents and the position-read calibration.

### Train

~~~powershell
# Parallel training: rollout workers on CPU, PPO updates on the GPU when available.
.\.venv\Scripts\python.exe -m server.rl.train --architecture conv --workers 10 --envs 12 --steps 128 `
  --updates 2400 --learning-rate 0.0005 --final-learning-rate 0.00005 --minibatch 2048 `
  --mix "random=0.1,tactical=0.15,lookahead=0.35" --fixed-opponent checkpoints/candidate/ppo.pt `
  --output checkpoints/conv/ppo.pt
# Continue a run (weights, optimizer and config are restored):
.\.venv\Scripts\python.exe -m server.rl.train --resume checkpoints/conv/ppo.pt --updates 500 --output checkpoints/conv/ppo.pt
~~~

- Snapshots are archived every 100 updates under `snapshots/`, and metrics go to `training.jsonl`.
- On Windows every process that imports PyTorch commits about 1.5 GB of memory, so keep `--workers` within what your page file allows.
- Train to a separate `--output` and promote a checkpoint only after evaluating it. The server reloads `checkpoints/ppo.pt` automatically when the file changes.

### Evaluate and promote

~~~powershell
# Head-to-head and baselines, seat-balanced, in parallel, with 95% intervals:
.\.venv\Scripts\python.exe -m server.rl.arena checkpoints/conv/ppo.pt --against random tactical lookahead checkpoints/ppo.pt --games 400
# After copying a checkpoint to checkpoints/ppo.pt, record its evaluation and fit the position read:
.\.venv\Scripts\python.exe -m server.rl.evaluate --games 400
.\.venv\Scripts\python.exe -m server.rl.calibration checkpoints/ppo.pt --fit --games 1200
~~~

The website shows an evaluation or a position read only when its step count matches the loaded weights.

## Verify

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q   # rules, API, PPO, trainer, arena, calibration, levels
npm run check                             # syntax of every module
npm test                                  # coaching rules, camera maths, tweens, 2D board geometry
npm run smoke -- http://127.0.0.1:8000    # real HTTP + WebSocket online match, reconnect, agent game
~~~

With Playwright for Python installed, `python scripts/browser_play.py http://127.0.0.1:8000 shots/game desktop` plays a complete game against the agent through real clicks on the 3D board, checking every pick against the server.

After changing which Three.js classes `web/three-board.js` or `web/island-models.js` import, run `npm install` and then `npm run vendor`.

## Deploy for internet play

The website and Python server share one origin. Deploy to a host that runs Docker or Python and supports WebSockets:

~~~sh
docker compose up --build -d
~~~

Persist `/app/data` to keep rooms, and put HTTPS in front of port 8000; the client switches to secure WebSockets on HTTPS. Keep the public Host header through the proxy for the WebSocket origin check.

| Setting | Default | Purpose |
| --- | --- | --- |
| PORT | 8000 | HTTP and WebSocket port |
| HOST | 0.0.0.0 | Listening interface |
| SANTORINI_DB | data/rooms.sqlite3 | Room database |
| SANTORINI_MODEL | checkpoints/ppo.pt | Served checkpoint; `levels.json` beside it lists lighter strengths |

Run **one process with one worker**: live broadcasts and room locks are in memory, while state persists in SQLite. Rooms untouched for seven days are removed at startup. No public host has been set up for this project.

## Project map

| Path | Responsibility |
| --- | --- |
| web/app.js | Lobby, sessions, WebSockets, turn flow, coach panel |
| web/three-board.js, web/island-models.js | Three.js scene, animation, picking, camera |
| web/board.js | 2D fallback board |
| web/coach.js, web/orbit.js, web/tween.js | Pure helpers: rule facts, camera maths, tweens |
| server/game.py | Authoritative rules and the observation/action encoding |
| server/app.py | HTTP API, seats, WebSockets, agent turns, hints, position reads |
| server/inference.py | Checkpoint loading, agent strengths, hints, calibrated reads |
| server/rl/ | Networks, PPO, parallel training, baselines, arena, calibration |
| checkpoints/ | Served weights, evaluations, earlier agents, strength ladder |

API documentation is at `/docs`. Seat tokens are returned once and stored only as hashes on the server.

Santorini was designed by Gordon Hamilton. This is an unofficial fan implementation with original geometry and no game artwork. [Publisher rulebook](https://s3.amazonaws.com/spinmasterstudioscdn/helpshift/Santorini.pdf).
