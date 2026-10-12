# Custom PPO-Clip

The training code is an original implementation that uses PyTorch only for tensors, gradients and Adam. Conceptual references: the [PPO paper](https://arxiv.org/abs/1707.06347), [Spinning Up PPO-Clip](https://spinningup.openai.com/en/latest/algorithms/ppo.html), and the [Higgsfield notebook](https://github.com/higgsfield-ai/higgsfield/blob/main/higgsfield/rl/rl_adventure_2/3.ppo.ipynb). Santorini uses a masked categorical policy instead of the notebook's continuous Pendulum distribution.

## Observations and actions

Each observation has 230 values, seen from the player to act:

- 125 values: five one-hot height planes, including domes.
- 100 values: four worker planes, the mover's two workers first, then the opponent's.
- 5 values: three phase flags (place, move, build) and two selected-worker flags.

| Phase | Action encoding |
| --- | --- |
| Place | Cell 0–24; fills the next unplaced worker |
| Move | 25 + worker × 25 + target cell |
| Build | Cell 0–24 |

Cells run row-major from A1 (0) to E5 (24). The rules engine builds every legal-action mask. Masked logits are set to −1e9, so illegal actions have zero probability; an all-false mask raises an error. Both sampling and the gradient update use each observation's stored mask.

## Networks

Two architectures share the same interface (`model.py`):

- **mlp** (checkpoint format 1, the original agent): two 128-unit tanh layers, a 75-logit actor and a scalar critic.
- **conv** (format 2, the current agent): the observation is reshaped into 15 planes on the 5×5 board. The planes are five heights, four worker positions, five broadcast flags, and a constant plane that marks the edge under zero padding.
  - A 3×3 stem and four residual blocks of 64 channels follow. The last convolution in each block is zero-initialised, so every block starts as the identity.
  - The policy head is a 1×1 convolution with three output planes, one per action family: place/build target, worker-one move target, and worker-two move target. Flattening the planes gives exactly the 75-action layout.
  - The value head is a 1×1 convolution followed by a two-layer MLP.

Convolutions see neighbourhoods directly, which is what Santorini's rules are about. The flat MLP has to learn adjacency from scratch.

## Reward perspective

An episode has one learner seat, which alternates between episodes. When control passes to the opponent, the environment plays the opponent's actions until control returns or the game ends. Every stored observation and value therefore belongs to the learner.

Rewards are +1 for a win, −1 for a loss and 0 otherwise; there is no reward shaping. Episodes stop at 400 actions:

- Truncation bootstraps the value but stops the GAE recursion at the reset.
- A terminal state bootstraps zero.
- The final decision of a rollout bootstraps its successor's value.

## Objective

~~~text
ratio = exp(new_log_probability - old_log_probability)
policy_loss = -mean(min(ratio * advantage, clamp(ratio, 1-ε, 1+ε) * advantage))
value_loss = mean((value - return)^2)
loss = policy_loss + 0.5 * value_loss - c_entropy * entropy
~~~

The defaults are ε = 0.2, γ = 0.99, λ = 0.95, and Adam with ε = 1e-5. Advantages are normalised per batch. Updates run shuffled minibatch epochs with the gradient norm clipped to 0.5, and stop early if the approximate KL exceeds 0.02. The learning rate can anneal linearly (`--learning-rate`, `--final-learning-rate`). Every update's losses, KL, early-stop flag and per-opponent win rates go to `training.jsonl`.

## Parallel collection

`train.py` spawns rollout workers (`--workers`). Each worker steps `--envs` environments in lockstep, with one batched forward pass per step. Each update:

1. The learner sends the current weights to every worker, plus any new frozen snapshots.
2. The workers collect `--steps` decisions per environment.
3. GAE is computed per stream on the learner.
4. The PPO update runs on `--device`, which is CUDA when available.

Successor values reuse the next decision's value, so no extra forward pass is needed. Weights travel between processes as numpy arrays. Torch tensors would be shared through memory-mapped files, and on Windows those exhausted the page file (error 1455).

On Windows every process that imports PyTorch commits about 1.5 GB, because its CUDA libraries are preloaded. Size `--workers` to fit the machine's commit limit.

## Opponents

At the start of each episode, an environment draws its opponent from `--mix`:

- **random**: uniform legal moves.
- **tactical**: wins when it can, climbs, and domes immediate threats.
- **lookahead**: a one-ply complete-turn search. It wins at once when possible, never leaves an immediate winning reply when it can avoid it, and scores positions by height, mobility and threats.
- **frozen** (the remaining probability): a snapshot of the learner from the last ten snapshots (every `--snapshot-every` updates), or a fixed checkpoint passed with `--fixed-opponent`. The current run includes both earlier MLP agents.

The opponent stays fixed for the whole episode.

## Evaluation

`arena.py` plays seat-balanced matches in parallel and reports wins, losses, draws, per-seat results, mean game length and a 95% Wilson interval. `evaluate.py` runs the served checkpoint against the three baselines and writes `ppo.evaluation.json`; the website shows it only when its step count matches the loaded weights. The arena samples from the policy exactly as the server does. These results measure play against fixed programs, not human strength.

## The critic as a position read

`calibration.py` tests whether the critic can be shown to players. It can't be shown directly. The critic predicts the learner's return against its training opponents, so it is optimistic for whoever moves, and its raw Brier score was worse than a constant guess.

Scoring the board from both sides and taking the difference cancels that bias. A logistic fit on self-play outcomes then maps the difference to a win chance between evenly matched players. The website shows the read only when the calibration file matches the loaded weights and its held-out Brier score beats a constant guess by at least 10%. Even then it names a clear leader or says the game is even; it never shows a percentage.

| Model | Raw critic Brier | Symmetric | Calibrated (held out) | Constant |
| --- | --- | --- | --- | --- |
| Conv at update 1,080, vs lookahead | 0.260 | 0.160 | n/a | 0.250 |
| Conv at update 1,080, self-play | 0.285 | 0.227 | 0.228 | 0.250 |
| Served model, self-play (1,200 games, 17,003 held-out positions) | 0.284 | 0.245 | 0.242 | 0.250 |

The served model's calibrated read is well calibrated, but it rarely leaves the 30–70% range: between two copies of a strong agent, the critic does not predict who wins. It is therefore hidden.

## Checkpoint contract

`ppo.pt` contains:

- `format_version`: 2.
- `model_spec`: the architecture and its sizes.
- `model`: the state dict.
- `optimizer`: the optimizer state, or none for archived snapshots.
- The PPO `config` and `metadata`.

It is loaded with `weights_only=True` and written atomically through a temporary file. Format 1 (MLP with `hidden`) still loads. Resume restores the weights, optimizer and config. It starts fresh environments and a new opponent pool, so it is not a bit-for-bit continuation.
