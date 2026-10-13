"""PPO-Clip following OpenAI Spinning Up's pseudocode, implemented in PyTorch.

For each batch of trajectories (Algorithm 1, lines 4-7):
  4. rewards-to-go R_t, the targets for the value network;
  5. GAE-lambda advantage estimates from the current value network;
  6. the policy update: Adam steps on the clipped objective, stopping early once the approximate
     KL from the policy that played exceeds 1.5 x `target_kl`;
  7. the value fit: Adam steps on the mean-squared error to R_t.

How lines 6 and 7 spend their steps is a setting. By default each makes `epochs` shuffled passes
over the batch in minibatches of `minibatch` decisions, the cheap schedule the project's first
agents used (4 x 8 steps of 2,048). With `minibatch` = 0 they instead take Spinning Up's own
default of `train_pi_iters` / `train_v_iters` full-batch steps (80 each), which costs about
ten times more for this game. Illegal moves are masked out of the policy (see model.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from server.rl.model import SeparateActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.01
    epochs: int = 4
    minibatch: int = 2048
    train_policy: bool = True


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                terminated: torch.Tensor, boundaries: torch.Tensor,
                gamma: float = 0.99, gae_lambda: float = 0.97) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE-lambda advantages, computed backwards along each stream. Time limits and rollout ends
    bootstrap from the next value; the recursion never crosses an episode boundary.
    Also returns advantages + values (the lambda-return)."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1:], device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]).float() - values[t]
        running = delta + gamma * gae_lambda * (~boundaries[t]).float() * running
        advantages[t] = running
    return advantages, advantages + values


def rewards_to_go(rewards: torch.Tensor, next_values: torch.Tensor, terminated: torch.Tensor,
                  boundaries: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Discounted rewards from each decision to the end of its episode (Spinning Up's R_t).
    Where the data stops before the game does (the rollout ends, or the 400-action limit), the
    sum is completed with the value of the next position, as Spinning Up does with `last_val`."""
    returns = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        bootstrap = next_values[t] * (~terminated[t]).float()
        tail = bootstrap if t == len(rewards) - 1 else torch.where(boundaries[t], bootstrap, returns[t + 1])
        returns[t] = rewards[t] + gamma * tail
    return returns


def clipped_policy_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                        advantages: torch.Tensor, clip: float) -> torch.Tensor:
    """Minus the PPO-Clip objective (optimisers minimise)."""
    ratio = (log_probs - old_log_probs).exp()
    ordinary = ratio * advantages
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages
    return -torch.minimum(ordinary, clipped).mean()


def step_schedule(count: int, config: PPOConfig, iterations: int, device) -> list[torch.Tensor | None]:
    """The decisions each gradient step uses: shuffled minibatches for `epochs` passes, or
    (with `minibatch` = 0) the whole batch, marked None, for `iterations` steps."""
    if config.minibatch <= 0:
        return [None] * iterations
    steps = []
    for _ in range(config.epochs):
        steps.extend(torch.randperm(count, device=device).split(config.minibatch))
    return steps


def update(model: SeparateActorCritic, optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
           batch: dict[str, torch.Tensor], config: PPOConfig) -> dict[str, float]:
    """Lines 6 and 7 of the pseudocode on one batch. The stored masks are reused so ratios compare like with like."""
    pi_optimizer, v_optimizer = optimizers
    observations, masks, actions = batch["observations"], batch["masks"], batch["actions"]
    old_log_probs, returns = batch["log_probs"], batch["returns"]
    advantage = batch["advantages"]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    count = len(advantage)
    take = lambda tensor, ids: tensor if ids is None else tensor[ids]

    # Line 6: policy steps until the policy has moved far enough from the one that played.
    with torch.no_grad():
        start = model.pi(observations, masks)
        policy_before = clipped_policy_loss(start.log_prob(actions), old_log_probs, advantage, config.clip).item()
        entropy = start.entropy().mean().item()
    # With train_policy off (refitting only the value network), line 6 is skipped.
    policy_schedule = step_schedule(count, config, config.train_pi_iters, observations.device) if config.train_policy else []
    steps, kl, stopped = 0, 0.0, False
    for ids in policy_schedule:
        old = take(old_log_probs, ids)
        log_probs = model.pi(take(observations, ids), take(masks, ids)).log_prob(take(actions, ids))
        loss = clipped_policy_loss(log_probs, old, take(advantage, ids), config.clip)
        kl = (old - log_probs).mean().item()
        if kl > 1.5 * config.target_kl:
            stopped = True
            break
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy loss; refusing to save invalid weights.")
        pi_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pi_optimizer.step()
        steps += 1
    with torch.no_grad():
        ratio = (model.pi(observations, masks).log_prob(actions) - old_log_probs).exp()
        clip_fraction = ((ratio - 1).abs() > config.clip).float().mean().item()

    # Line 7: regression of the value network on the rewards-to-go.
    with torch.no_grad():
        value_before = (model.v(observations) - returns).square().mean().item()
    value_schedule = step_schedule(count, config, config.train_v_iters, observations.device)
    for ids in value_schedule:
        loss = (model.v(take(observations, ids)) - take(returns, ids)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite value loss; refusing to save invalid weights.")
        v_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        v_optimizer.step()
    with torch.no_grad():
        values = model.v(observations)
        value_after = (values - returns).square().mean().item()
        # A value network whose ReLUs have all died predicts one number everywhere; this shows it.
        value_spread = values.std(unbiased=False).item()
    return {"policy_loss": policy_before, "value_loss": value_before, "entropy": entropy, "kl": kl,
            "clip_fraction": clip_fraction, "policy_steps": float(steps), "early_stopped": float(stopped),
            "step_limit": float(len(policy_schedule)), "value_steps": float(len(value_schedule)),
            "value_loss_after": value_after, "value_spread": value_spread}
