"""PPO-Clip and generalized advantage estimation implemented in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from server.rl.model import ActorCritic


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    epochs: int = 4
    minibatch: int = 256
    max_grad_norm: float = 0.5
    target_kl: float = 0.02


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                terminated: torch.Tensor, boundaries: torch.Tensor,
                gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap time limits, but never propagate GAE across reset boundaries."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]).float() - values[t]
        running = delta + gamma * gae_lambda * (~boundaries[t]).float() * running
        advantages[t] = running
    return advantages, advantages + values


def clipped_policy_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                        advantages: torch.Tensor, clip: float) -> torch.Tensor:
    ratio = (log_probs - old_log_probs).exp()
    ordinary = ratio * advantages
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages
    return -torch.minimum(ordinary, clipped).mean()


def update(model: ActorCritic, optimizer: torch.optim.Optimizer,
           batch: dict[str, torch.Tensor], config: PPOConfig) -> dict[str, float]:
    """Reuse rollout masks for both policies so probability ratios are valid."""
    advantage = batch["advantages"]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    count = len(advantage)
    metrics: list[dict[str, float]] = []
    stopped = False
    for _ in range(config.epochs):
        order = torch.randperm(count, device=advantage.device)
        for ids in order.split(config.minibatch):
            dist, values = model(batch["observations"][ids], batch["masks"][ids])
            log_probs = dist.log_prob(batch["actions"][ids])
            difference = log_probs - batch["log_probs"][ids]
            with torch.no_grad():
                kl = ((difference.exp() - 1) - difference).mean()
            if kl.item() > config.target_kl:
                stopped = True
                break
            policy = clipped_policy_loss(log_probs, batch["log_probs"][ids], advantage[ids], config.clip)
            value = (values - batch["returns"][ids]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy + config.value_coefficient * value - config.entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss; refusing to save invalid weights.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append({"policy_loss": policy.item(), "value_loss": value.item(), "entropy": entropy.item(), "kl": kl.item()})
        if stopped:
            break
    result = {key: sum(row[key] for row in metrics) / max(len(metrics), 1) for key in ("policy_loss", "value_loss", "entropy", "kl")}
    result["early_stopped"] = float(stopped)
    return result
