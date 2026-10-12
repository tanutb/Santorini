"""Categorical actor-critics for masked Santorini actions; no reinforcement-learning framework."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from server.game import ACTION_COUNT, OBSERVATION_SIZE


def masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    if not masks.any(dim=-1).all():
        raise ValueError("A policy decision requires at least one legal action.")
    return Categorical(logits=logits.masked_fill(~masks.bool(), -1e9))


class ActorCritic(nn.Module):
    """Two-layer MLP over the flat 230-feature observation (checkpoint format 1)."""

    architecture = "mlp"

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.trunk = nn.Sequential(nn.Linear(OBSERVATION_SIZE, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, ACTION_COUNT)
        self.critic = nn.Linear(hidden, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @property
    def spec(self) -> dict:
        return {"architecture": "mlp", "hidden": self.hidden}

    def forward(self, observations: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        encoded = self.trunk(observations)
        return masked_distribution(self.actor(encoded), masks), self.critic(encoded).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 3, padding=1)
        self.second = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.second(torch.relu(self.first(x))))


class ConvActorCritic(nn.Module):
    """Residual CNN on the 5x5 island.

    The flat observation is reshaped into planes: five height levels, four worker
    positions (mine then theirs), five broadcast phase/selection flags and a constant
    plane that marks the board edge under zero padding. The policy head emits one
    logit plane per action family, matching the 75-action layout:
    place/build target, worker-one move target, worker-two move target.
    """

    architecture = "conv"
    PLANES = 15

    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(nn.Conv2d(self.PLANES, channels, 3, padding=1), nn.ReLU())
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Conv2d(channels, 3, 1)
        self.value = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.normal_(self.policy.weight, std=0.01)
        nn.init.zeros_(self.policy.bias)

    @property
    def spec(self) -> dict:
        return {"architecture": "conv", "channels": self.channels, "blocks": self.blocks}

    @staticmethod
    def planes(observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        board = observations[:, :225].reshape(batch, 9, 5, 5)
        flags = observations[:, 225:230].reshape(batch, 5, 1, 1).expand(batch, 5, 5, 5)
        edge = torch.ones(batch, 1, 5, 5, dtype=observations.dtype, device=observations.device)
        return torch.cat((board, flags, edge), dim=1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.tower(self.stem(self.planes(observations)))
        logits = self.policy(features).flatten(1)
        return masked_distribution(logits, masks), self.value(features).squeeze(-1)


def create_model(spec: dict | None = None) -> nn.Module:
    spec = dict(spec or {"architecture": "mlp"})
    architecture = spec.pop("architecture", "mlp")
    if architecture == "mlp":
        return ActorCritic(**spec)
    if architecture == "conv":
        return ConvActorCritic(**spec)
    raise ValueError(f"Unknown model architecture: {architecture}")


def checkpoint_spec(checkpoint: dict) -> dict:
    """Format 1 stored only the MLP hidden width; format 2 stores a full model spec."""
    if checkpoint.get("format_version") == 1:
        return {"architecture": "mlp", "hidden": checkpoint["hidden"]}
    if checkpoint.get("format_version") == 2:
        return dict(checkpoint["model_spec"])
    raise ValueError("Unsupported agent checkpoint.")


def load_model(checkpoint: dict) -> nn.Module:
    model = create_model(checkpoint_spec(checkpoint))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
