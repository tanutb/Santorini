"""Distilling the network-plus-search agent into the network."""
import numpy as np
import torch

from server.game import Game
from server.rl.distill import distill, targets
from server.rl.model import SeparateActorCritic
from server.rl.ppo import PPOConfig
from server.rl.train import save_checkpoint


def ladder() -> Game:
    game = Game(workers=[[24, 0], [17, 23]], current=1, phase="move", turn=8)
    for cell, height in {0: 2, 1: 2, 19: 2, 23: 1}.items():
        game.heights[cell] = height
    return game


def test_targets_keep_the_network_where_the_search_allows_everything():
    torch.manual_seed(0)
    model = SeparateActorCritic(channels=8, blocks=1, value_channels=8, value_blocks=1)
    game = ladder()
    observations = torch.tensor([game.observation()] * 2)
    legal = torch.tensor([game.mask()] * 2)
    allowed = legal.clone()
    allowed[1] = False
    allowed[1, [36, 37]] = True
    target = targets(model, observations, legal, allowed).exp()
    probabilities = model.pi(observations, legal).probs.detach()
    assert torch.allclose(target[0], probabilities[0], atol=1e-6)
    assert torch.allclose(target[1, [36, 37]], probabilities[1, [36, 37]] / probabilities[1, [36, 37]].sum(), atol=1e-6)
    assert target[1].sum().item() == target[1, [36, 37]].sum().item()


def test_distillation_moves_probability_onto_allowed_actions(tmp_path):
    torch.manual_seed(1)
    model = SeparateActorCritic(channels=8, blocks=1, value_channels=8, value_blocks=1)
    path = tmp_path / "ppo.pt"
    save_checkpoint(path, model, None, {"steps": 0, "updates": 0, "episodes": 0}, PPOConfig())
    game, other = ladder(), Game(workers=[[6, 18], [8, 16]], phase="move")
    observations = np.asarray([game.observation()] * 200 + [other.observation()] * 200, dtype=np.uint8)
    legal = np.asarray([game.mask()] * 200 + [other.mask()] * 200)
    allowed = legal.copy()
    allowed[:200] = False
    allowed[:200, [36, 37]] = True
    data = {"observations": observations, "legal": legal, "allowed": allowed}
    distilled, history = distill(path, data, epochs=12, lr=3e-3, batch=64, device="cpu", seed=0)
    assert history[-1]["allowed_mass_taught"] > history[0]["allowed_mass_taught"] + 0.2
    assert history[-1]["kl_untaught"] < 0.05, "positions the search left alone keep the original policy"
