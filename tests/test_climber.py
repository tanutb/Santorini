"""Regression coverage for the corner tower-racing training opponent."""
import random

import pytest

from server.game import Game
from server.rl.opponents import climber_action
from server.rl.train import parse_mix


@pytest.mark.parametrize("corner,other", [(0, 24), (4, 20), (20, 4), (24, 0)])
def test_climber_wins_quickly_when_ignored(corner, other):
    game = Game(workers=[[corner, other], [7, 12]], phase="move")
    rng = random.Random(17)
    for _ in range(15):
        action = climber_action(game, rng)
        assert action in game.legal_actions()
        game.apply(action)
        if game.winner is not None:
            break
        # The distant opponent leaves these towers alone.
        game.current = 0
    assert game.winner == 0
    assert game.reason == "Reached the third level"


def test_climber_uses_other_worker_when_first_is_trapped():
    game = Game(workers=[[0, 24], [7, 12]], phase="move")
    for cell in (1, 5, 6):
        game.heights[cell] = 4
    assert climber_action(game, random.Random(1)) in game.legal_actions()
    assert climber_action(game, random.Random(1)) >= 50


def test_climber_handles_occupied_corners_and_mix():
    game = Game(workers=[[0, 4], [-1, -1]], current=1)
    rng = random.Random(3)
    game.apply(climber_action(game, rng))
    game.apply(climber_action(game, rng))
    assert set(game.workers[1]) == {20, 24}
    assert parse_mix("climber=0.2,lookahead=0.3")["climber"] == .2


def test_entropy_bonus_encourages_exploration_without_reward_signal():
    import torch
    from server.rl.model import SeparateActorCritic
    from server.rl.ppo import PPOConfig, update

    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = SeparateActorCritic(channels=4, blocks=1, value_channels=4, value_blocks=1)
    with torch.no_grad():
        model.pi.policy.weight.mul_(100)
    game = Game()
    observations = torch.tensor([game.observation()] * 8)
    masks = torch.tensor([game.mask()] * 8)
    with torch.no_grad():
        dist = model.pi(observations, masks)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
        entropy_before = dist.entropy().mean().item()
    batch = {"observations": observations, "masks": masks, "actions": actions,
             "log_probs": old_log_probs, "advantages": torch.zeros(8), "returns": torch.zeros(8)}
    optimizers = (torch.optim.Adam(model.pi.parameters(), lr=.001),
                  torch.optim.Adam(model.v.parameters(), lr=.001))
    config = PPOConfig(entropy_coef=.1, minibatch=0, train_pi_iters=10, train_v_iters=0)
    update(model, optimizers, batch, config)
    assert model.pi(observations, masks).entropy().mean().item() > entropy_before
