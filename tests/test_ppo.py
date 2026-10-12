import random

import pytest
import torch

from server.game import Game
from server.rl.environment import TrainingEnvironment
from server.rl.model import ActorCritic
from server.rl.ppo import PPOConfig, clipped_policy_loss, compute_gae, update


def test_clip_uses_pessimistic_bound_for_both_advantage_signs():
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 0.5]))
    advantages = torch.tensor([1.0, -1.0])
    loss = clipped_policy_loss(new, old, advantages, 0.2)
    assert loss.item() == pytest.approx(-0.2)


def test_gae_termination_and_truncation_are_distinct():
    rewards = torch.tensor([1., 0., 9.])
    values = torch.tensor([0.2, 0.4, 0.])
    next_values = torch.tensor([99., 2., 0.])
    terminal = torch.tensor([True, False, True])
    boundaries = torch.tensor([True, True, True])
    advantages, returns = compute_gae(rewards, values, next_values, terminal, boundaries, gamma=0.9)
    assert torch.allclose(advantages, torch.tensor([0.8, 1.4, 9.]))
    assert torch.allclose(returns, torch.tensor([1., 1.8, 9.]))


def test_policy_never_samples_masked_actions():
    torch.set_num_threads(2)
    model = ActorCritic()
    game = Game()
    game.apply(12)
    mask = torch.tensor([game.mask()])
    dist, value = model(torch.tensor([game.observation()]), mask)
    samples = dist.sample((1000,))
    assert all(int(action) in game.legal_actions() for action in samples.flatten())
    assert dist.probs[0, 12] == 0
    assert torch.isfinite(value).all()
    with pytest.raises(ValueError):
        model(torch.tensor([game.observation()]), torch.zeros_like(mask))


def test_update_changes_parameters_and_is_finite():
    torch.manual_seed(10)
    torch.set_num_threads(2)
    model = ActorCritic()
    obs = torch.tensor([Game().observation()] * 32)
    masks = torch.tensor([Game().mask()] * 32)
    with torch.no_grad():
        dist, value = model(obs, masks)
        actions = dist.sample()
        old = dist.log_prob(actions)
    batch = {"observations": obs, "masks": masks, "actions": actions, "log_probs": old,
             "advantages": torch.linspace(-1, 1, 32), "returns": torch.ones(32)}
    before = model.critic.weight.detach().clone()
    metrics = update(model, torch.optim.Adam(model.parameters(), lr=3e-4), batch, PPOConfig())
    assert not torch.equal(before, model.critic.weight)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_environment_returns_learner_perspective_and_alternates_seats():
    rng = random.Random(4)
    env = TrainingEnvironment(lambda game: rng.choice(game.legal_actions()))
    for episode in range(6):
        game = env.reset()
        assert env.learner == episode % 2
        while game.winner is None:
            assert game.current == env.learner
            game, reward, terminated, truncated = env.step(rng.choice(game.legal_actions()))
            assert not truncated
            if terminated:
                assert reward == (1 if game.winner == env.learner else -1)
            else:
                assert reward == 0 and game.current == env.learner
