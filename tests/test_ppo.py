import random

import pytest
import torch

from server.game import Game
from server.rl.environment import TrainingEnvironment
from server.rl.model import ActorCritic, SeparateActorCritic
from server.rl.ppo import PPOConfig, clipped_policy_loss, compute_gae, rewards_to_go, update


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


def test_rewards_to_go_sum_to_the_end_and_bootstrap_only_cut_data():
    gamma = 0.5
    # One game: three decisions ending in a win; then a game cut by the time limit; then the rollout ends mid-game.
    rewards = torch.tensor([0., 0., 1., 0., 0.])
    next_values = torch.tensor([7., 7., 7., 0.4, 0.8])
    terminated = torch.tensor([False, False, True, False, False])
    boundaries = torch.tensor([False, False, True, True, False])
    returns = rewards_to_go(rewards, next_values, terminated, boundaries, gamma)
    assert torch.allclose(returns, torch.tensor([0.25, 0.5, 1., 0.2, 0.4]))


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


def small_batch(model, size=32):
    obs = torch.tensor([Game().observation()] * size)
    masks = torch.tensor([Game().mask()] * size)
    with torch.no_grad():
        dist, _ = model(obs, masks)
        actions = dist.sample()
        old = dist.log_prob(actions)
    return {"observations": obs, "masks": masks, "actions": actions, "log_probs": old,
            "advantages": torch.linspace(-1, 1, size), "returns": torch.ones(size)}


def test_update_trains_policy_and_value_networks_separately():
    torch.manual_seed(10)
    torch.set_num_threads(2)
    model = SeparateActorCritic(channels=8, blocks=1, value_channels=8, value_blocks=1)
    batch = small_batch(model)
    optimizers = (torch.optim.Adam(model.pi.parameters(), lr=3e-4), torch.optim.Adam(model.v.parameters(), lr=1e-3))
    policy_before = model.pi.policy.weight.detach().clone()
    value_before = model.v.value[-1].weight.detach().clone()
    metrics = update(model, optimizers, batch, PPOConfig(train_pi_iters=5, train_v_iters=5, minibatch=0))
    assert not torch.equal(policy_before, model.pi.policy.weight)
    assert not torch.equal(value_before, model.v.value[-1].weight)
    assert metrics["policy_steps"] == 5 and metrics["value_steps"] == 5 and metrics["value_loss_after"] < metrics["value_loss"]
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_minibatch_schedule_makes_epoch_passes_like_the_first_agents():
    torch.manual_seed(12)
    model = SeparateActorCritic(channels=8, blocks=1, value_channels=8, value_blocks=1)
    batch = small_batch(model, size=40)
    optimizers = (torch.optim.Adam(model.pi.parameters(), lr=1e-4), torch.optim.Adam(model.v.parameters(), lr=1e-3))
    metrics = update(model, optimizers, batch, PPOConfig(epochs=3, minibatch=16))
    # 40 decisions in minibatches of 16 is 3 steps per pass; 3 passes make 9 steps each.
    assert metrics["step_limit"] == 9 and metrics["value_steps"] == 9
    assert metrics["policy_steps"] <= 9 and metrics["value_loss_after"] < metrics["value_loss"]


def test_policy_steps_stop_once_kl_passes_one_and_a_half_times_the_target():
    torch.manual_seed(11)
    model = SeparateActorCritic(channels=8, blocks=1, value_channels=8, value_blocks=1)
    batch = small_batch(model)
    optimizers = (torch.optim.Adam(model.pi.parameters(), lr=0.5), torch.optim.Adam(model.v.parameters(), lr=1e-3))
    metrics = update(model, optimizers, batch, PPOConfig(train_pi_iters=80, train_v_iters=1, target_kl=1e-4, minibatch=0))
    assert metrics["early_stopped"] == 1 and 1 <= metrics["policy_steps"] < 80
    assert metrics["kl"] > 1.5e-4


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
