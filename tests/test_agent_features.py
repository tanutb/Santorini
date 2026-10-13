import argparse
import math
import random

import pytest
import torch

from server.game import Game
from server.rl.arena import match, wilson
from server.rl.calibration import fit_platt
from server.rl.model import ActorCritic, ConvActorCritic, SeparateActorCritic, create_model, load_model
from server.rl.opponents import lookahead_action
from server.rl.ppo import PPOConfig
from server.rl.train import parse_mix, save_checkpoint, train


def placed() -> Game:
    game = Game()
    for cell in (0, 1, 23, 24):
        game.apply(cell)
    return game


def test_lookahead_takes_a_win_and_blocks_a_threat():
    rng = random.Random(1)
    game = placed()
    game.heights[0], game.heights[5] = 2, 3
    assert lookahead_action(game, rng) == 25 + 5
    # The opponent's level-two worker on 24 can climb to 18: our build must dome it.
    game = placed()
    game.heights[24], game.heights[18] = 2, 3
    game.workers[0] = [12, 1]
    game.apply(25 + 13)
    assert lookahead_action(game, rng) == 18


def test_conv_policy_masks_illegal_actions_and_matches_layout():
    torch.manual_seed(0)
    model = ConvActorCritic(channels=16, blocks=1)
    game = placed()
    dist, value = model(torch.tensor([game.observation()]), torch.tensor([game.mask()]))
    assert dist.probs.shape == (1, 75) and value.shape == (1,)
    legal = set(game.legal_actions())
    assert all(int(a) in legal for a in dist.sample((500,)).flatten())
    with pytest.raises(ValueError):
        model(torch.tensor([game.observation()]), torch.zeros(1, 75, dtype=torch.bool))


def test_checkpoint_format_three_round_trips_and_older_formats_still_load(tmp_path):
    spec = {"architecture": "separate", "channels": 8, "blocks": 1, "value_channels": 4, "value_blocks": 1}
    model = create_model(spec)
    save_checkpoint(tmp_path / "separate.pt", model, None, {"steps": 1, "updates": 1}, PPOConfig())
    loaded = load_model(torch.load(tmp_path / "separate.pt", weights_only=True))
    assert isinstance(loaded, SeparateActorCritic) and loaded.spec == spec
    game = placed()
    dist, value = loaded(torch.tensor([game.observation()]), torch.tensor([game.mask()]))
    assert dist.probs.shape == (1, 75) and value.shape == (1,)
    conv = create_model({"architecture": "conv", "channels": 8, "blocks": 1})
    torch.save({"format_version": 2, "model_spec": conv.spec, "model": conv.state_dict()}, tmp_path / "conv.pt")
    assert isinstance(load_model(torch.load(tmp_path / "conv.pt", weights_only=True)), ConvActorCritic)
    mlp = ActorCritic()
    torch.save({"format_version": 1, "hidden": 128, "model": mlp.state_dict()}, tmp_path / "old.pt")
    assert isinstance(load_model(torch.load(tmp_path / "old.pt", weights_only=True)), ActorCritic)
    with pytest.raises(ValueError):
        load_model({"format_version": 9})


def test_in_process_training_writes_a_loadable_checkpoint(tmp_path):
    args = argparse.Namespace(updates=2, steps=8, envs=2, workers=0, seed=3, threads=1, device="cpu",
                              output=tmp_path / "ppo.pt", resume=None, channels=8, blocks=1, value_channels=4, value_blocks=1,
                              gamma=None, gae_lambda=None, clip=None, pi_lr=None, vf_lr=None, train_pi_iters=3,
                              train_v_iters=3, target_kl=None, mix="random=0.5,lookahead=0.2", fixed_opponent=[],
                              snapshot_every=1, pool_size=2, archive_every=0)
    train(args)
    checkpoint = torch.load(tmp_path / "ppo.pt", weights_only=True)
    assert checkpoint["format_version"] == 3 and set(checkpoint["optimizer"]) == {"pi", "v"}
    assert checkpoint["metadata"]["updates"] == 2 and checkpoint["metadata"]["steps"] == 32
    assert checkpoint["config"]["train_pi_iters"] == 3 and checkpoint["config"]["gae_lambda"] == 0.97
    args.resume, args.updates = tmp_path / "ppo.pt", 1
    args.vf_lr = 2e-4
    train(args)
    resumed = torch.load(tmp_path / "ppo.pt", weights_only=True)
    assert resumed["metadata"]["updates"] == 3
    assert resumed["optimizer"]["v"]["param_groups"][0]["lr"] == 2e-4
    assert len((tmp_path / "training.jsonl").read_text().splitlines()) == 3
    with pytest.raises(ValueError):
        parse_mix("random=0.8,lookahead=0.5")


def test_training_stops_once_enough_games_are_played(tmp_path):
    args = argparse.Namespace(updates=40, steps=8, envs=2, workers=0, seed=5, threads=1, device="cpu",
                              output=tmp_path / "ppo.pt", resume=None, channels=8, blocks=1, value_channels=4, value_blocks=1,
                              gamma=None, gae_lambda=None, clip=None, pi_lr=None, vf_lr=None, train_pi_iters=1,
                              train_v_iters=1, target_kl=None, mix="random=1", fixed_opponent=[],
                              snapshot_every=5, pool_size=2, archive_every=0, total_games=1)
    train(args)
    metadata = torch.load(tmp_path / "ppo.pt", weights_only=True)["metadata"]
    assert metadata["episodes"] >= 1 and metadata["updates"] < 40


def test_refitting_the_value_network_leaves_the_policy_untouched(tmp_path):
    args = argparse.Namespace(updates=1, steps=8, envs=2, workers=0, seed=6, threads=1, device="cpu",
                              output=tmp_path / "ppo.pt", resume=None, channels=8, blocks=1, value_channels=4, value_blocks=1,
                              gamma=None, gae_lambda=None, clip=None, pi_lr=None, vf_lr=None, train_pi_iters=2,
                              train_v_iters=2, target_kl=None, mix="random=1", fixed_opponent=[],
                              snapshot_every=5, pool_size=2, archive_every=0)
    train(args)
    before = torch.load(tmp_path / "ppo.pt", weights_only=True)["model"]
    args.resume, args.output, args.refit_value = tmp_path / "ppo.pt", tmp_path / "refit.pt", True
    train(args)
    refit = torch.load(tmp_path / "refit.pt", weights_only=True)
    after = refit["model"]
    assert refit["metadata"]["updates"] == 1 and refit["config"]["train_policy"] is False
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("pi."))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("v."))


def test_arena_balances_seats_and_reports_intervals():
    result = match("lookahead", "random", games=4, workers=2)
    assert [seat["games"] for seat in result["seats"]] == [2, 2]
    assert result["wins"] + result["losses"] + result["draws"] == 4
    low, high = wilson(50, 100)
    assert low < .5 < high


def test_platt_fit_recovers_a_known_logistic():
    rng = random.Random(5)
    rows = []
    for _ in range(4000):
        x = rng.uniform(-2, 2)
        rows.append((0, 0, x, int(rng.random() < 1 / (1 + math.exp(-(1.5 * x + .3))))))
    fit = fit_platt(rows)
    assert fit["a"] == pytest.approx(1.5, abs=.2) and fit["b"] == pytest.approx(.3, abs=.15)
