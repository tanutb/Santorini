import argparse
import json
import math
import random

import pytest
import torch

from server.game import Game
from server.inference import AgentService
from server.rl.arena import match, wilson
from server.rl.calibration import fit_platt
from server.rl.model import ActorCritic, ConvActorCritic, create_model, load_model
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


def test_checkpoint_format_two_round_trips_and_format_one_still_loads(tmp_path):
    conv = create_model({"architecture": "conv", "channels": 8, "blocks": 1})
    save_checkpoint(tmp_path / "conv.pt", conv, None, {"steps": 1, "updates": 1}, PPOConfig())
    loaded = load_model(torch.load(tmp_path / "conv.pt", weights_only=True))
    assert isinstance(loaded, ConvActorCritic) and loaded.spec == conv.spec
    mlp = ActorCritic()
    torch.save({"format_version": 1, "hidden": 128, "model": mlp.state_dict()}, tmp_path / "old.pt")
    assert isinstance(load_model(torch.load(tmp_path / "old.pt", weights_only=True)), ActorCritic)
    with pytest.raises(ValueError):
        load_model({"format_version": 9})


def test_in_process_training_writes_a_loadable_checkpoint(tmp_path):
    args = argparse.Namespace(updates=2, steps=8, envs=2, workers=0, seed=3, threads=1, device="cpu",
                              output=tmp_path / "ppo.pt", resume=None, architecture="conv", hidden=128, channels=8,
                              blocks=1, learning_rate=None, final_learning_rate=None, entropy_coefficient=None,
                              epochs=1, minibatch=16, mix="random=0.5,lookahead=0.2", fixed_opponent=[],
                              snapshot_every=1, pool_size=2, archive_every=0)
    train(args)
    checkpoint = torch.load(tmp_path / "ppo.pt", weights_only=True)
    assert checkpoint["metadata"]["updates"] == 2 and checkpoint["metadata"]["steps"] == 32
    assert len((tmp_path / "training.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError):
        parse_mix("random=0.8,lookahead=0.5")


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


def test_insight_requires_a_matching_calibration(tmp_path, monkeypatch):
    path = tmp_path / "ppo.pt"
    save_checkpoint(path, create_model({"architecture": "conv", "channels": 8, "blocks": 1}), None,
                    {"steps": 42, "updates": 1}, PPOConfig())
    monkeypatch.setenv("SANTORINI_MODEL", str(path))
    service = AgentService()
    game = placed()
    assert service.insight(game) is None
    fit = {"checkpoint_steps": 41, "a": 1, "b": 0, "held_out_brier": .2, "constant_brier": .25}
    path.with_suffix(".calibration.json").write_text(json.dumps(fit))
    assert service.insight(game) is None, "a fit for other weights is ignored"
    path.with_suffix(".calibration.json").write_text(json.dumps({**fit, "checkpoint_steps": 42, "held_out_brier": .245}))
    assert service.insight(game) is None, "a fit with almost no skill is not shown"
    path.with_suffix(".calibration.json").write_text(json.dumps({**fit, "checkpoint_steps": 42}))
    insight = service.insight(game)
    assert insight["revision"] == game.revision and 0 < insight["p0"] < 1
    suggestion = service.suggest(game)
    assert suggestion["action"] in game.legal_actions() and 0 < suggestion["confidence"] <= 1


def test_levels_manifest_offers_earlier_checkpoints(tmp_path):
    from server.inference import AgentLevels
    small = {"architecture": "conv", "channels": 8, "blocks": 1}
    save_checkpoint(tmp_path / "ppo.pt", create_model(small), None, {"steps": 100, "updates": 2}, PPOConfig())
    save_checkpoint(tmp_path / "levels" / "early.pt", create_model(small), None, {"steps": 10, "updates": 1}, PPOConfig())
    (tmp_path / "levels.json").write_text(json.dumps({"default": "early", "levels": [
        {"id": "early", "name": "Early", "checkpoint": "levels/early.pt"},
        {"id": "../escape", "name": "Bad", "checkpoint": "x.pt"},
        {"id": "full", "name": "Full strength"}]}))
    levels = AgentLevels(AgentService(tmp_path / "ppo.pt"))
    assert levels.default == "early" and set(levels.services) == {"early", "full"}
    described = levels.describe()
    assert [row["id"] for row in described] == ["early", "full"] and described[0]["steps"] == 10
    assert levels.get(None) is levels.services["early"] and levels.get("missing") is levels.main
    assert levels.get("full").action(placed()) in placed().legal_actions()


def test_without_a_manifest_only_full_strength_exists(tmp_path):
    from server.inference import AgentLevels
    levels = AgentLevels(AgentService(tmp_path / "ppo.pt"))
    assert levels.default == "full" and [entry["id"] for entry in levels.entries] == ["full"]
