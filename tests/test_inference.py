import torch

from server.game import Game
from server.inference import AgentService
from server.rl.model import ActorCritic
from server.rl.ppo import PPOConfig
from server.rl.train import save_checkpoint


def test_missing_and_corrupt_checkpoints_report_unavailable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ppo.pt"
    monkeypatch.setenv("SANTORINI_MODEL", str(checkpoint))
    service = AgentService()
    assert not service.status()["available"]
    checkpoint.write_bytes(b"not a torch checkpoint")
    assert not service.status()["available"]


def test_inference_loads_saved_weights_and_only_selects_legal_actions(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    checkpoint = tmp_path / "ppo.pt"
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(checkpoint, model, optimizer, {"steps": 10, "updates": 1}, PPOConfig())
    monkeypatch.setenv("SANTORINI_MODEL", str(checkpoint))
    service = AgentService()
    assert service.status()["available"]
    assert service.status()["metadata"]["steps"] == 10
    game = Game()
    game.apply(12)
    for _ in range(20):
        assert service.action(game) in game.legal_actions()
