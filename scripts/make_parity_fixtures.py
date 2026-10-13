"""Record Python rules and PyTorch outputs so the browser port can be checked against them.

Run after changing server/game.py, the models or the export:  python scripts/make_parity_fixtures.py
Writes tests/fixtures/parity.json, read by tests/engine.test.js and tests/policy.test.js.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.game import Game  # noqa: E402
from server.rl.model import load_model  # noqa: E402
from server.rl.opponents import lookahead_action  # noqa: E402

GAMES = 40


def record_games() -> list[dict]:
    games = []
    for index in range(GAMES):
        rng = random.Random(1000 + index)
        game, steps = Game(), []
        clever = index % 2 == 1
        while game.winner is None and game.revision < 300:
            legal = game.legal_actions()
            action = lookahead_action(game, rng) if clever and rng.random() < .8 else rng.choice(legal)
            observation = "".join("1" if value else "0" for value in game.observation())
            message = game.apply(action)
            steps.append({"legal": legal, "observation": observation, "action": action, "message": message, "after": game.to_dict()})
        games.append({"steps": steps})
    return games


def record_policies(games: list[dict]) -> dict:
    manifest = json.loads((ROOT / "web/models/manifest.json").read_text(encoding="utf8"))
    rng = random.Random(7)
    positions = []
    for game in games:
        state = Game()
        for step in game["steps"]:
            if rng.random() < .06:
                positions.append(state.to_dict())
            state.apply(step["action"])
    torch.set_num_threads(1)
    outputs = {}
    for level in manifest["levels"]:
        path = ROOT / "checkpoints" / ("ppo.pt" if level["id"] == "full" else f"levels/{level['id']}.pt")
        model = load_model(torch.load(path, map_location="cpu", weights_only=True))
        rows = []
        with torch.inference_mode():
            for position in positions:
                game = Game.from_dict(position)
                distribution, value = model(torch.tensor([game.observation()]), torch.tensor([game.mask()]))
                rows.append({"probabilities": [round(p, 7) for p in distribution.probs[0].tolist()], "value": round(float(value[0]), 7)})
        outputs[level["id"]] = rows
    return {"positions": positions, "outputs": outputs}


if __name__ == "__main__":
    games = record_games()
    fixture = {"games": games, "policies": record_policies(games)}
    target = ROOT / "tests/fixtures/parity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf8")
    steps = sum(len(game["steps"]) for game in games)
    print(f"{target.relative_to(ROOT)}: {steps} steps, {len(fixture['policies']['positions'])} policy positions, {target.stat().st_size // 1024} KiB")
