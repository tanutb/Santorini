"""Evaluate a checkpoint on both seats against named baselines and record the report beside it.

Run: python -m server.rl.evaluate --checkpoint checkpoints/ppo.pt --games 400
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from server.rl.arena import match

BASELINE_ORDER = ("random", "tactical", "lookahead", "climber")


def evaluate(checkpoint_path: Path, games: int = 200, seed: int = 123, workers: int = 8) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with ProcessPoolExecutor(workers) as pool:
        results = {name: match(str(checkpoint_path), name, games, seed, workers, False, pool) for name in BASELINE_ORDER}
    report = {"checkpoint_steps": checkpoint["metadata"]["steps"], "seed": seed,
              "policy": "stochastic masked categorical", "results": results}
    checkpoint_path.with_suffix(".evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.pt"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.games, args.seed, args.workers), indent=2))
