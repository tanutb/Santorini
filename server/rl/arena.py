"""Seat-balanced, parallel matches between PPO checkpoints and named baselines.

Run: python -m server.rl.arena checkpoints/candidate/ppo.pt --against random tactical lookahead checkpoints/ppo.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from server.game import Game
from server.rl.opponents import BASELINES

MAX_ACTIONS = 400
_models: dict[str, object] = {}


def load_policy(path: str):
    """Load a checkpoint once per process; evaluation uses a single CPU thread per worker."""
    import torch
    from server.rl.model import load_model
    if path not in _models:
        torch.set_num_threads(1)
        _models[path] = load_model(torch.load(path, map_location="cpu", weights_only=True))
    return _models[path]


def make_player(spec: str, rng: random.Random, greedy: bool):
    if spec in BASELINES:
        baseline = BASELINES[spec]
        return lambda game: baseline(game, rng)
    from server.rl.train import choose
    model = load_policy(spec)
    return lambda game: choose(model, game, deterministic=greedy)


def play_block(first: str, second: str, indices: list[int], seed: int, greedy: bool) -> list[tuple[int, int | None, int]]:
    """Play games where `first` takes seat index % 2; return (seat, winner, actions)."""
    import torch
    results = []
    for index in indices:
        rng = random.Random(seed * 1_000_003 + index)
        torch.manual_seed(seed * 1_000_003 + index)
        seat = index % 2
        players = {seat: make_player(first, rng, greedy), 1 - seat: make_player(second, rng, greedy)}
        game = Game()
        while game.winner is None and game.revision < MAX_ACTIONS:
            game.apply(players[game.current](game))
        results.append((seat, game.winner, game.revision))
    return results


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    centre = (p + z * z / (2 * games)) / (1 + z * z / games)
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def match(first: str, second: str, games: int = 200, seed: int = 123, workers: int = 8,
          greedy: bool = False, pool: ProcessPoolExecutor | None = None) -> dict:
    if games < 2 or games % 2:
        raise ValueError("Use an even number of games so both seats are covered equally.")
    chunks = [list(range(start, games, workers)) for start in range(min(workers, games))]
    if pool is None:
        rows = [row for chunk in chunks for row in play_block(first, second, chunk, seed, greedy)]
    else:
        futures = [pool.submit(play_block, first, second, chunk, seed, greedy) for chunk in chunks]
        rows = [row for future in futures for row in future.result()]
    seats = [{"wins": 0, "games": 0}, {"wins": 0, "games": 0}]
    wins = losses = draws = length = 0
    for seat, winner, actions in rows:
        seats[seat]["games"] += 1
        seats[seat]["wins"] += int(winner == seat)
        wins += int(winner == seat)
        losses += int(winner == 1 - seat)
        draws += int(winner is None)
        length += actions
    return {"games": games, "wins": wins, "losses": losses, "draws": draws, "win_rate": wins / games,
            "confidence_95": wilson(wins, games), "seats": seats, "mean_actions": round(length / games, 1)}


def label(spec: str) -> str:
    return spec if spec in BASELINES else Path(spec).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--against", nargs="+", default=["random", "tactical", "lookahead"])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", help="argmax instead of sampling as the server does")
    args = parser.parse_args()
    report = {"checkpoint": label(str(args.checkpoint)), "seed": args.seed,
              "policy": "greedy masked categorical" if args.greedy else "stochastic masked categorical", "results": {}}
    with ProcessPoolExecutor(args.workers) as pool:
        for opponent in args.against:
            report["results"][label(opponent)] = match(str(args.checkpoint), opponent, args.games, args.seed,
                                                       args.workers, args.greedy, pool)
            print(label(opponent), json.dumps(report["results"][label(opponent)]), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
