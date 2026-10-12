"""Turn the critic into a calibrated win estimate, or show that it is not one.

The critic predicts the learner's return against its training opponents, so it is optimistic for
whoever moves. Comparing its view from both sides cancels that bias; a logistic (Platt) fit on
self-play outcomes then maps the difference to "win chance between evenly matched players".

Check: python -m server.rl.calibration checkpoints/ppo.pt --against lookahead
Fit:   python -m server.rl.calibration checkpoints/ppo.pt --fit --games 1200
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from server.game import Game
from server.rl.arena import load_policy, make_player


def critic(model, game: Game) -> float:
    with torch.inference_mode():
        _, value = model(torch.tensor([game.observation()]), torch.tensor([game.mask()]))
    return max(-1.0, min(1.0, float(value[0])))


def swapped(game: Game) -> Game:
    """The same board with the other player to move; the critic's view from the other side."""
    other = Game.from_dict(game.to_dict())
    other.current, other.phase, other.selected = 1 - game.current, "move", -1
    return other


def advantage(model, game: Game) -> float:
    """Critic value for the mover minus the critic value if the opponent were to move instead."""
    other = swapped(game)
    return critic(model, game) - (critic(model, other) if other.legal_actions() else -1.0)


def calibrated(advantage_value: float, fit: dict) -> float:
    return 1 / (1 + math.exp(-(fit["a"] * advantage_value + fit["b"])))


def estimates(model, game: Game) -> tuple[float, float, float]:
    """(raw probability, symmetric probability, advantage) for the player to move."""
    raw = (critic(model, game) + 1) / 2
    difference = advantage(model, game)
    return raw, min(1.0, max(0.0, .5 + difference / 4)), difference


def collect(checkpoint: str, opponent: str, indices: list[int], seed: int) -> list[tuple[float, float, int]]:
    model = load_policy(checkpoint)
    rows = []
    for index in indices:
        rng = random.Random(seed * 7_919 + index)
        torch.manual_seed(seed * 7_919 + index)
        seat = index % 2
        players = {seat: make_player(checkpoint, rng, False), 1 - seat: make_player(checkpoint if opponent == "self" else opponent, rng, False)}
        game, samples = Game(), []
        while game.winner is None and game.revision < 400:
            if game.phase == "move":
                samples.append((*estimates(model, game), game.current))
            game.apply(players[game.current](game))
        rows += [(raw, symmetric, difference, int(game.winner == mover))
                 for raw, symmetric, difference, mover in samples if game.winner is not None]
    return rows


def fit_platt(rows: list[tuple[float, float, float, int]]) -> dict:
    """Two-parameter logistic regression on the advantage, by Newton's method."""
    a, b = 1.0, 0.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for *_, x, y in rows:
            p = 1 / (1 + math.exp(-(a * x + b)))
            w = p * (1 - p)
            ga += (p - y) * x
            gb += p - y
            haa += w * x * x
            hab += w * x
            hbb += w
        determinant = haa * hbb - hab * hab
        if abs(determinant) < 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / determinant, (haa * gb - hab * ga) / determinant
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-9:
            break
    return {"a": round(a, 5), "b": round(b, 5)}


def report(rows: list[tuple[float, float, float, int]], fit: dict | None = None) -> dict:
    outcomes = [won for *_, won in rows]
    base = sum(outcomes) / len(outcomes)
    result = {"positions": len(rows), "base_rate": round(base, 3), "brier_constant": round(sum((base - y) ** 2 for y in outcomes) / len(rows), 4)}
    columns = [("raw", [row[0] for row in rows]), ("symmetric", [row[1] for row in rows])]
    if fit:
        columns.append(("calibrated", [calibrated(row[2], fit) for row in rows]))
    for name, predictions in columns:
        brier = sum((p - y) ** 2 for p, y in zip(predictions, outcomes)) / len(rows)
        bins = []
        for low in [i / 10 for i in range(10)]:
            members = [(p, y) for p, y in zip(predictions, outcomes) if low <= p < low + .1 or (low == .9 and p == 1)]
            if members:
                bins.append([round(low, 1), len(members), round(sum(p for p, _ in members) / len(members), 3), round(sum(y for _, y in members) / len(members), 3)])
        result[name] = {"brier": round(brier, 4), "bins[low,count,mean_prediction,observed]": bins}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--against", default="lookahead")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--fit", action="store_true", help="fit on self-play and write <checkpoint>.calibration.json")
    args = parser.parse_args()
    opponent = "self" if args.fit else args.against
    with ProcessPoolExecutor(args.workers) as pool:
        chunks = [list(range(i, args.games, args.workers)) for i in range(args.workers)]
        parts = list(pool.map(collect, [args.checkpoint] * args.workers, [opponent] * args.workers, chunks, [args.seed] * args.workers))
    if not args.fit:
        print(json.dumps(report([row for part in parts for row in part]), indent=2))
    else:
        # Fit on the even worker chunks, report on the odd ones, so the score is out of sample.
        train = [row for i, part in enumerate(parts) for row in part if i % 2 == 0]
        held_out = [row for i, part in enumerate(parts) for row in part if i % 2 == 1]
        fit = fit_platt(train)
        result = report(held_out, fit)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        sidecar = {"checkpoint_steps": checkpoint["metadata"]["steps"], **fit, "games": args.games, "seed": args.seed,
                   "opponent": "self-play", "held_out_positions": result["positions"],
                   "held_out_brier": result["calibrated"]["brier"], "constant_brier": result["brier_constant"]}
        Path(args.checkpoint).with_suffix(".calibration.json").write_text(json.dumps(sidecar, indent=2), encoding="utf8")
        print(json.dumps({"fit": sidecar, "held_out": result}, indent=2))
