"""Distil the browser agent (network plus forced-win search) into the network itself.

The website's agent samples from its network after the search (web/tactics.js) removes the turns it
proves worse. This module plays games, labels every decision of the agent with the actions the search
allows, and trains a copy of the network toward that agent's own distribution: the original network's
probabilities, restricted to the allowed actions and renormalised. Where the search allows every legal
action the target is the original network unchanged, so the copy keeps its play there and changes only
where the search proves it wrong. The value network is kept as it is.

Run: python -m server.rl.distill --games 10000 --output checkpoints/distill/ppo.pt
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from server.game import Game
from server.rl.arena import load_policy
from server.rl.model import load_model
from server.rl.opponents import BASELINES
from server.rl.teacher import SearchTeacher
from server.rl.train import save_checkpoint
from server.rl.ppo import PPOConfig

OPPONENTS = ("climber", "lookahead", "tactical", "random", "self")
MAX_ACTIONS = 400


def play(checkpoint: str, indices: list[int], seed: int, depth: int, budget: int) -> dict[str, np.ndarray]:
    """Games between the agent and a rotation of opponents. Half the games the agent samples like the
    website (search applied), half from its network alone, so the data also holds the network's own mistakes."""
    torch.set_num_threads(1)
    model = load_policy(checkpoint)
    teacher = SearchTeacher(depth, budget)
    observations, legal, allowed = [], [], []
    try:
        for index in indices:
            rng = random.Random(seed * 1_000_003 + index)
            torch.manual_seed(seed * 1_000_003 + index)
            seat, opponent, searched = index % 2, OPPONENTS[(index // 2) % len(OPPONENTS)], (index // 10) % 2 == 0
            game = Game()
            while game.winner is None and game.revision < MAX_ACTIONS:
                mask = torch.tensor([game.mask()])
                if game.current != seat and opponent != "self":
                    game.apply(BASELINES[opponent](game, rng))
                    continue
                with torch.no_grad():
                    probabilities = model.pi(torch.tensor([game.observation()]), mask).probs[0]
                if game.current == seat and game.phase != "place":
                    allow, _ = teacher.masks([game])
                    observations.append(np.asarray(game.observation(), dtype=np.uint8))
                    legal.append(mask[0].numpy())
                    allowed.append(allow[0])
                    if searched:
                        probabilities = probabilities * torch.from_numpy(allow[0])
                game.apply(int(torch.multinomial(probabilities, 1)))
    finally:
        teacher.close()
    return {"observations": np.asarray(observations), "legal": np.asarray(legal), "allowed": np.asarray(allowed)}


def collect(checkpoint: str, games: int, seed: int, depth: int, budget: int, workers: int) -> dict[str, np.ndarray]:
    chunks = [list(range(start, games, workers)) for start in range(min(workers, games))]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(play, [checkpoint] * len(chunks), chunks, [seed] * len(chunks), [depth] * len(chunks), [budget] * len(chunks)))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def targets(reference: torch.nn.Module, observations: torch.Tensor, legal: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """The website agent's distribution: the reference network's probabilities on the allowed actions, renormalised."""
    with torch.no_grad():
        logits = reference.pi(observations, legal).logits
        return torch.log_softmax(logits.masked_fill(~allowed, -1e9), dim=-1)


def divergence(model: torch.nn.Module, observations: torch.Tensor, legal: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || model) per decision."""
    log_probs = model.pi(observations, legal).logits
    return (target.exp() * (target - log_probs)).sum(-1)


def distill(checkpoint_path: Path, data: dict[str, np.ndarray], epochs: int, lr: float, batch: int, device: str,
            seed: int) -> tuple[torch.nn.Module, list[dict]]:
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, reference = load_model(checkpoint).to(device), load_model(checkpoint).to(device).eval()
    reference.requires_grad_(False)
    observations = torch.from_numpy(data["observations"]).float().to(device)
    legal = torch.from_numpy(data["legal"]).to(device)
    allowed = torch.from_numpy(data["allowed"]).to(device)
    taught = (legal != allowed).any(-1)
    target = torch.cat([targets(reference, observations[i:i + 4096], legal[i:i + 4096], allowed[i:i + 4096])
                        for i in range(0, len(observations), 4096)])
    order = torch.randperm(len(observations), device=device)
    held = order[: len(order) // 20]
    train_ids = order[len(order) // 20:]
    optimizer = torch.optim.Adam(model.pi.parameters(), lr=lr)
    history = []

    def report(epoch: int) -> dict:
        model.eval()
        with torch.no_grad():
            kl = divergence(model, observations[held], legal[held], target[held])
            mass = (model.pi(observations[held], legal[held]).probs * allowed[held]).sum(-1)
            row = {"epoch": epoch, "held_out": len(held), "kl_taught": kl[taught[held]].mean().item(),
                   "kl_untaught": kl[~taught[held]].mean().item(), "allowed_mass_taught": mass[taught[held]].mean().item()}
        model.train()
        print(json.dumps(row), flush=True)
        return row

    history.append(report(0))
    for epoch in range(1, epochs + 1):
        for ids in train_ids[torch.randperm(len(train_ids), device=device)].split(batch):
            loss = divergence(model, observations[ids], legal[ids], target[ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss; refusing to save invalid weights.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(report(epoch))
    return model.cpu().eval(), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/distill/ppo.pt"))
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--depth", type=int, default=4, help="opponent turns the search looks ahead (the website's Hard uses 4)")
    parser.add_argument("--budget", type=int, default=1_000_000, help="the search's node budget per decision")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, help="reuse labelled decisions saved by an earlier run (an .npz file)")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.data or args.output.with_name("decisions.npz")
    if args.data:
        data = dict(np.load(args.data))
    else:
        data = collect(str(args.checkpoint), args.games, args.seed, args.depth, args.budget, args.workers)
        np.savez_compressed(data_path, **data)
    taught = int((data["legal"] != data["allowed"]).any(-1).sum())
    print(f"{len(data['legal'])} decisions, {taught} where the search excluded actions", flush=True)
    model, history = distill(args.checkpoint, data, args.epochs, args.lr, args.batch, args.device, args.seed)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = {**source["metadata"], "status": "distilled from the network with the browser's search; evaluate before promotion",
                "distillation": {"source_steps": source["metadata"]["steps"], "games": args.games, "decisions": len(data["legal"]),
                                 "taught": taught, "depth": args.depth, "budget": args.budget, "epochs": args.epochs,
                                 "lr": args.lr, "history": history}}
    save_checkpoint(args.output, model, None, metadata, PPOConfig(**source["config"]))


if __name__ == "__main__":
    main()
