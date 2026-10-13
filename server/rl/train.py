"""Parallel PPO-Clip training (Spinning Up's process) against baselines, frozen snapshots and fixed checkpoints.

Run: python -m server.rl.train --workers 8 --envs 16 --updates 1200 --device cuda --output checkpoints/spinup/ppo.pt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from server.game import ACTION_COUNT, OBSERVATION_SIZE, Game
from server.rl.environment import TrainingEnvironment
from server.rl.model import SeparateActorCritic, create_model, load_model
from server.rl.opponents import BASELINES
from server.rl.ppo import PPOConfig, compute_gae, rewards_to_go, update

FORMAT_VERSION = 3


@torch.no_grad()
def choose(model: nn.Module, game: Game, device: str = "cpu", deterministic: bool = False) -> int:
    observation = torch.tensor(game.observation(), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(game.mask(), dtype=torch.bool, device=device).unsqueeze(0)
    dist, _ = model(observation, mask)
    return int(dist.probs.argmax(-1).item() if deterministic else dist.sample().item())


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}


def wire_state(model: nn.Module) -> dict[str, np.ndarray]:
    """Weights as numpy arrays for pipes: torch tensors would travel as shared-memory file
    mappings, which leak on Windows until the page file is exhausted (error 1455)."""
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def from_wire(state: dict) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in state.items()}


def save_checkpoint(path: Path, model: nn.Module, optimizers: tuple | None,
                    metadata: dict, config: PPOConfig) -> None:
    """Write atomically so a reader never loads a half-written file. Format 3 keeps the policy
    and value optimizers separately, as the two networks are trained separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": FORMAT_VERSION, "model_spec": model.spec, "model": cpu_state(model),
                "optimizer": {"pi": optimizers[0].state_dict(), "v": optimizers[1].state_dict()} if optimizers else None,
                "metadata": metadata, "config": asdict(config)}, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf8")


def parse_mix(text: str) -> dict[str, float]:
    """`random=0.1,lookahead=0.3`; the remaining probability plays frozen policies."""
    mix = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline opponent: {name}")
        mix[name] = float(value)
    if any(p < 0 for p in mix.values()) or sum(mix.values()) > 1:
        raise ValueError("Opponent probabilities must be non-negative and sum to at most 1.")
    return mix


class RolloutCollector:
    """Steps several learner-perspective environments in lockstep with one batched forward pass."""

    def __init__(self, seed: int, spec: dict, environments: int, mix: dict[str, float],
                 fixed_paths: list[str], max_actions: int = 400) -> None:
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.spec, self.mix = spec, mix
        self.model = create_model(spec).eval()
        self.snapshots: dict[int, nn.Module] = {}
        self.fixed = [load_model(torch.load(path, map_location="cpu", weights_only=True)) for path in fixed_paths]
        self.environments = []
        for index in range(environments):
            env = TrainingEnvironment(lambda game: 0, seed=seed + index, max_actions=max_actions)
            env.episodes = index
            env.kind = ""
            self.environments.append(env)
        self.started = False

    def _pick(self, env: TrainingEnvironment) -> None:
        draw, cumulative = self.rng.random(), 0.0
        for kind, probability in self.mix.items():
            cumulative += probability
            if draw < cumulative:
                baseline = BASELINES[kind]
                env.kind, env.opponent = kind, lambda game, b=baseline: b(game, self.rng)
                return
        frozen = list(self.snapshots.values())
        if self.fixed and (not frozen or self.rng.random() < 0.25):
            env.kind, model = "fixed", self.rng.choice(self.fixed)
        else:
            env.kind, model = "snapshot", self.rng.choice(frozen or [self.model])
        env.opponent = lambda game, m=model: choose(m, game)

    def sync(self, weights: dict, snapshot_updates: dict[int, dict], snapshot_ids: list[int]) -> None:
        self.model.load_state_dict(from_wire(weights))
        for key, state in snapshot_updates.items():
            frozen = create_model(self.spec).eval()
            frozen.load_state_dict(from_wire(state))
            self.snapshots[key] = frozen
        for key in list(self.snapshots):
            if key not in snapshot_ids:
                del self.snapshots[key]
        if not self.started:
            for env in self.environments:
                self._pick(env)
                env.reset()
            self.started = True

    @torch.no_grad()
    def _evaluate(self, games: list[Game]) -> tuple[np.ndarray, np.ndarray, torch.distributions.Categorical, torch.Tensor]:
        observations = np.asarray([game.observation() for game in games], dtype=np.float32)
        masks = np.asarray([game.mask() for game in games], dtype=bool)
        dist, values = self.model(torch.from_numpy(observations), torch.from_numpy(masks))
        return observations, masks, dist, values

    @torch.no_grad()
    def collect(self, steps: int) -> dict:
        streams = len(self.environments)
        out = {"observations": np.zeros((steps, streams, OBSERVATION_SIZE), np.uint8),
               "masks": np.zeros((steps, streams, ACTION_COUNT), bool),
               "actions": np.zeros((steps, streams), np.int64), "log_probs": np.zeros((steps, streams), np.float32),
               "values": np.zeros((steps, streams), np.float32), "rewards": np.zeros((steps, streams), np.float32),
               "next_values": np.zeros((steps, streams), np.float32),
               "terminated": np.zeros((steps, streams), bool), "boundaries": np.zeros((steps, streams), bool)}
        stats: dict[str, list[int]] = {}
        for t in range(steps):
            observations, masks, dist, values = self._evaluate([env.game for env in self.environments])
            actions = dist.sample()
            out["observations"][t], out["masks"][t] = observations, masks
            out["actions"][t], out["log_probs"][t] = actions.numpy(), dist.log_prob(actions).numpy()
            out["values"][t] = values.numpy()
            for s, env in enumerate(self.environments):
                state, reward, terminated, truncated = env.step(int(actions[s]))
                out["rewards"][t, s], out["terminated"][t, s] = reward, terminated
                out["boundaries"][t, s] = terminated or truncated
                if truncated:
                    out["next_values"][t, s] = float(self._evaluate([state])[3][0])
                if terminated or truncated:
                    record = stats.setdefault(env.kind, [0, 0])
                    record[0] += 1
                    record[1] += int(reward > 0)
                    self._pick(env)
                    env.reset()
        # A step's successor is the next recorded decision unless an episode boundary intervened.
        final = self._evaluate([env.game for env in self.environments])[3].numpy()
        successors = np.concatenate((out["values"][1:], final[None]), axis=0)
        out["next_values"] = np.where(out["boundaries"], out["next_values"], successors).astype(np.float32)
        out["stats"] = stats
        return out


def _worker(connection, seed: int, spec: dict, environments: int, mix: dict, fixed: list[str]) -> None:
    torch.set_num_threads(1)
    collector = RolloutCollector(seed, spec, environments, mix, fixed)
    while True:
        message = connection.recv()
        if message is None:
            break
        weights, updates, ids, steps = message
        collector.sync(weights, updates, ids)
        connection.send(collector.collect(steps))
    connection.close()


def assemble(results: list[dict], config: PPOConfig, device: str) -> tuple[dict, dict]:
    """Join worker streams along the stream axis, then compute per stream the GAE advantages
    (pseudocode line 5) and the rewards-to-go that the value network is fitted to (line 4)."""
    joined = {key: np.concatenate([r[key] for r in results], axis=1) for key in results[0] if key != "stats"}
    tensors = {key: torch.from_numpy(joined[key]) for key in ("rewards", "values", "next_values", "terminated", "boundaries")}
    advantages, _ = compute_gae(tensors["rewards"], tensors["values"], tensors["next_values"], tensors["terminated"],
                                tensors["boundaries"], config.gamma, config.gae_lambda)
    returns = rewards_to_go(tensors["rewards"], tensors["next_values"], tensors["terminated"], tensors["boundaries"], config.gamma)
    flat = lambda array: torch.as_tensor(array).flatten(0, 1).to(device)
    batch = {"observations": flat(joined["observations"]).float(), "masks": flat(joined["masks"]),
             "actions": flat(joined["actions"]), "log_probs": flat(joined["log_probs"]),
             "advantages": flat(advantages), "returns": flat(returns)}
    stats: dict[str, list[int]] = {}
    for result in results:
        for kind, (games, wins) in result["stats"].items():
            record = stats.setdefault(kind, [0, 0])
            record[0] += games
            record[1] += wins
    return batch, stats


def train(args: argparse.Namespace) -> None:
    if args.updates < 1 or args.steps < 2 or args.threads < 1 or args.workers < 0 or args.envs < 1:
        raise ValueError("Use at least one update, two steps per environment, one thread and one environment.")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mix = parse_mix(args.mix)
    config = PPOConfig()
    start, total_steps, total_episodes = 0, 0, 0
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        model = load_model(checkpoint)
        if not isinstance(model, SeparateActorCritic):
            raise ValueError("Only separate policy and value networks (checkpoint format 3) can resume training.")
        config = PPOConfig(**checkpoint["config"])
        start = checkpoint["metadata"]["updates"]
        total_steps = checkpoint["metadata"]["steps"]
        total_episodes = checkpoint["metadata"]["episodes"]
    else:
        model = create_model({"architecture": "separate", "channels": args.channels, "blocks": args.blocks,
                              "value_channels": args.value_channels, "value_blocks": args.value_blocks})
    refit = bool(getattr(args, "refit_value", False))
    if refit:
        # Keep the policy exactly as it is and start a fresh value network (line 7 only).
        # The counters restart so the log describes the refit, not the policy's training.
        model.v = create_model(model.spec).v
        start, total_steps, total_episodes = 0, 0, 0
    config.train_policy = not refit
    model = model.to(args.device).train()
    for name in ("gamma", "gae_lambda", "clip", "pi_lr", "vf_lr", "train_pi_iters", "train_v_iters", "target_kl", "epochs", "minibatch"):
        override = getattr(args, name, None)
        if override is not None:
            setattr(config, name, override)
    # Separate Adam optimizers for the policy (theta) and value (phi) parameters, as in Spinning Up.
    optimizers = (torch.optim.Adam(model.pi.parameters(), lr=config.pi_lr), torch.optim.Adam(model.v.parameters(), lr=config.vf_lr))
    if checkpoint and checkpoint.get("optimizer") and not refit:
        optimizers[0].load_state_dict(checkpoint["optimizer"]["pi"])
        optimizers[1].load_state_dict(checkpoint["optimizer"]["v"])
    # Optional linear decay of the policy learning rate over this run (`--final-pi-lr`).
    initial_rate = config.pi_lr
    final_rate = getattr(args, "final_pi_lr", None) or initial_rate

    spec, fixed = model.spec, [str(path) for path in args.fixed_opponent]
    snapshots: dict[int, dict] = {start: wire_state(model)}
    sent: set[int] = set()
    seeds = [args.seed * 7919 + total_episodes + index for index in range(max(args.workers, 1))]
    if args.workers:
        context = mp.get_context("spawn")
        connections, processes = [], []
        for seed in seeds:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, seed, spec, args.envs, mix, fixed), daemon=True)
            process.start()
            connections.append(parent)
            processes.append(process)
    else:
        local = RolloutCollector(seeds[0], spec, args.envs, mix, fixed)

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"
    began = time.monotonic()
    try:
        for iteration in range(start + 1, start + args.updates + 1):
            progress = (iteration - start - 1) / args.updates
            config.pi_lr = initial_rate + (final_rate - initial_rate) * progress
            for group in optimizers[0].param_groups:
                group["lr"] = config.pi_lr
            weights = wire_state(model)
            updates = {key: state for key, state in snapshots.items() if key not in sent}
            sent.update(updates)
            message = (weights, updates, list(snapshots), args.steps)
            collect_began = time.monotonic()
            if args.workers:
                for connection in connections:
                    connection.send(message)
                results = [connection.recv() for connection in connections]
            else:
                local.sync(*message[:3])
                results = [local.collect(args.steps)]
            collect_seconds = time.monotonic() - collect_began
            batch, stats = assemble(results, config, args.device)
            learn_began = time.monotonic()
            metrics = update(model, optimizers, batch, config)
            samples = len(batch["actions"])
            total_steps += samples
            total_episodes += sum(games for games, _ in stats.values())
            metadata = {"algorithm": "PPO-Clip, Spinning Up process (custom PyTorch)", "model": spec, "updates": iteration,
                        "steps": total_steps, "episodes": total_episodes, "seed": args.seed,
                        "status": "value network refit on a frozen policy" if refit else "trained candidate; evaluate before promotion",
                        "opponents": {**mix, "frozen": round(1 - sum(mix.values()), 3)},
                        "elapsed_seconds": round(time.monotonic() - began, 1)}
            row = {**metadata, **metrics, "pi_lr": config.pi_lr, "samples": samples, "collect_seconds": round(collect_seconds, 2),
                   "learn_seconds": round(time.monotonic() - learn_began, 2),
                   "win_rates": {kind: [wins, games] for kind, (games, wins) in sorted(stats.items())}}
            with log_path.open("a", encoding="utf8") as log:
                log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if metrics["value_spread"] < 1e-3:
                # The served run lost its value network this way at update 550 (see the README).
                print(f"warning: the value network predicts one value everywhere at update {iteration}; "
                      "its ReLUs have probably died. Refit it with --refit-value.", file=sys.stderr, flush=True)
            if iteration % args.snapshot_every == 0:
                snapshots[iteration] = wire_state(model)
                for key in sorted(snapshots)[:-args.pool_size]:
                    del snapshots[key]
            if args.archive_every and iteration % args.archive_every == 0:
                save_checkpoint(output_dir / "snapshots" / f"ppo-{iteration:05d}.pt", model, None, metadata, config)
            # Optionally stop once the run has played enough games (`--total-games`).
            enough = bool(getattr(args, "total_games", None)) and total_episodes >= args.total_games
            if iteration % 10 == 0 or iteration == start + args.updates or enough:
                save_checkpoint(args.output, model, optimizers, metadata, config)
            if enough:
                if args.archive_every and iteration % args.archive_every:
                    save_checkpoint(output_dir / "snapshots" / f"ppo-{iteration:05d}.pt", model, None, metadata, config)
                break
    finally:
        if args.workers:
            for connection in connections:
                try:
                    connection.send(None)
                except OSError:
                    pass
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--updates", type=int, default=1000, help="most updates to run")
    parser.add_argument("--total-games", dest="total_games", type=int, help="stop once the run has played this many games")
    parser.add_argument("--steps", type=int, default=128, help="decisions per environment per update")
    parser.add_argument("--envs", type=int, default=8, help="lockstep environments per rollout worker")
    parser.add_argument("--workers", type=int, default=8, help="rollout processes; 0 collects in-process")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4, help="learner CPU threads")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/candidate/ppo.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--channels", type=int, default=64, help="policy network width")
    parser.add_argument("--blocks", type=int, default=4, help="policy network residual blocks")
    parser.add_argument("--value-channels", dest="value_channels", type=int, default=32)
    parser.add_argument("--value-blocks", dest="value_blocks", type=int, default=4)
    # Spinning Up's hyperparameters; omitted ones keep its defaults (see PPOConfig).
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lam", dest="gae_lambda", type=float)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--pi-lr", dest="pi_lr", type=float)
    parser.add_argument("--vf-lr", dest="vf_lr", type=float)
    parser.add_argument("--train-pi-iters", dest="train_pi_iters", type=int)
    parser.add_argument("--train-v-iters", dest="train_v_iters", type=int)
    parser.add_argument("--target-kl", dest="target_kl", type=float)
    parser.add_argument("--epochs", type=int, help="passes over the batch for each of lines 6 and 7")
    parser.add_argument("--minibatch", type=int, help="decisions per gradient step; 0 = Spinning Up's full-batch steps")
    parser.add_argument("--final-pi-lr", dest="final_pi_lr", type=float, help="decay the policy learning rate linearly to this")
    parser.add_argument("--refit-value", dest="refit_value", action="store_true",
                        help="with --resume: freeze the policy, reset the value network and train only it")
    parser.add_argument("--mix", default="random=0.1,tactical=0.15,lookahead=0.35")
    parser.add_argument("--fixed-opponent", action="append", default=[], type=Path)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--archive-every", type=int, default=50)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
