"""Load a genuine PPO checkpoint; never silently substitute a heuristic."""
from __future__ import annotations

import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any

from server.game import Game

ROOT = Path(__file__).resolve().parents[1]


class AgentService:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.getenv("SANTORINI_MODEL", str(ROOT / "checkpoints/ppo.pt")))
        self._model: Any = None
        self._stamp = 0
        self._metadata: dict = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from server.rl.model import load_model
        stamp = self.path.stat().st_mtime_ns
        if self._model is not None and self._stamp == stamp:
            return
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        model = load_model(checkpoint)
        torch.set_num_threads(2)
        self._model, self._stamp = model, stamp
        self._metadata = checkpoint["metadata"]

    def status(self) -> dict:
        try:
            with self._lock:
                self._load()
                metadata = dict(self._metadata)
            evaluation_path = self.path.with_suffix(".evaluation.json")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf8")) if evaluation_path.exists() else None
            if evaluation and evaluation.get("checkpoint_steps") != metadata.get("steps"):
                evaluation = None
            return {"available": True, "metadata": metadata, "evaluation": evaluation}
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            return {"available": False, "message": "Train a PPO checkpoint to enable the agent.", "detail": type(error).__name__}

    def action(self, game: Game) -> int:
        from server.rl.train import choose
        with self._lock:
            self._load()
            return choose(self._model, game)

    def _forward(self, game: Game):
        import torch
        with self._lock:
            self._load()
            with torch.inference_mode():
                observation = torch.tensor([game.observation()], dtype=torch.float32)
                mask = torch.tensor([game.mask()], dtype=torch.bool)
                distribution, value = self._model(observation, mask)
                return distribution.probs[0].tolist(), float(value[0])

    def suggest(self, game: Game) -> dict:
        """Return the policy's highest-probability legal action without applying it."""
        from server.coach import explain_action
        probabilities, _ = self._forward(game)
        action = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"action": action, "revision": game.revision, "source": "PPO policy suggestion",
                "confidence": round(probabilities[action], 3), **explain_action(game, action)}

    MIN_SKILL = 0.1

    def calibration(self) -> dict | None:
        """Platt fit from `server.rl.calibration --fit`, used only if it matches the loaded weights and
        its held-out Brier score beats a constant guess by at least MIN_SKILL; otherwise the read is noise."""
        path = self.path.with_suffix(".calibration.json")
        try:
            fit = json.loads(path.read_text(encoding="utf8"))
            skill = 1 - fit["held_out_brier"] / fit["constant_brier"]
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            return None
        with self._lock:
            self._load()
            steps = self._metadata.get("steps")
        return fit if fit.get("checkpoint_steps") == steps and skill >= self.MIN_SKILL else None

    def insight(self, game: Game) -> dict | None:
        """Calibrated win chance for seat 0 between evenly matched players, or None without a fit.

        The raw critic is optimistic for whoever moves, so the board is scored from both sides
        and the difference is mapped through the self-play calibration.
        """
        from server.rl.calibration import calibrated, swapped
        fit = self.calibration()
        if fit is None or game.winner is not None:
            return None
        _, mover_value = self._forward(game)
        other = swapped(game)
        other_value = self._forward(other)[1] if other.legal_actions() else -1.0
        mover = calibrated(max(-1.0, min(1.0, mover_value)) - max(-1.0, min(1.0, other_value)), fit)
        return {"revision": game.revision, "p0": round(mover if game.current == 0 else 1 - mover, 3)}


class AgentLevels:
    """Earlier checkpoints of the same PPO run, offered as lighter opponents.

    `levels.json` beside the served model lists them, strongest last, e.g.
    {"default": "full", "levels": [{"id": "early", "name": "Early", "checkpoint": "levels/early.pt"}, ...]}.
    The served model is always available as "full"; hints and insight always use it.
    """

    def __init__(self, main: AgentService) -> None:
        self.main = main
        self.entries = [{"id": "full", "name": "Full strength"}]
        self.services = {"full": main}
        self.default = "full"
        try:
            manifest = json.loads(main.path.with_name("levels.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            return
        entries = []
        for entry in manifest.get("levels", []):
            level = str(entry.get("id", ""))
            if level == "full":
                entries.append({"id": "full", "name": entry.get("name", "Full strength")})
            elif level.isidentifier() and entry.get("checkpoint"):
                self.services[level] = AgentService(main.path.parent / entry["checkpoint"])
                entries.append({"id": level, "name": str(entry.get("name", level))})
        if not any(entry["id"] == "full" for entry in entries):
            entries.append(self.entries[0])
        self.entries = entries
        self.default = manifest.get("default", "full") if manifest.get("default") in self.services else "full"

    def get(self, level: str | None) -> AgentService:
        return self.services.get(level or self.default, self.main)

    def describe(self) -> list[dict]:
        levels = []
        for entry in self.entries:
            status = self.services[entry["id"]].status()
            row = {**entry, "available": status["available"], "default": entry["id"] == self.default}
            if status["available"]:
                row["steps"] = status["metadata"].get("steps")
                results = (status.get("evaluation") or {}).get("results", {})
                row["win_rates"] = {name: result["win_rate"] for name, result in results.items()}
            levels.append(row)
        return levels
