"""The browser's forced-win search as a teacher for training (see distill.py).

The search (web/tactics.js) ranks every complete turn by searching all replies. Where it proves that
some actions lose and others do not, it allows only the best-ranked ones. Distillation trains the
network toward those actions, so it learns what the search would stop. The search runs in Node
(scripts/teacher.mjs), the same code the website plays with.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from server.game import ACTION_COUNT, Game

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "teacher.mjs"


class SearchTeacher:
    def __init__(self, depth: int = 4, budget: int = 1_000_000) -> None:
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Training with the search teacher needs Node.js 22 or newer on the PATH.")
        self.depth, self.budget = depth, budget
        self.process = subprocess.Popen([node, str(SCRIPT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf8", bufsize=1)

    def masks(self, games: list[Game]) -> tuple[np.ndarray, np.ndarray]:
        """Per game, the allowed actions as a boolean mask (the legal mask where the search has no
        preference) and whether the search excluded anything."""
        request = {"depth": self.depth, "budget": self.budget, "games": [game.to_dict() for game in games]}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The search teacher stopped; run `node scripts/teacher.mjs` to see its error.")
        answers = json.loads(line)["masks"]
        masks = np.zeros((len(games), ACTION_COUNT), dtype=bool)
        taught = np.zeros(len(games), dtype=bool)
        for row, (game, actions) in enumerate(zip(games, answers)):
            if actions is None:
                masks[row] = game.mask()
            else:
                masks[row, actions] = True
                taught[row] = True
        return masks, taught

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
