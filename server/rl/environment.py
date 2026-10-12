"""Learner-perspective transitions against fixed opponents."""
from __future__ import annotations

import random
from collections.abc import Callable

from server.game import Game

Policy = Callable[[Game], int]


class TrainingEnvironment:
    def __init__(self, opponent: Policy, seed: int = 0, max_actions: int = 400) -> None:
        self.opponent = opponent
        self.rng = random.Random(seed)
        self.max_actions = max_actions
        self.episodes = 0
        self.learner = 0
        self.game = Game()

    def _opponent_turn(self) -> None:
        while self.game.winner is None and self.game.current != self.learner:
            self.game.apply(self.opponent(self.game))

    def reset(self) -> Game:
        self.learner = self.episodes % 2
        self.episodes += 1
        self.game = Game()
        self._opponent_turn()
        return self.game

    def step(self, action: int) -> tuple[Game, float, bool, bool]:
        if self.game.current != self.learner or self.game.winner is not None:
            raise ValueError("Reset the environment before the next learner decision.")
        self.game.apply(action)
        self._opponent_turn()
        terminated = self.game.winner is not None
        truncated = not terminated and self.game.revision >= self.max_actions
        reward = (1.0 if self.game.winner == self.learner else -1.0) if terminated else 0.0
        return self.game, reward, terminated, truncated
