"""Pure, authoritative classic Santorini rules shared by play and training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

BOARD_SIZE = 5
ACTION_COUNT = 75
OBSERVATION_SIZE = 230
NEIGHBORS = tuple(
    tuple(j for j in range(25) if j != i and abs(i // 5 - j // 5) <= 1 and abs(i % 5 - j % 5) <= 1)
    for i in range(25)
)


@dataclass
class Game:
    heights: list[int] = field(default_factory=lambda: [0] * 25)
    workers: list[list[int]] = field(default_factory=lambda: [[-1, -1], [-1, -1]])
    current: int = 0
    phase: str = "place"
    selected: int = -1
    winner: int | None = None
    reason: str = ""
    turn: int = 1
    revision: int = 0

    def occupied(self) -> set[int]:
        return {cell for pair in self.workers for cell in pair if cell >= 0}

    def legal_actions(self) -> list[int]:
        """Return only actions that allow the required turn to complete."""
        if self.winner is not None:
            return []
        occupied = self.occupied()
        if self.phase == "place":
            return [i for i in range(25) if i not in occupied]
        if self.phase == "build":
            origin = self.workers[self.current][self.selected]
            return [i for i in NEIGHBORS[origin] if i not in occupied and self.heights[i] < 4]
        result = []
        for worker, origin in enumerate(self.workers[self.current]):
            for target in NEIGHBORS[origin]:
                if target in occupied or self.heights[target] == 4:
                    continue
                if self.heights[target] > self.heights[origin] + 1:
                    continue
                wins = self.heights[origin] == 2 and self.heights[target] == 3
                next_occupied = (occupied - {origin}) | {target}
                can_build = any(i not in next_occupied and self.heights[i] < 4 for i in NEIGHBORS[target])
                if wins or can_build:
                    result.append(25 + worker * 25 + target)
        return result

    def apply(self, action: int) -> str:
        """Validate and apply one phase action; return a readable history entry."""
        if type(action) is not int or action not in self.legal_actions():
            raise ValueError("That action is not legal in the current position.")
        actor = self.current
        if self.phase == "place":
            worker = self.workers[actor].index(-1)
            self.workers[actor][worker] = action
            message = f"Placed worker {worker + 1} on {coordinate(action)}"
            if -1 not in self.workers[actor]:
                if actor == 0:
                    self.current = 1
                else:
                    self.current, self.phase = 0, "move"
        elif self.phase == "move":
            worker, target = divmod(action - 25, 25)
            origin = self.workers[actor][worker]
            self.workers[actor][worker] = target
            self.selected = worker
            message = f"Moved {coordinate(origin)} to {coordinate(target)}"
            if self.heights[origin] == 2 and self.heights[target] == 3:
                self.winner, self.reason = actor, "Reached the third level"
            else:
                self.phase = "build"
        else:
            self.heights[action] += 1
            kind = "a dome" if self.heights[action] == 4 else f"level {self.heights[action]}"
            message = f"Built {kind} on {coordinate(action)}"
            self.current, self.phase, self.selected = 1 - actor, "move", -1
            self.turn += 1
        self.revision += 1
        if self.winner is None and self.phase == "move" and not self.legal_actions():
            self.winner, self.reason = 1 - self.current, "Opponent has no complete legal turn"
        return message

    def observation(self) -> list[float]:
        """Encode the board from the player-to-act perspective (230 floats)."""
        features = [float(height == level) for level in range(5) for height in self.heights]
        for player in (self.current, 1 - self.current):
            for cell in self.workers[player]:
                features.extend(float(cell == i) for i in range(25))
        features.extend(float(self.phase == phase) for phase in ("place", "move", "build"))
        features.extend(float(self.selected == worker) for worker in range(2))
        return features

    def mask(self) -> list[bool]:
        legal = set(self.legal_actions())
        return [i in legal for i in range(ACTION_COUNT)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Game:
        return cls(**data)


def coordinate(cell: int) -> str:
    return f"{chr(65 + cell % 5)}{cell // 5 + 1}"
