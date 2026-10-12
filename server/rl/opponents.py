"""Explicit baselines for training and evaluation, not presented as PPO."""
from __future__ import annotations

import random
from server.game import Game, NEIGHBORS

CENTER_DISTANCE = tuple(abs(cell // 5 - 2) + abs(cell % 5 - 2) for cell in range(25))


def tactical_action(game: Game, rng: random.Random) -> int:
    legal = game.legal_actions()
    if game.phase == "place":
        return max(legal, key=lambda cell: -abs(cell // 5 - 2) - abs(cell % 5 - 2) + rng.random())
    if game.phase == "build":
        def score(cell: int) -> float:
            threat = any(p >= 0 and game.heights[p] == 2 and cell in NEIGHBORS[p] for p in game.workers[1 - game.current])
            return (12 if game.heights[cell] == 3 and threat else 0) + rng.random()
        return max(legal, key=score)
    def score_move(action: int) -> float:
        worker, target = divmod(action - 25, 25)
        origin = game.workers[game.current][worker]
        return (100 if game.heights[origin] == 2 and game.heights[target] == 3 else 0) + game.heights[target] * 2 + rng.random()
    return max(legal, key=score_move)


def _can_win(heights: list[int], movers: list[int], occupied: set[int]) -> bool:
    return any(heights[origin] == 2 and heights[target] == 3 and target not in occupied
               for origin in movers for target in NEIGHBORS[origin])


def _position_score(heights: list[int], mine: list[int], theirs: list[int]) -> float:
    """Static evaluation after a complete turn, from the mover's perspective."""
    occupied = set(mine) | set(theirs)
    score = 0.0
    for workers, sign in ((mine, 1.0), (theirs, -1.0)):
        threats = set()
        for origin in workers:
            height = heights[origin]
            score += sign * 3.0 * height
            for target in NEIGHBORS[origin]:
                level = heights[target]
                if target in occupied or level == 4 or level > height + 1:
                    continue
                score += sign * (0.3 + 0.45 * level)
                if height == 2 and level == 3:
                    threats.add(target)
            score -= sign * 0.15 * CENTER_DISTANCE[origin]
        score += sign * (4.0 * len(threats) + (12.0 if len(threats) > 1 else 0.0))
    return score


def _turn_score(heights: list[int], mine: list[int], theirs: list[int]) -> float:
    occupied = set(mine) | set(theirs)
    if _can_win(heights, theirs, occupied):
        return -1000.0 + _position_score(heights, mine, theirs)
    if not any(target not in occupied and heights[target] < 4 and heights[target] <= heights[origin] + 1
               for origin in theirs for target in NEIGHBORS[origin]):
        return 1000.0
    return _position_score(heights, mine, theirs)


def _build_scores(game: Game, worker: int, target: int, rng: random.Random) -> list[tuple[float, int]]:
    me = game.current
    mine = list(game.workers[me])
    mine[worker] = target
    theirs = list(game.workers[1 - me])
    occupied = set(mine) | set(theirs)
    heights = list(game.heights)
    scores = []
    for build in NEIGHBORS[target]:
        if build in occupied or heights[build] == 4:
            continue
        heights[build] += 1
        scores.append((_turn_score(heights, mine, theirs) + 0.05 * rng.random(), build))
        heights[build] -= 1
    return scores


def lookahead_action(game: Game, rng: random.Random) -> int:
    """One-ply complete-turn search: win now, never allow an immediate reply win when avoidable."""
    legal = game.legal_actions()
    me = game.current
    if game.phase == "place":
        def place_score(cell: int) -> float:
            friends = [c for c in game.workers[me] if c >= 0]
            near = sum(1 for c in friends if cell in NEIGHBORS[c])
            return -CENTER_DISTANCE[cell] + 0.6 * near + rng.random()
        return max(legal, key=place_score)
    if game.phase == "build":
        worker = game.selected
        target = game.workers[me][worker]
        return max(_build_scores(game, worker, target, rng))[1]
    best_action, best_score = legal[0], float("-inf")
    for action in legal:
        worker, target = divmod(action - 25, 25)
        if game.heights[game.workers[me][worker]] == 2 and game.heights[target] == 3:
            return action
        scores = _build_scores(game, worker, target, rng)
        score = max(scores)[0] if scores else float("-inf")
        if score > best_score:
            best_action, best_score = action, score
    return best_action


BASELINES = {
    "random": lambda game, rng: rng.choice(game.legal_actions()),
    "tactical": tactical_action,
    "lookahead": lookahead_action,
}
