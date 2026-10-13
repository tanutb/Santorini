"""The browser's search, served to Python training code."""
import shutil

import pytest

from server.game import Game
from server.rl.teacher import SearchTeacher

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="the search teacher runs in Node")


def ladder() -> Game:
    """A reported game: the opponent ladders A1 and B1 a tempo ahead of the agent's own race."""
    game = Game(workers=[[24, 0], [17, 23]], current=1, phase="move", turn=8)
    for cell, height in {0: 2, 1: 2, 19: 2, 23: 1}.items():
        game.heights[cell] = height
    return game


@needs_node
def test_teacher_returns_the_browser_search_answer():
    teacher = SearchTeacher(depth=4, budget=1_000_000)
    try:
        masks, taught = teacher.masks([ladder(), Game()])
    finally:
        teacher.close()
    assert [action for action in range(75) if masks[0, action]] == [36, 37]  # C4 to B3 or C3, toward A1.
    assert taught.tolist() == [True, False]
    assert masks[1].tolist() == Game().mask(), "no preference: the legal mask, untaught"
