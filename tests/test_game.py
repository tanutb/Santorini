import random

import pytest

from server.game import ACTION_COUNT, OBSERVATION_SIZE, Game


def placed():
    game = Game()
    for cell in (0, 1, 23, 24):
        game.apply(cell)
    return game


def test_placement_and_move_build_sequence():
    game = placed()
    assert game.current == 0 and game.phase == "move"
    game.apply(25 + 5)
    assert game.phase == "build" and game.selected == 0
    game.apply(0)
    assert game.heights[0] == 1 and game.current == 1 and game.phase == "move"


def test_winning_move_is_immediate():
    game = placed()
    game.heights[0], game.heights[5] = 2, 3
    game.apply(30)
    assert game.winner == 0 and game.heights[5] == 3
    assert game.legal_actions() == []


def test_illegal_moves_are_atomic():
    game = placed()
    game.heights[5], game.heights[6] = 2, 4
    before = game.to_dict()
    for action in (30, 31, 26, -1, 75, True):
        with pytest.raises(ValueError):
            game.apply(action)
        assert game.to_dict() == before


def test_can_descend_multiple_levels_and_build_vacated_square():
    game = placed()
    game.heights[0] = 3
    game.apply(30)
    assert 0 in game.legal_actions()
    game.apply(0)
    assert game.heights[0] == 4


def test_dome_blocked_opponent_loses():
    game = placed()
    game.workers = [[0, 1], [23, 24]]
    game.phase, game.selected = "build", 0
    for cell in (17, 18, 19, 22):
        game.heights[cell] = 4
    game.apply(5)
    assert game.winner == 0 and "no complete" in game.reason


def test_serialization_and_observation():
    game = placed()
    assert Game.from_dict(game.to_dict()).to_dict() == game.to_dict()
    assert len(game.observation()) == OBSERVATION_SIZE
    assert len(game.mask()) == ACTION_COUNT
    assert sum(game.mask()) == len(game.legal_actions())


def test_random_games_always_terminate_with_valid_boards():
    rng = random.Random(42)
    for _ in range(100):
        game = Game()
        for _ in range(250):
            if game.winner is not None:
                break
            assert game.legal_actions()
            game.apply(rng.choice(game.legal_actions()))
            positions = [cell for pair in game.workers for cell in pair if cell >= 0]
            assert len(positions) == len(set(positions))
            assert all(0 <= height <= 4 for height in game.heights)
            assert all(game.heights[cell] < 4 for cell in positions)
        assert game.winner in (0, 1)
