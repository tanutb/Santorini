"""Factual descriptions of policy suggestions, never invented agent reasoning."""
from server.game import Game, NEIGHBORS, coordinate


def explain_action(game: Game, action: int) -> dict:
    if action not in game.legal_actions():
        raise ValueError("The suggested action must be legal.")
    if game.phase == "place":
        exits = sum(cell not in game.occupied() for cell in NEIGHBORS[action])
        return {"cell": action, "worker": None, "title": f"Place on {coordinate(action)}",
                "detail": f"This square has {exits} unoccupied neighbors. Your two workers will need room to move and build."}
    if game.phase == "build":
        dome = game.heights[action] == 3
        return {"cell": action, "worker": game.selected,
                "title": f"{'Cap the tower' if dome else 'Build a floor'} on {coordinate(action)}",
                "detail": "A dome permanently blocks this square." if dome else f"This creates level {game.heights[action] + 1}. Both players can use the tower."}
    worker, target = divmod(action - 25, 25)
    origin = game.workers[game.current][worker]
    height = game.heights[target]
    if game.heights[origin] == 2 and height == 3:
        detail = "Moving up to the third level wins immediately."
    elif height == 2:
        detail = "This puts your worker on level two, ready to climb to an adjacent third level on a later turn."
    elif height > game.heights[origin]:
        detail = "This climbs one level. You must build next with this worker."
    else:
        detail = "After moving, build on a free square beside this worker."
    return {"cell": target, "worker": worker, "title": f"Move {coordinate(origin)} to {coordinate(target)}", "detail": detail}
