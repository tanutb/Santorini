"""Authoritative multiplayer API and same-origin website."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import pickle
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from server.game import Game
from server.inference import AgentLevels, AgentService
from server.storage import RoomStore

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
# Pause before each agent action so the previous animation can finish and the turn reads clearly.
AI_DELAY = 0.6
INFERENCE_ERRORS = (OSError, RuntimeError, ValueError, ImportError, KeyError, EOFError, pickle.UnpicklingError)
store = RoomStore(Path(os.getenv("SANTORINI_DB", str(ROOT / "data/rooms.sqlite3"))))
agent_service = AgentService()
agent_levels = AgentLevels(agent_service)
rooms: dict[str, dict] = {}
locks: dict[str, asyncio.Lock] = {}
connections: dict[str, dict[WebSocket, int]] = {}
ai_tasks: dict[str, asyncio.Task] = {}
background: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.cleanup()
    yield
    tasks = [*ai_tasks.values(), *background]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    store.close()


app = FastAPI(title="Santorini", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def cache_policy(request: Request, call_next):
    """Revalidate code on every load (cheap 304s via ETag); fonts never change between releases."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        immutable = request.url.path.startswith("/static/fonts/")
        response.headers["Cache-Control"] = "public, max-age=2592000" if immutable else "no-cache"
    return response


class CreateRoom(BaseModel):
    mode: Literal["online", "local", "ai"]
    name: str = Field(default="You", min_length=1, max_length=24)
    level: str | None = Field(default=None, max_length=24)


class JoinRoom(BaseModel):
    name: str = Field(default="Guest", min_length=1, max_length=24)


class HintRequest(BaseModel):
    revision: StrictInt = Field(ge=0)


class ActionRequest(BaseModel):
    action: StrictInt = Field(ge=0, lt=75)
    revision: StrictInt = Field(ge=0)


def get_room(code: str) -> dict:
    code = code.upper()
    if len(code) != 6 or not code.isalnum():
        raise HTTPException(404, "Room not found. Check the six-character code.")
    if code not in rooms:
        room = store.load(code)
        if room is None:
            raise HTTPException(404, "Room not found or expired. Create a new room.")
        rooms[code] = room
    return rooms[code]


def room_lock(code: str) -> asyncio.Lock:
    return locks.setdefault(code, asyncio.Lock())


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def seat_for(room: dict, token: str) -> int:
    for seat, stored in enumerate(room["tokens"]):
        if stored and hmac.compare_digest(stored, digest(token)):
            return seat
    raise HTTPException(401, "This player session is not valid. Rejoin with your original browser.")


def authenticate(room: dict, authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "A player session is required.")
    return seat_for(room, authorization[7:])


def snapshot(room: dict, seat: int) -> dict:
    game = Game.from_dict(room["game"])
    active = list(connections.get(room["code"], {}).values())
    return {"code": room["code"], "mode": room["mode"], "seat": seat,
            "players": room["players"], "connected": [i in active for i in range(2)],
            "game": room["game"], "legal": game.legal_actions(), "history": room["history"][-100:],
            "rematch": room["rematch"], "ai_error": room.get("ai_error"),
            "insight": room.get("insight") if room["mode"] != "online" else None,
            "level": room.get("level"),
            "ready": room["players"][1] is not None}


async def broadcast(room: dict) -> None:
    for socket, seat in list(connections.get(room["code"], {}).items()):
        try:
            await asyncio.wait_for(socket.send_json({"type": "state", "room": snapshot(room, seat)}), timeout=3)
        except (OSError, RuntimeError, asyncio.TimeoutError, WebSocketDisconnect):
            connections.get(room["code"], {}).pop(socket, None)


def apply_action(room: dict, action: int) -> None:
    game = Game.from_dict(room["game"])
    actor = game.current
    message = game.apply(action)
    room["game"] = game.to_dict()
    room["history"].append({"player": actor, "text": message, "revision": game.revision})
    room["ai_error"] = None
    store.save(room)


def schedule_insight(room: dict) -> None:
    """Coached games get the critic's read of each new position, pushed when ready."""
    game = room["game"]
    # Scored at turn boundaries only: the calibration was fitted on move-phase positions.
    if room["mode"] == "online" or game["winner"] is not None or game["phase"] != "move":
        return
    task = asyncio.create_task(update_insight(room, Game.from_dict(game)))
    background.add(task)
    task.add_done_callback(background.discard)


async def update_insight(room: dict, game: Game) -> None:
    try:
        insight = await asyncio.to_thread(agent_service.insight, game)
    except INFERENCE_ERRORS:
        return
    if insight is None:
        return
    async with room_lock(room["code"]):
        if room["game"]["revision"] == insight["revision"]:
            room["insight"] = insight
            await broadcast(room)


async def run_ai(room: dict) -> None:
    try:
        while True:
            async with room_lock(room["code"]):
                state = Game.from_dict(room["game"])
                if state.current != 1 or state.winner is not None:
                    return
            await asyncio.sleep(AI_DELAY)
            action = await asyncio.to_thread(agent_levels.get(room.get("level")).action, state)
            async with room_lock(room["code"]):
                if room["game"]["revision"] != state.revision:
                    return
                apply_action(room, action)
                await broadcast(room)
                schedule_insight(room)
    except INFERENCE_ERRORS:
        logger.exception("Agent inference failed")
        room["ai_error"] = "The agent could not load its checkpoint. Restore it, then reconnect to retry."
        await broadcast(room)


def schedule_ai(room: dict) -> None:
    if room["mode"] != "ai":
        return
    task = ai_tasks.get(room["code"])
    if task is None or task.done():
        ai_tasks[room["code"]] = asyncio.create_task(run_ai(room))


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "rules": "classic", "version": "2.0.0"}


@app.get("/api/agent")
async def agent_status() -> dict:
    status = await asyncio.to_thread(agent_service.status)
    status["levels"] = await asyncio.to_thread(agent_levels.describe)
    return status


@app.post("/api/rooms", status_code=201)
async def create_room(request: CreateRoom) -> dict:
    level = None
    if request.mode == "ai":
        level = request.level or agent_levels.default
        if level not in agent_levels.services:
            raise HTTPException(422, "Choose one of the listed agent strengths.")
        if not (await asyncio.to_thread(agent_levels.get(level).status))["available"]:
            raise HTTPException(503, "The PPO agent needs a trained checkpoint. You can still play locally or online.")
    if store.count() >= 10000:
        raise HTTPException(503, "The server is at room capacity. Try again later.")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    code = "".join(secrets.choice(alphabet) for _ in range(6))
    while store.load(code):
        code = "".join(secrets.choice(alphabet) for _ in range(6))
    token = secrets.token_urlsafe(32)
    opponent = {"online": None, "local": "Player two", "ai": "PPO agent"}[request.mode]
    room = {"code": code, "mode": request.mode, "players": [request.name.strip() or "Player one", opponent],
            "tokens": [digest(token), None], "game": Game().to_dict(), "history": [], "rematch": [], "ai_error": None,
            "level": level}
    rooms[code] = room
    store.save(room)
    return {"token": token, "room": snapshot(room, 0)}


@app.post("/api/rooms/{code}/join")
async def join_room(code: str, request: JoinRoom) -> dict:
    room = get_room(code)
    async with room_lock(room["code"]):
        if room["mode"] != "online" or room["players"][1] is not None:
            raise HTTPException(409, "This room already has two players, or is not an online room.")
        token = secrets.token_urlsafe(32)
        room["players"][1] = request.name.strip() or "Guest"
        room["tokens"][1] = digest(token)
        store.save(room)
        await broadcast(room)
        return {"token": token, "room": snapshot(room, 1)}


@app.get("/api/rooms/{code}")
async def read_room(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    schedule_ai(room)
    return snapshot(room, seat)


@app.post("/api/rooms/{code}/actions")
async def action(code: str, request: ActionRequest, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["players"][1] is None:
            raise HTTPException(409, "Wait for your friend to join before placing workers.")
        if request.revision != room["game"]["revision"]:
            raise HTTPException(409, "The board changed. Your position has been refreshed; choose again.")
        if room["mode"] != "local" and seat != room["game"]["current"]:
            raise HTTPException(403, "It is the other player's turn.")
        try:
            apply_action(room, request.action)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        await broadcast(room)
        schedule_ai(room)
        schedule_insight(room)
        return snapshot(room, seat)



@app.post("/api/rooms/{code}/hint")
async def hint(code: str, request: HintRequest, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["mode"] == "online":
            raise HTTPException(403, "Practice hints are available in agent and pass-and-play games.")
        game = Game.from_dict(room["game"])
        if game.winner is not None or (room["mode"] == "ai" and game.current != seat):
            raise HTTPException(409, "Wait until your turn to request a hint.")
        if request.revision != game.revision:
            raise HTTPException(409, "The position changed. Request a hint for the current board.")
    try:
        suggestion = await asyncio.to_thread(agent_service.suggest, game)
    except INFERENCE_ERRORS as error:
        raise HTTPException(503, "Hints need a working PPO checkpoint. You can continue playing.") from error
    if room["game"]["revision"] != game.revision:
        raise HTTPException(409, "The position changed while the hint was being prepared.")
    return suggestion


@app.post("/api/rooms/{code}/resign")
async def resign(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["game"]["winner"] is not None:
            raise HTTPException(409, "This game has already finished.")
        if room["mode"] == "local":
            seat = room["game"]["current"]
        room["game"]["winner"] = 1 - seat
        room["game"]["reason"] = "Opponent resigned"
        room["game"]["revision"] += 1
        store.save(room)
        await broadcast(room)
        return snapshot(room, seat)


@app.post("/api/rooms/{code}/rematch")
async def rematch(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["game"]["winner"] is None:
            raise HTTPException(409, "Finish the current game before requesting a rematch.")
        if seat not in room["rematch"]:
            room["rematch"].append(seat)
        if room["mode"] != "online" or len(room["rematch"]) == 2:
            # Monotonic revisions also invalidate delayed requests from previous games.
            room["game"] = Game(revision=room["game"]["revision"] + 1).to_dict()
            room["history"], room["rematch"], room["ai_error"], room["insight"] = [], [], None, None
        store.save(room)
        await broadcast(room)
        return snapshot(room, seat)


def same_origin(socket: WebSocket) -> bool:
    """Block cross-site WebSocket use. Tunnels and proxies may rewrite Host, so also accept the
    forwarded host; browsers cannot set that header on a WebSocket handshake."""
    origin = socket.headers.get("origin")
    if not origin:
        return True
    forwarded = [host.strip() for host in socket.headers.get("x-forwarded-host", "").split(",") if host.strip()]
    return urlparse(origin).netloc in {socket.headers.get("host"), *forwarded}


@app.websocket("/ws/{code}")
async def websocket_room(socket: WebSocket, code: str) -> None:
    if not same_origin(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    room = None
    try:
        auth = await asyncio.wait_for(socket.receive_json(), timeout=10)
        if not isinstance(auth, dict) or not isinstance(auth.get("token"), str):
            await socket.close(code=1008)
            return
        room = get_room(code)
        seat = seat_for(room, auth["token"])
        connections.setdefault(room["code"], {})[socket] = seat
        await broadcast(room)
        schedule_ai(room)
        while True:
            message = await socket.receive_text()
            if message == "ping":
                await socket.send_json({"type": "pong"})
    except (WebSocketDisconnect, asyncio.TimeoutError, ValueError):
        pass
    except HTTPException:
        await socket.close(code=1008)
    finally:
        if room is not None:
            connections.get(room["code"], {}).pop(socket, None)
            await broadcast(room)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "web/index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
