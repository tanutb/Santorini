"""Online rooms for two players, plus the website itself.

Games against the agent and pass-and-play games run entirely in the browser (web/local-game.js),
so this server needs no machine-learning libraries. Configuration lives in server/security.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from server.game import Game
from server.security import BodyLimit, RateLimiter, Settings, clean_name, client_ip, key_matches, security_headers
from server.storage import RoomStore

ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.0.0"
CLEANUP_INTERVAL = 6 * 3600
logger = logging.getLogger(__name__)
settings = Settings()
limiter = RateLimiter(settings.rate_limits)
store = RoomStore(Path(os.getenv("SANTORINI_DB", str(ROOT / "data/rooms.sqlite3"))))
rooms: dict[str, dict] = {}
last_used: dict[str, float] = {}
locks: dict[str, asyncio.Lock] = {}
connections: dict[str, dict[WebSocket, int]] = {}
sockets_by_ip: dict[str, int] = {}


async def housekeeping() -> None:
    """Expire week-old rooms and drop idle rooms from memory, so a long-running server stays small."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        store.cleanup()
        idle = time.monotonic() - CLEANUP_INTERVAL
        for code in [code for code, used in last_used.items() if used < idle and not connections.get(code)]:
            rooms.pop(code, None)
            locks.pop(code, None)
            last_used.pop(code, None)
            connections.pop(code, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.cleanup()
    task = asyncio.create_task(housekeeping())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    store.close()


app = FastAPI(title="Santorini online rooms", version=VERSION, lifespan=lifespan,
              docs_url="/docs" if settings.api_docs else None, redoc_url=None,
              openapi_url="/openapi.json" if settings.api_docs else None)
app.add_middleware(GZipMiddleware, minimum_size=1024)
if settings.allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type", "X-API-Key"], max_age=600)
app.add_middleware(BodyLimit, limit=settings.max_body)


@app.middleware("http")
async def headers_and_caching(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=2592000" if path.startswith("/static/fonts/") else "no-cache"
    elif path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    security_headers(request, response.headers, is_page=path in ("/", "/learn.html"))
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_keys and not key_matches(x_api_key, settings.api_keys):
        raise HTTPException(401, "A valid X-API-Key header is required.")


def limit(action: str):
    def dependency(request: Request) -> None:
        limiter.check(client_ip(request), action)
    return Depends(dependency)


api_key = Depends(require_api_key)


class CreateRoom(BaseModel):
    mode: str = Field(default="online", max_length=12)
    name: str = Field(default="Player one", min_length=1, max_length=48)


class JoinRoom(BaseModel):
    name: str = Field(default="Guest", min_length=1, max_length=48)


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
    last_used[code] = time.monotonic()
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
    return {"code": room["code"], "mode": "online", "seat": seat, "players": room["players"],
            "connected": [i in active for i in range(2)], "game": room["game"], "legal": game.legal_actions(),
            "history": room["history"][-100:], "rematch": room["rematch"], "ready": room["players"][1] is not None}


async def broadcast(room: dict) -> None:
    for socket, seat in list(connections.get(room["code"], {}).items()):
        try:
            await asyncio.wait_for(socket.send_json({"type": "state", "room": snapshot(room, seat)}), timeout=3)
        except (OSError, RuntimeError, asyncio.TimeoutError, WebSocketDisconnect):
            connections.get(room["code"], {}).pop(socket, None)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "rules": "classic", "version": VERSION, "online": True}


@app.post("/api/rooms", status_code=201, dependencies=[api_key, limit("create")])
async def create_room(request: CreateRoom) -> dict:
    if request.mode != "online":
        raise HTTPException(422, "This server hosts online games. Games against the agent and pass and play run in your browser.")
    if store.count() >= settings.max_rooms:
        raise HTTPException(503, "The server is at room capacity. Try again later.")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    code = "".join(secrets.choice(alphabet) for _ in range(6))
    while store.load(code):
        code = "".join(secrets.choice(alphabet) for _ in range(6))
    token = secrets.token_urlsafe(32)
    room = {"code": code, "mode": "online", "players": [clean_name(request.name, "Player one"), None],
            "tokens": [digest(token), None], "game": Game().to_dict(), "history": [], "rematch": []}
    rooms[code] = room
    last_used[code] = time.monotonic()
    store.save(room)
    return {"token": token, "room": snapshot(room, 0)}


@app.post("/api/rooms/{code}/join", dependencies=[api_key, limit("join")])
async def join_room(code: str, request: JoinRoom) -> dict:
    room = get_room(code)
    async with room_lock(room["code"]):
        if room["players"][1] is not None:
            raise HTTPException(409, "This room already has two players.")
        token = secrets.token_urlsafe(32)
        room["players"][1] = clean_name(request.name, "Guest")
        room["tokens"][1] = digest(token)
        store.save(room)
        await broadcast(room)
        return {"token": token, "room": snapshot(room, 1)}


@app.get("/api/rooms/{code}", dependencies=[api_key, limit("read")])
async def read_room(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    return snapshot(room, authenticate(room, authorization))


@app.post("/api/rooms/{code}/actions", dependencies=[api_key, limit("write")])
async def action(code: str, request: ActionRequest, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["players"][1] is None:
            raise HTTPException(409, "Wait for your friend to join before placing workers.")
        if request.revision != room["game"]["revision"]:
            raise HTTPException(409, "The board changed. Your position has been refreshed; choose again.")
        if seat != room["game"]["current"]:
            raise HTTPException(403, "It is the other player's turn.")
        game = Game.from_dict(room["game"])
        try:
            message = game.apply(request.action)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        room["game"] = game.to_dict()
        room["history"].append({"player": seat, "text": message, "revision": game.revision})
        store.save(room)
        await broadcast(room)
        return snapshot(room, seat)


@app.post("/api/rooms/{code}/resign", dependencies=[api_key, limit("write")])
async def resign(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["game"]["winner"] is not None:
            raise HTTPException(409, "This game has already finished.")
        room["game"].update(winner=1 - seat, reason="Opponent resigned", revision=room["game"]["revision"] + 1)
        store.save(room)
        await broadcast(room)
        return snapshot(room, seat)


@app.post("/api/rooms/{code}/rematch", dependencies=[api_key, limit("write")])
async def rematch(code: str, authorization: str | None = Header(default=None)) -> dict:
    room = get_room(code)
    seat = authenticate(room, authorization)
    async with room_lock(room["code"]):
        if room["game"]["winner"] is None:
            raise HTTPException(409, "Finish the current game before requesting a rematch.")
        if seat not in room["rematch"]:
            room["rematch"].append(seat)
        if len(room["rematch"]) == 2:
            # Monotonic revisions also invalidate delayed requests from previous games.
            room["game"] = Game(revision=room["game"]["revision"] + 1).to_dict()
            room["history"], room["rematch"] = [], []
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
    return urlparse(origin).netloc in {socket.headers.get("host"), *forwarded} or origin.rstrip("/") in settings.allowed_origins


@app.websocket("/ws/{code}")
async def websocket_room(socket: WebSocket, code: str) -> None:
    ip = client_ip(socket)
    try:
        limiter.check(ip, "socket")
    except HTTPException:
        await socket.close(code=1013)
        return
    if not same_origin(socket) or sockets_by_ip.get(ip, 0) >= settings.max_sockets_per_ip:
        await socket.close(code=1008)
        return
    await socket.accept()
    sockets_by_ip[ip] = sockets_by_ip.get(ip, 0) + 1
    room = None
    try:
        auth = await asyncio.wait_for(socket.receive_json(), timeout=10)
        if not isinstance(auth, dict) or not isinstance(auth.get("token"), str):
            await socket.close(code=1008)
            return
        if settings.api_keys and not key_matches(auth.get("key") if isinstance(auth.get("key"), str) else None, settings.api_keys):
            await socket.close(code=1008)
            return
        candidate = get_room(code)
        seat = seat_for(candidate, auth["token"])
        if len(connections.get(candidate["code"], {})) >= settings.max_sockets_per_room:
            await socket.close(code=1013)
            return
        room = candidate
        connections.setdefault(room["code"], {})[socket] = seat
        await broadcast(room)
        window, count = time.monotonic(), 0
        while True:
            message = await socket.receive_text()
            # Clients ping every 25 s; more than 20 messages in 10 s ends the connection.
            now = time.monotonic()
            if now - window > 10:
                window, count = now, 0
            count += 1
            if count > 20:
                await socket.close(code=1008)
                return
            if message == "ping":
                await socket.send_json({"type": "pong"})
    except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, RuntimeError):
        pass
    except HTTPException:
        await socket.close(code=1008)
    finally:
        sockets_by_ip[ip] = max(0, sockets_by_ip.get(ip, 1) - 1)
        if not sockets_by_ip[ip]:
            sockets_by_ip.pop(ip, None)
        if room is not None:
            connections.get(room["code"], {}).pop(socket, None)
            await broadcast(room)


@app.get("/static/config.js", include_in_schema=False)
async def page_config() -> Response:
    """The page served by this server plays online here, and sends this server's public key if one is set."""
    body = f"export const SERVER_URL = '';\nexport const API_KEY = {json.dumps(settings.public_key)};\n"
    return Response(body, media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "web/index.html", headers={"Cache-Control": "no-cache"})


@app.get("/learn.html", include_in_schema=False)
async def learn() -> FileResponse:
    return FileResponse(ROOT / "web/learn.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
