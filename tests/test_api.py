import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SANTORINI_DB", str(tmp_path / "rooms.sqlite3"))
    import server.app as module
    module = importlib.reload(module)
    with TestClient(module.app) as client:
        yield client, module


def create(client, mode="online"):
    response = client.post("/api/rooms", json={"mode": mode, "name": "Alice"})
    assert response.status_code == 201
    data = response.json()
    return data["room"]["code"], {"Authorization": "Bearer " + data["token"]}, data["token"]


def test_online_ownership_revision_and_persistence(service):
    client, module = service
    code, owner, token = create(client)
    endpoint = f"/api/rooms/{code}"
    assert client.post(endpoint + "/actions", headers=owner, json={"action": 0, "revision": 0}).status_code == 409
    guest = client.post(endpoint + "/join", json={"name": "Bob"}).json()
    guest_headers = {"Authorization": "Bearer " + guest["token"]}
    assert client.post(endpoint + "/join", json={"name": "Eve"}).status_code == 409
    assert client.get(endpoint).status_code == 401
    assert client.post(endpoint + "/actions", headers=guest_headers, json={"action": 0, "revision": 0}).status_code == 403
    response = client.post(endpoint + "/actions", headers=owner, json={"action": 0, "revision": 0})
    assert response.status_code == 200
    assert client.post(endpoint + "/actions", headers=owner, json={"action": 1, "revision": 0}).status_code == 409
    module.rooms.clear()
    state = client.get(endpoint, headers=owner).json()
    assert state["game"]["workers"][0][0] == 0
    assert "tokens" not in state
    stored = module.store.load(code)
    assert stored["tokens"][0] != token


def test_websocket_initial_state_and_live_update(service):
    client, _ = service
    code, owner, token = create(client, "local")
    with client.websocket_connect(f"/ws/{code}") as socket:
        socket.send_json({"token": token})
        assert socket.receive_json()["room"]["connected"][0]
        client.post(f"/api/rooms/{code}/actions", headers=owner, json={"action": 12, "revision": 0})
        assert socket.receive_json()["room"]["game"]["workers"][0][0] == 12
        socket.send_text("ping")
        assert socket.receive_json()["type"] == "pong"


def test_rematch_requires_both_online_players_and_preserves_revision(service):
    client, _ = service
    code, owner, _ = create(client)
    endpoint = f"/api/rooms/{code}"
    guest = client.post(endpoint + "/join", json={"name": "Bob"}).json()
    guest_headers = {"Authorization": "Bearer " + guest["token"]}
    assert client.post(endpoint + "/rematch", headers=owner).status_code == 409
    client.post(endpoint + "/resign", headers=owner)
    first = client.post(endpoint + "/rematch", headers=owner).json()
    assert first["game"]["winner"] == 1
    second = client.post(endpoint + "/rematch", headers=guest_headers).json()
    assert second["game"]["winner"] is None and second["game"]["revision"] == 2
    assert second["rematch"] == []


def test_invalid_action_types_and_no_fake_agent(service, monkeypatch):
    client, module = service
    code, headers, _ = create(client, "local")
    assert client.post(f"/api/rooms/{code}/actions", headers=headers, json={"action": True, "revision": 0}).status_code == 422
    monkeypatch.setattr(module.agent_service, "status", lambda: {"available": False})
    assert client.post("/api/rooms", json={"mode": "ai"}).status_code == 503


def test_disconnected_client_does_not_interrupt_broadcast(service):
    import asyncio
    from fastapi import WebSocketDisconnect

    client, module = service
    code, _, _ = create(client, "local")

    class ClosedSocket:
        async def send_json(self, message):
            raise WebSocketDisconnect(code=1006)

    class LiveSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, message):
            self.messages.append(message)

    closed, live = ClosedSocket(), LiveSocket()
    module.connections[code] = {closed: 0, live: 0}
    asyncio.run(module.broadcast(module.rooms[code]))
    assert closed not in module.connections[code]
    assert live.messages[0]["type"] == "state"
    module.connections[code].clear()


def test_hints_are_for_practice_games_and_reject_stale_positions(service, monkeypatch):
    client, module = service
    code, headers, _ = create(client, "online")
    assert client.post(f"/api/rooms/{code}/hint", headers=headers, json={"revision": 0}).status_code == 403
    code, headers, _ = create(client, "local")
    endpoint = f"/api/rooms/{code}"
    assert client.post(endpoint + "/hint", json={"revision": 0}).status_code == 401
    assert client.post(endpoint + "/hint", headers=headers, json={"revision": 3}).status_code == 409
    monkeypatch.setattr(module.agent_service, "suggest", lambda game: {"action": game.legal_actions()[0], "revision": game.revision,
                                                                       "cell": game.legal_actions()[0], "worker": None,
                                                                       "title": "Place", "detail": "", "confidence": .5})
    response = client.post(endpoint + "/hint", headers=headers, json={"revision": 0})
    assert response.status_code == 200 and response.json()["action"] == 0


def test_online_rooms_never_carry_insight_and_static_code_revalidates(service):
    client, module = service
    code, headers, _ = create(client, "online")
    module.rooms[code]["insight"] = {"revision": 0, "p0": .9}
    assert client.get(f"/api/rooms/{code}", headers=headers).json()["insight"] is None
    response = client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"
    assert response.headers.get("content-encoding") == "gzip"


def test_agent_rooms_record_a_known_level(service, monkeypatch):
    client, module = service
    monkeypatch.setattr(module.agent_service, "status", lambda: {"available": True, "metadata": {"steps": 1}})
    assert client.post("/api/rooms", json={"mode": "ai", "level": "nonsense"}).status_code == 422
    response = client.post("/api/rooms", json={"mode": "ai", "level": "full"})
    assert response.status_code == 201 and response.json()["room"]["level"] == "full"
    assert client.post("/api/rooms", json={"mode": "local"}).json()["room"]["level"] is None


def test_websocket_origin_must_match_host_or_forwarded_host(service):
    from starlette.websockets import WebSocketDisconnect as Closed
    client, _ = service
    code, _, token = create(client, "local")
    with pytest.raises(Closed):
        with client.websocket_connect(f"/ws/{code}", headers={"origin": "https://evil.example"}) as socket:
            socket.receive_json()
    headers = {"origin": "https://game.trycloudflare.com", "x-forwarded-host": "game.trycloudflare.com"}
    with client.websocket_connect(f"/ws/{code}", headers=headers) as socket:
        socket.send_json({"token": token})
        assert socket.receive_json()["type"] == "state"
