import asyncio
import importlib

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient


def start(tmp_path, monkeypatch, **env):
    """A fresh app with its own database; rate limits are off unless a test turns them on."""
    monkeypatch.setenv("SANTORINI_DB", str(tmp_path / "rooms.sqlite3"))
    monkeypatch.setenv("SANTORINI_RATE_LIMITS", "off")
    for name in ("SANTORINI_API_KEYS", "SANTORINI_ALLOWED_ORIGINS", "SANTORINI_API_DOCS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    import server.app as module
    return importlib.reload(module)


@pytest.fixture
def service(tmp_path, monkeypatch):
    module = start(tmp_path, monkeypatch)
    with TestClient(module.app) as client:
        yield client, module


def create(client, headers=None):
    response = client.post("/api/rooms", json={"name": "Alice"}, headers=headers or {})
    assert response.status_code == 201, response.text
    data = response.json()
    return data["room"]["code"], {"Authorization": "Bearer " + data["token"], **(headers or {})}, data["token"]


def pair(client, headers=None):
    code, owner, owner_token = create(client, headers)
    guest = client.post(f"/api/rooms/{code}/join", json={"name": "Bob"}, headers=headers or {}).json()
    return code, owner, owner_token, {"Authorization": "Bearer " + guest["token"], **(headers or {})}, guest["token"]


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
    assert client.post(endpoint + "/actions", headers=owner, json={"action": 0, "revision": 0}).status_code == 200
    assert client.post(endpoint + "/actions", headers=owner, json={"action": 1, "revision": 0}).status_code == 409
    module.rooms.clear()
    state = client.get(endpoint, headers=owner).json()
    assert state["game"]["workers"][0][0] == 0 and "tokens" not in state
    assert module.store.load(code)["tokens"][0] != token


def test_only_online_games_and_strict_action_types(service):
    client, _ = service
    for mode in ("ai", "local"):
        response = client.post("/api/rooms", json={"mode": mode})
        assert response.status_code == 422 and "browser" in response.json()["detail"]
    code, owner, _, _, _ = pair(client)
    for action in (True, "3", 75, -1, 1.5):
        assert client.post(f"/api/rooms/{code}/actions", headers=owner, json={"action": action, "revision": 0}).status_code == 422


def test_websocket_initial_state_and_live_update(service):
    client, _ = service
    code, owner, token, _, _ = pair(client)
    with client.websocket_connect(f"/ws/{code}") as socket:
        socket.send_json({"token": token})
        assert socket.receive_json()["room"]["connected"][0]
        client.post(f"/api/rooms/{code}/actions", headers=owner, json={"action": 12, "revision": 0})
        assert socket.receive_json()["room"]["game"]["workers"][0][0] == 12
        socket.send_text("ping")
        assert socket.receive_json()["type"] == "pong"


def test_rematch_requires_both_players_and_preserves_revision(service):
    client, _ = service
    code, owner, _, guest, _ = pair(client)
    endpoint = f"/api/rooms/{code}"
    assert client.post(endpoint + "/rematch", headers=owner).status_code == 409
    client.post(endpoint + "/resign", headers=owner)
    assert client.post(endpoint + "/rematch", headers=owner).json()["game"]["winner"] == 1
    second = client.post(endpoint + "/rematch", headers=guest).json()
    assert second["game"]["winner"] is None and second["game"]["revision"] == 2 and second["rematch"] == []


def test_disconnected_client_does_not_interrupt_broadcast(service):
    client, module = service
    code, _, _ = create(client)

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
    assert closed not in module.connections[code] and live.messages[0]["type"] == "state"
    module.connections[code].clear()


def test_names_are_cleaned(service):
    client, _ = service
    room = client.post("/api/rooms", json={"name": "  Ana\u0000‮  of\tThera  "}).json()["room"]
    assert room["players"][0] == "Ana of Thera"
    assert client.post("/api/rooms", json={"name": "​"}).json()["room"]["players"][0] == "Player one"


def test_api_key_is_required_for_http_and_websockets(tmp_path, monkeypatch):
    module = start(tmp_path, monkeypatch, SANTORINI_API_KEYS="first-key,second-key")
    with TestClient(module.app) as client:
        assert client.get("/api/health").status_code == 200, "health stays open for availability checks"
        assert client.post("/api/rooms", json={}).status_code == 401
        assert client.post("/api/rooms", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
        code, owner, token, _, _ = pair(client, {"X-API-Key": "second-key"})
        assert client.get(f"/api/rooms/{code}", headers={"Authorization": owner["Authorization"]}).status_code == 401
        assert client.get(f"/api/rooms/{code}", headers=owner).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws/{code}") as socket:
                socket.send_json({"token": token})
                socket.receive_json()
        with client.websocket_connect(f"/ws/{code}") as socket:
            socket.send_json({"token": token, "key": "first-key"})
            assert socket.receive_json()["type"] == "state"
        config = client.get("/static/config.js").text
        assert 'API_KEY = "first-key"' in config and "SERVER_URL = ''" in config


def test_rate_limits_answer_429_with_retry_after(tmp_path, monkeypatch):
    module = start(tmp_path, monkeypatch, SANTORINI_RATE_LIMITS="on")
    with TestClient(module.app) as client:
        statuses = [client.post("/api/rooms", json={}).status_code for _ in range(12)]
        assert statuses[:10] == [201] * 10 and statuses[10] == 429
        limited = client.post("/api/rooms", json={})
        assert int(limited.headers["retry-after"]) >= 1
        assert client.get("/api/health").status_code == 200


def test_rate_limiter_refills_over_time():
    from server.security import RateLimiter
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    for _ in range(10):
        limiter.check("1.2.3.4", "create")
    with pytest.raises(Exception):
        limiter.check("1.2.3.4", "create")
    limiter.check("5.6.7.8", "create")
    now[0] += 6
    limiter.check("1.2.3.4", "create")


def test_security_headers_csp_and_hidden_docs(service):
    client, _ = service
    page = client.get("/")
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "'unsafe-inline'" not in csp
    for response in (page, client.get("/api/health")):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/api/health").headers["cache-control"] == "no-store"
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404
    static = client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
    assert static.headers["cache-control"] == "no-cache" and static.headers.get("content-encoding") == "gzip"


def test_learning_page_is_served_with_the_page_policy(service):
    client, _ = service
    page = client.get("/learn.html")
    assert page.status_code == 200 and "How the agent learns" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    # Equations are pre-rendered MathML; inline styles would be blocked by the policy.
    assert "<math" in page.text and ' style="' not in page.text


def test_large_bodies_are_rejected(service):
    client, _ = service
    assert client.post("/api/rooms", content=b'{"name":"' + b"x" * 6000 + b'"}',
                       headers={"Content-Type": "application/json"}).status_code == 413

    def chunks():
        yield b'{"name":"'
        for _ in range(10):
            yield b"y" * 1000
        yield b'"}'
    assert client.post("/api/rooms", content=chunks(), headers={"Content-Type": "application/json"}).status_code == 413


def test_there_is_no_admin_endpoint(tmp_path, monkeypatch):
    module = start(tmp_path, monkeypatch)
    with TestClient(module.app) as client:
        assert client.get("/api/admin/stats", headers={"X-Admin-Key": "anything"}).status_code == 404


def test_websocket_origin_must_match_host_or_forwarded_host(service):
    client, _ = service
    code, _, token, _, _ = pair(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/{code}", headers={"origin": "https://evil.example"}) as socket:
            socket.receive_json()
    headers = {"origin": "https://game.trycloudflare.com", "x-forwarded-host": "game.trycloudflare.com"}
    with client.websocket_connect(f"/ws/{code}", headers=headers) as socket:
        socket.send_json({"token": token})
        assert socket.receive_json()["type"] == "state"


def test_a_separately_hosted_site_can_be_allowed(tmp_path, monkeypatch):
    module = start(tmp_path, monkeypatch, SANTORINI_ALLOWED_ORIGINS="https://example.github.io")
    with TestClient(module.app) as client:
        allowed = client.options("/api/rooms", headers={"Origin": "https://example.github.io", "Access-Control-Request-Method": "POST",
                                                        "Access-Control-Request-Headers": "x-api-key,content-type"})
        assert allowed.headers["access-control-allow-origin"] == "https://example.github.io"
        assert "x-api-key" in allowed.headers["access-control-allow-headers"].lower()
        assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "https://evil.example"}).headers
        code, _, token, _, _ = pair(client)
        with client.websocket_connect(f"/ws/{code}", headers={"origin": "https://example.github.io"}) as socket:
            socket.send_json({"token": token})
            assert socket.receive_json()["type"] == "state"
