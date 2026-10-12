"""Small persistent room store. Run one application worker per database."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class RoomStore:
    """One long-lived connection, used only from the event-loop thread.

    WAL with synchronous=NORMAL is crash-safe (no corruption); at worst the last few
    commits are lost on power failure, which a game room can afford.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS rooms (code TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL)")

    def load(self, code: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM rooms WHERE code = ?", (code,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, room: dict) -> None:
        self.connection.execute(
            "INSERT INTO rooms VALUES (?, ?, ?) ON CONFLICT(code) DO UPDATE SET payload=excluded.payload, updated=excluded.updated",
            (room["code"], json.dumps(room, separators=(",", ":")), time.time()))

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]

    def cleanup(self) -> None:
        self.connection.execute("DELETE FROM rooms WHERE updated < ?", (time.time() - 7 * 86400,))

    def close(self) -> None:
        self.connection.close()
