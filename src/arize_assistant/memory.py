"""Long-term memory: short facts stored in a local SQLite database.

Search uses SQLite's full-text index (FTS5) when available and falls back to
simple word matching otherwise.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_WORD = re.compile(r"[\w']+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "so", "that",
    "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "with",
    "you", "your",
}  # fmt: skip


@dataclass
class Memory:
    id: int
    text: str
    created: str


class MemoryStore:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created TEXT NOT NULL)"
        )
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "text, content='memories', content_rowid='id')"
            )
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False
        self._db.commit()

    def add(self, text: str) -> Memory:
        text = " ".join(text.split())
        if not text:
            raise ValueError("Memory text is empty")
        for m in self.all():
            if m.text.lower() == text.lower():
                return m
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self._db.execute(
            "INSERT INTO memories (text, created) VALUES (?, ?)", (text, created)
        )
        if self._fts:
            self._db.execute(
                "INSERT INTO memories_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, text)
            )
        self._db.commit()
        return Memory(cur.lastrowid, text, created)

    def forget(self, memory_id: int) -> bool:
        row = self._db.execute("SELECT text FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return False
        if self._fts:
            self._db.execute(
                "INSERT INTO memories_fts (memories_fts, rowid, text) VALUES ('delete', ?, ?)",
                (memory_id, row[0]),
            )
        self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._db.commit()
        return True

    def all(self) -> list[Memory]:
        rows = self._db.execute("SELECT id, text, created FROM memories ORDER BY id").fetchall()
        return [Memory(*r) for r in rows]

    def recent(self, limit: int = 5) -> list[Memory]:
        rows = self._db.execute(
            "SELECT id, text, created FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Memory(*r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        words = [w for w in _WORD.findall(query.lower()) if w not in _STOP and len(w) > 1]
        if not words:
            return []
        if self._fts:
            match = " OR ".join('"' + w.replace('"', "") + '"*' for w in words)
            rows = self._db.execute(
                "SELECT m.id, m.text, m.created FROM memories_fts f "
                "JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH ? "
                "ORDER BY bm25(memories_fts) LIMIT ?",
                (match, limit),
            ).fetchall()
            return [Memory(*r) for r in rows]
        scored = []
        for m in self.all():
            low = m.text.lower()
            score = sum(1 for w in words if w in low)
            if score:
                scored.append((score, m.id, m))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [m for _, _, m in scored[:limit]]

    def context_for(self, message: str, limit: int = 8) -> list[Memory]:
        """Memories worth showing the model for this message: relevant ones, then recent ones."""
        picked: dict[int, Memory] = {}
        for m in self.search(message, limit) + self.recent(limit):
            if len(picked) >= limit:
                break
            picked.setdefault(m.id, m)
        return sorted(picked.values(), key=lambda m: m.id)

    def close(self) -> None:
        self._db.close()
