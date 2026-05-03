import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from log import log
from memory import embedder

DB_PATH = Path(__file__).parent / "memories.db"
EMBED_DIM = 768

_conn: sqlite3.Connection | None = None
_index: faiss.IndexFlatIP | None = None
_id_map: list[int] = []  # FAISS position → SQLite id
_lock = threading.RLock()


def _db() -> sqlite3.Connection:
    assert _conn is not None, "call load() first"
    return _conn


def _idx() -> faiss.IndexFlatIP:
    assert _index is not None, "call load() first"
    return _index


def load():
    global _conn
    _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT    NOT NULL,
            embedding   BLOB    NOT NULL,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        )
    """)
    _conn.commit()
    _rebuild_index()
    count = _index.ntotal if _index else 0
    log("memory", f"store ready — {count} memories loaded")


def _rebuild_index():
    global _index, _id_map
    rows = _db().execute("SELECT id, embedding FROM memories").fetchall()
    _index = faiss.IndexFlatIP(EMBED_DIM)
    _id_map = []
    if rows:
        vecs = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        _index.add(vecs)  # type: ignore[call-arg]
        _id_map = [r[0] for r in rows]


def _relative(iso_ts: str) -> str:
    days = (datetime.utcnow() - datetime.fromisoformat(iso_ts)).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def retrieve(text: str, top_k: int = 5, threshold: float = 0.35) -> str:
    with _lock:
        if _index is None or _index.ntotal == 0:
            return ""
        vec = embedder.encode_query(text).reshape(1, -1).astype(np.float32)
        k = min(top_k, _index.ntotal)
        scores, positions = _idx().search(vec, k)  # type: ignore[call-arg]
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or score < threshold:
                continue
            row = _db().execute("SELECT text, created_at FROM memories WHERE id = ?", (_id_map[pos],)).fetchone()
            if row:
                results.append((row[0], row[1]))
        if not results:
            return ""
        lines = "\n".join(f"- ({_relative(ts)}) {m}" for m, ts in results)
        return f"[Relevant context — prefer more recent on conflicts]\n{lines}\n\n"


def search_similar(text: str, k: int = 3, threshold: float = 0.85) -> list[tuple[int, float, str]]:
    """Returns (id, score, text) tuples above threshold, for dedup/DECIDE."""
    with _lock:
        if _index is None or _index.ntotal == 0:
            return []
        vec = embedder.encode_doc(text).reshape(1, -1).astype(np.float32)
        n = min(k, _index.ntotal)
        scores, positions = _idx().search(vec, n)  # type: ignore[call-arg]
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or score < threshold:
                continue
            mem_id = _id_map[pos]
            row = _db().execute("SELECT text FROM memories WHERE id = ?", (mem_id,)).fetchone()
            if row:
                results.append((mem_id, float(score), row[0]))
        return results


def add(text: str) -> int:
    with _lock:
        vec = embedder.encode_doc(text).astype(np.float32)
        now = datetime.utcnow().isoformat()
        cur = _db().execute(
            "INSERT INTO memories (text, embedding, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (text, vec.tobytes(), now, now),
        )
        _db().commit()
        new_id = cur.lastrowid
        assert new_id is not None
        _idx().add(vec.reshape(1, -1))  # type: ignore[call-arg]
        _id_map.append(new_id)
        return new_id
