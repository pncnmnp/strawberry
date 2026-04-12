from datetime import datetime
from tools.db import get_conn, init
from pixies import time_parse, query_expand
from log import log

init()

def save_note(content: str, tag: str = "general") -> str:
    """Save a personal note or piece of information the user wants to remember.

    Args:
        content: The note to save.
        tag: Topic tag to categorise the note (e.g. 'reminders', 'ideas', 'work'). Defaults to 'general'.
    """
    tag = tag.lower().strip()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notes VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), tag, content),
        )
    log("notes", f"saved [{tag}] {content[:60]}")
    return f"Note saved under '{tag}'."


def recall_notes(
    query: str | None = None,
    tag: str | None = None,
    time_range: str | None = None,
) -> str:
    """Search and recall saved personal notes. Supports keyword search, tag filter, and natural language time range.

    Args:
        query: Keyword or topic to search notes for.
        tag: Filter notes by tag.
        time_range: Natural language time window, e.g. 'last hour', 'last 2 hours', 'today', 'yesterday', 'last week', 'in March'.
    """
    from_dt, to_dt = time_parse(time_range) if time_range else (None, None)

    if time_range:
        log("notes", f"time range: {from_dt} → {to_dt}")

    terms = query_expand(query) if query else []
    if terms:
        log("notes", f"expanded: {terms}")

    params: list = []
    where: list[str] = []

    if terms:
        fts_match = " OR ".join(f'"{t}"' for t in terms)
        base = "SELECT timestamp, tag, content FROM notes WHERE notes MATCH ?"
        params.append(fts_match)
    else:
        base = "SELECT timestamp, tag, content FROM notes WHERE 1=1"

    if tag:
        where.append("tag = ?")
        params.append(tag.lower().strip())
    if from_dt:
        where.append("timestamp >= ?")
        params.append(from_dt)
    if to_dt:
        where.append("timestamp <= ?")
        params.append(to_dt)

    suffix = (" AND " + " AND ".join(where) if where else "") + " ORDER BY timestamp DESC"
    sql = base + suffix

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "No notes found."

    return "\n".join(f"[{r['timestamp']}] ({r['tag']}) {r['content']}" for r in rows)
