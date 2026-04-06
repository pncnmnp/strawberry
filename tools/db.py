import sqlite3

DB_PATH = "notes.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with get_conn() as conn:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(
                timestamp UNINDEXED,
                tag,
                content,
                tokenize = 'porter ascii'
            )
        """)
