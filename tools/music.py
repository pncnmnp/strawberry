import subprocess, random, urllib.request, urllib.parse, base64, json, time, tempfile, os, re, sqlite3
from pathlib import Path
from log import log
import mutagen  # type: ignore[attr-defined]

VLC = "/Applications/VLC.app/Contents/MacOS/VLC"
HOST, PORT, PASS = "127.0.0.1", 8765, "strawberry"
EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg", ".opus"}
DB_PATH = Path(__file__).parent.parent / "music.db"

_proc = None
_m3u = None

def _dir() -> str | None:
    p = Path(__file__).parent.parent / "config.local.json"
    if not p.exists(): return None
    return json.loads(p.read_text()).get("MUSIC_DIR")

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS tracks (
        path TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT NOT NULL DEFAULT '',
        album TEXT NOT NULL DEFAULT '',
        mtime REAL NOT NULL
    )""")
    return conn

def _read_tags(path: Path) -> dict:
    """Read ID3/metadata tags from an audio file using mutagen."""
    title, artist, album = path.stem, "", ""
    try:
        f = mutagen.File(path, easy=True)  # type: ignore[attr-defined]
        if f:
            title = (f.get("title") or [path.stem])[0]
            artist = (f.get("artist") or [""])[0]
            album = (f.get("album") or [""])[0]
    except Exception:
        pass
    # mtime - time of most recent content modification
    return {"path": str(path), "title": title, "artist": artist, "album": album, "mtime": path.stat().st_mtime}

def _build_index() -> int:
    """Scan MUSIC_DIR, read tags, populate the DB. Returns track count."""
    d = _dir()
    if not d: return 0
    files = sorted(f for ext in EXTS for f in Path(d).rglob(f"*{ext}"))
    conn = _db()
    # grab existing mtimes so we only re-read changed files
    existing = {r["path"]: r["mtime"] for r in conn.execute("SELECT path, mtime FROM tracks").fetchall()}
    current_paths = set()
    batch = []
    for f in files:
        p = str(f)
        current_paths.add(p)
        mt = f.stat().st_mtime
        if p in existing and existing[p] == mt:
            continue
        batch.append(_read_tags(f))
    # upsert changed/new tracks
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO tracks (path, title, artist, album, mtime) VALUES (:path, :title, :artist, :album, :mtime)",
            batch)
    # remove tracks whose files no longer exist
    stale = set(existing) - current_paths
    if stale:
        conn.executemany("DELETE FROM tracks WHERE path = ?", [(p,) for p in stale])
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    conn.close()
    return count

def _ensure_db():
    """Build the index if the DB doesn't exist or is empty."""
    if not DB_PATH.exists():
        _build_index()
        return
    conn = _db()
    count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    conn.close()
    if count == 0:
        _build_index()

def _api(cmd="", **kw) -> dict:
    url = f"http://{HOST}:{PORT}/requests/status.json"
    qs = ({"command": cmd} | kw) if cmd else kw
    if qs: url += "?" + urllib.parse.urlencode(qs)
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + base64.b64encode(f":{PASS}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=2) as r: return json.loads(r.read())  # type: ignore[no-any-return]
    except: return {"error": "unreachable"}

_GENERIC = {"", "all", "any", "everything", "anything", "random", "music", "some music", "library"}

def _word_match_clause(query: str) -> tuple[str, list[str]]:
    """Build a WHERE clause that requires every word in the query to appear
    somewhere across title, artist, album, or path. Returns (sql, params).
    e.g. "Music Man" -> each word must independently match at least one column.
    This handles smashed-together tags like 'MusicManChannel'.
    """
    words = query.lower().split()
    if not words: return "1=1", []
    clauses, params = [], []
    for w in words:
        p = f"%{w}%"
        clauses.append("(LOWER(title) LIKE ? OR LOWER(artist) LIKE ? OR LOWER(album) LIKE ? OR LOWER(path) LIKE ?)")
        params.extend([p, p, p, p])
    return " AND ".join(clauses), params

def _clean(s: str) -> str:
    """Strip model special tokens (e.g. Gemma's <|"|>) and whitespace."""
    return re.sub(r"<\|[^|]*\|>", "", s).strip()

def _alive() -> bool:
    """Check if VLC is responding on our HTTP port — works even across Python restarts."""
    return "error" not in _api()

def _kill_stale():
    """Kill whatever is holding our HTTP control port.
    lsof -ti :PORT is definitive — no regex guessing about command lines.
    """
    result = subprocess.run(["lsof", "-ti", f":{PORT}"],
                            capture_output=True, text=True)
    for pid in result.stdout.split():
        subprocess.run(["kill", "-9", pid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)  # let the port actually free up

def _launch(files, shuffle=False):
    global _proc, _m3u
    _kill_stale()
    if shuffle: random.shuffle(files := list(files))

    # m3u is a simple plaintext playlist format that VLC can read.
    # Example .m3u content:
    '''
    #EXTM3U
    /Users/username/username/music/artist/song1.mp3
    /Users/username/username/music/artist/song2.mp3
    '''
    if _m3u and os.path.exists(_m3u): os.unlink(_m3u)
    with tempfile.NamedTemporaryFile("w", suffix=".m3u", delete=False, prefix="straw_") as f:
        f.write("#EXTM3U\n" + "\n".join(str(p) for p in files))
        _m3u = f.name

    # just in case, kill_stale should have taken care of this though
    if _proc and _proc.poll() is None: _proc.terminate()
    # --intf dummy is a non-GUI mode that still allows HTTP control, 
    # --extraintf http enables the HTTP interface,
    # --no-video disables any video output
    _proc = subprocess.Popen([VLC, "--intf", "dummy", "--extraintf", "http",
        f"--http-host={HOST}", f"--http-port={PORT}", f"--http-password={PASS}",
        "--no-video", _m3u], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)


def music_play(query: str = "", shuffle: bool = False) -> str:
    """Play music from the local library by artist, album, or song. Leave query empty to play all.

    Args:
        query: Artist, album, or song to play. Do not use query, if you want to play everything.
        shuffle: Shuffle the matching tracks.
    """
    if not _dir(): return "Set MUSIC_DIR in config.local.json first."
    _ensure_db()
    query = _clean(query)
    if query.lower() in _GENERIC: query = ""
    conn = _db()
    if query:
        where, params = _word_match_clause(query)
        rows = conn.execute(f"SELECT path FROM tracks WHERE {where}", params).fetchall()
    else:
        rows = conn.execute("SELECT path FROM tracks").fetchall()
    conn.close()
    files = [r["path"] for r in rows]
    if not files: return f"Nothing matched '{query}'."
    log("music", f"play '{query}' shuffle={shuffle} n={len(files)}")
    if _alive():
        _api("pl_empty") # clear current playlist
        for f in (random.sample(files, len(files)) if shuffle else files): _api("in_enqueue", input=f"file://{f}")
        _api("pl_play")
    else:
        _launch(files, shuffle)
    return f"Playing {len(files)} track(s){' shuffled' if shuffle else ''}."


def music_control(action: str) -> str:
    """Control playback: pause, resume, next, previous, stop.

    Args:
        action: pause | resume | next | previous | stop
    """
    global _proc
    action = _clean(action)
    if action == "stop":
        if _proc and _proc.poll() is None: _proc.terminate(); _proc = None
        return "Stopped."
    if not _alive(): return "Nothing is playing."
    _api({"pause": "pl_pause", "resume": "pl_play", "next": "pl_next", "previous": "pl_previous"}[action])
    return f"{action.capitalize()}."


def music_set_volume(level: int) -> str:
    """Set playback volume 0–100.

    Args:
        level: 0 (silent) to 100 (full).
    """
    if not _alive(): return "Nothing is playing."
    # why 256?
    # VLC's internal volume scale is 0–256, but the tool exposes 0–100. So level * 2.56 maps 100 -> 256.
    _api("volume", val=int(max(0, min(100, level)) * 2.56))
    return f"Volume → {level}%."


def music_now_playing() -> str:
    """Call this whenever the user asks what is playing, what song is this, what music is on, or anything about the current track. Do NOT guess — always call this tool first."""
    if not _alive(): return "Nothing is playing."
    s = _api() # gets current player status and track metadata
    meta = s.get("information", {}).get("category", {}).get("meta", {})
    t, pos, length, vol = s.get("time", 0), s.get("time", 0), s.get("length", 0), int(s.get("volume", 0) / 2.56)
    fmt = lambda s: f"{s//60}:{s%60:02d}"
    return "\n".join(filter(None, [
        f"state: {s.get('state')}",
        f"track: {meta.get('title') or meta.get('filename', '?')}",
        f"artist: {meta['artist']}" if meta.get("artist") else "",
        f"album: {meta['album']}" if meta.get("album") else "",
        f"position: {fmt(pos)}/{fmt(length)}  volume: {vol}%",
    ]))


def music_search_library(query: str, filter_by: str = "all") -> str:
    """Find what's in the music library. Use this before playing so you know exact names.

    Args:
        query: Name to search for. Use a specific word like "taylor" or "rock". Do NOT pass "all", "everything", or similar — pass an empty string "" to list the full library.
        filter_by: Narrow results — 'artist', 'album', 'song', or 'all' (default).
    """
    if not _dir(): return "Set MUSIC_DIR in config.local.json first."
    _ensure_db()
    query = _clean(query)
    filter_by = _clean(filter_by)
    if query.lower() in _GENERIC: query = ""
    conn = _db()
    where, params = _word_match_clause(query) if query else ("1=1", [])
    match filter_by:
        case "artist":
            rows = conn.execute(f"SELECT DISTINCT artist FROM tracks WHERE artist != '' AND {where}", params).fetchall()
            hits = sorted(r["artist"] for r in rows)
        case "album":
            rows = conn.execute(f"SELECT DISTINCT artist, album FROM tracks WHERE {where}", params).fetchall()
            hits = sorted(f"{r['artist']} — {r['album']}" for r in rows)
        case "song":
            rows = conn.execute(f"SELECT DISTINCT title FROM tracks WHERE {where}", params).fetchall()
            all_hits = sorted(r["title"] for r in rows)
            hits = random.sample(all_hits, min(10, len(all_hits)))
        case _:
            artists = sorted(r["artist"] for r in conn.execute(
                f"SELECT DISTINCT artist FROM tracks WHERE artist != '' AND {where}", params).fetchall())
            albums = sorted(f"{r['artist']} — {r['album']}" for r in conn.execute(
                f"SELECT DISTINCT artist, album FROM tracks WHERE {where}", params).fetchall())
            songs = [r["title"] for r in conn.execute(
                f"SELECT DISTINCT title FROM tracks WHERE {where}", params).fetchall()]
            song_sample = random.sample(sorted(songs), min(10, len(songs))) if songs else []
            conn.close()
            return "\n".join(filter(None, [
                ("Artists: " + ", ".join(artists))       if artists     else "",
                ("Albums: "  + ", ".join(albums))        if albums      else "",
                ("Some Random Songs: " + ", ".join(song_sample)) if song_sample else "",
            ])) or f"Nothing found for '{query}'."
    conn.close()
    return "\n".join(hits) if hits else f"No results for '{query}'."


def music_rebuild_index() -> str:
    """Rebuild the music library index. Call this after adding or removing music files.
    """
    if not _dir(): return "Set MUSIC_DIR in config.local.json first."
    count = _build_index()
    return f"Index rebuilt: {count} tracks."
