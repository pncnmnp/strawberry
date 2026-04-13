import subprocess, random, urllib.request, urllib.parse, base64, json, time, tempfile, os, re
from pathlib import Path
from log import log

VLC = "/Applications/VLC.app/Contents/MacOS/VLC"
HOST, PORT, PASS = "127.0.0.1", 8765, "strawberry"
EXTS = {".mp3", ".flac", ".m4a", ".aac", ".wav", ".ogg", ".opus"}

_proc = None
_m3u = None

def _dir() -> str | None:
    p = Path(__file__).parent.parent / "config.local.py"
    if not p.exists(): return None
    ns: dict = {}
    exec(p.read_text(), ns)
    return ns.get("MUSIC_DIR")

def _scan():
    d = _dir()
    return sorted(f for ext in EXTS for f in Path(d).rglob(f"*{ext}")) if d else []

def _index():
    return [{"path": f, "title": f.stem, "album": f.parent.name, "artist": f.parent.parent.name} for f in _scan()]

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

def _clean(s: str) -> str:
    """Strip model special tokens (e.g. Gemma's <|"|>) and whitespace."""
    return re.sub(r"<\|[^|]*\|>", "", s).strip()

def _alive() -> bool:
    return bool(_proc and _proc.poll() is None and "error" not in _api())

def _kill_stale():
    """Kill any lingering VLC processes from previous sessions using our port.
    This is IMPORTANT, else, we are not able to reliably perform actions.
    """
    subprocess.run(["pkill", "-f", f"{VLC}.*--http-port={PORT}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    if not _dir(): return "Set MUSIC_DIR in config.local.py first."
    query = _clean(query)
    if query.lower() in _GENERIC: query = ""
    q = query.lower()
    files = [e["path"] for e in _index() if not q or q in e["title"].lower() or q in e["album"].lower() or q in e["artist"].lower()]
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
    if not _dir(): return "Set MUSIC_DIR in config.local.py first."
    query = _clean(query)
    filter_by = _clean(filter_by)
    if query.lower() in _GENERIC: query = ""
    q, idx = query.lower(), _index()
    match filter_by:
        case "artist": hits = sorted({e["artist"] for e in idx if q in e["artist"].lower()} - {""})
        case "album":  hits = sorted({f"{e['artist']} — {e['album']}" for e in idx if q in e["album"].lower()})
        case "song":
            all_hits = sorted({e["title"] for e in idx if q in e["title"].lower()})
            hits = random.sample(all_hits, min(10, len(all_hits)))
        case _:
            artists = sorted({e["artist"] for e in idx if q in e["artist"].lower()} - {""})
            albums  = sorted({f"{e['artist']} — {e['album']}" for e in idx if q in e["album"].lower()})
            songs   = {e["title"] for e in idx if q in e["title"].lower()}
            song_sample = random.sample(sorted(songs), min(10, len(songs)))
            return "\n".join(filter(None, [
                ("Artists: " + ", ".join(artists))          if artists     else "",
                ("Albums: "  + ", ".join(albums))           if albums      else "",
                ("Some Random Songs: "   + ", ".join(song_sample))      if song_sample else "",
            ])) or f"Nothing found for '{query}'."
    return "Artists, albums, and some random songs:\n" + "\n".join(hits) if hits else f"No results for '{query}'."
