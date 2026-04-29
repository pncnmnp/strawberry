import asyncio
import http.server
import os
import queue
import socketserver
import threading

import numpy as np
import websockets

from log import log

WS_PORT   = 8768
HTTP_PORT = 8769
_DIR = os.path.dirname(os.path.abspath(__file__))

_mic_queue: queue.Queue = queue.Queue()
_connected = threading.Event()
_replay: list = []  # frames prepended for listen() after a barge-in


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_DIR, **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def _http_thread():
    with socketserver.TCPServer(("localhost", HTTP_PORT), _SilentHandler) as httpd:
        httpd.serve_forever()


async def _ws_handler(websocket):
    _connected.set()
    log("ws", "browser connected")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                _mic_queue.put(np.frombuffer(message, dtype=np.float32).copy())
    finally:
        _mic_queue.put(None)  # unblock any waiting read
        _connected.clear()
        log("ws", "browser disconnected")


def start():
    threading.Thread(target=_http_thread, daemon=True).start()

    async def _serve():
        async with websockets.serve(_ws_handler, "localhost", WS_PORT):
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True).start()


def wait_connected(timeout: float = 30.0) -> bool:
    return _connected.wait(timeout=timeout)


def replay(frames: list):
    """Prepend captured frames so the next listen() sees them first.

    Example: user says "stop what you're" while the assistant is talking. 
    The barge-in monitor captured those frames in barge_in_buf. 
    We call replay(barge_in_buf) so the next listen() reads "stop what you're" 
    before pulling any new live mic audio — otherwise those
    frames are lost and the user has to repeat themselves.
    """
    global _replay
    _replay = list(frames)


def has_replay() -> bool:
    return bool(_replay)


def drain():
    """Drop stale frames accumulated during TTS (call before listen() when no replay pending)."""
    while True:
        try:
            _mic_queue.get_nowait()
        except queue.Empty:
            break


def get_frame(timeout: float | None = None) -> np.ndarray | None:
    """Return the next mic frame. Drains replay buffer first. Returns None on timeout or disconnect."""
    global _replay
    if _replay:
        return _replay.pop(0)
    try:
        return _mic_queue.get() if timeout is None else _mic_queue.get(timeout=timeout)
    except queue.Empty:
        return None
