import os
import warnings
import logging
import _thread
import sys
import time
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from kokoro import KPipeline
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import json
import re
import threading
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any

from prompts.mk1 import SYSTEM_PROMPT
from tools import TOOL_FUNCTIONS
from tools.music import _alive as _music_alive, _kill_stale as _music_kill_stale
from lm import get_engine, cleanup as lm_cleanup, _suppress_stderr, _restore_stderr
from log import log, log_context, divider

SAMPLE_RATE = 16000        # samples per second
CHUNK_DURATION = 0.1       # seconds per chunk
SILENCE_THRESHOLD = 0.01   # RMS amplitude below this = silence
SILENCE_DURATION = 1.5     # seconds of silence to stop recording
MAX_DURATION = 30          # safety cap in seconds
PRE_SPEECH_TIMEOUT = 10    # seconds to wait before any speech is detected

whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
wake_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

WAKE_PHRASE = "hey strawberry"
WAKE_WINDOW = 2.0          # seconds of audio to check for wake word
WAKE_STRIDE = 0.5          # seconds between checks
tts = KPipeline(lang_code='a')

TYPING_MUSIC_FILE = os.path.join(os.path.dirname(__file__), "sounds", "typing.mp3")

SIGNALS = ("shutting_down", "resetting", "muting")


@dataclass
class AppState:
    signal: str | None = None
    conversation: Any = None
    music_thread: threading.Thread | None = None
    music_stop: threading.Event = field(default_factory=threading.Event)
    exit_requested: threading.Event = field(default_factory=threading.Event)
    tool_chars: int = 0  # cumulative chars from tool calls + results (not in history)


state = AppState()

def _watch_stdin():
    """Raise KeyboardInterrupt in main thread when Ctrl+D (EOF) is pressed."""
    while sys.stdin.read(1) != "":
        pass
    state.exit_requested.set()
    _thread.interrupt_main()


def _music_loop():
    while not state.music_stop.is_set():
        proc = subprocess.Popen(["afplay", TYPING_MUSIC_FILE])
        while proc.poll() is None:
            if state.music_stop.wait(timeout=0):
                proc.terminate()
                return


def start_music():
    state.music_stop.clear()
    state.music_thread = threading.Thread(target=_music_loop, daemon=True)
    state.music_thread.start()


def stop_music():
    state.music_stop.set()
    if state.music_thread:
        state.music_thread.join()


def wait_for_wake_word():
    chunk_size = int(SAMPLE_RATE * WAKE_STRIDE)
    window_chunks = int(WAKE_WINDOW / WAKE_STRIDE)
    buffer = []

    log("wake", f'listening for "{WAKE_PHRASE}"...')
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_size)
    stream.start()
    while True:
        chunk, _ = stream.read(chunk_size)
        buffer.append(chunk.copy())
        if len(buffer) > window_chunks:
            buffer.pop(0)

        # skip whisper if audio is silent
        window = np.concatenate(buffer).flatten()
        rms = np.sqrt(np.mean(window ** 2))
        if rms < SILENCE_THRESHOLD:
            continue

        segments, _ = wake_model.transcribe(
            window,
            language="en",
            beam_size=1,
            initial_prompt=WAKE_PHRASE,
            without_timestamps=True,
        )
        text = "".join(s.text for s in segments).lower().strip()
        if WAKE_PHRASE in text:
            log("wake", "detected!")
            subprocess.Popen(["afplay", "sounds/wake-up.mp3"])
            return stream  # keep stream open so listen() can use it immediately


def record_until_silence(stream=None, max_duration=MAX_DURATION):
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(max_duration / CHUNK_DURATION)
    pre_speech_chunks = int(PRE_SPEECH_TIMEOUT / CHUNK_DURATION)

    frames = []
    silent_count = 0
    has_speech = False

    owns_stream = stream is None
    if owns_stream:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=chunk_size)
        stream.start()

    try:
        while len(frames) < max_chunks:
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms > SILENCE_THRESHOLD:
                has_speech = True
                silent_count = 0
            elif has_speech:
                silent_count += 1
                if silent_count >= silence_chunks_needed:
                    break
            elif len(frames) >= pre_speech_chunks:
                break
    finally:
        if owns_stream:
            stream.stop()
            stream.close()

    return np.concatenate(frames).flatten()


def transcribe(audio) -> str:
    segments, _ = whisper_model.transcribe(audio, language="en")
    return "".join(s.text for s in segments).strip()


def _wrap_tool(fn):
    """Wrap a tool function to log calls and capture signal returns."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        call_str = f"{fn.__name__} {json.dumps(kwargs or list(args))}"
        log("tool", f"{fn.__name__}  {kwargs or args}")
        result = fn(*args, **kwargs)
        log("result", str(result)[:1000])
        if result in SIGNALS:
            state.signal = result
        state.tool_chars += len(call_str) + len(str(result))
        return result
    return wrapper


def _make_conversation():
    saved = _suppress_stderr()
    try:
        state.conversation = get_engine().create_conversation(
            messages=[{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}],
            tools=[_wrap_tool(fn) for fn in TOOL_FUNCTIONS],
        )
    finally:
        _restore_stderr(saved)


def think(text: str) -> str:
    state.signal = None
    assert state.conversation is not None
    saved = _suppress_stderr()
    try:
        response = state.conversation.send_message(text)
    finally:
        _restore_stderr(saved)
    if state.signal:
        return state.signal
    content = response.get("content", [])
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if c.get("type") == "text")
    return str(content)


def dump_history(history: list):
    os.makedirs("history", exist_ok=True)
    filename = f"history/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    def _serialize(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with open(filename, "w") as f:
        json.dump(history, f, indent=2, default=_serialize)
    log("dump", filename)


def listen(stream=None) -> str | None:
    # HACK: music bleeds into the mic, causing whisper to keep "hearing" audio and
    # never hit the silence threshold — so we clamp the window hard when VLC is running.
    max_dur = 7 if _music_alive() else MAX_DURATION
    log("listen", "waiting for speech...")
    audio = record_until_silence(stream, max_duration=max_dur)
    if stream:
        stream.stop()
        stream.close()
    log("listen", "recorded...")
    text = transcribe(audio.flatten())
    if not text:
        log("warn", "no speech detected")
        return None
    log("input", f'"{text}"')
    return text


def clean(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'#+\s*', '', text)              # ## headings
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'\n+', ' ', text)               # newlines → space
    text = re.sub(r'\s{2,}', ' ', text)            # collapse spaces
    return text.strip()


def speak(text: str):
    for _, (gs, ps, audio) in enumerate(tts(text, voice="af_heart")):
        sd.play(audio, samplerate=24000)
        try:
            while sd.get_stream().active:
                time.sleep(0.05)
        except KeyboardInterrupt:
            sd.stop()
            raise


def respond(text: str, history: list) -> str | None:
    start_music()
    try:
        return think(text)
    except KeyboardInterrupt:
        if state.exit_requested.is_set():
            dump_history(history)
            log("interrupt", "exiting...")
            divider()
            state.conversation = None
            lm_cleanup()
            raise SystemExit(0)
        log("interrupt", "aborted, re-prompting...")
        return None
    finally:
        stop_music()


def main():
    threading.Thread(target=_watch_stdin, daemon=True).start()
    _make_conversation()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    awake = False

    while True:
        divider()

        try:
            stream = None
            if not awake:
                stream = wait_for_wake_word()
                awake = True
            text = listen(stream)
        except KeyboardInterrupt:
            log("interrupt", "listening interrupted, re-prompting...")
            continue
        if not text:
            continue

        if (reply := respond(text, history)) is None:
            continue

        match reply:
            case "resetting":
                dump_history(history)
                log("reset", "clearing conversation history...")
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                state.tool_chars = 0
                state.conversation = None
                _make_conversation()
                speak("Sure, starting fresh.")

            case "shutting_down":
                dump_history(history)
                log("speak", "shutting down...")
                divider()
                _music_kill_stale()
                state.conversation = None
                lm_cleanup()
                subprocess.run(["afplay", "sounds/shutdown.mp3"])
                break

            case "muting":
                log("wake", "muting — wake word required next time")
                awake = False

            case _:
                log("reply", f'"{reply}"')
                history.append({"role": "user", "content": text})
                history.append({"role": "assistant", "content": reply})
                log_context(history, tool_chars=state.tool_chars)
                try:
                    speak(clean(reply))
                except KeyboardInterrupt:
                    log("interrupt", "speech interrupted, re-prompting...")
                    continue
                log("speak", f"playing response...")

if __name__ == "__main__":
    main()
