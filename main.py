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
import ollama
import numpy as np
import json
import os
import re
import threading
import subprocess
from datetime import datetime

from prompts.mk1 import SYSTEM_PROMPT
from tools import TOOLS, dispatch
from log import log, divider

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
_music_stop = threading.Event()
_music_thread = None

_exit_requested = threading.Event()

def _watch_stdin():
    """Raise KeyboardInterrupt in main thread when Ctrl+D (EOF) is pressed."""
    while sys.stdin.read(1) != "":
        pass
    _exit_requested.set()
    _thread.interrupt_main()


def _music_loop():
    while not _music_stop.is_set():
        proc = subprocess.Popen(["afplay", TYPING_MUSIC_FILE])
        while proc.poll() is None:
            if _music_stop.wait(timeout=0):
                proc.terminate()
                return


def start_music():
    global _music_thread
    _music_stop.clear()
    _music_thread = threading.Thread(target=_music_loop, daemon=True)
    _music_thread.start()


def stop_music():
    _music_stop.set()
    if _music_thread:
        _music_thread.join()


def wait_for_wake_word():
    chunk_size = int(SAMPLE_RATE * WAKE_STRIDE)
    window_chunks = int(WAKE_WINDOW / WAKE_STRIDE)
    buffer = []

    log("wake", f'listening for "{WAKE_PHRASE}"...')
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_size) as stream:
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
                subprocess.run(["afplay", "sounds/wake-up.mp3"])
                return


def record_until_silence():
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(MAX_DURATION / CHUNK_DURATION)
    pre_speech_chunks = int(PRE_SPEECH_TIMEOUT / CHUNK_DURATION)

    frames = []
    silent_count = 0
    has_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_size) as stream:
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

    return np.concatenate(frames).flatten()


def transcribe(audio) -> str:
    segments, _ = whisper_model.transcribe(audio, language="en")
    return "".join(s.text for s in segments).strip()


def think(history: list) -> str:
    response = ollama.chat(model="gemma4:e2b", messages=history, tools=TOOLS)
    message = response["message"]

    if message.get("tool_calls"):
        history.append(message)
        for tool_call in message["tool_calls"]:
            name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            log("tool", f"{name}  {args}")
            tool_result = dispatch(name, args)
            log("result", tool_result)
            history.append({"role": "tool", "content": tool_result})
            if tool_result in ("shutting_down", "resetting"):
                return tool_result
        response = ollama.chat(model="gemma4:e2b", messages=history)
        message = response["message"]

    return message["content"]


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


def listen() -> str | None:
    log("listen", "waiting for speech...")
    audio = record_until_silence()
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


def respond(history: list) -> str | None:
    start_music()
    try:
        return think(history)
    except KeyboardInterrupt:
        if _exit_requested.is_set():
            dump_history(history)
            log("interrupt", "exiting...")
            divider()
            raise SystemExit(0)
        log("interrupt", "aborted, re-prompting...")
        return None
    finally:
        stop_music()


def main():
    threading.Thread(target=_watch_stdin, daemon=True).start()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        divider()

        try:
            wait_for_wake_word()
            text = listen()
        except KeyboardInterrupt:
            log("interrupt", "listening interrupted, re-prompting...")
            continue
        if not text:
            continue

        history.append({"role": "user", "content": text})
        if (reply := respond(history)) is None:
            history.pop()
            continue

        match reply:
            case "resetting":
                dump_history(history)
                log("reset", "clearing conversation history...")
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                speak("Sure, starting fresh.")

            case "shutting_down":
                dump_history(history)
                log("speak", "shutting down...")
                divider()
                subprocess.run(["afplay", "sounds/shutdown.mp3"])
                break

            case _:
                log("reply", f'"{reply}"')
                history.append({"role": "assistant", "content": reply})
                try:
                    speak(clean(reply))
                except KeyboardInterrupt:
                    log("interrupt", "speech interrupted, re-prompting...")
                    continue
                log("speak", f"playing response...")


if __name__ == "__main__":
    main()
