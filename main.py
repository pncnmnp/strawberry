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
import queue
import threading
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize

from prompts.mk1 import SYSTEM_PROMPT, PRIOR_CONVERSATION
from tools import TOOL_FUNCTIONS
from tools.music import _alive as _music_alive, _kill_stale as _music_kill_stale
from tools.code import start as _sandbox_start, stop as _sandbox_stop
from pixies import compact_history
from lm import get_engine, cleanup as lm_cleanup, _suppress_stderr, _restore_stderr
from log import log, log_context, context_pct, divider, compute_tool_schema_tokens
from vad import is_turn_complete
import audio_ws
import memory

# For sentence tokenization
nltk.download("punkt_tab", quiet=True)

SAMPLE_RATE = 16000        # samples per second
CHUNK_DURATION = 0.1       # seconds per chunk
SILENCE_THRESHOLD = 0.01   # RMS amplitude below this = silence
SILENCE_DURATION = 3       # seconds of silence — hard cutoff if smart-turn keeps saying incomplete
PAUSE_CHECK_DURATION = 0.75  # seconds of silence after speech before consulting smart-turn
MAX_DURATION = 30          # safety cap in seconds
PRE_SPEECH_TIMEOUT = 10    # seconds to wait before any speech is detected
COMPACT_THRESHOLD = 0.66   # context fraction that triggers auto-compaction
BARGE_IN_VERIFY_FRAMES = 8 # accumulate ~512 ms of above-threshold audio before whisper-verifying it's actually speech

# NOTE: small.en is about 3x slower (~1.2 sec) than base.en (~400 ms).
# However, we have made a lot of optimizations elsewhere, to enable this faster model.
# Frankly, the quality of base.en sucks. small.en is a tad bit better.
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
wake_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

WAKE_PHRASE = "hey strawberry"
WAKE_WINDOW = 2.0          # seconds of audio to check for wake word
WAKE_STRIDE = 0.5          # seconds between checks
tts = KPipeline(lang_code='a')

TYPING_MUSIC_FILE = os.path.join(os.path.dirname(__file__), "sounds", "typing.mp3")

SIGNALS = ("shutting_down", "resetting", "muting", "powercycling")


@dataclass
class AppState:
    signal: str | None = None
    conversation: Any = None
    music_thread: threading.Thread | None = None
    music_stop: threading.Event = field(default_factory=threading.Event)
    exit_requested: threading.Event = field(default_factory=threading.Event)
    tool_chars: int = 0  # cumulative chars from tool calls + results (not in history)
    last_thought: str | None = None
    partial_reply: str | None = None  # set when barge-in cuts off a response mid-sentence


state = AppState()


class _WSAudioStream:
    """
    Drop-in replacement for sd.InputStream backed by the WebSocket audio bridge.

    Why 'drop-in'? The initial code used sd.InputStream, however, with AEC + Barge-in support,
    the architecture evolved. This keeps the same interface for now to minimize changes.
    TODO: Refactoring - not a high priority right now.
    """

    def start(self): pass
    def stop(self):  pass
    def close(self): pass

    def __init__(self):
        self._buf = np.array([], dtype=np.float32)

    @property
    def read_available(self) -> int:
        return len(self._buf)

    def read(self, frames: int):
        while len(self._buf) < frames:
            chunk = audio_ws.get_frame()
            if chunk is None:
                raise RuntimeError("audio bridge disconnected")
            self._buf = np.concatenate([self._buf, chunk.flatten()])
        result = self._buf[:frames].reshape(-1, 1)
        self._buf = self._buf[frames:]
        # why false? mirrors overflow behavior of sd.InputStream
        return result, False


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
    stream = _WSAudioStream()
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
            # drain everything and keep stream open so listen() can use it immediately
            # NOTE: Without draining, the remnants of the audio that triggered 
            # the wake word gets leaked in the buffer and causes misinterpretation 
            # in the speech, leading to a false responses from the model.
            if stream.read_available > 0:
                stream.read(stream.read_available)
            return stream


def record_until_silence(stream=None, max_duration=MAX_DURATION):
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    # How many chunks of audio before we consult smart-turn?
    pause_check_chunks = int(PAUSE_CHECK_DURATION / CHUNK_DURATION)
    max_chunks = int(max_duration / CHUNK_DURATION)
    pre_speech_chunks = int(PRE_SPEECH_TIMEOUT / CHUNK_DURATION)

    # NOTE: Smart-turn votes once at PAUSE_CHECK_DURATION. If it says complete, we don't cut
    # immediately — we wait one more second_check_chunks window and vote again. If the
    # user resumes speaking in that gap, pending_complete resets and we carry on. Only
    # two consecutive "complete" votes (with no speech in between) actually end the turn.
    # This prevents filler phrases like "Well, let's see." from getting submitted mid-thought.
    # Why 0.5 seconds for the second check? Trial and error for now.
    second_check_chunks = int(0.5 / CHUNK_DURATION)

    frames = []
    silent_count = 0
    has_speech = False
    speech_start_idx: int | None = None
    next_vad_check = pause_check_chunks
    pending_complete = False

    owns_stream = stream is None
    if owns_stream:
        stream = _WSAudioStream()
        stream.start()
    else:
        # The wake phrase is still trickling through the audio pipeline even after
        # the buffer drain in wait_for_wake_word. Discard ~400ms to let it clear.
        for _ in range(int(0.4 / CHUNK_DURATION)):
            stream.read(chunk_size)

    try:
        while len(frames) < max_chunks:
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())
            rms = np.sqrt(np.mean(chunk ** 2))
            if rms > SILENCE_THRESHOLD:
                if not has_speech:
                    speech_start_idx = len(frames) - 1
                has_speech = True
                silent_count = 0
                next_vad_check = pause_check_chunks
                pending_complete = False
            elif has_speech:
                silent_count += 1
                # Do not fire until enough silence chunks have accumulated
                # Why do this? To give the user X seconds (0.5) to resume speaking
                # before the second vote cuts them off.
                if next_vad_check > 0 and silent_count >= next_vad_check:
                    next_vad_check = 0
                    turn_audio = np.concatenate(frames[speech_start_idx:]).flatten()
                    complete, prob = is_turn_complete(turn_audio)
                    log("vad", f"prob={prob:.2f} complete={complete}")
                    if complete:
                        if pending_complete:
                            break
                        pending_complete = True
                        # For the second vote
                        next_vad_check = silent_count + second_check_chunks
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
    return "".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()


def _wrap_tool(fn):
    """Wrap a tool function to log calls and capture signal returns."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs = {k: v.replace('<|"|>', "").replace("<|'|>", "").strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        call_str = f"{fn.__name__} {json.dumps(kwargs or list(args))}"
        log("tool", f"{fn.__name__}  {kwargs or args}")
        result = fn(*args, **kwargs)
        if isinstance(result, dict) and "thought" in result:
            state.last_thought = result["thought"]
            result = result["response"]
        log("result", str(result)[:1000])
        if result in SIGNALS:
            state.signal = result
        state.tool_chars += len(call_str) + len(str(result))
        return result
    return wrapper


def _make_conversation(system: str = SYSTEM_PROMPT):
    saved = _suppress_stderr()
    try:
        state.conversation = get_engine().create_conversation(
            messages=[{"role": "system", "content": [{"type": "text", "text": system}]}],
            tools=[_wrap_tool(fn) for fn in TOOL_FUNCTIONS],
        )
    finally:
        _restore_stderr(saved)


def _reset_history(system: str = SYSTEM_PROMPT) -> list:
    state.tool_chars = 0
    state.conversation = None
    _make_conversation(system)
    return [{"role": "system", "content": system}]


def _compact(history: list) -> list:
    log("compact", f"above threshold of {int(COMPACT_THRESHOLD * 100)} — auto-compacting...")
    speak("We're past the context threshold, Sir. Running auto-compaction — one moment.")
    memory.flush()
    start_music()
    try:
        summary = compact_history(history)
    finally:
        stop_music()
    dump_history(history)
    log("compact", summary.replace("\n", " / "))
    return _reset_history(SYSTEM_PROMPT + PRIOR_CONVERSATION.format(summary=summary))



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
    # Drop frames that accumulated in the queue during respond(). 
    # Skip when a barge-in replay is pending — those frames are the user's actual speech.
    if stream is None and not audio_ws.has_replay():
        time.sleep(0.15)  # let in-flight WebSocket frames from TTS land before draining
        audio_ws.drain()
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


def _split_sentence(buf: str) -> tuple[str, str]:
    """Split off the first complete sentence using Punkt. Returns (sentence, remainder)."""
    sentences = sent_tokenize(buf)
    if len(sentences) < 2:
        return "", buf
    return sentences[0], buf[len(sentences[0]):].lstrip()


def clean(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'#+\s*', '', text)              # ## headings
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
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


def _barge_in_monitor(stop_event: threading.Event, barge_in: threading.Event, buf: list, tts_active: threading.Event):
    """Drain mic queue during TTS; set barge_in when whisper-verified speech is detected.

    IMPORTANT DESIGN: 
    Above-threshold frames accumulate into buf; once ~512 ms have collected, 
    run tiny whisper to verify it's actual speech rather than typing/sneezing/coughing/etc. 
    RMS alone fires on any energy spike, which kept yanking TTS out for non-speech sounds.

    buf is populated only from speech onset + a small pre-roll, so replay never
    contains the long silence window captured during TTS before the user spoke.

    Speech tracking is gated on tts_active — during the LLM generation phase there
    is nothing to barge into, and ambient noise over 3-10 seconds easily produces
    sustained above-threshold frames. We still consume frames here (so the queue
    doesn't back up) — we just don't let them count toward barge-in.
    """
    speech_count = 0
    pre_buf: list = []   # rolling pre-speech context (a few frames)
    PRE_ROLL = 3         # frames to keep before speech onset (~192 ms)

    while not stop_event.is_set() and not barge_in.is_set():
        frame = audio_ws.get_frame(timeout=0.1)
        if frame is None:
            continue

        if not tts_active.is_set():
            speech_count = 0
            pre_buf.clear()
            buf.clear()
            continue

        rms = np.sqrt(np.mean(frame ** 2))
        if rms > SILENCE_THRESHOLD:
            if speech_count == 0:
                buf.clear()
                buf.extend(pre_buf)
            speech_count += 1
            buf.append(frame)
            if speech_count >= BARGE_IN_VERIFY_FRAMES:
                audio = np.concatenate(buf).flatten()
                segments, _ = wake_model.transcribe(
                    audio, language="en", beam_size=1, without_timestamps=True,
                )
                text = "".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()
                if text:
                    log("barge", f'"{text}"')
                    barge_in.set()
                else:
                    log("barge", "ignored (no speech)")
                    speech_count = 0
                    buf.clear()
                    pre_buf.clear()
        else:
            speech_count = max(0, speech_count - 1)
            if speech_count == 0:
                buf.clear()
                pre_buf.append(frame)
                if len(pre_buf) > PRE_ROLL:
                    pre_buf.pop(0)
            else:
                # silence within an active speech segment
                # this would be quiet frames mid-speech, like the breath, the pause between words, etc.
                # preserves the natural rhythm of what we said, so that STT gets the best possible input for barge-in verification
                buf.append(frame)


def respond(text: str, history: list) -> str | None:
    state.signal = None
    state.partial_reply = None
    assert state.conversation is not None

    sentence_queue: queue.Queue = queue.Queue()
    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    full_text = "" # what it said in total, for logging
    exc: Exception | None = None
    _ERROR = object()

    def _produce():
        nonlocal full_text, exc
        buf = ""
        saved = _suppress_stderr()
        try:
            for chunk in state.conversation.send_message_async(text):
                if stop_event.is_set():
                    break
                content = chunk.get("content", [])
                token = ""
                if isinstance(content, list):
                    token = "".join(c.get("text", "") for c in content if c.get("type") == "text")
                elif isinstance(content, str):
                    token = content
                if not token:
                    continue
                buf += token
                full_text += token
                sentence, buf = _split_sentence(buf)
                if sentence:
                    sentence_queue.put(sentence)
            if buf.strip() and not stop_event.is_set():
                sentence_queue.put(buf.strip())
        except Exception as e:
            exc = e
            sentence_queue.put(_ERROR)
        finally:
            _restore_stderr(saved)
            sentence_queue.put(None)

    def _synthesize():
        try:
            while True:
                sentence = sentence_queue.get()
                if sentence is None:
                    break
                if sentence is _ERROR:
                    audio_queue.put(_ERROR)
                    return
                for _, _, audio in tts(clean(sentence), voice="af_heart"):
                    if stop_event.is_set():
                        break
                    audio_queue.put(audio)
        finally:
            audio_queue.put(None)

    barge_in = threading.Event()
    tts_active = threading.Event()
    barge_in_buf: list = []
    barge_in_thread = threading.Thread(
        target=_barge_in_monitor, args=(stop_event, barge_in, barge_in_buf, tts_active), daemon=True
    )

    producer = threading.Thread(target=_produce, daemon=True)
    synthesizer = threading.Thread(target=_synthesize, daemon=True)
    start_music()
    producer.start()
    synthesizer.start()
    barge_in_thread.start()

    try:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.05)
            except queue.Empty:
                if barge_in.is_set():
                    sd.stop()
                    stop_event.set()
                    state.partial_reply = full_text or None
                    # prepend captured speech so listen() doesn't miss what the user said
                    audio_ws.replay(barge_in_buf)
                    return None
                continue
            if chunk is None:
                break
            if chunk is _ERROR:
                raise exc or RuntimeError("unknown producer error")
            stop_music()
            tts_active.set()
            sd.play(chunk, samplerate=24000)
            while sd.get_stream().active:
                time.sleep(0.05)
                if barge_in.is_set():
                    sd.stop()
                    stop_event.set()
                    state.partial_reply = full_text or None
                    audio_ws.replay(barge_in_buf)
                    return None

        if state.signal:
            return state.signal

        return full_text or None

    except KeyboardInterrupt:
        stop_event.set()
        try:
            state.conversation.cancel_process()
        except Exception:
            pass
        sd.stop()
        producer.join()
        synthesizer.join()

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
        stop_event.set()
        barge_in_thread.join(timeout=0.5)
        if producer.is_alive():
            producer.join()
        if synthesizer.is_alive():
            synthesizer.join()


def _warmup():
    """Force system-prompt + tool-schema prefill so first reply is fast."""
    log("warmup", "warming up llm...")
    saved = _suppress_stderr()
    try:
        state.conversation.send_message("hi")
    finally:
        _restore_stderr(saved)
    log("warmup", "llm warmup done")


def _ensure_syncthing():
    try:
        subprocess.run(["pgrep", "-x", "syncthing"], check=True, capture_output=True)
        log("syncthing", "already running")
    except subprocess.CalledProcessError:
        # No syncthing process found, start it
        subprocess.Popen(["syncthing", "--no-browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("syncthing", "started")


def _commit_turn(history: list, user_text: str, assistant_text: str):
    history.append({"role": "user", "content": user_text})
    if state.last_thought:
        history.append({"role": "assistant", "content": f"[Thought]\n{state.last_thought}"})
        state.last_thought = None
    history.append({"role": "assistant", "content": assistant_text})
    log_context(history, tool_chars=state.tool_chars)
    memory.queue_turn(user_text, assistant_text)


def main():
    threading.Thread(target=_watch_stdin, daemon=True).start()
    _ensure_syncthing()
    audio_ws.start()
    subprocess.Popen(["open", f"http://localhost:{audio_ws.HTTP_PORT}/audio_bridge.html"])
    compute_tool_schema_tokens(TOOL_FUNCTIONS)
    _make_conversation()
    sandbox_thread = threading.Thread(target=_sandbox_start, daemon=True)
    sandbox_thread.start()
    memory_thread = threading.Thread(target=memory.load, daemon=True)
    memory_thread.start()
    _warmup()
    sandbox_thread.join()
    memory_thread.join()
    if not audio_ws.wait_connected(timeout=5):
        log("ws", f"waiting — open http://localhost:{audio_ws.HTTP_PORT}/audio_bridge.html and click Connect")
        audio_ws.wait_connected(timeout=120)
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

        # FYI: we are prepending memory block on top of user input on every turn.
        augmented = (prefix + text) if (prefix := memory.retrieve(text)) else text

        if context_pct(history, state.tool_chars) >= COMPACT_THRESHOLD:
            history = _compact(history)

        if (reply := respond(augmented, history)) is None:
            if partial := state.partial_reply:
                _commit_turn(history, text, partial)
                state.partial_reply = None
            continue

        match reply:
            case "resetting":
                dump_history(history)
                log("reset", "clearing conversation history...")
                history = _reset_history()
                speak("Sure, starting fresh.")

            case "shutting_down":
                memory.flush()
                dump_history(history)
                log("speak", "shutting down...")
                divider()
                _music_kill_stale()
                state.conversation = None
                lm_cleanup()
                subprocess.run(["afplay", "sounds/shutdown.mp3"])
                break

            case "powercycling":
                memory.flush()
                dump_history(history)
                log("speak", "powercycling...")
                divider()
                _music_kill_stale()
                _sandbox_stop()
                state.conversation = None
                lm_cleanup()
                subprocess.run(["afplay", "sounds/shutdown.mp3"])
                os.execv(sys.executable, [sys.executable] + sys.argv)

            case "muting":
                # No memory flush on mute - we want to have short wake-up latency
                log("wake", "muting — wake word required next time")
                awake = False

            case _:
                log("reply", f'"{reply}"')
                _commit_turn(history, text, reply)

if __name__ == "__main__":
    main()
