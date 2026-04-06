import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from kokoro import KPipeline
import sounddevice as sd
import whisper
import ollama
import numpy as np
import json
import os
import re
from datetime import datetime

from prompts.mk1 import SYSTEM_PROMPT
from tools import TOOLS, dispatch
from log import log, divider

SAMPLE_RATE = 16000        # samples per second
CHUNK_DURATION = 0.1       # seconds per chunk
SILENCE_THRESHOLD = 0.01   # RMS amplitude below this = silence
SILENCE_DURATION = 1.5     # seconds of silence to stop recording
MAX_DURATION = 30          # safety cap in seconds

whisper_model = whisper.load_model("base")
tts = KPipeline(lang_code='a')


def record_until_silence():
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(MAX_DURATION / CHUNK_DURATION)

    frames = []
    silent_count = 0
    has_speech = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
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

    return np.concatenate(frames).flatten()


def transcribe(audio) -> str:
    transcription = whisper_model.transcribe(audio)  # type: ignore[union-attr]
    return transcription["text"].strip()  # type: ignore


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
    text = re.sub(r'\n+', ' ', text)               # newlines → space
    text = re.sub(r'\s{2,}', ' ', text)            # collapse spaces
    return text.strip()


def speak(text: str):
    for _, (gs, ps, audio) in enumerate(tts(text, voice="af_heart")):
        sd.play(audio, samplerate=24000)
        sd.wait()


def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        divider()

        if not (text := listen()):
            continue

        history.append({"role": "user", "content": text})
        reply = think(history)

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
                break

            case _:
                log("reply", f'"{reply}"')
                history.append({"role": "assistant", "content": reply})
                log("speak", "playing response...")
                speak(clean(reply))


if __name__ == "__main__":
    main()
