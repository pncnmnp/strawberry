# Strawberry

My local voice assistant. Listens via microphone, transcribes with Whisper, thinks with Gemma via LiteRT-LM, speaks with Kokoro TTS.

## What it can do
- Answer questions (Wikipedia or DuckDuckGo, auto-routed)
- Extended reasoning for math, logic, and multi-step problems
- Execute Python code in an isolated sandbox
- Play and control music (requires `MUSIC_DIR` in `config.local.json`)
- Check weather and current date/time
- Save and recall personal notes (Markdown supported) with keyword, tag, and time-range search
- Recall and complete todos from a synced `TODO.md`
- Persistent memory — extracts durable facts from conversations and surfaces them in future sessions
- Barge-in — interrupt Strawberry mid-response and it will hear you
- Mute until next wake word
- Reset conversation history + auto-compaction
- Shut down and powercycle on command

## Stack
- **STT**: Faster Whisper (`small.en` for transcription, `tiny.en` for wake word)
- **LLM**: Gemma 4 E4B via LiteRT-LM
- **TTS**: Kokoro (`af_heart` voice)
- **VAD**: [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) (ONNX) for turn completion
- **Memory**: nomic-embed-text-v1.5 + FAISS + SQLite
- **Search**: Wikipedia + DuckDuckGo
- **Notes**: Markdown files in an Obsidian vault, synced to mobile via Syncthing

## Setup

The quickest path is the setup script — it installs `uv`, syncs dependencies, downloads the VAD + Gemma models, and scaffolds `config.local.json`. It's idempotent, so re-running it only does outstanding work (and resumes interrupted downloads):

```bash
./setup.sh
```

You'll still need VLC and OrbStack (steps 1 and 5 below) for music playback and the code sandbox. To set things up by hand instead:

1. Install [VLC](https://www.videolan.org/vlc/) (required for music playback).

2. Download the Gemma model weights from [HuggingFace](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm) and note the path to the `.litertlm` file.

3. Download the smart-turn VAD model:
   ```bash
   curl -L -o models/smart-turn-v3.2-cpu.onnx \
     "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
   ```

4. Create `config.local.json` in the project root:
   ```json
   {
     "MODEL_PATH": "/path/to/model.litertlm",
     "MUSIC_DIR": "/path/to/your/music"
   }
   ```
   `MUSIC_DIR` is optional — omit it if you don't want music playback.

5. Open OrbStack (required for the code execution sandbox).

6. Set up notes sync (optional):
   - Install [Obsidian](https://obsidian.md/download) and open `obsidian/strawberry` as a vault.
   - Install [Syncthing](https://syncthing.net/) on macOS (`brew install syncthing`) and on Android (Syncthing-Fork).
   - Pair devices at `http://127.0.0.1:8384`, share the `obsidian/strawberry` folder, and open the synced folder as a vault in Obsidian on Android.
   - Add todos manually to `obsidian/strawberry/TODO.md` — Strawberry can read and tick them off.

## Run

```bash
uv run main.py
```

On startup, Strawberry opens the audio bridge page in your default browser. Click **Connect Microphone** — this is how the microphone is captured (with browser-side AEC). The assistant won't start listening until the browser is connected.

Say **"hey strawberry"** to wake it.

## Contributions
I would like to take a different approach with regards to contributions for this project. I strongly believe that the future is gravitating towards what I call "Take It Home OSS" — i.e. fork freely, modify it to your liking using AI coding agents, and stop waiting for upstream permission. This means you do not need to submit PRs or issues, except for critical bugs or security fixes. It is as if OSS is raw material, and your fork is your product.

## Credits

- [Keyboard typing](https://pixabay.com/sound-effects/film-special-effects-typing-on-laptop-keyboard-308455/), [bubble pop](https://pixabay.com/sound-effects/technology-bubble-pop-07-487896/) and [shutdown](https://pixabay.com/sound-effects/film-special-effects-beep-401570/) sounds are from Pixabay, licensed under the [Pixabay Content License](https://pixabay.com/service/license-summary/).
