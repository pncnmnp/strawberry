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
- Mute until next wake word
- Reset conversation history + auto-compaction
- Shut down and powercycle on command

## Stack
- **STT**: Faster Whisper (`small.en`)
- **LLM**: Gemma via LiteRT-LM
- **TTS**: Kokoro (`af_heart` voice)
- **Search**: Wikipedia + DuckDuckGo
- **Notes**: Markdown files in an Obsidian vault, synced to mobile via Syncthing

## Setup

1. Install [VLC](https://www.videolan.org/vlc/) (required for music playback).

2. Download the Gemma model weights from [HuggingFace](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) and place the `.litertlm` file at:
   ```
   ~/.litert-lm/models/gemma-e2b/model.litertlm
   ```

3. Open OrbStack (required for the code execution sandbox).

4. Set up notes sync (optional):
   - Install [Obsidian](https://obsidian.md/download) and open `obsidian/strawberry` as a vault.
   - Install [Syncthing](https://syncthing.net/) on macOS (`brew install syncthing && brew services start syncthing`) and on Android (Syncthing-Fork).
   - Pair devices at `http://127.0.0.1:8384`, share the `obsidian/strawberry` folder, and open the synced folder as a vault in Obsidian on Android.
   - Add todos manually to `obsidian/strawberry/TODO.md` — Strawberry can read and tick them off.

5. Download the smart-turn VAD model weights:
   ```bash
   curl -L -o models/smart-turn-v3.2-cpu.onnx \
     "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
   ```

## Run

```bash
uv run main.py
```

## Contributions
I would like to take a different approach with regards to contributions for this project. I strongly believe that the future is gravitating towards what I call "Take It Home OSS" — i.e. fork freely, modify it to your liking using AI coding agents, and stop waiting for upstream permission. This means you do not need to submit PRs or issues, except for critical bugs or security fixes. It is as if OSS is raw material, and your fork is your product.

## Credits

- [Keyboard typing](https://pixabay.com/sound-effects/film-special-effects-typing-on-laptop-keyboard-308455/), [bubble pop](https://pixabay.com/sound-effects/technology-bubble-pop-07-487896/) and [shutdown](https://pixabay.com/sound-effects/film-special-effects-beep-401570/) sounds are from Pixabay, licensed under the [Pixabay Content License](https://pixabay.com/service/license-summary/).
