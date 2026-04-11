# Strawberry

My local voice assistant. Listens via microphone, transcribes with Whisper, thinks with Gemma via Ollama, speaks with Kokoro TTS.

## What it can do
- Answer questions (Wikipedia or DuckDuckGo, auto-routed)
- Check weather
- Save and recall personal notes with keyword, tag, and time-range search
- Reset conversation history
- Shut down on command

## Stack
- **STT**: Faster Whisper (`base`)
- **LLM**: `gemma4:e2b` via Ollama (100% GPU on Apple Silicon)
- **TTS**: Kokoro (`af_heart` voice)
- **Search**: Wikipedia + DuckDuckGo
- **Notes**: SQLite FTS5 with stemming, time parsing, and query expansion

## Run

```bash
ollama serve
uv run main.py
```

## Credits

- [Keyboard typing](https://pixabay.com/sound-effects/film-special-effects-typing-on-laptop-keyboard-308455/), [bubble pop](https://pixabay.com/sound-effects/technology-bubble-pop-07-487896/) and [shutdown](https://pixabay.com/sound-effects/film-special-effects-beep-401570/) sounds are from Pixabay, licensed under the [Pixabay Content License](https://pixabay.com/service/license-summary/).
