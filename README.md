# Strawberry

My local voice assistant. Listens via microphone, transcribes with Whisper, thinks with Gemma via LiteRT-LM, speaks with Kokoro TTS.

## What it can do
- Answer questions (Wikipedia or DuckDuckGo, auto-routed)
- Play music (requires `MUSIC_DIR` in `config.local.py`)
- Check weather
- Save and recall personal notes with keyword, tag, and time-range search
- Reset conversation history
- Shut down on command

## Stack
- **STT**: Faster Whisper (`base`)
- **LLM**: Gemma via LiteRT-LM (100% GPU on Apple Silicon)
- **TTS**: Kokoro (`af_heart` voice)
- **Search**: Wikipedia + DuckDuckGo
- **Notes**: SQLite FTS5 with stemming, time parsing, and query expansion

## Run

```bash
ollama serve
uv run main.py
```

## Contributions
I would like to take a different approach with regards to contributions for this project. I strongly believe that the future is gravitating towards what I call "Take It Home OSS" — i.e. fork freely, modify it to your liking using AI coding agents, and stop waiting for upstream permission. This means you do not need to submit PRs or issues, except for critical bugs or security fixes. It is as if OSS is raw material, and your fork is your product.

## Credits

- [Keyboard typing](https://pixabay.com/sound-effects/film-special-effects-typing-on-laptop-keyboard-308455/), [bubble pop](https://pixabay.com/sound-effects/technology-bubble-pop-07-487896/) and [shutdown](https://pixabay.com/sound-effects/film-special-effects-beep-401570/) sounds are from Pixabay, licensed under the [Pixabay Content License](https://pixabay.com/service/license-summary/).
