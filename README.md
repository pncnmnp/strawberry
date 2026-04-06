# hello-world

A local voice assistant. Listens via microphone, transcribes with Whisper, thinks with Gemma via Ollama, speaks with Kokoro TTS.

## What it can do
- Answer questions (Wikipedia or DuckDuckGo, auto-routed)
- Check weather
- Reset its conversation history
- Shut down on command

## Stack
- **STT**: OpenAI Whisper (`base`)
- **LLM**: `gemma4:e2b` via Ollama (100% GPU on Apple Silicon)
- **TTS**: Kokoro (`af_heart` voice)
- **Search**: Wikipedia + DuckDuckGo, with a pixie-based router

## Run

```bash
ollama serve
uv run main.py
```
