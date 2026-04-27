import os
import sys

os.environ.setdefault("GLOG_minloglevel", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "4")

import litert_lm

MODEL_PATH = "/Users/parth/.litert-lm/models/gemma-e4b/model.litertlm"
PIXIE_MODEL_PATH = "/Users/parth/.litert-lm/models/gemma-e4b/model.litertlm"
MAX_TOKENS = 16384
# NOTE: Why is it low?
# When the MAX TOKENS is set high for deep think calls, the latency can become unacceptably long.
# We are talking about several minutes here.
# As our deep think calls are solely used for single-use scenarios (i.e not part of a multi-turn conversation),
# we can afford to do this to keep latency reasonable. An okay compromise for now.
THINK_MAX_TOKENS = 4096

_engine = None
_pixie_engine = None
_think_engine = None


def _suppress_stderr():
    """Redirect native stderr to /dev/null, return fd to restore later."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    return saved_fd


def _restore_stderr(saved_fd):
    os.dup2(saved_fd, sys.stderr.fileno())
    os.close(saved_fd)


# NOTE: Gemma 4 E2B supports 128K context but uses a 4:1 local-to-global attention pattern —
# 4 layers of 512-token sliding window followed by 1 global layer. 
# Local layers (4/5) only cache 512 tokens regardless of context length, so per-layer KV cache is fixed.
# Only the global layers (1/5) scale with context, and those use 8:1 GQA + K=V sharing.
# https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
# 16K is comfortable on 16GB RAM.
def _quiet_engine(path: str, max_tokens: int = MAX_TOKENS) -> litert_lm.Engine:
    saved_fd = _suppress_stderr()
    # TODO: GPU backend is ~4x faster but corrupts token sampling — "Don't" becomes
    # "You't", tool args get mangled, math breaks. Revisit when LiteRT-LM fixes GPU accuracy.
    try:
        return litert_lm.Engine(path, max_num_tokens=max_tokens)
    finally:
        _restore_stderr(saved_fd)

# NOTE: Why are we not using audio input since the model is multimodal?
# I tried it. The problem is that audio tokens are orders of magnitude larger than
# text tokens. In a multi-turn conversation the KV cache fills up within a few turns,
# after which the model produces degenerate single-token outputs ("A", "The", "I", "").
# Whisper transcribes each utterance to ~10-50 text tokens instead, so the context
# stays manageable across a long session.
# Audio input drastically decreases the latency (~1 sec), but is only practical for single-turn or short-session use.
def get_engine() -> litert_lm.Engine:
    """Engine for the main conversation session."""
    global _engine
    if _engine is None:
        _engine = _quiet_engine(MODEL_PATH)
    return _engine


def _get_pixie_engine() -> litert_lm.Engine:
    """Separate engine for pixie one-shot calls (avoids session conflict)."""
    global _pixie_engine
    if _pixie_engine is None:
        _pixie_engine = _quiet_engine(PIXIE_MODEL_PATH)
    return _pixie_engine


def _get_think_engine() -> litert_lm.Engine:
    """Capped engine for thinking calls — limits total output (thinking + response)."""
    global _think_engine
    if _think_engine is None:
        _think_engine = _quiet_engine(MODEL_PATH, max_tokens=THINK_MAX_TOKENS)
    return _think_engine


def cleanup():
    """Delete engines to avoid nanobind leak warnings on exit."""
    global _engine, _pixie_engine, _think_engine
    _engine = None
    _pixie_engine = None
    _think_engine = None


def chat(system: str, user: str) -> str:
    """One-shot chat for pixies — no conversation history."""
    engine = _get_pixie_engine()
    saved = _suppress_stderr()
    try:
        with engine.create_conversation(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system}]}
            ],
        ) as conv:
            response = conv.send_message(user)
    finally:
        _restore_stderr(saved)
    content = response.get("content", [])
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if c.get("type") == "text")
    return str(content)


def think(system: str, question: str) -> dict:
    """One-shot thinking chat. Returns {"thought": ..., "response": ...}.

    To enable thinking, include <|think|> at the start of the system prompt or message.
    See: https://huggingface.co/google/gemma-4-E2B

    Why a separate tool call instead of per-message <|think|>?
    - Per-message <|think|> in multi-turn is unreliable: thinking bleeds across turns,
      causing the model to think even when not prompted.
    - System-prompt <|think|> works but configures the entire session for thinking,
      increasing average latency on every turn.
    - Per-message <|think|> in isolated (single-turn) conversations is reliable.

    Verified experimentally.
    So we spin up a fresh tool-based conversation per thinking request.
    """
    engine = _get_think_engine()
    saved = _suppress_stderr()
    try:
        with engine.create_conversation(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "<|think|>\n" + system}]}
            ],
        ) as conv:
            response = conv.send_message(question)
    finally:
        _restore_stderr(saved)
    thought = response.get("channels", {}).get("thought", "")
    content = response.get("content", [])
    if isinstance(content, list):
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
    else:
        text = str(content)
    return {"thought": thought, "response": text}
