import os
import sys

os.environ.setdefault("GLOG_minloglevel", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "4")

import litert_lm

MODEL_PATH = "/Users/parth/.litert-lm/models/gemma-e2b/model.litertlm"

_engine = None
_pixie_engine = None


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


def _quiet_engine(path: str) -> litert_lm.Engine:
    saved_fd = _suppress_stderr()
    try:
        return litert_lm.Engine(path, max_num_tokens=4096)
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
        _pixie_engine = _quiet_engine(MODEL_PATH)
    return _pixie_engine


def cleanup():
    """Delete engines to avoid nanobind leak warnings on exit."""
    global _engine, _pixie_engine
    _engine = None
    _pixie_engine = None


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
