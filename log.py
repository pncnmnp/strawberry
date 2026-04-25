from rich.console import Console
from rich.rule import Rule
import inspect
import time

from lm import MAX_TOKENS

console = Console(highlight=False)

_LABELS: dict[str, tuple[str, str]] = {
    "listen": ("LISTEN",  "bold cyan"),
    "input":  ("YOU",     "bold white"),
    "tool":   ("TOOL",    "bold yellow"),
    "result": ("RESULT",  "yellow"),
    "reply":  ("GEMMA",   "bold green"),
    "speak":  ("SPEAK",   "bold cyan"),
    "warn":   ("WARN",    "bold red"),
    "search":    ("SEARCH",    "bold blue"),
    "wikipedia": ("WIKIPEDIA", "bold cyan"),
    "duckduckgo":("DUCKDUCKGO","bold yellow"),
    "notes":  ("NOTES",   "bold green"),
    "reset":  ("RESET",   "bold magenta"),
    "interrupt": ("ABORT", "bold red"),
    "dump":   ("DUMP",    "dim magenta"),
    "wake":   ("WAKE",    "bold magenta"),
    "vad":    ("VAD",     "bold color(45)"),
    "warmup": ("WARMUP",  "bold color(135)"),
    "music":  ("MUSIC",   "bold color(208)"),
    "compact":("COMPACT", "bold magenta"),
}

_last_time: float = time.perf_counter()


def divider():
    global _last_time
    console.print(Rule(style="dim"))
    _last_time = time.perf_counter()

_tool_schema_tokens: int = 0

def compute_tool_schema_tokens(tools: list) -> None:
    """Call once at startup with the tool functions to measure their schema overhead."""
    global _tool_schema_tokens
    total = 0
    for fn in tools:
        doc = inspect.getdoc(fn) or ""
        total += len(f"{fn.__name__} {doc} {inspect.signature(fn)}")
    _tool_schema_tokens = total // 4

# NOTE: Is this accurate?
# https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
# Somewhat. We use chars // 4 as a rough token estimate.
def _usage(history: list, tool_chars: int, total: int) -> tuple[int, float]:
    turn_overhead = len(history) * 6  # <|turn>role\n...<turn|>\n per message
    used = (sum(len(m["content"]) for m in history if isinstance(m.get("content"), str)) + tool_chars) // 4 + _tool_schema_tokens + turn_overhead
    return used, min(used / total, 1.0)


def context_pct(history: list, tool_chars: int = 0, total: int = MAX_TOKENS) -> float:
    return _usage(history, tool_chars, total)[1]


def log_context(history: list, total: int = MAX_TOKENS, tool_chars: int = 0):
    used, pct = _usage(history, tool_chars, total)
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    color = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "bold red"
    console.print(f"  [dim]CONTEXT [/dim]  [{color}]{bar}[/{color}]  [dim]{used}/{total} tokens[/dim]")


def log(kind: str, message: str = ""):
    global _last_time
    now = time.perf_counter()
    delta_ms = (now - _last_time) * 1000
    _last_time = now

    label, style = _LABELS[kind]
    console.print(f"  [{style}]{label:<8}[/{style}]  {message}  [dim]+{delta_ms:.0f}ms[/dim]")
