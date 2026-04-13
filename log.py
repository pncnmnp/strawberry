from rich.console import Console
from rich.rule import Rule
import time

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
    "music":  ("MUSIC",   "bold color(208)")
}

_last_time: float = time.perf_counter()


def divider():
    global _last_time
    console.print(Rule(style="dim"))
    _last_time = time.perf_counter()

# NOTE: Is this accurate?
# https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
# Somewhat. We use chars // 4 as a rough token estimate, which misses chat template overhead
# (~16 tokens/turn for role markers) and JSON scaffolding around tool calls. So we undercount
# by maybe 200-400 tokens. But the model also seems to degrade in quality well before the KV
# cache is actually full (~3K tokens), some limitation somewhere that I am overlooking rn.
def log_context(history: list, total: int = 4096, tool_chars: int = 0):
    used = (sum(len(m["content"]) for m in history if isinstance(m.get("content"), str)) + tool_chars) // 4
    pct = min(used / total, 1.0)
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
