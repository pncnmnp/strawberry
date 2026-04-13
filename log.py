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


def log(kind: str, message: str = ""):
    global _last_time
    now = time.perf_counter()
    delta_ms = (now - _last_time) * 1000
    _last_time = now

    label, style = _LABELS[kind]
    console.print(f"  [{style}]{label:<8}[/{style}]  {message}  [dim]+{delta_ms:.0f}ms[/dim]")
