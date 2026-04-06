from rich.console import Console
from rich.rule import Rule

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
    "reset":  ("RESET",   "bold magenta"),
    "dump":   ("DUMP",    "dim magenta"),
}


def divider():
    console.print(Rule(style="dim"))


def log(kind: str, message: str = ""):
    label, style = _LABELS[kind]
    console.print(f"  [{style}]{label:<8}[/{style}]  {message}")
