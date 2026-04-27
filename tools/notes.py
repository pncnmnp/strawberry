from datetime import datetime
from pathlib import Path
from pixies import time_parse
from log import log

VAULT = Path(__file__).parent.parent / "obsidian" / "strawberry" / "notes"
VAULT.mkdir(parents=True, exist_ok=True)
TODO_FILE = Path(__file__).parent.parent / "obsidian" / "strawberry" / "TODO.md"


def save_note(content: str, tag: str = "general") -> str:
    """Save a personal note or piece of information the user wants to remember. Markdown is supported.

    Args:
        content: The note to save. Markdown formatting is supported (bold, lists, links, etc.).
        tag: Topic tag to categorise the note (e.g. 'reminders', 'ideas', 'work'). Defaults to 'general'.
    """
    tag = tag.lower().strip()
    now = datetime.now()
    daily = VAULT / f"{now.date()}.md"

    if not daily.exists():
        daily.write_text(f"# {now.date()}\n\n")

    with daily.open("a") as f:
        f.write(f"- {now.strftime('%H:%M:%S')} [{tag}] {content}\n")

    log("notes", f"saved [{tag}] {content[:60]}")
    return f"Note saved under '{tag}'."


def recall_notes(
    query: str | None = None,
    tag: str | None = None,
    time_range: str | None = None,
) -> str:
    """Search and recall saved personal notes. Supports keyword search, tag filter, and natural language time range.

    Args:
        query: Keyword or topic to search notes for.
        tag: Filter notes by tag.
        time_range: Natural language time window, e.g. 'last hour', 'last 2 hours', 'today', 'yesterday', 'last week', 'in March'.
    """
    from_dt, to_dt = time_parse(time_range) if time_range else (None, None)

    if time_range:
        log("notes", f"time range: {from_dt} → {to_dt}")

    results = []

    for daily in sorted(VAULT.glob("*.md"), reverse=True):
        try:
            file_date = daily.stem
        except ValueError:
            continue

        for line in daily.read_text().splitlines():
            if not line.startswith("- "):
                continue

            # parse: - HH:MM:SS [tag] content
            rest = line[2:]
            try:
                time_str, rest = rest.split(" ", 1)
                line_tag = rest[rest.index("[") + 1 : rest.index("]")]
                line_content = rest[rest.index("]") + 2 :]
            except (ValueError, IndexError):
                continue

            timestamp = f"{file_date}T{time_str}"

            if from_dt and timestamp < from_dt:
                continue
            if to_dt and timestamp > to_dt:
                continue
            if tag and line_tag != tag.lower().strip():
                continue
            if query and query.lower() not in line_content.lower():
                continue

            results.append(f"[{timestamp}] ({line_tag}) {line_content}")

    if not results:
        return "No notes found."

    return "\n".join(results)


def add_todo(content: str) -> str:
    """Add a new item to the top of the TODO list.

    Args:
        content: The todo item to add.
    """
    lines = TODO_FILE.read_text().splitlines() if TODO_FILE.exists() else ["# TODO", ""]

    # insert after the heading block
    insert_at = next((i + 1 for i, l in enumerate(lines) if l.startswith("# ")), 0) + 1
    lines.insert(insert_at, f"- [ ] {content}")
    TODO_FILE.write_text("\n".join(lines) + "\n")
    log("notes", f"added todo: {content[:60]}")
    return f"Todo added: {content}"


def recall_todos(include_done: bool = False) -> str:
    """Read the TODO list. By default returns only pending items.

    Args:
        include_done: If True, also return completed todos.
    """
    if not TODO_FILE.exists():
        return "No TODO file found."

    items = []
    for line in TODO_FILE.read_text().splitlines():
        if line.startswith("- [ ]"):
            items.append(line)
        elif include_done and line.startswith("- [x]"):
            items.append(line)

    if not items:
        return "No todos found."

    return "\n".join(items)


def complete_todo(query: str) -> str:
    """Mark a TODO item as done by matching its text.

    Args:
        query: Substring to identify the todo item to mark as complete.
    """
    if not TODO_FILE.exists():
        return "No TODO file found."

    lines = TODO_FILE.read_text().splitlines()
    matches = [i for i, l in enumerate(lines) if l.startswith("- [ ]") and query.lower() in l.lower()]

    if not matches:
        return f"No pending todo matching '{query}'."
    if len(matches) > 1:
        previews = "\n".join(lines[i] for i in matches)
        return f"Multiple todos match '{query}' — be more specific:\n{previews}"

    idx = matches[0]
    lines[idx] = lines[idx].replace("- [ ]", "- [x]", 1)
    TODO_FILE.write_text("\n".join(lines) + "\n")
    log("notes", f"completed todo: {lines[idx][:60]}")
    return f"Marked done: {lines[idx][6:]}"
