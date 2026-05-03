import json
import re

from lm import chat
from log import log
from memory import store

_EXTRACT_SYSTEM = """\
Extract durable facts about the user from this conversation — facts that would still be useful months from now.

Include: stable preferences, personal details, goals, habits, relationships, recurring interests the user reveals about themselves.

Skip:
- Transient state ("user is asking about X", "user wants the weather right now")
- Meta-narration about the conversation ("user is engaged in conversation", "user is being prompted to choose…")
- Things the assistant said, assumed, or how the assistant addressed the user — only what the user revealed
- Tool invocations and their results
- One-off observations that won't matter later

Return ONLY a JSON array of short declarative sentences (at most 5).
If nothing durable was revealed, return []."""

_turn_buffer: list[tuple[str, str]] = []


def queue_turn(user_turn: str, assistant_turn: str):
    """Accumulate a turn for later extraction. Does no LLM work."""
    _turn_buffer.append((user_turn, assistant_turn))


def flush():
    """Process all buffered turns. Blocks — call only at idle points (shutdown, compact)."""
    if not _turn_buffer:
        return
    turns = _turn_buffer.copy()
    _turn_buffer.clear()
    log("memory", f"flushing {len(turns)} buffered turn(s)...")
    _extract_and_store(turns)


def _parse_json_list(text: str) -> list[str]:
    try:
        if match := re.search(r'\[.*?\]', text, re.DOTALL):
            result = json.loads(match.group())
            if isinstance(result, list):
                return [s for s in result if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _process_fact(fact: str):
    # NOTE: No updates or deletes: they require extra LLM calls to choose the
    # actions, and overwriting an existing fact can erase a better prior version.
    # Retrieval surfaces a date per memory so the LLM can prefer recent
    # facts over stale ones on conflicts.
    if store.search_similar(fact, k=1, threshold=0.95):
        log("memory", f"NOOP (duplicate) {fact!r}")
        return
    mem_id = store.add(fact)
    log("memory", f"ADD [{mem_id}] {fact!r}")


def _extract_and_store(turns: list[tuple[str, str]]):
    try:
        convo = "\n\n".join(f'User: "{u}"\nAssistant: "{a}"' for u, a in turns)
        facts = _parse_json_list(chat(_EXTRACT_SYSTEM, convo))
        if not facts:
            log("memory", "no durable facts")
            return
        log("memory", f"extracted {len(facts)} fact(s)")
        for fact in facts:
            _process_fact(fact)
    except Exception as e:
        log("memory", f"extraction error: {e}")
