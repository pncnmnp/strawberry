import json
import ollama
from datetime import datetime

_SYSTEM = """You parse natural language time expressions into a JSON date range.

Think step by step:
1. Identify the unit (minutes, hours, days, weeks, months, years).
2. Identify the quantity.
3. Subtract that exact amount from the current time to get "from".
4. "to" is the current time.

IMPORTANT: 1 hour = subtract 1 from the HOUR only. 1 day = subtract 1 from the DAY. Do NOT confuse them.

Return ONLY valid JSON in this exact format:
{"unit": "<the unit>", "qty": <number>, "from": "YYYY-MM-DDTHH:MM:SS", "to": "YYYY-MM-DDTHH:MM:SS"}

For expressions like "today", "yesterday", "this month" etc. set unit to the period and qty to 1.
If you cannot determine a range, return: {"unit": null, "qty": null, "from": null, "to": null}

Examples (assuming now is 2026-04-06T23:30:00):
- "last hour"     → {"unit": "hour", "qty": 1, "from": "2026-04-06T22:30:00", "to": "2026-04-06T23:30:00"}
- "last 2 hours"  → {"unit": "hour", "qty": 2, "from": "2026-04-06T21:30:00", "to": "2026-04-06T23:30:00"}
- "today"         → {"unit": "day", "qty": 1, "from": "2026-04-06T00:00:00", "to": "2026-04-06T23:59:59"}
- "yesterday"     → {"unit": "day", "qty": 1, "from": "2026-04-05T00:00:00", "to": "2026-04-05T23:59:59"}
- "last week"     → {"unit": "week", "qty": 1, "from": "2026-03-30T00:00:00", "to": "2026-04-05T23:59:59"}
- "last 3 days"   → {"unit": "day", "qty": 3, "from": "2026-04-03T00:00:00", "to": "2026-04-06T23:59:59"}"""


_UNIT_SECONDS = {
    "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000, "year": 31536000,
}


def _validate(data: dict, now: datetime) -> tuple[str | None, str | None]:
    """If the model reported unit+qty, verify the delta is in the right ballpark."""
    fr, to = data.get("from"), data.get("to")
    unit, qty = data.get("unit"), data.get("qty")
    if not fr or not unit or not qty or unit not in _UNIT_SECONDS:
        return fr, to
    try:
        from_dt = datetime.fromisoformat(fr)
        expected_secs = _UNIT_SECONDS[unit] * qty
        actual_secs = (now - from_dt).total_seconds()
        # if the actual delta is off by more than 3x, recompute from Python
        if actual_secs > expected_secs * 3 or actual_secs < expected_secs / 3:
            from_dt = now - __import__("datetime").timedelta(seconds=expected_secs)
            return from_dt.strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        pass
    return fr, to


def parse(time_range: str) -> tuple[str | None, str | None]:
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    response = ollama.chat(
        model="gemma4:e2b",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Now is {now_str}. Parse: {time_range}"},
        ],
    )
    try:
        data = json.loads(response["message"]["content"].strip())
        return _validate(data, now)
    except (json.JSONDecodeError, KeyError):
        return None, None
