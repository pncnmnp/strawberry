from datetime import datetime
import zoneinfo
import subprocess
import json

def get_current_datetime(timezone: str = "") -> str:
    """Get the current date and time with location.

    Args:
        timezone: IANA timezone name (e.g. 'America/New_York'). Omit for local system time.
    """
    if timezone:
        try:
            tz = zoneinfo.ZoneInfo(timezone)
            now = datetime.now(tz)
        except zoneinfo.ZoneInfoNotFoundError:
            return f"Unknown timezone: {timezone}"
    else:
        now = datetime.now().astimezone()

    location = ""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://ip-api.com/json/?fields=city"],
            capture_output=True, text=True, timeout=3,
        )
        location = json.loads(result.stdout).get("city", "")
    except Exception:
        pass

    hour = now.hour % 12 or 12
    minute = now.minute
    period = "AM" if now.hour < 12 else "PM"
    time_str = f"{hour} {minute:02d} {period}" if minute else f"{hour} {period}"

    lines = [
        f"date: {now.strftime('%A, %B %-d')}",
        f"time: {time_str}",
        f"timezone: {now.strftime('%Z')}",
    ]
    if location:
        lines.append(f"location: {location}")
    return "\n".join(lines)
