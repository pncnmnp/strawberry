import subprocess
import json
from datetime import datetime

def get_weather(location: str = "") -> str:
    """Get the current weather. Location is optional — auto-detects via IP if omitted.

    Args:
        location: The city or location to get weather for. Omit to auto-detect.
    """
    path = location.replace(" ", "+") if location else ""
    result = subprocess.run(
        ["curl", "-s", f"wttr.in/{path}?format=j1"],
        capture_output=True,
        text=True,
    )
    try:
        w = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "Weather service is unavailable right now."

    curr = w["current_condition"][0]
    today = w["weather"][0]
    city = w["nearest_area"][0]["areaName"][0]["value"]
    next_hour = today["hourly"][0]

    desc = curr["weatherDesc"][0]["value"].strip()
    temp_f = int(curr.get("temp_F", 0) or 0)
    feels_f = int(curr.get("FeelsLikeF", temp_f) or temp_f)
    uv = int(curr.get("uvIndex", 0) or 0)
    wind_mph = int(curr.get("windspeedMiles", 0) or 0)
    wind_dir = curr.get("winddir16Point", "").strip()
    rain_next = int(next_hour.get("chanceofrain", 0) or 0)
    sunrise = today["astronomy"][0]["sunrise"]
    sunset = today["astronomy"][0]["sunset"]
    local_time_str = curr.get("localObsDateTime", "")

    layer = ("Grab a warm jacket." if feels_f <= 45
             else "Light layer suggested." if feels_f <= 60
             else "Long sleeves optional." if feels_f <= 75
             else "Short sleeves are fine.")

    try:
        hour = datetime.strptime(local_time_str, "%Y-%m-%d %I:%M %p").hour
        time_of_day = "night" if hour < 5 else "morning" if hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"
    except ValueError:
        time_of_day = "unknown"

    facts = "\n".join([
        f"location: {city}",
        f"local_time: {local_time_str} ({time_of_day})",
        f"condition: {desc}",
        f"temp: {temp_f}°F (feels like {feels_f}°F)",
        f"wind: {wind_mph} mph {wind_dir}",
        f"rain_next_hour: {rain_next}%",
        f"uv_index: {uv}",
        f"sunrise: {sunrise} / sunset: {sunset}",
        f"clothing: {layer}",
    ])
    return f"Present this weather report like a lively TV weatherman — natural, conversational, not a dry recitation:\n{facts}"
