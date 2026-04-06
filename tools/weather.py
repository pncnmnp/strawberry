import subprocess

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for",
                }
            },
            "required": ["location"],
        },
    },
}


def get_weather(location: str) -> str:
    location = location.replace(" ", "+")
    result = subprocess.run(
        ["curl", "-s", f"wttr.in/{location}?format=3"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
