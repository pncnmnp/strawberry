DEFINITION = {
    "type": "function",
    "function": {
        "name": "mute",
        "description": "Go to sleep. Call this when the user says thanks, goodbye, nothing, or is done talking.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def mute() -> str:
    return "muting"
