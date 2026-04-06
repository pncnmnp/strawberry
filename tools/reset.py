DEFINITION = {
    "type": "function",
    "function": {
        "name": "reset_history",
        "description": "Clear the conversation history and start fresh when the user asks to reset, forget everything, start over, or similar.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def reset_history() -> str:
    return "resetting"
