DEFINITION = {
    "type": "function",
    "function": {
        "name": "shutdown",
        "description": "Shut down and exit the assistant when the user asks to stop, quit, exit, goodbye, or similar.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def shutdown() -> str:
    return "shutting_down"
