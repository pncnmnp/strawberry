DEFINITION = {
    "type": "function",
    "function": {
        "name": "mute",
        "description": "Call this EVERY time the user says thank you, goodbye, see you later, that's all, or otherwise signals they are done. Also call it when you answer a one-off question that needs no follow-up. You will go to sleep and the user must say the wake word to talk again.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def mute() -> str:
    return "muting"
