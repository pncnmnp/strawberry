from lm import think

_THINK_PROMPT = "You are the reasoning engine behind a real-time voice assistant. Think through the problem carefully and provide a clear, well-reasoned answer."

def deep_think(question: str) -> dict:
    """Engage extended reasoning for problems that require multi-step logic, math, analysis, or planning.
    Do NOT use this for simple factual recall, greetings, or anything answerable in one step.
    Latency is high — only invoke when the quality gain justifies the wait.

    Args:
        question: The full question or problem to reason through.
    """
    return think(_THINK_PROMPT, question)
