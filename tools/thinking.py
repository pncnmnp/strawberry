from lm import think
from prompts.mk1 import SYSTEM_PROMPT

def deep_think(question: str) -> str:
    """Engage extended reasoning for problems that require multi-step logic, math, analysis, or planning.
    Do NOT use this for simple factual recall, greetings, or anything answerable in one step.
    Latency is high — only invoke when the quality gain justifies the wait.

    Args:
        question: The full question or problem to reason through.
    """
    result = think(SYSTEM_PROMPT, question)
    return result["response"]
