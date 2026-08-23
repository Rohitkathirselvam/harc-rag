import re


SAFE_REFUSAL = "I don't have enough information from the provided documents to answer this question."


def is_safe_refusal(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer).strip().rstrip(".")
    expected = re.sub(r"\s+", " ", SAFE_REFUSAL).strip().rstrip(".")
    return normalized == expected
