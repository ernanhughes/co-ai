REASONING_FORMATS = {
    "direct": "<Direct>",
    "short_cot": "<Short_CoT>",
    "code": "<Code>",
    "long_cot": "<Long_CoT>"
}

FORMAT_END_TAGS = {
    "direct": "</Direct>",
    "short_cot": "</Short_CoT>",
    "code": "</Code>",
    "long_cot": "</Long_CoT>"
}

def detect_difficulty(text: str) -> str:
    words = text.split()
    if len(words) < 20:
        return "easy"
    elif len(words) < 50:
        return "medium"
    else:
        return "hard"


def detect_format(text: str) -> str:
    text = text.strip().lower()
    if not text:
        return "unknown"
    if "<direct>" in text:
        return "direct"
    elif "<short_cot>" in text:
        return "short_cot"
    elif "<code>" in text:
        return "code"
    elif "<long_cot>" in text:
        return "long_cot"
    
    # Direct Answer
    if text.startswith("the answer is") or text.startswith("answer:"):
        return "direct"

    # Short CoT
    elif text.startswith("let me think briefly"):
        return "short_cot"

    # Long CoT
    elif text.startswith("let's analyze this step-by-step"):
        return "long_cot"

    # Code
    elif any(kw in text for kw in ["def", "return", "solve()", "print(", "for ", "if "]):
        return "code"

    else:
        print(f"[WARNING] Unknown format:\n{text[:100]}...")
        return "unknown"