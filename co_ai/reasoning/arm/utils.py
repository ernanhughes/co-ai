REASONING_FORMATS = {
    "direct": "<Direct>",
    "short_cot": "<Short_CoT>",
    "code": "<Code>",
    "long_cot": "<Long_CoT>"
}


def detect_format(text):
    for fmt, token in REASONING_FORMATS.items():
        if token in text:
            return fmt
    return "unknown"