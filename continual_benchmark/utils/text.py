"""Text processing utilities."""

import json
import re
from typing import Any


def normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", s).strip()


def extract_json(text: str) -> Any:
    """Extract and parse JSON from a string that may contain extra text.

    Tries to find a JSON object or array in the text.
    Returns the parsed JSON value, or raises ValueError.
    """
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract valid JSON from: {text[:200]}")


def extract_code_block(text: str) -> str:
    """Extract code from a markdown code block, or return the text as-is.

    Handles ```python ... ``` and ``` ... ``` fences.
    """
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
