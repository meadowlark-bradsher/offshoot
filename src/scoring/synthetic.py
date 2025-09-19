"""Scoring for synthetic tasks."""

import re
from typing import Optional, Dict, Any, Union


def extract_boxed_value(response: str) -> Optional[int]:
    """Extract integer value from \\boxed{} if it exists."""
    match = re.search(r"\\boxed\{([^\}]+)\}", response)
    if match:
        try:
            return int(match.group(1).strip())
        except ValueError:
            return None
    return None


def extract_any_number(response: str) -> Optional[int]:
    """Extract any integer from the response as fallback."""
    numbers = re.findall(r'-?\d+', response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return None
    return None


def score_synthetic_response(
    response: str,
    expected: int,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Score a synthetic task response.

    Returns scoring results including multiple evaluation criteria.
    """
    predicted_boxed = extract_boxed_value(response)
    predicted_any = extract_any_number(response)

    exact_match = predicted_boxed == expected if predicted_boxed is not None else False
    value_present = str(expected) in response
    used_code = "```python" in response.lower() or "print(" in response.lower()

    scores = {
        "predicted_boxed": predicted_boxed,
        "predicted_any": predicted_any,
        "expected": expected,
        "exact_match": exact_match,
        "value_present": value_present,
        "used_code": used_code,
        "response_length": len(response),
        "contains_calculation": any(op in response for op in ['+', '-', '*', '/', '=']),
    }

    return scores


def determine_failure_reason(
    scores: Dict[str, Any],
    token_limit_reached: bool = False
) -> str:
    """Determine the primary failure reason."""
    # We don't treat token limits as failures - only accuracy matters
    if not scores["exact_match"]:
        if scores["predicted_boxed"] is None:
            return "no_boxed_answer"
        else:
            return "wrong_answer"
    else:
        return "none"