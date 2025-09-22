"""Arithmetic chain generation based on reference notebook."""

from typing import Tuple
from ...common.rng import SeededRNG


def generate_chain_task(
    initial_value: int,
    depth: int,
    condition: str = "terse",
    rng: SeededRNG = None
) -> str:
    """
    Generate arithmetic chain task prompt.

    Based on the reference notebook pattern but with condition variations.
    Now uses RNG for randomized operation generation.
    """
    operations = []

    for i in range(1, depth + 1):
        # Randomize operation type and operand
        op_type = rng.choice(["add", "subtract", "multiply"])

        if op_type == "add":
            operand = rng.randint(1, 20)
            operations.append(f"add {operand}")
        elif op_type == "subtract":
            operand = rng.randint(1, 20)
            operations.append(f"subtract {operand}")
        else:  # multiply
            operand = rng.randint(2, 12)
            operations.append(f"multiply by {operand}")

    if condition == "terse":
        task = (
            f"Start with {initial_value}, then " +
            ", then ".join(operations) +
            ". What is the final result?"
        )
        prompt = f"Evaluate the following step by step. Put the final result inside \\boxed{{}}: {task}"

    elif condition == "verbose":
        task = (
            f"We begin with the initial value of {initial_value}. " +
            "Next, we apply the following sequence of mathematical operations in order: " +
            ", followed by ".join(operations) +
            ". Please compute the final numerical result."
        )
        prompt = (
            f"Please carefully evaluate the following mathematical expression step by step, "
            f"showing your work for each operation. "
            f"Put the final result inside \\boxed{{}}: {task}"
        )

    elif condition == "redundant":
        operations_verbose = []
        for i, op in enumerate(operations, 1):
            operations_verbose.append(f"Step {i}: {op}")

        task = (
            f"Initial value: {initial_value}. "
            f"Mathematical operations to perform sequentially: " +
            "; ".join(operations_verbose) +
            ". Calculate the final result by applying each operation in the specified order."
        )
        prompt = (
            f"Please solve this step-by-step mathematical problem. "
            f"Show intermediate calculations for clarity. "
            f"Verify your work at each step. "
            f"Put the final result inside \\boxed{{}}: {task}"
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return prompt, operations


def calculate_chain_answer(initial_value: int, operations: list) -> int:
    """Calculate the expected result of the arithmetic chain."""
    result = initial_value
    for operation in operations:
        parts = operation.split()
        if parts[0] == "add":
            result += int(parts[1])
        elif parts[0] == "subtract":
            result -= int(parts[1])
        elif parts[0] == "multiply" and parts[1] == "by":
            result *= int(parts[2])
    return result


def generate_chain_instance(
    initial_value: int,
    depth: int,
    condition: str,
    instance_id: str,
    rng: SeededRNG = None
) -> Tuple[str, int, dict]:
    """
    Generate a single chain instance with metadata.

    Returns:
        (prompt, expected_answer, metadata)
    """
    prompt, operations = generate_chain_task(initial_value, depth, condition, rng)
    expected = calculate_chain_answer(initial_value, operations)

    metadata = {
        "initial_value": initial_value,
        "depth": depth,
        "condition": condition,
        "instance_id": instance_id,
        "task_type": "arithmetic_chain"
    }

    return prompt, expected, metadata