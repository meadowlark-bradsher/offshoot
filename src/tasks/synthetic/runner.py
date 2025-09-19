"""Synthetic task runner based on reference notebook survival experiment."""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .gen_arith import generate_chain_instance, calculate_chain_answer
from ...modeling.run import GenerationRunner
from ...scoring.synthetic import score_synthetic_response, determine_failure_reason
from ...common.rng import SeededRNG
from ...common.logging import setup_logging


class SyntheticRunner:
    """Runs synthetic survival experiments."""

    def __init__(
        self,
        generation_runner: GenerationRunner,
        config: Dict[str, Any],
        logger = None
    ):
        self.generation_runner = generation_runner
        self.config = config
        self.logger = logger or setup_logging()

    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run synthetic survival experiment.

        Returns list of experimental records in the common schema.
        """
        data_config = self.config["data"]
        model_config = self.config["model"]

        max_depth = data_config["max_depth"]
        n_instances = data_config["n_instances"]
        condition = data_config["condition"]
        initial_value = data_config.get("initial_value", 1)

        results = []

        self.logger.info(f"Starting synthetic experiment: {condition}")
        self.logger.info(f"Max depth: {max_depth}, Instances: {n_instances}")

        for run_idx in tqdm(range(n_instances), desc="Instances"):
            rng = SeededRNG(self.config["seed"] + run_idx)
            instance_id = str(uuid.uuid4())[:8]

            for depth in range(1, max_depth + 1):
                prompt, expected, metadata = generate_chain_instance(
                    initial_value=initial_value,
                    depth=depth,
                    condition=condition,
                    instance_id=instance_id,
                    rng=rng
                )

                tokens_in = self.generation_runner.count_tokens(prompt)

                try:
                    response, tokens_out, gen_info = self.generation_runner.generate(
                        prompt=prompt,
                        max_new_tokens=model_config.get("max_new_tokens", 256),
                        temperature=model_config.get("temperature", 0.0)
                    )

                    scores = score_synthetic_response(response, expected, metadata)

                    failure_reason = determine_failure_reason(scores)

                    record = {
                        "task_family": "synthetic",
                        "condition": condition,
                        "seed": self.config["seed"] + run_idx,
                        "instance_id": instance_id,
                        "depth_step": depth,
                        "depth_max": max_depth,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "tokens_total": tokens_in + tokens_out,
                        "answer": scores.get("predicted_boxed"),
                        "gold": expected,
                        "correct": scores["exact_match"],
                        "failure_reason": failure_reason,
                        "time_iso": datetime.utcnow().isoformat() + "Z",
                        "model": self.generation_runner.llm_manager.model_name,
                        "model_rev": "hf-default",
                        "raw_response": response.replace("\n", " ").strip(),
                        "used_code": scores["used_code"],
                        "value_present": scores["value_present"],
                    }

                    results.append(record)

                    # Only break on incorrect answers, not token limits
                    if not scores["exact_match"]:
                        break

                except Exception as e:
                    self.logger.error(f"Error processing depth {depth}: {e}")
                    break

        self.logger.info(f"Completed experiment with {len(results)} records")
        return results