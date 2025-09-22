"""Process-isolated vLLM manager to avoid initialization conflicts."""

import multiprocessing as mp
import queue
import time
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Single generation result with timing info."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float


def vllm_worker_process(
    model_name: str,
    config: Dict[str, Any],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    error_queue: mp.Queue
):
    """Worker process for vLLM inference."""
    try:
        # Set environment variables in the worker process
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Import vLLM in the isolated process
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[Worker] Loading {model_name} with vLLM...")
        start_time = time.time()

        # Initialize vLLM with conservative settings
        llm = LLM(
            model=model_name,
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 2048),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.6),
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_time = time.time() - start_time
        print(f"[Worker] Model loaded in {load_time:.2f} seconds")

        # Signal ready
        output_queue.put(("ready", load_time))

        # Process requests
        while True:
            try:
                # Get request with timeout
                try:
                    request = input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if request is None:  # Shutdown signal
                    break

                request_id, task_type, data = request

                if task_type == "generate_batch":
                    prompts = data["prompts"]
                    max_new_tokens = data.get("max_new_tokens", 256)
                    temperature = data.get("temperature", 0.0)
                    top_p = data.get("top_p", 1.0)

                    # Count input tokens
                    prompt_token_counts = [
                        len(tokenizer.encode(prompt))
                        for prompt in prompts
                    ]

                    # Create sampling parameters
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                    )

                    # Generate
                    start_time = time.time()
                    outputs = llm.generate(prompts, sampling_params)
                    generation_time = time.time() - start_time

                    # Process results
                    results = []
                    for i, output in enumerate(outputs):
                        completion = output.outputs[0].text
                        completion_tokens = len(output.outputs[0].token_ids)

                        result = GenerationResult(
                            text=prompts[i] + completion,
                            prompt_tokens=prompt_token_counts[i],
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_token_counts[i] + completion_tokens,
                            generation_time=generation_time / len(prompts)
                        )
                        results.append(result)

                    output_queue.put((request_id, "success", results))

                elif task_type == "count_tokens":
                    text = data["text"]
                    token_count = len(tokenizer.encode(text))
                    output_queue.put((request_id, "success", token_count))

                else:
                    output_queue.put((request_id, "error", f"Unknown task type: {task_type}"))

            except Exception as e:
                error_queue.put(f"[Worker] Error processing request: {e}")
                output_queue.put((request_id, "error", str(e)))

    except Exception as e:
        error_queue.put(f"[Worker] Fatal error: {e}")
        import traceback
        error_queue.put(traceback.format_exc())


class VLLMIsolatedManager:
    """vLLM manager that runs in an isolated process."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.6,
        **kwargs
    ):
        self.model_name = model_name
        self.config = {
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            **kwargs
        }

        self.worker_process = None
        self.input_queue = None
        self.output_queue = None
        self.error_queue = None
        self.request_id = 0
        self.is_ready = False

        self._start_worker()

    def _start_worker(self) -> None:
        """Start the worker process."""
        # Create queues
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.error_queue = mp.Queue()

        # Start worker process
        self.worker_process = mp.Process(
            target=vllm_worker_process,
            args=(
                self.model_name,
                self.config,
                self.input_queue,
                self.output_queue,
                self.error_queue
            )
        )
        self.worker_process.start()

        # Wait for ready signal
        print("Waiting for vLLM worker to initialize...")
        timeout = 120  # 2 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                message = self.output_queue.get(timeout=1.0)
                if message[0] == "ready":
                    load_time = message[1]
                    print(f"vLLM worker ready in {load_time:.2f} seconds")
                    self.is_ready = True
                    return
            except queue.Empty:
                # Check for errors
                try:
                    error = self.error_queue.get_nowait()
                    raise RuntimeError(f"Worker initialization failed: {error}")
                except queue.Empty:
                    continue

        raise TimeoutError("vLLM worker did not initialize within timeout")

    def _send_request(self, task_type: str, data: Dict[str, Any]) -> Any:
        """Send a request to the worker and get response."""
        if not self.is_ready:
            raise RuntimeError("Worker is not ready")

        request_id = self.request_id
        self.request_id += 1

        # Send request
        self.input_queue.put((request_id, task_type, data))

        # Wait for response
        timeout = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.output_queue.get(timeout=1.0)
                resp_id, status, result = response

                if resp_id == request_id:
                    if status == "success":
                        return result
                    else:
                        raise RuntimeError(f"Worker error: {result}")
                else:
                    # Put back response for another request
                    self.output_queue.put(response)
                    continue

            except queue.Empty:
                # Check for errors
                try:
                    error = self.error_queue.get_nowait()
                    raise RuntimeError(f"Worker error: {error}")
                except queue.Empty:
                    continue

        raise TimeoutError("Request timed out")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs
    ) -> List[GenerationResult]:
        """Generate completions for a batch of prompts."""
        data = {
            "prompts": prompts,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **generation_kwargs
        }
        return self._send_request("generate_batch", data)

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Generate completion for a single prompt."""
        results = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **generation_kwargs
        )

        result = results[0]
        generation_info = {
            "prompt_length": result.prompt_tokens,
            "total_length": result.total_tokens,
            "new_tokens": result.completion_tokens,
            "generation_time": result.generation_time,
            "temperature": temperature,
        }

        return result.text, result.completion_tokens, generation_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        data = {"text": text}
        return self._send_request("count_tokens", data)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_name": self.model_name,
            "backend": "vllm_isolated",
            **self.config
        }

    def cleanup(self) -> None:
        """Clean up worker process."""
        if self.worker_process and self.worker_process.is_alive():
            print("Shutting down vLLM worker...")
            # Send shutdown signal
            self.input_queue.put(None)
            # Wait for graceful shutdown
            self.worker_process.join(timeout=5)
            if self.worker_process.is_alive():
                self.worker_process.terminate()
                self.worker_process.join()
            print("vLLM worker shutdown complete")

    def __del__(self):
        """Destructor cleanup."""
        self.cleanup()


class VLLMIsolatedGenerationRunner:
    """Generation runner compatible with existing API but using isolated vLLM."""

    def __init__(self, model_name: str, **vllm_kwargs):
        self.vllm_manager = VLLMIsolatedManager(model_name, **vllm_kwargs)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Generate text (compatible with existing GenerationRunner API)."""
        return self.vllm_manager.generate_single(
            prompt, max_new_tokens, temperature, **generation_kwargs
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.vllm_manager.count_tokens(text)

    @property
    def llm_manager(self):
        """Compatibility property for existing code."""
        return self.vllm_manager