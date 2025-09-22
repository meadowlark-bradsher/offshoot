"""vLLM server-based manager for robust high-performance inference."""

import time
import subprocess
import requests
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import threading
import atexit
import signal
import os


@dataclass
class GenerationResult:
    """Single generation result with timing info."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float


class VLLMServerManager:
    """vLLM manager using server mode for stability and performance."""

    def __init__(
        self,
        model_name: str,
        host: str = "localhost",
        port: int = 8000,
        gpu_memory_utilization: float = 0.72,
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        auto_start: bool = True,
        **server_kwargs
    ):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.server_kwargs = server_kwargs

        self.server_process = None
        self.base_url = f"http://{host}:{port}"
        self.tokenizer = None

        if auto_start:
            self.start_server()

        # Register cleanup
        atexit.register(self.cleanup)

    def start_server(self) -> None:
        """Start vLLM server with conservative settings."""
        if self.is_server_running():
            print(f"vLLM server already running on {self.base_url}")
            return

        print(f"Starting vLLM server for {self.model_name}...")

        # Conservative server command following runbook
        cmd = [
            "vllm", "serve", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--enforce-eager",  # Disable CUDA graphs for stability
            "--disable-log-requests",  # Reduce logging overhead
        ]

        # Add any additional server kwargs
        for key, value in self.server_kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Set environment variables for stability
        env = os.environ.copy()
        env["VLLM_USE_V1"] = "0"  # Use legacy engine
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU for now

        print(f"Starting server with command: {' '.join(cmd)}")

        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )

        # Wait for server to be ready
        print("Waiting for server to start...")
        start_time = time.time()
        timeout = 120  # 2 minutes timeout

        while time.time() - start_time < timeout:
            if self.is_server_running():
                ready_time = time.time() - start_time
                print(f"vLLM server ready in {ready_time:.1f} seconds")
                self._load_tokenizer()
                return
            time.sleep(2)

        # Check if process died
        if self.server_process.poll() is not None:
            stdout, stderr = self.server_process.communicate()
            raise RuntimeError(f"vLLM server failed to start. Stderr: {stderr.decode()}")

        raise TimeoutError(f"vLLM server did not start within {timeout} seconds")

    def is_server_running(self) -> bool:
        """Check if vLLM server is running and responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _load_tokenizer(self) -> None:
        """Load tokenizer separately for token counting."""
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs
    ) -> List[GenerationResult]:
        """Generate completions for a batch of prompts using OpenAI-compatible API."""
        if not self.is_server_running():
            raise RuntimeError("vLLM server is not running")

        # Count input tokens if tokenizer available
        prompt_token_counts = []
        if self.tokenizer:
            prompt_token_counts = [
                len(self.tokenizer.encode(prompt))
                for prompt in prompts
            ]
        else:
            # Rough estimate if no tokenizer
            prompt_token_counts = [len(prompt.split()) * 1.3 for prompt in prompts]

        # Prepare batch request using OpenAI-compatible completions API
        start_time = time.time()

        # Process batch using individual requests (can be optimized later)
        results = []
        for i, prompt in enumerate(prompts):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    **generation_kwargs
                }

                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                choice = data["choices"][0]
                completion = choice["text"]

                # Extract token counts from usage if available
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", len(completion.split()) * 1.3)

                result = GenerationResult(
                    text=prompt + completion,
                    prompt_tokens=int(prompt_token_counts[i]),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(prompt_token_counts[i] + completion_tokens),
                    generation_time=(time.time() - start_time) / (i + 1)  # Average so far
                )
                results.append(result)

            except Exception as e:
                print(f"Error generating for prompt {i}: {e}")
                # Create dummy result to maintain batch size
                result = GenerationResult(
                    text=prompt + f" [ERROR: {str(e)}]",
                    prompt_tokens=int(prompt_token_counts[i]),
                    completion_tokens=0,
                    total_tokens=int(prompt_token_counts[i]),
                    generation_time=0.0
                )
                results.append(result)

        total_time = time.time() - start_time
        # Update generation times with actual total
        for result in results:
            result.generation_time = total_time / len(prompts)

        return results

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **generation_kwargs
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Generate completion for a single prompt (compatible with existing API)."""
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
        if self.tokenizer is None:
            self._load_tokenizer()

        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate
            return int(len(text.split()) * 1.3)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_name": self.model_name,
            "backend": "vllm_server",
            "server_url": self.base_url,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }

    def cleanup(self) -> None:
        """Clean up server process."""
        if self.server_process and self.server_process.poll() is None:
            print("Shutting down vLLM server...")
            try:
                # Try graceful shutdown first
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=10)
            except:
                # Force kill if needed
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except:
                    pass
            print("vLLM server shutdown complete")

    def __del__(self):
        """Destructor cleanup."""
        self.cleanup()


class VLLMServerGenerationRunner:
    """Generation runner compatible with existing API but using vLLM server."""

    def __init__(self, model_name: str, **server_kwargs):
        self.vllm_manager = VLLMServerManager(model_name, **server_kwargs)
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