# -------- Settings --------
PYTHON ?= python3
PORT   ?= 8000
MODEL  ?= Qwen/Qwen2.5-7B-Instruct

# Toggle vLLM engine (0 = v0 / 1 = v1). Start with v0 for stability.
VLLM_USE_V1 ?= 0

# GPU memory fraction (prod); keep margin for contiguous allocs
GPU_MEM_UTIL ?= 0.72
SWAP_SPACE   ?= 16
MAX_LEN      ?= 2048
BATCH_TOKENS ?= 4096

# -------- macOS DEV (CPU) --------
dev.create-env:
	conda create -n vllm-mac-dev python=3.11 -y

dev.activate:
	@echo "Run: source $$(conda info --base)/etc/profile.d/conda.sh && conda activate vllm-mac-dev"

dev.install:  ## CPU-only validation on macOS (no Metal)
	$(PYTHON) -m pip install -U pip
	# pins that vLLM expects; CPU torch
	pip install -r requirements-dev-mac.txt -c constraints-dev-mac.txt
	# install vLLM WITHOUT deps to dodge outlines->pyairports
	pip install --no-deps vllm==0.10.2

dev.serve:
	export VLLM_USE_V1=$(VLLM_USE_V1); \
	vllm serve $(MODEL) --port $(PORT) \
	  --max-model-len $(MAX_LEN) \
	  --max-num-batched-tokens $(BATCH_TOKENS) \
	  --enforce-eager --disable-log-stats --swap-space $(SWAP_SPACE)

dev.test:
	curl -s http://localhost:$(PORT)/v1/completions \
	  -H "Content-Type: application/json" \
	  -d '{"model":"$(MODEL)","prompt":"Say hello in 10 words.","max_tokens":32,"temperature":0}' | jq .

# -------- LINUX CUDA PROD --------
prod.create-venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip

prod.install:
	. .venv/bin/activate && \
	pip install -r requirements-prod-linux.txt -c constraints-prod-linux.txt

prod.serve:
	. .venv/bin/activate && \
	export VLLM_USE_V1=$(VLLM_USE_V1); \
	vllm serve $(MODEL) --port $(PORT) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL) \
	  --max-model-len $(MAX_LEN) \
	  --max-num-batched-tokens $(BATCH_TOKENS) \
	  --enforce-eager --swap-space $(SWAP_SPACE)

prod.test:
	curl -s http://localhost:$(PORT)/v1/completions \
	  -H "Content-Type: application/json" \
	  -d '{"model":"$(MODEL)","prompt":"Two bullet points about GPUs.","max_tokens":64,"temperature":0}' | jq .

# -------- Bench (both envs; requires Python client) --------
bench:
	. .venv/bin/activate 2>/dev/null || true; \
	$(PYTHON) bench_client.py --host http://localhost:$(PORT) --concurrency 8 --max-tokens 200

# -------- Utilities --------
clean:
	rm -rf .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: dev.create-env dev.activate dev.install dev.serve dev.test prod.create-venv prod.install prod.serve prod.test bench clean help