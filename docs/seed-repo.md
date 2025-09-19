Absolutely—let’s shape the repo so it cleanly hosts **all Paper-1 experiments** (synthetic, software, and dialog), while keeping runners, survival models, plotting, and configs shared.

---

# Repo layout (Paper-1, end-to-end)

```
paper1-survival/
├── README.md
├── LICENSE
├── pyproject.toml                 # or setup.cfg/setup.py
├── requirements.txt               # core: hf transformers, torch, lifelines, pandas, pyyaml, matplotlib, tqdm
├── requirements-dev.txt           # dev: pytest, ruff, black, mypy (optional)
├── Makefile                       # one-liners for reproducibility
├── .github/workflows/ci.yml       # pytest + ruff + black
├── docker/
│   └── Dockerfile                 # pinned env for full reproducibility
├── data/                          # (empty; .gitkeep) small fixtures only; bigger via DVC if desired
│   ├── fixtures/                  # tiny examples for tests/docs
│   └── external/                  # optional: downloaded datasets (ignored by git)
├── configs/
│   ├── global.yaml                # seeds, device, output dirs, logging verbosity
│   ├── synthetic/
│   │   ├── terse.yaml
│   │   ├── verbose.yaml
│   │   └── redundant.yaml
│   ├── software/
│   │   ├── humaneval_incremental.yaml
│   │   └── swebench_threaded.yaml
│   └── dialog/
│       ├── qa_clean.yaml
│       └── qa_noisy.yaml
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── io.py                  # load/save, hashed filenames, parquet/csv/jsonl helpers
│   │   ├── config.py              # OmegaConf/YAML loader + overrides
│   │   ├── logging.py
│   │   ├── tokenize.py            # token counting (HF tokenizer)
│   │   └── rng.py                 # seeded RNG utilities
│   ├── modeling/
│   │   ├── llm.py                 # model/ tokenizer loader (GPT-2 by default)
│   │   └── run.py                 # unified generation API with streaming capture of tokens
│   ├── scoring/
│   │   ├── synthetic.py           # exact match vs gold for composition tasks
│   │   ├── software.py            # unit-tests/spec checks → pass/fail
│   │   └── dialog.py              # gold QA (EM/F1) or LLM-as-judge adapter (optional)
│   ├── survival/
│   │   ├── prepare.py             # build survival-ready DataFrames from raw logs
│   │   ├── km.py                  # Kaplan–Meier utilities (steps vs tokens)
│   │   ├── cox.py                 # Cox/AFT helpers
│   │   └── plots.py               # plotting helpers (overlays, CI ribbons)
│   ├── tasks/
│   │   ├── synthetic/
│   │   │   ├── gen_pointer.py     # pointer-chasing generator
│   │   │   ├── gen_arith.py       # nested arithmetic/boolean
│   │   │   └── runner.py          # emits rows: {depth, tokens_in, tokens_out, correct, condition}
│   │   ├── software/
│   │   │   ├── adapters/
│   │   │   │   ├── humaneval.py   # incremental variants with staged specs/tests
│   │   │   │   └── sweb.py        # (optional) SWE-bench-style issue→patch threads
│   │   │   └── runner.py
│   │   └── dialog/
│   │       ├── adapters/
│   │       │   ├── qa_gold.py     # NQ/HotpotQA/GSM8K-style with turn chaining
│   │       │   └── probes.py      # entity-memory probes, contradiction checks
│   │       └── runner.py
│   ├── pipeline/
│   │   ├── run_synthetic.py
│   │   ├── run_software.py
│   │   └── run_dialog.py
│   └── reporting/
│       ├── tables.py              # summary stats → CSV/Markdown
│       └── figures.py             # paper-ready figures (steps vs tokens overlays)
├── results/
│   ├── synthetic/
│   │   ├── raw/                   # per-seed JSONL logs (one row per instance)
│   │   ├── processed/             # survival-ready Parquet/CSV
│   │   └── figures/
│   ├── software/
│   │   ├── raw/ processed/ figures/
│   └── dialog/
│       ├── raw/ processed/ figures/
└── tests/
    ├── test_generators.py
    ├── test_tokenize.py
    ├── test_scoring.py
    └── test_survival.py
```

---

## What’s different vs “synthetic-only”

* Three **task families** live under `src/tasks/…` with their own runners and adapters.
* A single **modeling** layer (`modeling/llm.py`, `modeling/run.py`) so you can swap GPT-2 ↔ other models without touching task code.
* One **survival** package reused everywhere (KM by steps/tokens; Cox/AFT). Fine–Gray is saved for Paper-2, but you can park scaffolding later under `survival/competing.py` if you want the import path stable now.
* **Configs** split by track (synthetic/software/dialog) so experiments are reproducible and comparable.
* **reporting/** centralizes paper figures/tables so every pipeline writes to the same shapes.

---

## Data contracts (so everything clicks)

### Raw log schema (JSONL; one row per attempt)

```
{
  "task_family": "synthetic|software|dialog",
  "condition": "terse|verbose|redundant|…",
  "seed": 123,
  "instance_id": "uuid",
  "depth_step": 17,                 # composition step index (or turn index)
  "depth_max": 50,
  "tokens_in": 83,                  # prompt/context tokens consumed
  "tokens_out": 12,                 # generated tokens until stop
  "tokens_total": 95,               # in+out
  "answer": "...",
  "gold": "...",                    # when applicable
  "correct": true|false|null,       # null if judged later
  "failure_reason": "first_error|test_fail|hallucination|none",
  "time_iso": "2025-09-17T18:05:00Z",
  "model": "gpt2",
  "model_rev": "hf-tag"
}
```

### Survival-ready table (CSV/Parquet)

Columns (minimal):

* `time_var` ∈ {`step`, `tokens_total`}
* `event` ∈ {0,1}  (first failure occurred by time\_var)
* `condition`, `task_family`, `seed`, `model`

---

## Makefile (repro-friendly)

```Makefile
setup:
\tpip install -r requirements.txt

# Synthetic
synthetic-terse:
\tpython -m src.pipeline.run_synthetic --config configs/synthetic/terse.yaml
synthetic-verbose:
\tpython -m src.pipeline.run_synthetic --config configs/synthetic/verbose.yaml

# Software
software-humaneval:
\tpython -m src.pipeline.run_software --config configs/software/humaneval_incremental.yaml

# Dialog
dialog-qa-clean:
\tpython -m src.pipeline.run_dialog --config configs/dialog/qa_clean.yaml

# Figures
figures:
\tpython -m src.reporting.figures
```

---

## Config shape (example)

```yaml
# configs/synthetic/terse.yaml
experiment_name: "synthetic_terse_L50"
seed: 123
model:
  name: "gpt2"
  max_new_tokens: 32
  temperature: 0.0
data:
  generator: "pointer"        # or "arith"
  n_instances: 2000
  max_depth: 50
  condition: "terse"          # "verbose" | "redundant"
logging:
  out_dir: "results/synthetic"
analysis:
  survival:
    time_axes: ["step", "tokens_total"]
    curves: ["km", "weibull"]
```

---

## How each track runs

* **Synthetic**
  `run_synthetic.py` → generate chains → run GPT-2 → score exactness → build two survival views:

  * Steps as time (pure depth)
  * Tokens as time (no redundancy benefit expected)

* **Software**
  `run_software.py` → load incremental variants (e.g., HumanEval staged specs) → run → execute tests → first failing stage = event → survival by steps & tokens (expect **tokens** to buy some robustness due to clarifications/tests).

* **Dialog**
  `run_dialog.py` → chain QA turns with clean/noisy conditions → score via EM/F1 (or optional judge) → mark first degradation → survival by steps & tokens (expect redundancy to help).

---

## Testing hooks

* `test_generators.py` verifies depth-L gold correctness.
* `test_scoring.py` checks exact match and unit-test harness.
* `test_survival.py` asserts KM shapes on tiny toy sets (no plotting).

---

## Why this scales to Paper-2 later

* You’ll be able to add `src/survival/competing.py` (Fine–Gray) and a `pipeline/run_20q.py` without moving anything. Results drop into `results/dialog/…` (or `results/20q/` if you want a new root).

---

If you want, I can drop in **starter files** (empty modules with docstrings + a minimal `run_synthetic.py`) so you can ‘pip install -e .’ and run a toy experiment immediately.
