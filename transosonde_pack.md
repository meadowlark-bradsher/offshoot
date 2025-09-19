# Transosonde Pack for Offshoot Survival Experiments

This document outlines the Transosonde pack configuration for running LLM survival analysis experiments efficiently on GPU instances.

## Pack Structure

```
packs/offshoot-survival/
├── pack.yml                  # Pack configuration
├── env/
│   ├── conda.yaml           # Dependencies
│   └── requirements.txt     # Additional pip packages
├── hooks/
│   ├── pre_run.sh           # Setup script
│   ├── run.sh               # Main experiment runner
│   └── post_run.sh          # Results sync and cleanup
└── assets/
    └── configs/             # Experiment configurations
```

## pack.yml

```yaml
name: offshoot-survival
runtime: conda
entry: hooks/run.sh
pre: hooks/pre_run.sh
post: hooks/post_run.sh

env:
  conda: env/conda.yaml
  pip: env/requirements.txt

params:
  experiment_type: synthetic    # synthetic, software, dialog
  condition: terse             # terse, verbose, redundant
  model_name: Qwen/Qwen2.5-3B-Instruct
  n_instances: 20              # Number of survival instances
  max_depth: 30                # Safety ceiling for chains
  wandb_project: offshoot-survival
```

## env/conda.yaml

```yaml
name: transosonde-offshoot
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=11.8
  - pip
  - pip:
    - transformers>=4.30.0
    - torch>=2.0.0
    - pandas>=1.5.0
    - numpy>=1.20.0
    - matplotlib>=3.5.0
    - seaborn>=0.11.0
    - lifelines>=0.27.0
    - pyyaml>=6.0
    - tqdm>=4.64.0
    - omegaconf>=2.3.0
    - scikit-learn>=1.1.0
    - accelerate
    - wandb
    - google-cloud-storage
```

## hooks/pre_run.sh

```bash
#!/bin/bash
set -euo pipefail
echo "🔧 Setting up Offshoot survival experiments..."

# Activate environment
source "$(dirname "${BASH_SOURCE[0]}")/../.activate_env.sh"

# Auth with GCS
~/transosonde/storage/gcs/login.sh
~/transosonde/storage/gcs/verify.sh

# Create run directories
mkdir -p "~/transosonde/runs/$RUN_ID"/{logs,results,configs}

# Clone offshoot repository
cd ~/transosonde/runs/$RUN_ID
git clone https://github.com/user/offshoot.git .
pip install -e .

echo "✅ Pre-run setup completed"
```

## hooks/run.sh

```bash
#!/bin/bash
set -euo pipefail
echo "🚀 Starting Offshoot survival experiment..."

# Activate environment
source "$(dirname "${BASH_SOURCE[0]}")/../.activate_env.sh"

cd ~/transosonde/runs/$RUN_ID

# Set defaults
EXPERIMENT_TYPE=${EXPERIMENT_TYPE:-synthetic}
CONDITION=${CONDITION:-terse}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
N_INSTANCES=${N_INSTANCES:-20}
MAX_DEPTH=${MAX_DEPTH:-30}

# Create dynamic config
cat > config_dynamic.yaml << EOF
experiment_name: "${RUN_ID}_${CONDITION}"
seed: 42

model:
  name: "${MODEL_NAME}"
  max_new_tokens: 2048
  temperature: 0.0
  model_kwargs:
    device_map: "auto"

data:
  generator: "arith"
  n_instances: ${N_INSTANCES}
  max_depth: ${MAX_DEPTH}
  condition: "${CONDITION}"
  initial_value: 1

logging:
  out_dir: "results/${EXPERIMENT_TYPE}"
  level: "INFO"
EOF

echo "📊 Running survival experiment: ${CONDITION} with ${MODEL_NAME}"
echo "Instances: ${N_INSTANCES}, Max depth: ${MAX_DEPTH}"

# Run the experiment
python -m src.pipeline.run_synthetic --config config_dynamic.yaml

# Log completion
echo "experiment_completed_at=$(date -Iseconds)" >> "results/${EXPERIMENT_TYPE}/${RUN_ID}_${CONDITION}/DONE"
echo "total_records=$(wc -l < results/${EXPERIMENT_TYPE}/${RUN_ID}_${CONDITION}/raw/results.jsonl)" >> "results/${EXPERIMENT_TYPE}/${RUN_ID}_${CONDITION}/DONE"

echo "✅ Experiment completed successfully!"
```

## hooks/post_run.sh

```bash
#!/bin/bash
set -euo pipefail
echo "🧹 Running post-run cleanup and sync..."

cd ~/transosonde/runs/$RUN_ID

# Generate quick analysis
python -c "
import sys
sys.path.append('.')
from analyze_test_results import analyze_results

# Analyze the results
results_dir = f'results/{EXPERIMENT_TYPE}/{RUN_ID}_{CONDITION}'
analyze_results(results_dir)
" > results/analysis_summary.txt

# Sync everything to GCS
~/transosonde/storage/gcs/sync.sh "results/" "$OUTPUT_PATH"

echo "✅ Results synced to $OUTPUT_PATH"
echo "📊 Analysis summary saved"
```

## Usage Examples

### Single Experiment
```bash
# Run terse condition with Qwen 3B
RUN_ID=surv-$(date +%Y%m%d-%H%M) \
OUTPUT_PATH=gs://your-bucket/runs/surv-$(date +%Y%m%d-%H%M) \
CONDITION=terse \
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
N_INSTANCES=50 \
bin/ts launch --pack offshoot-survival --run $RUN_ID
```

### Batch Comparison
```bash
# Run all three conditions for comparison
for condition in terse verbose redundant; do
  RUN_ID=surv-${condition}-$(date +%Y%m%d-%H%M)
  OUTPUT_PATH=gs://your-bucket/runs/$RUN_ID
  CONDITION=$condition \
  N_INSTANCES=30 \
  bin/ts launch --pack offshoot-survival --run $RUN_ID
done
```

### GitHub Actions Workflow
```yaml
name: Offshoot Survival Experiment
on:
  workflow_dispatch:
    inputs:
      condition:
        description: 'Experiment condition'
        required: true
        default: 'terse'
        type: choice
        options:
        - terse
        - verbose
        - redundant
      model_name:
        description: 'Model to test'
        required: true
        default: 'Qwen/Qwen2.5-3B-Instruct'
      n_instances:
        description: 'Number of instances'
        required: true
        default: '20'

jobs:
  survival-experiment:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run Survival Experiment
      env:
        LAMBDA_LABS_API_KEY: ${{ secrets.LAMBDA_LABS_API_KEY }}
        GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GOOGLE_APPLICATION_CREDENTIALS }}
      run: |
        RUN_ID=surv-${{ github.event.inputs.condition }}-$(date +%Y%m%d-%H%M)
        OUTPUT_PATH=gs://your-bucket/survival-experiments/$RUN_ID

        CONDITION=${{ github.event.inputs.condition }} \
        MODEL_NAME=${{ github.event.inputs.model_name }} \
        N_INSTANCES=${{ github.event.inputs.n_instances }} \
        bin/ts launch --pack offshoot-survival --run $RUN_ID
```

## Expected Performance

- **Local (MPS)**: 12 hours for 10 instances
- **A10 GPU**: ~1-2 hours for 20-50 instances
- **H100 GPU**: ~30-60 minutes for 20-50 instances

This setup will allow rapid iteration on survival experiments with proper GPU acceleration and automated results management.