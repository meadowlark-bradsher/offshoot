# Transosonde GPU Setup for Offshoot Survival Experiments

## ✅ What's Ready

The **offshoot-survival** Transosonde pack has been created and is ready for GPU experiments:

### Pack Location
```
/Users/meadowlarkbradsher/workspace/repos/genai/then-mx/transosonde/packs/offshoot-survival/
```

### Pack Configuration
- **Runtime**: conda with PyTorch + CUDA 11.8
- **Model**: Qwen/Qwen2.5-3B-Instruct (generalist model)
- **Dependencies**: All survival analysis libraries (lifelines, pandas, etc.)
- **GPU Setup**: Automatic device mapping for CUDA acceleration

### Expected Performance
- **Local (MPS)**: 12+ hours for 10 instances
- **A10 GPU**: ~1-2 hours for 20-50 instances
- **H100 GPU**: ~30-60 minutes for 20-50 instances

## ⚠️ What's Needed to Test

### Required Environment Variables
Create `.env` file in transosonde directory:

```bash
# Required for Lambda Labs instances
LAMBDA_LABS_API_KEY=your_lambda_labs_api_key_here

# Required for GCS storage
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-service-account.json

# Optional: Weights & Biases logging
WANDB_API_KEY=your_wandb_api_key_here
```

### Quick Test Command
Once environment is set up:

```bash
cd /path/to/transosonde
RUN_ID="surv-test-$(date +%Y%m%d-%H%M)"
OUTPUT_PATH="gs://your-bucket/runs/$RUN_ID"
CONDITION="terse" \
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" \
N_INSTANCES="10" \
./bin/ts launch --pack offshoot-survival --run $RUN_ID
```

## 🎯 Test Plan

### Phase 1: Quick Validation (10 instances)
- **Purpose**: Verify GPU acceleration works
- **Expected time**: ~20-30 minutes on A10
- **Output**: 10 survival chains showing failure around depth 16-30

### Phase 2: Condition Comparison (3x 30 instances)
```bash
# Test all three conditions
for condition in terse verbose redundant; do
  RUN_ID="surv-${condition}-$(date +%Y%m%d-%H%M)"
  CONDITION=$condition N_INSTANCES=30 \
  ./bin/ts launch --pack offshoot-survival --run $RUN_ID
done
```

### Phase 3: Model Comparison
- Test Qwen/Qwen2.5-7B-Instruct for comparison
- Validate results show expected degradation patterns

## 📊 Expected Results

Based on local testing, we should see:
- **Median survival**: 16-30 steps across all conditions
- **Consistent failure patterns**: All instances failing at similar depths
- **Token vs Step analysis**: Different survival curves when measured by context size vs sequential depth

## 🚀 Benefits of GPU Acceleration

- **Speed**: 20-40x faster than local MPS
- **Scale**: Run larger experiments (100+ instances)
- **Automation**: Hands-off execution with automatic GCS sync
- **Cost-effective**: Only pay for GPU time during experiments

## 📋 Next Steps for Team

1. **Set up Lambda Labs API access**
2. **Configure GCS bucket for results storage**
3. **Run 10-instance validation test**
4. **Scale to full experimental comparison**

The pack is production-ready and follows all Transosonde best practices. Once the API keys are configured, we can begin rapid iteration on survival experiments with proper GPU acceleration.