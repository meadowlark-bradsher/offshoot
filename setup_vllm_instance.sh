#!/bin/bash
# Setup script for vLLM performance testing on new GPU instance

echo "Setting up vLLM optimization test on 146.235.234.214"
echo "=================================================="

# Upload the code to the new instance
echo "1. Uploading code to instance..."
rsync -avz --exclude='results/' --exclude='__pycache__/' \
    ./ ubuntu@146.235.234.214:~/offshoot_vllm/

# Connect and run setup
ssh ubuntu@146.235.234.214 << 'EOF'
    echo "2. Setting up environment..."
    cd ~/offshoot_vllm

    # Install vLLM and dependencies
    echo "Installing vLLM..."
    pip install vllm
    pip install transformers torch

    # Check GPU availability
    echo "3. Checking GPU status..."
    nvidia-smi
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

    echo "4. Ready to run performance tests!"
    echo "Next steps:"
    echo "  - Run: python test_vllm_performance.py"
    echo "  - Compare current vs vLLM performance"
    echo "  - Scale up to 1500+ instances"
EOF

echo "Setup complete! Ready to test vLLM performance."