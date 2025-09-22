# 🔧 vLLM Infrastructure Setup Guide

**For Transosonde DevOps/ML Engineering Teams**

This guide provides step-by-step instructions for deploying production-ready vLLM infrastructure that delivers 10-20x LLM inference performance improvements.

---

## 🚀 Quick Start (Docker Method)

### Prerequisites
- Ubuntu 22.04 server with NVIDIA GPU
- NVIDIA drivers >= 525.60.13
- Docker with NVIDIA container toolkit

### One-Command Deployment
```bash
#!/bin/bash
# deploy-vllm-quick.sh

# Install dependencies
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update && apt install -y nvidia-container-toolkit
systemctl restart docker

# Deploy vLLM service
docker run -d \
  --name vllm-qwen-7b \
  --gpus all \
  -p 8000:8000 \
  -e VLLM_USE_V1=0 \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --disable-log-stats

# Test deployment
sleep 30
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"Hello, world!","max_tokens":50}'
```

---

## 🏗️ Production Deployment Options

### Option 1: Multi-Model Docker Compose

Create `docker-compose.vllm.yml`:
```yaml
version: '3.8'

services:
  # NGINX load balancer and model router
  vllm-router:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - vllm-qwen-7b
      - vllm-llama-7b
    restart: unless-stopped

  # Qwen model service
  vllm-qwen-7b:
    image: vllm/vllm-openai:latest
    environment:
      - VLLM_USE_V1=0
      - CUDA_VISIBLE_DEVICES=0
    command: >
      vllm serve Qwen/Qwen2.5-7B-Instruct
      --host 0.0.0.0 --port 8000
      --gpu-memory-utilization 0.7
      --max-model-len 4096
      --enforce-eager
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Llama model service (if multiple GPUs available)
  vllm-llama-7b:
    image: vllm/vllm-openai:latest
    environment:
      - VLLM_USE_V1=0
      - CUDA_VISIBLE_DEVICES=1
    command: >
      vllm serve meta-llama/Llama-2-7b-chat-hf
      --host 0.0.0.0 --port 8000
      --gpu-memory-utilization 0.7
      --max-model-len 4096
      --enforce-eager
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

NGINX configuration (`nginx.conf`):
```nginx
upstream vllm_qwen {
    server vllm-qwen-7b:8000;
}

upstream vllm_llama {
    server vllm-llama-7b:8000;
}

server {
    listen 80;
    server_name vllm.transosonde.internal;

    # Route based on model parameter
    location /v1/completions {
        # Default to Qwen
        set $backend "vllm_qwen";

        # Route to Llama if specified
        if ($request_body ~ "llama") {
            set $backend "vllm_llama";
        }

        proxy_pass http://$backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://vllm_qwen/health;
    }
}
```

### Option 2: Kubernetes Deployment

Create `vllm-k8s-manifest.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: ml-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-a100
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        env:
        - name: VLLM_USE_V1
          value: "0"
        command:
        - vllm
        - serve
        - "Qwen/Qwen2.5-7B-Instruct"
        - --host=0.0.0.0
        - --port=8000
        - --gpu-memory-utilization=0.8
        - --enforce-eager
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ml-inference
spec:
  selector:
    app: vllm-inference
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: ml-inference
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  rules:
  - host: vllm.transosonde.internal
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 80
```

Deploy with:
```bash
kubectl create namespace ml-inference
kubectl apply -f vllm-k8s-manifest.yaml
```

---

## 📊 Monitoring & Observability

### Prometheus Configuration (`prometheus.yml`)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-qwen-7b:8000', 'vllm-llama-7b:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'gpu'
    static_configs:
      - targets: ['nvidia-dcgm-exporter:9400']
```

### Grafana Dashboard JSON
```json
{
  "dashboard": {
    "title": "vLLM Performance Dashboard",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [
          {
            "expr": "rate(vllm_request_total[5m])",
            "legendFormat": "{{instance}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL",
            "legendFormat": "GPU {{gpu}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Tokens per Second",
        "targets": [
          {
            "expr": "rate(vllm_token_total[5m])",
            "legendFormat": "{{instance}}"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

---

## 🔒 Security & Authentication

### API Gateway with Authentication
```yaml
# kong-vllm-config.yaml
services:
- name: vllm-service
  url: http://vllm-qwen-7b:8000
  routes:
  - name: vllm-route
    paths:
    - /v1
    methods:
    - POST
    - GET
    plugins:
    - name: key-auth
      config:
        key_names:
        - apikey
    - name: rate-limiting
      config:
        minute: 100
        hour: 1000
```

### SSL/TLS Termination
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout vllm.key -out vllm.crt \
  -subj "/CN=vllm.transosonde.internal"

# Update nginx.conf
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/vllm.crt;
    ssl_certificate_key /etc/nginx/ssl/vllm.key;

    # ... rest of config
}
```

---

## 📚 Client Libraries

### Python Client (`transosonde_vllm.py`)
```python
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import logging

class TransosondeVLLM:
    def __init__(
        self,
        base_url: str = "http://vllm.transosonde.internal",
        api_key: Optional[str] = None,
        timeout: int = 300
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def complete(
        self,
        prompt: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion for a single prompt."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        response = self.session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    async def batch_complete(
        self,
        prompts: List[str],
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Async batch completion for high throughput."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            tasks = []
            for prompt in prompts:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    **kwargs
                }
                task = self._async_request(session, payload)
                tasks.append(task)

            return await asyncio.gather(*tasks)

    async def _async_request(self, session, payload):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers=headers
        ) as response:
            response.raise_for_status()
            return await response.json()

    def health_check(self) -> bool:
        """Check if vLLM service is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False

# Usage example
if __name__ == "__main__":
    llm = TransosondeVLLM()

    # Test connection
    if llm.health_check():
        print("✅ vLLM service is healthy")

        # Single completion
        result = llm.complete("Explain quantum computing in one sentence.")
        print(f"Response: {result['choices'][0]['text']}")

        # Batch completion
        prompts = [
            "What is machine learning?",
            "Explain neural networks.",
            "Define artificial intelligence."
        ]

        import asyncio
        results = asyncio.run(llm.batch_complete(prompts))
        for i, result in enumerate(results):
            print(f"Prompt {i+1}: {result['choices'][0]['text']}")
    else:
        print("❌ vLLM service is not available")
```

### Node.js Client (`transosonde-vllm.js`)
```javascript
const axios = require('axios');

class TransosondeVLLM {
    constructor(baseUrl = 'http://vllm.transosonde.internal', apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 300000,
            headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
        });
    }

    async complete(prompt, options = {}) {
        const payload = {
            model: options.model || 'Qwen/Qwen2.5-7B-Instruct',
            prompt,
            max_tokens: options.maxTokens || 256,
            temperature: options.temperature || 0.0,
            ...options
        };

        const response = await this.client.post('/v1/completions', payload);
        return response.data;
    }

    async batchComplete(prompts, options = {}) {
        const requests = prompts.map(prompt => this.complete(prompt, options));
        return Promise.all(requests);
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health', { timeout: 10000 });
            return response.status === 200;
        } catch {
            return false;
        }
    }
}

module.exports = TransosondeVLLM;
```

---

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

#### 1. "CUDA out of memory" errors
```bash
# Check GPU memory usage
nvidia-smi

# Reduce memory utilization
docker exec vllm-container pkill -f vllm
docker run ... --gpu-memory-utilization 0.6 ...
```

#### 2. Slow inference after startup
```bash
# Check if model is still loading
docker logs vllm-qwen-7b

# Wait for "Model loaded successfully" message
# First few requests are slower due to CUDA warm-up
```

#### 3. Connection timeouts
```bash
# Increase timeout in nginx.conf
proxy_read_timeout 600s;
proxy_send_timeout 600s;

# Or in client code
requests.post(url, json=data, timeout=600)
```

#### 4. Model download failures
```bash
# Pre-download models
docker run --rm -v /opt/models:/models \
  huggingface/transformers-pytorch-gpu \
  python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='/models')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='/models')
"

# Mount cached models
docker run -v /opt/models:/root/.cache/huggingface ...
```

### Performance Optimization

#### GPU Memory Optimization
```bash
# For multiple small models
--gpu-memory-utilization 0.5 --swap-space 16

# For single large model
--gpu-memory-utilization 0.9 --swap-space 0

# Enable tensor parallelism for larger models
--tensor-parallel-size 2  # For 2 GPUs
```

#### Batch Size Tuning
```bash
# Increase batch size for higher throughput
--max-num-batched-tokens 8192

# Reduce for lower latency
--max-num-batched-tokens 2048
```

---

## 📈 Performance Benchmarking

### Load Testing Script
```python
#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import statistics

async def benchmark_vllm(base_url, num_concurrent=10, num_requests=100):
    prompts = [
        "Explain machine learning in one sentence.",
        "What is artificial intelligence?",
        "Describe neural networks briefly.",
        "Define deep learning.",
        "What is computer vision?"
    ]

    async def make_request(session, prompt):
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.0
        }

        start_time = time.time()
        async with session.post(f"{base_url}/v1/completions", json=payload) as response:
            await response.json()
            return time.time() - start_time

    # Warm up
    async with aiohttp.ClientSession() as session:
        await make_request(session, prompts[0])

    # Benchmark
    start_time = time.time()
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=num_concurrent)
    ) as session:
        semaphore = asyncio.Semaphore(num_concurrent)

        async def controlled_request():
            async with semaphore:
                prompt = prompts[len(tasks) % len(prompts)]
                return await make_request(session, prompt)

        tasks = [controlled_request() for _ in range(num_requests)]
        latencies = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    print(f"Benchmark Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Concurrent requests: {num_concurrent}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Requests/sec: {num_requests/total_time:.1f}")
    print(f"  Average latency: {statistics.mean(latencies):.2f}s")
    print(f"  P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}s")

if __name__ == "__main__":
    asyncio.run(benchmark_vllm("http://localhost:8000"))
```

---

## 🚀 Next Steps

1. **Choose deployment method** (Docker Compose recommended for start)
2. **Provision GPU servers** with adequate VRAM (24GB+ recommended)
3. **Deploy monitoring stack** for observability
4. **Create client libraries** for your programming languages
5. **Migrate pilot project** to validate performance improvements
6. **Scale up** based on usage patterns and performance metrics

**Expected Results**: 10-20x performance improvement over remote API calls, with typical throughput of 100-500+ tokens/sec depending on hardware.

For support, create an issue in the infrastructure repository or contact the ML platform team.