import argparse, time, json, concurrent.futures, requests, random

PROMPTS = [
    "Summarize why batching improves LLM throughput (2 sentences).",
    "Name three GPU memory fragmentation causes.",
    "Explain KV cache in one paragraph.",
    "Write a haiku about inference servers.",
    "List four vLLM flags that affect memory.",
    "Why does contiguous VRAM matter at startup?",
    "Give two pros/cons of CUDA graphs.",
    "Explain top-p vs temperature in 3 lines."
]

def one_call(host, model, max_tokens):
    payload = {
        "model": model,
        "prompt": random.choice(PROMPTS),
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    t0 = time.time()
    r = requests.post(f"{host}/v1/completions", json=payload, timeout=60)
    dt = time.time() - t0
    r.raise_for_status()
    out = r.json()
    toks = out.get("usage", {}).get("completion_tokens") or max_tokens
    return toks, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_call, args.host, args.model, args.max_tokens)
                for _ in range(args.concurrency)]
        results = [f.result() for f in futs]
    total_toks = sum(t for t, _ in results)
    wall = time.time() - t0
    tps = total_toks / wall if wall > 0 else 0.0

    print(json.dumps({
        "concurrency": args.concurrency,
        "total_completion_tokens": total_toks,
        "wall_seconds": round(wall, 3),
        "approx_tokens_per_second": round(tps, 1)
    }, indent=2))

if __name__ == "__main__":
    main()