# vllm-serving-bench

How do batch size and context length affect LLM serving performance?
This project serves a small model with [vLLM](https://docs.vllm.ai/), sends it many requests at once, and measures **throughput** and **p50/p99 latency** while changing two things:

- **Concurrency** (how many requests are in flight at the same time ≈ batch size): 1 → 64
- **Context length** (prompt length in tokens): 128, 512, 2048

## Questions

1. How much does throughput grow as concurrency increases, and where does it stop growing?
2. How much slower does each user's generation get (TPOT) as the batch grows?
3. When does time to first token (TTFT) jump because requests start waiting in a queue?
4. How does a longer context change all of the above? (Longer prompts → bigger KV cache per request → fewer requests fit in GPU memory.)

**Expectation (from ORCA and vLLM):** decode is memory-bound, so batching should raise throughput almost for free, until compute or KV-cache memory runs out.

## Metrics

| Metric | Meaning |
|---|---|
| Output throughput | Generated tokens per second, summed over all requests |
| TTFT | Time to first token: how long a user waits before the answer starts |
| TPOT | Time per output token after the first one: how fast the answer streams |
| E2E latency | Total time for one request, start to finish |
| p50 / p99 | Median / 99th percentile: the typical request vs. the slowest 1% |

## Environment

| Item | Value |
|---|---|
| GPU | TBD |
| vLLM version | TBD |
| Model | TBD |
| Server flags | TBD |
| KV cache capacity (tokens) | TBD |

## How to run

TBD

## Results

TBD

## Repo layout

```
serve.sh      # starts the vLLM server
sweep.sh      # runs every concurrency × context-length combination
plot.py       # turns results/*.json into plots/
results/      # raw benchmark results (JSON)
plots/        # generated figures
```
