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
| GPU | Tesla T4 on Google Colab (14.56 GiB usable) |
| vLLM version | 0.34.0 (Colab Python 3.13) |
| Model | Qwen/Qwen2.5-1.5B-Instruct (float16) |
| Server flags | `--dtype half --max-model-len 4096 --gpu-memory-utilization 0.90` |
| KV cache capacity (tokens) | 335,872 (5,248 blocks × 64 tokens, 8.97 GiB) |
| Attention | FlashAttention 2 unavailable on T4 (compute capability 7.5); vLLM falls back to another backend |

## How to run

| Notebook | What it runs |
|---|---|
| [`colab_vllm_t4gpu.ipynb`](colab_vllm_t4gpu.ipynb) | vLLM: server setup, the sweep (`sweep.sh`), prefix caching ON/OFF |
| [`colab_sglang.ipynb`](colab_sglang.ipynb) | SGLang: the same experiments, measured with the same vLLM benchmark client |

Open a notebook in Colab (**File → Open notebook → GitHub**), pick a T4 runtime, and run the cells top to bottom.
`sweep.sh` can also be run on any machine with a server already running; see the comments at the top of the script.

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
