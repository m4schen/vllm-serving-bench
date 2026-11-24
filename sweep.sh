#!/usr/bin/env bash
# Sweep vLLM serving performance over concurrency × input length.
#
# Usage (the vLLM server must already be running on localhost:8000):
#   bash sweep.sh
#   TAG=t4_kv1g CONCURRENCIES="16 32 64" bash sweep.sh    # override any setting
#
# Each run is saved as results/<TAG>/c<concurrency>_in<input_len>.json.
# Runs whose result file already exists are skipped, so the sweep can be
# resumed after a Colab disconnect.

set -uo pipefail

# ---- Settings (override with environment variables) ----
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TAG="${TAG:-t4_qwen1.5b}"                      # one folder per experiment setup
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
INPUT_LENS="${INPUT_LENS:-128 512 2048}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MIN_PROMPTS="${MIN_PROMPTS:-32}"               # at least this many requests per run
PROMPTS_PER_SLOT="${PROMPTS_PER_SLOT:-5}"      # requests per concurrency slot
BASE_URL="${BASE_URL:-http://localhost:8000}"

RESULT_DIR="results/${TAG}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---- Make sure the server is up ----
if ! curl -sf "${BASE_URL}/v1/models" > /dev/null; then
    echo "ERROR: no vLLM server at ${BASE_URL}. Start it first." >&2
    exit 1
fi

total=$(( $(wc -w <<< "${CONCURRENCIES}") * $(wc -w <<< "${INPUT_LENS}") ))
run=0
seed=100   # a new seed for every run, so prompts never repeat (no prefix-cache hits)

for in_len in ${INPUT_LENS}; do
    for c in ${CONCURRENCIES}; do
        run=$((run + 1))
        seed=$((seed + 1))
        name="c${c}_in${in_len}"

        if [[ -f "${RESULT_DIR}/${name}.json" ]]; then
            echo "[${run}/${total}] ${name}: already done, skipping"
            continue
        fi

        # Enough requests that p99 is meaningful even at high concurrency.
        num_prompts=$(( c * PROMPTS_PER_SLOT ))
        (( num_prompts < MIN_PROMPTS )) && num_prompts=${MIN_PROMPTS}

        echo "[${run}/${total}] ${name}: ${num_prompts} prompts, seed ${seed}"

        if vllm bench serve \
            --backend vllm \
            --base-url "${BASE_URL}" \
            --model "${MODEL}" \
            --dataset-name random \
            --random-input-len "${in_len}" \
            --random-output-len "${OUTPUT_LEN}" \
            --ignore-eos \
            --num-prompts "${num_prompts}" \
            --max-concurrency "${c}" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --metric-percentiles 50,99 \
            --seed "${seed}" \
            --metadata "tag=${TAG}" "input_len=${in_len}" "output_len=${OUTPUT_LEN}" \
            --save-result --result-dir "${RESULT_DIR}" --result-filename "${name}.json" \
            > "${LOG_DIR}/${name}.log" 2>&1
        then
            # Print the key numbers so progress is visible in the notebook.
            grep -E "Output token throughput|P50 TPOT|P99 TTFT" "${LOG_DIR}/${name}.log" | sed 's/^/    /'
        else
            echo "    FAILED (see ${LOG_DIR}/${name}.log); continuing"
        fi
    done
done

echo "Done. Results are in ${RESULT_DIR}/"
