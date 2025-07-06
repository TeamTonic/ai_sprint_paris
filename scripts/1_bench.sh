#Usage:
# ./1_bench.sh server
# ./1_bench.sh perf
# ./1_bench.sh accuracy
# ./1_bench.sh profile
# ./1_bench.sh all (perf + accuracy + profile)
# ./1_bench.sh submit <team_name> (runs accuracy + perf + submits to leaderboard)

mkdir -p results
export MODEL="amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV"
export VLLM_TORCH_PROFILER_DIR=./profile
export PYTORCH_NO_HIP_MEMORY_CACHING=1
export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export VLLM_MAX_NUM_SEQS=32
export VLLM_GPU_MEMORY_UTILIZATION=0.95
export VLLM_ATTENTION_BACKEND=triton
export ROCM_USE_FLASH_ATTN_V2_TRITON=True
export VLLM_DTYPE=float16
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_ENGINE_USE_PARALLEL_SAMPLING=1
export VLLM_ENABLE_FUSED_MOE=1
export VLLM_MOE_TOPK=2
export VLLM_MOE_CAPACITY_FACTOR=2.0
export VLLM_ENABLE_LOG_STATS=0
export VLLM_USE_TRITON_FLASH_ATTN=0

# export TORCH_NCCL_HIGH_PRIORITY=1
# export GPU_MAX_HW_QUEUES=2

LB_URL="https://siro1-amd-leaderboard.hf.space"

# Check team name for submit mode
if [ $1 == "submit" ]; then
    if [ ! -z "$2" ]; then
        TEAM_NAME="$2"
    elif [ ! -z "$TEAM_NAME" ]; then
        TEAM_NAME="$TEAM_NAME"
    else
        echo "ERROR: Team name required for submit mode"
        echo "Usage: ./1_bench.sh submit <team_name>"
        echo "Or set TEAM_NAME environment variable"
        exit 1
    fi
    echo "INFO: Using team name: $TEAM_NAME"
fi

if [ $1 == "server" ]; then
    echo "INFO: server"
    # Set environment variable correctly
    # Launch vLLM server with ROCm profiling and enforce eager execution for better profiling compatibility
    rocprofv3 --hip-trace --hsa-trace -o vllm_server_trace.json -- vllm serve $MODEL \
        --enforce-eager \
        --disable-log-requests \
        --no-enable-prefix-caching \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --cuda-graph-sizes 64 \
        --dtype float16 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 1024 \
        --attention-backend triton \
        --no-enable-chunked-prefill \
        --num-scheduler-steps 15 \
        --max-seq-len-to-capture 16384 \
        --port 8000 \
        # Add any additional model-specific or optimization flags here
fi


if [ $1 == "perf" ] || [ $1 == "all" ] || [ $1 == "submit" ]; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
	sleep 1
    done
    echo "INFO: performance"
    INPUT_LENGTH=2048
    OUTPUT_LENGTH=2048
    CONCURRENT=32
    date=$(date +'%b%d_%H_%M_%S')
    rpt=result_${date}.json
    python /vllm-dev/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len ${INPUT_LENGTH} \
        --random-output-len ${OUTPUT_LENGTH} \
        --num-prompts $(( $CONCURRENT * 2 )) \
        --max-concurrency $CONCURRENT \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --result-dir ./results/ \
        --result-filename $rpt \
        --percentile-metrics ttft,tpot,itl,e2el \
        --seed 92100 \
        --disable-log-requests

    PERF_OUTPUT=$(python show_results.py)
    echo "$PERF_OUTPUT"
fi


# TODO: do not use 8 months old baberabb/lm-evaluation-harness/wikitext-tokens
if [ $1 == "accuracy" ] || [ $1 == "all" ] || [ $1 == "submit" ]; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
	sleep 1
    done
    echo "INFO: accuracy"
    if [ "$(which lm_eval)" == "" ] ; then
    git clone https://github.com/baberabb/lm-evaluation-harness.git -b wikitext-tokens
    cd lm-evaluation-harness
    pip install -e .
    pip install lm-eval[api]
    fi
    
    ACCURACY_OUTPUT=$(lm_eval --model local-completions --model_args model=$MODEL,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=10,max_retries=3 --tasks wikitext 2>&1)
    echo "$ACCURACY_OUTPUT"
fi

if [ $1 == "profile" ] || [ $1 == "all" ] ; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
	sleep 1
    done
    echo "INIFO: performance"
    INPUT_LENGTH=2048
    OUTPUT_LENGTH=2048
    CONCURRENT=16
    date=$(date +'%b%d_%H_%M_%S')
    rpt=result_${date}.json
    python /vllm-dev/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len ${INPUT_LENGTH} \
        --random-output-len ${OUTPUT_LENGTH} \
        --num-prompts $(( $CONCURRENT * 2 )) \
        --max-concurrency $CONCURRENT \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --profile \
        --result-dir ./results_with_profile/ \
        --result-filename $rpt \
        --percentile-metrics ttft,tpot,itl,e2el \
        --seed 92100 \
        --disable-log-requests
fi

if [ "$1" == "profile_fused_moe" ]; then
    echo "INFO: Profiling fused_moe_kernel with rocprofiler-compute"
    # Ensure rocprofiler-compute is installed and available
    if ! command -v rocprofiler-compute &> /dev/null; then
        echo "ERROR: rocprofiler-compute not found. Please install it first."
        exit 1
    fi
    # Run the profiler targeting the fused_moe_kernel
    CUDA_VISIBLE_DEVICES=0 ROCPROFCOMPUTE_LOGLEVEL=debug rocprofiler-compute profile --name fused_moe_profile --device 0 --kernel fused_moe_kernel -- python /vllm-dev/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 128 \
        --random-output-len 10 \
        --num-prompts 32 \
        --max-concurrency 16 \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --profile \
        --result-dir ./results_with_profile/ \
        --result-filename fused_moe_profile.json \
        --percentile-metrics ttft,tpot,itl,e2el
    echo "INFO: Profiling complete. Results are in workloads/fused_moe_profile/MI300/"
    exit 0
fi

if [ $1 == "submit" ]; then
    echo "INFO: Submitting results for team: $TEAM_NAME"
    
    PERF_LINE=$(echo "$PERF_OUTPUT" | grep -E "[0-9]+\.[0-9]+.*,[[:space:]]*[0-9]+\.[0-9]+" | tail -1)
    TTFT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}')     # Convert ms to seconds
    TPOT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')     # Convert ms to seconds  
    ITL=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}')      # Convert ms to seconds
    E2E=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}')      # Convert ms to seconds
    THROUGHPUT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}')
    
    # Parse accuracy metrics from lm_eval output
    BITS_PER_BYTE=$(echo "$ACCURACY_OUTPUT" | grep -oE "bits_per_byte[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    BYTE_PERPLEXITY=$(echo "$ACCURACY_OUTPUT" | grep -oE "byte_perplexity[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    WORD_PERPLEXITY=$(echo "$ACCURACY_OUTPUT" | grep -oE "word_perplexity[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    
    # Default to 0.0 if parsing fails
    TTFT=${TTFT:-0.0}
    TPOT=${TPOT:-0.0}
    ITL=${ITL:-0.0}
    E2E=${E2E:-0.0}
    THROUGHPUT=${THROUGHPUT:-0.0}
    BITS_PER_BYTE=${BITS_PER_BYTE:-0.0}
    BYTE_PERPLEXITY=${BYTE_PERPLEXITY:-0.0}
    WORD_PERPLEXITY=${WORD_PERPLEXITY:-0.0}
    
    echo "Performance metrics:"
    echo "  TTFT: ${TTFT}ms"
    echo "  TPOT: ${TPOT}ms"
    echo "  ITL: ${ITL}ms"
    echo "  E2E: ${E2E}ms"
    echo "  Throughput: ${THROUGHPUT} tokens/s"
    echo "Accuracy metrics:"
    echo "  Bits per Byte: ${BITS_PER_BYTE}"
    echo "  Byte Perplexity: ${BYTE_PERPLEXITY}"
    echo "  Word Perplexity: ${WORD_PERPLEXITY}"
    
    # Submit to leaderboard
    echo "Submitting to leaderboard..."
    curl -X POST $LB_URL/gradio_api/call/submit_results -s -H "Content-Type: application/json" -d "{
        \"data\": [
            \"$TEAM_NAME\",
            $TTFT,
            $TPOT,
            $ITL,
            $E2E,
            $THROUGHPUT,
            $BITS_PER_BYTE,
            $BYTE_PERPLEXITY,
            $WORD_PERPLEXITY
        ]
    }" | awk -F'"' '{ print $4}' | read EVENT_ID

    sleep 2

    echo "SUCCESS: Results submitted to leaderboard! 🤗 Check it out @ $LB_URL"
fi
