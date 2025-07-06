CONTAINER_NAME="vllm-container"
DOCKER_IMG="rocm/vllm-dev:nightly"


running_container=$(docker ps -q --filter "name=$CONTAINER_NAME")

if [ $running_container ]; then
    echo "Stopping the already running $CONTAINER_NAME container"
    docker stop $CONTAINER_NAME
fi


if ! test -f vllm/setup.py; then echo "WARNING: This script assumes it is launched from a directory containing a cloned vllm, but it was not found. Make sure vllm is cloned at ${PWD}/vllm."; fi

echo "Starting a container based off $DOCKER_IMG..."
echo "With the following mounted folders:"
echo "$PWD/.hf_cache -> /root/.cache/huggingface/hub"
echo "$PWD/.vllm_cache -> /root/.cache/vllm/"
echo "$PWD -> /workspace"
echo "$PWD/vllm -> /vllm-dev"

# Optimization: Set environment variables for maximum throughput and e2e
# Remove conflicting or redundant settings, and set optimal values
# Use ROCm backend for attention, maximize concurrency, and memory utilization
# Add comments for clarity

docker run \
    --rm \
    -it \
    --ipc host \
    --name $CONTAINER_NAME \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e PYTORCH_ROCM_ARCH="gfx942" \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -e PYTORCH_NO_HIP_MEMORY_CACHING=1 \
    -e HSA_DISABLE_FRAGMENT_ALLOCATOR=1 \
    -e VLLM_MAX_NUM_SEQS=1024 \
    -e VLLM_GPU_MEMORY_UTILIZATION=0.98 \
    -e VLLM_ATTENTION_BACKEND=rocm \
    -e VLLM_DTYPE=float16 \
    -e VLLM_ENABLE_CHUNKED_PREFILL=1 \
    -e TORCH_NCCL_HIGH_PRIORITY=1 \
    -e GPU_MAX_HW_QUEUES=4 \
    -e VLLM_ENGINE_USE_PARALLEL_SAMPLING=1 \
    -e VLLM_ENABLE_FUSED_MOE=1 \
    -e VLLM_ROCM_USE_SKINNY_GEMM=1 \
    -e VLLM_MOE_TOPK=2 \
    -e VLLM_MOE_CAPACITY_FACTOR=2.0 \
    -e VLLM_ENABLE_LOG_STATS=0 \
    -v "$PWD/.hf_cache/":/root/.cache/huggingface/hub/ \
    -v "$PWD/.vllm_cache/":/root/.cache/vllm/ \
    -v "$PWD":/workspace \
    -v ${PWD}/vllm:/vllm-dev \
    -w /workspace \
    $DOCKER_IMG 
