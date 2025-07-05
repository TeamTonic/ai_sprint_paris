# ai_sprint_paris

# Running Benchmarks, Profiling, and Downloading Results

## 1. Running Performance Benchmarks

- Start the vLLM server in one terminal or tmux pane:
  ```sh
  docker exec -it vllm-container /bin/bash
  cd /workspace
  ./1_bench.sh server
  ```
- In another terminal or tmux pane, run the performance benchmark:
  ```sh
  docker exec -it vllm-container /bin/bash
  cd /workspace
  ./1_bench.sh perf
  ```
- Results will be saved in `/workspace/results/` inside the container.

## 2. Running Profiling Traces

- To run a profiling trace (standard):
  ```sh
  docker exec -it vllm-container /bin/bash
  cd /workspace
  ./1_bench.sh profile
  ```
- Profiling traces are saved in `/workspace/profile/` inside the container (e.g., `*.pt.trace.json.gz`).

- To profile a specific kernel (e.g., `fused_moe_kernel`) with ROCm Compute Profiler:
  ```sh
  docker exec -it vllm-container /bin/bash
  cd /workspace
  ./1_bench.sh profile_fused_moe
  ```
- Results will be saved in `workloads/fused_moe_profile/MI300/` inside the container.

## 3. Copying Results from the Container to the Host VM

- First, create the destination directory on the host if it doesn't exist:
  ```sh
  mkdir -p ~/ai_sprint_paris/scripts/profile/
  mkdir -p ~/ai_sprint_paris/scripts/results/
  ```
- Copy a profiling trace from the container to the host:
  ```sh
  docker cp vllm-container:/workspace/profile/your_trace_file.pt.trace.json.gz ~/ai_sprint_paris/scripts/profile/
  ```
- Copy a benchmark result from the container to the host:
  ```sh
  docker cp vllm-container:/workspace/results/your_result_file.json ~/ai_sprint_paris/scripts/results/
  ```

## 4. Downloading Results to Your Local Machine

- From your local machine, use `scp` to download the file:
  ```sh
  scp root@YOUR_IP:/root/ai_sprint_paris/scripts/profile/your_trace_file.pt.trace.json.gz .
  scp root@YOUR_IP:/root/ai_sprint_paris/scripts/results/your_result_file.json .
  ```
  Replace `YOUR_IP` and filenames as appropriate.

## 5. Visualizing Profiling Traces

- Go to [https://ui.perfetto.dev/](https://ui.perfetto.dev/) in your browser.
- Upload the `.pt.trace.json.gz` file to visualize the trace.

## Troubleshooting
- If you get a permissions error with your SSH key, restrict permissions using:
  ```sh
  icacls "C:\Users\MeMyself\.ssh\id_rsa" /inheritance:r
  icacls "C:\Users\MeMyself\.ssh\id_rsa" /grant:r "$($env:USERNAME):(R)"
  icacls "C:\Users\MeMyself\.ssh\id_rsa" /remove "Users" "Authenticated Users" "Everyone"
  # Repeat for id_ed25519 if needed
  ```
- If a directory does not exist, create it with `mkdir -p ...` before copying files.

For more details on profiling with ROCm Compute Profiler, see the [official documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) and [GitHub repo](https://github.com/ROCm/rocprofiler-compute).
