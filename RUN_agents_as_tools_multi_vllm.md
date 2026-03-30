# `agents_as_tools_multi_vllm.py` runtime notes

This script now supports:

- multi-GPU training through `accelerate launch`
- TRL GRPO with `vLLM` in both `server` and `colocate` modes
- native TRL tool calling with automatic fallback between `environment` and `argument` binding

## Environment

- Use Linux or WSL2 for `vLLM`
- Use a CUDA-enabled PyTorch install
- Verified stack on 2026-03-30: `trl==0.29.1` and `vllm==0.17.0`
- Install dependencies with:
vllm==0.17.0
accelerate>=1.12.0
```bash
pip install --upgrade --upgrade-strategy eager -r requirements-grpo-vllm-linux.txt
```

## Recommended 4-GPU layout

Use 2 GPUs for training and 2 GPUs for the dedicated `vLLM` server.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.35
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --manager_base_model Qwen/Qwen3-0.6B \
  --tool_base_model Qwen/Qwen3-0.6B \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_num_generations 4 \
  --mgr_max_completion_length 2048 \
  --mgr_use_vllm \
  --mgr_vllm_mode server \
  --mgr_vllm_server_base_url http://127.0.0.1:8000 \
  --mgr_vllm_visible_devices 2,3 \
  --mgr_vllm_tensor_parallel_size 2
```

## Colocate mode

If you do not have dedicated inference GPUs, use colocate mode instead:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --manager_base_model Qwen/Qwen3-0.6B \
  --tool_base_model Qwen/Qwen3-0.6B \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_num_generations 4 \
  --mgr_max_completion_length 2048 \
  --mgr_use_vllm \
  --mgr_vllm_mode colocate \
  --mgr_vllm_gpu_memory_utilization 0.25 \
  --mgr_vllm_enable_sleep_mode
```

## Notes

- On native Windows, the script will now fail fast with a clear error if `vLLM` is requested.
- Multi-GPU training expects `accelerate` to set `LOCAL_RANK` and `WORLD_SIZE`.
- The manager training model is no longer forced onto a single device before `Trainer` starts, so DDP can shard work correctly.
