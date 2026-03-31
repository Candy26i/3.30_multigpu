# `agents_as_tools_multi_vllm.py` RunPod Guide

This guide matches the current code in this repo and is intended for:

- Linux / RunPod environments
- multi-GPU launches through `accelerate`
- `TRL GRPOTrainer`
- both `vLLM` modes: `colocate` and `server`
- shared tool runtime: one tool base model plus two PEFT adapters

Important behavior in the current code:

- manager GRPO defaults to `grpo_beta=0.0`, which avoids loading a reference model and saves VRAM
- when `--mgr_use_vllm` is enabled, the default mode is `--mgr_vllm_mode colocate`
- GRPO validates that
  `num_processes * per_device_train_batch_size * gradient_accumulation_steps`
  is divisible by `num_generations`
- if `reasoning_tool_out` and `context_tool_out` are PEFT adapters from the same `tool_base_model`, the tool runtime will try to load only one shared base model
- if shared tool initialization fails, the code automatically falls back to the older split-model tool runtime

## 1. Recommended Environment

Use a dedicated Linux virtual environment for this project.

Do not reuse an environment that already has `transformers>=5.0.0` installed if you want to use `vLLM` in the same environment.

Recommended stack for this repo:

- Python `3.11`
- `torch==2.10.0`
- `transformers==4.57.6`
- `trl==1.0.0rc1`
- `vllm==0.17.1`
- `accelerate==1.12.0`
- `peft==0.18.1`
- `datasets` from `requirements-grpo-vllm-linux.txt`

Why this exact direction:

- official `vLLM 0.17.1` requires `torch==2.10.0`
- official `vLLM 0.17.1` requires `transformers>=4.56.0,<5`
- official `TRL 1.0.0rc1` allows `vllm<=0.17.1`

Practical conclusion:

- if you need `vLLM`, you should stay on `transformers 4.x`
- if you insist on `transformers 5.x`, the practical solution is a separate environment, or no `vLLM`
- if your served model is so new that `vLLM` only supports it with newer `transformers`, use `server` mode with a separate `vLLM` environment and keep the trainer environment pinned to the versions in this repo

Recommended:

- RunPod Linux pod
- Python 3.11
- CUDA available
- persistent volume or network storage for checkpoints

Not recommended:

- native Windows for `vLLM`
- `server` mode on a 2-GPU pod

Why:

- in `server` mode, TRL expects the `vLLM` server and the trainer to use different CUDA devices
- a 2-GPU pod is usually better suited for `colocate`
- a 4-GPU pod is better suited for `server`

## 2. Cache and Environment Variables

On RunPod, set cache directories first so models are not repeatedly downloaded:

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If you need access to gated or private Hugging Face models:

```bash
huggingface-cli login
```

## 3. Install Dependencies

This repo already includes:

- `requirements-grpo-vllm-linux.txt`
- `accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml`

On a fresh RunPod Linux shell, create a dedicated environment first:

```bash
cd /workspace/MedQA
python3.11 -m venv .venv
source .venv/bin/activate
python -V
```

Then install with:

```bash
pip install --upgrade pip
pip install --upgrade --upgrade-strategy eager -r requirements-grpo-vllm-linux.txt
pip check
```

Recommended sanity checks:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import transformers, trl, vllm, accelerate, peft, datasets; print(transformers.__version__, trl.__version__, vllm.__version__)"
```

Expected direction:

- `transformers` should print `4.57.6`
- `trl` should print `1.0.0rc1`
- `vllm` should print `0.17.1`

If you see `transformers 5.x` in this environment, fix that before training:

```bash
pip uninstall -y transformers
pip install --upgrade --upgrade-strategy eager -r requirements-grpo-vllm-linux.txt
```

## 3.1 If Your Model Needs `transformers 5.x`

This can happen for very new model families, for example the `qwen3_5` architecture discussed in the `vLLM` issue tracker.

Do not try to force one shared environment to satisfy all of these at once:

- `TRL` trainer
- `vLLM`
- `transformers 5.x`

For this repo, the clean solution is:

- trainer environment: use the pinned stack in `requirements-grpo-vllm-linux.txt`
- `vLLM` server environment: use a separate environment or container that matches the model's own serving requirements
- connect them through `--mgr_vllm_mode server` and `--mgr_vllm_server_base_url`

Why this works well here:

- your script already supports external `vLLM server` mode
- the trainer and the `vLLM` server do not need to import the exact same Python packages in the same environment
- this avoids the `transformers 4.x` vs `transformers 5.x` conflict entirely

## 4. Main Stages

The main stages in the current script are:

- `make_splits`
- `build_tool_sft`
- `train_tool_reasoning`
- `train_tool_context`
- `train_manager_grpo`
- `evolve_build_manager_sft`
- `train_manager_sft`
- `evolve_round`

Important manager GRPO flags:

- `--mgr_bs`
- `--mgr_grad_accum`
- `--mgr_num_generations`
- `--mgr_max_completion_length`
- `--grpo_beta`
- `--mgr_use_vllm`
- `--mgr_vllm_mode`
- `--mgr_vllm_gpu_memory_utilization`
- `--mgr_use_lora`
- `--mgr_gradient_checkpointing`

## 5. Important Runtime Constraints

### 5.1 GRPO Batch Geometry

The current code enforces:

```text
num_processes * mgr_bs * mgr_grad_accum % mgr_num_generations == 0
```

Examples:

- 2 GPUs, `mgr_bs=1`, `mgr_grad_accum=2`, `mgr_num_generations=4` -> valid
- 2 GPUs, `mgr_bs=1`, `mgr_grad_accum=1`, `mgr_num_generations=4` -> invalid
- 2 GPUs, `mgr_bs=1`, `mgr_grad_accum=1`, `mgr_num_generations=2` -> valid

### 5.2 Conditions for Shared Tool Base

The shared-base tool runtime only works when all of the following are true:

- `reasoning_tool_out` is a PEFT adapter
- `context_tool_out` is a PEFT adapter
- both adapters come from the same `tool_base_model`
- `peft` is installed

So if you want the lower-VRAM shared tool path, train the tool models with:

```bash
--tool_use_lora
```

When shared loading succeeds, you should see:

```text
[TOOLS] runtime=shared_base adapters=reasoning_tool,context_tool
```

If you instead see:

```text
[TOOLS] runtime=split_models
```

then the code fell back to loading two separate tool models.

## 6. Recommended RunPod Modes

### 6.1 Two GPUs

Recommended:

- `colocate`
- same model family for manager and tool base
- LoRA for manager GRPO
- gradient checkpointing enabled

Not recommended:

- `server` mode on a 2-GPU pod

### 6.2 Four or More GPUs

Recommended:

- 2 GPUs for trainer
- 2 GPUs for `vLLM server`
- trainer launched with `accelerate`
- `vLLM server` and trainer running in the same pod using `127.0.0.1`

If both trainer and server are inside the same RunPod pod:

- you usually do not need to expose port `8000` publicly
- prefer `http://127.0.0.1:8000`

## 7. Basic RunPod Commands

Enter the project directory:

```bash
cd /workspace/MedQA
```

Using `tmux` is strongly recommended, especially for `server` mode:

```bash
tmux new -s medqa
```

Inspect GPUs:

```bash
nvidia-smi
```

Inspect the accelerate config:

```bash
cat accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml
```

## 8. Full Pipeline Example

The examples below assume:

- task: `pubmedqa`
- manager base: `Qwen/Qwen3-8B`
- tool base: `Qwen/Qwen3-8B`
- tool checkpoints are saved as adapters
- outputs are written to the current workspace

### 8.1 Create Train / Dev / Test Splits

```bash
python agents_as_tools_multi_vllm.py \
  --stage make_splits \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --seed 42
```

### 8.2 Build Tool SFT Data

```bash
python agents_as_tools_multi_vllm.py \
  --stage build_tool_sft \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --tool_sft_out_dir tool_sft_data_evolving \
  --seed 42
```

### 8.3 Train the Reasoning Tool

You should use `--tool_use_lora` here, otherwise you will not get the shared tool runtime memory savings later.

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  agents_as_tools_multi_vllm.py \
  --stage train_tool_reasoning \
  --task_name pubmedqa \
  --tool_base_model Qwen/Qwen3-8B \
  --tool_sft_out_dir tool_sft_data_evolving \
  --reasoning_tool_out reasoning_tool_adapter \
  --tool_bs 1 \
  --tool_grad_accum 8 \
  --tool_epochs 2 \
  --tool_max_seq_len 4096 \
  --tool_use_lora
```

### 8.4 Train the Context Tool

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  agents_as_tools_multi_vllm.py \
  --stage train_tool_context \
  --task_name pubmedqa \
  --tool_base_model Qwen/Qwen3-8B \
  --tool_sft_out_dir tool_sft_data_evolving \
  --context_tool_out context_tool_adapter \
  --tool_bs 1 \
  --tool_grad_accum 8 \
  --tool_epochs 2 \
  --tool_max_seq_len 4096 \
  --tool_use_lora
```

## 9. Recommended 2-GPU RunPod Command: `colocate`

This is the best starting point for your current setup.

Best fit for:

- 2 x A100 80GB
- 8B manager
- 8B tool base
- both tool checkpoints saved as adapters

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_grad_accum 2 \
  --mgr_num_generations 4 \
  --mgr_max_completion_length 1024 \
  --grpo_beta 0.0 \
  --mgr_use_vllm \
  --mgr_vllm_mode colocate \
  --mgr_vllm_gpu_memory_utilization 0.10 \
  --mgr_vllm_enable_sleep_mode \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```

Why this configuration:

- `2 * 1 * 2 = 4`, so the effective batch is divisible by `mgr_num_generations=4`
- `grpo_beta=0.0` avoids loading a reference model
- `mgr_use_lora` reduces manager GRPO memory usage
- `mgr_gradient_checkpointing` further reduces memory usage
- `mgr_max_completion_length=1024` keeps rollout memory under control
- `mgr_vllm_gpu_memory_utilization=0.10` makes colocate mode more stable

If you still hit OOM, reduce further:

- change `--mgr_num_generations` from `4` to `2`
- change `--mgr_grad_accum` from `2` to `1`
- reduce `--mgr_max_completion_length` from `1024` to `512`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_grad_accum 1 \
  --mgr_num_generations 2 \
  --mgr_max_completion_length 512 \
  --grpo_beta 0.0 \
  --mgr_use_vllm \
  --mgr_vllm_mode colocate \
  --mgr_vllm_gpu_memory_utilization 0.08 \
  --mgr_vllm_enable_sleep_mode \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```

## 10. Recommended 4-GPU RunPod Command: `server`

Best fit for:

- 4 or more GPUs
- separating trainer and `vLLM server`

Assume:

- `GPU 0,1` for trainer
- `GPU 2,3` for `vLLM server`

### 10.1 Start the vLLM Server

In the first shell or tmux pane:

```bash
CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.35
```

### 10.2 Start the Trainer

In the second shell or tmux pane:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_grad_accum 2 \
  --mgr_num_generations 4 \
  --mgr_max_completion_length 1024 \
  --grpo_beta 0.0 \
  --mgr_use_vllm \
  --mgr_vllm_mode server \
  --mgr_vllm_server_base_url http://127.0.0.1:8000 \
  --mgr_vllm_visible_devices 2,3 \
  --mgr_vllm_tensor_parallel_size 2 \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```

## 11. Evolve Stage Usage

If you already finished one GRPO run and produced a fail buffer:

### 11.1 Build Manager SFT Data from Failures

```bash
python agents_as_tools_multi_vllm.py \
  --stage evolve_build_manager_sft \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --fail_buffer_jsonl manager_grpo_binary/fail_buffer.jsonl \
  --evolve_out_dir evolve_manager_sft
```

### 11.2 Train Manager SFT

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_sft \
  --task_name pubmedqa \
  --manager_base_model Qwen/Qwen3-8B \
  --evolve_out_dir evolve_manager_sft \
  --manager_sft_out manager_sft_evolved \
  --manager_sft_bs 1 \
  --manager_sft_grad_accum 8 \
  --manager_sft_use_lora
```

## 12. What to Check When Debugging on RunPod

### 12.1 Check Whether Shared Tool Runtime Is Active

Success:

```text
[TOOLS] runtime=shared_base adapters=reasoning_tool,context_tool
```

Fallback:

```text
[WARN] shared tool base init failed ...
[TOOLS] runtime=split_models
```

Common reasons:

- the tools were not trained with `--tool_use_lora`
- `reasoning_tool_out` / `context_tool_out` are not adapter directories
- the adapters do not match the same base model

### 12.2 If You Hit OOM

Reduce in this order:

1. `--mgr_num_generations 4 -> 2`
2. `--mgr_max_completion_length 1024 -> 512`
3. `--mgr_vllm_gpu_memory_utilization 0.10 -> 0.08`
4. keep `--grpo_beta 0.0`
5. keep `--mgr_use_lora`
6. keep `--mgr_gradient_checkpointing`

### 12.3 If GRPO Batch Geometry Is Invalid

Reconfigure these three flags:

- `--mgr_bs`
- `--mgr_grad_accum`
- `--mgr_num_generations`

Easy valid combinations:

- 2 GPUs: `mgr_bs=1, mgr_grad_accum=1, mgr_num_generations=2`
- 2 GPUs: `mgr_bs=1, mgr_grad_accum=2, mgr_num_generations=4`

### 12.4 If `server` Mode Cannot Connect

Check:

```bash
curl http://127.0.0.1:8000/health
```

If trainer and server are in the same RunPod pod:

- prefer `127.0.0.1`
- do not start by debugging public networking

## 13. Best Practical Starting Command for Your Current Setup

If your setup is:

- 2 x A100 80GB
- 8B manager
- 8B tool base
- both tools saved as adapters

Start with:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_grad_accum 2 \
  --mgr_num_generations 4 \
  --mgr_max_completion_length 1024 \
  --grpo_beta 0.0 \
  --mgr_use_vllm \
  --mgr_vllm_mode colocate \
  --mgr_vllm_gpu_memory_utilization 0.10 \
  --mgr_vllm_enable_sleep_mode \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```

If that still OOMs, switch to this more conservative version:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml \
  agents_as_tools_multi_vllm.py \
  --stage train_manager_grpo \
  --task_name pubmedqa \
  --data_path pubmedqa \
  --split_path splits_pubmedqa_1000.json \
  --manager_base_model Qwen/Qwen3-8B \
  --tool_base_model Qwen/Qwen3-8B \
  --reasoning_tool_out reasoning_tool_adapter \
  --context_tool_out context_tool_adapter \
  --manager_out manager_grpo_binary \
  --mgr_bs 1 \
  --mgr_grad_accum 1 \
  --mgr_num_generations 2 \
  --mgr_max_completion_length 512 \
  --grpo_beta 0.0 \
  --mgr_use_vllm \
  --mgr_vllm_mode colocate \
  --mgr_vllm_gpu_memory_utilization 0.08 \
  --mgr_vllm_enable_sleep_mode \
  --mgr_use_lora \
  --mgr_gradient_checkpointing
```
