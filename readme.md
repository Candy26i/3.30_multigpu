# 🧠 Agents-as-Tools: GRPO Training Pipeline (PubMedQA)

This repository contains a **multi-agent training pipeline** for PubMedQA using:

* Tool-augmented LLMs (Reasoning + Context agents)
* LoRA fine-tuning
* GRPO (Generalized Reinforcement Policy Optimization)

---

# ⚙️ Environment Setup (RunPod + A100)

## ✅ Recommended Setup

* GPU: **A100**
* Storage: **Network Volume (≥20GB)**
  ⚠️ Important: Prevents data loss when pod restarts

---

## ⚠️ RunPod Tips (VERY IMPORTANT)

* SSH connections may drop frequently

* ✅ Use **Web Terminal** instead (stable)

* ✅ Launch **JupyterLab** to:

  * inspect files
  * preview JSON outputs

* ❗ If you stop the pod:

  * `/workspace` is **persistent**
  * anything outside it is **lost**

---

# 🚀 Setup Instructions

## 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

---

## 2. Clone Repository

```bash
git clone https://github.com/Candy26i/3.30_multigpu.git
cd 3.30_research
```

---

## 3. Create Python Environment

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
```

---

## 4. Install Dependencies

# 🔥 Critical Dependencies (MUST MATCH)

## PyTorch (CUDA 12.8)

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers  # ensure it is >= 5.0.0 but dont get the newest version
uv pip install git+https://github.com/huggingface/trl.git
uv pip install jmespath
uv pip install peft
uv pip install wandb
wandb login


```

---

## Transformers (>= 5.0.0.dev)

```bash

python -c "import transformers; print(transformers.__version__)"
```

✅ Must be **5.0.0.dev or higher**
Otherwise GRPO tool-calling will fail.

---

## TRL (>= 1.0.0.dev)

```bash

python -c "import trl; print(trl.__version__)"
```

---


# 🧩 Training Pipeline

## 1️⃣ Train Reasoning Tool (LoRA)

```bash
python agents_as_tools.py --stage train_tool_reasoning --task_name pubmedqa --base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_pubmedqa_500 --reasoning_tool_out reasoning_lora_mvp_split --tool_use_lora --tool_max_seq_len 4096 --tool_lr 2e-4 --tool_epochs 2 --tool_bs 10 --tool_grad_accum 8 --seed 42
```

⏱ ~7 minutes on A100

---

## 2️⃣ Train Context Tool (LoRA)

```bash
python agents_as_tools.py --stage train_tool_context --task_name pubmedqa --base_model Qwen/Qwen3-8B --tool_sft_out_dir tool_sft_pubmedqa_500 --context_tool_out context_lora_mvp_split --tool_use_lora --tool_max_seq_len 4096 --tool_lr 2e-4 --tool_epochs 2 --tool_bs 10 --tool_grad_accum 8 --seed 42
```

⏱ ~7 minutes on A100

---


## MULTIPLE GPU

wandb change to your own
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_configs/trl_multi_gpu_2gpu_bf16.yaml agents_as_tools_multi_vllm.py --stage train_manager_grpo --task_name pubmedqa --data_path pubmedqa --split_path splits_pubmedqa_500.json --manager_base_model Qwen/Qwen3-8B --tool_base_model Qwen/Qwen3-8B --tool_binding_mode argument --reasoning_tool_out reasoning_lora_mvp_split --context_tool_out context_lora_mvp_split --manager_out manager_grpo_binary_no_vllm --mgr_bs 8 --mgr_grad_accum 2 --mgr_num_generations 8 --mgr_max_prompt_length 4000 --mgr_max_completion_length 4000 --grpo_beta 0.001 --mgr_use_lora --mgr_gradient_checkpointing --grpo_use_wandb --wandb_project runpod_pubmedqa_grpo --wandb_entity madisonlijingxuan-ucla --wandb_run_name manager_grpo_binary_8_inter_kl
```
⏱ ~10 hours on A100

---

# 📂 Output Structure

```
.
├── reasoning_lora_mvp_split/
├── context_lora_mvp_split/
├── manager_grpo_mvp_split/
│   ├── fail_buffer.jsonl
│   └── train_raw_trace.jsonl
```
# Evaluation
```
python evaluate_pipeline_vs_baselines.py --task_name pubmedqa --data_path pubmedqa --split_path splits_pubmedqa_500.json --split_key test_ids --max_eval_samples 0 --pipeline_manager_dir manager_grpo_binary_no_vllm --pipeline_base_model_for_tools Qwen/Qwen3-8B --pipeline_reasoning_adapter reasoning_lora_mvp_split --pipeline_context_adapter context_lora_mvp_split --add_pipeline_no_tools_baseline --baseline_model_dirs "Qwen/Qwen3-8B" --baseline_model_names "qwen3_base_direct" --add_random_baseline --add_majority_baseline --temperature 0.0 --max_new_tokens 4000 --max_tool_calls 2 --out_dir eval_pubmedqa_compare

```





---

# 🧪 Notes & Debugging

## 🐢 Low GPU Usage (~20%)

Common causes:

* CPU bottleneck
* data loading too slow
* small batch size

---

## ⚠️ "transformers version error"

```
ImportError: Using tools with GRPOTrainer requires transformers>=5.0.0
```

✅ Fix: install from `main` branch (see above)

---

## 💥 Data Lost After Restart

If `/workspace` is empty:

👉 You saved files outside persistent volume

---

## 🧱 Cannot delete folder

```bash
rm -r folder_name
```

---

# 💡 Tips

* Use `tmux` for long training jobs
* Save logs frequently (`jsonl`)
* Push only results to GitHub (NOT models)

---
