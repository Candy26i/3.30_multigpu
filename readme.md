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
git clone https://github.com/Candy26i/3_20_research.git
cd 3_20_research
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

Ensure `pyproject.toml` exists, then:

```bash
uv sync
```

---

# 🔥 Critical Dependencies (MUST MATCH)

## PyTorch (CUDA 12.8)

```bash
uv pip uninstall -y torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Transformers (>= 5.0.0.dev)

```bash
uv pip uninstall -y transformers
uv pip install git+https://github.com/huggingface/transformers.git@main

python -c "import transformers; print(transformers.__version__)"
```

✅ Must be **5.0.0.dev or higher**
Otherwise GRPO tool-calling will fail.

---

## TRL (>= 1.0.0.dev)

```bash
uv pip uninstall -y trl
uv pip install git+https://github.com/huggingface/trl.git

python -c "import trl; print(trl.__version__)"
```

---

## Other

```bash
uv pip install jmespath
```

---

# 🧩 Training Pipeline

## 1️⃣ Train Reasoning Tool (LoRA)

```bash
python agents_as_tools.py --stage train_tool_reasoning --task_name pubmedqa --base_model Qwen/Qwen3-0.6B --tool_sft_out_dir tool_sft_pubmedqa_500 --reasoning_tool_out reasoning_lora_mvp_split --tool_use_lora --tool_max_seq_len 4096 --tool_lr 2e-4 --tool_epochs 2 --tool_bs 1 --tool_grad_accum 8 --seed 42
```

⏱ ~7 minutes on A100

---

## 2️⃣ Train Context Tool (LoRA)

```bash
python agents_as_tools.py --stage train_tool_context --task_name pubmedqa --base_model Qwen/Qwen3-0.6B --tool_sft_out_dir tool_sft_pubmedqa_500 --context_tool_out context_lora_mvp_split --tool_use_lora --tool_max_seq_len 4096 --tool_lr 2e-4 --tool_epochs 2 --tool_bs 1 --tool_grad_accum 8 --seed 42
```

⏱ ~7 minutes on A100

---

## 3️⃣ Train Manager (GRPO)

```bash
python agents_as_tools.py --stage train_manager_grpo --task_name pubmedqa --base_model Qwen/Qwen3-0.6B --data_path pubmedqa --split_path splits_pubmedqa_500.json --reasoning_tool_out reasoning_lora_mvp_split --context_tool_out context_lora_mvp_split --manager_out manager_grpo_mvp_split --mgr_bs 8 --mgr_max_completion_length 512 --mgr_temperature 0.9 --mgr_num_generations 2 --grpo_beta 0.01 --fail_buffer_jsonl manager_grpo_mvp_split/fail_buffer.jsonl --raw_trace_jsonl manager_grpo_mvp_split/train_raw_trace.jsonl --seed 42
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
