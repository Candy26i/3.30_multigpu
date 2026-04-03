# agents_as_tools_evolving_binary_v2.py
# -*- coding: utf-8 -*-

import argparse
import glob
import importlib
import inspect
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import transformers
from packaging.version import Version

DATASETS_AVAILABLE = False
DATASETS_IMPORT_ERROR: Optional[Exception] = None
try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except Exception as _datasets_import_error:
    Dataset = Any
    load_dataset = None
    DATASETS_IMPORT_ERROR = _datasets_import_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

TRL_AVAILABLE = False
TRL_IMPORT_ERROR: Optional[Exception] = None
try:
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import add_response_schema
    TRL_AVAILABLE = True
except Exception as _trl_import_error:
    GRPOConfig = None
    GRPOTrainer = None
    TRL_IMPORT_ERROR = _trl_import_error

    def add_response_schema(tokenizer: Any) -> Any:
        return tokenizer

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def require_trl(stage_name: str) -> None:
    if TRL_AVAILABLE:
        return
    detail = "" if TRL_IMPORT_ERROR is None else f" Original import error: {type(TRL_IMPORT_ERROR).__name__}: {TRL_IMPORT_ERROR}"
    raise RuntimeError(
        f"{stage_name} requires `trl`, but it is not installed in the current Python environment.{detail}"
    )


def require_datasets(stage_name: str) -> None:
    if DATASETS_AVAILABLE:
        return
    detail = (
        ""
        if DATASETS_IMPORT_ERROR is None
        else f" Original import error: {type(DATASETS_IMPORT_ERROR).__name__}: {DATASETS_IMPORT_ERROR}"
    )
    raise RuntimeError(
        f"{stage_name} requires `datasets`, but it is not installed in the current Python environment.{detail}"
    )


def get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        return -1


def get_global_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def get_world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1


def configure_cuda_runtime() -> None:
    local_rank = get_local_rank()
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)


def get_runtime_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = get_local_rank()
    if local_rank >= 0:
        return f"cuda:{local_rank}"
    return "cuda"


def get_runtime_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _import_optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _signature_parameter_names(callable_obj: Any) -> Optional[set]:
    try:
        return set(inspect.signature(callable_obj).parameters.keys())
    except Exception:
        return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    supported = _signature_parameter_names(callable_obj)
    if supported is None:
        return dict(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted([k for k in kwargs.keys() if k not in filtered])
    if dropped:
        print(f"[{label}] skipped unsupported kwargs: {', '.join(dropped)}")
    return filtered


def _grpo_supports_environment_factory() -> bool:
    if not TRL_AVAILABLE:
        return False
    supported = _signature_parameter_names(GRPOTrainer.__init__)
    return bool(supported and "environment_factory" in supported)


def _get_transformers_version() -> str:
    try:
        return str(getattr(transformers, "__version__", "0"))
    except Exception:
        return "0"


def _transformers_version_at_least(min_version: str) -> bool:
    try:
        return Version(_get_transformers_version()) >= Version(min_version)
    except Exception:
        return False


def _trainer_processing_kwargs(processing_obj: Any) -> Dict[str, Any]:
    if not TRL_AVAILABLE:
        return {}
    supported = _signature_parameter_names(GRPOTrainer.__init__) or set()
    if "processing_class" in supported:
        return {"processing_class": processing_obj}
    if "tokenizer" in supported:
        return {"tokenizer": processing_obj}
    return {}


def _platform_label() -> str:
    return f"{os.name}:{sys.platform}"


def _build_vllm_server_launch_hint(
    model_name: str,
    server_base_url: str,
    visible_devices: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> str:
    parsed = urlparse(server_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    prefix = f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices.strip() else ""
    return (
        f"{prefix}trl vllm-serve --model {model_name} "
        f"--tensor-parallel-size {max(1, int(tensor_parallel_size))} "
        f"--host {host} --port {port} "
        f"--gpu-memory-utilization {float(gpu_memory_utilization):.2f}"
    )


def validate_distributed_runtime(stage_name: str, require_cuda: bool = False, use_vllm: bool = False) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage_name} requires CUDA, but torch.cuda is unavailable in the current Python environment. "
            "This usually means you installed a CPU-only PyTorch build."
        )
    if get_world_size() > 1 and not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage_name} detected WORLD_SIZE={get_world_size()} but torch.cuda is unavailable. "
            "Multi-GPU training requires a CUDA-enabled PyTorch install."
        )
    if use_vllm:
        if os.name == "nt":
            raise RuntimeError(
                "vLLM is not supported in this native Windows runtime. "
                "Run this script under Linux or WSL2 with a CUDA-enabled PyTorch environment."
            )
        if _import_optional_module("vllm") is None:
            raise RuntimeError(
                f"{stage_name} requested vLLM, but `vllm` is not importable in the current environment."
            )


def validate_grpo_batch_geometry(
    per_device_train_bs: int,
    grad_accum: int,
    num_generations: int,
) -> None:
    world_size = max(1, get_world_size())
    effective_batch = int(per_device_train_bs) * world_size * int(grad_accum)
    if effective_batch <= 0:
        raise RuntimeError("GRPO effective batch size must be positive.")
    if int(num_generations) <= 0:
        raise RuntimeError("GRPO num_generations must be positive.")
    if effective_batch % int(num_generations) != 0:
        raise RuntimeError(
            "Invalid GRPO batch geometry: "
            f"per_device_train_batch_size({per_device_train_bs}) * world_size({world_size}) * "
            f"gradient_accumulation_steps({grad_accum}) = {effective_batch}, "
            f"which must be evenly divisible by num_generations({num_generations}). "
            "This follows the TRL GRPO requirement for effective batch size."
        )


# =========================================================
# Seed
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# JSON helpers
# =========================================================
def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first top-level JSON object from raw text.
    Returns only if the parsed object is a dict.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None

    return None

def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# =========================================================
# Task config (dataset/task-agnostic)
# =========================================================
TASK_NAME = "pubmedqa"
ANSWER_LABELS: List[str] = ["yes", "no", "maybe"]  # canonical labels
ANSWER_TOKEN_TO_CANONICAL: Dict[str, str] = {"YES": "yes", "NO": "no", "MAYBE": "maybe"}
ANSWER_CANONICAL_TO_TOKEN: Dict[str, str] = {"yes": "YES", "no": "NO", "maybe": "MAYBE"}
ANSWER_LASTLINE_RE = re.compile(
    r"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_(YES|NO|MAYBE)\b[^\w]*$",
    re.IGNORECASE,
)


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"Invalid label for tokenization: {label!r}")
    return s.upper()


def _default_labels_for_task(task_name: str) -> List[str]:
    t = (task_name or "").strip().lower()
    if t == "medqa":
        return ["A", "B", "C", "D", "E"]
    if t == "pubmedqa":
        return ["yes", "no", "maybe"]
    return ["yes", "no", "maybe"]


def _build_answer_regex(tokens: List[str]) -> re.Pattern:
    alts = "|".join([re.escape(t) for t in sorted(tokens, key=len, reverse=True)])
    # Last line must be exactly ANSWER_<LABEL> (optionally prefixed by "answer:")
    return re.compile(
        rf"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_({alts})\b[^\w]*$",
        re.IGNORECASE,
    )


def _parse_label_space_arg(label_space: str) -> List[str]:
    if not label_space or not label_space.strip():
        return []
    parts = []
    for p in label_space.split(","):
        s = p.strip()
        if s:
            parts.append(s)
    return parts


def _normalize_label(raw: Any) -> str:
    s = str(raw).strip()
    if not s:
        return s
    tok = _label_to_token(s)
    if tok in ANSWER_TOKEN_TO_CANONICAL:
        return ANSWER_TOKEN_TO_CANONICAL[tok]
    if s in ANSWER_CANONICAL_TO_TOKEN:
        return s
    return s


def configure_task(task_name: str, label_space: str = "") -> Tuple[str, List[str]]:
    """Configure task name + answer label space for parsing/reward/prompt."""
    global TASK_NAME, ANSWER_LABELS, ANSWER_TOKEN_TO_CANONICAL, ANSWER_CANONICAL_TO_TOKEN, ANSWER_LASTLINE_RE

    t = (task_name or "pubmedqa").strip().lower()
    labels = _parse_label_space_arg(label_space)
    if not labels:
        labels = _default_labels_for_task(t)

    token_to_canonical: Dict[str, str] = {}
    canonical_to_token: Dict[str, str] = {}
    canonical_labels: List[str] = []

    for lb in labels:
        canonical = str(lb).strip()
        if not canonical:
            continue
        token = _label_to_token(canonical)
        if token in token_to_canonical and token_to_canonical[token] != canonical:
            raise ValueError(
                f"Label collision after tokenization: token={token}, "
                f"labels={token_to_canonical[token]!r} and {canonical!r}"
            )
        token_to_canonical[token] = canonical
        canonical_to_token[canonical] = token
        if canonical not in canonical_labels:
            canonical_labels.append(canonical)

    if not canonical_labels:
        raise ValueError("Empty label space after parsing.")

    TASK_NAME = t
    ANSWER_LABELS = canonical_labels
    ANSWER_TOKEN_TO_CANONICAL = token_to_canonical
    ANSWER_CANONICAL_TO_TOKEN = canonical_to_token
    ANSWER_LASTLINE_RE = _build_answer_regex(list(token_to_canonical.keys()))
    return TASK_NAME, ANSWER_LABELS


def _read_json_or_jsonl(path: str) -> Any:
    p = str(path)
    if p.lower().endswith(".jsonl"):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error in {p}:{i}: {e}") from e
        return rows
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_data_path_for_task(task_name: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    t = (task_name or "").strip().lower()
    if t == "medqa":
        # Prefer the question folder so all JSONL shards can be discovered.
        return os.path.join(base_dir, "MedQA", "data_clean", "questions")
    # Default pubmedqa
    return os.path.join(base_dir, "Pubmedqa", "pqal_question_context_groundtruth.json")


def resolve_data_path_arg(data_path_arg: str, task_name: str) -> str:
    arg = (data_path_arg or "").strip()
    if not arg:
        return _default_data_path_for_task(task_name)

    alias = arg.lower()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if alias in {"pubmedqa", "pubmed", "pqal"}:
        return os.path.join(base_dir, "Pubmedqa")
    if alias in {"medqa"}:
        return os.path.join(base_dir, "MedQA")
    return arg


def _discover_data_files(path: str, task_name: str) -> List[str]:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Data path not found: {path}")

    t = (task_name or "").strip().lower()

    if t == "pubmedqa":
        preferred = [
            os.path.join(p, "pqal_question_context_groundtruth.json"),
            os.path.join(p, "Pubmedqa", "pqal_question_context_groundtruth.json"),
            os.path.join(p, "pubmedqa", "pqal_question_context_groundtruth.json"),
        ]
        for c in preferred:
            if os.path.isfile(c):
                return [c]

        fallback = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        if fallback:
            return [fallback[0]]
        raise FileNotFoundError(
            f"No PubMedQA json file found under directory: {path}. "
            "Expected pqal_question_context_groundtruth.json."
        )

    if t == "medqa":
        # Prefer QA shards under questions/ and avoid unrelated jsonl files.
        primary = sorted(glob.glob(os.path.join(p, "**", "questions", "**", "*.jsonl"), recursive=True))
        primary = [x for x in primary if os.path.isfile(x)]
        if primary:
            return primary

        fallback_patterns = [
            os.path.join(p, "**", "*.jsonl"),
            os.path.join(p, "**", "*.json"),
        ]
        files: List[str] = []
        for pat in fallback_patterns:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted([x for x in files if os.path.isfile(x)])
        if files:
            return files
        raise FileNotFoundError(f"No MedQA json/jsonl files found under directory: {path}")

    # Generic fallback
    files = sorted(
        [x for x in glob.glob(os.path.join(p, "**", "*.json*"), recursive=True) if os.path.isfile(x)]
    )
    if files:
        return files
    raise FileNotFoundError(f"No json/jsonl files found under directory: {path}")


def _sorted_choice_items(choices: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(choices, dict):
        return out
    for k, v in choices.items():
        kk = str(k).strip()
        vv = str(v).strip()
        if kk:
            out.append((kk, vv))
    out.sort(key=lambda x: x[0])
    return out


def _build_default_context(ex: Dict[str, Any], choices: Dict[str, str]) -> str:
    parts: List[str] = []
    if "context" in ex and str(ex.get("context", "")).strip():
        parts.append(str(ex.get("context", "")).strip())

    if choices:
        choice_lines = [f"{k}. {v}" for k, v in _sorted_choice_items(choices)]
        parts.append("Options:\n" + "\n".join(choice_lines))

    meta = str(ex.get("meta_info", "")).strip()
    if meta:
        parts.append(f"Meta: {meta}")

    phrases = ex.get("metamap_phrases", [])
    if isinstance(phrases, list) and phrases:
        pv = ", ".join([str(x) for x in phrases if str(x).strip()])[:1000]
        if pv:
            parts.append(f"MetaMap phrases: {pv}")

    return "\n\n".join([p for p in parts if p.strip()])


# =========================================================
# OpenAI-compatible chat client (optional for synthesis)
# - Works with OpenAI and local OpenAI-compatible servers
# =========================================================
class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = int(timeout)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        import requests  # local import to keep dependency optional
        max_retries = int(os.environ.get("TEACHER_MAX_RETRIES", "8"))
        base_sleep = float(os.environ.get("TEACHER_RETRY_BASE_SLEEP", "2.0"))
        url = self.base_url
        if not url.endswith("/v1"):
            url = url + "/v1"
        url = url + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After", "").strip()
                    try:
                        sleep_s = float(retry_after) if retry_after else (base_sleep * (2 ** attempt))
                    except Exception:
                        sleep_s = base_sleep * (2 ** attempt)
                    if attempt >= max_retries:
                        r.raise_for_status()
                    print(
                        f"[TEACHER RETRY] status={r.status_code} attempt={attempt + 1}/{max_retries + 1} "
                        f"sleep={sleep_s:.1f}s"
                    )
                    time.sleep(min(sleep_s, 120.0))
                    continue
                r.raise_for_status()
                data = r.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    return ""
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt >= max_retries:
                    raise
                sleep_s = min(base_sleep * (2 ** attempt), 120.0)
                print(
                    f"[TEACHER RETRY] exception={type(e).__name__} attempt={attempt + 1}/{max_retries + 1} "
                    f"sleep={sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
        if last_err is not None:
            raise last_err
        return ""


def get_teacher_client_from_env() -> Optional[OpenAICompatClient]:
    base_url = os.environ.get("TEACHER_BASE_URL", "").strip()
    api_key = os.environ.get("TEACHER_API_KEY", "").strip()
    model = os.environ.get("TEACHER_MODEL", "").strip()
    timeout = int(os.environ.get("TEACHER_TIMEOUT", "60"))
    if not base_url or not model:
        return None
    return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


# =========================================================
# Load task dataset as unified rows.
# Supports:
# 1) dict keyed by example_id string
# 2) list of examples
# 3) jsonl list of examples
# =========================================================
def _next_unique_id(seen: set, next_auto: int, candidate: Optional[Any]) -> Tuple[int, int]:
    if candidate is not None and str(candidate).strip():
        try:
            eid = int(candidate)
        except Exception:
            eid = None
        if eid is not None and eid not in seen:
            seen.add(eid)
            return eid, max(next_auto, eid + 1)

    while next_auto in seen:
        next_auto += 1
    eid = int(next_auto)
    seen.add(eid)
    return eid, next_auto + 1


def load_raw_dataset(path: str, task_name: str = "") -> List[Dict[str, Any]]:
    effective_task = (task_name or TASK_NAME).strip().lower()
    files = _discover_data_files(path, effective_task)
    rows: List[Dict[str, Any]] = []

    seen_ids: set = set()
    next_auto_id = 0

    for fp in files:
        raw = _read_json_or_jsonl(fp)

        if isinstance(raw, dict):
            iterable = raw.items()
            keyed = True
        elif isinstance(raw, list):
            iterable = enumerate(raw)
            keyed = False
        else:
            raise ValueError(f"Unsupported JSON top-level type in {fp}: {type(raw)}")

        for k, ex in iterable:
            if not isinstance(ex, dict):
                raise ValueError(f"Unsupported example type in {fp} key/index {k!r}: {type(ex)}")

            eid_candidate = k if keyed else ex.get("example_id", None)
            eid, next_auto_id = _next_unique_id(seen_ids, next_auto_id, eid_candidate)

            question = str(ex.get("question", "")).strip()
            choices = ex.get("choices", ex.get("options", {}))
            if not isinstance(choices, dict):
                choices = {}
            norm_choices = {str(kk).strip(): str(vv).strip() for kk, vv in choices.items() if str(kk).strip()}

            context = str(ex.get("context", "")).strip()
            if not context:
                context = _build_default_context(ex, norm_choices)

            gt_raw = ex.get("ground_truth", ex.get("answer_idx", ex.get("label", ex.get("answer_label", ""))))
            gt = _normalize_label(gt_raw)

            rows.append({
                "example_id": eid,
                "question": question,
                "context": context,
                "ground_truth": gt,
                "choices": norm_choices,
                "task_name": TASK_NAME,
                "source_file": fp,
            })

    allowed = set(ANSWER_LABELS)
    cleaned_rows: List[Dict[str, Any]] = []
    dropped_missing_q = 0
    dropped_bad_label = 0

    for r in rows:
        if not r["question"]:
            dropped_missing_q += 1
            continue
        if r["ground_truth"] not in allowed:
            dropped_bad_label += 1
            continue
        cleaned_rows.append(r)

    if not cleaned_rows:
        raise ValueError(
            f"No valid rows loaded from {path}. "
            f"dropped_missing_question={dropped_missing_q} dropped_bad_label={dropped_bad_label}"
        )
    if dropped_missing_q or dropped_bad_label:
        print(
            f"[DATA] dropped invalid rows: missing_question={dropped_missing_q} "
            f"bad_label={dropped_bad_label} kept={len(cleaned_rows)}"
        )
    return cleaned_rows


def load_raw_pubmedqa(path: str) -> List[Dict[str, Any]]:
    # Backward-compatible wrapper used by existing pipeline stages.
    return load_raw_dataset(path=path, task_name=TASK_NAME)


# =========================================================
# Stratified split: Train / Dev / Test (fixed ids)
# =========================================================
def _alloc_counts_stratified(label_counts: Dict[str, int], target: int) -> Dict[str, int]:
    labels = sorted(label_counts.keys())
    total = sum(label_counts.values())
    if total == 0:
        return {lab: 0 for lab in labels}

    floor_counts = {}
    frac = {}
    for lab in labels:
        x = target * (label_counts[lab] / total)
        floor_counts[lab] = int(np.floor(x))
        frac[lab] = x - floor_counts[lab]

    remainder = target - sum(floor_counts.values())
    order = sorted(labels, key=lambda lab: frac[lab], reverse=True)
    i = 0
    while remainder > 0 and i < 100000:
        lab = order[i % len(order)]
        floor_counts[lab] += 1
        remainder -= 1
        i += 1

    return floor_counts


def make_splits(
    rows: List[Dict[str, Any]],
    test_size: int = 200,
    dev_size: int = 160,
    seed: int = 42,
) -> Dict[str, List[int]]:
    if test_size + dev_size >= len(rows):
        raise ValueError("test_size + dev_size must be < total dataset size")

    rng = random.Random(seed)

    by_label: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_label[r["ground_truth"]].append(r["example_id"])

    for lab in by_label:
        rng.shuffle(by_label[lab])

    full_counts = {lab: len(ids) for lab, ids in by_label.items()}
    test_counts = _alloc_counts_stratified(full_counts, test_size)

    test_ids = set()
    remaining_by_label: Dict[str, List[int]] = {}
    for lab, ids in by_label.items():
        n = min(test_counts.get(lab, 0), len(ids))
        test_part = ids[:n]
        rest = ids[n:]
        test_ids.update(test_part)
        remaining_by_label[lab] = rest

    if len(test_ids) < test_size:
        need = test_size - len(test_ids)
        pool = []
        for lab in remaining_by_label:
            pool.extend(remaining_by_label[lab])
        rng.shuffle(pool)
        take = pool[:need]
        test_ids.update(take)
        take_set = set(take)
        for lab in remaining_by_label:
            remaining_by_label[lab] = [x for x in remaining_by_label[lab] if x not in take_set]

    assert len(test_ids) == test_size

    rem_counts = {lab: len(ids) for lab, ids in remaining_by_label.items()}
    dev_counts = _alloc_counts_stratified(rem_counts, dev_size)

    dev_ids = set()
    train_ids = set()
    for lab, ids in remaining_by_label.items():
        n = min(dev_counts.get(lab, 0), len(ids))
        dev_part = ids[:n]
        train_part = ids[n:]
        dev_ids.update(dev_part)
        train_ids.update(train_part)

    if len(dev_ids) < dev_size:
        need = dev_size - len(dev_ids)
        pool = list(train_ids)
        rng.shuffle(pool)
        take = pool[:need]
        dev_ids.update(take)
        train_ids.difference_update(set(take))

    assert len(dev_ids) == dev_size
    assert len(train_ids) + len(dev_ids) + len(test_ids) == len(rows)

    return {
        "train_ids": sorted(list(train_ids)),
        "dev_ids": sorted(list(dev_ids)),
        "test_ids": sorted(list(test_ids)),
    }


def subsample_rows(rows: List[Dict[str, Any]], max_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= len(rows):
        return rows

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[str(r["ground_truth"])].append(r)
    for lb in by_label:
        rng.shuffle(by_label[lb])

    wanted = _alloc_counts_stratified({lb: len(items) for lb, items in by_label.items()}, max_samples)
    sampled: List[Dict[str, Any]] = []
    for lb, items in by_label.items():
        sampled.extend(items[:wanted.get(lb, 0)])

    if len(sampled) < max_samples:
        sampled_ids = {int(x["example_id"]) for x in sampled}
        rest = [r for r in rows if int(r["example_id"]) not in sampled_ids]
        rng.shuffle(rest)
        sampled.extend(rest[: max_samples - len(sampled)])

    sampled = sampled[:max_samples]
    sampled.sort(key=lambda x: int(x["example_id"]))
    return sampled


# =========================================================
# Sentence splitting + candidate selection
# =========================================================
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def split_into_sentences(context: str) -> List[str]:
    if not context:
        return []
    txt = context.replace("\n", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    parts = re.split(r"(?<=[\.\?\!;])\s+", txt)
    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 30:
            continue
        if len(p) > 500:
            p = p[:500]
        sents.append(p)
    if not sents and txt:
        sents = [txt[:500]]
    return sents


def build_numbered_sentences(context: str) -> List[Dict[str, Any]]:
    sents = split_into_sentences(context)
    return [{"sid": i, "text": s} for i, s in enumerate(sents)]


def format_numbered_sentences(items: List[Dict[str, Any]], header: str) -> str:
    lines = [f"[{int(it['sid'])}] {str(it['text'])}" for it in items if isinstance(it, dict) and "sid" in it and "text" in it]
    if not lines:
        return f"{header}:\n"
    return f"{header}:\n" + "\n".join(lines) + "\n"


def format_choices_block(choices: Optional[Dict[str, str]], header: str = "Choices") -> str:
    if not isinstance(choices, dict) or not choices:
        return ""
    items = _sorted_choice_items(choices)
    if not items:
        return ""
    lines = "\n".join([f"{k}. {v}" for k, v in items])
    return f"{header}:\n{lines}\n\n"


def overlap_score(q_words: List[str], s_words: List[str]) -> float:
    if not q_words or not s_words:
        return 0.0
    qs = set(q_words)
    ss = set(s_words)
    inter = len(qs.intersection(ss))
    return inter / (1.0 + 0.05 * len(ss))


def build_candidates(
    question: str,
    context: str,
    top_k: int,
    rng: random.Random,
    choices: Optional[Dict[str, str]] = None,
    task_name: str = "",
) -> List[Dict[str, Any]]:
    q_words = tokenize_words(question)
    t = (task_name or TASK_NAME).strip().lower()
    choice_text = " ".join([str(v) for _, v in _sorted_choice_items(choices or {})])
    choice_words = tokenize_words(choice_text)
    sents = split_into_sentences(context)
    cands = []
    for i, s in enumerate(sents):
        s_words = tokenize_words(s)
        q_sc = overlap_score(q_words, s_words)
        sc = q_sc
        if t == "medqa" and choice_words:
            choice_sc = overlap_score(choice_words, s_words)
            sc = (0.85 * q_sc) + (0.35 * choice_sc)
        sc = sc + rng.uniform(-0.02, 0.02)
        cands.append({"sid": i, "text": s, "score": float(sc)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:top_k]


def pick_evidence(candidates: List[Dict[str, Any]], n_min: int, n_max: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    n = rng.randint(n_min, n_max)
    n = min(n, len(candidates))

    scores = np.array([max(0.001, c["score"] + 0.05) for c in candidates], dtype=np.float32)
    probs = scores / scores.sum()
    idxs = rng.choices(list(range(len(candidates))), weights=probs.tolist(), k=n * 2)

    uniq, seen = [], set()
    for ix in idxs:
        if ix in seen:
            continue
        uniq.append(ix)
        seen.add(ix)
        if len(uniq) >= n:
            break
    ev = [candidates[ix] for ix in uniq]
    ev.sort(key=lambda x: x["score"], reverse=True)
    return ev[:n]


# =========================================================
# Tool schemas (no final label)
# =========================================================
def get_reasoning_system_prompt(task_name: str = "") -> str:
    t = (task_name or TASK_NAME).strip().lower()
    if t == "medqa":
        intro = (
            "You are a clinical reasoning tool for medical multiple-choice questions.\n"
            "Given a QUESTION, CHOICES, and NUMBERED CANDIDATE SENTENCES, extract grounded reasoning signals that help discriminate between answer options.\n"
            "Focus on patient clues, labs, imaging, mechanism, disease scripts, and option discriminators.\n"
        )
        task_specific = (
            "Polarity means whether a sentence supports, opposes, or is unclear with respect to the best answer implied by the QUESTION and CHOICES.\n"
            "Counterpoints should highlight tempting distractors or why a clue may be nonspecific.\n"
        )
    elif t == "pubmedqa":
        intro = (
            "You are a clinical reasoning tool for PubMedQA-style questions.\n"
            "Given a QUESTION and NUMBERED CANDIDATE SENTENCES, output grounded reasoning signals.\n"
        )
        task_specific = (
            "Polarity means whether a sentence supports, opposes, or is unclear with respect to the proposition asked in the QUESTION.\n"
        )
    else:
        intro = (
            "You are a clinical reasoning tool.\n"
            "Given a QUESTION and NUMBERED CANDIDATE SENTENCES, output grounded reasoning signals.\n"
        )
        task_specific = "Polarity means whether a sentence supports, opposes, or is unclear with respect to the main question.\n"

    return (
        intro
        + "Use only the provided candidate sentences.\n"
        + task_specific
        + "Do not invent sentence ids or sentence text.\n"
        + "Do not provide the final answer label.\n"
        + "Return ONLY a valid JSON object with keys:\n"
        + "  evidence: list of {sid:int, text:str, polarity:str in [support, oppose, unclear]}\n"
        + "  reasoning_steps: list of short strings (2~5 items)\n"
        + "  counterpoints: list of short strings (0~3 items)\n"
        + "  uncertainty_flags: list[str] (0~3)\n"
        + "  confidence: float 0.0~1.0\n"
        + "Constraints:\n"
        + "- evidence length <= 6\n"
        + "- every evidence sid must come from the provided candidate sentences\n"
        + "- every evidence text must copy or minimally trim a provided candidate sentence\n"
        + "- each evidence text <= 240 chars\n"
        + "- each reasoning step <= 180 chars\n"
        + "- output JSON only, no extra text.\n"
    )


def get_context_system_prompt(task_name: str = "") -> str:
    t = (task_name or TASK_NAME).strip().lower()
    if t == "medqa":
        intro = (
            "You are a context extraction tool for medical multiple-choice questions.\n"
            "Given a QUESTION, CHOICES, and NUMBERED CONTEXT SENTENCES, extract the most decision-relevant clues.\n"
            "Focus on demographics, timeline, exam findings, labs, imaging, mechanism clues, and sentences that separate plausible options.\n"
        )
    elif t == "pubmedqa":
        intro = (
            "You are a context extraction tool for PubMedQA-style questions.\n"
            "Given a QUESTION and NUMBERED CONTEXT SENTENCES, extract decision-relevant signals.\n"
        )
    else:
        intro = (
            "You are a context extraction tool.\n"
            "Given a QUESTION and NUMBERED CONTEXT SENTENCES, extract decision-relevant signals.\n"
        )

    return (
        intro
        + "Use only the provided numbered sentences.\n"
        + "Do not invent sentence ids or sentence text.\n"
        + "Do not provide the final answer label.\n"
        + "Return ONLY a valid JSON object with keys:\n"
        + "  key_sentences: list of {sid:int, text:str} (up to 6)\n"
        + "  context_summary: short string (<=260 chars)\n"
        + "  uncertainty_flags: list[str] (0~3)\n"
        + "  confidence: float 0.0~1.0\n"
        + "Constraints:\n"
        + "- every key_sentences sid must come from the provided numbered sentences\n"
        + "- every key_sentences text must copy or minimally trim a provided sentence\n"
        + "- output JSON only, no extra text.\n"
    )

# Weak mode: DO NOT use ground_truth for any field.
# Keep signals generic to avoid label leakage.
WEAK_UNCERTAINTY_FLAGS = ["weak_supervision_generation"]


# =========================================================
# Build tool SFT datasets (optional GPT synth)
# =========================================================
def build_tool_sft_data_from_splits(
    data_path: str,
    split_path: str,
    out_dir: str,
    seed: int = 42,
    top_k: int = 20,
    variants_train: int = 3,
    variants_dev: int = 2,
    ev_min: int = 3,
    ev_max: int = 6,
    synth_mode: str = "weak",  # weak | gpt
):
    set_seed(seed)
    rows = load_raw_pubmedqa(data_path)
    splits = read_json(split_path)
    train_ids = set(splits["train_ids"])
    dev_ids = set(splits["dev_ids"])
    test_ids = set(splits["test_ids"])
    assert len(train_ids & dev_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(dev_ids & test_ids) == 0

    id2ex = {r["example_id"]: r for r in rows}
    teacher = get_teacher_client_from_env() if synth_mode == "gpt" else None
    if synth_mode == "gpt" and teacher is None:
        raise ValueError(
            "synth_mode='gpt' requires TEACHER_BASE_URL and TEACHER_MODEL. "
            "Set env vars or switch to --tool_synth_mode weak."
        )

    tool_reason_train, tool_reason_dev = [], []
    tool_ctx_train, tool_ctx_dev = [], []

    def gpt_make_reasoning(
        q: str,
        candidates: List[Dict[str, Any]],
        choices: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if teacher is None:
            return None

        cand_block = format_numbered_sentences(candidates, "NUMBERED CANDIDATE SENTENCES")
        choices_block = format_choices_block(choices)
        user = f"QUESTION:\n{q}\n\n{choices_block}{cand_block}"

        last_raw = None
        for attempt in range(3):
            raw = teacher.chat(
                [
                    {"role": "system", "content": get_reasoning_system_prompt()},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            last_raw = raw
            obj = extract_first_json(raw)
            print(f"[DEBUG] GPT reasoning synthesis raw response:\n{raw}\nExtracted obj:\n{obj}\n")
            if obj is not None:
                return obj

        raise ValueError(
            "GPT synthesis failed for reasoning_tool after 3 retries: response is not valid JSON object.\n"
            f"Last raw response:\n{last_raw}"
        )


    def gpt_make_context(
        q: str,
        numbered_context: List[Dict[str, Any]],
        choices: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if teacher is None:
            return None
        ctx_block = format_numbered_sentences(numbered_context, "NUMBERED CONTEXT SENTENCES")
        choices_block = format_choices_block(choices)
        user = f"QUESTION:\n{q}\n\n{choices_block}{ctx_block}"
        raw = teacher.chat([{"role": "system", "content": get_context_system_prompt()}, {"role": "user", "content": user}], temperature=0.2)
        obj = extract_first_json(raw)
        if obj is None:
            raise ValueError(
                "GPT synthesis failed for context_tool: response is not valid JSON object. "
                "Use --tool_synth_mode weak if you want fallback behavior."
            )
        return obj

    def normalize_reasoning_obj(
        obj_r: Dict[str, Any],
        candidate_text_by_sid: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        ev = obj_r.get("evidence", [])
        if not isinstance(ev, list):
            ev = []
        norm_ev = []
        for it in ev[:6]:
            if isinstance(it, dict):
                sid_raw = it.get("sid", -1)
                sid = int(sid_raw) if str(sid_raw).lstrip("-").isdigit() else -1
                if candidate_text_by_sid is not None and sid not in candidate_text_by_sid:
                    continue
                txt = str(it.get("text", ""))[:240]
                if candidate_text_by_sid is not None:
                    txt = candidate_text_by_sid[sid][:240]
                pol = str(it.get("polarity", "unclear"))
                if pol not in ["support", "oppose", "unclear"]:
                    pol = "unclear"
                norm_ev.append({"sid": sid, "text": txt, "polarity": pol})
        obj_r["evidence"] = norm_ev

        if not isinstance(obj_r.get("reasoning_steps"), list):
            obj_r["reasoning_steps"] = []
        obj_r["reasoning_steps"] = [str(x)[:180] for x in obj_r["reasoning_steps"][:5]]

        if not isinstance(obj_r.get("counterpoints"), list):
            obj_r["counterpoints"] = []
        obj_r["counterpoints"] = [str(x)[:180] for x in obj_r["counterpoints"][:3]]

        uf = obj_r.get("uncertainty_flags", [])
        if not isinstance(uf, list):
            uf = []
        obj_r["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]]

        try:
            obj_r["confidence"] = float(obj_r.get("confidence", 0.6))
        except Exception:
            obj_r["confidence"] = 0.6
        obj_r["confidence"] = max(0.0, min(1.0, obj_r["confidence"]))
        return obj_r

    def normalize_context_obj(
        obj_c: Dict[str, Any],
        sentence_text_by_sid: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        ks = obj_c.get("key_sentences", [])
        if not isinstance(ks, list):
            ks = []
        norm_ks = []
        for it in ks[:6]:
            if isinstance(it, dict):
                sid_raw = it.get("sid", -1)
                sid = int(sid_raw) if str(sid_raw).lstrip("-").isdigit() else -1
                if sentence_text_by_sid is not None and sid not in sentence_text_by_sid:
                    continue
                txt = str(it.get("text", ""))[:240]
                if sentence_text_by_sid is not None:
                    txt = sentence_text_by_sid[sid][:240]
                norm_ks.append({"sid": sid, "text": txt})
        obj_c["key_sentences"] = norm_ks

        if not isinstance(obj_c.get("context_summary"), str):
            obj_c["context_summary"] = ""
        obj_c["context_summary"] = obj_c["context_summary"][:260]

        uf = obj_c.get("uncertainty_flags", [])
        if not isinstance(uf, list):
            uf = []
        obj_c["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]]

        try:
            obj_c["confidence"] = float(obj_c.get("confidence", 0.6))
        except Exception:
            obj_c["confidence"] = 0.6
        obj_c["confidence"] = max(0.0, min(1.0, obj_c["confidence"]))
        return obj_c

    def add_one(
        eid: int,
        variants: int,
        reason_list: List[Dict[str, Any]],
        ctx_list: List[Dict[str, Any]],
    ):
        ex = id2ex[eid]
        q, ctx = ex["question"], ex["context"]
        choices = ex.get("choices", {})
        base_rng = random.Random(seed * 100000 + eid)
        numbered_context = build_numbered_sentences(ctx)
        sentence_text_by_sid = {int(it["sid"]): str(it["text"]) for it in numbered_context}

        for _ in range(variants):
            rng = random.Random(base_rng.randint(0, 10**9))
            candidates = build_candidates(q, ctx, top_k=top_k, rng=rng, choices=choices, task_name=TASK_NAME)
            evidence = pick_evidence(candidates, n_min=ev_min, n_max=ev_max, rng=rng)
            candidate_text_by_sid = {int(c["sid"]): str(c["text"]) for c in candidates}

            # ---------- reasoning tool target ----------
            obj_r = None
            if synth_mode == "gpt":
                obj_r = gpt_make_reasoning(q, candidates, choices=choices)

            if obj_r is None:
                if synth_mode == "gpt":
                    raise RuntimeError("Unexpected empty reasoning object in strict gpt synthesis mode.")
                # Weak mode: no label usage, keep polarity unclear
                ev_items = [{"sid": int(e["sid"]), "text": str(e["text"])[:240], "polarity": "unclear"} for e in evidence[:6]]
                obj_r = {
                    "evidence": ev_items,
                    "reasoning_steps": [
                        f"Restate the question focus: {q[:120]}",
                        "Review candidate sentences and identify relevant study signals.",
                        "Note limitations and indirect evidence.",
                    ][:5],
                    "counterpoints": ["Evidence may be indirect or underpowered."][:3],
                    "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:3],
                    "confidence": 0.6,
                }

            obj_r = normalize_reasoning_obj(obj_r, candidate_text_by_sid=candidate_text_by_sid)

            cand_block = format_numbered_sentences(candidates, "NUMBERED CANDIDATE SENTENCES")
            choices_block = format_choices_block(choices)
            user_r = f"Example ID: {eid}\nQUESTION:\n{q}\n\n{choices_block}{cand_block}"
            reason_list.append({
                "example_id": eid,
                "prompt": [{"role": "system", "content": get_reasoning_system_prompt()}, {"role": "user", "content": user_r}],
                "response": dumps_json(obj_r),
            })

            # ---------- context tool target ----------
            obj_c = None
            if synth_mode == "gpt":
                obj_c = gpt_make_context(q, numbered_context, choices=choices)

            if obj_c is None:
                if synth_mode == "gpt":
                    raise RuntimeError("Unexpected empty context object in strict gpt synthesis mode.")
                key_sentences = [{"sid": int(c["sid"]), "text": str(c["text"])[:240]} for c in candidates[:6]]
                obj_c = {
                    "key_sentences": key_sentences,
                    "context_summary": "Decision-relevant signals extracted from context.",
                    "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:3],
                    "confidence": 0.6,
                }

            obj_c = normalize_context_obj(obj_c, sentence_text_by_sid=sentence_text_by_sid)

            ctx_block = format_numbered_sentences(numbered_context, "NUMBERED CONTEXT SENTENCES")
            choices_block = format_choices_block(choices)
            user_c = f"Example ID: {eid}\nQUESTION:\n{q}\n\n{choices_block}{ctx_block}"
            ctx_list.append({
                "example_id": eid,
                "prompt": [{"role": "system", "content": get_context_system_prompt()}, {"role": "user", "content": user_c}],
                "response": dumps_json(obj_c),
            })

    for eid in sorted(list(train_ids)):
        add_one(eid, variants_train, tool_reason_train, tool_ctx_train)
    for eid in sorted(list(dev_ids)):
        add_one(eid, variants_dev, tool_reason_dev, tool_ctx_dev)

    os.makedirs(out_dir, exist_ok=True)
    reason_train_path = os.path.join(out_dir, "tool_reasoning_train.jsonl")
    reason_dev_path = os.path.join(out_dir, "tool_reasoning_dev.jsonl")
    ctx_train_path = os.path.join(out_dir, "tool_context_train.jsonl")
    ctx_dev_path = os.path.join(out_dir, "tool_context_dev.jsonl")

    write_jsonl(reason_train_path, tool_reason_train)
    write_jsonl(reason_dev_path, tool_reason_dev)
    write_jsonl(ctx_train_path, tool_ctx_train)
    write_jsonl(ctx_dev_path, tool_ctx_dev)

    print(f"[TOOL SFT DATA] reasoning train/dev: {len(tool_reason_train)} / {len(tool_reason_dev)}")
    print(f"[TOOL SFT DATA] context   train/dev: {len(tool_ctx_train)} / {len(tool_ctx_dev)}")
    print(f"[TOOL SFT DATA] wrote: {out_dir}")
    return reason_train_path, reason_dev_path, ctx_train_path, ctx_dev_path


# =========================================================
# Tokenize SFT dataset
# =========================================================
def tokenize_sft_dataset(ds: Dataset, tokenizer: Any, max_seq_len: int) -> Dataset:
    eos = tokenizer.eos_token or ""

    def _normalize_response_messages(response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, str):
            return [{"role": "assistant", "content": response}]
        if isinstance(response, dict):
            msg = dict(response)
            msg.setdefault("role", "assistant")
            return [msg]
        if isinstance(response, list):
            out = []
            for item in response:
                if not isinstance(item, dict):
                    raise ValueError(f"Unsupported response message type: {type(item)}")
                msg = dict(item)
                msg.setdefault("role", "assistant")
                out.append(msg)
            return out
        raise ValueError(f"Unsupported response type for SFT tokenization: {type(response)}")

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response_msgs = _normalize_response_messages(ex["response"])

        prompt_text = render_chat_messages(tokenizer, prompt_msgs, add_generation_prompt=True)
        full_text = render_chat_messages(tokenizer, prompt_msgs + response_msgs, add_generation_prompt=False) + eos

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False)

        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]

        prompt_len = min(len(prompt_ids), max_seq_len)
        labels = ([-100] * prompt_len) + input_ids[prompt_len:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return ds.map(_map, remove_columns=ds.column_names)


# =========================================================
# Train SFT agent (tool models)
# =========================================================
def train_sft_agent(
    tool_base_model: str,
    train_jsonl: str,
    dev_jsonl: str,
    out_dir: str,
    seed: int = 42,
    max_seq_len: int = 2048,
    lr: float = 2e-4,
    epochs: int = 2,
    per_device_bs: int = 1,
    grad_accum: int = 8,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    require_datasets("train_sft_agent")
    set_seed(seed)
    device = get_runtime_device()
    validate_distributed_runtime("train_sft_agent", require_cuda=(get_world_size() > 1))

    tok = AutoTokenizer.from_pretrained(tool_base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = get_runtime_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        tool_base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_lora:
        if not PEFT_AVAILABLE:
            print("[WARN] peft not available -> full finetune.")
        else:
            common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            present = set([name.split(".")[-1] for name, _ in model.named_modules()])
            target_modules = [m for m in common if m in present]
            if not target_modules:
                target_modules = ["q_proj", "v_proj"]

            lconf = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lconf)
            print(f"[LoRA] target_modules={target_modules}")

    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": dev_jsonl})
    train_ds = tokenize_sft_dataset(ds["train"], tok, max_seq_len=max_seq_len)
    dev_ds = tokenize_sft_dataset(ds["validation"], tok, max_seq_len=max_seq_len)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[SFT] saved to: {out_dir}")


# =========================================================
# Manager + Tools runtime
# =========================================================
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*.+?\s*</tool_call>", re.IGNORECASE | re.DOTALL)
TOOLS_TAG_RE = re.compile(r"<tools>.*?</tools>", re.IGNORECASE | re.DOTALL)
TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
TOOL_JSON_RE = re.compile(
    r'"name"\s*:\s*"(reasoning_tool|context_tool)".*?"arguments"\s*:',
    re.IGNORECASE | re.DOTALL,
)
PLAIN_TOOL_NAME_RE = re.compile(
    r"^\s*(reasoning_tool|context_tool)\s*(?:\(|\{|:|$)",
    re.IGNORECASE | re.MULTILINE,
)

MANAGER_TOOL_BINDING_MODE = "argument"  # argument | environment


def _manager_tools_require_example_id() -> bool:
    return MANAGER_TOOL_BINDING_MODE != "environment"


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(dumps_json(item) if isinstance(item, (dict, list)) else str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        return dumps_json(content)
    return str(content)


def _fallback_render_messages(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip() or "user"
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name = str(fn.get("name", "")).strip()
                arguments = fn.get("arguments", "{}")
                payload = {"name": tool_name, "arguments": arguments}
                parts.append(f"assistant_tool_call: {dumps_json(payload)}")
            content = _message_content_to_text(m.get("content"))
            if content:
                parts.append(f"assistant: {content}")
            continue
        if role == "tool":
            tool_name = str(m.get("name", "tool")).strip() or "tool"
            tool_call_id = str(m.get("tool_call_id", "")).strip()
            prefix = f"tool[{tool_name}]"
            if tool_call_id:
                prefix += f"#{tool_call_id}"
            parts.append(f"{prefix}: {_message_content_to_text(m.get('content'))}")
            continue
        parts.append(f"{role}: {_message_content_to_text(m.get('content'))}")
    if add_generation_prompt:
        parts.append("assistant: ")
    return "\n".join(parts)


def render_chat_messages(tokenizer: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return _fallback_render_messages(messages, add_generation_prompt=add_generation_prompt)
    except Exception:
        return _fallback_render_messages(messages, add_generation_prompt=add_generation_prompt)


def parse_answer_label_lastline(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    m = ANSWER_LASTLINE_RE.match(last_line)
    if not m:
        return None
    tok = m.group(1).upper()
    return ANSWER_TOKEN_TO_CANONICAL.get(tok)


def final_has_tool_call_artifacts(text: str) -> bool:
    if not text:
        return False
    txt = str(text)
    if TOOL_CALL_TAG_RE.search(txt):
        return True
    if TOOLS_TAG_RE.search(txt):
        return True
    if TOOL_CALLS_FIELD_RE.search(txt):
        return True
    if TOOL_JSON_RE.search(txt):
        return True
    if PLAIN_TOOL_NAME_RE.search(txt):
        return True
    return False


def ensure_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if len(x) == 0:
            return [None] * n
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n


def extract_stats(completion_msgs: Any) -> Dict[str, Any]:
    if not isinstance(completion_msgs, list):
        txt = _message_content_to_text(completion_msgs)
        has_tool_text = final_has_tool_call_artifacts(txt)
        return {
            "assistant_texts": [txt],
            "tool_msg_count": 0,
            "tool_call_count": 0,
            "tool_names": [],
            "tool_payloads": [],
            "last_assistant_text": txt,
            "last_assistant_has_tool_calls": False,
            "last_assistant_plaintext_tool_artifacts": bool(has_tool_text),
            "fake_tool_text_attempt": bool(has_tool_text),
        }

    assistant_msgs = [m for m in completion_msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs = [m for m in completion_msgs if isinstance(m, dict) and m.get("role") == "tool"]
    tool_msg_count = len(tool_msgs)

    tool_call_count = 0
    for m in assistant_msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            tool_call_count += len(tc)

    tool_names = []
    tool_payloads = []
    for m in tool_msgs:
        tool_names.append("" if m.get("name") is None else str(m.get("name")))
        tool_payloads.append(_message_content_to_text(m.get("content")))

    assistant_texts = []
    for m in assistant_msgs:
        assistant_texts.append(_message_content_to_text(m.get("content")))

    last_assistant_text = assistant_texts[-1] if assistant_texts else ""
    last_assistant_has_tool_calls = bool(assistant_msgs[-1].get("tool_calls")) if assistant_msgs else False

    any_tool_artifacts_anywhere = any(final_has_tool_call_artifacts(t) for t in assistant_texts)
    fake_tool_text_attempt = bool(any_tool_artifacts_anywhere and (tool_msg_count == 0 and tool_call_count == 0))

    return {
        "assistant_texts": assistant_texts,
        "tool_msg_count": tool_msg_count,
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "tool_payloads": tool_payloads,
        "last_assistant_text": last_assistant_text,
        "last_assistant_has_tool_calls": last_assistant_has_tool_calls,
        "last_assistant_plaintext_tool_artifacts": bool(final_has_tool_call_artifacts(last_assistant_text)),
        "fake_tool_text_attempt": fake_tool_text_attempt,
    }


# ---------- globals for tools ----------
# IMPORTANT: ID2EX must NOT contain ground_truth, to avoid any label leakage via tools.
ID2EX: Dict[int, Dict[str, Any]] = {}
REASONING_CACHE: Dict[int, str] = {}
CONTEXT_CACHE: Dict[int, str] = {}
REASONING_RAW_CACHE: Dict[int, str] = {}
CONTEXT_RAW_CACHE: Dict[int, str] = {}
ALLOWED_TOOL_IDS: Optional[set] = None

_IS_MAIN_PROCESS = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


@dataclass
class FrozenAgent:
    """HF model id or path for loading the tool (plus optional LoRA adapter)."""
    tool_base_model: str
    adapter_path: Optional[str] = None
    device: str = "cpu"
    max_new_tokens: int = 512

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)

        if self.adapter_path:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not available but adapter_path is set.")
            model = PeftModel.from_pretrained(model, self.adapter_path).to(self.device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)

        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = (temperature > 1e-6)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)

        out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolBase:
    """One base model with multiple PEFT adapters, switched at generation time."""
    tool_base_model: str
    reasoning_adapter_path: Optional[str] = None
    context_adapter_path: Optional[str] = None
    device: str = "cpu"

    def __post_init__(self):
        if not PEFT_AVAILABLE:
            raise RuntimeError("SharedToolBase requires `peft`.")

        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)

        primary_name = None
        primary_path = None
        if self.reasoning_adapter_path:
            primary_name = "reasoning_tool"
            primary_path = self.reasoning_adapter_path
        elif self.context_adapter_path:
            primary_name = "context_tool"
            primary_path = self.context_adapter_path

        if primary_path is None:
            raise RuntimeError("SharedToolBase requires at least one adapter path.")

        model = PeftModel.from_pretrained(base_model, primary_path, adapter_name=primary_name).to(self.device)
        self.adapter_names: Dict[str, Optional[str]] = {
            "reasoning_tool": primary_name if self.reasoning_adapter_path == primary_path else None,
            "context_tool": primary_name if self.context_adapter_path == primary_path else None,
        }

        if self.reasoning_adapter_path and self.reasoning_adapter_path != primary_path:
            model.load_adapter(self.reasoning_adapter_path, adapter_name="reasoning_tool")
            self.adapter_names["reasoning_tool"] = "reasoning_tool"
        if self.context_adapter_path and self.context_adapter_path != primary_path:
            model.load_adapter(self.context_adapter_path, adapter_name="context_tool")
            self.adapter_names["context_tool"] = "context_tool"

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.lock = threading.Lock()

    @torch.no_grad()
    def generate(
        self,
        tool_name: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        adapter_name = self.adapter_names.get(tool_name)
        if adapter_name is None:
            raise RuntimeError(f"Adapter for tool `{tool_name}` is not loaded.")

        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = (temperature > 1e-6)
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)

        with self.lock:
            self.model.set_adapter(adapter_name)
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolView:
    shared_base: SharedToolBase
    tool_name: str
    max_new_tokens: int = 512

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        return self.shared_base.generate(
            tool_name=self.tool_name,
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
        )


_shared_tool_base: Optional[SharedToolBase] = None
_reasoning_agent: Optional[Any] = None
_context_agent: Optional[Any] = None
TOOL_TOP_K: int = 20


def init_tool_agents(tool_base_model: str, reasoning_adapter: str, context_adapter: str, device: str):
    global _shared_tool_base, _reasoning_agent, _context_agent
    can_share = bool(PEFT_AVAILABLE and reasoning_adapter and context_adapter)

    if can_share and _shared_tool_base is None:
        try:
            _shared_tool_base = SharedToolBase(
                tool_base_model=tool_base_model,
                reasoning_adapter_path=reasoning_adapter,
                context_adapter_path=context_adapter,
                device=device,
            )
            _reasoning_agent = SharedToolView(_shared_tool_base, "reasoning_tool", max_new_tokens=640)
            _context_agent = SharedToolView(_shared_tool_base, "context_tool", max_new_tokens=400)
            print("[TOOLS] runtime=shared_base adapters=reasoning_tool,context_tool")
            return
        except Exception as e:
            print(f"[WARN] shared tool base init failed; falling back to split tool models. {type(e).__name__}: {e}")

    if _reasoning_agent is None:
        _reasoning_agent = FrozenAgent(tool_base_model, reasoning_adapter, device=device, max_new_tokens=640)
    if _context_agent is None:
        _context_agent = FrozenAgent(tool_base_model, context_adapter, device=device, max_new_tokens=400)
    if _shared_tool_base is None:
        print("[TOOLS] runtime=split_models")


def _tool_guard(eid: int) -> Optional[str]:
    if ALLOWED_TOOL_IDS is not None and eid not in ALLOWED_TOOL_IDS:
        return dumps_json({"error": f"example_id {eid} not allowed in current split"})
    return None


def _normalize_reasoning_output(
    obj: Dict[str, Any],
    candidate_text_by_sid: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    ev = obj.get("evidence", [])
    if not isinstance(ev, list):
        ev = []
    norm_ev = []
    for it in ev[:6]:
        if isinstance(it, dict):
            sid_raw = it.get("sid", -1)
            sid = int(sid_raw) if str(sid_raw).lstrip("-").isdigit() else -1
            if candidate_text_by_sid is not None and sid not in candidate_text_by_sid:
                continue
            txt = str(it.get("text", ""))[:240]
            if candidate_text_by_sid is not None:
                txt = candidate_text_by_sid[sid][:240]
            pol = str(it.get("polarity", "unclear"))
            if pol not in ["support", "oppose", "unclear"]:
                pol = "unclear"
            norm_ev.append({"sid": sid, "text": txt, "polarity": pol})
    obj["evidence"] = norm_ev

    if not isinstance(obj.get("reasoning_steps"), list):
        obj["reasoning_steps"] = []
    obj["reasoning_steps"] = [str(x)[:180] for x in obj["reasoning_steps"][:5]]

    if not isinstance(obj.get("counterpoints"), list):
        obj["counterpoints"] = []
    obj["counterpoints"] = [str(x)[:180] for x in obj["counterpoints"][:3]]

    uf = obj.get("uncertainty_flags", [])
    if not isinstance(uf, list):
        uf = []
    obj["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]]

    try:
        obj["confidence"] = float(obj.get("confidence", 0.6))
    except Exception:
        obj["confidence"] = 0.6
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def _normalize_context_output(
    obj: Dict[str, Any],
    sentence_text_by_sid: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    ks = obj.get("key_sentences", [])
    if not isinstance(ks, list):
        ks = []
    norm_ks = []
    for it in ks[:6]:
        if isinstance(it, dict):
            sid_raw = it.get("sid", -1)
            sid = int(sid_raw) if str(sid_raw).lstrip("-").isdigit() else -1
            if sentence_text_by_sid is not None and sid not in sentence_text_by_sid:
                continue
            txt = str(it.get("text", ""))[:240]
            if sentence_text_by_sid is not None:
                txt = sentence_text_by_sid[sid][:240]
            norm_ks.append({"sid": sid, "text": txt})
    obj["key_sentences"] = norm_ks

    if not isinstance(obj.get("context_summary"), str):
        obj["context_summary"] = ""
    obj["context_summary"] = obj["context_summary"][:260]

    uf = obj.get("uncertainty_flags", [])
    if not isinstance(uf, list):
        uf = []
    obj["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]]

    try:
        obj["confidence"] = float(obj.get("confidence", 0.6))
    except Exception:
        obj["confidence"] = 0.6
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def reasoning_tool(example_id: int) -> str:
    """Return reasoning signals for one example.

    Args:
        example_id: PubMedQA example id.

    Returns:
        JSON string with evidence and reasoning signals.
    """
    eid = int(example_id)
    guard = _tool_guard(eid)
    if guard is not None:
        REASONING_RAW_CACHE[eid] = guard
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "reasoning_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": False,
            "guard_error": True,
            "raw_output": guard,
        }])
        return guard
    if eid in REASONING_CACHE:
        raw_cached = REASONING_RAW_CACHE.get(eid, REASONING_CACHE[eid])
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "reasoning_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": True,
            "guard_error": False,
            "raw_output": raw_cached,
        }])
        return REASONING_CACHE[eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = dumps_json({"error": "example_id not found"})
        REASONING_CACHE[eid] = out
        REASONING_RAW_CACHE[eid] = out
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "reasoning_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": False,
            "guard_error": True,
            "raw_output": out,
        }])
        return out

    q = ex["question"]
    ctx = ex["context"]
    choices = ex.get("choices", {})

    rng = random.Random(12345 + eid)
    candidates = build_candidates(q, ctx, top_k=TOOL_TOP_K, rng=rng, choices=choices, task_name=TASK_NAME)
    candidate_text_by_sid = {int(c["sid"]): str(c["text"]) for c in candidates}
    cand_block = format_numbered_sentences(candidates, "NUMBERED CANDIDATE SENTENCES")
    choices_block = format_choices_block(choices)

    user = f"Example ID: {eid}\nQUESTION:\n{q}\n\n{choices_block}{cand_block}"
    msgs = [{"role": "system", "content": get_reasoning_system_prompt()}, {"role": "user", "content": user}]
    raw = _reasoning_agent.generate(msgs, temperature=0.0) if _reasoning_agent else ""
    REASONING_RAW_CACHE[eid] = raw
    _append_raw_trace_rows([{
        "ts": int(time.time()),
        "agent": "reasoning_tool",
        "event": "tool_call",
        "example_id": eid,
        "cache_hit": False,
        "guard_error": False,
        "raw_output": raw,
    }])
    obj = extract_first_json(raw)

    if obj is None:
        # Fallback without any label usage
        ev = [{"sid": int(c["sid"]), "text": c["text"][:240], "polarity": "unclear"} for c in candidates[:4]]
        obj = {
            "evidence": ev,
            "reasoning_steps": [
                f"Restate the question focus: {q[:120]}",
                "Scan evidence for study design, endpoints, and direction of effect.",
                "Note limitations and indirectness.",
            ],
            "counterpoints": ["Evidence may be indirect or underpowered."],
            "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:3],
            "confidence": 0.6,
        }

    obj = _normalize_reasoning_output(obj, candidate_text_by_sid=candidate_text_by_sid)
    out = dumps_json(obj)
    REASONING_CACHE[eid] = out
    return out


def context_tool(example_id: int) -> str:
    """Return context extraction signals for one example.

    Args:
        example_id: PubMedQA example id.

    Returns:
        JSON string with key context signals.
    """
    eid = int(example_id)
    guard = _tool_guard(eid)
    if guard is not None:
        CONTEXT_RAW_CACHE[eid] = guard
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "context_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": False,
            "guard_error": True,
            "raw_output": guard,
        }])
        return guard
    if eid in CONTEXT_CACHE:
        raw_cached = CONTEXT_RAW_CACHE.get(eid, CONTEXT_CACHE[eid])
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "context_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": True,
            "guard_error": False,
            "raw_output": raw_cached,
        }])
        return CONTEXT_CACHE[eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = dumps_json({"error": "example_id not found"})
        CONTEXT_CACHE[eid] = out
        CONTEXT_RAW_CACHE[eid] = out
        _append_raw_trace_rows([{
            "ts": int(time.time()),
            "agent": "context_tool",
            "event": "tool_call",
            "example_id": eid,
            "cache_hit": False,
            "guard_error": True,
            "raw_output": out,
        }])
        return out

    q = ex["question"]
    ctx = ex["context"]
    choices = ex.get("choices", {})
    numbered_context = build_numbered_sentences(ctx)
    sentence_text_by_sid = {int(it["sid"]): str(it["text"]) for it in numbered_context}
    ctx_block = format_numbered_sentences(numbered_context, "NUMBERED CONTEXT SENTENCES")
    choices_block = format_choices_block(choices)

    user = f"Example ID: {eid}\nQUESTION:\n{q}\n\n{choices_block}{ctx_block}"
    msgs = [{"role": "system", "content": get_context_system_prompt()}, {"role": "user", "content": user}]
    raw = _context_agent.generate(msgs, temperature=0.0) if _context_agent else ""
    CONTEXT_RAW_CACHE[eid] = raw
    _append_raw_trace_rows([{
        "ts": int(time.time()),
        "agent": "context_tool",
        "event": "tool_call",
        "example_id": eid,
        "cache_hit": False,
        "guard_error": False,
        "raw_output": raw,
    }])
    obj = extract_first_json(raw)

    if obj is None:
        rng = random.Random(67890 + eid)
        candidates = build_candidates(q, ctx, top_k=TOOL_TOP_K, rng=rng, choices=choices, task_name=TASK_NAME)
        key_sentences = [{"sid": int(c["sid"]), "text": str(c["text"])[:240]} for c in candidates[:6]]
        obj = {
            "key_sentences": key_sentences,
            "context_summary": "Decision-relevant signals extracted from context.",
            "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:3],
            "confidence": 0.6,
        }

    obj = _normalize_context_output(obj, sentence_text_by_sid=sentence_text_by_sid)
    out = dumps_json(obj)
    CONTEXT_CACHE[eid] = out
    return out


# =========================================================
# Manager prompt
# =========================================================
def _tool_call_args_for_current_mode(eid: int) -> Dict[str, Any]:
    if _manager_tools_require_example_id():
        return {"example_id": int(eid)}
    return {}


def _tool_call_message(tool_name: str, eid: int, call_id: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": dumps_json(_tool_call_args_for_current_mode(eid)),
                },
            }
        ],
    }


class ManagerToolEnvironment:
    def reset(self, example_id: int, **kwargs) -> Optional[str]:
        self.example_id = int(example_id)
        return None

    def reasoning_tool(self) -> str:
        return reasoning_tool(getattr(self, "example_id", -1))

    def context_tool(self) -> str:
        return context_tool(getattr(self, "example_id", -1))


def build_manager_system_prompt() -> str:
    if TASK_NAME == "medqa":
        task_line = "You are a manager agent solving medical multiple-choice questions."
    elif TASK_NAME == "pubmedqa":
        task_line = "You are a manager agent solving PubMedQA-style clinical questions."
    else:
        task_line = "You are a manager agent solving clinical QA tasks."

    answer_lines = "\n".join([f"  ANSWER_{ANSWER_CANONICAL_TO_TOKEN[lab]}" for lab in ANSWER_LABELS])
    if _manager_tools_require_example_id():
        tool_binding_line = "The available native tools expect the current Example ID from the user message."
        tool_arg_rule = "If you call a tool, pass the current Example ID exactly as shown in the user message."
    else:
        tool_binding_line = "The available native tools are already bound to the current example."
        tool_arg_rule = "If you call a tool, do not invent arguments."
    return (
        task_line + "\n"
        "Calling tools is OPTIONAL.\n"
        "You have up to TWO tool calls total.\n"
        "- Final answer must end with exactly one line:\n"
        f"{answer_lines}\n"
        "Do not write anything after that last line.\n"
    )


MANAGER_SYSTEM = build_manager_system_prompt()


def _format_choices_block(choices: Optional[Dict[str, str]]) -> str:
    return format_choices_block(choices)


def build_manager_messages(eid: int, q: str, ctx: str, choices: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    choices_block = _format_choices_block(choices)
    return [
        {"role": "system", "content": MANAGER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"example_id: {eid}\n\n"
                f"Question:\n{q}\n\n"
                f"{choices_block}"
                f"Context:\n{ctx}\n\n"
                "You may call reasoning_tool and/or context_tool.\n"
                "If you call a tool, copy the exact example_id provided above into the arguments.\n"
                "If you do NOT call tools, answer directly.\n\n"
                "When calling a tool, output exactly in one or more blocks like these:\n"
                "<tool_call>\n"
                f'{{"name": "reasoning_tool", "arguments": {{"example_id": {eid}}}}}\n'
                "</tool_call>\n\n"
                "<tool_call>\n"
                f'{{"name": "context_tool", "arguments": {{"example_id": {eid}}}}}\n'
                "</tool_call>\n"
            ),
        },
    ]

# =========================================================
# Binary outcome reward + failure logging (self-evolving)
# =========================================================
FAIL_BUFFER_JSONL: Optional[str] = None
RAW_TRACE_JSONL: Optional[str] = None


def _append_raw_trace_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if (not _IS_MAIN_PROCESS) or (not RAW_TRACE_JSONL):
        return
    append_jsonl(RAW_TRACE_JSONL, rows)


def binary_outcome_reward(prompts=None, completions=None, ground_truth=None, example_id=None, **kwargs):
    n = len(completions)
    gts = ensure_list(ground_truth, n)
    exids = ensure_list(example_id, n)

    rewards = []
    fail_rows = []
    manager_rows = []
    for c, gt, eid in zip(completions, gts, exids):
        gt = _normalize_label(gt)
        st = extract_stats(c)
        pred = parse_answer_label_lastline(st["last_assistant_text"])
        valid_format = (pred is not None)

        final_has_artifacts = bool(
            st["last_assistant_has_tool_calls"] or st.get("last_assistant_plaintext_tool_artifacts")
        )
        fake_tool_text = bool(st.get("fake_tool_text_attempt"))

        ok = bool(valid_format and (not final_has_artifacts) and (not fake_tool_text) and (pred == gt))
        r = 1.0 if ok else 0.0
        rewards.append(float(r))
        manager_rows.append({
            "ts": int(time.time()),
            "agent": "manager",
            "event": "completion",
            "example_id": int(eid) if eid is not None else None,
            "ground_truth": gt,
            "pred": pred,
            "reward": float(r),
            "valid_format": bool(valid_format),
            "final_has_tool_artifacts": bool(final_has_artifacts),
            "fake_tool_text_attempt": bool(fake_tool_text),
            "tool_names": st.get("tool_names", []),
            "tool_payloads": st.get("tool_payloads", []),
            "assistant_texts": st.get("assistant_texts", []),
            "last_assistant_text": st.get("last_assistant_text", ""),
            "completion_raw": c,
        })

        if (not ok) and _IS_MAIN_PROCESS and FAIL_BUFFER_JSONL:
            fail_rows.append({
                "ts": int(time.time()),
                "example_id": int(eid) if eid is not None else None,
                "ground_truth": gt,
                "pred": pred,
                "valid_format": bool(valid_format),
                "final_has_tool_artifacts": bool(final_has_artifacts),
                "fake_tool_text_attempt": bool(fake_tool_text),
                "tool_names": st.get("tool_names", []),
                "tool_payloads": st.get("tool_payloads", []),
                "assistant_texts": st.get("assistant_texts", []),
            })

    if fail_rows and _IS_MAIN_PROCESS and FAIL_BUFFER_JSONL:
        append_jsonl(FAIL_BUFFER_JSONL, fail_rows)
    _append_raw_trace_rows(manager_rows)

    return rewards


# =========================================================
# Manager SFT from failures (teacher chooses tool sequence)
# =========================================================
def _final_answer_str(gt: str) -> str:
    canonical = _normalize_label(gt)
    if canonical not in ANSWER_CANONICAL_TO_TOKEN:
        canonical = ANSWER_LABELS[0]
    return f"ANSWER_{ANSWER_CANONICAL_TO_TOKEN[canonical]}"


def teacher_choose_tool_sequence(
    teacher: Optional[OpenAICompatClient],
    q: str,
    ctx: str,
    choices: Optional[Dict[str, str]] = None,
    planning_mode: str = "realistic",  # realistic | oracle
    reasoning_json: str = "",
    context_json: str = "",
) -> List[str]:
    """
    realistic: teacher only sees QUESTION + CONTEXT (no tool outputs).
    oracle: teacher also sees tool outputs (useful for debugging, but optimistic).
    """
    if teacher is None:
        # Simple heuristic without any tool-output peeking:
        # - long context: use context_tool first
        # - otherwise: use reasoning_tool first
        if len(ctx) > 2000:
            return ["context_tool", "reasoning_tool"]
        return ["reasoning_tool"]

    sys = (
        "You are helping decide an efficient tool-use plan.\n"
        "Choose a tool sequence of length 0 to 2 from: [reasoning_tool, context_tool].\n"
        "Goal: maximize correctness but use as few tools as possible.\n"
        "Return ONLY JSON: {\"tool_sequence\": [\"context_tool\", \"reasoning_tool\"]}\n"
        "Do not include any other keys.\n"
    )

    choices_block = _format_choices_block(choices)
    if planning_mode == "oracle":
        user = (
            f"QUESTION:\n{q}\n\n"
            f"{choices_block}"
            f"CONTEXT (truncated):\n{ctx[:2500]}\n\n"
            f"reasoning_tool output:\n{reasoning_json[:2000]}\n\n"
            f"context_tool output:\n{context_json[:2000]}\n"
        )
    else:
        user = (
            f"QUESTION:\n{q}\n\n"
            f"{choices_block}"
            f"CONTEXT (truncated):\n{ctx[:3000]}\n"
        )

    raw = teacher.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.2)
    obj = extract_first_json(raw)
    if not obj or "tool_sequence" not in obj:
        return ["context_tool", "reasoning_tool"] if len(ctx) > 2000 else ["reasoning_tool"]

    seq = obj.get("tool_sequence", [])
    if not isinstance(seq, list):
        return ["context_tool", "reasoning_tool"] if len(ctx) > 2000 else ["reasoning_tool"]

    out = []
    for x in seq:
        x = str(x)
        if x in ["reasoning_tool", "context_tool"] and x not in out:
            out.append(x)
        if len(out) >= 2:
            break
    return out


def build_manager_sft_from_failures(
    tool_base_model: str,
    reasoning_adapter: str,
    context_adapter: str,
    data_path: str,
    split_path: str,
    fail_jsonl: str,
    out_dir: str,
    seed: int = 42,
    max_fail_samples: int = 2000,
    use_teacher: bool = True,
    planning_mode: str = "realistic",
    device: Optional[str] = None,
):
    """
    IMPORTANT: we initialize tool agents here, so evolve stage really uses trained tool adapters.
    Tools do not have access to ground_truth.
    """
    set_seed(seed)
    rows = load_raw_pubmedqa(data_path)
    splits = read_json(split_path)
    train_ids = set(splits["train_ids"])
    id2ex_full = {r["example_id"]: r for r in rows}

    # Read failures; keep only train ids
    fails = []
    with open(fail_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("example_id")
            if eid is None:
                continue
            try:
                eid = int(eid)
            except Exception:
                continue
            if eid not in train_ids:
                continue
            fails.append(eid)

    fails = sorted(list(set(fails)))
    rng = random.Random(seed)
    rng.shuffle(fails)
    fails = fails[:max_fail_samples]

    teacher = get_teacher_client_from_env() if use_teacher else None

    if device is None:
        device = get_runtime_device()

    # Init tool agents (trained adapters; base must match tool SFT / adapter)
    init_tool_agents(tool_base_model, reasoning_adapter, context_adapter, device=device)

    # Ensure globals for tools (NO ground_truth)
    ID2EX.clear()
    for r in rows:
        ID2EX[int(r["example_id"])] = {
            "question": r["question"],
            "context": r["context"],
            "choices": r.get("choices", {}),
        }

    global ALLOWED_TOOL_IDS
    ALLOWED_TOOL_IDS = set(train_ids)
    REASONING_CACHE.clear()
    CONTEXT_CACHE.clear()
    REASONING_RAW_CACHE.clear()
    CONTEXT_RAW_CACHE.clear()

    sft_rows = []

    for eid in fails:
        ex = id2ex_full.get(eid)
        if ex is None:
            continue

        q, ctx, gt = ex["question"], ex["context"], ex["ground_truth"]
        choices = ex.get("choices", {})

        # Compute tool outputs for step2/step3 prompts only (teacher does not need to see them in realistic mode)
        rj = reasoning_tool(eid)
        cj = context_tool(eid)

        seq = teacher_choose_tool_sequence(
            teacher=teacher,
            q=q,
            ctx=ctx,
            choices=choices,
            planning_mode=planning_mode,
            reasoning_json=rj,
            context_json=cj,
        )

        # step 1
        prompt1 = build_manager_messages(eid, q, ctx, choices=choices)
        if len(seq) == 0:
            resp1 = _final_answer_str(gt)
            sft_rows.append({"example_id": eid, "prompt": prompt1, "response": resp1})
            continue

        call1_id = f"call_{eid}_1"
        resp1 = _tool_call_message(seq[0], eid, call1_id)
        sft_rows.append({"example_id": eid, "prompt": prompt1, "response": resp1})

        # step 2 prompt: add tool output as tool message
        tool1_name = seq[0]
        tool1_out = rj if tool1_name == "reasoning_tool" else cj
        prompt2 = prompt1 + [
            resp1,
            {"role": "tool", "tool_call_id": call1_id, "name": tool1_name, "content": tool1_out},
        ]

        if len(seq) == 1:
            resp2 = _final_answer_str(gt)
            sft_rows.append({"example_id": eid, "prompt": prompt2, "response": resp2})
            continue

        call2_id = f"call_{eid}_2"
        resp2 = _tool_call_message(seq[1], eid, call2_id)
        sft_rows.append({"example_id": eid, "prompt": prompt2, "response": resp2})

        # step 3 prompt: add tool2 output
        tool2_name = seq[1]
        tool2_out = rj if tool2_name == "reasoning_tool" else cj
        prompt3 = prompt2 + [
            resp2,
            {"role": "tool", "tool_call_id": call2_id, "name": tool2_name, "content": tool2_out},
        ]
        resp3 = _final_answer_str(gt)
        sft_rows.append({"example_id": eid, "prompt": prompt3, "response": resp3})

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "manager_sft_from_failures.jsonl")
    write_jsonl(out_path, sft_rows)
    print(f"[EVOLVE SFT] failures={len(fails)} rows={len(sft_rows)} wrote={out_path}")
    return out_path


# =========================================================
# Manager SFT training (on per-turn data)
# =========================================================
def train_manager_sft(
    manager_base_model: str,
    train_jsonl: str,
    out_dir: str,
    seed: int = 42,
    max_seq_len: int = 4096,
    lr: float = 2e-5,
    epochs: int = 1,
    per_device_bs: int = 1,
    grad_accum: int = 8,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    require_datasets("train_manager_sft")
    set_seed(seed)
    device = get_runtime_device()
    validate_distributed_runtime("train_manager_sft", require_cuda=(get_world_size() > 1))

    tok = AutoTokenizer.from_pretrained(manager_base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    try:
        tok = add_response_schema(tok)
    except Exception:
        pass

    dtype = get_runtime_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        manager_base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_lora:
        if not PEFT_AVAILABLE:
            print("[WARN] peft not available -> full finetune.")
        else:
            common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            present = set([name.split(".")[-1] for name, _ in model.named_modules()])
            target_modules = [m for m in common if m in present]
            if not target_modules:
                target_modules = ["q_proj", "v_proj"]
            lconf = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lconf)
            print(f"[LoRA] target_modules={target_modules}")

    ds = load_dataset("json", data_files={"train": train_jsonl})
    train_ds = tokenize_sft_dataset(ds["train"], tok, max_seq_len=max_seq_len)

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[MANAGER SFT] saved to: {out_dir}")


# =========================================================
# Train manager with GRPO (binary reward) on train split
# =========================================================
def train_manager_grpo_from_splits(
    manager_base_model: str,
    tool_base_model: str,
    data_path: str,
    split_path: str,
    save_dir: str,
    reasoning_adapter: str,
    context_adapter: str,
    seed: int = 42,
    per_device_train_bs: int = 2,
    grad_accum: int = 1,
    max_prompt_length: int = 2048,
    max_completion_length: int = 2048,
    temperature: float = 0.9,
    num_generations: int = 6,
    grpo_beta: float = 0.0,
    fail_buffer_jsonl: Optional[str] = None,
    raw_trace_jsonl: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "agents_as_tools_grpo",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    wandb_mode: str = "online",  # online | offline
    use_vllm: bool = False,
    vllm_mode: str = "colocate",  # colocate | server
    vllm_server_base_url: str = "http://127.0.0.1:8000",
    vllm_gpu_memory_utilization: float = 0.35,
    vllm_enable_sleep_mode: bool = False,
    vllm_visible_devices: str = "",
    vllm_tensor_parallel_size: int = 1,
    manager_use_lora: bool = False,
    manager_lora_r: int = 8,
    manager_lora_alpha: int = 16,
    manager_lora_dropout: float = 0.05,
    manager_gradient_checkpointing: bool = False,
    tool_top_k: int = 20,
):
    require_datasets("train_manager_grpo")
    require_trl("train_manager_grpo")
    set_seed(seed)
    device = get_runtime_device()
    global TOOL_TOP_K
    TOOL_TOP_K = max(1, int(tool_top_k))

    validate_distributed_runtime(
        "train_manager_grpo",
        require_cuda=(get_world_size() > 1 or use_vllm),
        use_vllm=use_vllm,
    )
    validate_grpo_batch_geometry(
        per_device_train_bs=per_device_train_bs,
        grad_accum=grad_accum,
        num_generations=num_generations,
    )
    if _IS_MAIN_PROCESS:
        print(
            f"[RUNTIME] platform={_platform_label()} world_size={get_world_size()} "
            f"rank={get_global_rank()} local_rank={get_local_rank()} device={device}"
        )
        if use_vllm:
            print(
                f"[VLLM] enabled mode={vllm_mode} base_url={vllm_server_base_url} "
                f"gpu_mem_util={float(vllm_gpu_memory_utilization):.2f}"
            )
            if vllm_mode == "colocate" and (float(grpo_beta) > 0.0) and (not manager_use_lora):
                print(
                    "[WARN] colocate + full-finetune + grpo_beta>0 will load a reference model "
                    "and often OOMs on 8B-class models. Prefer --grpo_beta 0.0 and --mgr_use_lora."
                )
            if vllm_mode == "server":
                print(
                    "[VLLM] launch hint: "
                    + _build_vllm_server_launch_hint(
                        model_name=manager_base_model,
                        server_base_url=vllm_server_base_url,
                        visible_devices=vllm_visible_devices,
                        tensor_parallel_size=vllm_tensor_parallel_size,
                        gpu_memory_utilization=vllm_gpu_memory_utilization,
                    )
                )

    rows = load_raw_pubmedqa(data_path)
    splits = read_json(split_path)
    train_ids = set(splits["train_ids"])

    if use_wandb:
        try:
            import wandb  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested but `wandb` is not available. "
                "Please install it with `pip install wandb`."
            ) from e

        if wandb_project.strip():
            os.environ["WANDB_PROJECT"] = wandb_project.strip()
        if wandb_entity.strip():
            os.environ["WANDB_ENTITY"] = wandb_entity.strip()
        os.environ["WANDB_MODE"] = (wandb_mode or "online").strip().lower()
        save_dir_tag = os.path.basename(save_dir.rstrip("/\\"))
        run_name = wandb_run_name.strip() or f"grpo_{save_dir_tag}_{int(time.time())}"
        os.environ["WANDB_NAME"] = run_name
        print(
            f"[WANDB] enabled project={os.environ.get('WANDB_PROJECT','')} "
            f"entity={os.environ.get('WANDB_ENTITY','')} mode={os.environ.get('WANDB_MODE','online')} "
            f"run_name={run_name}"
        )
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    # globals for tools (NO ground_truth)
    ID2EX.clear()
    for r in rows:
        ID2EX[int(r["example_id"])] = {
            "question": r["question"],
            "context": r["context"],
            "choices": r.get("choices", {}),
        }

    global ALLOWED_TOOL_IDS
    ALLOWED_TOOL_IDS = set(train_ids)
    REASONING_CACHE.clear()
    CONTEXT_CACHE.clear()
    REASONING_RAW_CACHE.clear()
    CONTEXT_RAW_CACHE.clear()

    init_tool_agents(tool_base_model, reasoning_adapter, context_adapter, device=device)

    # setup fail buffer
    global FAIL_BUFFER_JSONL
    FAIL_BUFFER_JSONL = fail_buffer_jsonl or os.path.join(save_dir, "fail_buffer.jsonl")
    if _IS_MAIN_PROCESS and FAIL_BUFFER_JSONL:
        os.makedirs(os.path.dirname(FAIL_BUFFER_JSONL) or ".", exist_ok=True)
        if os.environ.get("FAIL_BUFFER_APPEND", "0") == "0":
            with open(FAIL_BUFFER_JSONL, "w", encoding="utf-8"):
                pass
        print(f"[FAIL_BUFFER] writing failures -> {FAIL_BUFFER_JSONL}")

    global RAW_TRACE_JSONL
    RAW_TRACE_JSONL = raw_trace_jsonl             or os.path.join(save_dir, "train_raw_trace.jsonl")
    if _IS_MAIN_PROCESS and RAW_TRACE_JSONL:
        os.makedirs(os.path.dirname(RAW_TRACE_JSONL) or ".", exist_ok=True)
        if os.environ.get("RAW_TRACE_APPEND", "0") == "0":
            with open(RAW_TRACE_JSONL, "w", encoding="utf-8"):
                pass
        print(f"[RAW_TRACE] writing raw outputs -> {RAW_TRACE_JSONL}")

    train_rows = [r for r in rows if r["example_id"] in train_ids]
    dataset = Dataset.from_list(train_rows)

    manager_tok = AutoTokenizer.from_pretrained(manager_base_model, trust_remote_code=True)
    manager_tok.padding_side = "left"
    if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
        manager_tok.pad_token_id = manager_tok.eos_token_id
    try:
        manager_tok = add_response_schema(manager_tok)
    except Exception:
        pass

    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        eid = int(ex["example_id"])
        msgs = build_manager_messages(eid, ex["question"], ex["context"], choices=ex.get("choices", {}))
        return {"prompt": msgs, "ground_truth": ex["ground_truth"], "example_id": eid}

    train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    grpo_kwargs = {
        "output_dir": save_dir,
        "remove_unused_columns": False,
        "gradient_accumulation_steps": int(grad_accum),
        "max_prompt_length": int(max_prompt_length),
        "max_completion_length": int(max_completion_length),
        "temperature": float(temperature),
        "num_generations": int(num_generations),
        "bf16": torch.cuda.is_available(),
        "beta": float(grpo_beta),
        "scale_rewards": "group",
        "report_to": (["wandb"] if use_wandb else []),
        "use_vllm": bool(use_vllm),
        "per_device_train_batch_size": int(per_device_train_bs),
        "max_tool_calling_iterations": 2,
        "chat_template_kwargs": {"enable_thinking": False},
        "gradient_checkpointing": bool(manager_gradient_checkpointing),
        "logging_steps": 1,
        "log_completions": True,
        "num_completions_to_print": None,
        "log_unique_prompts": False,
    }
    if use_vllm:
        grpo_kwargs.update(
            {
                "vllm_mode": vllm_mode,
                "vllm_server_base_url": vllm_server_base_url,
                "vllm_gpu_memory_utilization": float(vllm_gpu_memory_utilization),
                "vllm_enable_sleep_mode": bool(vllm_enable_sleep_mode),
            }
        )

    grpo_args = GRPOConfig(
        **_filter_supported_kwargs(GRPOConfig.__init__, grpo_kwargs, "GRPOConfig")
    )

    dtype = get_runtime_dtype()
    manager_model = AutoModelForCausalLM.from_pretrained(
        manager_base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    manager_model.config.use_cache = False
    if not hasattr(manager_model, "warnings_issued") or manager_model.warnings_issued is None:
        manager_model.warnings_issued = {}

    if manager_use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("`--mgr_use_lora` requires `peft` to be installed.")
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set([name.split(".")[-1] for name, _ in manager_model.named_modules()])
        target_modules = [m for m in common if m in present]
        if not target_modules:
            target_modules = ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=manager_lora_r,
            lora_alpha=manager_lora_alpha,
            lora_dropout=manager_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        manager_model = get_peft_model(manager_model, lconf)
        print(f"[MANAGER LoRA] target_modules={target_modules}")

    if manager_gradient_checkpointing:
        try:
            manager_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            manager_model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(manager_model, "enable_input_require_grads"):
            try:
                manager_model.enable_input_require_grads()
            except Exception:
                pass

    trainer_kwargs = {
        "model": manager_model,
        "args": grpo_args,
        "train_dataset": train_dataset,
        "reward_funcs": [binary_outcome_reward],
        "rollout_func": None,
    }
    trainer_kwargs.update(_trainer_processing_kwargs(manager_tok))
    if MANAGER_TOOL_BINDING_MODE == "environment":
        trainer_kwargs["environment_factory"] = ManagerToolEnvironment
    else:
        trainer_kwargs["tools"] = [reasoning_tool, context_tool]

    trainer = GRPOTrainer(
        **_filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs, "GRPOTrainer")
    )

    trainer.train()
    trainer.model.save_pretrained(save_dir)
    manager_tok.save_pretrained(save_dir)
    print(f"[GRPO] Saved manager to: {save_dir}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "make_splits",
            "build_tool_sft",
            "train_tool_reasoning",
            "train_tool_context",
            "train_manager_grpo",
            "evolve_build_manager_sft",
            "train_manager_sft",
            "evolve_round",
        ],
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help=(
            "If set: use the same HF id/path for both manager and tool bases "
            "(overrides --manager_base_model and --tool_base_model). "
            "Otherwise use the two flags below."
        ),
    )
    parser.add_argument(
        "--manager_base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HF id/path for the manager (GRPO + manager SFT). Chat template must match training.",
    )
    parser.add_argument(
        "--tool_base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help=(
            "HF id/path for reasoning_tool and context_tool (SFT adapters and GRPO tool rollouts). "
            "Must match the base used to train --reasoning_tool_out / --context_tool_out."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help=(
            "Dataset file or directory. "
            "Supports aliases: pubmedqa, medqa. "
            "If empty, use task-specific default path."
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="pubmedqa",
        choices=["pubmedqa", "medqa", "generic"],
        help="Task preset. Controls default label space and manager prompt wording.",
    )
    parser.add_argument(
        "--tool_binding_mode",
        type=str,
        default="auto",
        choices=["auto", "environment", "argument"],
        help="How manager tools bind to the current example. `environment` avoids example_id hallucination but requires transformers>=5.2.0.",
    )
    parser.add_argument(
        "--label_space",
        type=str,
        default="",
        help="Comma-separated canonical labels. Example: yes,no,maybe or A,B,C,D,E.",
    )

    # split
    parser.add_argument("--split_path", type=str, default="splits_pubmedqa_1000.json")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--dev_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, subsample this many rows before split generation (stratified by label).",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=-1,
        help="Random seed for subsampling. If <0, use --seed.",
    )

    # tool SFT data
    parser.add_argument("--tool_sft_out_dir", type=str, default="tool_sft_data_evolving")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tool_variants_train", type=int, default=3)
    parser.add_argument("--tool_variants_dev", type=int, default=2)
    parser.add_argument("--tool_synth_mode", type=str, default="weak", choices=["weak", "gpt"])

    # tool SFT training (load weights from --tool_base_model; saves adapters to --reasoning_tool_out / --context_tool_out)
    parser.add_argument("--tool_max_seq_len", type=int, default=4096)
    parser.add_argument("--tool_lr", type=float, default=2e-4)
    parser.add_argument("--tool_epochs", type=int, default=2)
    parser.add_argument("--tool_bs", type=int, default=1)
    parser.add_argument("--tool_grad_accum", type=int, default=8)
    parser.add_argument("--tool_use_lora", action="store_true")
    parser.add_argument("--reasoning_tool_out", type=str, default="reasoning_tool_adapter")
    parser.add_argument("--context_tool_out", type=str, default="context_tool_adapter")

    # manager GRPO
    parser.add_argument("--manager_out", type=str, default="manager_grpo_binary")
    parser.add_argument("--mgr_bs", type=int, default=4)
    parser.add_argument("--mgr_grad_accum", type=int, default=1)
    parser.add_argument("--mgr_max_prompt_length", type=int, default=2048)
    parser.add_argument("--mgr_max_completion_length", type=int, default=4096)
    parser.add_argument("--mgr_temperature", type=float, default=0.9)
    parser.add_argument("--mgr_num_generations", type=int, default=4)
    parser.add_argument(
        "--grpo_beta",
        type=float,
        default=0.0,
        help="KL coefficient. 0.0 avoids loading a GRPO reference model and saves substantial VRAM.",
    )
    parser.add_argument("--fail_buffer_jsonl", type=str, default="")
    parser.add_argument("--raw_trace_jsonl", type=str, default="")
    parser.add_argument("--grpo_use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="agents_as_tools_grpo")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--mgr_use_vllm", action="store_true")
    parser.add_argument("--mgr_vllm_mode", type=str, default="colocate", choices=["server", "colocate"])
    parser.add_argument("--mgr_vllm_server_base_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--mgr_vllm_gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--mgr_vllm_enable_sleep_mode", action="store_true")
    parser.add_argument(
        "--mgr_vllm_visible_devices",
        type=str,
        default="",
        help="Optional CUDA_VISIBLE_DEVICES string used only when printing the vLLM server launch hint.",
    )
    parser.add_argument(
        "--mgr_vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size used only when printing the vLLM server launch hint.",
    )
    parser.add_argument("--mgr_use_lora", action="store_true")
    parser.add_argument("--mgr_lora_r", type=int, default=8)
    parser.add_argument("--mgr_lora_alpha", type=int, default=16)
    parser.add_argument("--mgr_lora_dropout", type=float, default=0.05)
    parser.add_argument("--mgr_gradient_checkpointing", action="store_true")

    # evolve manager SFT
    parser.add_argument("--evolve_out_dir", type=str, default="evolve_manager_sft")
    parser.add_argument("--max_fail_samples", type=int, default=2000)
    parser.add_argument("--use_teacher", action="store_true")
    parser.add_argument("--planning_mode", type=str, default="realistic", choices=["realistic", "oracle"])

    # manager SFT train
    parser.add_argument("--manager_sft_out", type=str, default="manager_sft_evolved")
    parser.add_argument("--manager_sft_lr", type=float, default=2e-5)
    parser.add_argument("--manager_sft_epochs", type=int, default=1)
    parser.add_argument("--manager_sft_max_seq_len", type=int, default=4096)
    parser.add_argument("--manager_sft_bs", type=int, default=1)
    parser.add_argument("--manager_sft_grad_accum", type=int, default=8)
    parser.add_argument("--manager_sft_use_lora", action="store_true")

    args = parser.parse_args()
    configure_cuda_runtime()
    if args.base_model.strip():
        shared = args.base_model.strip()
        args.manager_base_model = shared
        args.tool_base_model = shared

    configured_task, configured_labels = configure_task(args.task_name, args.label_space)
    print(f"[TASK] task_name={configured_task} labels={configured_labels}")
    data_path = resolve_data_path_arg(args.data_path, configured_task)
    print(f"[DATA] data_path={data_path}")
    print(f"[MODELS] manager_base_model={args.manager_base_model} tool_base_model={args.tool_base_model}")
    binding_mode = (args.tool_binding_mode or "auto").strip().lower()
    binding_decision = "explicit"
    if binding_mode == "auto":
        if _grpo_supports_environment_factory() and _transformers_version_at_least("5.2.0"):
            binding_mode = "environment"
            binding_decision = "auto(environment_factory supported and transformers>=5.2.0)"
        else:
            binding_mode = "argument"
            if _grpo_supports_environment_factory():
                binding_decision = "auto(transformers<5.2.0 -> argument)"
            else:
                binding_decision = "auto(environment_factory unsupported -> argument)"
    if binding_mode == "environment" and not _grpo_supports_environment_factory():
        raise RuntimeError(
            "tool_binding_mode=environment requires a TRL version that supports `environment_factory`."
        )
    if binding_mode == "environment" and not _transformers_version_at_least("5.2.0"):
        raise RuntimeError(
            "tool_binding_mode=environment requires transformers>=5.2.0 according to the TRL GRPO tool-calling docs. "
            f"Current transformers version: {_get_transformers_version()}."
        )
    global MANAGER_TOOL_BINDING_MODE, MANAGER_SYSTEM
    MANAGER_TOOL_BINDING_MODE = binding_mode
    MANAGER_SYSTEM = build_manager_system_prompt()
    print(
        f"[MANAGER_TOOLS] binding_mode={MANAGER_TOOL_BINDING_MODE} decision={binding_decision} "
        f"transformers={_get_transformers_version()} "
        f"world_size={get_world_size()} local_rank={get_local_rank()} device={get_runtime_device()}"
    )
    set_seed(args.seed)

    if args.stage == "make_splits":
        rows = load_raw_pubmedqa(data_path)
        print(f"[SPLIT] loaded rows = {len(rows)}")
        if args.max_samples > 0:
            sample_seed = args.seed if args.sample_seed < 0 else args.sample_seed
            rows = subsample_rows(rows, max_samples=args.max_samples, seed=sample_seed)
            print(f"[SPLIT] subsampled rows = {len(rows)} (max_samples={args.max_samples}, sample_seed={sample_seed})")
        splits = make_splits(rows, test_size=args.test_size, dev_size=args.dev_size, seed=args.seed)
        write_json(args.split_path, splits)
        print(f"[SPLIT] wrote -> {args.split_path}")
        print(f"[SPLIT] train/dev/test sizes = {len(splits['train_ids'])}/{len(splits['dev_ids'])}/{len(splits['test_ids'])}")
        return

    if args.stage == "build_tool_sft":
        build_tool_sft_data_from_splits(
            data_path=data_path,
            split_path=args.split_path,
            out_dir=args.tool_sft_out_dir,
            seed=args.seed,
            top_k=args.top_k,
            variants_train=args.tool_variants_train,
            variants_dev=args.tool_variants_dev,
            ev_min=3,
            ev_max=6,
            synth_mode=args.tool_synth_mode,
        )
        return

    if args.stage == "train_tool_reasoning":
        train_jsonl = os.path.join(args.tool_sft_out_dir, "tool_reasoning_train.jsonl")
        dev_jsonl = os.path.join(args.tool_sft_out_dir, "tool_reasoning_dev.jsonl")
        train_sft_agent(
            tool_base_model=args.tool_base_model,
            train_jsonl=train_jsonl,
            dev_jsonl=dev_jsonl,
            out_dir=args.reasoning_tool_out,
            seed=args.seed,
            max_seq_len=args.tool_max_seq_len,
            lr=args.tool_lr,
            epochs=args.tool_epochs,
            per_device_bs=args.tool_bs,
            grad_accum=args.tool_grad_accum,
            use_lora=args.tool_use_lora,
        )
        return

    if args.stage == "train_tool_context":
        train_jsonl = os.path.join(args.tool_sft_out_dir, "tool_context_train.jsonl")
        dev_jsonl = os.path.join(args.tool_sft_out_dir, "tool_context_dev.jsonl")
        train_sft_agent(
            tool_base_model=args.tool_base_model,
            train_jsonl=train_jsonl,
            dev_jsonl=dev_jsonl,
            out_dir=args.context_tool_out,
            seed=args.seed,
            max_seq_len=args.tool_max_seq_len,
            lr=args.tool_lr,
            epochs=args.tool_epochs,
            per_device_bs=args.tool_bs,
            grad_accum=args.tool_grad_accum,
            use_lora=args.tool_use_lora,
        )
        return

    if args.stage == "train_manager_grpo":
        fb = args.fail_buffer_jsonl.strip() or None
        rt = args.raw_trace_jsonl.strip() or None
        train_manager_grpo_from_splits(
            manager_base_model=args.manager_base_model,
            tool_base_model=args.tool_base_model,
            data_path=data_path,
            split_path=args.split_path,
            save_dir=args.manager_out,
            reasoning_adapter=args.reasoning_tool_out,
            context_adapter=args.context_tool_out,
            seed=args.seed,
            per_device_train_bs=args.mgr_bs,
            grad_accum=args.mgr_grad_accum,
            max_prompt_length=args.mgr_max_prompt_length,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.grpo_beta,
            fail_buffer_jsonl=fb,
            raw_trace_jsonl=rt,
            use_wandb=args.grpo_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            use_vllm=args.mgr_use_vllm,
            vllm_mode=args.mgr_vllm_mode,
            vllm_server_base_url=args.mgr_vllm_server_base_url,
            vllm_gpu_memory_utilization=args.mgr_vllm_gpu_memory_utilization,
            vllm_enable_sleep_mode=args.mgr_vllm_enable_sleep_mode,
            vllm_visible_devices=args.mgr_vllm_visible_devices,
            vllm_tensor_parallel_size=args.mgr_vllm_tensor_parallel_size,
            manager_use_lora=args.mgr_use_lora,
            manager_lora_r=args.mgr_lora_r,
            manager_lora_alpha=args.mgr_lora_alpha,
            manager_lora_dropout=args.mgr_lora_dropout,
            manager_gradient_checkpointing=args.mgr_gradient_checkpointing,
            tool_top_k=args.top_k,
        )
        return

    if args.stage == "evolve_build_manager_sft":
        fail_path = args.fail_buffer_jsonl.strip()
        if not fail_path:
            fail_path = os.path.join(args.manager_out, "fail_buffer.jsonl")

        out_path = build_manager_sft_from_failures(
            tool_base_model=args.tool_base_model,
            reasoning_adapter=args.reasoning_tool_out,
            context_adapter=args.context_tool_out,
            data_path=data_path,
            split_path=args.split_path,
            fail_jsonl=fail_path,
            out_dir=args.evolve_out_dir,
            seed=args.seed,
            max_fail_samples=args.max_fail_samples,
            use_teacher=args.use_teacher,
            planning_mode=args.planning_mode,
        )
        print(f"[EVOLVE] manager_sft_jsonl = {out_path}")
        return

    if args.stage == "train_manager_sft":
        sft_jsonl = os.path.join(args.evolve_out_dir, "manager_sft_from_failures.jsonl")
        if not os.path.exists(sft_jsonl):
            raise FileNotFoundError(f"Missing evolve SFT file: {sft_jsonl}")
        train_manager_sft(
            manager_base_model=args.manager_base_model,
            train_jsonl=sft_jsonl,
            out_dir=args.manager_sft_out,
            seed=args.seed,
            max_seq_len=args.manager_sft_max_seq_len,
            lr=args.manager_sft_lr,
            epochs=args.manager_sft_epochs,
            per_device_bs=args.manager_sft_bs,
            grad_accum=args.manager_sft_grad_accum,
            use_lora=args.manager_sft_use_lora,
        )
        return

    if args.stage == "evolve_round":
        fb = args.fail_buffer_jsonl.strip() or os.path.join(args.manager_out, "fail_buffer.jsonl")
        rt = args.raw_trace_jsonl.strip() or None

        train_manager_grpo_from_splits(
            manager_base_model=args.manager_base_model,
            tool_base_model=args.tool_base_model,
            data_path=data_path,
            split_path=args.split_path,
            save_dir=args.manager_out,
            reasoning_adapter=args.reasoning_tool_out,
            context_adapter=args.context_tool_out,
            seed=args.seed,
            per_device_train_bs=args.mgr_bs,
            grad_accum=args.mgr_grad_accum,
            max_prompt_length=args.mgr_max_prompt_length,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.grpo_beta,
            fail_buffer_jsonl=fb,
            raw_trace_jsonl=rt,
            use_wandb=args.grpo_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_mode=args.wandb_mode,
            use_vllm=args.mgr_use_vllm,
            vllm_mode=args.mgr_vllm_mode,
            vllm_server_base_url=args.mgr_vllm_server_base_url,
            vllm_gpu_memory_utilization=args.mgr_vllm_gpu_memory_utilization,
            vllm_enable_sleep_mode=args.mgr_vllm_enable_sleep_mode,
            vllm_visible_devices=args.mgr_vllm_visible_devices,
            vllm_tensor_parallel_size=args.mgr_vllm_tensor_parallel_size,
            manager_use_lora=args.mgr_use_lora,
            manager_lora_r=args.mgr_lora_r,
            manager_lora_alpha=args.mgr_lora_alpha,
            manager_lora_dropout=args.mgr_lora_dropout,
            manager_gradient_checkpointing=args.mgr_gradient_checkpointing,
            tool_top_k=args.top_k,
        )

        build_manager_sft_from_failures(
            tool_base_model=args.tool_base_model,
            reasoning_adapter=args.reasoning_tool_out,
            context_adapter=args.context_tool_out,
            data_path=data_path,
            split_path=args.split_path,
            fail_jsonl=fb,
            out_dir=args.evolve_out_dir,
            seed=args.seed,
            max_fail_samples=args.max_fail_samples,
            use_teacher=args.use_teacher,
            planning_mode=args.planning_mode,
        )

        sft_jsonl = os.path.join(args.evolve_out_dir, "manager_sft_from_failures.jsonl")
        train_manager_sft(
            manager_base_model=args.manager_base_model,
            train_jsonl=sft_jsonl,
            out_dir=args.manager_sft_out,
            seed=args.seed,
            max_seq_len=args.manager_sft_max_seq_len,
            lr=args.manager_sft_lr,
            epochs=args.manager_sft_epochs,
            per_device_bs=args.manager_sft_bs,
            grad_accum=args.manager_sft_grad_accum,
            use_lora=args.manager_sft_use_lora,
        )

        print("[EVOLVE_ROUND] done")
        return


if __name__ == "__main__":
    main()
