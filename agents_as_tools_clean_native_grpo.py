# agents_as_tools_clean_native_grpo.py
# -*- coding: utf-8 -*-

"""
Cleaned native-tool-call GRPO pipeline for PubMedQA / MedQA.

Design choices:
1) fixed native tool-calling path only
2) fixed argument-based tools only (example_id is always passed)
3) explicit generation config for manager sampling
4) reward shaping for final correctness + valid tool use
5) lightweight planner prompt with subgoal / memory / uncertainty guidance

Recommended environment:
  transformers >= 4.53
  trl >= 0.19
  datasets
  peft (optional)
"""

import argparse
import glob
import inspect
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

try:
    from datasets import Dataset, load_dataset
except Exception as e:
    raise RuntimeError(f"`datasets` is required. Import error: {type(e).__name__}: {e}")

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception as e:
    raise RuntimeError(f"`trl` is required. Import error: {type(e).__name__}: {e}")

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# =========================
# Runtime and version checks
# =========================
MIN_TRANSFORMERS = "4.53.0"
MIN_TRL = "0.19.0"  # adjust if your local known-good version differs


def _version_of(mod: Any) -> str:
    return str(getattr(mod, "__version__", "0"))


def require_clean_runtime() -> None:
    tf_ver = _version_of(transformers)
    if Version(tf_ver) < Version(MIN_TRANSFORMERS):
        raise RuntimeError(
            f"transformers>={MIN_TRANSFORMERS} is required for this cleaned script. Found {tf_ver}."
        )
    # trl.__version__ is not always exported the same way
    import trl
    trl_ver = _version_of(trl)
    if Version(trl_ver) < Version(MIN_TRL):
        raise RuntimeError(
            f"trl>={MIN_TRL} is required for this cleaned script. Found {trl_ver}."
        )
    print(f"[ENV] transformers={tf_ver} trl={trl_ver}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def runtime_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Task config
# =========================
TASK_NAME = "pubmedqa"
ANSWER_LABELS = ["yes", "no", "maybe"]
ANSWER_TOKEN_TO_CANONICAL = {"YES": "yes", "NO": "no", "MAYBE": "maybe"}
ANSWER_CANONICAL_TO_TOKEN = {"yes": "YES", "no": "NO", "maybe": "MAYBE"}
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
        raise ValueError(f"Invalid label: {label!r}")
    return s.upper()


def _default_labels_for_task(task_name: str) -> List[str]:
    t = (task_name or "").strip().lower()
    if t == "medqa":
        return ["A", "B", "C", "D", "E"]
    return ["yes", "no", "maybe"]


def _build_answer_regex(tokens: List[str]) -> re.Pattern:
    alts = "|".join(re.escape(x) for x in sorted(tokens, key=len, reverse=True))
    return re.compile(
        rf"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_({alts})\b[^\w]*$",
        re.IGNORECASE,
    )


def _parse_label_space_arg(label_space: str) -> List[str]:
    return [x.strip() for x in label_space.split(",") if x.strip()] if label_space else []


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
    global TASK_NAME, ANSWER_LABELS, ANSWER_TOKEN_TO_CANONICAL, ANSWER_CANONICAL_TO_TOKEN, ANSWER_LASTLINE_RE
    t = (task_name or "pubmedqa").strip().lower()
    labels = _parse_label_space_arg(label_space) or _default_labels_for_task(t)

    token_to_canonical: Dict[str, str] = {}
    canonical_to_token: Dict[str, str] = {}
    canonical_labels: List[str] = []

    for lb in labels:
        canonical = str(lb).strip()
        token = _label_to_token(canonical)
        token_to_canonical[token] = canonical
        canonical_to_token[canonical] = token
        if canonical not in canonical_labels:
            canonical_labels.append(canonical)

    TASK_NAME = t
    ANSWER_LABELS = canonical_labels
    ANSWER_TOKEN_TO_CANONICAL = token_to_canonical
    ANSWER_CANONICAL_TO_TOKEN = canonical_to_token
    ANSWER_LASTLINE_RE = _build_answer_regex(list(token_to_canonical.keys()))
    return TASK_NAME, ANSWER_LABELS


# =========================
# Data loading
# =========================
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
                    raise ValueError(f"JSONL parse error at {p}:{i}: {e}") from e
        return rows
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_data_path_for_task(task_name: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    t = (task_name or "").strip().lower()
    if t == "medqa":
        return os.path.join(base_dir, "MedQA", "data_clean", "questions")
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
        for cand in preferred:
            if os.path.isfile(cand):
                return [cand]
        fallback = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        if fallback:
            return [fallback[0]]
        raise FileNotFoundError(f"No PubMedQA json file found under {path}")

    if t == "medqa":
        primary = sorted(glob.glob(os.path.join(p, "**", "questions", "**", "*.jsonl"), recursive=True))
        primary = [x for x in primary if os.path.isfile(x)]
        if primary:
            return primary

        files: List[str] = []
        for pattern in [os.path.join(p, "**", "*.jsonl"), os.path.join(p, "**", "*.json")]:
            files.extend(glob.glob(pattern, recursive=True))
        files = sorted([x for x in files if os.path.isfile(x)])
        if files:
            return files
        raise FileNotFoundError(f"No MedQA data file found under {path}")

    generic = sorted([x for x in glob.glob(os.path.join(p, "**", "*.json*"), recursive=True) if os.path.isfile(x)])
    if generic:
        return generic
    raise FileNotFoundError(f"No data file found under {path}")


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
    eid = next_auto
    seen.add(eid)
    return eid, next_auto + 1


def _sorted_choice_items(choices: Dict[str, Any]) -> List[Tuple[str, str]]:
    items = []
    if not isinstance(choices, dict):
        return items
    for k, v in choices.items():
        kk, vv = str(k).strip(), str(v).strip()
        if kk:
            items.append((kk, vv))
    items.sort(key=lambda x: x[0])
    return items


def _build_default_context(ex: Dict[str, Any], choices: Dict[str, str]) -> str:
    parts: List[str] = []
    if str(ex.get("context", "")).strip():
        parts.append(str(ex.get("context", "")).strip())
    if choices:
        parts.append("Options:\n" + "\n".join(f"{k}. {v}" for k, v in _sorted_choice_items(choices)))
    meta = str(ex.get("meta_info", "")).strip()
    if meta:
        parts.append(f"Meta: {meta}")
    phrases = ex.get("metamap_phrases", [])
    if isinstance(phrases, list) and phrases:
        pv = ", ".join([str(x) for x in phrases if str(x).strip()])[:1000]
        if pv:
            parts.append(f"MetaMap phrases: {pv}")
    return "\n\n".join(p for p in parts if p.strip())


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
            raise ValueError(f"Unsupported top-level type in {fp}: {type(raw)}")

        for k, ex in iterable:
            if not isinstance(ex, dict):
                raise ValueError(f"Unsupported example type in {fp}: {type(ex)}")
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

            rows.append(
                {
                    "example_id": eid,
                    "question": question,
                    "context": context,
                    "ground_truth": gt,
                    "choices": norm_choices,
                    "task_name": TASK_NAME,
                    "source_file": fp,
                }
            )

    allowed = set(ANSWER_LABELS)
    cleaned: List[Dict[str, Any]] = []
    dropped_missing_q = 0
    dropped_bad_label = 0
    for r in rows:
        if not r["question"]:
            dropped_missing_q += 1
            continue
        if r["ground_truth"] not in allowed:
            dropped_bad_label += 1
            continue
        cleaned.append(r)

    if not cleaned:
        raise ValueError(
            f"No valid rows loaded from {path}. dropped_missing_question={dropped_missing_q} dropped_bad_label={dropped_bad_label}"
        )
    if dropped_missing_q or dropped_bad_label:
        print(
            f"[DATA] dropped invalid rows: missing_question={dropped_missing_q} bad_label={dropped_bad_label} kept={len(cleaned)}"
        )
    return cleaned


def load_raw_pubmedqa(path: str) -> List[Dict[str, Any]]:
    return load_raw_dataset(path=path, task_name=TASK_NAME)


# =========================
# Split helpers
# =========================
def _alloc_counts_stratified(label_counts: Dict[str, int], target: int) -> Dict[str, int]:
    labels = sorted(label_counts.keys())
    total = sum(label_counts.values())
    if total == 0:
        return {lab: 0 for lab in labels}

    floors = {}
    frac = {}
    for lab in labels:
        x = target * (label_counts[lab] / total)
        floors[lab] = int(np.floor(x))
        frac[lab] = x - floors[lab]

    remainder = target - sum(floors.values())
    order = sorted(labels, key=lambda lab: frac[lab], reverse=True)
    i = 0
    while remainder > 0:
        lab = order[i % len(order)]
        floors[lab] += 1
        remainder -= 1
        i += 1
    return floors


def make_splits(rows: List[Dict[str, Any]], test_size: int = 200, dev_size: int = 160, seed: int = 42) -> Dict[str, List[int]]:
    if test_size + dev_size >= len(rows):
        raise ValueError("test_size + dev_size must be < total size")

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
        test_ids.update(ids[:n])
        remaining_by_label[lab] = ids[n:]

    if len(test_ids) < test_size:
        need = test_size - len(test_ids)
        pool = []
        for lab in remaining_by_label:
            pool.extend(remaining_by_label[lab])
        rng.shuffle(pool)
        extra = pool[:need]
        test_ids.update(extra)
        extra_set = set(extra)
        for lab in remaining_by_label:
            remaining_by_label[lab] = [x for x in remaining_by_label[lab] if x not in extra_set]

    rem_counts = {lab: len(ids) for lab, ids in remaining_by_label.items()}
    dev_counts = _alloc_counts_stratified(rem_counts, dev_size)

    dev_ids = set()
    train_ids = set()
    for lab, ids in remaining_by_label.items():
        n = min(dev_counts.get(lab, 0), len(ids))
        dev_ids.update(ids[:n])
        train_ids.update(ids[n:])

    if len(dev_ids) < dev_size:
        need = dev_size - len(dev_ids)
        pool = list(train_ids)
        rng.shuffle(pool)
        extra = pool[:need]
        dev_ids.update(extra)
        train_ids.difference_update(set(extra))

    return {
        "train_ids": sorted(train_ids),
        "dev_ids": sorted(dev_ids),
        "test_ids": sorted(test_ids),
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
    wanted = _alloc_counts_stratified({lb: len(v) for lb, v in by_label.items()}, max_samples)
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


# =========================
# Candidate sentences
# =========================
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


def overlap_score(q_words: List[str], s_words: List[str]) -> float:
    if not q_words or not s_words:
        return 0.0
    qs, ss = set(q_words), set(s_words)
    inter = len(qs.intersection(ss))
    return inter / (1.0 + 0.05 * len(ss))


def build_candidates(question: str, context: str, top_k: int, rng: random.Random) -> List[Dict[str, Any]]:
    q_words = tokenize_words(question)
    sents = split_into_sentences(context)
    cands = []
    for i, s in enumerate(sents):
        s_words = tokenize_words(s)
        sc = overlap_score(q_words, s_words) + rng.uniform(-0.02, 0.02)
        cands.append({"sid": i, "text": s, "score": float(sc)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:top_k]


def pick_evidence(candidates: List[Dict[str, Any]], n_min: int, n_max: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    n = min(rng.randint(n_min, n_max), len(candidates))
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


# =========================
# Tool schemas and SFT data
# =========================
REASONING_SYS = (
    "You are a clinical reasoning tool for PubMedQA-style questions.\n"
    "Given a QUESTION and CANDIDATE SENTENCES (each with id), output a valid JSON object with:\n"
    "evidence: list[{sid:int,text:str,polarity:str in [support, oppose, unclear]}]\n"
    "reasoning_steps: list[str]\n"
    "counterpoints: list[str]\n"
    "uncertainty_flags: list[str]\n"
    "confidence: float 0.0-1.0\n"
    "JSON only."
)

CONTEXT_SYS = (
    "You are a context extraction tool for PubMedQA-style questions.\n"
    "Given a QUESTION and full CONTEXT, output a valid JSON object with:\n"
    "key_sentences: list[{sid:int,text:str}]\n"
    "context_summary: str\n"
    "uncertainty_flags: list[str]\n"
    "confidence: float 0.0-1.0\n"
    "JSON only."
)

WEAK_UNCERTAINTY_FLAGS = ["weak_supervision_generation"]


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
) -> Tuple[str, str, str, str]:
    set_seed(seed)
    rows = load_raw_pubmedqa(data_path)
    splits = read_json(split_path)
    train_ids = set(splits["train_ids"])
    dev_ids = set(splits["dev_ids"])
    id2ex = {r["example_id"]: r for r in rows}

    tool_reason_train, tool_reason_dev = [], []
    tool_ctx_train, tool_ctx_dev = [], []

    def normalize_reasoning_obj(obj_r: Dict[str, Any]) -> Dict[str, Any]:
        ev = obj_r.get("evidence", [])
        if not isinstance(ev, list):
            ev = []
        norm_ev = []
        for it in ev[:6]:
            if isinstance(it, dict):
                sid = int(it.get("sid", -1)) if str(it.get("sid", "-1")).lstrip("-").isdigit() else -1
                txt = str(it.get("text", ""))[:240]
                pol = str(it.get("polarity", "unclear"))
                if pol not in ["support", "oppose", "unclear"]:
                    pol = "unclear"
                norm_ev.append({"sid": sid, "text": txt, "polarity": pol})
        obj_r["evidence"] = norm_ev
        obj_r["reasoning_steps"] = [str(x)[:180] for x in obj_r.get("reasoning_steps", [])[:5]] if isinstance(obj_r.get("reasoning_steps", []), list) else []
        obj_r["counterpoints"] = [str(x)[:180] for x in obj_r.get("counterpoints", [])[:3]] if isinstance(obj_r.get("counterpoints", []), list) else []
        uf = obj_r.get("uncertainty_flags", [])
        obj_r["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]] if isinstance(uf, list) else []
        try:
            obj_r["confidence"] = float(obj_r.get("confidence", 0.6))
        except Exception:
            obj_r["confidence"] = 0.6
        obj_r["confidence"] = max(0.0, min(1.0, obj_r["confidence"]))
        return obj_r

    def normalize_context_obj(obj_c: Dict[str, Any]) -> Dict[str, Any]:
        ks = obj_c.get("key_sentences", [])
        if not isinstance(ks, list):
            ks = []
        norm_ks = []
        for it in ks[:6]:
            if isinstance(it, dict):
                sid = int(it.get("sid", -1)) if str(it.get("sid", "-1")).lstrip("-").isdigit() else -1
                txt = str(it.get("text", ""))[:240]
                norm_ks.append({"sid": sid, "text": txt})
        obj_c["key_sentences"] = norm_ks
        obj_c["context_summary"] = str(obj_c.get("context_summary", ""))[:260]
        uf = obj_c.get("uncertainty_flags", [])
        obj_c["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]] if isinstance(uf, list) else []
        try:
            obj_c["confidence"] = float(obj_c.get("confidence", 0.6))
        except Exception:
            obj_c["confidence"] = 0.6
        obj_c["confidence"] = max(0.0, min(1.0, obj_c["confidence"]))
        return obj_c

    def add_one(eid: int, variants: int, reason_list: List[Dict[str, Any]], ctx_list: List[Dict[str, Any]]) -> None:
        ex = id2ex[eid]
        q, ctx = ex["question"], ex["context"]
        base_rng = random.Random(seed * 100000 + eid)
        for _ in range(variants):
            rng = random.Random(base_rng.randint(0, 10**9))
            candidates = build_candidates(q, ctx, top_k=top_k, rng=rng)
            evidence = pick_evidence(candidates, ev_min, ev_max, rng)

            obj_r = {
                "evidence": [{"sid": int(e["sid"]), "text": str(e["text"])[:240], "polarity": "unclear"} for e in evidence[:6]],
                "reasoning_steps": [
                    f"Question focus: {q[:120]}",
                    "Inspect candidate sentences for outcome direction and study signal.",
                    "Track conflicting or incomplete evidence.",
                ],
                "counterpoints": ["Evidence may be indirect or limited."],
                "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:],
                "confidence": 0.6,
            }
            obj_r = normalize_reasoning_obj(obj_r)
            cand_lines = "\n".join([f"[{c['sid']}] {c['text']}" for c in candidates])
            user_r = f"Example ID: {eid}\nQUESTION:\n{q}\n\nCANDIDATE SENTENCES:\n{cand_lines}\n"
            reason_list.append(
                {
                    "example_id": eid,
                    "prompt": [{"role": "system", "content": REASONING_SYS}, {"role": "user", "content": user_r}],
                    "response": json.dumps(obj_r, ensure_ascii=False),
                }
            )

            obj_c = {
                "key_sentences": [{"sid": int(e["sid"]), "text": str(e["text"])[:240]} for e in evidence[:6]],
                "context_summary": "Decision-relevant signals extracted from context.",
                "uncertainty_flags": WEAK_UNCERTAINTY_FLAGS[:],
                "confidence": 0.6,
            }
            obj_c = normalize_context_obj(obj_c)
            user_c = f"Example ID: {eid}\nQUESTION:\n{q}\n\nCONTEXT:\n{ctx}\n"
            ctx_list.append(
                {
                    "example_id": eid,
                    "prompt": [{"role": "system", "content": CONTEXT_SYS}, {"role": "user", "content": user_c}],
                    "response": json.dumps(obj_c, ensure_ascii=False),
                }
            )

    for eid in sorted(train_ids):
        add_one(eid, variants_train, tool_reason_train, tool_ctx_train)
    for eid in sorted(dev_ids):
        add_one(eid, variants_dev, tool_reason_dev, tool_ctx_dev)

    ensure_dir(out_dir)
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
    return reason_train_path, reason_dev_path, ctx_train_path, ctx_dev_path


# =========================
# SFT training
# =========================
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
                parts.append(json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _fallback_render_messages(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip() or "user"
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                payload = {"name": str(fn.get("name", "")).strip(), "arguments": fn.get("arguments", "{}")}
                parts.append(f"assistant_tool_call: {json.dumps(payload, ensure_ascii=False)}")
            content = _message_content_to_text(m.get("content"))
            if content:
                parts.append(f"assistant: {content}")
            continue
        if role == "tool":
            name = str(m.get("name", "tool")).strip() or "tool"
            parts.append(f"tool[{name}]: {_message_content_to_text(m.get('content'))}")
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
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return _fallback_render_messages(messages, add_generation_prompt)
    except Exception:
        return _fallback_render_messages(messages, add_generation_prompt)


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
        raise ValueError(f"Unsupported response type: {type(response)}")

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
) -> None:
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(tool_base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        tool_base_model,
        torch_dtype=runtime_dtype(),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("`peft` is required for LoRA.")
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set(name.split(".")[-1] for name, _ in model.named_modules())
        target_modules = [m for m in common if m in present] or ["q_proj", "v_proj"]
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
    ensure_dir(out_dir)
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[SFT] saved to: {out_dir}")


# =========================
# Native tool runtime
# =========================
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*.+?\s*</tool_call>", re.IGNORECASE | re.DOTALL)
TOOLS_TAG_RE = re.compile(r"<tools>.*?</tools>", re.IGNORECASE | re.DOTALL)
TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
TOOL_JSON_RE = re.compile(r'"name"\s*:\s*"(reasoning_tool|context_tool)".*?"arguments"\s*:', re.IGNORECASE | re.DOTALL)
PLAIN_TOOL_NAME_RE = re.compile(r"^\s*(reasoning_tool|context_tool)\s*(?:\(|\{|:|$)", re.IGNORECASE | re.MULTILINE)

# fixed: native tool call + argument mode only
MANAGER_TOOL_BINDING_MODE = "argument"
MAX_MANAGER_TOOL_CALLS = 2

ID2EX: Dict[int, Dict[str, Any]] = {}
REASONING_CACHE: Dict[int, str] = {}
CONTEXT_CACHE: Dict[int, str] = {}
REASONING_RAW_CACHE: Dict[int, str] = {}
CONTEXT_RAW_CACHE: Dict[int, str] = {}
ALLOWED_TOOL_IDS: Optional[set] = None
FAIL_BUFFER_JSONL: Optional[str] = None
RAW_TRACE_JSONL: Optional[str] = None
_IS_MAIN_PROCESS = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _append_raw_trace_rows(rows: List[Dict[str, Any]]) -> None:
    if rows and _IS_MAIN_PROCESS and RAW_TRACE_JSONL:
        append_jsonl(RAW_TRACE_JSONL, rows)


def _manager_tools_require_example_id() -> bool:
    return True


def _tool_call_args_for_current_mode(eid: int) -> Dict[str, Any]:
    return {"example_id": int(eid)}


def parse_answer_label_lastline(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    m = ANSWER_LASTLINE_RE.match(lines[-1])
    if not m:
        return None
    tok = m.group(1).upper()
    return ANSWER_TOKEN_TO_CANONICAL.get(tok)


def final_has_tool_call_artifacts(text: str) -> bool:
    if not text:
        return False
    txt = str(text)
    return bool(
        TOOL_CALL_TAG_RE.search(txt)
        or TOOLS_TAG_RE.search(txt)
        or TOOL_CALLS_FIELD_RE.search(txt)
        or TOOL_JSON_RE.search(txt)
        or PLAIN_TOOL_NAME_RE.search(txt)
    )


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

    assistant_texts = [_message_content_to_text(m.get("content")) for m in assistant_msgs]
    last_assistant_text = assistant_texts[-1] if assistant_texts else ""
    last_assistant_has_tool_calls = bool(assistant_msgs[-1].get("tool_calls")) if assistant_msgs else False
    any_tool_artifacts_anywhere = any(final_has_tool_call_artifacts(t) for t in assistant_texts)
    fake_tool_text_attempt = bool(any_tool_artifacts_anywhere and (len(tool_msgs) == 0 and tool_call_count == 0))

    return {
        "assistant_texts": assistant_texts,
        "tool_msg_count": len(tool_msgs),
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "tool_payloads": tool_payloads,
        "last_assistant_text": last_assistant_text,
        "last_assistant_has_tool_calls": last_assistant_has_tool_calls,
        "last_assistant_plaintext_tool_artifacts": bool(final_has_tool_call_artifacts(last_assistant_text)),
        "fake_tool_text_attempt": fake_tool_text_attempt,
    }


@dataclass
class FrozenAgent:
    tool_base_model: str
    adapter_path: Optional[str] = None
    device: str = "cpu"
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        if self.adapter_path:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft is required when adapter_path is set.")
            model = PeftModel.from_pretrained(model, self.adapter_path).to(self.device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 1e-6
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolBase:
    tool_base_model: str
    reasoning_adapter_path: Optional[str] = None
    context_adapter_path: Optional[str] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not PEFT_AVAILABLE:
            raise RuntimeError("SharedToolBase requires peft.")
        self.tok = AutoTokenizer.from_pretrained(self.tool_base_model, trust_remote_code=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.tool_base_model,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
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
        import threading
        self.lock = threading.Lock()

    @torch.no_grad()
    def generate(self, tool_name: str, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float = 0.0) -> str:
        adapter_name = self.adapter_names.get(tool_name)
        if adapter_name is None:
            raise RuntimeError(f"Adapter for tool `{tool_name}` is not loaded.")
        prompt = render_chat_messages(self.tok, messages, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 1e-6
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tok.pad_token_id,
            "eos_token_id": self.tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        with self.lock:
            self.model.set_adapter(adapter_name)
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1] :]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


@dataclass
class SharedToolView:
    shared_base: SharedToolBase
    tool_name: str
    max_new_tokens: int = 512

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        return self.shared_base.generate(self.tool_name, messages, self.max_new_tokens, temperature)


_shared_tool_base: Optional[SharedToolBase] = None
_reasoning_agent: Optional[Any] = None
_context_agent: Optional[Any] = None


def init_tool_agents(tool_base_model: str, reasoning_adapter: str, context_adapter: str, device: str) -> None:
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
            print(f"[WARN] shared tool base init failed; fallback to split tool models. {type(e).__name__}: {e}")

    if _reasoning_agent is None:
        _reasoning_agent = FrozenAgent(tool_base_model, reasoning_adapter, device=device, max_new_tokens=640)
    if _context_agent is None:
        _context_agent = FrozenAgent(tool_base_model, context_adapter, device=device, max_new_tokens=400)
    print("[TOOLS] runtime=split_models")


def _tool_guard(eid: int) -> Optional[str]:
    if ALLOWED_TOOL_IDS is not None and eid not in ALLOWED_TOOL_IDS:
        return json.dumps({"error": f"example_id {eid} not allowed in current split"}, ensure_ascii=False)
    return None


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
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
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def _normalize_reasoning_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    ev = obj.get("evidence", [])
    if not isinstance(ev, list):
        ev = []
    norm_ev = []
    for it in ev[:6]:
        if isinstance(it, dict):
            sid = int(it.get("sid", -1)) if str(it.get("sid", "-1")).lstrip("-").isdigit() else -1
            txt = str(it.get("text", ""))[:240]
            pol = str(it.get("polarity", "unclear"))
            if pol not in ["support", "oppose", "unclear"]:
                pol = "unclear"
            norm_ev.append({"sid": sid, "text": txt, "polarity": pol})
    obj["evidence"] = norm_ev
    obj["reasoning_steps"] = [str(x)[:180] for x in obj.get("reasoning_steps", [])[:5]] if isinstance(obj.get("reasoning_steps", []), list) else []
    obj["counterpoints"] = [str(x)[:180] for x in obj.get("counterpoints", [])[:3]] if isinstance(obj.get("counterpoints", []), list) else []
    uf = obj.get("uncertainty_flags", [])
    obj["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]] if isinstance(uf, list) else []
    try:
        obj["confidence"] = float(obj.get("confidence", 0.6))
    except Exception:
        obj["confidence"] = 0.6
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def _normalize_context_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    ks = obj.get("key_sentences", [])
    if not isinstance(ks, list):
        ks = []
    norm_ks = []
    for it in ks[:6]:
        if isinstance(it, dict):
            sid = int(it.get("sid", -1)) if str(it.get("sid", "-1")).lstrip("-").isdigit() else -1
            txt = str(it.get("text", ""))[:240]
            norm_ks.append({"sid": sid, "text": txt})
    obj["key_sentences"] = norm_ks
    obj["context_summary"] = str(obj.get("context_summary", ""))[:260]
    uf = obj.get("uncertainty_flags", [])
    obj["uncertainty_flags"] = [str(x)[:120] for x in uf[:3]] if isinstance(uf, list) else []
    try:
        obj["confidence"] = float(obj.get("confidence", 0.6))
    except Exception:
        obj["confidence"] = 0.6
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))
    return obj


def reasoning_tool(example_id: int) -> str:
    eid = int(example_id)
    guard = _tool_guard(eid)
    if guard is not None:
        return guard
    if eid in REASONING_CACHE:
        return REASONING_CACHE[eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = json.dumps({"error": "example_id not found"}, ensure_ascii=False)
        REASONING_CACHE[eid] = out
        REASONING_RAW_CACHE[eid] = out
        return out

    q, ctx = ex["question"], ex["context"]
    rng = random.Random(12345 + eid)
    candidates = build_candidates(q, ctx, top_k=20, rng=rng)
    cand_lines = "\n".join([f"[{c['sid']}] {c['text']}" for c in candidates])
    user = f"Example ID: {eid}\nQUESTION:\n{q}\n\nCANDIDATE SENTENCES:\n{cand_lines}\n"
    msgs = [{"role": "system", "content": REASONING_SYS}, {"role": "user", "content": user}]
    raw = _reasoning_agent.generate(msgs, temperature=0.0) if _reasoning_agent else ""
    REASONING_RAW_CACHE[eid] = raw
    obj = extract_first_json(raw)

    if obj is None:
        # explicit low-confidence fallback instead of pretending high-quality success
        obj = {
            "evidence": [{"sid": int(c["sid"]), "text": c["text"][:240], "polarity": "unclear"} for c in candidates[:4]],
            "reasoning_steps": [f"Question focus: {q[:120]}", "Tool output parsing failed; use evidence cautiously."],
            "counterpoints": ["reasoning_tool returned invalid JSON"],
            "uncertainty_flags": ["invalid_tool_output"],
            "confidence": 0.0,
        }

    obj = _normalize_reasoning_output(obj)
    out = json.dumps(obj, ensure_ascii=False)
    REASONING_CACHE[eid] = out
    _append_raw_trace_rows([{
        "ts": int(time.time()),
        "agent": "reasoning_tool",
        "event": "tool_call",
        "example_id": eid,
        "raw_output": raw,
        "normalized_output": out,
    }])
    return out


def context_tool(example_id: int) -> str:
    eid = int(example_id)
    guard = _tool_guard(eid)
    if guard is not None:
        return guard
    if eid in CONTEXT_CACHE:
        return CONTEXT_CACHE[eid]

    ex = ID2EX.get(eid)
    if ex is None:
        out = json.dumps({"error": "example_id not found"}, ensure_ascii=False)
        CONTEXT_CACHE[eid] = out
        CONTEXT_RAW_CACHE[eid] = out
        return out

    q, ctx = ex["question"], ex["context"]
    user = f"Example ID: {eid}\nQUESTION:\n{q}\n\nCONTEXT:\n{ctx}\n"
    msgs = [{"role": "system", "content": CONTEXT_SYS}, {"role": "user", "content": user}]
    raw = _context_agent.generate(msgs, temperature=0.0) if _context_agent else ""
    CONTEXT_RAW_CACHE[eid] = raw
    obj = extract_first_json(raw)

    if obj is None:
        rng = random.Random(67890 + eid)
        candidates = build_candidates(q, ctx, top_k=20, rng=rng)
        obj = {
            "key_sentences": [{"sid": int(c["sid"]), "text": c["text"][:240]} for c in candidates[:6]],
            "context_summary": "context_tool returned invalid JSON; use context summary cautiously.",
            "uncertainty_flags": ["invalid_tool_output"],
            "confidence": 0.0,
        }

    obj = _normalize_context_output(obj)
    out = json.dumps(obj, ensure_ascii=False)
    CONTEXT_CACHE[eid] = out
    _append_raw_trace_rows([{
        "ts": int(time.time()),
        "agent": "context_tool",
        "event": "tool_call",
        "example_id": eid,
        "raw_output": raw,
        "normalized_output": out,
    }])
    return out


# =========================
# Manager prompt: lightweight subgoal + memory
# =========================
def build_manager_system_prompt() -> str:
    if TASK_NAME == "medqa":
        task_line = "You are a planner-manager agent solving medical multiple-choice questions."
    elif TASK_NAME == "pubmedqa":
        task_line = "You are a planner-manager agent solving PubMedQA-style clinical questions."
    else:
        task_line = "You are a planner-manager agent solving clinical QA tasks."

    answer_lines = "\n".join([f"  ANSWER_{ANSWER_CANONICAL_TO_TOKEN[lab]}" for lab in ANSWER_LABELS])
    return (
        task_line + "\n"
        "Use ONLY the model's native tool-calling interface when calling tools.\n"
        "Available tools: reasoning_tool, context_tool.\n"
        "Tool arguments: always pass the exact current example_id from the user message.\n"
        f"You may make up to {MAX_MANAGER_TOOL_CALLS} tool calls total.\n\n"
        "Planning policy:\n"
        "- First form a brief internal plan around current uncertainty.\n"
        "- If direct answering is reliable, answer directly.\n"
        "- If uncertain, call ONE tool first.\n"
        "- Call the second tool only if uncertainty remains after reading the first tool result.\n"
        "- Use previous tool outputs as memory of what is already known; do not repeat the same tool call unless needed.\n"
        "- Prefer concise progress updates over long free-form reasoning.\n"
        "- Do not write XML tags, tool-call JSON, or pseudo-tool calls in plain text.\n\n"
        "If you answer, the final line must be exactly one of:\n"
        f"{answer_lines}\n"
        "Do not write anything after that last line.\n"
    )


MANAGER_SYSTEM = build_manager_system_prompt()


def _format_choices_block(choices: Optional[Dict[str, str]]) -> str:
    if not isinstance(choices, dict) or not choices:
        return ""
    items = _sorted_choice_items(choices)
    if not items:
        return ""
    return "Choices:\n" + "\n".join([f"{k}. {v}" for k, v in items]) + "\n\n"


def build_manager_messages(eid: int, q: str, ctx: str, choices: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    choices_block = _format_choices_block(choices)
    user_text = (
        f"Example ID: {eid}\n\n"
        f"Question:\n{q}\n\n"
        f"{choices_block}"
        f"Context:\n{ctx}\n\n"
        "Decide the next best action. Use native tool call if needed.\n"
        "When you call a tool, pass the exact Example ID shown above as the only argument.\n"
    )
    return [
        {"role": "system", "content": MANAGER_SYSTEM},
        {"role": "user", "content": user_text},
    ]


# =========================
# Reward shaping
# =========================
def shaped_manager_reward(prompts=None, completions=None, ground_truth=None, example_id=None, **kwargs):
    n = len(completions)
    gts = ensure_list(ground_truth, n)
    exids = ensure_list(example_id, n)

    rewards: List[float] = []
    fail_rows: List[Dict[str, Any]] = []
    manager_rows: List[Dict[str, Any]] = []

    for c, gt, eid in zip(completions, gts, exids):
        gt = _normalize_label(gt)
        st = extract_stats(c)
        pred = parse_answer_label_lastline(st["last_assistant_text"])
        valid_format = pred is not None
        final_has_artifacts = bool(
            st["last_assistant_has_tool_calls"] or final_has_tool_call_artifacts(st["last_assistant_text"])
        )
        fake_tool_text = bool(st["fake_tool_text_attempt"])
        tool_call_count = int(st["tool_call_count"])

        reward = 0.0
        if valid_format:
            reward += 0.1
        if tool_call_count > 0:
            reward += 0.05
        if 0 < tool_call_count <= MAX_MANAGER_TOOL_CALLS:
            reward += 0.05
        if tool_call_count > MAX_MANAGER_TOOL_CALLS:
            reward -= 0.15 * float(tool_call_count - MAX_MANAGER_TOOL_CALLS)
        if final_has_artifacts:
            reward -= 0.25
        if fake_tool_text:
            reward -= 0.35
        if valid_format and (pred == gt):
            reward += 1.0
            if tool_call_count > 0:
                reward += 0.15
        else:
            reward -= 0.05

        reward = float(max(-1.0, min(1.5, reward)))
        rewards.append(reward)

        row = {
            "ts": int(time.time()),
            "agent": "manager",
            "event": "completion",
            "example_id": int(eid) if eid is not None else None,
            "ground_truth": gt,
            "pred": pred,
            "reward": reward,
            "valid_format": bool(valid_format),
            "tool_call_count": tool_call_count,
            "tool_names": st.get("tool_names", []),
            "final_has_tool_artifacts": bool(final_has_artifacts),
            "fake_tool_text_attempt": bool(fake_tool_text),
            "assistant_texts": st.get("assistant_texts", []),
            "last_assistant_text": st.get("last_assistant_text", ""),
            "completion_raw": c,
        }
        manager_rows.append(row)
        if reward < 1.0 and _IS_MAIN_PROCESS and FAIL_BUFFER_JSONL:
            fail_rows.append(row)

    if fail_rows and _IS_MAIN_PROCESS and FAIL_BUFFER_JSONL:
        append_jsonl(FAIL_BUFFER_JSONL, fail_rows)
    _append_raw_trace_rows(manager_rows)
    return rewards


# =========================
# Manager GRPO
# =========================
def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    try:
        supported = set(inspect.signature(callable_obj).parameters.keys())
    except Exception:
        supported = None
    if supported is None:
        return dict(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted([k for k in kwargs if k not in filtered])
    if dropped:
        print(f"[{label}] skipped unsupported kwargs: {', '.join(dropped)}")
    return filtered


def _trainer_processing_kwargs(processing_obj: Any) -> Dict[str, Any]:
    try:
        supported = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    except Exception:
        supported = set()
    if "processing_class" in supported:
        return {"processing_class": processing_obj}
    if "tokenizer" in supported:
        return {"tokenizer": processing_obj}
    return {}


def validate_grpo_batch_geometry(per_device_train_bs: int, grad_accum: int, num_generations: int) -> None:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = int(per_device_train_bs) * world_size * int(grad_accum)
    if effective_batch <= 0:
        raise RuntimeError("effective batch size must be positive")
    if int(num_generations) <= 0:
        raise RuntimeError("num_generations must be positive")
    if effective_batch % int(num_generations) != 0:
        raise RuntimeError(
            f"Invalid GRPO batch geometry: {effective_batch} not divisible by num_generations={num_generations}"
        )


def train_manager_grpo_from_splits(
    manager_base_model: str,
    tool_base_model: str,
    data_path: str,
    split_path: str,
    save_dir: str,
    reasoning_adapter: str,
    context_adapter: str,
    seed: int = 42,
    per_device_train_bs: int = 8,
    grad_accum: int = 1,
    max_prompt_length: int = 2048,
    max_completion_length: int = 4096,
    temperature: float = 0.5,
    num_generations: int = 8,
    grpo_beta: float = 0.001,
    fail_buffer_jsonl: str = "",
    raw_trace_jsonl: str = "",
    use_wandb: bool = False,
    manager_use_lora: bool = False,
    manager_lora_r: int = 16,
    manager_lora_alpha: int = 32,
    manager_lora_dropout: float = 0.05,
    manager_gradient_checkpointing: bool = True,
) -> None:
    require_clean_runtime()
    validate_grpo_batch_geometry(per_device_train_bs, grad_accum, num_generations)
    set_seed(seed)

    global FAIL_BUFFER_JSONL, RAW_TRACE_JSONL, ALLOWED_TOOL_IDS
    FAIL_BUFFER_JSONL = fail_buffer_jsonl.strip() or None
    RAW_TRACE_JSONL = raw_trace_jsonl.strip() or None

    rows = load_raw_pubmedqa(data_path)
    splits = read_json(split_path)
    train_ids = set(splits["train_ids"])
    id2ex_full = {r["example_id"]: r for r in rows}

    ID2EX.clear()
    for r in rows:
        ID2EX[int(r["example_id"])] = {"question": r["question"], "context": r["context"]}
    ALLOWED_TOOL_IDS = set(train_ids)
    REASONING_CACHE.clear()
    CONTEXT_CACHE.clear()
    REASONING_RAW_CACHE.clear()
    CONTEXT_RAW_CACHE.clear()

    device = device_str()
    init_tool_agents(tool_base_model, reasoning_adapter, context_adapter, device=device)

    manager_tok = AutoTokenizer.from_pretrained(manager_base_model, trust_remote_code=True)
    manager_tok.padding_side = "left"
    if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
        manager_tok.pad_token_id = manager_tok.eos_token_id

    train_rows = [id2ex_full[eid] for eid in sorted(train_ids)]
    dataset = Dataset.from_list(train_rows)

    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        eid = int(ex["example_id"])
        msgs = build_manager_messages(eid, ex["question"], ex["context"], choices=ex.get("choices", {}))
        return {"prompt": msgs, "ground_truth": ex["ground_truth"], "example_id": eid}

    train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    grpo_kwargs = {
        "output_dir": save_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": int(per_device_train_bs),
        "gradient_accumulation_steps": int(grad_accum),
        "max_prompt_length": int(max_prompt_length),
        "max_completion_length": int(max_completion_length),
        "num_generations": int(num_generations),
        "temperature": float(temperature),
        "do_sample": True,
        "beta": float(grpo_beta),
        "scale_rewards": "group",
        "bf16": torch.cuda.is_available(),
        "logging_steps": 1,
        "log_completions": True,
        "num_completions_to_print": None,
        "log_unique_prompts": False,
        "report_to": (["wandb"] if use_wandb else []),
        "max_tool_calling_iterations": int(MAX_MANAGER_TOOL_CALLS),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    grpo_args = GRPOConfig(**_filter_supported_kwargs(GRPOConfig.__init__, grpo_kwargs, "GRPOConfig"))

    manager_model = AutoModelForCausalLM.from_pretrained(
        manager_base_model,
        torch_dtype=runtime_dtype(),
        trust_remote_code=True,
    )

    # clean and explicit generation config: do not inherit noisy sampling defaults
    manager_model.config.use_cache = False
    manager_model.generation_config = GenerationConfig.from_model_config(manager_model.config)
    manager_model.generation_config.do_sample = True
    manager_model.generation_config.temperature = float(temperature)
    manager_model.generation_config.top_p = 1.0
    manager_model.generation_config.top_k = 0
    manager_model.generation_config.pad_token_id = manager_tok.pad_token_id
    manager_model.generation_config.eos_token_id = manager_tok.eos_token_id
    if not hasattr(manager_model, "warnings_issued") or manager_model.warnings_issued is None:
        manager_model.warnings_issued = {}

    print("[GENCFG] do_sample =", manager_model.generation_config.do_sample)
    print("[GENCFG] temperature =", manager_model.generation_config.temperature)
    print("[GENCFG] top_p =", manager_model.generation_config.top_p)
    print("[GENCFG] top_k =", manager_model.generation_config.top_k)

    if manager_use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("`peft` is required for manager LoRA.")
        common = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = set(name.split(".")[-1] for name, _ in manager_model.named_modules())
        target_modules = [m for m in common if m in present] or ["q_proj", "v_proj"]
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
        "reward_funcs": [shaped_manager_reward],
        "rollout_func": None,
        "tools": [reasoning_tool, context_tool],
    }
    trainer_kwargs.update(_trainer_processing_kwargs(manager_tok))
    print("[TRAINER] native tools =", [reasoning_tool.__name__, context_tool.__name__])

    trainer = GRPOTrainer(**_filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs, "GRPOTrainer"))
    trainer.train()

    ensure_dir(save_dir)
    trainer.model.save_pretrained(save_dir)
    manager_tok.save_pretrained(save_dir)
    print(f"[GRPO] saved manager to: {save_dir}")


# =========================
# CLI
# =========================
def main() -> None:
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
        ],
    )

    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--manager_base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tool_base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--task_name", type=str, default="pubmedqa", choices=["pubmedqa", "medqa", "generic"])
    parser.add_argument("--label_space", type=str, default="")

    # split
    parser.add_argument("--split_path", type=str, default="splits_pubmedqa_1000.json")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--dev_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=-1)

    # tool SFT data
    parser.add_argument("--tool_sft_out_dir", type=str, default="tool_sft_data_clean")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tool_variants_train", type=int, default=3)
    parser.add_argument("--tool_variants_dev", type=int, default=2)
    parser.add_argument("--ev_min", type=int, default=3)
    parser.add_argument("--ev_max", type=int, default=6)

    # tool SFT train
    parser.add_argument("--reasoning_tool_out", type=str, default="reasoning_lora_clean")
    parser.add_argument("--context_tool_out", type=str, default="context_lora_clean")
    parser.add_argument("--tool_lr", type=float, default=2e-4)
    parser.add_argument("--tool_epochs", type=int, default=2)
    parser.add_argument("--tool_bs", type=int, default=1)
    parser.add_argument("--tool_grad_accum", type=int, default=8)
    parser.add_argument("--tool_max_seq_len", type=int, default=2048)
    parser.add_argument("--tool_use_lora", action="store_true")
    parser.add_argument("--tool_lora_r", type=int, default=8)
    parser.add_argument("--tool_lora_alpha", type=int, default=16)
    parser.add_argument("--tool_lora_dropout", type=float, default=0.05)

    # manager GRPO
    parser.add_argument("--manager_out", type=str, default="manager_grpo_clean")
    parser.add_argument("--mgr_bs", type=int, default=8)
    parser.add_argument("--mgr_grad_accum", type=int, default=1)
    parser.add_argument("--mgr_max_prompt_length", type=int, default=2048)
    parser.add_argument("--mgr_max_completion_length", type=int, default=512)
    parser.add_argument("--mgr_temperature", type=float, default=0.5)
    parser.add_argument("--mgr_num_generations", type=int, default=8)
    parser.add_argument("--grpo_beta", type=float, default=0.001)
    parser.add_argument("--fail_buffer_jsonl", type=str, default="")
    parser.add_argument("--raw_trace_jsonl", type=str, default="")
    parser.add_argument("--grpo_use_wandb", action="store_true")

    # manager lora (optional)
    parser.add_argument("--mgr_use_lora", action="store_true")
    parser.add_argument("--mgr_lora_r", type=int, default=16)
    parser.add_argument("--mgr_lora_alpha", type=int, default=32)
    parser.add_argument("--mgr_lora_dropout", type=float, default=0.05)
    parser.add_argument("--mgr_gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    configure_task(args.task_name, args.label_space)
    data_path = resolve_data_path_arg(args.data_path, TASK_NAME)

    if args.base_model.strip():
        args.manager_base_model = args.base_model
        args.tool_base_model = args.base_model

    if args.stage == "make_splits":
        rows = load_raw_dataset(data_path, task_name=TASK_NAME)
        if args.max_samples > 0:
            sample_seed = args.seed if args.sample_seed < 0 else args.sample_seed
            rows = subsample_rows(rows, args.max_samples, seed=sample_seed)
            print(f"[SUBSAMPLE] kept {len(rows)} rows with seed={sample_seed}")
        splits = make_splits(rows, test_size=args.test_size, dev_size=args.dev_size, seed=args.seed)
        write_json(args.split_path, splits)
        print(
            f"[SPLITS] train/dev/test = {len(splits['train_ids'])}/{len(splits['dev_ids'])}/{len(splits['test_ids'])}"
        )
        print(f"[SPLITS] wrote: {args.split_path}")
        return

    if args.stage == "build_tool_sft":
        paths = build_tool_sft_data_from_splits(
            data_path=data_path,
            split_path=args.split_path,
            out_dir=args.tool_sft_out_dir,
            seed=args.seed,
            top_k=args.top_k,
            variants_train=args.tool_variants_train,
            variants_dev=args.tool_variants_dev,
            ev_min=args.ev_min,
            ev_max=args.ev_max,
        )
        print("[TOOL SFT DATA] files =", paths)
        return

    if args.stage == "train_tool_reasoning":
        train_sft_agent(
            tool_base_model=args.tool_base_model,
            train_jsonl=os.path.join(args.tool_sft_out_dir, "tool_reasoning_train.jsonl"),
            dev_jsonl=os.path.join(args.tool_sft_out_dir, "tool_reasoning_dev.jsonl"),
            out_dir=args.reasoning_tool_out,
            seed=args.seed,
            max_seq_len=args.tool_max_seq_len,
            lr=args.tool_lr,
            epochs=args.tool_epochs,
            per_device_bs=args.tool_bs,
            grad_accum=args.tool_grad_accum,
            use_lora=args.tool_use_lora,
            lora_r=args.tool_lora_r,
            lora_alpha=args.tool_lora_alpha,
            lora_dropout=args.tool_lora_dropout,
        )
        return

    if args.stage == "train_tool_context":
        train_sft_agent(
            tool_base_model=args.tool_base_model,
            train_jsonl=os.path.join(args.tool_sft_out_dir, "tool_context_train.jsonl"),
            dev_jsonl=os.path.join(args.tool_sft_out_dir, "tool_context_dev.jsonl"),
            out_dir=args.context_tool_out,
            seed=args.seed,
            max_seq_len=args.tool_max_seq_len,
            lr=args.tool_lr,
            epochs=args.tool_epochs,
            per_device_bs=args.tool_bs,
            grad_accum=args.tool_grad_accum,
            use_lora=args.tool_use_lora,
            lora_r=args.tool_lora_r,
            lora_alpha=args.tool_lora_alpha,
            lora_dropout=args.tool_lora_dropout,
        )
        return

    if args.stage == "train_manager_grpo":
        fb = args.fail_buffer_jsonl.strip() or os.path.join(args.manager_out, "fail_buffer.jsonl")
        rt = args.raw_trace_jsonl.strip() or os.path.join(args.manager_out, "train_raw_trace.jsonl")
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
            manager_use_lora=args.mgr_use_lora,
            manager_lora_r=args.mgr_lora_r,
            manager_lora_alpha=args.mgr_lora_alpha,
            manager_lora_dropout=args.mgr_lora_dropout,
            manager_gradient_checkpointing=args.mgr_gradient_checkpointing,
        )
        return


if __name__ == "__main__":
    main()
