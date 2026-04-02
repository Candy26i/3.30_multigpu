# -*- coding: utf-8 -*-
"""
Evaluate and compare:
1) pipeline manager + tools
2) manager without tools
3) direct baseline models
4) random / majority baselines


Model paths (pipeline with tools):
- Manager LLM: --pipeline_manager_dir (always separate from tools).
- Tool LLMs: each tool is a FrozenAgent. Use either a full checkpoint dir
 (--pipeline_reasoning_model_dir / --pipeline_context_model_dir) or a base
 model plus optional LoRA (--pipeline_*_adapter). By default both tools
 share --pipeline_base_model_for_tools; override per tool with
 --pipeline_reasoning_base_model / --pipeline_context_base_model when you
 want different bases without loading a merged full dir for each tool.
"""


import argparse
import csv
import gc
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


import agents_as_tools as m




def set_seed(seed: int) -> None:
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)




def write_json(path: str, obj: Any) -> None:
   os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
   with open(path, "w", encoding="utf-8") as f:
       json.dump(obj, f, ensure_ascii=False, indent=2)




def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
   os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
   with open(path, "w", encoding="utf-8") as f:
       for r in rows:
           f.write(json.dumps(r, ensure_ascii=False) + "\n")




def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
   os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
   with open(path, "w", encoding="utf-8", newline="") as f:
       writer = csv.DictWriter(f, fieldnames=fieldnames)
       writer.writeheader()
       for r in rows:
           writer.writerow({k: r.get(k, "") for k in fieldnames})




def parse_csv_arg(text: str) -> List[str]:
   if not text:
       return []
   out = []
   for x in text.split(","):
       s = x.strip()
       if s:
           out.append(s)
   return out




def unload_cuda() -> None:
   gc.collect()
   if torch.cuda.is_available():
       torch.cuda.empty_cache()




def load_lm(model_dir: str, device: str) -> Tuple[Any, Any]:
   tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
   tok.padding_side = "left"
   if tok.pad_token_id is None and tok.eos_token_id is not None:
       tok.pad_token_id = tok.eos_token_id
   dtype = torch.bfloat16 if device == "cuda" else torch.float32
   model = AutoModelForCausalLM.from_pretrained(
       model_dir,
       torch_dtype=dtype,
       trust_remote_code=True,
   ).to(device)
   model.eval()
   model.config.use_cache = False
   return tok, model




def render_prompt(tok: Any, messages: List[Dict[str, Any]]) -> str:
   try:
       return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
   except TypeError:
       return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   except Exception:
       flat = []
       for mm in messages:
           if mm.get("role") == "tool":
               nm = mm.get("name", "tool")
               ct = mm.get("content", "")
               flat.append({"role": "user", "content": f"TOOL_OUTPUT {nm}:\n{ct}\n"})
           else:
               flat.append(mm)
       try:
           return tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
       except TypeError:
           return tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True)
       except Exception:
           parts = []
           for mm in flat:
               parts.append(f"{mm.get('role','')}: {mm.get('content','')}")
           return "\n".join(parts) + "\nassistant: "




@torch.no_grad()
def generate_text(model: Any, tok: Any, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
   prompt = render_prompt(tok, messages)
   inputs = tok(prompt, return_tensors="pt").to(model.device)
   out = model.generate(
       **inputs,
       max_new_tokens=int(max_new_tokens),
       do_sample=(temperature > 1e-6),
       temperature=max(float(temperature), 1e-6),
       pad_token_id=tok.pad_token_id,
       eos_token_id=tok.eos_token_id,
   )
   gen = out[0, inputs["input_ids"].shape[1]:]
   return tok.decode(gen, skip_special_tokens=True).strip()




TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)




def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
   if not text:
       return None
   m1 = TOOL_CALL_RE.search(text)
   if m1:
       raw = m1.group(1)
       try:
           obj = json.loads(raw)
       except Exception:
           return None
   else:
       obj = m.extract_first_json(text)
   if not isinstance(obj, dict):
       return None
   if "name" not in obj or "arguments" not in obj:
       return None
   name = str(obj.get("name"))
   args = obj.get("arguments", {})
   if isinstance(args, str):
       try:
           args = json.loads(args)
       except Exception:
           args = {}
   if not isinstance(args, dict):
       args = {}
   return name, args




def build_direct_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
   answer_lines = "\n".join([f"  ANSWER_{m.ANSWER_CANONICAL_TO_TOKEN[lb]}" for lb in m.ANSWER_LABELS])
   choices_block = m._format_choices_block(ex.get("choices", {}))
   sys = (
       "You solve medical QA tasks.\n"
       "Do NOT call tools.\n"
       "Final answer must end with exactly one line:\n"
       f"{answer_lines}\n"
       "Do not write anything after that last line.\n"
   )
   usr = (
       f"Example ID: {int(ex['example_id'])}\n\n"
       f"Question:\n{ex['question']}\n\n"
       f"{choices_block}"
       f"Context:\n{ex['context']}\n\n"
       "Answer now.\n"
   )
   return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]




def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
   n = len(y_true)
   assert n == len(y_pred)
   if n == 0:
       return {
           "n": 0,
           "accuracy": 0.0,
           "macro_f1": 0.0,
           "weighted_f1": 0.0,
           "invalid_rate": 0.0,
           "per_label": {},
           "confusion": {},
           "invalid_by_true": {},
       }


   correct = 0
   invalid = 0
   for yt, yp in zip(y_true, y_pred):
       if yp not in labels:
           invalid += 1
       if yp == yt and yp in labels:
           correct += 1
   acc = float(correct) / float(n)
   invalid_rate = float(invalid) / float(n)


   per_label: Dict[str, Dict[str, float]] = {}
   f1s = []
   supports = []
   for lb in labels:
       tp = fp = fn = support = 0
       for yt, yp in zip(y_true, y_pred):
           if yt == lb:
               support += 1
           if yt == lb and yp == lb:
               tp += 1
           elif yt != lb and yp == lb:
               fp += 1
           elif yt == lb and yp != lb:
               fn += 1
       p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
       r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
       f1 = 0.0 if (p + r) <= 0.0 else (2.0 * p * r / (p + r))
       per_label[lb] = {"precision": p, "recall": r, "f1": f1, "support": support}
       f1s.append(f1)
       supports.append(support)


   macro_f1 = float(np.mean(f1s)) if f1s else 0.0
   weighted_f1 = 0.0
   total_support = float(sum(supports))
   if total_support > 0:
       weighted_f1 = float(sum([f * s for f, s in zip(f1s, supports)]) / total_support)


   confusion = {yt: {yp: 0 for yp in labels} for yt in labels}
   invalid_by_true = {yt: 0 for yt in labels}
   for yt, yp in zip(y_true, y_pred):
       if yt not in confusion:
           continue
       if yp in confusion[yt]:
           confusion[yt][yp] += 1
       else:
           invalid_by_true[yt] += 1


   return {
       "n": n,
       "accuracy": acc,
       "macro_f1": macro_f1,
       "weighted_f1": weighted_f1,
       "invalid_rate": invalid_rate,
       "per_label": per_label,
       "confusion": confusion,
       "invalid_by_true": invalid_by_true,
   }




def build_eval_rows(
   all_rows: List[Dict[str, Any]],
   split_obj: Dict[str, Any],
   split_key: str,
   max_eval_samples: int,
   seed: int,
) -> List[Dict[str, Any]]:
   ids = split_obj.get(split_key, [])
   if not isinstance(ids, list) or not ids:
       raise ValueError(f"split key {split_key!r} missing or empty")
   wanted = set([int(x) for x in ids])
   rows = [r for r in all_rows if int(r["example_id"]) in wanted]
   if not rows:
       raise ValueError(f"No rows matched split_key={split_key}")
   if max_eval_samples > 0 and len(rows) > int(max_eval_samples):
       rng = random.Random(seed)
       rng.shuffle(rows)
       rows = rows[: int(max_eval_samples)]
   rows.sort(key=lambda x: int(x["example_id"]))
   return rows




def init_eval_state(eval_rows: List[Dict[str, Any]]) -> None:
   m.ID2EX.clear()
   m.REASONING_CACHE.clear()
   m.CONTEXT_CACHE.clear()
   m.REASONING_RAW_CACHE.clear()
   m.CONTEXT_RAW_CACHE.clear()
   for r in eval_rows:
       m.ID2EX[int(r["example_id"])] = {"question": r["question"], "context": r["context"]}
   m.ALLOWED_TOOL_IDS = set([int(r["example_id"]) for r in eval_rows])




def init_tool_agents_for_system(system: Dict[str, Any], device: str) -> None:
   m._reasoning_agent = None
   m._context_agent = None


   base_model_for_tools = str(system.get("base_model_for_tools", "") or "").strip()
   if not base_model_for_tools:
       base_model_for_tools = "Qwen/Qwen3-0.6B"
   reasoning_base = str(system.get("reasoning_base_model", "") or "").strip() or base_model_for_tools
   context_base = str(system.get("context_base_model", "") or "").strip() or base_model_for_tools
   reasoning_model_dir = str(system.get("reasoning_model_dir", "") or "").strip()
   context_model_dir = str(system.get("context_model_dir", "") or "").strip()
   reasoning_adapter = str(system.get("reasoning_adapter", "") or "").strip()
   context_adapter = str(system.get("context_adapter", "") or "").strip()


   if reasoning_model_dir:
       m._reasoning_agent = m.FrozenAgent(reasoning_model_dir, adapter_path=None, device=device, max_new_tokens=640)
   else:
       m._reasoning_agent = m.FrozenAgent(
           reasoning_base,
           adapter_path=(reasoning_adapter or None),
           device=device,
           max_new_tokens=640,
       )


   if context_model_dir:
       m._context_agent = m.FrozenAgent(context_model_dir, adapter_path=None, device=device, max_new_tokens=400)
   else:
       m._context_agent = m.FrozenAgent(
           context_base,
           adapter_path=(context_adapter or None),
           device=device,
           max_new_tokens=400,
       )




def eval_manager_system(
   system: Dict[str, Any],
   eval_rows: List[Dict[str, Any]],
   device: str,
   max_new_tokens: int,
   temperature: float,
   max_tool_calls: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
   manager_dir = str(system["manager_dir"])
   use_tools = bool(system.get("use_tools", True))


   if use_tools:
       init_tool_agents_for_system(system=system, device=device)
   else:
       m._reasoning_agent = None
       m._context_agent = None


   tok, model = load_lm(manager_dir, device=device)


   y_true: List[str] = []
   y_pred: List[str] = []
   pred_rows: List[Dict[str, Any]] = []
   tool_calls_list: List[int] = []
   t0 = time.time()


   for i, ex in enumerate(eval_rows, start=1):
       eid = int(ex["example_id"])
       gt = str(ex["ground_truth"])
       messages = m.build_manager_messages(eid, ex["question"], ex["context"], choices=ex.get("choices", {}))


       final_text = ""
       tool_calls_used = 0
       tool_trace: List[Dict[str, Any]] = []


       for _ in range(int(max_tool_calls) + 1):
           final_text = generate_text(
               model=model,
               tok=tok,
               messages=messages,
               max_new_tokens=max_new_tokens,
               temperature=temperature,
           ).strip()


           if not use_tools or tool_calls_used >= int(max_tool_calls):
               break


           tc = parse_tool_call(final_text)
           if tc is None:
               break


           tool_name, args = tc
           tool_eid = int(args.get("example_id", eid))
           if tool_name == "reasoning_tool":
               tool_out = m.reasoning_tool(tool_eid)
           elif tool_name == "context_tool":
               tool_out = m.context_tool(tool_eid)
           else:
               tool_out = m.dumps_json({"error": f"unknown tool: {tool_name}"})


           tool_trace.append({"tool": tool_name, "example_id": tool_eid, "tool_output": tool_out})
           messages.append({"role": "assistant", "content": final_text})
           messages.append({"role": "tool", "name": tool_name, "content": tool_out})
           tool_calls_used += 1


       pred = m.parse_answer_label_lastline(final_text)
       pred = pred if pred is not None else "__INVALID__"
       y_true.append(gt)
       y_pred.append(pred)
       tool_calls_list.append(tool_calls_used)
       pred_rows.append(
           {
               "system": system["name"],
               "mode": system["mode"],
               "example_id": eid,
               "ground_truth": gt,
               "pred": pred,
               "correct": bool(pred == gt),
               "valid_format": bool(pred in m.ANSWER_LABELS),
               "tool_calls_used": tool_calls_used,
               "final_text": final_text,
               "tool_trace": tool_trace,
           }
       )


       if i % 50 == 0:
           done_acc = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)]))
           print(f"[{system['name']}] {i}/{len(eval_rows)} acc={done_acc:.4f}")


   metrics = compute_metrics(y_true, y_pred, labels=m.ANSWER_LABELS)
   metrics["avg_tool_calls"] = float(np.mean(tool_calls_list)) if tool_calls_list else 0.0
   metrics["elapsed_sec"] = float(time.time() - t0)


   del model
   del tok
   unload_cuda()
   return metrics, pred_rows




def eval_direct_system(
   system: Dict[str, Any],
   eval_rows: List[Dict[str, Any]],
   device: str,
   max_new_tokens: int,
   temperature: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
   model_dir = str(system["model_dir"])
   tok, model = load_lm(model_dir, device=device)


   y_true: List[str] = []
   y_pred: List[str] = []
   pred_rows: List[Dict[str, Any]] = []
   t0 = time.time()
   for i, ex in enumerate(eval_rows, start=1):
       eid = int(ex["example_id"])
       gt = str(ex["ground_truth"])
       final_text = generate_text(
           model=model,
           tok=tok,
           messages=build_direct_messages(ex),
           max_new_tokens=max_new_tokens,
           temperature=temperature,
       ).strip()
       pred = m.parse_answer_label_lastline(final_text)
       pred = pred if pred is not None else "__INVALID__"


       y_true.append(gt)
       y_pred.append(pred)
       pred_rows.append(
           {
               "system": system["name"],
               "mode": system["mode"],
               "example_id": eid,
               "ground_truth": gt,
               "pred": pred,
               "correct": bool(pred == gt),
               "valid_format": bool(pred in m.ANSWER_LABELS),
               "tool_calls_used": 0,
               "final_text": final_text,
           }
       )
       if i % 50 == 0:
           done_acc = float(np.mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)]))
           print(f"[{system['name']}] {i}/{len(eval_rows)} acc={done_acc:.4f}")


   metrics = compute_metrics(y_true, y_pred, labels=m.ANSWER_LABELS)
   metrics["avg_tool_calls"] = 0.0
   metrics["elapsed_sec"] = float(time.time() - t0)
   del model
   del tok
   unload_cuda()
   return metrics, pred_rows




def eval_random_system(system: Dict[str, Any], eval_rows: List[Dict[str, Any]], seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
   rng = random.Random(seed + 17)
   y_true = []
   y_pred = []
   rows = []
   t0 = time.time()
   for ex in eval_rows:
       gt = str(ex["ground_truth"])
       pred = rng.choice(m.ANSWER_LABELS)
       y_true.append(gt)
       y_pred.append(pred)
       rows.append(
           {
               "system": system["name"],
               "mode": system["mode"],
               "example_id": int(ex["example_id"]),
               "ground_truth": gt,
               "pred": pred,
               "correct": bool(pred == gt),
               "valid_format": True,
               "tool_calls_used": 0,
               "final_text": pred,
           }
       )
   metrics = compute_metrics(y_true, y_pred, labels=m.ANSWER_LABELS)
   metrics["avg_tool_calls"] = 0.0
   metrics["elapsed_sec"] = float(time.time() - t0)
   return metrics, rows




def eval_majority_system(
   system: Dict[str, Any],
   eval_rows: List[Dict[str, Any]],
   majority_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
   y_true = []
   y_pred = []
   rows = []
   t0 = time.time()
   for ex in eval_rows:
       gt = str(ex["ground_truth"])
       pred = str(majority_label)
       y_true.append(gt)
       y_pred.append(pred)
       rows.append(
           {
               "system": system["name"],
               "mode": system["mode"],
               "example_id": int(ex["example_id"]),
               "ground_truth": gt,
               "pred": pred,
               "correct": bool(pred == gt),
               "valid_format": True,
               "tool_calls_used": 0,
               "final_text": pred,
           }
       )
   metrics = compute_metrics(y_true, y_pred, labels=m.ANSWER_LABELS)
   metrics["avg_tool_calls"] = 0.0
   metrics["elapsed_sec"] = float(time.time() - t0)
   return metrics, rows




def build_system_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
   systems: List[Dict[str, Any]] = []
   if args.pipeline_manager_dir.strip():
       systems.append(
           {
               "name": "pipeline_tools",
               "mode": "manager_tools",
               "manager_dir": args.pipeline_manager_dir.strip(),
               "use_tools": True,
               "base_model_for_tools": args.pipeline_base_model_for_tools,
               "reasoning_base_model": args.pipeline_reasoning_base_model,
               "context_base_model": args.pipeline_context_base_model,
               "reasoning_adapter": args.pipeline_reasoning_adapter,
               "context_adapter": args.pipeline_context_adapter,
               "reasoning_model_dir": args.pipeline_reasoning_model_dir,
               "context_model_dir": args.pipeline_context_model_dir,
           }
       )
       if args.add_pipeline_no_tools_baseline:
           systems.append(
               {
                   "name": "pipeline_no_tools",
                   "mode": "manager_no_tools",
                   "manager_dir": args.pipeline_manager_dir.strip(),
                   "use_tools": False,
               }
           )


   baseline_dirs = parse_csv_arg(args.baseline_model_dirs)
   baseline_names = parse_csv_arg(args.baseline_model_names)
   for i, d in enumerate(baseline_dirs):
       nm = baseline_names[i] if i < len(baseline_names) else f"direct_baseline_{i+1}"
       systems.append({"name": nm, "mode": "direct_model", "model_dir": d})


   if args.add_random_baseline:
       systems.append({"name": "random_baseline", "mode": "random"})
   if args.add_majority_baseline:
       systems.append({"name": "majority_baseline", "mode": "majority"})


   if not systems:
       raise ValueError("No systems specified. Provide pipeline manager and/or baseline models.")
   return systems




def infer_majority_label(all_rows: List[Dict[str, Any]], split_obj: Dict[str, Any]) -> str:
   counts = {lb: 0 for lb in m.ANSWER_LABELS}
   train_ids = split_obj.get("train_ids", [])
   train_set = set([int(x) for x in train_ids]) if isinstance(train_ids, list) else set()
   for r in all_rows:
       if train_set and int(r["example_id"]) not in train_set:
           continue
       gt = str(r["ground_truth"])
       if gt in counts:
           counts[gt] += 1
   return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]




def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--task_name", type=str, default="pubmedqa", choices=["pubmedqa", "medqa", "generic"])
   parser.add_argument("--label_space", type=str, default="")
   parser.add_argument("--data_path", type=str, default="")
   parser.add_argument("--split_path", type=str, required=True)
   parser.add_argument("--split_key", type=str, default="test_ids")
   parser.add_argument("--max_eval_samples", type=int, default=0)
   parser.add_argument("--seed", type=int, default=42)


   parser.add_argument(
       "--pipeline_manager_dir",
       type=str,
       default="",
       help="HF or local path for the manager (orchestrator) model; independent from tool models.",
   )
   parser.add_argument(
       "--pipeline_base_model_for_tools",
       type=str,
       default="Qwen/Qwen3-0.6B",
       help="Default base for both tools when --pipeline_reasoning_model_dir / "
       "--pipeline_context_model_dir are unset and per-tool *_base_model args are unset.",
   )
   parser.add_argument(
       "--pipeline_reasoning_base_model",
       type=str,
       default="",
       help="Base model for reasoning_tool only (when --pipeline_reasoning_model_dir is unset). "
       "Defaults to --pipeline_base_model_for_tools.",
   )
   parser.add_argument(
       "--pipeline_context_base_model",
       type=str,
       default="",
       help="Base model for context_tool only (when --pipeline_context_model_dir is unset). "
       "Defaults to --pipeline_base_model_for_tools.",
   )
   parser.add_argument(
       "--pipeline_reasoning_adapter",
       type=str,
       default="",
       help="LoRA/PEFT path for reasoning_tool; used with reasoning base (ignored if reasoning_model_dir is set).",
   )
   parser.add_argument(
       "--pipeline_context_adapter",
       type=str,
       default="",
       help="LoRA/PEFT path for context_tool; used with context base (ignored if context_model_dir is set).",
   )
   parser.add_argument(
       "--pipeline_reasoning_model_dir",
       type=str,
       default="",
       help="Full merged checkpoint for reasoning_tool; if set, overrides reasoning base and adapter.",
   )
   parser.add_argument(
       "--pipeline_context_model_dir",
       type=str,
       default="",
       help="Full merged checkpoint for context_tool; if set, overrides context base and adapter.",
   )
   parser.add_argument("--add_pipeline_no_tools_baseline", action="store_true")


   parser.add_argument("--baseline_model_dirs", type=str, default="")
   parser.add_argument("--baseline_model_names", type=str, default="")
   parser.add_argument("--add_random_baseline", action="store_true")
   parser.add_argument("--add_majority_baseline", action="store_true")


   parser.add_argument("--temperature", type=float, default=0.0)
   parser.add_argument("--max_new_tokens", type=int, default=1024)
   parser.add_argument("--max_tool_calls", type=int, default=2)
   parser.add_argument("--out_dir", type=str, default="eval_compare_outputs")
   args = parser.parse_args()


   set_seed(args.seed)
   configured_task, configured_labels = m.configure_task(args.task_name, args.label_space)
   m.MANAGER_SYSTEM = m.build_manager_system_prompt()


   data_path = m.resolve_data_path_arg(args.data_path, configured_task)
   all_rows = m.load_raw_dataset(path=data_path, task_name=configured_task)
   split_obj = m.read_json(args.split_path)
   eval_rows = build_eval_rows(
       all_rows=all_rows,
       split_obj=split_obj,
       split_key=args.split_key,
       max_eval_samples=args.max_eval_samples,
       seed=args.seed,
   )
   systems = build_system_specs(args)
   majority_label = infer_majority_label(all_rows, split_obj)
   device = "cuda" if torch.cuda.is_available() else "cpu"


   print(f"[TASK] {configured_task} labels={configured_labels}")
   print(f"[DATA] data_path={data_path}")
   print(f"[EVAL] split_key={args.split_key} eval_size={len(eval_rows)}")
   print(f"[EVAL] systems={[s['name'] for s in systems]}")


   leaderboard_rows: List[Dict[str, Any]] = []
   pred_rows_all: List[Dict[str, Any]] = []
   details: Dict[str, Any] = {}


   for system in systems:
       init_eval_state(eval_rows)
       print(f"\n[RUN] system={system['name']} mode={system['mode']}")
       if system["mode"] in ["manager_tools", "manager_no_tools"]:
           metrics, pred_rows = eval_manager_system(
               system=system,
               eval_rows=eval_rows,
               device=device,
               max_new_tokens=args.max_new_tokens,
               temperature=args.temperature,
               max_tool_calls=args.max_tool_calls,
           )
       elif system["mode"] == "direct_model":
           metrics, pred_rows = eval_direct_system(
               system=system,
               eval_rows=eval_rows,
               device=device,
               max_new_tokens=args.max_new_tokens,
               temperature=args.temperature,
           )
       elif system["mode"] == "random":
           metrics, pred_rows = eval_random_system(system=system, eval_rows=eval_rows, seed=args.seed)
       elif system["mode"] == "majority":
           metrics, pred_rows = eval_majority_system(
               system=system,
               eval_rows=eval_rows,
               majority_label=majority_label,
           )
       else:
           raise ValueError(f"Unknown system mode: {system['mode']}")


       print(
           f"[RUN] {system['name']} acc={metrics['accuracy']:.4f} "
           f"macro_f1={metrics['macro_f1']:.4f} invalid={metrics['invalid_rate']:.4f} "
           f"avg_tool_calls={metrics.get('avg_tool_calls', 0.0):.3f}"
       )


       pred_rows_all.extend(pred_rows)
       details[system["name"]] = {"system": system, "metrics": metrics}
       leaderboard_rows.append(
           {
               "system": system["name"],
               "mode": system["mode"],
               "n": metrics["n"],
               "accuracy": round(float(metrics["accuracy"]), 6),
               "macro_f1": round(float(metrics["macro_f1"]), 6),
               "weighted_f1": round(float(metrics["weighted_f1"]), 6),
               "invalid_rate": round(float(metrics["invalid_rate"]), 6),
               "avg_tool_calls": round(float(metrics.get("avg_tool_calls", 0.0)), 6),
               "elapsed_sec": round(float(metrics.get("elapsed_sec", 0.0)), 3),
           }
       )


   leaderboard_rows.sort(key=lambda x: (x["accuracy"], x["macro_f1"]), reverse=True)
   for i, row in enumerate(leaderboard_rows, start=1):
       row["rank"] = i


   out_dir = args.out_dir
   os.makedirs(out_dir, exist_ok=True)
   summary_json = os.path.join(out_dir, "summary.json")
   pred_jsonl = os.path.join(out_dir, "predictions.jsonl")
   leaderboard_csv = os.path.join(out_dir, "leaderboard.csv")


   write_json(
       summary_json,
       {
           "task_name": configured_task,
           "labels": configured_labels,
           "data_path": data_path,
           "split_path": args.split_path,
           "split_key": args.split_key,
           "eval_size": len(eval_rows),
           "systems": systems,
           "leaderboard": leaderboard_rows,
           "details": details,
       },
   )
   write_jsonl(pred_jsonl, pred_rows_all)
   write_csv(
       leaderboard_csv,
       leaderboard_rows,
       fieldnames=[
           "rank",
           "system",
           "mode",
           "n",
           "accuracy",
           "macro_f1",
           "weighted_f1",
           "invalid_rate",
           "avg_tool_calls",
           "elapsed_sec",
       ],
   )


   print(f"\n[DONE] summary -> {summary_json}")
   print(f"[DONE] predictions -> {pred_jsonl}")
   print(f"[DONE] leaderboard -> {leaderboard_csv}")
   if leaderboard_rows:
       best = leaderboard_rows[0]
       print(f"[BEST] rank1={best['system']} acc={best['accuracy']:.4f} macro_f1={best['macro_f1']:.4f}")




if __name__ == "__main__":
   main()

