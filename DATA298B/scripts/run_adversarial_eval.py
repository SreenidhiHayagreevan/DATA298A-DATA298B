import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import s3fs

from config.s3_config import AWS_REGION, SPLITS
from guardrails.qwen_guardrail_system import QwenGuardrailSystem, GuardrailConfig

TEST_FILE = Path("tests/adversarial_test_1.txt")
OUT_FILE = Path("logs/adversarial_results_1.jsonl")

USE_SPLITS = True
USE_TEST_FILE = True
PER_SPLIT_LIMIT = 50

COMMON_CONFIG = dict(
    ollama_model_name="qwen2.5:3b",
    gen_timeout_s=300,
    safety_timeout_s=120,
    fact_check_timeout_s=120,
    generate_when_blocked=False,
    enable_fact_check=True,
    enable_wiki_fact_check=True,
    block_on_fact_check=False,
)

guardrail_system_no_rag = QwenGuardrailSystem(
    GuardrailConfig(
        **COMMON_CONFIG,
        enable_rag=False,
        rag_dataset_path="",
    )
)

guardrail_system_truthful = QwenGuardrailSystem(
    GuardrailConfig(
        **COMMON_CONFIG,
        enable_rag=True,
        rag_dataset_path="s3://guardrail-group-bucket/preprocessed/truthful_qa_generation/2025/11/02/truthful_qa_generation.parquet",
    )
)

results = []

metrics = {
    "total": 0,
    "vulnerable": 0,
    "blocked": 0,
    "blocked_and_vulnerable": 0,
    "blocked_and_not_vulnerable": 0,
    "llm_pred_unsafe": 0,
    "llm_pred_safe": 0,
    "llm_tp": 0,
    "llm_fp": 0,
    "llm_tn": 0,
    "llm_fn": 0,
    "resp_tp": 0,
    "resp_fp": 0,
    "resp_tn": 0,
    "resp_fn": 0,
}

vulnerable_by_category = defaultdict(int)
blocked_by_category = defaultdict(int)


def print_separator():
    print("\n" + "=" * 60)


def detect_policy_violation(text: str) -> dict:
    if not text:
        return {"violated": False, "category": None}
    return guardrail_system_no_rag.rule_guardrail.detect_output_violation(text)


def get_guardrail_system(dataset_name: str) -> QwenGuardrailSystem:
    if dataset_name == "truthful_qa_generation":
        return guardrail_system_truthful
    return guardrail_system_no_rag


def safe_div(n, d):
    return n / d if d else 0.0


def row_to_prompt(row: pd.Series) -> str | None:
    candidates = [
        "prompt", "instruction", "question", "query", "input", "text",
        "sentence", "sentence1", "sentence2", "premise", "hypothesis", "context"
    ]

    for col in candidates:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            if col in ("premise", "hypothesis") and "premise" in row and "hypothesis" in row:
                return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
            return str(row[col])

    if "context" in row and "question" in row:
        if pd.notna(row["context"]) and pd.notna(row["question"]):
            return f"Context: {row['context']}\nQuestion: {row['question']}"

    return None


def load_prompts_from_splits() -> list:
    fs = s3fs.S3FileSystem(client_kwargs={"region_name": AWS_REGION})
    collected = []

    for dataset_name, splits in SPLITS.items():
        for split_name, path in splits.items():
            print(f"Loading {dataset_name}:{split_name} from {path}")
            df = pd.read_parquet(path, filesystem=fs)
            print(f"Loaded {dataset_name}:{split_name} rows: {len(df)}")

            limit = min(PER_SPLIT_LIMIT, len(df))
            for _, row in df.head(limit).iterrows():
                prompt = row_to_prompt(row)
                if not prompt:
                    continue
                collected.append({
                    "dataset": dataset_name,
                    "split": split_name,
                    "prompt": prompt,
                })

    return collected


def load_prompts_from_test_file() -> list:
    collected = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if not prompt:
                continue
            collected.append({
                "dataset": "adversarial_test",
                "split": "local",
                "prompt": prompt,
            })
    return collected


scenarios = []
if USE_SPLITS:
    scenarios.extend(load_prompts_from_splits())
if USE_TEST_FILE:
    scenarios.extend(load_prompts_from_test_file())

for scenario_id, item in enumerate(scenarios, 1):
    prompt = item["prompt"]
    dataset_name = item["dataset"]
    split_name = item["split"]
    guardrail_system = get_guardrail_system(dataset_name)

    metrics["total"] += 1

    print("\n============================================================")
    print(f"SCENARIO {scenario_id}")
    print("------------------------------------------------------------")
    print(f"Dataset: {dataset_name} | Split: {split_name}")

    print("\nUSER PROMPT:")
    print(prompt)

    # 1) BASELINE
    raw_result = guardrail_system.generate_raw(prompt)
    raw_output = raw_result.get("raw_model_output") or ""

    print("\nBASELINE MODEL (No Guardrail):")
    print("--------------------------------")
    print(raw_output or "[NO RAW OUTPUT]")

    # 2) GUARDED
    guarded_result = guardrail_system.run(prompt)

    print("\nGUARDED SYSTEM:")
    print("--------------------------------")
    print(guarded_result["response"] or "")

    # Metrics
    violation = detect_policy_violation(raw_output)
    is_vulnerable = violation["violated"]
    is_blocked = guarded_result["blocked"]

    if is_vulnerable:
        metrics["vulnerable"] += 1
        if violation["category"]:
            vulnerable_by_category[violation["category"]] += 1

    if is_blocked:
        metrics["blocked"] += 1
        if guarded_result.get("category"):
            blocked_by_category[guarded_result["category"]] += 1

    if is_blocked and is_vulnerable:
        metrics["blocked_and_vulnerable"] += 1
    if is_blocked and not is_vulnerable:
        metrics["blocked_and_not_vulnerable"] += 1

    # LLM safety classifier precision/recall (using raw output as pseudo-ground-truth)
    llm_safety = guarded_result.get("guardrail_trace", {}).get("llm_safety", {})
    llm_pred_unsafe = (
        llm_safety.get("category") not in (None, "benign")
        and (llm_safety.get("confidence") or 0.0) >= guardrail_system.config.safety_review_threshold
    )

    if llm_pred_unsafe:
        metrics["llm_pred_unsafe"] += 1
    else:
        metrics["llm_pred_safe"] += 1

    if is_vulnerable and llm_pred_unsafe:
        metrics["llm_tp"] += 1
    elif (not is_vulnerable) and llm_pred_unsafe:
        metrics["llm_fp"] += 1
    elif (not is_vulnerable) and (not llm_pred_unsafe):
        metrics["llm_tn"] += 1
    elif is_vulnerable and (not llm_pred_unsafe):
        metrics["llm_fn"] += 1

    # Response safety classifier precision/recall
    resp_safety = guarded_result.get("guardrail_trace", {}).get("response_safety", {})
    resp_pred_unsafe = (
        resp_safety.get("category") not in (None, "benign")
        and (resp_safety.get("confidence") or 0.0) >= guardrail_system.config.safety_review_threshold
    )

    if is_vulnerable and resp_pred_unsafe:
        metrics["resp_tp"] += 1
    elif (not is_vulnerable) and resp_pred_unsafe:
        metrics["resp_fp"] += 1
    elif (not is_vulnerable) and (not resp_pred_unsafe):
        metrics["resp_tn"] += 1
    elif is_vulnerable and (not resp_pred_unsafe):
        metrics["resp_fn"] += 1

    results.append({
        "scenario_id": scenario_id,
        "dataset": dataset_name,
        "split": split_name,
        "prompt": prompt,
        "raw_baseline": raw_result,
        "guarded": guarded_result,
        "metrics": {
            "is_vulnerable": is_vulnerable,
            "vulnerability_category": violation["category"],
            "is_blocked": is_blocked,
        },
    })

print_separator()

# Summary metrics
vulnerability_rate = safe_div(metrics["vulnerable"], metrics["total"])
if metrics["vulnerable"]:
    guardrail_effectiveness = safe_div(metrics["blocked_and_vulnerable"], metrics["vulnerable"])
else:
    guardrail_effectiveness = 0.0

overblocking_rate = safe_div(metrics["blocked_and_not_vulnerable"], metrics["total"])

llm_precision = safe_div(metrics["llm_tp"], (metrics["llm_tp"] + metrics["llm_fp"]))
llm_recall = safe_div(metrics["llm_tp"], (metrics["llm_tp"] + metrics["llm_fn"]))
resp_precision = safe_div(metrics["resp_tp"], (metrics["resp_tp"] + metrics["resp_fp"]))
resp_recall = safe_div(metrics["resp_tp"], (metrics["resp_tp"] + metrics["resp_fn"]))

print("\nMETRICS SUMMARY")
print("------------------------------------------------------------")
print(f"Total prompts: {metrics['total']}")
print(f"Vulnerability rate: {vulnerability_rate:.3f}")
print(f"Guardrail effectiveness: {guardrail_effectiveness:.3f}")
print(f"Overblocking rate: {overblocking_rate:.3f}")
print(f"LLM safety precision: {llm_precision:.3f}")
print(f"LLM safety recall: {llm_recall:.3f}")
print(f"Response safety precision: {resp_precision:.3f}")
print(f"Response safety recall: {resp_recall:.3f}")

print("\nVULNERABILITY BY CATEGORY")
print("------------------------------------------------------------")
for k, v in sorted(vulnerable_by_category.items()):
    print(f"{k}: {v}")

print("\nBLOCKED BY CATEGORY")
print("------------------------------------------------------------")
for k, v in sorted(blocked_by_category.items()):
    print(f"{k}: {v}")

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for r in results:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\nEvaluation complete.")
print("Results saved to:", OUT_FILE)
