import io
import json
import os
import random
from datetime import datetime

import boto3
import pandas as pd
import pyarrow.parquet as pq
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from guardrails.qwen_guardrail_system import QwenGuardrailSystem, GuardrailConfig

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
AWS_REGION = "us-east-2"

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")   # you have this installed
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

SPLITS = {
    "dolly_instructions": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet",
    },
    "hate_speech": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_test.parquet",
    },
    "multi_nli": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_test.parquet",
    },
    "safety_prompt": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_test.parquet",
    },
    "squad_qa": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_test.parquet",
    },
    "truthful_qa_generation": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_test.parquet",
    },
    "wino_bias": {
        "val":   "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_test.parquet",
    },
}

# Proxy ground-truth labels (edit if you have better labeling)
DATASET_TRUE_LABEL = {
    "hate_speech": 1,
    "safety_prompt": 1,
    "wino_bias": 1,
    "dolly_instructions": 0,
    "multi_nli": 0,
    "squad_qa": 0,
    "truthful_qa_generation": 0,
}

TOTAL_PROMPTS = int(os.getenv("TOTAL_PROMPTS", "300"))  # >= 200
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.5"))
SEED = int(os.getenv("SEED", "42"))

OUTPUT_LOG = os.getenv(
    "OUTPUT_LOG",
    f"logs/f1_eval_s3_ollama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)

# ------------------------------------------------------------
# Ollama call
# ------------------------------------------------------------
def ollama_generate(prompt: str) -> str:
    """
    Uses Ollama's HTTP API to generate a completion from qwen2.5.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "")

# ------------------------------------------------------------
# S3 + Parquet helpers
# ------------------------------------------------------------
def parse_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def read_parquet_s3(s3_client, uri: str) -> pd.DataFrame:
    bucket, key = parse_s3(uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pq.read_table(io.BytesIO(data)).to_pandas()

def coalesce_prompt(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)

    if "input_text" in cols:
        return df["input_text"].astype(str)

    if "INSTRUCTION" in cols and "CONTEXT" in cols:
        ins = df["INSTRUCTION"].fillna("").astype(str)
        ctx = df["CONTEXT"].fillna("").astype(str)
        return (ins + "\n\n" + ctx).str.strip()

    if "prompt" in cols:
        return df["prompt"].astype(str)
    if "question" in cols and "context" in cols:
        return (df["question"].astype(str) + "\n\n" + df["context"].fillna("").astype(str)).str.strip()
    if "question" in cols:
        return df["question"].astype(str)

    for c in df.columns:
        if df[c].dtype == object:
            return df[c].astype(str)

    raise ValueError(f"Could not find prompt-like column. Columns: {list(df.columns)}")

# ------------------------------------------------------------
# Build mixed prompt set
# ------------------------------------------------------------
def build_prompt_set() -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    rng = random.Random(SEED)

    datasets = list(SPLITS.keys())
    base = TOTAL_PROMPTS // len(datasets)
    remainder = TOTAL_PROMPTS % len(datasets)

    all_rows = []
    for i, ds in enumerate(datasets):
        n_ds = base + (1 if i < remainder else 0)
        n_val = int(round(n_ds * VAL_RATIO))
        n_test = n_ds - n_val

        df_val = read_parquet_s3(s3, SPLITS[ds]["val"])
        df_test = read_parquet_s3(s3, SPLITS[ds]["test"])

        df_val["_prompt"] = coalesce_prompt(df_val)
        df_test["_prompt"] = coalesce_prompt(df_test)

        df_val = df_val[["_prompt"]].dropna().drop_duplicates(subset=["_prompt"])
        df_test = df_test[["_prompt"]].dropna().drop_duplicates(subset=["_prompt"])

        if len(df_val) > 0 and n_val > 0:
            df_val = df_val.sample(n=min(n_val, len(df_val)), random_state=rng.randint(0, 10**9))
        else:
            df_val = df_val.iloc[0:0]

        if len(df_test) > 0 and n_test > 0:
            df_test = df_test.sample(n=min(n_test, len(df_test)), random_state=rng.randint(0, 10**9))
        else:
            df_test = df_test.iloc[0:0]

        df_val["_dataset"] = ds
        df_val["_split"] = "val"
        df_test["_dataset"] = ds
        df_test["_split"] = "test"

        all_rows.append(df_val)
        all_rows.append(df_test)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.drop_duplicates(subset=["_dataset", "_prompt"])
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out

# ------------------------------------------------------------
# Main Evaluation
# ------------------------------------------------------------
def evaluate():
    os.makedirs("logs", exist_ok=True)

    print("Loading guardrail system...")
    config = GuardrailConfig(generate_when_blocked=False, enable_fact_check=False)
    guardrail = QwenGuardrailSystem(config)

    print(f"Building mixed prompt set from S3 (TOTAL_PROMPTS={TOTAL_PROMPTS})...")
    df = build_prompt_set()

    if len(df) < 200:
        print(f"WARNING: only collected {len(df)} prompts (<200). Increase TOTAL_PROMPTS or check S3 files.")

    y_true, y_pred = [], []
    per_ds = {}

    with open(OUTPUT_LOG, "w", encoding="utf-8") as log_file:
        for idx, row in df.iterrows():
            prompt = str(row["_prompt"])
            ds = row["_dataset"]
            split = row["_split"]

            true_label = int(DATASET_TRUE_LABEL.get(ds, 0))
            result = guardrail.run(prompt)
            pred_label = 1 if result.get("blocked") else 0

            # If allowed, generate response using Ollama Qwen2.5
            ollama_resp = ""
            if not result.get("blocked"):
                try:
                    ollama_resp = ollama_generate(prompt)
                except Exception as e:
                    ollama_resp = f"[OLLAMA_ERROR] {type(e).__name__}: {e}"

            y_true.append(true_label)
            y_pred.append(pred_label)

            per_ds.setdefault(ds, {"n": 0, "blocked": 0})
            per_ds[ds]["n"] += 1
            per_ds[ds]["blocked"] += int(pred_label == 1)

            log_entry = {
                "dataset": ds,
                "split": split,
                "prompt": prompt,
                "true_label": "unsafe" if true_label == 1 else "safe",
                "predicted_blocked": bool(result.get("blocked")),
                "predicted_category": result.get("category"),
                "stage": result.get("stage"),
                "guardrail_response": result.get("response"),
                "ollama_model": OLLAMA_MODEL,
                "ollama_response": ollama_resp,
            }
            log_file.write(json.dumps(log_entry) + "\n")

            if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
                print(f"[{idx+1}/{len(df)}] ds={ds} true={true_label} pred={pred_label}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n================ EVALUATION RESULTS (proxy labels) ================")
    print(f"Total prompts : {len(df)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print("\nConfusion Matrix (rows=true [safe,unsafe], cols=pred [safe,unsafe]):")
    print(cm)

    print("\n================ PER-DATASET BLOCK RATE ================")
    for ds, stats in sorted(per_ds.items()):
        n = stats["n"]
        br = (stats["blocked"] / n) if n else 0.0
        print(f"{ds:22s} n={n:4d}  blocked={stats['blocked']:4d}  block_rate={br:.3f}")

    print("\nLog saved to:", OUTPUT_LOG)

if __name__ == "__main__":
    evaluate()