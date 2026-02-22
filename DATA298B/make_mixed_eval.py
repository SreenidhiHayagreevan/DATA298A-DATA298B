import os, io, random
import boto3
import pandas as pd
import pyarrow.parquet as pq

AWS_REGION = "us-east-2"

SPLITS = {
    "dolly_instructions": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet",
    },
    "hate_speech": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_test.parquet",
    },
    "multi_nli": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_test.parquet",
    },
    "safety_prompt": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_test.parquet",
    },
    "squad_qa": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_test.parquet",
    },
    "truthful_qa_generation": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_test.parquet",
    },
    "wino_bias": {
        "val":  "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_val.parquet",
        "test": "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_test.parquet",
    },
}

def parse_s3(uri: str):
    assert uri.startswith("s3://")
    rest = uri[5:]
    bucket, key = rest.split("/", 1)
    return bucket, key

def read_parquet_s3(s3_client, uri: str) -> pd.DataFrame:
    bucket, key = parse_s3(uri)
    data = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    table = pq.read_table(io.BytesIO(data))
    return table.to_pandas()

def coalesce_prompt(df: pd.DataFrame) -> pd.Series:
    # Your dolly split has INSTRUCTION/CONTEXT and also input_text.
    # For other datasets, we pick best available.
    cols = set(df.columns)
    if "input_text" in cols:
        return df["input_text"].astype(str)
    if "INSTRUCTION" in cols and "CONTEXT" in cols:
        ctx = df["CONTEXT"].fillna("").astype(str)
        ins = df["INSTRUCTION"].fillna("").astype(str)
        return (ins + "\n\n" + ctx).str.strip()
    if "prompt" in cols:
        return df["prompt"].astype(str)
    if "question" in cols and "context" in cols:
        return (df["question"].astype(str) + "\n\n" + df["context"].fillna("").astype(str)).str.strip()
    if "question" in cols:
        return df["question"].astype(str)
    # fallback: first string column
    for c in df.columns:
        if df[c].dtype == object:
            return df[c].astype(str)
    raise ValueError(f"Can't find a prompt column. Columns: {list(df.columns)}")

def sample_split(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).copy()

def main():
    # Controls
    N_PER_DATASET = int(os.getenv("N_PER_DATASET", "200"))     # total per dataset (val+test mixed)
    VAL_RATIO     = float(os.getenv("VAL_RATIO", "0.5"))       # fraction from val
    SEED          = int(os.getenv("SEED", "42"))
    OUT_CSV       = os.getenv("OUT_CSV", "mixed_eval_prompts.csv")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    rng = random.Random(SEED)

    rows = []
    for ds, split_map in SPLITS.items():
        n_val = int(round(N_PER_DATASET * VAL_RATIO))
        n_test = N_PER_DATASET - n_val

        df_val = read_parquet_s3(s3, split_map["val"])
        df_test = read_parquet_s3(s3, split_map["test"])

        # Build prompt text
        df_val["_prompt"] = coalesce_prompt(df_val)
        df_test["_prompt"] = coalesce_prompt(df_test)

        # Sample
        seed_val = rng.randint(0, 10**9)
        seed_test = rng.randint(0, 10**9)
        sv = sample_split(df_val, n_val, seed_val)
        st = sample_split(df_test, n_test, seed_test)

        sv["_dataset"] = ds
        sv["_split"] = "val"
        st["_dataset"] = ds
        st["_split"] = "test"

        keep = ["_dataset", "_split", "_prompt"]
        rows.append(sv[keep])
        rows.append(st[keep])

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["_dataset","_prompt"])
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle

    out.to_csv(OUT_CSV, index=False)
    print(f" Wrote {len(out)} prompts -> {OUT_CSV}")
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()