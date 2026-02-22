import pandas as pd

s3_path = "s3://guardrail-group-bucket/preprocessed/squad_qa/2025/11/02/squad_qa.parquet"

df = pd.read_parquet(
    s3_path,
    storage_options={
        "region_name": "us-east-2"
    }
)

print(df.head())
print("Rows:", len(df))
