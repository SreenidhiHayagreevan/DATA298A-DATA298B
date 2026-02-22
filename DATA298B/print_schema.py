import io
import boto3
import pyarrow.parquet as pq

S3_BUCKET="guardrail-group-bucket"

files = [
    "prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet",
    "prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet"
]

s3=boto3.client("s3")

for key in files:
    print(f"\n===== {key.split('/')[-1]} =====")
    data=s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    schema=pq.ParquetFile(io.BytesIO(data)).schema_arrow
    for f in schema:
        print(f"{f.name} : {f.type}")