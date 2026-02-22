import io, boto3, pyarrow.parquet as pq

AWS_REGION="us-east-2"

FILES = [
    ("dolly_val",  "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet"),
    ("dolly_test", "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet"),

    ("hate_val",   "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_val.parquet"),
    ("hate_test",  "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_test.parquet"),

    ("nli_val",    "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_val.parquet"),
    ("nli_test",   "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_test.parquet"),

    ("safety_val", "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_val.parquet"),
    ("safety_test","s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_test.parquet"),

    ("squad_val",  "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_val.parquet"),
    ("squad_test", "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_test.parquet"),

    ("truth_val",  "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_val.parquet"),
    ("truth_test", "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_test.parquet"),

    ("wino_val",   "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_val.parquet"),
    ("wino_test",  "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_test.parquet"),
]

def parse_s3(uri: str):
    rest = uri[5:]
    b, k = rest.split("/", 1)
    return b, k

s3 = boto3.client("s3", region_name=AWS_REGION)

for name, uri in FILES:
    b, k = parse_s3(uri)
    data = s3.get_object(Bucket=b, Key=k)["Body"].read()
    schema = pq.ParquetFile(io.BytesIO(data)).schema_arrow
    print(f"\n===== {name} =====")
    for f in schema:
        print(f"{f.name} : {f.type}")