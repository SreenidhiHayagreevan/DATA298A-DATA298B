# config/s3_config.py

AWS_REGION = "us-east-2"

S3_DATASETS = {
    "dolly_instructions": "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_train.parquet",
    "hate_speech": "s3://guardrail-group-bucket/preprocessed/hate_speech/2025/11/02/hate_speech.parquet",
    "multi_nli": "s3://guardrail-group-bucket/preprocessed/multi_nli/2025/11/02/multi_nli.parquet",
    "safety_prompt": "s3://guardrail-group-bucket/preprocessed/safety_prompt/2025/11/02/safety_prompt.parquet",
    "squad_qa": "s3://guardrail-group-bucket/preprocessed/squad_qa/2025/11/02/squad_qa.parquet",
    "truthful_qa_generation": "s3://guardrail-group-bucket/preprocessed/truthful_qa_generation/2025/11/02/truthful_qa_generation.parquet",
    "wino_bias": "s3://guardrail-group-bucket/preprocessed/wino_bias/2025/11/02/wino_bias.parquet",
}

SPLITS = {
    "dolly_instructions": {
        "train": "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet",
    },
    "hate_speech": {
        "train": "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_test.parquet",
    },
    "multi_nli": {
        "train": "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_test.parquet",
    },
    "safety_prompt": {
        "train": "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_test.parquet",
    },
    "squad_qa": {
        "train": "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_test.parquet",
    },
    "truthful_qa_generation": {
        "train": "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_test.parquet",
    },
    "wino_bias": {
        "train": "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_test.parquet",
    },
}
