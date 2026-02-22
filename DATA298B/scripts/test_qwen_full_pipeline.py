from guardrails.qwen_guardrail_system import (
    QwenGuardrailSystem,
    GuardrailConfig
)

config = GuardrailConfig(
    rag_dataset_path="s3://guardrail-group-bucket/preprocessed/squad_qa/2025/11/02/squad_qa.parquet"
)

system = QwenGuardrailSystem(config)

result = system.generate_with_guardrails(
    "What is the oldest university in the world?"
)

print("\nRESPONSE:\n", result["response"])
print("\nGUARDRAILS:\n", result["guardrails"])
