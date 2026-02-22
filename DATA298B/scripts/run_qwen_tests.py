import json
from pathlib import Path
from guardrails.qwen_guardrail_system import QwenGuardrailSystem, GuardrailConfig

TEST_FILE = Path("tests/qwen_test_cases.txt")
OUT_FILE = Path("logs/qwen_guardrail_responses.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

config = GuardrailConfig(
    rag_dataset_path="s3://guardrail-group-bucket/preprocessed/squad_qa/2025/11/02/squad_qa.parquet",
    ollama_model_name="qwen2.5:7b",
)

system = QwenGuardrailSystem(config)

with open(TEST_FILE, "r", encoding="utf-8") as f, open(OUT_FILE, "w", encoding="utf-8") as out:
    for line in f:
        prompt = line.strip()
        if not prompt:
            continue
        res = system.generate_with_guardrails(prompt)
        out.write(json.dumps(res, ensure_ascii=False) + "\n")

print("Saved:", OUT_FILE)
