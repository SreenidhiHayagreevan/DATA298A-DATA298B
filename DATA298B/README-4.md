# Qwen Guardrail System

## Overview

Qwen Guardrail System is a modular, multi-stage safety framework built
around the Qwen model (via Ollama) to enable controlled, safe, and
verifiable LLM generation.

The system integrates:

-   Rule-based input filtering
-   LLM-based safety classification
-   Retrieval-Augmented Generation (RAG)
-   Embedding-based hallucination detection
-   Optional fact-checking (including Wikipedia-based validation)
-   Structured evaluation pipelines (F1 scoring + adversarial testing)
-   Optional S3 dataset integration

This project is designed for LLM safety research, guardrail
experimentation, and evaluation benchmarking.

------------------------------------------------------------------------

## Architecture

The guardrail pipeline follows a layered safety architecture:

User Input\
↓\
1. Rule-Based Filter\
↓\
2. LLM Safety Classification\
↓\
3. Optional RAG Context Injection\
↓\
4. Controlled Qwen Generation (Ollama)\
↓\
5. Output Verification (Embedding Similarity)\
↓\
6. Optional Fact Check\
↓\
Final Decision (Allow / Refuse / Review)

Core implementation: `guardrails/qwen_guardrail_system.py`

------------------------------------------------------------------------

## Project Structure

    qwen_guardrail_project/
    │
    ├── guardrails/
    │   ├── qwen_guardrail_system.py
    │   ├── rag_cache.pt
    │
    ├── config/
    │   ├── qwen_config.txt
    │   ├── dataset_paths.py
    │   ├── s3_config.py
    │
    ├── scripts/
    │   ├── run_f1_eval.py
    │   ├── run_adversarial_eval.py
    │   ├── summarize_eval.py
    │   ├── test_qwen_connection.py
    │   ├── test_s3_connection.py
    │   ├── test_all_datasets.py
    │   └── test_qwen_full_pipeline.py
    │
    ├── tests/
    │   ├── adversarial_test_1.txt
    │   └── print_all_schemas.py
    │
    ├── logs/
    │   ├── f1_eval_*.jsonl
    │   ├── adversarial_results_*.jsonl
    │
    ├── make_mixed_eval.py
    ├── print_schema.py
    └── mixed_eval_prompts.csv

------------------------------------------------------------------------

## Installation

### 1. Install Python Dependencies

    pip install torch pandas sentence-transformers

### 2. Install Ollama

Download from: https://ollama.com

Pull Qwen model:

    ollama pull qwen2.5:7b

------------------------------------------------------------------------

## Configuration

Primary configuration is located in:

`guardrails/qwen_guardrail_system.py`

Key configurable parameters include:

-   Safety thresholds
-   Hallucination similarity threshold
-   RAG enable/disable
-   Fact-check enable/disable
-   Model name
-   Token limits
-   Timeout settings

Additional configuration files are available in:

`config/`

------------------------------------------------------------------------

## Running the System

### Test Qwen Connection

    python scripts/test_qwen_connection.py

### Run Full Guardrail Pipeline

    python scripts/test_qwen_full_pipeline.py

### Run F1 Evaluation

    python scripts/run_f1_eval.py

### Run Adversarial Evaluation

    python scripts/run_adversarial_eval.py

### Summarize Results

    python scripts/summarize_eval.py

------------------------------------------------------------------------

## Evaluation Framework

The system supports:

-   F1 scoring on labeled datasets
-   Adversarial prompt testing
-   Mixed evaluation prompts
-   Structured JSONL logging

Each evaluation run logs:

-   Input prompt
-   Safety classification decision
-   Model output
-   Verification metrics
-   Final guardrail action

------------------------------------------------------------------------

## S3 Integration

Configured in:

`config/s3_config.py`

Allows:

-   Pulling evaluation datasets from S3
-   Running guardrail pipeline at scale

------------------------------------------------------------------------

## Research Use Cases

This project is suitable for:

-   LLM safety research
-   Guardrail benchmarking
-   Hallucination detection experiments
-   RAG reliability studies
-   Adversarial robustness testing
-   Policy enforcement prototyping

------------------------------------------------------------------------

## Design Principles

-   Modular safety layers
-   Configurable thresholds
-   Toggle-based experimentation
-   Research-grade logging
-   Reproducibility

Each safety stage can be independently enabled or disabled for ablation
studies.

------------------------------------------------------------------------

