# GuardRail for LLMs

## Overview

This project builds a **multi-layered Guardrail System for Large
Language Models (LLMs)**\
to ensure safety, reduce hallucinations, and enforce policy compliance.

Our system is designed as a **three-layer intelligent guardrail
architecture**:

1.  **Gatekeeper Layer** -- Filters unsafe or malicious inputs.
2.  **Knowledge Anchor Layer (RAG)** -- Grounds responses using external
    knowledge sources.
3.  **Parametric Layer** -- Uses aligned LLM reasoning for safe and
    reliable outputs.

------------------------------------------------------------------------

## Architecture Blueprint
![WhatsApp Image 2026-02-17 at 17 11 58](https://github.com/user-attachments/assets/0b4a0672-4a53-4f7d-ae0f-5b54879388ad)


The architecture ensures:

-   Early detection of malicious or biased prompts\
-   Factual grounding using Retrieval-Augmented Generation (RAG)\
-   Output verification and hallucination detection\
-   Safe fallback responses when violations are detected

------------------------------------------------------------------------

## Project Structure

### DATA298A

This phase focused on:

-   Initial experimentation\
-   Data pipeline creation (ETL, Airflow, S3 integration)\
-   Dataset ingestion (Safety, TruthfulQA, Hate Speech, etc.)\
-   Basic guardrail implementation for **Phi-3**\
-   Early safety classification and hallucination experiments

### DATA298B

This phase includes:

-   Higher-level guardrail architecture\
-   Multi-layer implementation (Input + Output guardrails)\
-   Integration of multiple LLMs via **Ollama**
-   RAG-based grounding using Wikipedia knowledge base\
-   Hallucination detection using embedding similarity\
-   Evaluation using F1-score, Precision, Recall, Confusion Matrix

------------------------------------------------------------------------

## Key Features

-   Multi-model guardrail orchestration\
-   Input and Output validation\
-   Real-time safety classification\
-   RAG grounding for factual reliability\
-   Hallucination detection\
-   Local LLM execution using Ollama\
-   Scalable data pipeline with Airflow and S3

------------------------------------------------------------------------

## Goal

To build a **robust, scalable, and research-driven guardrail
framework**\
that enables safe, reliable, and policy-compliant deployment of LLM
systems.

------------------------------------------------------------------------

San Jose State University\
MS Data Analytics -- DATA298A & DATA298B
