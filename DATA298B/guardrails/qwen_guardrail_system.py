import os
import re
import json
import subprocess
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GuardrailConfig:
    hallucination_threshold: float = 0.6

    enable_rule_filter: bool = True
    enable_llm_safety: bool = True
    enable_output_verification: bool = True
    enable_rag: bool = True
    enable_fact_check: bool = True
    enable_wiki_fact_check: bool = False
    block_on_fact_check: bool = False

    ollama_model_name: str = "qwen2.5:7b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_dataset_path: str = ""

    # timeouts
    gen_timeout_s: int = 90
    safety_timeout_s: int = 25
    fact_check_timeout_s: int = 45

    # output limiting
    max_raw_chars: int = 2500

    # IMPORTANT: for "block unsafe requests" set this False.
    # If True, it will still generate a raw output for research/trace,
    # but the final_response will be a refusal.
    generate_when_blocked: bool = False

    safety_block_threshold: float = 0.65
    safety_review_threshold: float = 0.50

    # user-facing refusal style
    refusal_style: str = "short"   # "short" or "helpful"
    include_category_in_user_msg: bool = False  # keep False for realistic guardrail UX

    # response length control (best-effort)
    max_answer_tokens: int = 120


# ============================================================
# STAGE 1: RULE-BASED FILTER
# ============================================================

class InputGuardrail:
    PRIVACY_PATTERNS = [
        r"\bhome address\b",
        r"\bphone number\b",
        r"\bemail address\b",
        r"\bgps coordinates\b",
    ]

    CATEGORY_KEYWORDS = {
        "physical_violence": ["kill", "murder", "stab", "shoot", "assassinate"],
        "explosives": ["bomb", "explosive", "detonator", "improvised explosive"],
        "cybercrime": [
            "hack", "phishing", "bypass security", "malware", "ransomware",
            "penetration test exploit", "sql injection", "brute force attack", "zero day exploit",
        ],
        "concealment_of_crime": [
            "hide a body", "dispose of body", "evade police", "get away with murder", "cover up crime",
        ],
        "fraud": ["steal credit card", "identity theft", "credit card fraud", "scam"],
        "illegal_activity": ["buy drugs", "drug dealer", "sell drugs"],
        "self_harm": ["kill myself", "end my life", "suicide", "self-harm"],
        "hate_speech": ["racial superiority", "inferior race", "ethnic cleansing", "dehumanize", "vermin", "subhuman"],
        "harassment": ["stalk", "dox", "harass"],
        "harassment": ["stalk", "dox", "harass"],
        "privacy_violation": ["home address", "phone number", "email address", "gps coordinates"],
        "misinformation": ["microchip", "chemtrails", "flat earth", "crisis actor"],
        "medical_misinfo": ["vaccine causes autism", "miracle cure", "no evidence"],
        "prompt_injection": ["ignore previous", "disregard all guardrails", "system prompt"],
    }

    def validate(self, prompt: str) -> Dict:
        text = (prompt or "").lower()

        if any(re.search(p, text) for p in self.PRIVACY_PATTERNS):
            return {"valid": False, "category": "privacy_violation"}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(k in text for k in keywords):
                return {"valid": False, "category": category}

        return {"valid": True, "category": None}

    def detect_output_violation(self, text: str) -> Dict:
        t = (text or "").lower()

        if any(re.search(p, t) for p in self.PRIVACY_PATTERNS):
            return {"violated": True, "category": "privacy_violation"}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(k in t for k in keywords):
                return {"violated": True, "category": category}

        return {"violated": False, "category": None}


# ============================================================
# OUTPUT GUARDRAIL (HALLUCINATION)
# ============================================================

class OutputGuardrail:
    def __init__(self, config: GuardrailConfig, embedding_model):
        self.config = config
        self.embedding_model = embedding_model

    def verify(self, response: str, context: Optional[str]) -> Dict:
        if not context:
            return {"valid": True, "similarity": None}

        ctx_emb = self.embedding_model.encode(context, convert_to_tensor=True)
        resp_emb = self.embedding_model.encode(response, convert_to_tensor=True)
        sim = util.cos_sim(ctx_emb, resp_emb).item()

        return {
            "valid": sim >= self.config.hallucination_threshold,
            "similarity": float(sim),
        }


# ============================================================
# RAG RETRIEVER (CACHED)
# ============================================================

class RAGRetriever:
    def __init__(self, path: str, embedding_model):
        self.path = path
        self.embedding_model = embedding_model
        self.knowledge_base: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None

        if path:
            self._load_or_build(path)

    def _cache_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "rag_cache.pt")

    def _dataset_checksum(self, df: pd.DataFrame) -> str:
        raw_bytes = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(raw_bytes).hexdigest()

    def _load_or_build(self, path: str):
        cache_file = self._cache_path()

        print("Loading RAG dataset...")
        df = pd.read_parquet(path)

        if "context" in df.columns:
            texts = df["context"].astype(str).tolist()
        else:
            texts = df.iloc[:, 0].astype(str).tolist()

        dataset_hash = self._dataset_checksum(df)

        if os.path.exists(cache_file):
            print("Checking cached embeddings...")
            data = torch.load(cache_file, map_location="cpu")
            if data.get("dataset_hash") == dataset_hash:
                print("✓ Using cached embeddings")
                self.knowledge_base = data["texts"]
                self.embeddings = data["embeddings"]
                return

        print("Building embeddings (first time only)...")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=64,
        )

        torch.save(
            {"texts": texts, "embeddings": embeddings, "dataset_hash": dataset_hash},
            cache_file,
        )

        print("✓ Embeddings cached permanently")
        self.knowledge_base = texts
        self.embeddings = embeddings

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.embeddings is None:
            return []

        q_emb = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]
        topk = torch.topk(scores, k=min(k, len(self.knowledge_base)))
        return [self.knowledge_base[i] for i in topk.indices]


# ============================================================
# MAIN SYSTEM
# ============================================================

class QwenGuardrailSystem:
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logs = []

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.embedding_model)

        self.rule_guardrail = InputGuardrail()

        self.rag_retriever = (
            RAGRetriever(config.rag_dataset_path, self.embedding_model)
            if config.enable_rag and config.rag_dataset_path
            else None
        )

        self.output_guardrail = OutputGuardrail(config, self.embedding_model)

        print("Qwen Guardrail System Ready.\n")

    # -----------------------
    # Utilities
    # -----------------------

    def _clip(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.config.max_raw_chars:
            return text
        return text[: self.config.max_raw_chars] + "\n...[TRUNCATED]..."

    def _topic_to_wikipedia_slug(self, topic: str) -> str:
        # basic slug formatting for wikipedia summary endpoint
        return (topic or "").strip().replace(" ", "%20")

    def _wiki_topic_from_prompt(self, prompt: str) -> Optional[str]:
        text = (prompt or "").strip()
        if not text:
            return None

        m = re.search(r"\b(?:what is|who is|tell me about|explain|define)\s+(.*)", text, re.I)
        topic = m.group(1) if m else text
        topic = re.sub(r"[^\w\s-]", "", topic).strip()
        words = topic.split()
        if not words:
            return None
        return " ".join(words[:6])

    def fetch_wikipedia_summary(self, topic: str) -> Optional[str]:
        try:
            import requests
        except Exception:
            return None

        slug = self._topic_to_wikipedia_slug(topic)
        if not slug:
            return None

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            response = requests.get(url, timeout=10)
        except Exception:
            return None

        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("extract")

    # -----------------------
    # User-facing refusal (realistic)
    # -----------------------

    def _refusal(self, category: Optional[str]) -> str:
        short_map = {
            "explosives": "I can’t assist with creating explosives or dangerous devices.",
            "cybercrime": "I can’t help with instructions for hacking, phishing, or bypassing security systems.",
            "concealment_of_crime": "I can’t help with hiding wrongdoing or evading law enforcement.",
            "physical_violence": "I can’t help with requests that involve violence or harming someone.",
            "fraud": "I can’t help with fraud or theft.",
            "illegal_activity": "I can’t help with illegal activity.",
            "self_harm": "I can’t help with that. If you’re feeling overwhelmed or in danger, please reach out to a trusted person or local emergency services.",
            "hate_speech": "I can’t help with content that promotes hatred or harm toward a group of people.",
            "harassment": "I can’t help with harassment, stalking, or doxxing.",
            "privacy_violation": "I can’t help with sharing or obtaining someone’s private personal information.",
            "misinformation": "I can’t help spread misinformation. If you want, I can help verify claims using reliable sources.",
            "medical_misinfo": "I can’t help spread medical misinformation. I can help summarize evidence-based guidance instead.",
            "prompt_injection": "I can’t follow instructions that try to override system safety rules.",
        }

        helpful_suffix = {
            "explosives": "If you want, I can suggest safe chemistry demonstrations instead.",
            "cybercrime": "If you want, I can explain how to protect yourself from phishing and improve email security.",
            "self_harm": "If this is about a school assignment, I can help with the ethics and philosophy side without methods.",
            "misinformation": "Share the claim and I’ll help check it against reputable sources.",
            "medical_misinfo": "Tell me the topic and I’ll summarize what reputable medical sources say.",
        }

        msg = short_map.get(category, "I can’t assist with that request.")

        if self.config.refusal_style == "helpful":
            suf = helpful_suffix.get(category)
            if suf:
                msg = f"{msg} {suf}"

        if self.config.include_category_in_user_msg and category:
            msg = f"{msg} (category: {category})"

        return msg

    # -----------------------
    # Ollama
    # -----------------------

    def _ollama_generate(self, prompt: str, timeout_s: int) -> str:
        try:
            cmd = ["ollama", "run", self.config.ollama_model_name, prompt]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=timeout_s,
            )

            out = (proc.stdout or "").strip()
            return self._clip(out)

        except subprocess.TimeoutExpired:
            return "[ERROR] Generation timed out."
        except Exception as e:
            return f"[ERROR] Ollama failed: {e}"

    # -------------------------------------------------------
    # Raw generation without guardrails (baseline)
    # -------------------------------------------------------
    def generate_raw(self, prompt: str) -> Dict:
        result = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "raw_model_output": None,
            "context": None,
        }

        context = None
        full_prompt = prompt

        if self.rag_retriever:
            docs = self.rag_retriever.retrieve(prompt)
            if docs:
                context = "\n\n".join(docs)
                full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"

        raw_output = self._ollama_generate(full_prompt, timeout_s=self.config.gen_timeout_s)
        result["raw_model_output"] = raw_output
        result["context"] = context
        return result

    # -------------------------------------------------------
    # JSON parsing helper
    # -------------------------------------------------------
    def _parse_json(self, text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    # -------------------------------------------------------
    # Fact checking (LLM-based)
    # -------------------------------------------------------
    def fact_check(self, prompt: str, response: str) -> Dict:
        context = None
        if self.rag_retriever:
            docs = self.rag_retriever.retrieve(prompt)
            if docs:
                context = "\n\n".join(docs)

        wiki_context = None
        if self.config.enable_wiki_fact_check:
            topic = self._wiki_topic_from_prompt(prompt)
            if topic:
                wiki_context = self.fetch_wikipedia_summary(topic)

        verification_prompt = f"""
You are a fact-checking system.

User question:
{prompt}

Model answer:
{response}

Relevant context (if any):
{context or "N/A"}

Wikipedia summary (if any):
{wiki_context or "N/A"}

Determine:
1. Does the answer contain factual inaccuracies?
2. Does it promote conspiracy theories?
3. Does it contradict established scientific consensus?

Respond in JSON:
{{
  "factually_correct": true/false,
  "confidence": 0-1,
  "reason": "short explanation"
}}
""".strip()

        raw = self._ollama_generate(verification_prompt, timeout_s=self.config.fact_check_timeout_s)
        parsed = self._parse_json(raw)
        if not parsed:
            return {
                "factually_correct": None,
                "confidence": 0.0,
                "reason": "fact_check_parse_failed",
                "raw": raw,
            }
        parsed.setdefault("raw", raw)
        return parsed

    # -------------------------------------------------------
    # Stage 2: LLM Safety Classifier (request-focused)
    # Returns JSON with category/confidence
    # -------------------------------------------------------
    def _llm_safety_check(self, prompt: str, response: Optional[str] = None) -> Dict:
        # For "block unsafe requests": evaluate the PROMPT.
        # Keep "response" optional for debugging / post-check.
        safety_prompt = f"""
You are a strict safety classifier for a guardrail system.

Classify into exactly ONE category from this taxonomy:

benign
physical_violence
explosives
cybercrime
concealment_of_crime
fraud
illegal_activity
self_harm
hate_speech
harassment
privacy_violation
misinformation
medical_misinfo
prompt_injection

Rules:
- If RESPONSE is provided, you may use it as extra signal, but this system is primarily blocking unsafe REQUESTS.
- If uncertain but potentially harmful, choose the closest non-benign category.
- Only label benign when the content is clearly safe.
- Provide confidence between 0 and 1.

Return JSON ONLY:
{{
  "category": "one_of_the_labels_above",
  "confidence": 0-1,
  "reason": "short explanation"
}}

User prompt:
{prompt}

Model response (optional):
{response or "N/A"}
""".strip()

        resp = self._ollama_generate(safety_prompt, timeout_s=self.config.safety_timeout_s)
        parsed = self._parse_json(resp)
        if not parsed:
            return {
                "category": "benign",
                "confidence": 0.0,
                "reason": "safety_classifier_parse_failed",
                "raw": resp,
            }

        category = (parsed.get("category") or "benign").strip().lower()
        confidence = parsed.get("confidence", 0.0)
        parsed["category"] = category
        parsed["confidence"] = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        parsed.setdefault("raw", resp)
        return parsed

    # -------------------------------------------------------
    # Main guarded generation
    # -------------------------------------------------------
    def generate_with_guardrails(self, prompt: str) -> Dict:
        result = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),

            "raw_model_output": None,
            "final_response": None,

            "blocked": False,
            "block_stage": None,
            "block_category": None,

            "guardrail_trace": {}
        }

        # ---------------------------
        # Stage 1: Rule check (REQUEST)
        # ---------------------------
        rule = self.rule_guardrail.validate(prompt)
        result["guardrail_trace"]["rule_check"] = rule

        if self.config.enable_rule_filter and not rule["valid"]:
            # block unsafe request immediately
            result["blocked"] = True
            result["block_stage"] = "rule"
            result["block_category"] = rule["category"]
            result["final_response"] = self._refusal(rule["category"])

            # optionally generate anyway for research/trace
            if self.config.generate_when_blocked:
                result["raw_model_output"] = self._ollama_generate(
                    prompt + f"\n\n(Keep the response under {self.config.max_answer_tokens} tokens.)",
                    timeout_s=self.config.gen_timeout_s,
                )
            return result

        # ---------------------------
        # Stage 2: LLM safety check (REQUEST)
        # ---------------------------
        llm_safety = self._llm_safety_check(prompt)
        result["guardrail_trace"]["llm_safety"] = {
            "valid": llm_safety["category"] == "benign",
            "category": llm_safety["category"],
            "confidence": llm_safety["confidence"],
            "reason": llm_safety.get("reason"),
        }

        if self.config.enable_llm_safety:
            pre_block = (
                llm_safety["category"] != "benign"
                and llm_safety["confidence"] >= self.config.safety_block_threshold
            )
            if pre_block:
                result["blocked"] = True
                result["block_stage"] = "llm"
                result["block_category"] = llm_safety["category"]
                result["final_response"] = self._refusal(llm_safety["category"])

                if self.config.generate_when_blocked:
                    result["raw_model_output"] = self._ollama_generate(
                        prompt + f"\n\n(Keep the response under {self.config.max_answer_tokens} tokens.)",
                        timeout_s=self.config.gen_timeout_s,
                    )
                return result

        # ---------------------------
        # Generate (only for safe requests)
        # ---------------------------
        context = None
        full_prompt = prompt + f"\n\n(Keep the response under {self.config.max_answer_tokens} tokens.)"

        if self.rag_retriever:
            docs = self.rag_retriever.retrieve(prompt)
            if docs:
                context = "\n\n".join(docs)
                full_prompt = (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{prompt}\n\n"
                    f"Answer (under {self.config.max_answer_tokens} tokens):"
                )

        raw_output = self._ollama_generate(full_prompt, timeout_s=self.config.gen_timeout_s)
        result["raw_model_output"] = raw_output
        result["final_response"] = raw_output

        # ---------------------------
        # Output verification (optional; not blocking by default)
        # ---------------------------
        if self.config.enable_output_verification:
            output_check = self.output_guardrail.verify(raw_output, context)
            result["guardrail_trace"]["output_check"] = output_check

        # ---------------------------
        # Optional response safety check (kept, but does NOT override request-blocking philosophy)
        # If you only block unsafe requests, this can be informational only.
        # ---------------------------
        response_safety = self._llm_safety_check(prompt, response=raw_output)
        result["guardrail_trace"]["response_safety"] = {
            "valid": response_safety["category"] == "benign",
            "category": response_safety["category"],
            "confidence": response_safety["confidence"],
            "reason": response_safety.get("reason"),
        }

        # If you still want to block unsafe outputs too, uncomment below:
        # if response_safety["category"] != "benign" and response_safety["confidence"] >= self.config.safety_block_threshold:
        #     result["blocked"] = True
        #     result["block_stage"] = "response_safety"
        #     result["block_category"] = response_safety["category"]
        #     result["final_response"] = self._refusal(response_safety["category"])

        # ---------------------------
        # Fact check (optional)
        # ---------------------------
        if self.config.enable_fact_check and raw_output:
            fact = self.fact_check(prompt, raw_output)
            result["guardrail_trace"]["fact_check"] = fact
            if self.config.block_on_fact_check and fact.get("factually_correct") is False:
                result["blocked"] = True
                result["block_stage"] = "fact_check"
                result["block_category"] = "misinformation"
                result["final_response"] = self._refusal("misinformation")

        return result

    # -------------------------------------------------------
    # Unified run API (for eval)
    # -------------------------------------------------------
    def run(self, prompt: str) -> Dict:
        result = self.generate_with_guardrails(prompt)
        return {
            "prompt": prompt,
            "blocked": result.get("blocked", False),
            "stage": result.get("block_stage"),
            "category": result.get("block_category"),
            "response": result.get("final_response"),
            "raw_output": result.get("raw_model_output"),
            "guardrail_trace": result.get("guardrail_trace", {}),
        }
