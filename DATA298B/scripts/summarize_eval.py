import json
from collections import Counter
from pathlib import Path

IN_FILE = Path("logs/adversarial_results_1.jsonl")

blocked = 0
total = 0
timeouts = 0
reasons = Counter()
llm_cats = Counter()
rule_cats = Counter()
halluc_fail = 0

with open(IN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        total += 1

        if r.get("blocked"):
            blocked += 1
            reasons[r.get("block_reason") or "unknown"] += 1

            if (r.get("block_reason") or "").startswith("rule:"):
                rule_cats[(r.get("block_reason") or "").split("rule:", 1)[-1]] += 1
            if (r.get("block_reason") or "").startswith("llm:"):
                llm_cats[(r.get("block_reason") or "").split("llm:", 1)[-1]] += 1

        raw = r.get("raw_model_output") or ""
        if "TIMED OUT" in raw.upper():
            timeouts += 1

        oc = (r.get("guardrails") or {}).get("output_check") or {}
        if oc.get("valid") is False:
            halluc_fail += 1

print("\n=== SUMMARY ===")
print("Total prompts:", total)
print("Blocked:", blocked, f"({(blocked/total*100 if total else 0):.1f}%)")
print("Timeouts:", timeouts, f"({(timeouts/total*100 if total else 0):.1f}%)")
print("Hallucination check failed (when run):", halluc_fail)

print("\n=== BLOCK REASONS ===")
for k, v in reasons.most_common():
    print(f"{k}: {v}")

print("\n=== RULE CATEGORIES ===")
for k, v in rule_cats.most_common():
    print(f"{k}: {v}")

print("\n=== LLM SAFETY CATEGORIES ===")
for k, v in llm_cats.most_common():
    print(f"{k}: {v}")
