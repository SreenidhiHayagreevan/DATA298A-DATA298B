import subprocess

prompt = "Say hello in one sentence."

result = subprocess.run(
    ["ollama", "run", "qwen2.5:7b", prompt],
    capture_output=True,
    text=True,
    encoding="utf-8",     # 🔥 force correct encoding
    errors="ignore"       # avoid decode crashes
)

print(result.stdout)
