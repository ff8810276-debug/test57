from fastapi import FastAPI
from llama_cpp import Llama
import json
import re

app = FastAPI()

# مدل را یکبار لود می‌کنیم (بسیار سریع برای 0.5B)
llm = Llama(
    model_path="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=8,        # اگر CPU بیشتر داری بیشتر کن
    verbose=False
)

def extract_json(txt):
    match = re.search(r"\{.*\}", txt, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

@app.get("/skills")
def get_skills(role: str):
    prompt = f"""
You MUST return ONLY valid JSON.
No explanations. No markdown.

Generate EXACTLY 10 professional skills for the role "{role}".
Return JSON in this format:

{{
  "role": "{role}",
  "skills": [
    "skill1",
    "skill2",
    "skill3",
    "skill4",
    "skill5",
    "skill6",
    "skill7",
    "skill8",
    "skill9",
    "skill10"
  ]
}}
"""

    out = llm(prompt, max_tokens=200)
    text = out["choices"][0]["text"]

    data = extract_json(text)
    if data:
        return data
    return {"error": "Invalid JSON", "raw": text}
