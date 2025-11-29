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



# ---------- /courses ----------
@app.get("/courses")
def get_courses(role: str):
    prompt = f"""
Return ONLY valid JSON.

The role is: "{role}"
If the role is NON-TECHNICAL (like accountant, hr, teacher, lawyer, manager, doctor):
- DO NOT generate technical courses like SQL, Backend, Frontend, API, Servers, DevOps.
- Generate ONLY real courses related to this profession.
- Produce practical and educational steps used in real-world training.

Rules:
- MUST generate exactly 10 steps.
- Steps MUST be named Step 1 to Step 10.
- Steps MUST be meaningful and domain-specific.
- JSON ONLY. No explanations.

Correct output format:
{{
  "role": "{role}",
  "courses": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ...",
    "Step 5: ...",
    "Step 6: ...",
    "Step 7: ...",
    "Step 8: ...",
    "Step 9: ...",
    "Step 10: ..."
  ]
}}
"""


    out = llm(prompt, max_tokens=350)
    text = out["choices"][0]["text"]
    data = extract_json(text)
    return data or {"error": "Invalid JSON", "raw": text}


# http://127.0.0.1:8000/skills?role=DevOps  test skill 
# http://127.0.0.1:8000/courses?role=devops
