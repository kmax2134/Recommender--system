import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_json(text):
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
    return None

def preprocess_query(user_input):
    prompt = f"""
You are an AI assistant for SHL assessment matching.

From the user input below, extract:
- a clean, rewritten query
- estimated duration in minutes (if provided)
- whether remote testing is preferred (Yes/No/Unknown)
- whether adaptive testing is preferred (Yes/No/Unknown)
- inferred test type (cognitive, behavioral, etc.)
- job level from this list: Director, Entry-Level, Executive, Front Line Manager, General Population, Graduate, Manager, Mid-Professional, Professional Individual Contributor, Supervisor

Respond ONLY in this strict JSON format:

{{
  "query": "<rewritten query>",
  "duration_minutes": <int or null>,
  "remote": "Yes"|"No"|"Unknown",
  "adaptive": "Yes"|"No"|"Unknown",
  "test_type": "<string or null>",
  "job_level": "<string or null>"
}}

User input:
{user_input}
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        print("🧠 Model Output:", content)
        structured = extract_json(content)
        if structured:
            return structured
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

    # fallback if extraction fails
    return {
        "query": user_input,
        "duration_minutes": None,
        "remote": "Unknown",
        "adaptive": "Unknown",
        "test_type": None,
        "job_level": None
    }

# ✅ Quick test
if __name__ == "__main__":
    test_input = "Need a short adaptive test for entry-level remote software engineers."
    result = preprocess_query(test_input)
    print("✅ Final JSON Output:\n", result)
