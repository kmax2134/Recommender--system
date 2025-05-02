import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
    return None

def preprocess_query(user_input: str) -> dict:
    prompt = f"""
You are an AI assistant for SHL assessment matching.
From the user input below, extract:
- a clean, rewritten query
- estimated duration in minutes (if provided)
- whether remote testing is preferred (Yes/No/Unknown)
- whether adaptive testing is preferred (Yes/No/Unknown)
- inferred test type (cognitive, behavioral, etc.)
- job level from this list: Director, Entry-Level, Executive, Front Line Manager, General Population, Graduate, Manager, Mid-Professional, Professional Individual Contributor, Supervisor

Respond ONLY in JSON format exactly as follows:
{{
  "query": "<rewritten query>",
  "duration_minutes": <int or null>,
  "remote": "Yes"|"No"|"Unknown",
  "adaptive": "Yes"|"No"|"Unknown",
  "test_type": "<string or null>",
  "job_level": "<string or null>"
}}
User input: {user_input}
"""
    structured = None

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract structured info from job-desc."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        structured = extract_json(content)
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")

    # Fallback to barebones
    if not structured:
        structured = {
            "query": user_input,
            "duration_minutes": None,
            "remote": "Unknown",
            "adaptive": "Unknown",
            "test_type": None,
            "job_level": None
        }

    # —— NEW: simple regex skill extractor
    skills = re.findall(
        r"\b(Python|Java|SQL|JavaScript|C\+\+|C#|Go|Ruby)\b",
        user_input,
        flags=re.IGNORECASE
    )
    structured["skills"] = [s.lower() for s in skills] if skills else []

    return structured

if __name__ == "__main__":
    sample = "looking for python"
    print("→", preprocess_query(sample))
