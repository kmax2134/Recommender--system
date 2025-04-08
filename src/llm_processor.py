import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
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

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured information from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        print("🧠 Model Output:", content)
        structured = extract_json(content)
        if structured:
            return structured
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")

    # Fallback
    return {
        "query": user_input,
        "duration_minutes": None,
        "remote": "Unknown",
        "adaptive": "Unknown",
        "test_type": None,
        "job_level": None
    }

if __name__ == "__main__":
    test_input = "Need a short adaptive test for entry-level remote software engineers."
    result = preprocess_query(test_input)
    print("✅ Final JSON Output:\n", result)
