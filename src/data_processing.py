import pandas as pd
import os
import ast
import openai
from dotenv import load_dotenv
from tqdm import tqdm  # ✅ Progress bar

# Load environment variables
load_dotenv()

# Initialize OpenAI client (SDK v1+)
client = openai.OpenAI()

# ✅ Parse test_type column from string to list if necessary
def parse_test_type_column(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.loc[0, 'test_type'], str):
        try:
            df['test_type'] = df['test_type'].apply(ast.literal_eval)
        except Exception:
            print("⚠️ Warning: Could not parse 'test_type' column as list.")
    return df

# ✅ Combine relevant fields into a single text string for embeddings
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_test_type_column(df)

    def make_combined_text(row):
        parts = [
            f"{row['name']}.",
            f"Test types: {', '.join(row['test_type'])}.",
            f"Remote: {row['remote']}.",
            f"Adaptive: {row['adaptive']}."
        ]
        if pd.notna(row.get('duration_minutes', None)):
            parts.insert(2, f"Duration: {row['duration_minutes']} minutes.")
        if pd.notna(row.get('job_levels', None)):
            parts.append(f"Job Levels: {row['job_levels']}.")
        if pd.notna(row.get('description', None)):
            parts.append(f"Description: {row['description']}")
        return " ".join(parts)

    df['combined_text'] = df.apply(make_combined_text, axis=1)
    return df

# ✅ Generate embedding using OpenAI
def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Failed to embed text: {text[:30]}... – {e}")
        return None

# ✅ Main pipeline
def main():
    input_path = "data/shl_combined (1).csv"
    output_path = "data/processed_shl_data.pkl"

    print("📥 Reading input CSV...")
    data = pd.read_csv(input_path)

    print("🔧 Preprocessing data...")
    data = preprocess_data(data)

    print("🧠 Generating embeddings using OpenAI...")
    tqdm.pandas(desc="Embedding Progress")
    data['embedding'] = data['combined_text'].progress_apply(get_embedding)

    data = data[data['embedding'].notnull()].reset_index(drop=True)

    print(f"💾 Saving processed data to: {output_path}")
    os.makedirs("data", exist_ok=True)
    data.to_pickle(output_path)

    print("✅ Data processing complete.")

# ✅ Entry point
if __name__ == "__main__":
    main()
