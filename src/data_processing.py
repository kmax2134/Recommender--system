# data_preprocessing.py
import pandas as pd
import os
import ast
from sentence_transformers import SentenceTransformer

def parse_test_type_column(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.loc[0, 'test_type'], str):
        try:
            df['test_type'] = df['test_type'].apply(ast.literal_eval)
        except Exception:
            print("⚠️ Warning: Could not parse 'test_type' column as list.")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_test_type_column(df)
    def make_combined_text(row):
        parts = [
            f"{row['name']}.",
            f"Test types: {', '.join(row['test_type'])}.",
            f"Remote: {row['remote']}.",
            f"Adaptive: {row['adaptive']}."
        ]
        if pd.notna(row['duration_minutes']):
            parts.insert(2, f"Duration: {row['duration_minutes']} minutes.")
        if pd.notna(row.get('job_levels', None)):
            parts.append(f"Job Levels: {row['job_levels']}.")
        if pd.notna(row.get('description', None)):
            parts.append(f"Description: {row['description']}")
        return " ".join(parts)
    df['combined_text'] = df.apply(make_combined_text, axis=1)
    return df

def main():
    input_path = "data/shl_combined (1).csv"
    output_path = "data/processed_shl_data.pkl"
    print("📥 Reading input CSV...")
    data = pd.read_csv(input_path)
    print("🔧 Preprocessing data...")
    data = preprocess_data(data)
    print("🧠 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data['combined_text'].tolist(), show_progress_bar=True)
    data['embedding'] = list(embeddings)
    print(f"💾 Saving processed data to: {output_path}")
    os.makedirs("data", exist_ok=True)
    data.to_pickle(output_path)
    print("✅ Data processing complete.")

if __name__ == "__main__":
    main()
