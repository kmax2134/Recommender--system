import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from .llm_processor import preprocess_query

load_dotenv()
client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

class SHLRecommender:
    def __init__(self, data_path='data/processed_shl_data.pkl'):
        self.data = pd.read_pickle(data_path)

    def recommend(self,
                  query: str,
                  max_results: int = 10,
                  duration_filter: int = None,
                  job_levels: list[str] = None):
        # 1) Parse out structured fields + skills
        parsed = preprocess_query(query)
        rewritten = parsed['query']
        skills = parsed.get('skills', [])

        # 2) Embed the rewritten query
        q_emb = get_embedding(rewritten)
        all_embs = np.stack(self.data['embedding'].values)
        sims = cosine_similarity([q_emb], all_embs)[0]

        df = self.data.copy()
        df['similarity'] = sims

        # 3) Skill-match boost: +0.2 per matching skill
        def skill_boost(text: str):
            return 0.2 * sum(bool(re.search(rf"\b{re.escape(s)}\b", text, re.IGNORECASE)) for s in skills)
        df['skill_match'] = df['combined_text'].apply(skill_boost)

        # 4) Duration penalty
        if duration_filter is None and parsed.get('duration_minutes'):
            duration_filter = parsed['duration_minutes']
        df['duration_penalty'] = df['duration_minutes'].apply(
            lambda x: -0.2 if pd.notna(x) and duration_filter and x > duration_filter else 0
        )

        # 5) Remote & adaptive boosts
        df['remote_boost'] = df['remote'].apply(
            lambda x: 0.1 if parsed['remote'] == 'Yes' and x == 'Yes' else 0
        )
        df['adaptive_boost'] = df['adaptive'].apply(
            lambda x: 0.1 if parsed['adaptive'] == 'Yes' and x == 'Yes' else 0
        )

        # 6) Test-type match (unchanged)
        if parsed.get('test_type'):
            target = parsed['test_type'].lower()
            df['test_type_match'] = df['test_type'].apply(
                lambda t: 0.1 if any(target in x.lower() for x in t) else 0
            )
        else:
            df['test_type_match'] = 0

        # 7) Job-level match (front-end override wins)
        if job_levels:
            levels = [jl.lower() for jl in job_levels]
        else:
            levels = [parsed['job_level'].lower()] if parsed.get('job_level') else []
        def jl_boost(jl_field):
            if isinstance(jl_field, str):
                return 0.1 if any(lvl in jl_field.lower() for lvl in levels) else 0
            return 0
        df['job_level_match'] = df['job_levels'].apply(jl_boost)

        # 8) Hybrid scoring
        df['score'] = (
            0.8 * df['similarity']
            + 0.2 * df['skill_match']
            + df['duration_penalty']
            + df['remote_boost']
            + df['adaptive_boost']
            + df['test_type_match']
            + df['job_level_match']
        )

        df = df.sort_values('score', ascending=False).drop_duplicates('name')
        if len(df) < max_results:
            # fallback to pure embeddings if too few
            df['score'] = df['similarity']
            df = df.sort_values('score', ascending=False).drop_duplicates('name')

        return df.head(max_results)

    def format_recommendations(self, df: pd.DataFrame):
        return df[
            ['name', 'url', 'remote', 'adaptive', 'duration_minutes', 'test_type']
        ].fillna('').to_dict('records')

def get_top_k_recommendations(query: str, top_k: int = 3):
    recs = SHLRecommender().recommend(query, max_results=top_k)
    return recs['name'].tolist()
