import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from .llm_processor import preprocess_query

load_dotenv()
client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

class SHLRecommender:
    def __init__(self, data_path='data/processed_shl_data.pkl'):
        self.data = pd.read_pickle(data_path)

    def recommend(self, query, max_results=10, duration_filter=None):
        parsed = preprocess_query(query)
        rewritten_query = parsed['query']
        query_embedding = get_embedding(rewritten_query)
        embeddings = np.stack(self.data['embedding'].values)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        recommendations = self.data.copy()
        recommendations['similarity'] = similarities

        if duration_filter is None and parsed.get('duration_minutes'):
            duration_filter = parsed['duration_minutes']

        recommendations['duration_penalty'] = recommendations['duration_minutes'].apply(
            lambda x: -0.2 if pd.notna(x) and duration_filter is not None and x > duration_filter else 0
        )

        recommendations['remote_boost'] = recommendations['remote'].apply(
            lambda x: 0.1 if parsed['remote'] == 'Yes' and x == 'Yes' else 0
        )

        recommendations['adaptive_boost'] = recommendations['adaptive'].apply(
            lambda x: 0.1 if parsed['adaptive'] == 'Yes' and x == 'Yes' else 0
        )

        if parsed.get('test_type'):
            recommendations['test_type_match'] = recommendations['test_type'].apply(
                lambda t: 0.1 if parsed['test_type'].lower() in [x.lower() for x in t] else 0
            )
        else:
            recommendations['test_type_match'] = 0

        if parsed.get('job_level'):
            user_level = parsed['job_level'].strip().lower()
            recommendations['job_level_match'] = recommendations['job_levels'].apply(
                lambda jl: 0.1 if isinstance(jl, str) and user_level in jl.lower() else 0
            )
        else:
            recommendations['job_level_match'] = 0

        recommendations['score'] = (
            recommendations['similarity'] +
            recommendations['duration_penalty'] +
            recommendations['remote_boost'] +
            recommendations['adaptive_boost'] +
            recommendations['test_type_match'] +
            recommendations['job_level_match']
        )

        recommendations = recommendations.sort_values('score', ascending=False)
        recommendations = recommendations.drop_duplicates(subset='name')

        if len(recommendations) < max_results:
            print("🔁 Relaxing filters due to low recommendations")
            recommendations = self.data.copy()
            embeddings = np.stack(recommendations['embedding'].values)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            recommendations['similarity'] = similarities
            recommendations['score'] = recommendations['similarity']
            recommendations = recommendations.sort_values('score', ascending=False)
            recommendations = recommendations.drop_duplicates(subset='name')

        return recommendations.head(max_results)

    def format_recommendations(self, recommendations):
        return recommendations[['name', 'url', 'remote', 'adaptive', 'duration_minutes', 'test_type']]\
            .replace([np.inf, -np.inf], np.nan).fillna("").to_dict('records')

def get_top_k_recommendations(query, top_k=3):
    recommender = SHLRecommender()
    recommendations = recommender.recommend(query, max_results=top_k)
    return recommendations['name'].tolist() if isinstance(recommendations, pd.DataFrame) else []

if __name__ == "__main__":
    query = "Looking for a cognitive test for entry-level candidates under 30 minutes."
    recommender = SHLRecommender()
    recs = recommender.recommend(query, max_results=5)
    print(recommender.format_recommendations(recs))
