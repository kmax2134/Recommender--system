import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from .llm_processor import preprocess_query
from sentence_transformers import SentenceTransformer

load_dotenv()

class SHLRecommender:
    def __init__(self, data_path='data/processed_shl_data.pkl'):
        self.data = pd.read_pickle(data_path)
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    def recommend(self, query, max_results=10, duration_filter=None):
        parsed = preprocess_query(query)
        rewritten_query = parsed['query']
        query_embedding = self.embedding_model.embed_query(rewritten_query)

        similarities = cosine_similarity(
            [query_embedding],
            np.stack(self.data['embedding'].values)
        )[0]

        recommendations = self.data.copy()
        recommendations['similarity'] = similarities

        if duration_filter is None and parsed.get('duration_minutes'):
            duration_filter = parsed['duration_minutes']

        recommendations['duration_penalty'] = recommendations['duration_minutes'].apply(
            lambda x: -0.2 if pd.notna(x) and x > duration_filter else 0
        ) if duration_filter is not None else 0

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
            user_levels = [lvl.strip().lower() for lvl in parsed['job_level']]
            recommendations['job_level_match'] = recommendations['job_levels'].apply(
                lambda jl: 0.1 if isinstance(jl, str) and any(level in jl.lower() for level in user_levels) else 0
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
            recommendations['similarity'] = cosine_similarity(
                [query_embedding],
                np.stack(recommendations['embedding'].values)
            )[0]
            recommendations['score'] = recommendations['similarity']
            recommendations = recommendations.sort_values('score', ascending=False)
            recommendations = recommendations.drop_duplicates(subset='name')

        return recommendations.head(max_results)

    def format_recommendations(self, recommendations):
        return recommendations[[
            'name', 'url', 'remote', 'adaptive', 'duration_minutes'
        ]].replace([np.inf, -np.inf], np.nan).fillna("").to_dict('records')


def get_top_k_recommendations(query, top_k=3):
    model = SHLRecommender()
    recommendations = model.recommend(query, max_results=top_k)
    return recommendations['name'].tolist() if isinstance(recommendations, pd.DataFrame) else []
