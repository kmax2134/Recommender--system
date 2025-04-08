import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.recommender import get_top_k_recommendations

# Load OpenAI key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------
# 🔍 Ground Truth Dataset for Evaluation
# -----------------------------------------
ground_truth = {
    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.":
        ["Java Developer Test", "Java (Coding)", "Back-End Developer Assessment"],
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.":
        ["Python (Coding)", "SQL (Coding)", "Full Stack Developer Assessment"],
    "Here is a JD text, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes.":
        ["JD Matching Assessment", "Screening Assessment", "Test Authoring Assessment"],
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.":
        ["Cognitive Ability Test", "Personality Profile", "Behavioral Test"]
}


# -----------------------------------------
# 🔎 Embedding + Semantic Similarity
# -----------------------------------------
def get_embedding(text: str, model: str = "text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)

def compute_similarity_score(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return cosine_similarity(emb1, emb2)


# -----------------------------------------
# 🎯 Evaluation Metrics
# -----------------------------------------
def semantic_recall_at_k(actual, predicted, k, threshold=0.6):
    hits = 0
    for a in actual:
        for p in predicted[:k]:
            if compute_similarity_score(a, p) >= threshold:
                hits += 1
                break
    return hits / len(actual) if actual else 0

def semantic_map_at_k(actual, predicted, k, threshold=0.6):
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k]):
        for a in actual:
            if compute_similarity_score(p, a) >= threshold:
                hits += 1
                score += hits / (i + 1)
                break
    return score / min(len(actual), k) if actual else 0


# -----------------------------------------
# 📊 Evaluation Runner
# -----------------------------------------
def evaluate_model(ground_truth_dict, k=3, similarity_threshold=0.6):
    recalls = []
    maps = []
    for query, actual_labels in ground_truth_dict.items():
        predicted = get_top_k_recommendations(query, top_k=k)
        print(f"\nQuery: {query}")
        print(f"Ground Truth: {actual_labels}")
        print(f"Predicted: {predicted}")
        recalls.append(semantic_recall_at_k(actual_labels, predicted, k, threshold=similarity_threshold))
        maps.append(semantic_map_at_k(actual_labels, predicted, k, threshold=similarity_threshold))
    return {
        f"Mean Recall@{k}": round(np.mean(recalls), 4),
        f"MAP@{k}": round(np.mean(maps), 4)
    }


# -----------------------------------------
# 🔧 Manual Run for Debug/Test
# -----------------------------------------
if __name__ == "__main__":
    results = evaluate_model(ground_truth, k=3, similarity_threshold=0.6)
    print("\n📈 Evaluation Results:")
    print(results)
