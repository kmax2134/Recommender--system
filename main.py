# main.py
from fastapi import FastAPI
from src.recommender import get_top_k_recommendations

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "SHL Assessment Recommender is running!"}

@app.post("/recommend")
def recommend(query: str):
    results = get_top_k_recommendations(query)
    return {"recommendations": results}
