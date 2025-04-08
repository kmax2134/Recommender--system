# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.recommender import SHLRecommender

app = FastAPI()
recommender = SHLRecommender()

class RecommendationRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    max_duration: Optional[int] = None

@app.get("/")
def read_root():
    return {"message": "SHL Assessment Recommender is running!"}

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    results = recommender.recommend(
        request.query,
        max_results=request.max_results,
        duration_filter=request.max_duration
    )
    formatted = recommender.format_recommendations(results)
    return {"recommendations": formatted}
