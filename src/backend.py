# src/backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .recommender import SHLRecommender  # Note the dot for a relative import
import uvicorn

app = FastAPI()
recommender = SHLRecommender()

class RecommendationRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    max_duration: Optional[int] = None

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    recommendations = recommender.recommend(
        request.query,
        max_results=request.max_results,
        duration_filter=request.max_duration
    )
    formatted = recommender.format_recommendations(recommendations)
    return {"recommendations": formatted}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
