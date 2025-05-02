from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from .recommender import SHLRecommender
import uvicorn

app = FastAPI()
recommender = SHLRecommender()

class RecommendationRequest(BaseModel):
    query: str
    max_results: Optional[int]   = 10
    max_duration: Optional[int]  = None
    job_levels: Optional[List[str]] = None

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    recs = recommender.recommend(
        request.query,
        max_results=request.max_results,
        duration_filter=request.max_duration,
        job_levels=request.job_levels
    )
    return {"recommendations": recommender.format_recommendations(recs)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
