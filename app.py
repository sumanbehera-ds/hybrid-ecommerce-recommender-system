from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from src.models.hybrid_recommender import load_artifacts, hybrid_recommend


app = FastAPI(
    title="E-commerce Recommendation System",
    description="Hybrid recommender using GRU4Rec, item-item CF, and popularity fallback",
    version="1.0.0"
)

artifacts = load_artifacts()


class RecommendationRequest(BaseModel):
    item_sequence: List[int] = Field(..., min_length=1)
    top_n: int = Field(default=10, ge=1, le=50)


@app.get("/")
def home():
    return {"message": "E-commerce Recommendation API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "available_models": ["GRU4Rec", "Item-CF", "Popularity"]
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        recommendations = hybrid_recommend(
            user_sequence=request.item_sequence,
            artifacts=artifacts,
            top_n=request.top_n
        )

        return {
            "input_sequence": request.item_sequence,
            "top_n": request.top_n,
            "recommendations": recommendations,
            "model_used": "hybrid"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/popular")
def recommend_popular(top_n: int = 10):
    top_n = min(max(top_n, 1), 50)

    popularity_model = artifacts["popularity_model"]
    recommendations = popularity_model.head(top_n).index.tolist()

    return {
        "top_n": top_n,
        "recommendations": recommendations,
        "model_used": "popularity"
    }