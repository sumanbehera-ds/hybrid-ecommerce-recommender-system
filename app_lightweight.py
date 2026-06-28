from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

from src.models.lightweight_recommender import (
    hybrid_recommend,
    load_artifacts,
    recommend_popular as get_popular_recommendations,
)


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lightweight E-commerce Recommendation API",
    description="Deployment-optimized recommender using GRU4Rec + popularity fallback",
    version="1.0.0"
)

artifacts = None
artifact_load_error = None


def get_artifacts():
    global artifacts, artifact_load_error

    if artifacts is not None:
        return artifacts

    try:
        artifacts = load_artifacts()
        artifact_load_error = None
        return artifacts
    except Exception as exc:
        artifact_load_error = str(exc)
        logger.exception("Failed to load recommendation artifacts")
        raise


def require_artifacts():
    try:
        return get_artifacts()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Recommendation artifacts are not available: {exc}"
        )


@app.on_event("startup")
def load_models_on_startup():
    try:
        get_artifacts()
    except Exception:
        pass


class RecommendationRequest(BaseModel):
    item_sequence: List[int] = Field(..., min_length=1)
    top_n: int = Field(default=10, ge=1, le=50)


@app.get("/")
def home():
    return {
        "message": "Lightweight E-commerce Recommendation API is running"
    }


@app.get("/health")
def health_check():
    models_loaded = artifacts is not None

    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "available_models": ["GRU4Rec", "PopularityFallback"],
        "deployment_mode": "lightweight",
        "model_load_error": artifact_load_error
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        loaded_artifacts = require_artifacts()
        recommendations = hybrid_recommend(
            user_sequence=request.item_sequence,
            artifacts=loaded_artifacts,
            top_n=request.top_n
        )

        return {
            "input_sequence": request.item_sequence,
            "top_n": request.top_n,
            "recommendations": recommendations,
            "model_used": "GRU4Rec + popularity fallback"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/popular")
def recommend_popular(top_n: int = 10):
    top_n = min(max(top_n, 1), 50)

    loaded_artifacts = require_artifacts()
    popularity_model = loaded_artifacts["popularity_model"]
    recommendations = get_popular_recommendations(popularity_model, top_n)

    return {
        "top_n": top_n,
        "recommendations": recommendations,
        "model_used": "popularity"
    }
