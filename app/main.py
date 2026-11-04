
import os
import logging
import sys
import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import redis

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
MODEL_CARD_PATH = os.getenv("MODEL_CARD_PATH", "artifacts/model_card.json")

DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

COUNTER_KEY = "prediction_counter"


# --------------------------------------------------
# Logging - stdout only
# --------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Redis Configuration
# --------------------------------------------------

redis_client = None

if REDIS_HOST:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1
    )

    logger.info("Redis configured at %s:%s", REDIS_HOST, REDIS_PORT)
else:
    logger.info("Redis is disabled because REDIS_HOST is not set")


# --------------------------------------------------
# Global Model Variables
# --------------------------------------------------

model = None
model_card = None


# --------------------------------------------------
# Load Model Once at Startup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_card

    logger.info("Starting Breast Cancer Triage API")

    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found: %s", MODEL_PATH)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(MODEL_CARD_PATH):
        logger.error("Model card not found: %s", MODEL_CARD_PATH)
        raise RuntimeError(f"Model card not found: {MODEL_CARD_PATH}")

    try:
        model = joblib.load(MODEL_PATH)

        with open(MODEL_CARD_PATH, "r", encoding="utf-8") as file:
            model_card = json.load(file)

        logger.info("Model loaded successfully")
        logger.info("Model version: %s", model_card.get("model_version", MODEL_VERSION))

    except Exception:
        logger.exception("Failed to load model or model card")
        raise

    yield

    logger.info("Shutting down Breast Cancer Triage API")


app = FastAPI(
    title="Breast Cancer Triage Service",
    description="ML-powered breast cancer prediction API",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# Prediction Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: float = Field(ge=0)
    mean_texture: float = Field(ge=0)
    mean_perimeter: float = Field(ge=0)
    mean_area: float = Field(ge=0)
    mean_smoothness: float = Field(ge=0)
    mean_compactness: float = Field(ge=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    mean_symmetry: float = Field(ge=0)
    mean_fractal_dimension: float = Field(ge=0)

    radius_error: float = Field(ge=0)
    texture_error: float = Field(ge=0)
    perimeter_error: float = Field(ge=0)
    area_error: float = Field(ge=0)
    smoothness_error: float = Field(ge=0)
    compactness_error: float = Field(ge=0)
    concavity_error: float = Field(ge=0)
    concave_points_error: float = Field(ge=0)
    symmetry_error: float = Field(ge=0)
    fractal_dimension_error: float = Field(ge=0)

    worst_radius: float = Field(ge=0)
    worst_texture: float = Field(ge=0)
    worst_perimeter: float = Field(ge=0)
    worst_area: float = Field(ge=0)
    worst_smoothness: float = Field(ge=0)
    worst_compactness: float = Field(ge=0)
    worst_concavity: float = Field(ge=0)
    worst_concave_points: float = Field(ge=0)
    worst_symmetry: float = Field(ge=0)
    worst_fractal_dimension: float = Field(ge=0)


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None or model_card is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        # Convert request fields into the original
        # Scikit-learn feature names and order.
        request_data = request.model_dump()

        feature_names = model_card["feature_names"]

        input_data = {
            feature_name: request_data[feature_name.replace(" ", "_")]
            for feature_name in feature_names
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Get malignant probability.
        malignant_probability = float(
            model.predict_proba(input_df)[0][1]
        )

        # Apply configured decision threshold.
        prediction = (
            "malignant"
            if malignant_probability >= DECISION_THRESHOLD
            else "benign"
        )

        # Increment Redis counter.
        # Redis failure must not stop prediction.
        if redis_client is not None:
            try:
                redis_client.incr(COUNTER_KEY)
            except (redis.exceptions.RedisError, OSError):
                logger.exception(
                    "Redis unavailable. Prediction will continue."
                )

        logger.info("Prediction completed: %s", prediction)

        return {
            "label": prediction,
            "malignant_probability": round(malignant_probability, 4),
            "threshold": DECISION_THRESHOLD,
            "model_version": model_card.get(
                "model_version", MODEL_VERSION
            )
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# --------------------------------------------------
# Model Information Endpoint
# --------------------------------------------------

@app.get("/model/info")
def model_info():

    if model_card is None:
        raise HTTPException(
            status_code=503,
            detail="Model information is not available"
        )

    return {
        "model_version": model_card.get(
            "model_version", MODEL_VERSION
        ),
        "model_family": model_card.get(
            "model_family",
            model_card.get("model_type", "Unknown")
        ),
        "sklearn_version": model_card.get("sklearn_version"),
        "feature_count": len(model_card.get("feature_names", [])),
        "decision_threshold": DECISION_THRESHOLD,
        "git_sha": GIT_SHA,
        "test_metrics": model_card.get("test_metrics", {})
    }


# --------------------------------------------------
# Statistics Endpoint
# --------------------------------------------------

@app.get("/stats")
def get_stats():

    if redis_client is None:
        return {
            "prediction_count": None,
            "redis_enabled": False,
            "redis_available": False
        }

    try:
        count = redis_client.get(COUNTER_KEY)

        return {
            "prediction_count": int(count) if count else 0,
            "redis_enabled": True,
            "redis_available": True
        }

    except (redis.exceptions.RedisError, OSError):
        logger.exception("Could not retrieve Redis statistics")

        return {
            "prediction_count": None,
            "redis_enabled": True,
            "redis_available": False
        }
