import json
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request

from finsight.serving.api import (
    ask_endpoint,
    check_rate_limit,
    explain_endpoint,
    forecast_endpoint,
    recommendation_endpoint,
    verify_api_key
)

from finsight.serving.schemas import (
    AskRequest,
    AskResponse,
    ExplainRequest,
    ExplainResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    DriftResponse
)


ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="FinSight AI API",
    description="Adaptive, explainable and multi-asset AI-driven financial planning API",
    version="1.0.0"
)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    elapsed = time.perf_counter() - start

    response.headers["X-Process-Time"] = f"{elapsed:.6f}"

    return response


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "online",
        "service": "FinSight AI",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "service": "FinSight AI",
        "version": "1.0.0"
    }


@app.get(
    "/models",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def models():
    model_dir = ROOT / "models" / "forecasting"

    models = []

    if model_dir.exists():
        for path in model_dir.glob("*/model.keras"):
            models.append(path.parent.name)

    return {
        "status": "success",
        "count": len(models),
        "models": models
    }


@app.post(
    "/forecast",
    response_model=ForecastResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def forecast(request: ForecastRequest):
    return forecast_endpoint(request)


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def recommend(request: RecommendationRequest):
    return recommendation_endpoint(request)


@app.post(
    "/explain",
    response_model=ExplainResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def explain(request: ExplainRequest):
    return explain_endpoint(request)


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def ask(request: AskRequest):
    return ask_endpoint(request)


@app.get(
    "/drift/status",
    response_model=DriftResponse,
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit)
    ]
)
def drift_status():
    drift_dir = ROOT / "outputs" / "drift"

    summary_path = drift_dir / "drift_summary.json"

    if not summary_path.exists():
        return {
            "status": "not_available",
            "summary": {}
        }

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as f:
        summary = json.load(f)

    return {
        "status": "available",
        "summary": summary
    }