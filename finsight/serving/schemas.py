from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    asset_type: str = Field(default="crypto")
    symbol: str = Field(default="BTC_USD")
    model: str = Field(default="bilstm")


class RecommendationRequest(BaseModel):
    risk: str = Field(default="low")
    horizon: str = Field(default="medium")
    top_k: int = Field(default=5, ge=1, le=20)


class ExplainRequest(BaseModel):
    asset_type: str = Field(default="crypto")
    symbol: str = Field(default="BTC_USD")
    model: str = Field(default="bilstm")


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000
    )


class ForecastResponse(BaseModel):
    status: str
    asset_type: str
    symbol: str
    model: str
    prediction: Optional[float] = None
    cached: bool = False
    prediction_type: Optional[str] = None
    latest_close: Optional[float] = None
    predicted_next_close: Optional[float] = None
    lookback: Optional[int] = None
    feature_count: Optional[int] = None
    features: List[str] = []
    latest_date: Optional[str] = None

class RecommendationResponse(BaseModel):
    status: str
    risk: str
    horizon: str
    recommendations: List[Dict[str, Any]]
    cached: bool = False


class ExplainResponse(BaseModel):
    status: str
    asset_type: str
    symbol: str
    model: str
    explanation: Dict[str, Any]
    cached: bool = False


class AskResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class DriftResponse(BaseModel):
    status: str
    summary: Dict[str, Any]