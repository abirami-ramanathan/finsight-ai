import json
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from tensorflow.keras.models import load_model

from .schemas import (
    AskRequest,
    AskResponse,
    ExplainRequest,
    ExplainResponse,
    ForecastRequest,
    ForecastResponse,
    RecommendationRequest,
    RecommendationResponse
)


ROOT = Path(__file__).resolve().parents[2]

CACHE_TTL = 300
RATE_LIMIT = 30
RATE_WINDOW = 60

API_KEY = "finsight-demo-key"

FEATURES = [
    "close",
    "volume",
    "return_1d",
    "ema_12",
    "rsi_14",
    "mfi_14",
    "stoch_k",
    "stoch_d",
    "bb_high",
    "bb_low",
    "macd",
    "macd_signal",
    "volatility_20",
    "garch_style_vol",
    "volatility_5"
]

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


class TTLCache:
    def __init__(self, ttl=CACHE_TTL):
        self.ttl = ttl
        self.data = {}
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key not in self.data:
                return None

            value, timestamp = self.data[key]

            if time.time() - timestamp > self.ttl:
                del self.data[key]
                return None

            return value

    def set(self, key, value):
        with self.lock:
            self.data[key] = (
                value,
                time.time()
            )

    def clear(self):
        with self.lock:
            self.data.clear()


class RateLimiter:
    def __init__(
        self,
        limit=RATE_LIMIT,
        window=RATE_WINDOW
    ):
        self.limit = limit
        self.window = window
        self.requests = defaultdict(deque)
        self.lock = Lock()

    def allow(self, client_id):
        now = time.time()

        with self.lock:
            queue = self.requests[client_id]

            while queue and now - queue[0] > self.window:
                queue.popleft()

            if len(queue) >= self.limit:
                return False

            queue.append(now)

            return True


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold=3,
        recovery_timeout=30
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.opened_at = None
        self.lock = Lock()

    def allow(self):
        with self.lock:
            if self.opened_at is None:
                return True

            if (
                time.time() - self.opened_at
                >= self.recovery_timeout
            ):
                self.opened_at = None
                self.failures = 0
                return True

            return False

    def success(self):
        with self.lock:
            self.failures = 0
            self.opened_at = None

    def failure(self):
        with self.lock:
            self.failures += 1

            if self.failures >= self.failure_threshold:
                self.opened_at = time.time()


cache = TTLCache()
rate_limiter = RateLimiter()
circuit_breaker = CircuitBreaker()


def verify_api_key(
    x_api_key: str = Security(api_key_header)
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True


def check_rate_limit(
    request: Request
):
    client = request.client

    if client is None:
        client_id = "unknown"
    else:
        client_id = client.host

    if not rate_limiter.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )


def execute_protected(
    function,
    *args,
    **kwargs
):
    if not circuit_breaker.allow():
        raise HTTPException(
            status_code=503,
            detail="Circuit breaker is open"
        )

    try:
        result = function(
            *args,
            **kwargs
        )

        circuit_breaker.success()

        return result

    except Exception:
        circuit_breaker.failure()
        raise


def cache_key(
    prefix,
    payload
):
    return (
        prefix
        + ":"
        + json.dumps(
            payload,
            sort_keys=True
        )
    )


def infer_dataset(
    asset_type,
    symbol
):
    if asset_type == "crypto":
        dataset_map = {
            "BTC_USD": "BTC-USD.csv",
            "ETH_USD": "ETH-USD.csv"
        }

        if symbol not in dataset_map:
            raise ValueError(
                f"Unsupported crypto symbol: {symbol}"
            )

        return (
            ROOT
            / "data"
            / "processed"
            / "crypto"
            / dataset_map[symbol]
        )

    if asset_type == "stock":
        return (
            ROOT
            / "data"
            / "processed"
            / "stock"
            / f"{symbol}.csv"
        )

    raise ValueError(
        f"Unsupported asset type: {asset_type}"
    )


def load_model_metadata(
    model_dir
):
    metadata_path = (
        model_dir
        / "metadata.json"
    )

    if not metadata_path.exists():
        return {}

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_forecasting_model(
    asset_type,
    symbol,
    model_name
):
    model_dir = (
        ROOT
        / "models"
        / "forecasting"
        / f"{asset_type}_{symbol}_{model_name}"
    )

    model_path = (
        model_dir
        / "model.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = load_model(
        model_path
    )

    metadata = load_model_metadata(
        model_dir
    )

    return (
        model,
        metadata,
        model_dir
    )


def prepare_forecast_input(
    df,
    features,
    lookback
):
    from sklearn.preprocessing import MinMaxScaler

    data = df.copy()

    data["target_return"] = np.log(
        data["close"].shift(-1)
        / data["close"]
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = (
        data
        .dropna(
            subset=features + ["target_return"]
        )
        .reset_index(drop=True)
    )

    train_end = int(
        len(data) * 0.70
    )

    train_features = (
        data
        .iloc[:train_end][features]
        .astype(float)
        .values
    )

    scaler = MinMaxScaler()

    scaler.fit(
        train_features
    )

    all_features = (
        data[features]
        .astype(float)
        .values
    )

    scaled = scaler.transform(
        all_features
    )

    if len(scaled) < lookback:
        raise ValueError(
            "Not enough data for forecasting sequence."
        )

    sequence = scaled[
        -lookback:
    ]

    return (
        sequence.reshape(
            1,
            lookback,
            len(features)
        ),
        data,
        scaler
    )


def forecast_service(
    asset_type,
    symbol,
    model_name
):
    model, metadata, model_dir = (
        load_forecasting_model(
            asset_type,
            symbol,
            model_name
        )
    )

    dataset = infer_dataset(
        asset_type,
        symbol
    )

    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}"
        )

    df = pd.read_csv(
        dataset
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    metadata_features = metadata.get(
        "features"
    )

    if (
        metadata_features
        and all(
            feature in df.columns
            for feature in metadata_features
        )
    ):
        features = metadata_features
    else:
        features = [
            feature
            for feature in FEATURES
            if feature in df.columns
        ]

    expected_features = int(
        model.input_shape[2]
    )

    lookback = int(
        model.input_shape[1]
    )

    features = features[
        :expected_features
    ]

    if len(features) != expected_features:
        raise ValueError(
            f"Model expects "
            f"{expected_features} features, "
            f"but {len(features)} are available."
        )

    X, prepared_data, scaler = (
        prepare_forecast_input(
            df,
            features,
            lookback
        )
    )

    prediction = float(
        model.predict(
            X,
            verbose=0
        ).reshape(-1)[0]
    )

    latest_close = float(
        prepared_data[
            "close"
        ].iloc[-1]
    )

    predicted_next_close = (
        latest_close
        * np.exp(prediction)
    )

    latest_date = None

    if "date" in prepared_data.columns:
        value = prepared_data[
            "date"
        ].iloc[-1]

        if pd.notna(value):
            latest_date = (
                value.isoformat()
            )

    return {
        "status": "success",
        "asset_type": asset_type,
        "symbol": symbol,
        "model": model_name,
        "prediction": prediction,
        "cached": False,
        "prediction_type": "next_day_log_return",
        "latest_close": latest_close,
        "predicted_next_close": float(
            predicted_next_close
        ),
        "lookback": lookback,
        "feature_count": len(features),
        "features": features,
        "latest_date": latest_date
    }


def recommendation_service(
    risk,
    horizon,
    top_k
):
    model_path = (
        ROOT
        / "models"
        / "recommendation"
        / "recommender.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Recommendation model not found: {model_path}"
        )

    import joblib

    recommender = joblib.load(
        model_path
    )

    risk_value = {
        "low": 0.20,
        "medium": 0.50,
        "high": 0.80
    }.get(
        risk,
        0.50
    )

    horizon_value = {
        "short": 0.20,
        "medium": 0.50,
        "long": 0.80
    }.get(
        horizon,
        0.50
    )

    profile = np.array([
        0.75,
        0.75,
        risk_value,
        horizon_value,
        0.75
    ])

    recommendations = recommender.recommend(
        profile
    )

    return {
        "status": "success",
        "risk": risk,
        "horizon": horizon,
        "recommendations": recommendations[:top_k]
    }

def explain_service(
    asset_type,
    symbol,
    model_name
):
    explanation_dir = (
        ROOT
        / "outputs"
        / "explainability"
    )

    files = list(
        explanation_dir.glob(
            f"{symbol}_{model_name}_*.json"
        )
    )

    if not files:
        return {
            "status": "success",
            "asset_type": asset_type,
            "symbol": symbol,
            "model": model_name,
            "explanation": {
                "message": "No saved explanation found."
            }
        }

    with open(
        files[0],
        "r",
        encoding="utf-8"
    ) as f:
        explanation = json.load(f)

    return {
        "status": "success",
        "asset_type": asset_type,
        "symbol": symbol,
        "model": model_name,
        "explanation": explanation
    }


def ask_service(
    question
):
    from finsight.rag.service import RAGService

    service = RAGService(
        ROOT / "data" / "rag_index",
        root_dir=ROOT
    )

    result = service.ask(
        question
    )

    if isinstance(result, dict):
        answer = result.get(
            "answer",
            str(result)
        )

        sources = result.get(
            "citations",
            []
        )
    else:
        answer = str(result)
        sources = []

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "sources": sources
    }


def forecast_endpoint(
    request: ForecastRequest
):
    payload = request.model_dump()

    key = cache_key(
        "forecast",
        payload
    )

    cached = cache.get(
        key
    )

    if cached is not None:
        result = cached.copy()
        result["cached"] = True

        return ForecastResponse(
            **result
        )

    result = execute_protected(
        forecast_service,
        request.asset_type,
        request.symbol,
        request.model
    )

    cache.set(
        key,
        result.copy()
    )

    return ForecastResponse(
        **result
    )


def recommendation_endpoint(
    request: RecommendationRequest
):
    payload = request.model_dump()

    key = cache_key(
        "recommendation",
        payload
    )

    cached = cache.get(
        key
    )

    if cached is not None:
        result = cached.copy()
        result["cached"] = True

        return RecommendationResponse(
            **result
        )

    result = execute_protected(
        recommendation_service,
        request.risk,
        request.horizon,
        request.top_k
    )

    cache.set(
        key,
        result.copy()
    )

    return RecommendationResponse(
        **result
    )


def explain_endpoint(
    request: ExplainRequest
):
    payload = request.model_dump()

    key = cache_key(
        "explain",
        payload
    )

    cached = cache.get(
        key
    )

    if cached is not None:
        result = cached.copy()
        result["cached"] = True

        return ExplainResponse(
            **result
        )

    result = execute_protected(
        explain_service,
        request.asset_type,
        request.symbol,
        request.model
    )

    cache.set(
        key,
        result.copy()
    )

    return ExplainResponse(
        **result
    )


def ask_endpoint(
    request: AskRequest
):
    payload = request.model_dump()

    key = cache_key(
        "ask",
        payload
    )

    cached = cache.get(
        key
    )

    if cached is not None:
        result = cached.copy()
        result["cached"] = True

        return AskResponse(
            **result
        )

    result = execute_protected(
        ask_service,
        request.question
    )

    cache.set(
        key,
        result.copy()
    )

    return AskResponse(
        **result
    )