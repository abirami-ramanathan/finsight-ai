# FinSight AI

An executable end-to-end implementation of the architecture described in the FinSight AI paper.

> Important: the numerical results in the paper are reported first-cycle results. This repository does **not** claim to reproduce those numbers until the real datasets are downloaded and the training/evaluation commands are executed.

## Architecture

- Module A — data ingestion, validation, indicators, scaling, chronological splits
- Module B — stock/crypto/REIT forecasting with GRU/LSTM-style models and volatility/yield/occupancy features
- Module C — KNN investor recommendation + LinUCB contextual-bandit re-ranking
- Module D — FAISS RAG + multilingual query bridge (English/Hindi/Tamil)
- Module E — SHAP + Integrated Gradients explanations + groundedness score
- Module F — PSI/KL drift monitoring + retraining controller
- Module G — FastAPI serving + Streamlit dashboard

## Windows setup

Recommended: Python 3.11.

```powershell
cd FinSightAI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Copy environment configuration:

```powershell
Copy-Item .env.example .env
```

## Prepare data

The project supports live downloads where the upstream source is available and local CSV files for reproducible experiments.

```powershell
python scripts/download_data.py
```

Then build features:

```powershell
python scripts/build_features.py
```

## Train forecasting models

```powershell
python scripts/train_forecaster.py --asset stock --symbol RELIANCE.NS --model gru_vol
python scripts/train_forecaster.py --asset crypto --symbol BTC-USD --model gru_vol
python scripts/train_forecaster.py --asset reit --symbol EMBASSY.NS --model lstm_reit
```

Models are saved under `models/`.

## Train recommendation system

```powershell
python scripts/train_recommender.py
```

## Build RAG index

Put source documents in `data/documents/` and run:

```powershell
python scripts/build_rag_index.py
```

For a local LLaMA-3 setup, configure a Hugging Face token and model ID in `.env`. The system has an extractive/local fallback so the API remains runnable without an 8B GPU model.

## Run API

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run dashboard

In another terminal:

```powershell
streamlit run dashboard/streamlit_app.py
```

## Run tests

```powershell
pytest -q
```

## Reproduce paper-style evaluation

```powershell
python scripts/evaluate_forecaster.py --asset stock --symbol RELIANCE.NS --model gru_vol
python scripts/evaluate_recommender.py
python scripts/evaluate_drift.py
```

Use the generated files under `outputs/` for plots/tables.

## Model storage

All downloaded/trained model artifacts belong under:

```text
models/
├── forecasting/
├── recommender/
├── rag/
└── explainability/
```

Do not commit large model weights to Git. The included `.gitignore` excludes model binaries by default.
