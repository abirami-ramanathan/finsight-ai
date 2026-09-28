from pathlib import Path
import joblib
import pandas as pd

from finsight.data.split import prepare_data
from finsight.data.features import feature_columns
from finsight.models.forecaster import SequenceForecaster


SYMBOL = "GOLDBEES.NS"
ASSET = "gold"
MODELS = ["lstm", "bilstm", "gru", "gru_vol"]

path = Path(
    "data/processed/gold/GOLDBEES_NS.csv"
)

df = pd.read_csv(path)

columns = feature_columns("stock")

print("=" * 70)
print("GOLD ETF FORECASTING")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Features: {columns}")

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scaler,
    train_df,
    val_df,
    test_df
) = prepare_data(
    df,
    columns,
    lookback=30
)

results = []

for model_name in MODELS:

    print(f"\nTraining {model_name}...")

    model = SequenceForecaster(
        kind=model_name,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=30,
        batch_size=32
    )

    predictions = model.predict(X_test)

    pred_full = pd.DataFrame(
        0.0,
        index=range(len(predictions)),
        columns=range(len(columns))
    )

    true_full = pd.DataFrame(
        0.0,
        index=range(len(y_test)),
        columns=range(len(columns))
    )

    pred_full.iloc[:, 0] = predictions
    true_full.iloc[:, 0] = y_test

    predictions_original = scaler.inverse_transform(
        pred_full.values
    )[:, 0]

    true_original = scaler.inverse_transform(
        true_full.values
    )[:, 0]

    metrics = SequenceForecaster.metrics(
        true_original,
        predictions_original
    )

    model_dir = Path(
        f"models/forecasting/gold_{SYMBOL.replace('.', '_')}_{model_name}"
    )

    model.save(
        model_dir,
        {
            "asset_type": ASSET,
            "symbol": SYMBOL,
            "model": model_name,
            "features": columns,
            "lookback": 30
        }
    )

    joblib.dump(
        scaler,
        model_dir / "scaler.pkl"
    )

    results.append({
        "symbol": SYMBOL,
        "model": model_name,
        **metrics
    })

    print(
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"MAPE={metrics['mape']:.4f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/gold_forecasting_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("GOLD FORECASTING RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nSaved to:")
print("outputs/gold_forecasting_results.csv")