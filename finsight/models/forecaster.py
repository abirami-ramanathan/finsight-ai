from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SequenceForecaster:
    def __init__(self, kind="gru_vol", input_shape=None, learning_rate=0.001):
        self.kind = kind
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build() if input_shape else None

    def _build(self):
        model = Sequential([
            Input(shape=self.input_shape)
        ])

        if self.kind == "lstm":
            model.add(LSTM(64))

        elif self.kind == "bilstm":
            model.add(Bidirectional(LSTM(64)))

        elif self.kind in ["gru", "gru_vol"]:
            model.add(GRU(64))

        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

        model.compile(
            optimizer=Adam(self.learning_rate),
            loss="mse"
        )

        return model

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        batch_size=32
    ):
        if self.model is None:
            self.input_shape = X_train.shape[1:]
            self.model = self._build()

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ],
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0).reshape(-1)

    @staticmethod
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)

        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(
                np.mean(
                    np.abs(
                        (y_true - y_pred)
                        / np.clip(np.abs(y_true), 1e-8, None)
                    )
                ) * 100
            ),
        }

    def save(self, model_dir: Path, metadata: dict):
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(
            model_dir / "model.keras"
        )

        with open(
            model_dir / "metadata.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                default=str
            )

    @classmethod
    def load(cls, model_dir: Path):
        with open(
            model_dir / "metadata.json",
            encoding="utf-8"
        ) as f:
            meta = json.load(f)

        obj = cls(
            meta["kind"],
            tuple(meta["input_shape"]),
            meta.get("learning_rate", 0.001)
        )

        obj.model = tf.keras.models.load_model(
            model_dir / "model.keras"
        )

        return obj, meta