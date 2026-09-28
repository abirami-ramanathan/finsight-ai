from pathlib import Path
import json
import numpy as np
import pandas as pd


class DriftMonitor:
    def __init__(
        self,
        psi_threshold=0.20,
        kl_threshold=0.10,
        error_window=20,
        bins=10
    ):
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.error_window = error_window
        self.bins = bins

    def _clean(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return values

    def _edges(self, reference):
        reference = self._clean(reference)

        if len(reference) < 2:
            return np.array([-np.inf, np.inf])

        quantiles = np.linspace(
            0.0,
            1.0,
            self.bins + 1
        )

        edges = np.quantile(
            reference,
            quantiles
        )

        edges = np.unique(edges)

        if len(edges) < 3:
            minimum = np.min(reference)
            maximum = np.max(reference)

            if minimum == maximum:
                delta = max(
                    abs(minimum) * 0.01,
                    1e-6
                )

                minimum -= delta
                maximum += delta

            edges = np.linspace(
                minimum,
                maximum,
                self.bins + 1
            )

        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges

    def _distribution(self, reference, current):
        reference = self._clean(reference)
        current = self._clean(current)

        if len(reference) < 2 or len(current) < 2:
            return (
                np.array([1.0]),
                np.array([1.0])
            )

        edges = self._edges(reference)

        ref_counts, _ = np.histogram(
            reference,
            bins=edges
        )

        current_counts, _ = np.histogram(
            current,
            bins=edges
        )

        epsilon = 1e-4

        ref_probs = (
            ref_counts.astype(float)
            + epsilon
        )

        current_probs = (
            current_counts.astype(float)
            + epsilon
        )

        ref_probs /= ref_probs.sum()
        current_probs /= current_probs.sum()

        return ref_probs, current_probs

    def psi(self, reference, current):
        ref_probs, current_probs = (
            self._distribution(
                reference,
                current
            )
        )

        value = np.sum(
            (current_probs - ref_probs)
            * np.log(
                current_probs / ref_probs
            )
        )

        return float(value)

    def kl_divergence(self, reference, current):
        ref_probs, current_probs = (
            self._distribution(
                reference,
                current
            )
        )

        value = np.sum(
            ref_probs
            * np.log(
                ref_probs / current_probs
            )
        )

        return float(value)

    def rolling_error(self, actual, predicted):
        actual = np.asarray(
            actual,
            dtype=float
        )

        predicted = np.asarray(
            predicted,
            dtype=float
        )

        mask = (
            np.isfinite(actual)
            &
            np.isfinite(predicted)
        )

        actual = actual[mask]
        predicted = predicted[mask]

        if len(actual) == 0:
            return 0.0

        errors = np.abs(
            actual - predicted
        )

        window = min(
            self.error_window,
            len(errors)
        )

        return float(
            np.mean(
                errors[-window:]
            )
        )

    def feature_drift(
        self,
        reference_df,
        current_df,
        features
    ):
        rows = []

        for feature in features:
            if feature not in reference_df.columns:
                continue

            if feature not in current_df.columns:
                continue

            reference = (
                reference_df[feature]
                .astype(float)
                .values
            )

            current = (
                current_df[feature]
                .astype(float)
                .values
            )

            psi_value = self.psi(
                reference,
                current
            )

            kl_value = self.kl_divergence(
                reference,
                current
            )

            rows.append({
                "feature": feature,
                "psi": psi_value,
                "kl_divergence": kl_value,
                "psi_drift": (
                    psi_value
                    >= self.psi_threshold
                ),
                "kl_drift": (
                    kl_value
                    >= self.kl_threshold
                )
            })

        return pd.DataFrame(rows)

    def overall_drift(
        self,
        feature_results
    ):
        if feature_results.empty:
            return {
                "max_psi": 0.0,
                "mean_psi": 0.0,
                "max_kl": 0.0,
                "mean_kl": 0.0,
                "drifted_features": 0,
                "drift_detected": False
            }

        max_psi = float(
            feature_results["psi"].max()
        )

        mean_psi = float(
            feature_results["psi"].mean()
        )

        max_kl = float(
            feature_results[
                "kl_divergence"
            ].max()
        )

        mean_kl = float(
            feature_results[
                "kl_divergence"
            ].mean()
        )

        psi_drift_count = int(
            feature_results[
                "psi_drift"
            ].sum()
        )

        kl_drift_count = int(
            feature_results[
                "kl_drift"
            ].sum()
        )

        drift_detected = bool(
            (
                feature_results["psi"]
                >= self.psi_threshold
            ).any()
            or
            (
                feature_results[
                    "kl_divergence"
                ]
                >= self.kl_threshold
            ).any()
        )

        return {
            "max_psi": max_psi,
            "mean_psi": mean_psi,
            "max_kl": max_kl,
            "mean_kl": mean_kl,
            "drifted_features": max(
                psi_drift_count,
                kl_drift_count
            ),
            "drift_detected": drift_detected
        }

    def evaluate(
        self,
        reference_df,
        current_df,
        features,
        actual=None,
        predicted=None
    ):
        feature_results = self.feature_drift(
            reference_df,
            current_df,
            features
        )

        summary = self.overall_drift(
            feature_results
        )

        if (
            actual is not None
            and predicted is not None
        ):
            summary["rolling_error"] = (
                self.rolling_error(
                    actual,
                    predicted
                )
            )
        else:
            summary["rolling_error"] = None

        return (
            summary,
            feature_results
        )

    def save_json(
        self,
        data,
        path
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                indent=2
            )