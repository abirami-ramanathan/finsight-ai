import numpy as np


class SHAPExplainer:

    def __init__(
        self,
        model,
        feature_names
    ):
        self.model = model
        self.feature_names = list(
            feature_names
        )
        self.explainer = None

    def fit(
        self,
        background_data
    ):
        import shap

        background_data = np.asarray(
            background_data
        )

        self.explainer = (
            shap.Explainer(
                self.model,
                background_data
            )
        )

        return self

    def explain(
        self,
        x
    ):
        if self.explainer is None:
            raise RuntimeError(
                "SHAP explainer has not been fitted."
            )

        x = np.asarray(
            x
        )

        shap_values = self.explainer(
            x
        )

        values = shap_values.values

        if values.ndim > 2:
            values = np.mean(
                values,
                axis=1
            )

        values = np.asarray(
            values
        )

        if values.ndim == 2:
            values = values[0]

        results = []

        for index, feature in enumerate(
            self.feature_names
        ):
            value = float(
                values[index]
            )

            results.append(
                {
                    "feature": feature,
                    "shap_value": value,
                    "absolute_shap": abs(value)
                }
            )

        results.sort(
            key=lambda item:
            item["absolute_shap"],
            reverse=True
        )

        return results