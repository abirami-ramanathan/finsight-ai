from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from .integrated_gradients import (
    IntegratedGradientsExplainer
)


class ExplanationService:

    def __init__(
        self,
        output_dir
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def explain_recurrent_prediction(
        self,
        model,
        x,
        feature_names,
        asset_name,
        model_name,
        baseline=None,
        steps=64
    ):

        explainer = (
            IntegratedGradientsExplainer(
                model=model,
                feature_names=feature_names,
                steps=steps
            )
        )

        result = explainer.explain(
            x=x,
            baseline=baseline
        )

        result["asset"] = asset_name
        result["model"] = model_name
        result["method"] = (
            "Integrated Gradients"
        )

        self._save_json(
            result,
            asset_name,
            model_name
        )

        self._save_feature_plot(
            result,
            asset_name,
            model_name
        )

        self._save_temporal_heatmap(
            result,
            asset_name,
            model_name
        )

        result["explanation"] = (
            self._generate_text(
                result
            )
        )

        return result

    def _generate_text(
        self,
        result,
        top_k=5
    ):

        features = result[
            "feature_attributions"
        ][:top_k]

        temporal = sorted(
            result[
                "temporal_attributions"
            ],
            key=lambda item:
            item["absolute_attribution"],
            reverse=True
        )[:5]

        prediction = result[
            "prediction"
        ]

        baseline = result[
            "baseline_prediction"
        ]

        difference = result[
            "prediction_difference"
        ]

        lines = []

        lines.append(
            "WHY THIS FORECAST?"
        )

        lines.append(
            f"The {result['model']} model produced "
            f"a prediction of {prediction:.6f} "
            f"for {result['asset']}."
        )

        lines.append(
            f"The baseline prediction was "
            f"{baseline:.6f}, giving a prediction "
            f"difference of {difference:.6f}."
        )

        lines.append(
            "The strongest feature contributions "
            "were:"
        )

        for item in features:

            if item["attribution"] > 0:
                direction = "increased"
            elif item["attribution"] < 0:
                direction = "decreased"
            else:
                direction = "had no directional effect on"

            percentage = (
                item["relative_importance"] *
                100
            )

            lines.append(
                f"{item['feature']} "
                f"({direction} the model output; "
                f"attribution={item['attribution']:.6f}, "
                f"relative importance={percentage:.2f}%)."
            )

        lines.append(
            "The most influential historical "
            "time steps were:"
        )

        for item in temporal:

            lines.append(
                f"timestep {item['timestep']} "
                f"({item['relative_position']}) "
                f"with attribution "
                f"{item['attribution']:.6f}."
            )

        lines.append(
            "These attributions describe how the "
            "input features and historical time steps "
            "contributed to this specific model "
            "prediction; they do not represent "
            "causal relationships."
        )

        return "\n".join(
            lines
        )

    def _safe_name(
        self,
        value
    ):

        return (
            str(value)
            .replace(
                "/",
                "_"
            )
            .replace(
                "\\",
                "_"
            )
            .replace(
                " ",
                "_"
            )
        )

    def _save_json(
        self,
        result,
        asset_name,
        model_name
    ):

        safe_asset = self._safe_name(
            asset_name
        )

        safe_model = self._safe_name(
            model_name
        )

        path = (
            self.output_dir /
            f"{safe_asset}_{safe_model}_ig.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=2
            )

    def _save_feature_plot(
        self,
        result,
        asset_name,
        model_name
    ):

        features = result[
            "feature_attributions"
        ][:10]

        names = [
            item["feature"]
            for item in features
        ]

        values = [
            item["attribution"]
            for item in features
        ]

        names = names[::-1]
        values = values[::-1]

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            names,
            values
        )

        plt.axvline(
            0,
            linewidth=1
        )

        plt.xlabel(
            "Integrated Gradients Attribution"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            f"Why This Forecast? — "
            f"{asset_name} — {model_name}"
        )

        plt.tight_layout()

        safe_asset = self._safe_name(
            asset_name
        )

        safe_model = self._safe_name(
            model_name
        )

        path = (
            self.output_dir /
            f"{safe_asset}_{safe_model}_ig.png"
        )

        plt.savefig(
            path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

    def _save_temporal_heatmap(
        self,
        result,
        asset_name,
        model_name
    ):

        matrix = np.asarray(
            result[
                "timestep_feature_attributions"
            ],
            dtype=float
        )

        feature_names = [
            item["feature"]
            for item in result[
                "feature_attributions"
            ]
        ]

        original_feature_names = list(
            result[
                "timestep_feature_attributions"
            ][0:0]
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Expected temporal attribution matrix "
                "with shape (timesteps, features)."
            )

        feature_count = matrix.shape[1]

        if feature_count != len(
            result["feature_attributions"]
        ):
            raise ValueError(
                "Temporal attribution feature count "
                "does not match feature attribution results."
            )

        feature_names = [
            item["feature"]
            for item in sorted(
                result["feature_attributions"],
                key=lambda item:
                item["feature"]
            )
        ]

        feature_order = {
            name: index
            for index, name in enumerate(
                sorted(feature_names)
            )
        }

        matrix_for_plot = matrix.T

        plt.figure(
            figsize=(12, 8)
        )

        plt.imshow(
            matrix_for_plot,
            aspect="auto",
            interpolation="nearest"
        )

        plt.colorbar(
            label="Integrated Gradients Attribution"
        )

        plt.yticks(
            range(
                len(feature_names)
            ),
            feature_names
        )

        timestep_count = matrix.shape[0]

        tick_count = min(
            10,
            timestep_count
        )

        tick_positions = np.linspace(
            0,
            timestep_count - 1,
            tick_count,
            dtype=int
        )

        tick_labels = [
            f"t-{timestep_count - 1 - position}"
            for position in tick_positions
        ]

        plt.xticks(
            tick_positions,
            tick_labels,
            rotation=45
        )

        plt.xlabel(
            "Historical Time Step"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            f"Temporal Feature Attribution — "
            f"{asset_name} — {model_name}"
        )

        plt.tight_layout()

        safe_asset = self._safe_name(
            asset_name
        )

        safe_model = self._safe_name(
            model_name
        )

        path = (
            self.output_dir /
            f"{safe_asset}_{safe_model}_temporal_heatmap.png"
        )

        plt.savefig(
            path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()