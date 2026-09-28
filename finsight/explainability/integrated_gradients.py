import numpy as np
import tensorflow as tf


class IntegratedGradientsExplainer:

    def __init__(
        self,
        model,
        feature_names,
        steps=64
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.steps = steps

    def _prediction(self, x):
        output = self.model(
            x,
            training=False
        )

        if isinstance(output, (list, tuple)):
            output = output[0]

        return tf.reshape(
            output,
            [-1]
        )

    def explain(
        self,
        x,
        baseline=None
    ):
        x = np.asarray(
            x,
            dtype=np.float32
        )

        if x.ndim == 2:
            x = np.expand_dims(
                x,
                axis=0
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, timesteps, features)."
            )

        if baseline is None:
            baseline = np.zeros_like(x)
        else:
            baseline = np.asarray(
                baseline,
                dtype=np.float32
            )

            if baseline.ndim == 2:
                baseline = np.expand_dims(
                    baseline,
                    axis=0
                )

            if baseline.shape != x.shape:
                raise ValueError(
                    "Baseline shape must match input shape."
                )

        x_tensor = tf.convert_to_tensor(
            x,
            dtype=tf.float32
        )

        baseline_tensor = tf.convert_to_tensor(
            baseline,
            dtype=tf.float32
        )

        difference = (
            x_tensor -
            baseline_tensor
        )

        gradients = []

        for i in range(
            1,
            self.steps + 1
        ):

            alpha = tf.cast(
                i / self.steps,
                tf.float32
            )

            interpolated = (
                baseline_tensor +
                alpha * difference
            )

            with tf.GradientTape() as tape:

                tape.watch(
                    interpolated
                )

                prediction = self._prediction(
                    interpolated
                )

            gradient = tape.gradient(
                prediction,
                interpolated
            )

            gradients.append(
                gradient
            )

        gradients = tf.stack(
            gradients,
            axis=0
        )

        average_gradients = tf.reduce_mean(
            gradients,
            axis=0
        )

        attributions = (
            difference *
            average_gradients
        )

        attributions = attributions.numpy()

        prediction = float(
            self._prediction(
                x_tensor
            ).numpy()[0]
        )

        baseline_prediction = float(
            self._prediction(
                baseline_tensor
            ).numpy()[0]
        )

        timestep_feature_attributions = (
            attributions[0]
        )

        feature_attributions = np.sum(
            timestep_feature_attributions,
            axis=0
        )

        timestep_attributions = np.sum(
            timestep_feature_attributions,
            axis=1
        )

        absolute_importance = np.abs(
            feature_attributions
        )

        total_importance = (
            np.sum(
                absolute_importance
            ) + 1e-12
        )

        normalized_importance = (
            absolute_importance /
            total_importance
        )

        results = []

        for index, feature in enumerate(
            self.feature_names
        ):

            results.append(
                {
                    "feature": feature,
                    "attribution": float(
                        feature_attributions[index]
                    ),
                    "absolute_attribution": float(
                        absolute_importance[index]
                    ),
                    "relative_importance": float(
                        normalized_importance[index]
                    )
                }
            )

        results.sort(
            key=lambda item:
            item["absolute_attribution"],
            reverse=True
        )

        temporal_results = []

        timestep_count = len(
            timestep_attributions
        )

        for index, attribution in enumerate(
            timestep_attributions
        ):

            temporal_results.append(
                {
                    "timestep": index + 1,
                    "relative_position": -(
                        timestep_count - index
                    ),
                    "attribution": float(
                        attribution
                    ),
                    "absolute_attribution": float(
                        abs(attribution)
                    )
                }
            )

        temporal_results.sort(
            key=lambda item:
            item["absolute_attribution"],
            reverse=True
        )

        completeness = float(
            np.sum(
                feature_attributions
            )
        )

        prediction_difference = (
            prediction -
            baseline_prediction
        )

        return {
            "prediction": prediction,
            "baseline_prediction": baseline_prediction,
            "prediction_difference": prediction_difference,
            "feature_attributions": results,
            "timestep_feature_attributions": (
                timestep_feature_attributions.tolist()
            ),
            "temporal_attributions": temporal_results,
            "completeness_sum": completeness,
            "completeness_error": float(
                abs(
                    completeness -
                    prediction_difference
                )
            )
        }