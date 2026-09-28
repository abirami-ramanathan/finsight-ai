from pathlib import Path
from datetime import datetime
import json
import shutil


class DriftController:
    def __init__(
        self,
        monitor,
        model_dir="models/drift_versions",
        psi_threshold=0.20,
        retrain_cooldown=4
    ):
        self.monitor = monitor
        self.model_dir = Path(
            model_dir
        )
        self.psi_threshold = (
            psi_threshold
        )
        self.retrain_cooldown = (
            retrain_cooldown
        )
        self.events = []
        self.last_retrain_period = None

    def should_retrain(
        self,
        summary,
        period
    ):
        drift = bool(
            summary.get(
                "drift_detected",
                False
            )
        )

        if not drift:
            return False

        if self.last_retrain_period is None:
            return True

        return (
            period
            - self.last_retrain_period
            >= self.retrain_cooldown
        )

    def create_version(
        self,
        model_path,
        reason="drift"
    ):
        model_path = Path(
            model_path
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        version_name = (
            f"{model_path.stem}_"
            f"{reason}_"
            f"{timestamp}"
            f"{model_path.suffix}"
        )

        destination = (
            self.model_dir
            / version_name
        )

        shutil.copy2(
            model_path,
            destination
        )

        return destination

    def register_event(
        self,
        period,
        summary,
        retrained,
        model_version=None
    ):
        if retrained:
            self.last_retrain_period = (
                period
            )

        event = {
            "period": int(period),
            "max_psi": float(
                summary.get(
                    "max_psi",
                    0.0
                )
            ),
            "mean_psi": float(
                summary.get(
                    "mean_psi",
                    0.0
                )
            ),
            "max_kl": float(
                summary.get(
                    "max_kl",
                    0.0
                )
            ),
            "mean_kl": float(
                summary.get(
                    "mean_kl",
                    0.0
                )
            ),
            "rolling_error": (
                summary.get(
                    "rolling_error"
                )
            ),
            "drifted_features": int(
                summary.get(
                    "drifted_features",
                    0
                )
            ),
            "drift_detected": bool(
                summary.get(
                    "drift_detected",
                    False
                )
            ),
            "retrained": bool(
                retrained
            ),
            "model_version": (
                str(model_version)
                if model_version
                else None
            )
        }

        self.events.append(
            event
        )

        return event

    def save_events(self, path):
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
                self.events,
                f,
                indent=2
            )