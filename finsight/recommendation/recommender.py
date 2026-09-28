from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .bandit import LinUCB


class InvestorRecommender:

    def __init__(self, k=5, alpha=1.0):

        self.k = k
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.knn = None
        self.items = None
        self.item_features = None
        self.bandit = None

    def fit(self, item_features, item_names):

        self.items = list(item_names)

        self.item_features = np.asarray(
            item_features,
            dtype=float
        )

        scaled = self.scaler.fit_transform(
            self.item_features
        )

        self.knn = NearestNeighbors(
            n_neighbors=min(
                self.k,
                len(self.items)
            ),
            metric="euclidean"
        )

        self.knn.fit(scaled)

        self.bandit = LinUCB(
            n_arms=len(self.items),
            n_features=self.item_features.shape[1],
            alpha=self.alpha
        )

        return self

    def retrieve(self, profile):

        profile = np.asarray(
            profile,
            dtype=float
        ).reshape(1, -1)

        scaled_profile = self.scaler.transform(
            profile
        )

        distances, indices = self.knn.kneighbors(
            scaled_profile,
            n_neighbors=min(
                self.k,
                len(self.items)
            )
        )

        candidates = indices[0].tolist()

        return candidates, distances[0].tolist()

    def recommend(self, profile):

        candidates, distances = self.retrieve(
            profile
        )

        ranked = self.bandit.rerank(
            profile,
            candidates
        )

        recommendations = []

        for rank, arm in enumerate(
            ranked,
            start=1
        ):

            position = candidates.index(
                int(arm)
            )

            recommendations.append({
                "rank": rank,
                "item": self.items[int(arm)],
                "knn_distance": float(
                    distances[position]
                ),
                "linucb_score": float(
                    self.bandit.score(
                        profile,
                        int(arm)
                    )
                )
            })

        return recommendations

    def update(self, profile, arm, reward):

        self.bandit.update(
            profile,
            int(arm),
            reward
        )

    def save(self, path: Path):

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self,
            path
        )

    @staticmethod
    def load(path: Path):

        return joblib.load(path)