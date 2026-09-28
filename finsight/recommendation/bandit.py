import numpy as np

class LinUCB:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def score(self, x, arm):
        x = np.asarray(x, dtype=float)
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mean = x @ theta
        bonus = self.alpha * np.sqrt(x @ A_inv @ x)
        return float(mean + bonus)

    def select(self, x, arms=None):
        arms = list(range(self.n_arms)) if arms is None else list(arms)
        return max(arms, key=lambda a: self.score(x, a))

    def update(self, x, arm, reward):
        x = np.asarray(x, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x

    def rerank(self, x, candidates):
        return sorted(candidates, key=lambda a: self.score(x, int(a)), reverse=True)
