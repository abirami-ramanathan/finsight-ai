import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import MODEL_DIR
from finsight.recommendation.recommender import InvestorRecommender

r = InvestorRecommender.load(MODEL_DIR / "recommender" / "recommender.joblib")
profile = [100000, 20000, .65, 8, 30]
scores = np.linspace(.55, .90, len(r.items))
print(r.recommend(profile, scores, 5))
