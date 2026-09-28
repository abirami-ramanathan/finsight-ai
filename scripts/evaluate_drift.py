import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import PSI_THRESHOLD, KL_THRESHOLD
from finsight.drift.monitor import detect_drift

rng = np.random.default_rng(42)
reference = pd.DataFrame({"return_1d": rng.normal(0, .01, 1000), "volatility": rng.normal(.2, .03, 1000)})
current = pd.DataFrame({"return_1d": rng.normal(.02, .015, 1000), "volatility": rng.normal(.3, .04, 1000)})
rows = detect_drift(reference, current, list(reference.columns), PSI_THRESHOLD, KL_THRESHOLD)
print(pd.DataFrame(rows).to_string(index=False))
