import numpy as np
from finsight.drift.monitor import psi

def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    assert psi(x, x) < 0.01
