from finsight.recommendation.bandit import LinUCB

def test_bandit_updates():
    b = LinUCB(3, 2, alpha=1.0)
    x = [1.0, 0.5]
    arm = b.select(x)
    b.update(x, arm, 1.0)
    assert b.score(x, arm) >= 0
