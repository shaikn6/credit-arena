import numpy as np
from sklearn.metrics import roc_auc_score

from study_extra import TARGETS, delong, frontier


def test_delong_difference_matches_auc_and_detects_gap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p1, p2 = y * 0.5 + rng.random(1000), y * 0.05 + rng.random(1000)
    d, z, p = delong(y, p1, p2)
    assert abs(d - (roc_auc_score(y, p1) - roc_auc_score(y, p2))) < 1e-9
    assert z > 3 and p < 0.01
    d2, z2, p2_ = delong(y, p2, p1)
    assert abs(d2 + d) < 1e-12 and abs(z2 + z) < 1e-9 and abs(p2_ - p) < 1e-12


def test_frontier_meets_target_on_validation_and_costs_more():
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.random(n) < 0.3
    y = (rng.random(n) < np.where(g, 0.35, 0.15)).astype(int)
    p = np.clip(0.2 + 0.3 * y + 0.1 * g + 0.2 * rng.random(n), 0, 1)
    out = frontier(y, p, g, y, p, g)  # validation == test: the constraint must hold exactly
    assert len(out) == len(TARGETS)
    for tgt, f in zip(TARGETS, out):
        if f is not None:
            assert f["ratio"] >= tgt - 1e-12 and f["cost"] >= out[0]["cost"] - 1e-12
