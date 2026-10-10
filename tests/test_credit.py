import numpy as np
from credit.data import load, features, split, PROTECTED
from credit.metrics import ks, ece, best_threshold, cost, paired_bootstrap_auc_diff, FN_COST, FP_COST


def test_protected_attribute_not_a_feature():
    df = load()
    assert PROTECTED not in features(df).columns and "target" not in features(df).columns


def test_split_is_disjoint_stratified_and_complete():
    df = load(); tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not (set(tr.index) & set(va.index)) and not (set(va.index) & set(te.index)) and not (set(tr.index) & set(te.index))
    assert max(abs(s["target"].mean() - df["target"].mean()) for s in (tr, va, te)) < 0.005


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks(y, y.astype(float)) == 1.0
    assert ks(y, np.random.default_rng(0).random(100)) < 0.4


def test_cost_weights_and_threshold_search():
    y = np.array([1, 0]); assert cost(y, np.array([0.1, 0.9]), 0.5) == (FN_COST + FP_COST) / 2
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 2000); p = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    t = best_threshold(y, p); assert cost(y, p, t) <= min(cost(y, p, 0.05), cost(y, p, 0.95))


def test_ece_zero_for_calibrated_constant():
    y = np.array([0, 1] * 500); assert ece(y, np.full(1000, 0.5)) < 1e-9


def test_paired_bootstrap_auc_diff():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 1000)
    strong, weak, noise = y * 0.5 + rng.random(1000), y * 0.05 + rng.random(1000), rng.random(1000) * 1e-3
    r = paired_bootstrap_auc_diff(y, strong, weak, n=300)
    assert abs(r["diff"] - (roc_auc_score(y, strong) - roc_auc_score(y, weak))) < 1e-4
    assert 0 < r["ci95"][0] <= r["diff"] <= r["ci95"][1]                                    # a real gap excludes zero
    assert r == paired_bootstrap_auc_diff(y, strong, weak, n=300)                           # fixed seed: reproducible
    back = paired_bootstrap_auc_diff(y, weak, strong, n=300)
    assert back["diff"] == -r["diff"] and back["ci95"] == [-r["ci95"][1], -r["ci95"][0]]    # antisymmetric
    assert paired_bootstrap_auc_diff(y, strong, strong, n=50) == dict(diff=0.0, ci95=[0.0, 0.0])
    same = paired_bootstrap_auc_diff(y, strong, strong + noise, n=300)                      # near-identical models: includes zero
    assert same["ci95"][0] <= 0 <= same["ci95"][1]
