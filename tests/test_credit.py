import numpy as np
from sklearn.metrics import roc_auc_score
from credit.data import load, features, split, PROTECTED
from credit.metrics import (ks, ece, best_threshold, cost, score, bootstrap_auc_ci, paired_bootstrap_auc_diff,
                            approval_ratio, bootstrap_approval_ratio_ci, FN_COST, FP_COST)


def test_plot_writes_png(tmp_path):
    import plot
    output = tmp_path / "results.png"
    plot.main(output)
    assert output.stat().st_size > 8 and output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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


def test_ece_weights_bins_by_miscalibration():
    # two equal clusters: p=0.2 with a 0.4 default rate, p=0.8 with a 0.7 rate.
    # gaps 0.2 and 0.1, each weighted 0.5, so ECE = 0.5*0.2 + 0.5*0.1 = 0.15.
    y = np.array([1] * 200 + [0] * 300 + [1] * 350 + [0] * 150)
    p = np.array([0.2] * 500 + [0.8] * 500)
    assert abs(ece(y, p) - 0.15) < 1e-9


def test_score_auc_matches_hand_computed_value():
    # 3 of the 4 positive/negative pairs are correctly ordered -> AUC 0.75.
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    s = score(y, p, 0.5)
    assert s["auc"] == 0.75 and s["auc"] == round(roc_auc_score(y, p), 4)


def test_bootstrap_auc_ci_brackets_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0); y = rng.integers(0, 2, 1000); p = y * 0.4 + rng.random(1000)
    lo, hi = bootstrap_auc_ci(y, p, n=300)
    assert lo < roc_auc_score(y, p) < hi
    assert [lo, hi] == bootstrap_auc_ci(y, p, n=300)                       # fixed seed: reproducible
    yp = np.array([0] * 50 + [1] * 50)
    assert bootstrap_auc_ci(yp, yp.astype(float), n=100) == [1.0, 1.0]     # perfect ranking: CI collapses at 1


def test_approval_ratio_and_its_bootstrap_ci():
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    flag = np.array([1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)                  # approve 3/4 vs 2/4 -> ratio 0.5/0.75
    assert abs(approval_ratio(flag, g) - (0.5 / 0.75)) < 1e-12
    assert approval_ratio(np.array([1, 0, 1, 0], dtype=bool), np.array([0, 0, 1, 1])) == 1.0  # equal treatment
    rng = np.random.default_rng(0); grp = rng.integers(0, 2, 1000)
    decline = rng.random(1000) < np.where(grp == 1, 0.5, 0.3)              # group 1 declined more often
    lo, hi = bootstrap_approval_ratio_ci(decline, grp, n=300)
    assert lo <= approval_ratio(decline, grp) <= hi
    assert [lo, hi] == bootstrap_approval_ratio_ci(decline, grp, n=300)    # fixed seed: reproducible


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
