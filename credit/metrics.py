import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve

FN_COST, FP_COST = 5.0, 1.0  # missing a defaulter costs 5x refusing a good borrower


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def best_threshold(y, p):
    """Threshold minimising expected cost on (validation) data."""
    ts = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 91)))
    return float(ts[np.argmin([cost(y, p, t) for t in ts])])


def cost(y, p, t):
    pred = p >= t
    return float(((~pred) & (y == 1)).sum() * FN_COST + (pred & (y == 0)).sum() * FP_COST) / len(y)


def score(y, p, t):
    return dict(auc=round(roc_auc_score(y, p), 4), pr_auc=round(average_precision_score(y, p), 4),
                ks=round(ks(y, p), 4), brier=round(brier_score_loss(y, p), 4), logloss=round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
                ece=round(ece(y, p), 4), cost_per_account=round(cost(y, p, t), 4), threshold=round(t, 4))


def bootstrap_auc_ci(y, p, n=500, seed=0):
    rng = np.random.default_rng(seed)
    aucs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


def paired_bootstrap_auc_diff(y, p_a, p_b, n=2000, seed=0):
    """AUC(p_a) - AUC(p_b) on the same cases, with a 95% percentile interval from n paired resamples of the rows."""
    rng = np.random.default_rng(seed)
    d = [roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return dict(diff=round(float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), 4),
                ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)])
