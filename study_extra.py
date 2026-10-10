"""Follow-up analyses for the paper (keeps study.py intact). Writes study_extra.json.

1. learning curve: Taiwan, training-set size 500 ... 18,000, fixed validation/test sets per seed;
2. reliability bins (calibration curves) per model on both portfolios;
3. group error rates at the cost-minimising threshold: approval rate among good payers (equal opportunity) and among
   defaulters, by age group and by sex, plus observed and mean predicted default rate per group;
4. fine sweep of age / sex approval parity over cost ratios 1..10;
5. proxy analysis: how well the non-age inputs predict 'under 25', and which inputs carry most of it;
6. per-split AUC and expected cost (same splits as study.py) for the corrected resampled t-test, and a DeLong test on
   split 0;
7. accuracy-parity frontier: separate thresholds for under-25 and older applicants chosen on validation data to
   minimise expected cost subject to a minimum age approval ratio, then applied to test data.
"""
import json
import time

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from study import AGE_COL, COST_RATIOS, best_threshold, cost, german, make_models, parity, taiwan

BASE = ("logistic regression", "random forest", "gradient boosting", "MLP")
SIZES = (500, 1000, 2000, 5000, 10000, 18000)
FINE_R = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10)
GROUP_R = (2, 5)
TARGETS = (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)  # minimum age approval ratio required on validation data
FRONTIER_R = 5


def delong(y, p1, p2):
    """DeLong et al. (1988) test for two correlated AUCs on the same cases; returns (auc1 - auc2, z, two-sided p)."""
    from scipy.stats import norm
    out = []
    for p in (p1, p2):
        a, b = p[y == 1], p[y == 0]
        cmp = (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])
        out.append((cmp.mean(1), cmp.mean(0), cmp.mean()))
    v10 = np.cov(np.vstack([o[0] for o in out])); v01 = np.cov(np.vstack([o[1] for o in out]))
    S = v10 / (y == 1).sum() + v01 / (y == 0).sum()
    d = out[0][2] - out[1][2]
    z = d / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return float(d), float(z), float(2 * norm.sf(abs(z)))


def frontier(yv, pv, gv, yt, pt, gt, r=FRONTIER_R):
    """Group thresholds (t_young, t_old) from validation quantiles; min validation cost s.t. ratio >= target."""
    def grid(p, g):
        return np.unique(np.quantile(p[g], np.linspace(0.02, 0.98, 49)))
    ty, to = grid(pv, gv), grid(pv, ~gv)
    def stats(y, p, g, a, b):
        dy = p[g][None, :] >= a[:, None]; do = p[~g][None, :] >= b[:, None]  # decline flags
        yy, yo = y[g], y[~g]
        cy = ((~dy) & (yy == 1)).sum(1) * r + (dy & (yy == 0)).sum(1)
        co = ((~do) & (yo == 1)).sum(1) * r + (do & (yo == 0)).sum(1)
        ay, ao = 1 - dy.mean(1), 1 - do.mean(1)
        c = (cy[:, None] + co[None, :]) / len(y)
        ratio = np.minimum(ay[:, None], ao[None, :]) / np.maximum(np.maximum(ay[:, None], ao[None, :]), 1e-12)
        return c, ratio
    cv, rv = stats(yv, pv, gv, ty, to)
    out = []
    for tgt in TARGETS:
        ok = rv >= tgt
        if not ok.any():
            out.append(None); continue
        i, j = np.unravel_index(np.where(ok, cv, np.inf).argmin(), cv.shape)
        ct, rt = stats(yt, pt, gt, ty[i:i + 1], to[j:j + 1])
        out.append({"cost": float(ct[0, 0]), "ratio": float(rt[0, 0])})
    return out


def splits(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return tr, va, te


def fit_predict(X, y, tr, va, te, names=BASE):
    models = make_models(X)
    out = {}
    for n in names:
        m = models[n].fit(X.iloc[tr], y[tr])
        out[n] = (m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(X.iloc[te])[:, 1])
    return out


def learning_curve(X, y, seeds=5):
    res = {n: {s: [] for s in SIZES} for n in ("logistic regression", "random forest", "gradient boosting")}
    for seed in range(seeds):
        tr, va, te = splits(y, seed)
        for s in SIZES:
            sub = tr if s >= len(tr) else train_test_split(tr, train_size=s, stratify=y[tr], random_state=seed)[0]
            P = fit_predict(X, y, sub, va, te, names=list(res))
            for n, (_, pt) in P.items():
                res[n][s].append(roc_auc_score(y[te], pt))
        print("curve seed", seed, flush=True)
    return {n: {str(s): v for s, v in d.items()} for n, d in res.items()}


def reliability(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [[float(p[idx == b].mean()), float(y[idx == b].mean())] for b in range(bins) if (idx == b).any()]


def group_rates(y, flag, g):
    """Approval rate among good payers (y=0) and among defaulters (y=1) in group g."""
    approve = ~flag
    return {"approve_good": float(approve[g & (y == 0)].mean()), "approve_bad": float(approve[g & (y == 1)].mean()),
            "approve": float(approve[g].mean())}


def audit(X, y, female, young, seeds):
    rows = []
    for seed in range(seeds):
        tr, va, te = splits(y, seed)
        P = fit_predict(X, y, tr, va, te)
        yt = y[te]
        groups = {"young": young[te], "old": ~young[te], "female": female[te], "male": ~female[te]}
        row = {}
        for n, (pv, pt) in P.items():
            cell = {"rel": reliability(yt, pt), "auc": float(roc_auc_score(yt, pt)),
                    "frontier": frontier(y[va], pv, young[va], yt, pt, young[te])}
            for r in COST_RATIOS:
                cell[f"cost_r{r}"] = cost(yt, pt, best_threshold(y[va], pv, r), r)
            for gname, g in groups.items():
                cell[f"obs_{gname}"] = float(yt[g].mean())
                cell[f"pred_{gname}"] = float(pt[g].mean())
                cell[f"auc_{gname}"] = float(roc_auc_score(yt[g], pt[g]))
            for r in FINE_R:
                flag = pt >= best_threshold(y[va], pv, r)
                cell[f"pa_{r}"] = parity(flag, young[te])
                cell[f"ps_{r}"] = parity(flag, female[te])
                if r in GROUP_R:
                    for gname, g in groups.items():
                        cell[f"r{r}_{gname}"] = group_rates(yt, flag, g)
            row[n] = cell
        if seed == 0:
            row["delong"] = {n: delong(yt, P[n][1], P["logistic regression"][1]) for n in BASE[1:]}
        row["n_train"], row["n_test"] = len(tr), len(te)
        rows.append(row)
        print("audit seed", seed, flush=True)
    return rows


def proxy(X, young, seeds):
    """Predict under-25 from every input except age: AUC per seed and permutation importance on seed 0."""
    out = {"auc": {"logistic regression": [], "gradient boosting": []}}
    yy = young.astype(int)
    for seed in range(seeds):
        tr, _, te = splits(yy, seed)
        models = make_models(X)
        for n in out["auc"]:
            m = models[n].fit(X.iloc[tr], yy[tr])
            out["auc"][n].append(float(roc_auc_score(yy[te], m.predict_proba(X.iloc[te])[:, 1])))
            if seed == 0 and n == "gradient boosting":
                pi = permutation_importance(m, X.iloc[te], yy[te], scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=2)
                out["importance"] = sorted(zip(X.columns, pi.importances_mean.tolist()), key=lambda t: -t[1])[:6]
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"targets": TARGETS, "frontier_r": FRONTIER_R, "sizes": SIZES, "fine_r": FINE_R, "group_r": GROUP_R, "datasets": {}}
    Xt, yt, ft, yg = taiwan()
    res["learning_curve"] = learning_curve(Xt, yt)
    for name, loader, seeds in (("taiwan", taiwan, 20), ("german", german, 50)):
        X, y, female, young = loader()
        res["datasets"][name] = {"audit": audit(X, y, female, young, seeds),
                                 "proxy": proxy(X.drop(columns=[AGE_COL[name]]), young, 5 if name == "taiwan" else 20)}
        print(name, "done", round(time.time() - t0), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open("study_extra.json", "w"), indent=1)
