"""Credit-default model arena: accuracy, calibration, business cost, latency, fairness, and specialists vs global."""
import json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from credit.data import load, features, split, PROTECTED
from credit.metrics import score, best_threshold, bootstrap_auc_ci

df = load()
tr, va, te = split(df)
Xtr, Xva, Xte = features(tr), features(va), features(te)
ytr, yva, yte = tr["target"].values, va["target"].values, te["target"].values

models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
    "random forest": RandomForestClassifier(400, min_samples_leaf=10, n_jobs=4, random_state=0),
    "hist gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
    "MLP (64-32)": make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)),
}
P_va, P_te, res = {}, {}, {}
for name, m in models.items():
    t = time.perf_counter(); m.fit(Xtr, ytr); fit_s = time.perf_counter() - t
    P_va[name], P_te[name] = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    t = time.perf_counter(); m.predict_proba(Xte); inf = (time.perf_counter() - t) / len(Xte) * 1e6
    res[name] = dict(fit_seconds=round(fit_s, 1), inference_us_per_row=round(inf, 1))

# divide and conquer: average of the diverse members, weights fixed (no tuning on test)
ens = ["logistic regression", "hist gradient boosting", "MLP (64-32)"]
P_va["ensemble (LR+HGB+MLP)"] = np.mean([P_va[k] for k in ens], 0)
P_te["ensemble (LR+HGB+MLP)"] = np.mean([P_te[k] for k in ens], 0)
res["ensemble (LR+HGB+MLP)"] = dict(fit_seconds=sum(res[k]["fit_seconds"] for k in ens), inference_us_per_row=sum(res[k]["inference_us_per_row"] for k in ens))

# segment specialists: one HGB per delinquency segment vs the single global HGB
seg = lambda X: (X["max_delinq"] >= 1).values
spec = {s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
        .fit(Xtr[seg(Xtr) == s], ytr[seg(Xtr) == s]) for s in (True, False)}


def spec_proba(X):
    p = np.zeros(len(X)); s = seg(X)
    for k in (True, False):
        if (s == k).any():
            p[s == k] = spec[k].predict_proba(X[s == k])[:, 1]
    return p


P_va["segment specialists (HGB x2)"], P_te["segment specialists (HGB x2)"] = spec_proba(Xva), spec_proba(Xte)
res["segment specialists (HGB x2)"] = dict(fit_seconds=None, inference_us_per_row=None)

for name in P_te:
    th = best_threshold(yva, P_va[name])
    res[name].update(score(yte, P_te[name], th), auc_ci95=bootstrap_auc_ci(yte, P_te[name]))
    print(f"{name:32s} AUC {res[name]['auc']}  KS {res[name]['ks']}  ECE {res[name]['ece']}  cost {res[name]['cost_per_account']}", flush=True)

# fairness audit at each model's validation-chosen threshold (sex was NOT a feature)
sex = te[PROTECTED].values
audit = {}
for name in P_te:
    t = res[name]["threshold"]; flag = P_te[name] >= t
    rate = {int(s): float(flag[sex == s].mean()) for s in (1, 2)}          # share flagged as high risk (declined)
    appr = {s: 1 - r for s, r in rate.items()}
    audit[name] = dict(decline_rate_by_sex=rate, approval_ratio_min_over_max=round(min(appr.values()) / max(appr.values()), 4),
                       auc_by_sex={int(s): round(float(__import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(yte[sex == s], P_te[name][sex == s])), 4) for s in (1, 2)})
json.dump(dict(models=res, fairness=audit, n_train=len(tr), n_valid=len(va), n_test=len(te), default_rate=float(df["target"].mean())),
          open("results.json", "w"), indent=2)
np.savez("test_probs.npz", y=yte, **P_te)
