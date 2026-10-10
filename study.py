"""Paper study: when is a more complex credit model worth it? Two datasets, repeated splits, a sweep of cost ratios,
paired comparisons against logistic regression, and an approval-parity audit by sex and age. Writes study.json."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit.data import features as taiwan_features
from credit.metrics import ece

COST_RATIOS = (1, 2, 5, 10)  # cost of a missed defaulter relative to a wrongly declined good borrower
MODELS = ("logistic regression", "random forest", "gradient boosting", "MLP", "ensemble")


def taiwan():
    df = pd.read_csv("data.csv")
    X = taiwan_features(df)  # sex (x2) already excluded
    return X, df["target"].values, (df["x2"] == 2).values, (df["x5"] < 25).values  # female, under 25


def german():
    df = pd.read_csv("data_german.csv")
    female = df["personal_status"].str.startswith("female").values
    X = df.drop(columns=["target", "personal_status"])  # personal_status encodes sex: excluded (fair-lending practice)
    return X, df["target"].values, female, (df["age"] < 25).values


TUNE_LR_C = (0.01, 0.1, 1.0, 10.0)
TUNE_GB = [(lr, leaves) for lr in (0.03, 0.1) for leaves in (7, 15, 31)]


def make_models(X, lr_c=0.5, gb=(0.05, 15)):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    prep = lambda: make_column_transformer((StandardScaler(), num), (OneHotEncoder(handle_unknown="ignore"), cat))
    hgb = dict(max_iter=300, learning_rate=gb[0], max_leaf_nodes=gb[1], l2_regularization=1.0, random_state=0)
    return {"logistic regression": make_pipeline(prep(), LogisticRegression(max_iter=3000, C=lr_c)),
            "random forest": make_pipeline(prep(), RandomForestClassifier(400, min_samples_leaf=10, n_jobs=4, random_state=0)),
            "gradient boosting": make_pipeline(prep(), HistGradientBoostingClassifier(**hgb)),
            "MLP": make_pipeline(prep(), MLPClassifier((64, 32), alpha=1e-3, max_iter=500, random_state=0,
                                                          early_stopping=len(X) > 5000))}


def cost(y, p, t, r):
    pred = p >= t
    return float(((~pred) & (y == 1)).sum() * r + (pred & (y == 0)).sum()) / len(y)


def best_threshold(y, p, r):
    ts = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    return float(ts[np.argmin([cost(y, p, t, r) for t in ts])])


def parity(flag, group):
    """Approval-rate ratio, lower group / higher group (four-fifths rule compares it to 0.8)."""
    a, b = 1 - flag[group].mean(), 1 - flag[~group].mean()
    return float(min(a, b) / max(a, b)) if max(a, b) > 0 else None


def tune(X, y, tr, va):
    """Pick logistic regression's C and gradient boosting's learning rate / leaves by validation AUC."""
    auc = lambda m: roc_auc_score(y[va], m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1])
    lr_c = max(TUNE_LR_C, key=lambda c: auc(make_models(X, lr_c=c)["logistic regression"]))
    gb = max(TUNE_GB, key=lambda g: auc(make_models(X, gb=g)["gradient boosting"]))
    return dict(lr_c=lr_c, gb=gb)


def one_split(X, y, female, young, seed, tuned=False):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    P_va, P_te, fit_s = {}, {}, {}
    for name, m in make_models(X, **(tune(X, y, tr, va) if tuned else {})).items():
        t = time.perf_counter(); m.fit(X.iloc[tr], y[tr]); fit_s[name] = time.perf_counter() - t
        P_va[name], P_te[name] = m.predict_proba(X.iloc[va])[:, 1], m.predict_proba(X.iloc[te])[:, 1]
    for P in (P_va, P_te):
        P["ensemble"] = np.mean([P[k] for k in ("logistic regression", "gradient boosting", "MLP")], 0)
    out = {}
    for name in MODELS:
        p, yt = P_te[name], y[te]
        cell = {"auc": roc_auc_score(yt, p), "brier": brier_score_loss(yt, p), "ece": ece(yt, p), "fit_s": fit_s.get(name)}
        for r in COST_RATIOS:
            t = best_threshold(y[va], P_va[name], r)
            flag = p >= t
            cell[f"cost_r{r}"] = cost(yt, p, t, r)
            cell[f"decline_r{r}"] = float(flag.mean())
            cell[f"parity_sex_r{r}"] = parity(flag, female[te])
            cell[f"parity_age_r{r}"] = parity(flag, young[te])
        out[name] = cell
    return out


def summarise(runs):
    """Mean and 2.5/97.5 percentiles over splits; paired differences against logistic regression."""
    keys = [k for k in runs[0]["logistic regression"] if runs[0]["logistic regression"][k] is not None]
    s = {}
    for name in MODELS:
        s[name] = {}
        for k in keys:
            v = np.array([r[name][k] for r in runs if r[name][k] is not None], dtype=float)
            if not len(v):
                continue
            s[name][k] = {"mean": round(float(v.mean()), 4), "lo": round(float(np.percentile(v, 2.5)), 4), "hi": round(float(np.percentile(v, 97.5)), 4)}
        if name != "logistic regression":
            s[name]["vs_lr"] = {}
            for k in ["auc", "brier"] + [f"cost_r{r}" for r in COST_RATIOS]:
                d = np.array([r[name][k] - r["logistic regression"][k] for r in runs])
                s[name]["vs_lr"][k] = {"mean": round(float(d.mean()), 4), "lo": round(float(np.percentile(d, 2.5)), 4),
                                       "hi": round(float(np.percentile(d, 97.5)), 4), "share_better": round(float((d < 0).mean() if k != "auc" else (d > 0).mean()), 3)}
    return s


AGE_COL = {"taiwan": "x5", "german": "age"}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "no_age", "tuned"], default="base",
                    help="no_age: drop age from the inputs; tuned: pick key hyperparameters on validation data per split")
    variant = ap.parse_args().variant
    res = {"cost_ratios": COST_RATIOS, "variant": variant, "datasets": {}}
    for dname, loader, n_splits in (("taiwan", taiwan, 20), ("german", german, 50)):
        X, y, female, young = loader()
        if variant == "no_age":
            X = X.drop(columns=[AGE_COL[dname]])
        t0 = time.time()
        runs = [one_split(X, y, female, young, s, tuned=variant == "tuned") for s in range(n_splits)]
        res["datasets"][dname] = {"n": len(y), "default_rate": round(float(y.mean()), 4), "n_splits": n_splits,
                                  "female_share": round(float(female.mean()), 4), "young_share": round(float(young.mean()), 4),
                                  "summary": summarise(runs), "wall_s": round(time.time() - t0, 1)}
        print(dname, len(y), round(time.time() - t0), flush=True)
    json.dump(res, open("study.json" if variant == "base" else f"study_{variant}.json", "w"), indent=2)
