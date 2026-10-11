"""Third portfolio for the paper: Give Me Some Credit (150,000 accounts, OpenML 46929, Kaggle 2011). Same models, split
scheme and metrics as study.py, plus a learning curve that runs up to the full training set. Writes study_gmsc.json."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from study import COST_RATIOS, MODELS, one_split, summarise
from study_extra import fit_predict, splits

SIZES = (500, 1000, 2000, 5000, 10000, 18000, 45000, 90000)
N_SPLITS, CURVE_SEEDS = 20, 5
PAST_DUE = ("NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate")
LOG_COLS = ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome") + PAST_DUE  # heavy right tails
PER_SPLIT = ["auc", "brier", "ece"] + [f"cost_r{r}" for r in COST_RATIOS] + [f"parity_age_r{r}" for r in COST_RATIOS]


def gmsc(path="data_gmsc.csv"):
    """Features, default flag and the under-25 flag. The data has no sex attribute. Missing income and dependents become
    0 plus an indicator column, and the 96/98 codes in the past-due counts become 0 plus an indicator (no statistic is
    estimated, so nothing leaks across splits); log1p is monotone, so it changes nothing for the tree models and gives
    the linear model and the network a workable scale."""
    if not os.path.exists(path):
        from sklearn.datasets import fetch_openml
        fetch_openml(data_id=46929, as_frame=True, parser="auto").frame.to_csv(path, index=False)
    df = pd.read_csv(path)
    y = (df.pop("FinancialDistressNextTwoYears") == "Yes").values.astype(int)
    for c in ("MonthlyIncome", "NumberOfDependents"):
        df[c + "_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0)
    coded = (df[list(PAST_DUE)] >= 96).any(axis=1)  # 96 and 98 are codes in the source data, not counts
    df["past_due_coded"] = coded.astype(int)
    df.loc[coded, list(PAST_DUE)] = 0
    for c in LOG_COLS:
        df[c] = np.log1p(df[c])
    return df, y, (df["age"] < 25).values


def learning_curve(X, y):
    res = {n: {s: [] for s in SIZES} for n in ("logistic regression", "random forest", "gradient boosting")}
    for seed in range(CURVE_SEEDS):
        tr, va, te = splits(y, seed)
        for s in SIZES:
            sub = tr if s >= len(tr) else train_test_split(tr, train_size=s, stratify=y[tr], random_state=seed)[0]
            for n, (_, pt) in fit_predict(X, y, sub, va, te, names=list(res)).items():
                res[n][s].append(roc_auc_score(y[te], pt))
        print("curve seed", seed, flush=True)
    return {n: {str(s): v for s, v in d.items()} for n, d in res.items()}


if __name__ == "__main__":
    X, y, young = gmsc()
    t0 = time.time()
    runs = []
    for s in range(N_SPLITS):
        r = one_split(X, y, young, young, s)  # no sex attribute: the sex-parity cells are dropped below
        runs.append({m: {k: v for k, v in c.items() if "parity_sex" not in k} for m, c in r.items()})
        print("split", s, round(time.time() - t0), flush=True)
    res = {"dataset": "gmsc", "openml_id": 46929, "n": len(y), "default_rate": round(float(y.mean()), 4),
           "young_share": round(float(young.mean()), 4), "n_young": int(young.sum()), "n_splits": N_SPLITS,
           "n_features": X.shape[1], "cost_ratios": COST_RATIOS, "sizes": SIZES, "curve_seeds": CURVE_SEEDS,
           "summary": summarise(runs),
           "per_split": {m: {k: [r[m][k] for r in runs] for k in PER_SPLIT} for m in MODELS},
           "wall_s_main": round(time.time() - t0, 1)}
    json.dump(res, open("study_gmsc.json", "w"), indent=2)
    res["learning_curve"] = learning_curve(X, y)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open("study_gmsc.json", "w"), indent=2)
    print("done", res["wall_s"], flush=True)
