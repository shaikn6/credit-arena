"""Fourth portfolio for the paper: Lending Club 36-month loans issued 2012-2015 with a final outcome (Kaggle dataset
wordsforthewise/lending-club, CC0). Same models and metrics as study.py, with three designs: repeated random splits,
an out-of-time test (train on earlier issue years, test on later ones) and a learning curve. Writes study_lc.json.

Only fields known when the loan is issued are used. The lender's own grade, sub-grade, interest rate and instalment
are excluded from the main runs because they are the output of the lender's risk model; one variant adds the
sub-grade to ask whether a complex model still helps on top of the lender's own score."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from credit.metrics import ece
from study import COST_RATIOS, MODELS, best_threshold, cost, make_models

RAW, COHORT = "data_lc/accepted_2007_to_2018Q4.csv.gz", "data_lc/cohort_36m_2012_2015.csv.gz"
YEARS = (2012, 2013, 2014, 2015)
BAD, GOOD = ("Charged Off", "Default"), ("Fully Paid",)
NUM = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
       "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies"]
CAT = ["home_ownership", "verification_status", "purpose", "application_type"]
LOG = ["annual_inc", "revol_bal", "loan_amnt"]
N_SPLITS, CURVE_SEEDS = 10, 3
SIZES = (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 350000)
RANDOM_KEYS = ["auc", "brier", "ece"] + [f"cost_r{r}" for r in COST_RATIOS]


def midrank(x):
    """Ranks of x (1-based), with ties given the mean of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    xs, r = x[order], np.empty(len(x))
    starts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(x)]
    for a_, b_ in zip(starts[:-1], starts[1:]):
        r[order[a_:b_]] = 0.5 * (a_ + b_ - 1) + 1
    return r


def delong(y, p1, p2):
    """DeLong et al. (1988) test for two correlated AUCs, computed from mid-ranks in O(n log n) (Sun and Xu, 2014), so it
    runs on hundreds of thousands of loans. Same result as study_extra.delong: (auc1 - auc2, z, two-sided p)."""
    from scipy.stats import norm
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = midrank(p[pos]), midrank(p[neg]), midrank(np.r_[p[pos], p[neg]])
        aucs.append(tz[:m].sum() / m / n - (m + 1) / (2 * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    d = aucs[0] - aucs[1]
    z = d / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return float(d), float(z), float(2 * norm.sf(abs(z)))


def counts_path():
    return os.path.join(os.path.dirname(COHORT), "cohort_counts.json")


def cohort():
    """Features (issue-time fields only), default flag, issue year, and the lender's sub-grade as an ordinal (1 = A1 ... 35 = G5)."""
    if not os.path.exists(COHORT):
        use = NUM + CAT + ["term", "issue_d", "loan_status", "earliest_cr_line", "emp_length", "sub_grade"]
        df = pd.read_csv(RAW, usecols=use, low_memory=False)
        issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        keep = (df["term"].str.strip() == "36 months") & issue.dt.year.isin(YEARS) & df["loan_status"].isin(BAD + GOOD)
        n_term_years = int(((df["term"].str.strip() == "36 months") & issue.dt.year.isin(YEARS)).sum())
        df, issue = df[keep].copy(), issue[keep]
        df["issue_year"], df["issue_month"] = issue.dt.year.astype(int), issue.dt.month.astype(int)
        df["target"] = df["loan_status"].isin(BAD).astype(int)
        first = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["credit_history_years"] = ((issue - first).dt.days / 365.25).round(2)
        emp = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
        df["emp_years"] = np.where(df["emp_length"].str.startswith("<", na=False), 0.0, emp)
        df["sub_grade_num"] = df["sub_grade"].map(lambda s: (ord(s[0]) - 65) * 5 + int(s[1]))
        cols = NUM + CAT + ["credit_history_years", "emp_years", "sub_grade_num", "issue_year", "issue_month", "target"]
        df[cols].to_csv(COHORT, index=False)
        json.dump({"n_36m_in_years": n_term_years, "n_final_status": int(len(df))}, open(counts_path(), "w"))
    df = pd.read_csv(COHORT)
    y, year, month, sub = df.pop("target").values, df.pop("issue_year").values, df.pop("issue_month").values, df.pop("sub_grade_num").values
    for c in [c for c in df.columns if c not in CAT]:
        if df[c].isna().any():  # nothing is estimated from the data, so nothing leaks across splits or years
            df[c + "_missing"] = df[c].isna().astype(int)
            df[c] = df[c].fillna(0)
    for c in LOG:
        df[c] = np.log1p(df[c])
    return df, y, year, month, sub


def evaluate(X, y, tr, va, tests):
    """Fit the five models on tr, choose cost thresholds on va, score every test set in `tests` (name -> indices)."""
    P_va, P_te, fit_s = {}, {n: {} for n in tests}, {}
    for name, m in make_models(X).items():
        t = time.perf_counter(); m.fit(X.iloc[tr], y[tr]); fit_s[name] = time.perf_counter() - t
        P_va[name] = m.predict_proba(X.iloc[va])[:, 1]
        for n, te in tests.items():
            P_te[n][name] = m.predict_proba(X.iloc[te])[:, 1]
    P_va["ensemble"] = np.mean([P_va[k] for k in ("logistic regression", "gradient boosting", "MLP")], 0)
    out = {}
    for n, te in tests.items():
        P_te[n]["ensemble"] = np.mean([P_te[n][k] for k in ("logistic regression", "gradient boosting", "MLP")], 0)
        yt, cell = y[te], {}
        for name in MODELS:
            p = P_te[n][name]
            c = {"auc": roc_auc_score(yt, p), "brier": brier_score_loss(yt, p), "ece": ece(yt, p), "mean_pred": float(p.mean()),
                 "fit_s": fit_s.get(name)}
            for r in COST_RATIOS:
                t = best_threshold(y[va], P_va[name], r)
                c[f"cost_r{r}"], c[f"decline_r{r}"] = cost(yt, p, t, r), float((p >= t).mean())
            cell[name] = c
        d, z, pv = delong(yt, P_te[n]["gradient boosting"], P_te[n]["logistic regression"])
        out[n] = {"n": int(len(te)), "default_rate": float(yt.mean()), "models": cell,
                  "delong_gb_vs_lr": {"diff": float(d), "z": float(z), "p": float(pv)}}
    return out


def random_split(X, y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return evaluate(X, y, tr, va, {"test": te})["test"]["models"]


def out_of_time(X, y, year, month):
    """Two designs. A: train 2012-2013, thresholds on the first half of 2014, tests on the second half of 2014 and on 2015.
    B: the same amount of 2012-2015 data split at random, as the matched random-split comparison."""
    idx = np.arange(len(y))
    tr = idx[year <= 2013]
    va = idx[(year == 2014) & (month <= 6)]
    res = {"design": evaluate(X, y, tr, va, {"2014H2": idx[(year == 2014) & (month > 6)], "2015": idx[year == 2015]}),
           "n_train": int(len(tr)), "n_val": int(len(va)), "train_default_rate": float(y[tr].mean())}
    rng = np.random.RandomState(0)
    perm = rng.permutation(idx)
    rtr, rva, rte = perm[:len(tr)], perm[len(tr):len(tr) + len(va)], perm[len(tr) + len(va):]
    res["matched_random"] = evaluate(X, y, rtr, rva, {"random": rte})
    return res


def curve_seed(X, y, seed):
    """Test AUC of three models at every training size, for one seed."""
    names = ("logistic regression", "random forest", "gradient boosting")
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)[1]
    out = {n: [] for n in names}
    for s in SIZES:
        sub = tr if s >= len(tr) else train_test_split(tr, train_size=s, stratify=y[tr], random_state=seed)[0]
        models = make_models(X)
        for n in names:
            out[n].append(roc_auc_score(y[te], models[n].fit(X.iloc[sub], y[sub]).predict_proba(X.iloc[te])[:, 1]))
    return out


def save(res):
    json.dump(res, open("study_lc.json", "w"), indent=1)


if __name__ == "__main__":
    # Resumable: every finished piece is saved, and --budget SECONDS stops cleanly so the study can be run in several sittings.
    import sys
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else float("inf")
    t0 = time.time()
    over = lambda: time.time() - t0 > budget
    X, y, year, month, sub = cohort()
    res = json.load(open("study_lc.json")) if os.path.exists("study_lc.json") else {}
    res.update({"dataset": "lending club 36-month loans issued 2012-2015", "source": "kaggle wordsforthewise/lending-club",
                **json.load(open(counts_path())), "n": int(len(y)), "default_rate": round(float(y.mean()), 4),
                "n_features": int(X.shape[1]),
                "by_year": {str(k): {"n": int((year == k).sum()), "default_rate": round(float(y[year == k].mean()), 4)} for k in YEARS},
                "cost_ratios": COST_RATIOS, "sizes": SIZES, "curve_seeds": CURVE_SEEDS, "n_splits": N_SPLITS})
    if "out_of_time" not in res:
        res["out_of_time"] = out_of_time(X, y, year, month); save(res)
        print("out-of-time done", round(time.time() - t0), flush=True)
    if "out_of_time_with_grade" not in res and not over():
        res["out_of_time_with_grade"] = out_of_time(X.assign(sub_grade_num=sub), y, year, month); save(res)
        print("out-of-time with grade done", round(time.time() - t0), flush=True)
    runs = res.setdefault("runs", [])
    while len(runs) < N_SPLITS and not over():
        runs.append(random_split(X, y, len(runs))); save(res)
        print("split", len(runs) - 1, round(time.time() - t0), flush=True)
    curves = res.setdefault("curves", [])
    while len(runs) == N_SPLITS and len(curves) < CURVE_SEEDS and not over():
        curves.append(curve_seed(X, y, len(curves))); save(res)
        print("curve seed", len(curves) - 1, round(time.time() - t0), flush=True)
    if len(runs) == N_SPLITS and len(curves) == CURVE_SEEDS:
        res["per_split"] = {m: {k: [r[m][k] for r in runs] for k in RANDOM_KEYS} for m in MODELS}
        res["learning_curve"] = {n: {str(s): [c[n][i] for c in curves] for i, s in enumerate(SIZES)} for n in curves[0]}
        res["complete"] = True
        save(res)
        print("done", flush=True)
