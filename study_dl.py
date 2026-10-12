"""Modern-model extension for the paper: does a newer gradient-boosting library (XGBoost) or a tabular
deep model (FT-Transformer, Gorishniy et al. 2021) change the "is the complex model worth it" verdict
against HistGradientBoosting and logistic regression? Same preprocessing and splits for every model.

Writes study_dl.json. Reuses the portfolio loaders from study.py / study_gmsc.py.
Run: python3 study_dl.py [--repeats 10] [--gmsc-cap 50000]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from xgboost import XGBClassifier

import study
import study_gmsc

torch.manual_seed(0)
DEVICE = "cpu"  # CPU only: keep off the GPU so nothing else contends, and keep it deterministic


def prep(X):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return make_column_transformer((StandardScaler(), num), (OneHotEncoder(handle_unknown="ignore"), cat))


class FTTransformer(nn.Module):
    """Compact feature-tokenizer transformer: each scalar feature -> a d-dim token, a [CLS] token, a small encoder, CLS
    -> logit. A faithful-but-small version of Gorishniy et al. (2021).
    """

    def __init__(self, n_features, d=32, layers=2, heads=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b  # (B, F, d)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], 1)
        return self.head(self.enc(tok)[:, 0]).squeeze(-1)  # CLS -> logit


def fit_ftt(Xtr, ytr, Xva, yva, epochs=60, patience=8):
    m = FTTransformer(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    n = len(Xtr_t)
    for _ in range(epochs):
        m.train()
        for i in torch.randperm(n).split(256):
            opt.zero_grad()
            lossf(m(Xtr_t[i]), ytr_t[i]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(m(Xva_t)).numpy())
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m


def proba_models(Xtr, ytr, Xva, yva, Xte, pre):
    """Fit every model on the same preprocessed matrices; return test-set probabilities."""
    A = pre.fit_transform(Xtr)
    A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    Va = pre.transform(Xva)
    Va = Va.toarray() if hasattr(Va, "toarray") else np.asarray(Va)
    Te = pre.transform(Xte)
    Te = Te.toarray() if hasattr(Te, "toarray") else np.asarray(Te)
    out, fit_s = {}, {}
    sk = {
        "logistic regression": LogisticRegression(max_iter=3000, C=0.5),
        "HistGBM": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=0,
            n_jobs=4,
        ),
    }
    for name, clf in sk.items():
        t = time.perf_counter()
        clf.fit(A, ytr)
        fit_s[name] = time.perf_counter() - t
        out[name] = clf.predict_proba(Te)[:, 1]
    t = time.perf_counter()
    ftt = fit_ftt(A, ytr, Va, yva)
    fit_s["FT-Transformer"] = time.perf_counter() - t
    with torch.no_grad():
        out["FT-Transformer"] = torch.sigmoid(ftt(torch.tensor(Te, dtype=torch.float32))).numpy()
    return out, fit_s


def corrected_t(diffs, rho):
    """Nadeau-Bengio corrected resampled t-test over repeated splits; rho = n_test / n_train."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n < 2:
        return None
    var = d.var(ddof=1)
    if var == 0:
        return 0.0
    from scipy import stats

    t = d.mean() / np.sqrt(var * (1.0 / n + rho))
    return float(2 * stats.t.sf(abs(t), n - 1))


def run(name, X, y, repeats, test_frac=0.4):
    rows = {m: [] for m in ("logistic regression", "HistGBM", "XGBoost", "FT-Transformer")}
    fits = {m: [] for m in rows}
    for s in range(repeats):
        tr, rest = train_test_split(np.arange(len(y)), test_size=test_frac, stratify=y, random_state=s)
        va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=s)
        pre = prep(X)
        probs, fit_s = proba_models(X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], pre)
        for m, p in probs.items():
            rows[m].append(roc_auc_score(y[te], p))
            fits[m].append(fit_s[m])
        print(f"  {name} split {s}: " + ", ".join(f"{m} {rows[m][-1]:.4f}" for m in rows), flush=True)
    res = {}
    base = np.array(rows["HistGBM"])
    for m, aucs in rows.items():
        a = np.array(aucs)
        res[m] = {
            "auc_mean": float(a.mean()),
            "auc_lo": float(np.percentile(a, 2.5)),
            "auc_hi": float(np.percentile(a, 97.5)),
            "fit_s_mean": float(np.mean(fits[m])),
            "vs_histgbm_mean_diff": float((a - base).mean()),
            "vs_histgbm_corrected_p": corrected_t(a - base, (test_frac * 0.5) / (1 - test_frac))
            if m != "HistGBM"
            else None,
        }
    return {"n": len(y), "pos_rate": float(y.mean()), "repeats": repeats, "models": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--gmsc-cap", type=int, default=50000)
    a = ap.parse_args()
    out = {
        "note": "Same preprocessing and splits for every model; FT-Transformer is a compact tabular transformer "
        "(Gorishniy et al. 2021) trained on CPU with early stopping. AUC is test-set, repeated stratified splits.",
        "portfolios": {},
    }
    Xt, yt, _, _ = study.taiwan()
    print("Taiwan", Xt.shape)
    out["portfolios"]["taiwan"] = run("taiwan", Xt, yt, a.repeats)
    dfg, yg, _ = study_gmsc.gmsc()
    Xg = dfg.drop(columns=[c for c in dfg.columns if c.lower() in ("target", "y", "seriousdlqin2yrs")], errors="ignore")
    if len(Xg) > a.gmsc_cap:
        idx = np.random.RandomState(0).choice(len(Xg), a.gmsc_cap, replace=False)
        Xg, yg = Xg.iloc[idx].reset_index(drop=True), yg[idx]
    print("GMSC", Xg.shape)
    out["portfolios"]["gmsc"] = run("gmsc", Xg, yg, a.repeats)
    json.dump(out, open("study_dl.json", "w"), indent=1)
    print("wrote study_dl.json")


if __name__ == "__main__":
    main()
