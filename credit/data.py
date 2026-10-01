"""UCI 'Default of Credit Card Clients' (30k accounts, 22% default) with engineered features."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROTECTED = "x2"  # sex: excluded from features (fair lending), kept for the post-hoc audit only


def load(path="data.csv"):
    return pd.read_csv(path)


def features(df):
    X = df.drop(columns=["target", PROTECTED]).copy()
    bills, pays = [f"x{i}" for i in range(12, 18)], [f"x{i}" for i in range(18, 24)]
    X["util"] = df[bills].mean(axis=1) / df["x1"].clip(lower=1)
    X["pay_ratio"] = df[pays].sum(axis=1) / df[bills].sum(axis=1).clip(lower=1)
    X["max_delinq"] = df[[f"x{i}" for i in range(6, 12)]].max(axis=1)
    X["n_late"] = (df[[f"x{i}" for i in range(6, 12)]] > 0).sum(axis=1)
    return X


def split(df, seed=0):
    """60 / 20 / 20 stratified train / validation (threshold + stacking) / test."""
    tr, rest = train_test_split(df, test_size=0.4, stratify=df["target"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["target"], random_state=seed)
    return tr, va, te
