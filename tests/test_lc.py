import json

import numpy as np
import pandas as pd

import study_lc as L


def test_lending_club_cohort_filters_and_features(tmp_path, monkeypatch):
    base = {c: 1.0 for c in L.NUM}
    base.update(home_ownership="RENT", verification_status="Verified", purpose="car", application_type="Individual",
                earliest_cr_line="Jan-2003", emp_length="10+ years", sub_grade="B3")
    rows = [dict(base, term=" 36 months", issue_d="Jan-2013", loan_status="Fully Paid"),
            dict(base, term=" 36 months", issue_d="Jul-2014", loan_status="Charged Off", emp_length="< 1 year", sub_grade="G5"),
            dict(base, term=" 36 months", issue_d="Mar-2015", loan_status="Default", emp_length=np.nan, dti=np.nan),
            dict(base, term=" 60 months", issue_d="Jan-2013", loan_status="Fully Paid"),      # wrong term
            dict(base, term=" 36 months", issue_d="Jan-2017", loan_status="Fully Paid"),      # outside the issue years
            dict(base, term=" 36 months", issue_d="Jan-2015", loan_status="Current")]        # no final outcome
    pd.DataFrame(rows).to_csv(tmp_path / "raw.csv.gz", index=False)
    monkeypatch.setattr(L, "RAW", str(tmp_path / "raw.csv.gz"))
    monkeypatch.setattr(L, "COHORT", str(tmp_path / "cohort.csv.gz"))
    X, y, year, month, sub = L.cohort()
    assert y.tolist() == [0, 1, 1] and year.tolist() == [2013, 2014, 2015] and month.tolist() == [1, 7, 3]
    assert sub.tolist() == [8, 35, 8]                                    # B3 -> 8, G5 -> 35
    assert X["emp_years"].tolist() == [10.0, 0.0, 0.0] and X["emp_years_missing"].tolist() == [0, 0, 1]
    assert X["dti_missing"].tolist() == [0, 0, 1] and not X.isna().any().any()
    assert abs(X["credit_history_years"].iloc[0] - 10.0) < 0.01
    assert not {"sub_grade", "int_rate", "grade", "installment", "loan_status"} & set(X.columns)
    assert json.load(open(tmp_path / "cohort_counts.json")) == {"n_36m_in_years": 4, "n_final_status": 3}


def test_fast_delong_matches_the_pairwise_version_including_ties():
    from study_extra import delong as slow
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1500)
    p1 = np.round(y * 0.3 + rng.random(1500), 2)          # rounding creates ties
    p2 = np.round(y * 0.1 + rng.random(1500), 2)
    for a, b in zip(L.delong(y, p1, p2), slow(y, p1, p2)):
        assert abs(a - b) < 1e-9
