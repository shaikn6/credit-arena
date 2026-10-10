import numpy as np
from sklearn.metrics import roc_auc_score

from study_extra import TARGETS, delong, frontier


def test_delong_difference_matches_auc_and_detects_gap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p1, p2 = y * 0.5 + rng.random(1000), y * 0.05 + rng.random(1000)
    d, z, p = delong(y, p1, p2)
    assert abs(d - (roc_auc_score(y, p1) - roc_auc_score(y, p2))) < 1e-9
    assert z > 3 and p < 0.01
    d2, z2, p2_ = delong(y, p2, p1)
    assert abs(d2 + d) < 1e-12 and abs(z2 + z) < 1e-9 and abs(p2_ - p) < 1e-12


def test_frontier_meets_target_on_validation_and_costs_more():
    rng = np.random.default_rng(1)
    n = 3000
    g = rng.random(n) < 0.3
    y = (rng.random(n) < np.where(g, 0.35, 0.15)).astype(int)
    p = np.clip(0.2 + 0.3 * y + 0.1 * g + 0.2 * rng.random(n), 0, 1)
    out = frontier(y, p, g, y, p, g)  # validation == test: the constraint must hold exactly
    assert len(out) == len(TARGETS)
    for tgt, f in zip(TARGETS, out):
        if f is not None:
            assert f["ratio"] >= tgt - 1e-12 and f["cost"] >= out[0]["cost"] - 1e-12


def test_missing_data_file_names_the_download_step(tmp_path):
    import pytest
    from study import read
    with pytest.raises(SystemExit, match="make data"):
        read(str(tmp_path / "data_german.csv"))


def test_german_rows_relabels_codes_and_target():
    from download_data import GERMAN_COLUMNS, german_rows
    raw = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1\n" \
          "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2\n"
    header, a, b = german_rows(raw)
    assert header == list(GERMAN_COLUMNS) and len(a) == len(b) == 21
    row = dict(zip(header, b))
    assert row["personal_status"] == "female div/dep/mar" and row["age"] == "22" and row["target"] == "1" and a[-1] == "0"
