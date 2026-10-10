"""Download the two public datasets and verify them. Writes data.csv (Taiwan) and data_german.csv (German).

Taiwan: UCI 'Default of Credit Card Clients', fetched through its OpenML mirror (data_id 42477, columns x1..x23).
German: UCI 'Statlog (German Credit Data)', fetched from the UCI archive; the coded values (A11, A12, ...) are
replaced with the readable labels documented in the archive's german.doc, and the target is recoded to 1 = bad.
"""
import csv
import hashlib
import io
import urllib.request
import zipfile

GERMAN_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
SHA256 = {
    "german.zip": "e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b",
    "data_german.csv": "48a363675e091f451837fd5b00b9025005f6ed5148fcca429d27d132313c44c4",
    "data.csv": "a769ca8f15b6b382274e0c5be595e19aa6e2c12e7aeeb6c095ea17fb618a7120",
}

# column name -> {UCI code: label}, in the column order of german.data; None marks a numeric column
GERMAN_COLUMNS = {
    "checking_status": {"A11": "<0", "A12": "0<=X<200", "A13": ">=200", "A14": "no checking"},
    "duration": None,
    "credit_history": {"A30": "no credits/all paid", "A31": "all paid", "A32": "existing paid", "A33": "delayed previously",
                       "A34": "critical/other existing credit"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/tv", "A44": "domestic appliance",
                "A45": "repairs", "A46": "education", "A48": "retraining", "A49": "business", "A410": "other"},
    "credit_amount": None,
    "savings_status": {"A61": "<100", "A62": "100<=X<500", "A63": "500<=X<1000", "A64": ">=1000", "A65": "no known savings"},
    "employment": {"A71": "unemployed", "A72": "<1", "A73": "1<=X<4", "A74": "4<=X<7", "A75": ">=7"},
    "installment_commitment": None,
    "personal_status": {"A91": "male div/sep", "A92": "female div/dep/mar", "A93": "male single", "A94": "male mar/wid"},
    "other_parties": {"A101": "none", "A102": "co applicant", "A103": "guarantor"},
    "residence_since": None,
    "property_magnitude": {"A121": "real estate", "A122": "life insurance", "A123": "car", "A124": "no known property"},
    "age": None,
    "other_payment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "existing_credits": None,
    "job": {"A171": "unemp/unskilled non res", "A172": "unskilled resident", "A173": "skilled", "A174": "high qualif/self emp/mgmt"},
    "num_dependents": None,
    "own_telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
    "target": {"1": "0", "2": "1"},  # UCI: 1 = good, 2 = bad
}


def check(name, data):
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256[name]:
        raise SystemExit(f"{name}: sha256 {got} does not match the expected {SHA256[name]}")


def german_rows(raw):
    """Turn the lines of german.data into a header plus labelled rows."""
    maps = list(GERMAN_COLUMNS.values())
    rows = [list(GERMAN_COLUMNS)]
    for line in raw.splitlines():
        if line.strip():
            rows.append([v if m is None else m[v] for v, m in zip(line.split(), maps, strict=True)])
    return rows


def german(path="data_german.csv"):
    with urllib.request.urlopen(GERMAN_URL, timeout=60) as r:
        blob = r.read()
    check("german.zip", blob)
    raw = zipfile.ZipFile(io.BytesIO(blob)).read("german.data").decode("ascii")
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(german_rows(raw))
    check("data_german.csv", out.getvalue().encode())
    open(path, "w", newline="").write(out.getvalue())
    print(f"wrote {path} (sha256 ok)")


def taiwan(path="data.csv"):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    data = d.data.assign(target=d.target).to_csv(index=False).encode()
    check("data.csv", data)
    open(path, "wb").write(data)
    print(f"wrote {path} (sha256 ok)")


if __name__ == "__main__":
    taiwan()
    german()
