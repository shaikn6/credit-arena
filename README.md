# credit-arena

Which model should score credit-card default risk, and when is a bigger or more complex one worth it? Six
model families on the UCI *Default of Credit Card Clients* data (30,000 accounts, 22% default), compared on
discrimination, calibration, **business cost**, latency and **fair-lending** behaviour, with paired
bootstrap tests so that "better" means statistically better.

![results](results.png)

## Setup

- Split 60 / 20 / 20 (train / validation / test), stratified, seed 0. Test is touched once.
- **Sex is excluded from the features** (fair-lending practice) and used only in a post-hoc audit.
- Engineered features: credit utilisation, repayment ratio, worst delinquency, count of late months.
- Decision threshold is chosen on the *validation* set to minimise expected cost, with a missed defaulter
  costing 5x a wrongly refused good borrower. Cost is reported per account on the test set.
- "Divide and conquer" is tested directly: an averaged ensemble of diverse models, and two specialist
  gradient-boosting models (one for accounts with any delinquency, one for the rest) vs a single global model.

## Results (6,000 held-out accounts)

| Model | AUC [95% CI] | KS | Brier | ECE | Cost / account |
|---|---|---|---|---|---|
| Logistic regression | 0.746 [0.731, 0.760] | 0.387 | 0.143 | 0.009 | 0.609 |
| Random forest | 0.772 [0.759, 0.786] | 0.407 | 0.137 | 0.011 | 0.569 |
| **Hist. gradient boosting** | **0.774 [0.759, 0.788]** | 0.414 | 0.137 | 0.012 | 0.562 |
| MLP (64-32) | 0.760 [0.746, 0.776] | 0.393 | 0.140 | 0.018 | 0.587 |
| Ensemble (LR + HGB + MLP) | 0.769 [0.756, 0.784] | 0.401 | 0.139 | 0.013 | 0.562 |
| Segment specialists (HGB x2) | 0.770 [0.756, 0.784] | 0.412 | 0.138 | 0.019 | 0.561 |

Full metrics (Brier, log-loss, PR-AUC, thresholds) and the paired bootstrap comparisons: `results.json`, written by
`make run`. The run is seeded, so re-running reproduces the file exactly.

Fit time and per-row inference time (median of 5 passes over the test set) are written to `timings.json`, which is
not committed: they depend on the machine and on whatever else it is running, and they are kept out of `results.json`
so that file stays identical between runs. No timing figures are quoted here; run `make run` on an idle machine and
read `timings.json` (it also records `inference_ratio_hgb_over_lr`).

## What this says about model choice

- **Non-linear models beat logistic regression, and it is significant:** gradient boosting minus logistic
  regression = +0.0275 AUC (paired bootstrap 95% CI [0.0187, 0.0369]); expected cost drops about 8%.
- **Nothing beats gradient boosting significantly.** Its lead over the random forest (+0.0015, CI [-0.0038, 0.0067]),
  the ensemble (+0.0040, CI [-0.0006, 0.0086]) and the segment specialists (+0.0036, CI [-0.0006, 0.0082]) has
  intervals that include zero. On this data, stacking more models or splitting into segment specialists bought no
  measurable accuracy; it only added complexity and (for the ensemble) latency.
- The paired bootstrap resamples the 6,000 test accounts with replacement 2,000 times (seed 0), scores both models on
  the same resample and takes the 2.5 / 97.5 percentiles of the AUC difference (`paired_bootstrap_auc_diff` in
  `credit/metrics.py`). The numbers above are the `paired_bootstrap_auc` block of `results.json`. It measures test-set
  sampling noise for one fitted pair of models, not variation across training sets or split seeds.
- **When to use logistic regression anyway:** it is faster at inference than gradient boosting (see `timings.json`), best calibrated in practice
  (ECE 0.009), and the most interpretable. If a regulator needs reason codes, the cost of ~0.03 AUC is the price.
- **The MLP is dominated on accuracy:** gradient boosting leads it by +0.0134 AUC (CI [0.0060, 0.0208]) and it is
  worse calibrated.

## Fair-lending check

Sex was not a model input, yet decline rates still differ because the inputs correlate with it. Approval-rate
ratio (lower group / higher group) at each model's cost-optimal threshold, against the 0.80 four-fifths rule:

| Model | Approval ratio |
|---|---|
| Logistic regression | 0.96 |
| MLP | 0.92 |
| Segment specialists | 0.92 |
| Hist. gradient boosting | 0.89 |
| Random forest | 0.89 |
| Ensemble | 0.89 |

All pass, but the most accurate models sit closest to the line: a real accuracy-vs-parity trade-off. AUC by sex differs by
0.0001-0.020 depending on the model (largest for logistic regression). This is a screening check, not a full disparate-impact analysis.

## Caveats

- One dataset (Taiwan, 2005), one split seed. The ranking among the top models is not stable enough to claim;
  the gap to logistic regression is.
- At a 5:1 cost ratio and 22% base rate the cost-optimal threshold flags 45-55% of accounts for decline, which
  reflects the cost assumption, not a recommended policy.
- Models use near-default hyperparameters and were not exhaustively tuned; a tuned gradient-boosting library
  (XGBoost / LightGBM) may widen the gap.

## Repeated-split study

`run.py` uses one split of one dataset. `study.py` repeats the comparison over 20 splits of the Taiwan data and 50
splits of the German credit data (1,000 loans, 30% bad), sweeps the cost ratio over 1, 2, 5 and 10, and audits approval
parity by sex and by age (under 25). The committed outputs:

| File | Command | Contents |
|---|---|---|
| `study.json` | `python study.py` (`make study`, about 7 minutes) | per-model mean and 2.5 / 97.5 percentiles over splits, paired differences against logistic regression |
| `study_no_age.json` | `python study.py --variant no_age` | the same with age removed from the inputs |
| `study_tuned.json` | `python study.py --variant tuned` (about 45 minutes) | the same with logistic regression's C and gradient boosting's learning rate / leaves picked on validation data per split |
| `study_extra.json` | `python study_extra.py` (about 20 minutes) | learning curve, reliability bins, group error rates, fine parity sweep, age-proxy analysis, per-split results with a DeLong test on split 0, accuracy-parity frontier |

In `study.json`, gradient boosting minus logistic regression averages +0.0311 AUC on the Taiwan data (2.5 / 97.5
percentiles over splits 0.0241 to 0.0392, ahead in every split) and -0.0120 on the German data (-0.0508 to 0.0248, ahead in
38% of splits): on the small portfolio the more complex model is not better. The `fit_s` and `wall_s` fields in these
files are wall-clock times and change from run to run; every other field is seeded.

## Data

Neither dataset is stored in this repository; `make data` downloads both and checks SHA-256 checksums.

- **Default of Credit Card Clients** (Taiwan, 2005; 30,000 accounts). Yeh, I-C. (2009). *Default of Credit Card
  Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licensed CC BY 4.0. Introduced
  in: Yeh, I-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of
  probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480. Downloaded
  through the OpenML mirror (data id 42477), which names the columns `x1`..`x23`.
- **Statlog (German Credit Data)** (1,000 loans). Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5NC77. Licensed CC BY 4.0. Downloaded from the UCI archive;
  `download_data.py` replaces the coded values (`A11`, `A12`, ...) with readable labels and recodes the target to
  1 = bad.

## Intended use

A research comparison of model families on two public datasets: how much accuracy, calibration, cost and approval
parity change with model complexity. It is not a production credit model and should not be used to make or support
lending decisions.

## Limitations

- Both datasets are old and small by industry standards (Taiwan 2005; the German data was donated in 1994), each from a single lender, and
  contain only accepted applicants, so nothing here speaks to reject inference or to current portfolios.
- The German data has known coding problems documented by Groemping (2019, "South German Credit Data: Correcting a
  Widely Used Data Set"); this repository uses the original UCI file as distributed.
- The fairness checks are screening statistics (approval-rate ratios, AUC and error rates by group) on sex and age
  only. Passing the four-fifths rule here is not a disparate-impact analysis and not evidence of legal compliance;
  excluding a protected attribute does not remove its proxies.
- The cost ratios are assumptions, not estimates from a real portfolio.
- See also the caveats above: one split in the main table, near-default hyperparameters.

## Run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data    # download data.csv and data_german.csv, verify checksums
make run     # results.json, test_probs.npz, timings.json (about 1 minute)
make test    # ruff check + pytest
make study   # study.json (about 7 minutes)
```

Tested on Python 3.12 with the pinned versions in `requirements.txt`. `study.py` and `study_extra.py` stop with a
message pointing to `make data` if a data file is missing.
