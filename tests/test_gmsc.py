import numpy as np
import pandas as pd

from study_gmsc import PAST_DUE, gmsc


def test_gmsc_preparation_handles_missing_values_and_codes(tmp_path):
    df = pd.DataFrame({
        "FinancialDistressNextTwoYears": ["No", "Yes", "No", "No"],
        "RevolvingUtilizationOfUnsecuredLines": [0.1, 2.0, 0.0, 50708.0],
        "age": [24, 40, 61, 30],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 2, 0],
        "DebtRatio": [0.3, 0.5, 329664.0, 0.0],
        "MonthlyIncome": [5400.0, np.nan, 0.0, 3000.0],
        "NumberOfOpenCreditLinesAndLoans": [8, 3, 0, 12],
        "NumberOfTimes90DaysLate": [0, 98, 1, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2, 0],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 98, 0, 0],
        "NumberOfDependents": [0.0, 2.0, np.nan, 1.0]})
    path = tmp_path / "gmsc.csv"
    df.to_csv(path, index=False)
    X, y, young = gmsc(str(path))
    assert y.tolist() == [0, 1, 0, 0] and young.tolist() == [True, False, False, False]
    assert not X.isna().any().any() and "FinancialDistressNextTwoYears" not in X.columns
    assert X["MonthlyIncome_missing"].tolist() == [0, 1, 0, 0] and X["NumberOfDependents_missing"].tolist() == [0, 0, 1, 0]
    assert X["past_due_coded"].tolist() == [0, 1, 0, 0]
    assert (X.loc[1, list(PAST_DUE)] == 0).all()                      # the 98 codes are not treated as counts
    assert abs(X.loc[2, "NumberOfTime30-59DaysPastDueNotWorse"] - np.log1p(2)) < 1e-12
    assert abs(X.loc[3, "RevolvingUtilizationOfUnsecuredLines"] - np.log1p(50708.0)) < 1e-9
