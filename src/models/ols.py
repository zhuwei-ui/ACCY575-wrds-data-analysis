"""OLS fitting helper with heteroskedasticity-robust standard errors."""

import pandas as pd
import statsmodels.api as sm


def fit_ols(
    df: pd.DataFrame, target: str, features: list[str]
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":
    n_before = len(df)
    clean = df.dropna(subset=[target, *features])
    n_dropped = n_before - len(clean)
    print(f"fit_ols: dropped {n_dropped} of {n_before} rows with NaN in target/features")

    # WRDS-sourced frames commonly carry pandas nullable dtypes (Float64,
    # Int64), which statsmodels fails to convert to a numpy array. Cast
    # to plain float64 so sm.OLS gets a real numpy-backed array.
    y = clean[target].astype("float64")
    X = sm.add_constant(clean[features].astype("float64"))
    return sm.OLS(y, X).fit(cov_type="HC3")
