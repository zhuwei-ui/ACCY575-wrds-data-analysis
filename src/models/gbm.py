"""XGBoost regressor fitting helper with early stopping."""

import pandas as pd
import xgboost

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "early_stopping_rounds": 20,
    "eval_metric": "rmse",
}

N_VALIDATION_YEARS = 2


def fit_gbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    params: dict | None = None,
) -> tuple["xgboost.XGBRegressor", pd.Series]:
    model_params = {**DEFAULT_PARAMS, **(params or {})}

    # XGBoost handles NaN features natively (missing-aware splits), but a
    # NaN target crashes the fit outright, so only target rows are dropped.
    n_before = len(train)
    train = train.dropna(subset=[target])
    print(f"fit_gbm: dropped {n_before - len(train)} of {n_before} train rows with NaN target")

    # Early stopping needs its own held-out slice to monitor — carved from
    # the last two fiscal years present in train, never from test. Early
    # stopping against test would tune tree count on the rows being graded.
    val_years = sorted(train["fyear"].unique())[-N_VALIDATION_YEARS:]
    is_val = train["fyear"].isin(val_years)
    fit_rows, val_rows = train[~is_val], train[is_val]

    X_train, y_train = fit_rows[features], fit_rows[target]
    X_val, y_val = val_rows[features], val_rows[target]

    model = xgboost.XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    predictions = pd.Series(model.predict(test[features]), index=test.index, name="prediction")
    return model, predictions
