import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.models.gbm import fit_gbm


def test_fit_gbm_predicts_synthetic_pattern():
    rng = np.random.default_rng(0)
    n_per_year = 60
    years = range(2015, 2023)  # 8 years: enough for a 2-year validation carve-out

    rows = []
    for fyear in years:
        x1 = rng.normal(size=n_per_year)
        x2 = rng.normal(size=n_per_year)
        y = 2 * x1 - 3 * x2 + rng.normal(scale=0.05, size=n_per_year)
        rows.append(pd.DataFrame({"fyear": fyear, "x1": x1, "x2": x2, "y": y}))
    df = pd.concat(rows, ignore_index=True)

    train = df[df["fyear"] <= 2020]
    test = df[df["fyear"] > 2020]

    _, predictions = fit_gbm(
        train, test, target="y", features=["x1", "x2"],
        params={"n_estimators": 50, "early_stopping_rounds": 5},
    )

    assert r2_score(test["y"], predictions) > 0.8
