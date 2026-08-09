import numpy as np
import pandas as pd
import pytest

from src.models.ols import fit_ols


def test_fit_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(scale=0.01, size=n)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    df.loc[0, "x1"] = np.nan  # exercise the NaN-drop path
    df.loc[1, "y"] = np.nan

    results = fit_ols(df, target="y", features=["x1", "x2"])

    assert results.params["const"] == pytest.approx(2.0, abs=0.05)
    assert results.params["x1"] == pytest.approx(3.0, abs=0.05)
    assert results.params["x2"] == pytest.approx(-1.5, abs=0.05)
