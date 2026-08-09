"""Module 9: OLS of 1-year-ahead ROA on current-year ratios and industry dummies."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from src.models.ols import fit_ols
from src.schema import FundamentalsSchema

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "fundamentals.parquet"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "m9"

RATIO_FEATURES = ["asset_turnover", "rd_intensity", "cfo_to_at"]


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values(["gvkey", "fyear"]).copy()

    roa = df["ni"] / df["at"]
    df["target"] = roa.groupby(df["gvkey"]).shift(-1)

    df["asset_turnover"] = (df["revt"] / df["at"]).replace([np.inf, -np.inf], np.nan)
    # Compustat leaves xrd NULL for firms that don't disclose a separate
    # R&D line (not required unless material) — it does not code 0. That's
    # ~44% of the sample, concentrated in non-R&D sectors (utilities,
    # financials, retailers), so dropping those rows via dropna would bias
    # the regression toward R&D-heavy sectors. No disclosed R&D is treated
    # as economically zero R&D.
    df["rd_intensity"] = (df["xrd"].fillna(0) / df["at"]).replace([np.inf, -np.inf], np.nan)
    df["cfo_to_at"] = (df["oancf"] / df["at"]).replace([np.inf, -np.inf], np.nan)

    # Bucket to 2-digit SIC (major industry group) rather than the raw
    # 4-digit sich: at 4-digit granularity, 8 of 241 codes have exactly
    # one observation in the whole sample, and a singleton-category dummy
    # drives that row's leverage to exactly 1 — HC3 divides by (1-leverage)^2
    # in its sandwich formula, so leverage=1 is a division by zero that
    # corrupts the whole covariance matrix and crashes .summary(). 2-digit
    # grouping (241 -> 63 categories) eliminates every singleton.
    #
    # drop_first avoids the dummy trap: with an intercept already in the
    # design matrix, including every category would make the dummy block
    # collinear with the constant. statsmodels won't raise on this, but
    # .summary() flags it as a multicollinearity warning.
    industry_group = df["sich"] // 100
    industry_dummies = pd.get_dummies(industry_group, prefix="sic2", drop_first=True).astype(float)

    df = pd.concat([df, industry_dummies], axis=1)
    features = RATIO_FEATURES + list(industry_dummies.columns)
    return df, features


def save_diagnostics(results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = results.fittedvalues
    resid = results.resid
    influence = results.get_influence()
    standardized_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag

    fig, ax = plt.subplots()
    ax.scatter(fitted, resid, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    fig.savefig(out_dir / "residuals_vs_fitted.png")
    plt.close(fig)

    fig = sm.qqplot(resid, line="45")
    fig.suptitle("Normal Q-Q")
    fig.savefig(out_dir / "qq_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|Standardized residuals|)")
    ax.set_title("Scale-Location")
    fig.savefig(out_dir / "scale_location.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(leverage, standardized_resid, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Residuals vs Leverage")
    fig.savefig(out_dir / "leverage.png")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    FundamentalsSchema.validate(df, lazy=True)  # gate check only; keeps sich/oancf/xrd for features

    df, features = build_dataset(df)

    results = fit_ols(df, target="target", features=features)
    print(results.summary())

    save_diagnostics(results, RESULTS_DIR)
    print(f"Diagnostic plots written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
