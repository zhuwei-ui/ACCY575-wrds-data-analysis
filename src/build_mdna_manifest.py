"""Build a manifest of 10-K filings to parse for the S&P 500 / 2010-2024 panel.

Runs on the WRDS Cloud login node. Does no file parsing — it only figures
out which filings to parse and where they live on disk.
"""

from pathlib import Path

import wrds

WRDS_USERNAME = "zhuwei"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "interim" / "mdna_manifest.parquet"
FILINGS_ROOT = "/wrds/sec/wrds_clean_filings"

# Re-derives the same S&P 500 / 2010-2024 (gvkey, fyear) panel as
# pull_fundamentals.py so this script has no dependency on it.
QUERY = """
    WITH panel AS (
        SELECT DISTINCT
            f.gvkey,
            f.fyear::int AS fyear,
            f.datadate
        FROM comp.funda AS f
        JOIN crsp.ccmxpf_lnkhist AS l
            ON l.gvkey = f.gvkey
            AND l.linktype IN ('LU', 'LC')
            AND l.linkprim IN ('P', 'C')
            AND f.datadate >= l.linkdt
            AND f.datadate <= COALESCE(l.linkenddt, f.datadate)
        JOIN crsp_a_indexes.dsp500list_v2 AS sp
            ON sp.permno = l.lpermno
            AND f.datadate BETWEEN sp.mbrstartdt AND sp.mbrenddt
        WHERE f.indfmt = 'INDL'
            AND f.datafmt = 'STD'
            AND f.popsrc = 'D'
            AND f.consol = 'C'
            AND f.fyear BETWEEN 2010 AND 2024
    ),
    -- wciklink_gvkey carries a validity window per gvkey-cik link; pick the
    -- link active on datadate rather than the most recent one, and drop
    -- rows with no gvkey (trusts/funds with no Compustat counterpart).
    cik_map AS (
        SELECT DISTINCT
            p.gvkey,
            p.fyear,
            p.datadate,
            w.cik
        FROM panel AS p
        JOIN wrdssec.wciklink_gvkey AS w
            ON w.gvkey = p.gvkey
            AND w.gvkey IS NOT NULL
            AND p.datadate >= w.link_start_date
            AND p.datadate <= COALESCE(w.link_end_date, p.datadate)
    ),
    -- Window starts at fiscal year-end (not calendar fyear) so adjacent
    -- fiscal years' filing windows can't overlap for the same firm.
    filings AS (
        SELECT
            c.gvkey,
            c.fyear,
            c.cik,
            fm.fdate,
            fm.wrdsfname,
            ROW_NUMBER() OVER (
                PARTITION BY c.gvkey, c.fyear
                ORDER BY fm.fdate
            ) AS rn
        FROM cik_map AS c
        JOIN wrdssec_all.wrds_forms AS fm
            ON fm.cik = c.cik
            AND fm.form = '10-K'
            AND fm.fdate >= c.datadate
            AND fm.fdate <= make_date(c.fyear + 1, 3, 31)
    )
    SELECT gvkey, fyear, cik, fdate, wrdsfname
    FROM filings
    WHERE rn = 1
    ORDER BY gvkey, fyear
"""


def main() -> None:
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        df = db.raw_sql(QUERY, date_cols=["fdate"])
    finally:
        db.close()

    # wrds_clean_filings holds one normalized document per filing; the raw
    # warchives tree is the multi-exhibit SGML submission, not what the
    # downstream parser expects.
    df["filepath"] = FILINGS_ROOT + "/" + df["wrdsfname"]
    df = df.drop(columns=["wrdsfname"]).sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Rows pulled: {len(df)}")


if __name__ == "__main__":
    main()
