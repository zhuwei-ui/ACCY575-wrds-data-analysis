"""Pull annual Compustat fundamentals for S&P 500 firms, 2010-2024."""

from pathlib import Path

import wrds

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "fundamentals.parquet"

# S&P 500 membership comes from CRSP (crsp_a_indexes.dsp500list_v2), not
# comp.idxcst_his, which is materially incomplete. The linkdt/linkenddt
# window on ccmxpf_lnkhist restricts each permno-gvkey link to the period
# it was actually valid, since a permno can link to different gvkeys over time.
QUERY = """
    SELECT DISTINCT
        f.gvkey,
        f.datadate,
        f.fyear,
        f.tic,
        f.conm,
        f.at,
        f.lt,
        f.revt,
        f.ni,
        f.oancf,
        f.xrd,
        f.sich
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
"""


def main() -> None:
    db = wrds.Connection()
    try:
        df = db.raw_sql(QUERY, date_cols=["datadate"])
    finally:
        db.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Rows pulled: {len(df)}")


if __name__ == "__main__":
    main()
