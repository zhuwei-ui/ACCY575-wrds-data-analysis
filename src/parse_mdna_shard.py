"""Extract Item 7 (MD&A) text for one shard of the filing manifest.

Runs on a compute node inside a Grid Engine array task: no Postgres
connection, no prompts, no network. Only local file I/O and parsing.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Numeric item labels: not reliable alone. Some large filers (e.g. Intel)
# use a stylized annual-report layout with no literal "Item 7." next to
# the actual MD&A heading — "item 7" only appears in a cross-reference
# index table far from the real content. Bare section titles catch those.
ITEM7_RE = re.compile(r"item\s*7[.\s]", re.IGNORECASE)
ITEM7A_RE = re.compile(r"item\s*7a[.\s]", re.IGNORECASE)
ITEM8_RE = re.compile(r"item\s*8[.\s]", re.IGNORECASE)
MDNA_HEADING_RE = re.compile(r"management'?s discussion and analysis", re.IGNORECASE)
MARKET_RISK_RE = re.compile(r"quantitative and qualitative disclosures about market risk", re.IGNORECASE)
FINANCIAL_STATEMENTS_RE = re.compile(r"financial statements and supplementary data", re.IGNORECASE)

DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)
TYPE_RE = re.compile(r"<TYPE>(.*?)[\r\n]", re.IGNORECASE)
FILENAME_RE = re.compile(r"<FILENAME>(.*?)[\r\n]", re.IGNORECASE)
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.IGNORECASE | re.DOTALL)

# Matches MdnaSchema's char_count bounds (§9) — a candidate span outside
# this range is almost certainly a stray TOC/cross-reference match, not
# the real section, so it's rejected rather than accepted as-is.
MIN_PLAUSIBLE_LEN = 5_000
MAX_PLAUSIBLE_LEN = 300_000

FAILURE_RATE_THRESHOLD = 0.2


def extract_primary_document(raw: str) -> str:
    """wrds_clean_filings files are the full SGML submission, not a single
    clean document: multiple <DOCUMENT> blocks (the real 10-K HTML, a
    courtesy PDF also tagged TYPE 10-K, exhibits...). Isolate the first
    non-PDF block tagged TYPE 10-K so extraction runs on just that filing,
    not the whole multi-document blob. Falls back to the raw text if no
    <DOCUMENT> wrapper is found (already-unwrapped files)."""
    for match in DOCUMENT_RE.finditer(raw):
        block = match.group(1)
        type_match = TYPE_RE.search(block)
        if not type_match or type_match.group(1).strip().upper() != "10-K":
            continue

        filename_match = FILENAME_RE.search(block)
        filename = filename_match.group(1).strip().lower() if filename_match else ""
        if filename.endswith(".pdf"):
            continue

        text_match = TEXT_RE.search(block)
        if text_match:
            return text_match.group(1)

    return raw


def extract_via_regex(text: str) -> str | None:
    starts = sorted(m.start() for pattern in (ITEM7_RE, MDNA_HEADING_RE) for m in pattern.finditer(text))
    ends = sorted(
        m.start()
        for pattern in (ITEM7A_RE, MARKET_RISK_RE, ITEM8_RE, FINANCIAL_STATEMENTS_RE)
        for m in pattern.finditer(text)
    )
    if not starts or not ends:
        return None

    candidates = []
    for start in starts:
        end = next((e for e in ends if e > start), None)
        if end is not None:
            candidates.append((start, end))

    # Score every (start, end) pairing by plausible length rather than
    # trusting "last Item 7 before the next Item 7A" — that pairing is
    # wrong whenever a cross-reference table or TOC entry sits closer to
    # a boundary marker than the real section does.
    plausible = [(s, e) for s, e in candidates if MIN_PLAUSIBLE_LEN <= (e - s) <= MAX_PLAUSIBLE_LEN]
    if not plausible:
        return None

    start, end = max(plausible, key=lambda pair: pair[1] - pair[0])
    section = text[start:end].strip()
    return section or None


def extract_mdna(filepath: Path) -> str | None:
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    text = extract_primary_document(raw)
    return extract_via_regex(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest).reset_index(drop=True)
    shard = manifest[manifest.index % args.nshards == args.shard]

    rows = []
    n_failed = 0
    for row in shard.itertuples(index=False):
        try:
            mdna = extract_mdna(Path(row.filepath))
        except OSError as exc:
            print(f"FAIL gvkey={row.gvkey} fyear={row.fyear}: {exc}", file=sys.stderr)
            n_failed += 1
            continue

        if mdna is None:
            print(f"FAIL gvkey={row.gvkey} fyear={row.fyear}: no Item 7 match", file=sys.stderr)
            n_failed += 1
            continue

        rows.append(
            {
                "gvkey": row.gvkey,
                "fyear": row.fyear,
                "cik": row.cik,
                "fdate": row.fdate,
                "mdna_text": mdna,
                "char_count": len(mdna),
            }
        )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    n_total = len(shard)
    print(f"Wrote {len(rows)} rows, {n_failed} failures (of {n_total} in shard)")

    if n_total > 0 and n_failed / n_total > FAILURE_RATE_THRESHOLD:
        sys.exit(1)


if __name__ == "__main__":
    main()
