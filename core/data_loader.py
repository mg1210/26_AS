"""
core/data_loader.py
───────────────────
Shared, header-aware CSV loader used by BOTH the pipeline and the Streamlit UI.

Blind evaluation datasets may arrive without a header row. If the pipeline and
the UI disagree on whether row 0 is a header, their column names diverge and
every UI lookup against the pipeline schema silently breaks. Routing all reads
through smart_read_csv() guarantees headerless files get identical
``col_0, col_1, …`` names everywhere.
"""

import csv
import os
import pandas as pd


def _first_row_is_header_by_dtype(path: str):
    """Tri-state dtype heuristic, robust to sparse (missing-heavy) data.

    Looks only at columns that are clearly numeric in the body, and among
    those only at the ones whose first-row value is present (non-null). A real
    header makes ~all such columns show a text label as their first value;
    a headerless file makes ~none. Decide by that fraction.

    Returns:
      True  → header row present (most numeric columns start with a text label).
      False → headerless (most numeric columns start with a number).
      None  → undecidable (no judgeable numeric columns, or the fraction is
              ambiguous — caller falls back to csv.Sniffer).
    """
    try:
        peek = pd.read_csv(path, header=None, nrows=200, low_memory=False)
    except Exception:
        return None
    if len(peek) < 2:
        return None

    first, rest = peek.iloc[0], peek.iloc[1:]
    judged = 0
    text_first = 0
    for c in peek.columns:
        body_num = pd.to_numeric(rest[c], errors="coerce")
        if body_num.notna().mean() <= 0.8:           # not a dense numeric column
            continue
        v0 = first[c]
        if pd.isna(v0):                              # missing first value → can't judge
            continue
        judged += 1
        first_num = pd.to_numeric(pd.Series([v0]), errors="coerce").notna().all()
        if not first_num:
            text_first += 1

    if judged == 0:
        return None                                  # nothing numeric to judge → sniffer
    frac_text = text_first / judged
    if frac_text >= 0.7:
        return True                                  # numeric columns labelled by text → header
    if frac_text <= 0.3:
        return False                                 # numeric columns start with numbers → headerless
    return None                                      # ambiguous → sniffer


def detect_headerless(path: str, sample_bytes: int = 65536) -> bool:
    """Return True when the CSV appears to have NO header row.

    Combines a dtype-mismatch check (strong signal for the numeric-heavy
    credit datasets this factory handles) with Python's csv.Sniffer as a
    fallback for all-text data. Defaults to "has header" whenever the
    evidence is inconclusive, since that is the common case.
    """
    dtype_verdict = _first_row_is_header_by_dtype(path)
    if dtype_verdict is True:
        return False        # clear header
    if dtype_verdict is False:
        return True         # clear headerless

    # Undecidable on dtype → fall back to csv.Sniffer's heuristic.
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(sample_bytes)
    except Exception:
        return False
    if not sample.strip():
        return False
    try:
        return not csv.Sniffer().has_header(sample)
    except csv.Error:
        return False        # inconclusive → assume header present


def smart_read_csv(path: str, **kwargs):
    """Load a CSV, auto-detecting headerless files.

    Returns ``(df, headerless)``. When headerless, columns are named
    ``col_0, col_1, …`` so every consumer agrees on column identity.
    Extra kwargs are forwarded to ``pandas.read_csv``.
    """
    headerless = detect_headerless(path)
    if headerless:
        df = pd.read_csv(path, header=None, low_memory=False, **kwargs)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(path, low_memory=False, **kwargs)
    return df, headerless
