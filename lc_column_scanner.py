#!/usr/bin/env python3
"""
lc_column_scanner.py — Dynamic column identification for headerless, shuffled
LendingClub-style datasets, driven by a data dictionary + statistical
fingerprints (EDA, cardinalities, regex vocabularies, relational checks).

NO static positional mapping is used anywhere: every column of the input CSV
is profiled, scored against variable signatures, and assigned by optimal
matching. Works regardless of column order.

Pipeline:
  1. Load data dictionary (xlsx/xls/csv). Falls back to the embedded standard
     LendingClub dictionary if the file is unreadable (e.g. RMS-encrypted).
  2. Profile every column of the headerless CSV: dtype, ranges, quantiles,
     cardinality, missingness, roundness, regex/vocab patterns.
  3. Score (variable x column) and solve the assignment problem.
  4. Relational verification: amortization PMT check, open_acc<=total_acc,
     earliest_cr_line<=issue_d, grade<->sub_grade consistency, int_rate
     monotonic in sub_grade, revol_bal vs tot_cur_bal disambiguation.
  5. Select raw variables for the 9 model features, build features, export.

Usage:
  python3 lc_column_scanner.py --csv dataset2_headless.csv \
      [--dict Data_Dictionary.xlsx] [--out out]
"""

import argparse, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# The 9 final model features (from the Dataset1 model). Raw sources are NOT
# hardcoded: they are resolved at runtime from the feature-engineering lineage
# file (Derived Attributes.csv), the data dictionary, and naming conventions.
# --------------------------------------------------------------------------
TARGET_FEATURES = ["int_rate_clean", "loan_to_income", "dti", "annual_inc",
                   "revol_util", "tot_cur_bal", "loan_amnt", "open_acc_ratio",
                   "verification_ordinal"]

# Transform implementations (how to compute a feature once sources are known).
TRANSFORM_HOW = {
    "int_rate_clean":       "clean % -> float",
    "loan_to_income":       "loan_amnt / annual_inc",
    "open_acc_ratio":       "open_acc / total_acc",
    "verification_ordinal": "Not Verified=0 < Source Verified=1 < Verified=2",
    "payment_to_income":    "installment / (annual_inc / 12)",
}


def load_lineage(path):
    """Parse the feature-engineering lineage CSV (feature, source, rationale).
    Handles 'a + b' multi-sources and 'x / y' alternate feature names.
    Returns ({feature: {sources, rationale}}, note)."""
    if not path or not os.path.exists(path):
        return {}, "no lineage file supplied"
    try:
        ln = pd.read_csv(path, dtype=str).dropna(how="all")
    except Exception as e:
        return {}, f"WARNING: lineage file unreadable ({e})"
    cols = {c.lower().strip(): c for c in ln.columns}
    fcol = next((cols[k] for k in ("feature", "derived", "derived_feature", "name") if k in cols), ln.columns[0])
    scol = next((cols[k] for k in ("source", "raw", "raw_attribute", "from") if k in cols), ln.columns[1])
    rcol = next((cols[k] for k in ("rationale", "reason", "why", "description") if k in cols), None)
    lineage = {}
    for _, row in ln.iterrows():
        feats = [f.strip() for f in str(row[fcol]).split("/") if f.strip() and f.strip().lower() != "nan"]
        sources = [s.strip() for s in re.split(r"[+&,]", str(row[scol])) if s.strip() and s.strip().lower() != "nan"]
        if not feats or not sources:
            continue
        for f in feats:
            lineage[f.lower()] = {"sources": [s.lower() for s in sources],
                                  "rationale": (str(row[rcol]).strip() if rcol is not None else "")}
    return lineage, f"parsed {len(lineage)} derived features from {os.path.basename(path)}"


def resolve_features(targets, lineage, dictionary):
    """Classify each target feature as derived vs raw pass-through and resolve
    its raw source variables. Priority: lineage file > dictionary variable
    (raw pass-through) > naming convention fallback."""
    res = {}
    for feat in targets:
        f = feat.lower()
        if f in lineage:
            res[feat] = {"class": "derived", "sources": lineage[f]["sources"],
                         "origin": "Derived Attributes.csv",
                         "rationale": lineage[f]["rationale"]}
        elif f in dictionary:
            r = {"class": "raw pass-through", "sources": [f],
                 "origin": "dictionary variable", "rationale": ""}
            if f + "_clean" in lineage:  # e.g. final 'revol_util' vs lineage 'revol_util_clean'
                r["rationale"] = (f"note: lineage defines `{f}_clean` from the same source; "
                                  f"final feature treated as the cleaned raw variable")
                r["class"] = "derived (cleaning only)"
            res[feat] = r
        else:  # naming-convention fallback, e.g. verification_ordinal -> verification_status
            base = re.sub(r"_(clean|ordinal|flag|60|bin)$", "", f)
            cand = ([base] if base in dictionary else
                    [v for v in dictionary if v.startswith(base)])
            if len(cand) == 1:
                res[feat] = {"class": "derived", "sources": cand,
                             "origin": "naming convention (NOT listed in lineage file)",
                             "rationale": f"inferred: `{feat}` <- `{cand[0]}`"}
            else:
                res[feat] = {"class": "UNRESOLVED", "sources": [],
                             "origin": "no lineage/dictionary/convention match", "rationale": ""}
    return res

# --------------------------------------------------------------------------
# Embedded fallback dictionary (standard LendingClub LoanStatNew definitions).
# Used ONLY if the provided dictionary file cannot be read.
# --------------------------------------------------------------------------
FALLBACK_DICTIONARY = {
    "loan_amnt": "The listed amount of the loan applied for by the borrower.",
    "term": "The number of payments on the loan. Values are in months and can be either 36 or 60.",
    "int_rate": "Interest Rate on the loan.",
    "installment": "The monthly payment owed by the borrower if the loan originates.",
    "grade": "LC assigned loan grade.",
    "sub_grade": "LC assigned loan subgrade.",
    "emp_title": "The job title supplied by the Borrower when applying for the loan.",
    "emp_length": "Employment length in years. Possible values are between 0 and 10.",
    "home_ownership": "The home ownership status provided by the borrower during registration. Values: RENT, OWN, MORTGAGE, OTHER.",
    "annual_inc": "The self-reported annual income provided by the borrower during registration.",
    "verification_status": "Indicates if income was verified by LC, not verified, or if the income source was verified.",
    "issue_d": "The month which the loan was funded.",
    "loan_status": "Current status of the loan.",
    "purpose": "A category provided by the borrower for the loan request.",
    "title": "The loan title provided by the borrower.",
    "dti": "A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, divided by the borrower's self-reported monthly income.",
    "earliest_cr_line": "The month the borrower's earliest reported credit line was opened.",
    "open_acc": "The number of open credit lines in the borrower's credit file.",
    "pub_rec": "Number of derogatory public records.",
    "revol_bal": "Total credit revolving balance.",
    "revol_util": "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.",
    "total_acc": "The total number of credit lines currently in the borrower's credit file.",
    "initial_list_status": "The initial listing status of the loan. Possible values are W, F.",
    "application_type": "Indicates whether the loan is an individual application or a joint application with two co-borrowers.",
    "mort_acc": "Number of mortgage accounts.",
    "pub_rec_bankruptcies": "Number of public record bankruptcies.",
    "address": "The address of the borrower, including zip code and state.",
    "tot_cur_bal": "Total current balance of all accounts.",
    "fico_range_low": "The lower boundary range the borrower's FICO at loan origination belongs to.",
    "delinq_2yrs": "The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.",
    "inq_last_6mths": "The number of inquiries in past 6 months (excluding auto and mortgage inquiries).",
    "funded_amnt": "The total amount committed to that loan at that point in time.",
    "funded_amnt_inv": "The total amount committed by investors for that loan at that point in time.",
    "zip_code": "The first 3 numbers of the zip code provided by the borrower in the loan application.",
    "addr_state": "The state provided by the borrower in the loan application.",
    "pymnt_plan": "Indicates if a payment plan has been put in place for the loan.",
    "policy_code": "Publicly available policy_code=1; new products not publicly available policy_code=2.",
    "id": "A unique LC assigned ID for the loan listing.",
    "member_id": "A unique LC assigned Id for the borrower member.",
    "last_pymnt_d": "Last month payment was received.",
    "next_pymnt_d": "Next scheduled payment date.",
    "last_credit_pull_d": "The most recent month LC pulled credit for this loan.",
    "last_pymnt_amnt": "Last total payment amount received.",
    "out_prncp": "Remaining outstanding principal for total amount funded.",
    "out_prncp_inv": "Remaining outstanding principal for portion of total amount funded by investors.",
    "total_pymnt": "Payments received to date for total amount funded.",
    "total_pymnt_inv": "Payments received to date for portion of total amount funded by investors.",
    "total_rec_prncp": "Principal received to date.",
    "total_rec_int": "Interest received to date.",
    "total_rec_late_fee": "Late fees received to date.",
    "recoveries": "Post charge off gross recovery.",
    "collection_recovery_fee": "Post charge off collection fee.",
    "tot_coll_amt": "Total collection amounts ever owed.",
    "total_rev_hi_lim": "Total revolving high credit/credit limit.",
    "acc_now_delinq": "The number of accounts on which the borrower is now delinquent.",
    "mths_since_last_delinq": "The number of months since the borrower's last delinquency.",
    "mths_since_last_record": "The number of months since the last public record.",
}

MON_RE = re.compile(r"^[A-Z][a-z]{2}-\d{2,4}$")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


# ============================== dictionary =================================

def load_dictionary(path):
    """Load (variable -> description) from xlsx/xls/csv, tolerating layout
    variance. Returns (dict, source_note)."""
    if not path or not os.path.exists(path):
        return dict(FALLBACK_DICTIONARY), "embedded standard LendingClub dictionary (no file supplied)"
    frames = []
    try:
        if path.lower().endswith((".csv", ".txt")):
            frames = [pd.read_csv(path, header=None, dtype=str, on_bad_lines="skip")]
        else:
            for engine in ("openpyxl", "xlrd", None):
                try:
                    xl = pd.ExcelFile(path, engine=engine)
                    frames = [xl.parse(s, header=None, dtype=str) for s in xl.sheet_names]
                    break
                except Exception:
                    continue
            if not frames:  # last resort: LibreOffice convert (handles legacy .xls saved as .xlsx)
                with tempfile.TemporaryDirectory() as td:
                    tmp = os.path.join(td, "dict.xls")
                    import shutil; shutil.copy(path, tmp)
                    subprocess.run(["libreoffice", "--headless",
                                    "-env:UserInstallation=file:///tmp/lo_scan",
                                    "--convert-to", "csv", "--outdir", td, tmp],
                                   capture_output=True, timeout=120)
                    out = os.path.join(td, "dict.csv")
                    if os.path.exists(out):
                        frames = [pd.read_csv(out, header=None, dtype=str, on_bad_lines="skip")]
    except Exception:
        frames = []
    entries = {}
    name_re = re.compile(r"^[a-zA-Z][a-zA-Z0-9_ ]{1,40}$")
    for df in frames:
        for _, row in df.iterrows():
            cells = [c for c in row.tolist() if isinstance(c, str) and c.strip()]
            if len(cells) < 2:
                continue
            name, desc = cells[0].strip(), max(cells[1:], key=len).strip()
            if name_re.match(name) and len(desc) > 8 and name.lower() not in ("variable", "name", "field", "column"):
                entries[name.strip().lower().replace(" ", "_")] = desc
    if len(entries) >= 5:
        return entries, f"parsed {len(entries)} variables from {os.path.basename(path)}"
    return dict(FALLBACK_DICTIONARY), (
        f"WARNING: could not extract a usable variable table from {os.path.basename(path)} "
        f"(file may be encrypted/RMS-protected). Using embedded standard LendingClub dictionary.")


# ============================== profiling ==================================

def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(r"[%$,]", "", regex=True),
        errors="coerce")


def profile_column(s):
    """Compute the statistical fingerprint of one column."""
    p = {}
    n = len(s)
    p["missing"] = float(s.isna().mean())
    nn = s.dropna()
    p["card"] = int(nn.nunique())
    p["card_ratio"] = p["card"] / max(len(nn), 1)
    sn = clean_numeric(s)
    p["num_rate"] = float(sn.notna().sum() / max(len(nn), 1)) if len(nn) else 0.0
    d = sn.dropna()
    if len(d):
        p.update(min=float(d.min()), q01=float(d.quantile(.01)), q25=float(d.quantile(.25)),
                 med=float(d.median()), q75=float(d.quantile(.75)), q99=float(d.quantile(.99)),
                 max=float(d.max()), mean=float(d.mean()),
                 int_share=float((d == d.round()).mean()),
                 mult1000=float((d % 1000 == 0).mean()), mult25=float((d % 25 == 0).mean()),
                 zero_share=float((d == 0).mean()), gt100=float((d > 100).mean()))
    else:
        p.update({k: np.nan for k in ("min q01 q25 med q75 q99 max mean int_share "
                                      "mult1000 mult25 zero_share gt100").split()})
    st = nn.astype(str).str.strip()
    p["avglen"] = float(st.str.len().mean()) if len(st) else 0.0
    p["vocab"] = set(st.value_counts().head(12).index) if p["card"] <= 60 else set()
    p["vocab_lower"] = {v.lower() for v in p["vocab"]}
    p["empty"] = len(nn) == 0
    if len(st):
        p["share_grade"] = float(st.str.fullmatch(r"[A-G]").mean())
        p["share_subgrade"] = float(st.str.fullmatch(r"[A-G][1-5]").mean())
        p["share_term"] = float(st.str.contains(r"^\s*(?:36|60)\s*(?:months)?\s*$", regex=True).mean())
        # month-year in either direction: 'Jun-12' or '12-Jun'
        p["share_monyy"] = float((st.str.fullmatch(r"[A-Z][a-z]{2}-\d{2,4}") |
                                  st.str.fullmatch(r"\d{2}-[A-Z][a-z]{2}")).mean())
        p["share_zipxx"] = float(st.str.fullmatch(r"\d{3}xx").mean())
        p["share_zip_end"] = float(st.str.contains(r"\d{5}\s*$", regex=True).mean())
        p["share_multiline"] = float(st.str.contains("\n").mean())
    else:
        p["share_grade"] = p["share_subgrade"] = p["share_term"] = 0.0
        p["share_monyy"] = p["share_zip_end"] = p["share_multiline"] = p["share_zipxx"] = 0.0
    if p["share_monyy"] > 0.8:
        yr = st.str.extract(r"^(?:[A-Z][a-z]{2}-)?(\d{2,4})(?:-[A-Z][a-z]{2})?$")[0].astype(float)
        yr = yr.where(yr > 100, np.where(yr <= 26, yr + 2000, yr + 1900))
        p["year_min"], p["year_max"] = float(yr.min()), float(yr.max())
        p["year_span"] = p["year_max"] - p["year_min"]
    return p


# ============================== signatures =================================
# Each signature: profile -> (score 0..1, [evidence strings]).
# Scores combine hard gates (wrong type -> 0) and graded closeness.

def _band(x, lo, hi, soft=0.5):
    """1.0 inside [lo,hi], decaying outside (soft = fraction of band width)."""
    if np.isnan(x):
        return 0.0
    if lo <= x <= hi:
        return 1.0
    w = max((hi - lo) * soft, 1e-9)
    return float(max(0.0, 1.0 - (lo - x) / w if x < lo else 1.0 - (x - hi) / w))


def _num_gate(p, min_rate=0.95):
    # card >= 2: a constant column carries no discriminative signal and must
    # not be claimed by range/shape signatures (only explicit vocab sigs may).
    return p["num_rate"] >= min_rate and not np.isnan(p["med"]) and p["card"] >= 2


def sig_loan_amnt(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 400, 3000), _band(p["max"], 20000, 45000),
         _band(p["med"], 7000, 18000), p["int_share"], _band(p["mult25"], .9, 1),
         _band(p["zero_share"], 0, 0.001)]
    return float(np.mean(s)), [f"range {p['min']:.0f}-{p['max']:.0f}", f"{p['mult25']:.0%} multiples of 25"]

def sig_funded_amnt(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 400, 3000), _band(p["max"], 20000, 45000),
         _band(p["med"], 6000, 18000), p["int_share"], _band(p["mult25"], .85, 1),
         _band(p["zero_share"], 0, 0.001)]
    return float(np.mean(s)) * 0.98, [f"amount grid {p['min']:.0f}-{p['max']:.0f}"]

def sig_funded_amnt_inv(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 0, 3000, soft=1), _band(p["max"], 20000, 45000),
         _band(p["med"], 5000, 18000), _band(p["zero_share"], 0, 0.01, soft=5)]
    return float(np.mean(s)) * 0.9, [f"amount-like {p['min']:.0f}-{p['max']:.0f}, decimals ok"]

def sig_total_rec_prncp(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 0, 0), _band(p["max"], 20000, 45000), _band(p["med"], 4000, 12000),
         _band(p["int_share"], .5, .95, soft=1)]
    return float(np.mean(s)) * 0.85, [f"principal-repaid-like, min 0, med {p['med']:.0f}"]

def sig_total_pymnt(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 50, 500, soft=2), _band(p["med"], 4000, 16000),
         _band(p["card_ratio"], .95, 1), 1 - p["int_share"], _band(p["zero_share"], 0, 0.001)]
    return float(np.mean(s)) * 0.85, [f"payments-to-date-like, decimals, {p['card_ratio']:.0%} unique"]

def sig_total_rec_int(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 0, 60, soft=2), _band(p["med"], 600, 2600),
         _band(p["card_ratio"], .9, 1), 1 - p["int_share"]]
    return float(np.mean(s)) * 0.8, [f"interest-received-like, med {p['med']:.0f}"]

def sig_last_pymnt_amnt(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["med"], 150, 1200), _band(p["q99"], 8000, 40000, soft=1),
         1 - p["int_share"], _band(p["card_ratio"], .9, 1)]
    return float(np.mean(s)) * 0.8, [f"bimodal payment-like, med {p['med']:.0f}, q99 {p['q99']:.0f}"]

def sig_int_rate(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 4, 7), _band(p["max"], 24, 33), _band(p["med"], 10, 17),
         1 - p["int_share"], 1.0 if p["zero_share"] == 0 else 0.0]
    return float(np.mean(s)), [f"pct-like {p['min']:.2f}-{p['max']:.2f}, median {p['med']:.2f}, decimals"]

def sig_installment(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 10, 80), _band(p["max"], 900, 1900), _band(p["med"], 250, 600),
         1 - p["int_share"]]
    return float(np.mean(s)), [f"payment-like {p['min']:.0f}-{p['max']:.0f}, 2dp"]

def sig_annual_inc(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["med"], 40000, 95000), _band(p["max"], 250000, 1e7, soft=0.02),
         _band(p["mult1000"], .5, 1), _band(p["zero_share"], 0, .01)]
    return float(np.mean(s)), [f"income-like median {p['med']:.0f}", f"{p['mult1000']:.0%} round thousands"]

def sig_dti(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["min"], 0, 1), _band(p["q99"], 25, 45), _band(p["med"], 10, 25),
         1 - p["int_share"], 1 - p["gt100"]]
    return float(np.mean(s)), [f"ratio 0-{p['max']:.1f}, median {p['med']:.1f}"]

def sig_revol_util(p):
    if not _num_gate(p, 0.9):
        return 0, []
    s = [_band(p["med"], 35, 65), _band(p["q99"], 85, 130), _band(p["max"], 95, 200, soft=2),
         1 - p["int_share"]]
    return float(np.mean(s)), [f"utilization 0-{p['max']:.1f}%, median {p['med']:.1f}"]

def sig_open_acc(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 7, 14), _band(p["max"], 25, 95, soft=1), _band(p["min"], 0, 2),
         _band(p["card"], 20, 60, soft=1), _band(p["missing"], 0, .02, soft=25)]
    return float(np.mean(s)), [f"count int, median {p['med']:.0f}, max {p['max']:.0f}"]

def sig_total_acc(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 18, 32), _band(p["max"], 50, 190, soft=1), _band(p["min"], 1, 6),
         _band(p["missing"], 0, .02, soft=25)]
    return float(np.mean(s)), [f"count int, median {p['med']:.0f}, max {p['max']:.0f}"]

def sig_pub_rec(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 0, 0), _band(p["mean"], 0, .3, soft=1), _band(p["card"], 2, 25, soft=1),
         _band(p["zero_share"], .85, 1)]
    return float(np.mean(s)), [f"mostly zero int (mean {p['mean']:.2f})"]

def sig_delinq_2yrs(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 0, 0), _band(p["mean"], .1, .5, soft=1), _band(p["card"], 4, 15, soft=1),
         _band(p["zero_share"], .75, .95)]
    return float(np.mean(s)) * 0.95, [f"rare small int (mean {p['mean']:.2f})"]

def sig_inq_last_6mths(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 0, 1), _band(p["mean"], .4, 1.5), _band(p["max"], 4, 35, soft=1),
         _band(p["zero_share"], .3, .7, soft=1)]
    return float(np.mean(s)) * 0.95, [f"inquiry-count-like (mean {p['mean']:.2f})"]

def sig_mths_since_delinq(p):
    if not _num_gate(p, 0.9) or p["int_share"] < 0.95 or p["card"] < 20:
        return 0, []
    s = [_band(p["missing"], .4, .8, soft=1), _band(p["med"], 20, 45), _band(p["max"], 60, 200, soft=1)]
    return float(np.mean(s)) * 0.9, [f"months-since-like, {p['missing']:.0%} missing, med {p['med']:.0f}"]

def sig_mths_since_record(p):
    if not _num_gate(p, 0.9) or p["int_share"] < 0.95 or p["card"] < 15:
        return 0, []
    s = [_band(p["missing"], .8, .98), _band(p["med"], 55, 120), _band(p["max"], 90, 130, soft=1)]
    return float(np.mean(s)) * 0.9, [f"months-since-like, {p['missing']:.0%} missing, med {p['med']:.0f}"]

def sig_pub_rec_bk(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 0, 0), _band(p["mean"], 0, .3), _band(p["card"], 2, 10, soft=1),
         _band(p["max"], 1, 9, soft=1)]
    return float(np.mean(s)), [f"rare-event int, card {p['card']}"]

def sig_mort_acc(p):
    if not _num_gate(p) or p["int_share"] < 0.99:
        return 0, []
    s = [_band(p["med"], 0.5, 3, soft=.4), _band(p["mean"], .9, 3), _band(p["max"], 8, 45, soft=1),
         _band(p["missing"], .01, .15, soft=2)]
    return float(np.mean(s)), [f"small int, mean {p['mean']:.2f}, {p['missing']:.1%} missing"]

def sig_revol_bal(p):
    if not _num_gate(p):
        return 0, []
    s = [_band(p["med"], 4000, 22000), _band(p["q99"], 40000, 300000, soft=1),
         _band(p["zero_share"], 0, .1, soft=2), _band(p["int_share"], .95, 1)]
    return float(np.mean(s)), [f"balance median {p['med']:.0f}, q99 {p['q99']:.0f}, integer"]

def sig_tot_cur_bal(p):
    if not _num_gate(p) or p["med"] < 25000:   # hard gate: total balances are LARGE
        return 0, []
    s = [_band(p["med"], 50000, 250000), _band(p["q99"], 350000, 3e6, soft=1),
         _band(p["mean"], 70000, 400000)]
    return float(np.mean(s)), [f"large balance median {p['med']:.0f}"]

def _vocab_sig(target, min_overlap=0.99):
    def f(p):
        if not p["vocab_lower"]:
            return 0, []
        ov = len(p["vocab_lower"] & target) / max(len(p["vocab_lower"]), 1)
        return (ov if ov >= min_overlap else ov * 0.3), [f"vocab {sorted(p['vocab'])[:4]}"]
    return f

def sig_term(p):
    share = p["share_term"]
    if p["vocab"] and not p["vocab_lower"] - {"36", "60", "36 months", "60 months"}:
        share = max(share, 1.0 if p["card"] == 2 else 0.6)
    return share * (1.0 if p["card"] == 2 else 0.6), [f"36/60 vocab (card {p['card']})"]
sig_grade = lambda p: (p["share_grade"] * _band(p["card"], 5, 7), [f"letters A-G, card {p['card']}"])
sig_sub_grade = lambda p: (p["share_subgrade"] * _band(p["card"], 20, 35), [f"A1-G5 pattern, card {p['card']}"])
sig_home = _vocab_sig({"rent", "mortgage", "own", "other", "none", "any"})
sig_verif = _vocab_sig({"verified", "source verified", "not verified"})
sig_ils = _vocab_sig({"f", "w"})
sig_app_type = _vocab_sig({"individual", "joint", "direct_pay", "joint app"})
sig_loan_status = _vocab_sig({"fully paid", "charged off", "current", "default",
                              "late (31-120 days)", "late (16-30 days)", "in grace period",
                              "does not meet the credit policy. status:fully paid",
                              "does not meet the credit policy. status:charged off"}, 0.8)

def sig_emp_length(p):
    hits = len(p["vocab_lower"] & {"10+ years", "< 1 year", "1 year", "2 years", "3 years",
                                   "4 years", "5 years", "6 years", "7 years", "8 years", "9 years"})
    if hits:
        return hits / 11, [f"employment bins, card {p['card']}"]
    # numeric encoding variant: 0..10 (or <1 encoded fractionally), spike at 10
    if _num_gate(p, .9) and not np.isnan(p["med"]):
        s = [_band(p["card"], 10, 12, soft=1), _band(p["max"], 10, 10, soft=.05),
             _band(p["med"], 3, 8), _band(p["min"], 0, 1)]
        sc = float(np.mean(s)) * 0.9
        return (sc, [f"numeric tenure 0-10, card {p['card']}"]) if sc > 0.4 else (0, [])
    return 0, []

def sig_issue_d(p):
    if p["share_monyy"] < 0.9:
        return 0, []
    s = [_band(p.get("year_min", 0), 2006, 2016, soft=1), _band(p.get("year_max", 0), 2007, 2016.5),
         _band(p.get("year_span", 99), 0, 10, soft=1), _band(p["missing"], 0, 0.001),
         _band(p["card"], 1, 130, soft=2)]
    return float(np.mean(s)), [f"month-year, years {p.get('year_min'):.0f}-{p.get('year_max'):.0f}, card {p['card']}"]

def sig_earliest_cr(p):
    if p["share_monyy"] < 0.9:
        return 0, []
    s = [_band(p.get("year_min", 0), 1940, 1985, soft=2), _band(p.get("year_span", 0), 25, 75, soft=1),
         _band(p["card"], 150, 800, soft=1)]
    return float(np.mean(s)), [f"month-year, years {p.get('year_min'):.0f}-{p.get('year_max'):.0f}, card {p['card']}"]

def sig_last_pymnt_d(p):
    if p["share_monyy"] < 0.9:
        return 0, []
    s = [_band(p.get("year_max", 0), 2013.5, 2017, soft=1), _band(p.get("year_span", 0), 3, 10, soft=1),
         _band(p["missing"], 0.002, 0.2, soft=2), _band(p["card"], 30, 90, soft=1)]
    return float(np.mean(s)) * 0.9, [f"lifecycle date, {p['missing']:.1%} missing"]

def sig_next_pymnt_d(p):
    if p["share_monyy"] < 0.7:
        return 0, []
    s = [_band(p["missing"], .7, 1), _band(p["card"], 1, 5, soft=2)]
    return float(np.mean(s)) * 0.9, [f"mostly-null future date, {p['missing']:.0%} missing"]

def sig_last_credit_pull_d(p):
    if p["share_monyy"] < 0.9:
        return 0, []
    s = [_band(p.get("year_max", 0), 2014.5, 2017, soft=1), _band(p["missing"], 0, 0.01, soft=10),
         _band(p["card"], 30, 90, soft=1)]
    return float(np.mean(s)) * 0.85, [f"credit-pull date, card {p['card']}"]

def sig_purpose(p):
    known = {"debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
             "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
             "renewable_energy", "educational"}
    ov = len(p["vocab_lower"] & known)
    return (min(ov / 6, 1) * _band(p["card"], 8, 15), [f"snake_case categories, card {p['card']}"]) if ov else (0, [])

def sig_title(p):
    if p["num_rate"] > 0.5 or p["card"] < 60 or p["card"] > 2000:
        return 0, []
    s = [_band(p["card"], 150, 900, soft=1), _band(p["avglen"], 8, 30), _band(p["card_ratio"], .05, .5, soft=1)]
    return float(np.mean(s)) * 0.9, [f"free text, card {p['card']}, avglen {p['avglen']:.0f}"]

def sig_emp_title(p):
    if p["num_rate"] > 0.5 or p["card"] < 300:
        return 0, []
    s = [_band(p["card_ratio"], .5, 1, soft=1), _band(p["avglen"], 8, 30),
         _band(p["missing"], .02, .1, soft=2)]
    return float(np.mean(s)), [f"free text, card {p['card']} ({p['card_ratio']:.0%} unique)"]

def sig_address(p):
    if p["num_rate"] > 0.5:   # addresses are not numeric-parseable (ids are)
        return 0, []
    s = [p["share_zip_end"], p["share_multiline"], _band(p["card_ratio"], .95, 1)]
    return float(np.mean(s)), [f"street/zip text, {p['share_zip_end']:.0%} end in 5-digit zip"]

US_STATES = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
             "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY",
             "NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
             "WI","WY","DC"}

def sig_zip_code(p):
    return p["share_zipxx"], [f"{p['share_zipxx']:.0%} match 'NNNxx' zip pattern"]

def sig_addr_state(p):
    if not p["vocab"] or p["card"] > 55 or p["card"] < 15:
        return 0, []
    ov = len({v.upper() for v in p["vocab"]} & US_STATES) / max(len(p["vocab"]), 1)
    return ov, [f"US state codes, card {p['card']}"]

sig_pymnt_plan = _vocab_sig({"n", "y"})

def sig_policy_code(p):
    ok = p["vocab_lower"] and not p["vocab_lower"] - {"1", "2", "1.0", "2.0"}
    return (0.8 if ok else 0), [f"policy code vocab {sorted(p['vocab'])}"]

SIGNATURES = {
    "loan_amnt": sig_loan_amnt, "term": sig_term, "int_rate": sig_int_rate,
    "installment": sig_installment, "grade": sig_grade, "sub_grade": sig_sub_grade,
    "emp_title": sig_emp_title, "emp_length": sig_emp_length, "home_ownership": sig_home,
    "annual_inc": sig_annual_inc, "verification_status": sig_verif, "issue_d": sig_issue_d,
    "loan_status": sig_loan_status, "purpose": sig_purpose, "title": sig_title,
    "dti": sig_dti, "earliest_cr_line": sig_earliest_cr, "open_acc": sig_open_acc,
    "pub_rec": sig_pub_rec, "revol_bal": sig_revol_bal, "revol_util": sig_revol_util,
    "total_acc": sig_total_acc, "initial_list_status": sig_ils,
    "application_type": sig_app_type, "mort_acc": sig_mort_acc,
    "pub_rec_bankruptcies": sig_pub_rec_bk, "address": sig_address,
    "tot_cur_bal": sig_tot_cur_bal,
    "funded_amnt": sig_funded_amnt, "funded_amnt_inv": sig_funded_amnt_inv,
    "total_rec_prncp": sig_total_rec_prncp, "total_pymnt": sig_total_pymnt,
    "total_rec_int": sig_total_rec_int, "last_pymnt_amnt": sig_last_pymnt_amnt,
    "delinq_2yrs": sig_delinq_2yrs, "inq_last_6mths": sig_inq_last_6mths,
    "mths_since_last_delinq": sig_mths_since_delinq,
    "mths_since_last_record": sig_mths_since_record,
    "last_pymnt_d": sig_last_pymnt_d, "next_pymnt_d": sig_next_pymnt_d,
    "last_credit_pull_d": sig_last_credit_pull_d,
    "zip_code": sig_zip_code, "addr_state": sig_addr_state,
    "pymnt_plan": sig_pymnt_plan, "policy_code": sig_policy_code,
}

# Generic description-driven signature for dictionary variables we have no
# hand-tuned signature for (keeps the scanner dictionary-extensible).
KEYWORD_RULES = [
    (r"\brate\b|percent|utilization", lambda p: 0.3 * float(_num_gate(p, .9) and _band(p["max"], 20, 200) > 0)),
    (r"balance|amount|income",        lambda p: 0.3 * float(_num_gate(p) and p["med"] > 500)),
    (r"\bnumber of\b|count",          lambda p: 0.3 * float(_num_gate(p) and p["int_share"] > .99 and p["max"] < 200)),
    (r"month|date",                   lambda p: 0.3 * p["share_monyy"]),
]

def generic_signature(desc):
    desc_l = desc.lower()
    rules = [f for pat, f in KEYWORD_RULES if re.search(pat, desc_l)]
    def f(p):
        if not rules:
            return 0, []
        return max(r(p) for r in rules), ["generic description-keyword match only"]
    return f


# ============================ assignment ===================================

def assign(dictionary, profiles):
    """Score every dictionary variable against every column; solve one-to-one
    assignment maximizing total score."""
    MIN_SCORE = 0.55
    vars_ = list(dictionary.keys())
    score = np.zeros((len(vars_), len(profiles)))
    evid = {}
    for i, v in enumerate(vars_):
        sig = SIGNATURES.get(v, generic_signature(dictionary[v]))
        for j, p in enumerate(profiles):
            if p.get("empty"):          # fully-empty columns carry no signal
                continue
            sc, ev = sig(p)
            score[i, j] = sc
            evid[(i, j)] = ev
    # Zero out sub-threshold scores BEFORE assignment: a variable must never be
    # "compensated" with a junk column elsewhere so that it cedes a contested
    # column to a lower-scoring competitor (global-sum assignment pathology).
    score = np.where(score >= MIN_SCORE, score, 0.0)
    # Deterministic content-keyed tie-breaking: ties between near-identical
    # signatures must resolve the same way regardless of column ORDER, so the
    # jitter is derived from (variable, column-content fingerprint), never from
    # the column index.
    for j, p in enumerate(profiles):
        fp = f"{p.get('med')}|{p.get('card')}|{p.get('missing'):.5f}|{p.get('avglen'):.2f}|{p.get('mean')}"
        for i in range(len(vars_)):
            if score[i, j] > 0:
                h = int(hashlib.md5(f"{vars_[i]}|{fp}".encode()).hexdigest()[:8], 16)
                score[i, j] += (h / 0xFFFFFFFF) * 1e-6
    try:
        from scipy.optimize import linear_sum_assignment
        pad = max(len(vars_), len(profiles))
        m = np.zeros((pad, pad)); m[:len(vars_), :len(profiles)] = score
        ri, ci = linear_sum_assignment(-m)
        pairs = [(i, j) for i, j in zip(ri, ci) if i < len(vars_) and j < len(profiles)]
    except Exception:  # greedy fallback
        pairs, used_v, used_c = [], set(), set()
        for i, j in sorted(np.ndindex(score.shape), key=lambda ij: -score[ij]):
            if i not in used_v and j not in used_c and score[i, j] > 0:
                pairs.append((i, j)); used_v.add(i); used_c.add(j)
    mapping = {}
    for i, j in pairs:
        if score[i, j] >= MIN_SCORE:
            mapping[j] = {"variable": vars_[i], "confidence": round(float(score[i, j]), 3),
                          "evidence": evid[(i, j)]}
    # Transparency: flag statistically ambiguous label pairs — two variables
    # whose scores across two columns are swap-symmetric within 0.01 (the
    # assignment then rests on deterministic tie-breaking, not real evidence).
    assigned = [(i, j) for i, j in pairs if j in mapping and mapping[j]["variable"] == vars_[i]]
    for a in range(len(assigned)):
        for b in range(a + 1, len(assigned)):
            (i1, j1), (i2, j2) = assigned[a], assigned[b]
            if score[i1, j2] >= MIN_SCORE and score[i2, j1] >= MIN_SCORE and \
               abs((score[i1, j1] + score[i2, j2]) - (score[i1, j2] + score[i2, j1])) < 0.01:
                note = f"AMBIGUOUS with `{vars_[i2]}` (col {j2}) — swap-symmetric scores, resolved by deterministic tie-break"
                mapping[j1]["evidence"] = mapping[j1]["evidence"] + [note]
                mapping[j2]["evidence"] = mapping[j2]["evidence"] + [
                    f"AMBIGUOUS with `{vars_[i1]}` (col {j1}) — swap-symmetric scores, resolved by deterministic tie-break"]
    return mapping, score, vars_


# ======================== relational verification ==========================

def parse_monyy(s):
    def one(x):
        x = str(x).strip()
        m = re.match(r"^([A-Z][a-z]{2})-(\d{2,4})$", x)      # 'Jun-12'
        m2 = re.match(r"^(\d{2})-([A-Z][a-z]{2})$", x)       # '12-Jun'
        if m:
            mon, yr = MONTHS.get(m.group(1)), int(m.group(2))
        elif m2:
            mon, yr = MONTHS.get(m2.group(2)), int(m2.group(1))
        else:
            return np.nan
        if yr < 100:
            yr += 2000 if yr <= 26 else 1900
        return yr * 12 + (mon or 1)
    return s.map(one)


def relational_checks(df, mapping):
    """Cross-column consistency checks; returns list of (name, passed, detail)
    and may fix swapped assignments (open/total, issue/earliest, revol/tot)."""
    col_of = {m["variable"]: c for c, m in mapping.items()}
    checks = []

    def num(v):
        return clean_numeric(df[col_of[v]]) if v in col_of else None

    # 0a. Amount-triple dominance: loan_amnt >= funded_amnt >= funded_amnt_inv.
    # Reorders labels among the assigned amount columns if the Hungarian split
    # them arbitrarily (their marginal profiles are near-identical).
    trio = [v for v in ("loan_amnt", "funded_amnt", "funded_amnt_inv") if v in col_of]
    if len(trio) >= 2:
        cols = [col_of[v] for v in trio]
        data = {c: clean_numeric(df[c]) for c in cols}
        dom = {c: sum((data[c] >= data[o]).mean() for o in cols if o != c) + data[c].mean() * 1e-9
               for c in cols}
        ordered = sorted(cols, key=lambda c: -dom[c])
        relabel = dict(zip(ordered, ["loan_amnt", "funded_amnt", "funded_amnt_inv"][:len(ordered)]))
        changed = any(mapping[c]["variable"] != relabel[c] for c in cols)
        for c in cols:
            mapping[c]["variable"] = relabel[c]
        col_of = {m["variable"]: c for c, m in mapping.items()}
        checks.append(("amount dominance loan_amnt >= funded_amnt >= funded_amnt_inv"
                       + (" (auto-reordered)" if changed else ""), True,
                       " | ".join(f"{relabel[c]}=col{c}" for c in ordered)))

    # 1. Amortization: installment == PMT(amount, int_rate, term). LC computes
    # installment from funded_amnt; try both amounts and report the best.
    if all(v in col_of for v in ("int_rate", "term", "installment")):
        best = None
        for amt in ("funded_amnt", "loan_amnt"):
            if amt not in col_of:
                continue
            P, r = num(amt), num("int_rate") / 1200.0
            n = df[col_of["term"]].astype(str).str.extract(r"(\d+)")[0].astype(float)
            pmt = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
            rel = ((pmt - num("installment")).abs() / num("installment")).median()
            if best is None or rel < best[1]:
                best = (amt, rel)
        if best:
            checks.append((f"amortization PMT({best[0]},int_rate,term) ~ installment",
                           bool(best[1] < 0.02), f"median relative error {best[1]:.4%}"))

    # 2. open_acc <= total_acc (swap-fix if reversed)
    if all(v in col_of for v in ("open_acc", "total_acc")):
        a, b = num("open_acc"), num("total_acc")
        ok = (a <= b).mean()
        if ok < 0.5 and (b <= a).mean() > 0.95:
            mapping[col_of["open_acc"]], mapping[col_of["total_acc"]] = \
                mapping[col_of["total_acc"]], mapping[col_of["open_acc"]]
            col_of = {m["variable"]: c for c, m in mapping.items()}
            ok = 1 - ok
            checks.append(("open_acc/total_acc order (auto-swapped)", True, f"{ok:.2%} rows consistent"))
        else:
            checks.append(("open_acc <= total_acc row-wise", bool(ok > 0.99), f"{ok:.2%} of rows"))

    # 3. Date-role ordering: earliest_cr_line <= issue_d <= lifecycle dates.
    # issue_d must be the earliest of the loan-lifecycle dates; auto-corrects a
    # confusion between issue_d and last_pymnt_d / last_credit_pull_d.
    life = [v for v in ("issue_d", "last_pymnt_d", "last_credit_pull_d", "next_pymnt_d")
            if v in col_of]
    if len(life) >= 2 and "issue_d" in life:
        parsed = {v: parse_monyy(df[col_of[v]]) for v in life}
        first = min(life, key=lambda v: parsed[v].median())
        if first != "issue_d":
            mapping[col_of["issue_d"]]["variable"], mapping[col_of[first]]["variable"] = first, "issue_d"
            col_of = {m["variable"]: c for c, m in mapping.items()}
            checks.append(("issue_d must precede lifecycle dates (auto-swapped with "
                           f"{first})", True, "median-date ordering applied"))
        else:
            share = min(((parsed[v] >= parsed["issue_d"]) | parsed[v].isna()).mean()
                        for v in life if v != "issue_d")
            checks.append(("issue_d precedes lifecycle dates", bool(share > 0.97),
                           f"min row-wise share {share:.2%}"))
    if all(v in col_of for v in ("earliest_cr_line", "issue_d")):
        e, i = parse_monyy(df[col_of["earliest_cr_line"]]), parse_monyy(df[col_of["issue_d"]])
        ok = (e <= i).mean()
        checks.append(("earliest_cr_line <= issue_d row-wise", bool(ok > 0.995), f"{ok:.2%} of rows"))

    # 3b. Near-duplicate twin columns (e.g. *_inv variants): informational.
    numcols = [c for c in df.columns if clean_numeric(df[c]).notna().mean() > 0.9]
    twins = []
    for a_i, a in enumerate(numcols):
        for b in numcols[a_i + 1:]:
            x, y = clean_numeric(df[a]), clean_numeric(df[b])
            if x.std() > 0 and y.std() > 0 and x.corr(y) > 0.999:
                twins.append(f"col{a}~col{b}")
    if twins:
        checks.append(("near-duplicate column pairs (corr>0.999, *_inv style)", True,
                       ", ".join(twins[:8])))

    # 3c. Rare-count role resolution: delinq_2yrs / pub_rec / pub_rec_bankruptcies
    # have near-identical marginal profiles. Resolve roles by co-occurrence with
    # their companion columns: delinq_2yrs>0 implies mths_since_last_delinq is
    # populated; pub_rec>0 implies mths_since_last_record is populated.
    fam_cols = [col_of[v] for v in ("delinq_2yrs", "pub_rec", "pub_rec_bankruptcies")
                if v in col_of]
    helpers = {}
    if "mths_since_last_delinq" in col_of:
        helpers["delinq_2yrs"] = clean_numeric(df[col_of["mths_since_last_delinq"]]).notna()
    if "mths_since_last_record" in col_of:
        helpers["pub_rec"] = clean_numeric(df[col_of["mths_since_last_record"]]).notna()
    if len(fam_cols) >= 2 and helpers:
        aff = {}
        for c in fam_cols:
            pos = clean_numeric(df[c]) > 0
            aff[c] = {role: (float(ind[pos].mean()) if pos.any() else 0.0)
                      for role, ind in helpers.items()}
        roles, remaining = {}, set(fam_cols)
        for role in ("delinq_2yrs", "pub_rec"):
            if role in helpers and remaining:
                best = max(remaining, key=lambda c: aff[c].get(role, 0))
                if aff[best].get(role, 0) >= 0.5:
                    roles[best] = role
                    remaining.discard(best)
        if len(remaining) == 1 and "pub_rec_bankruptcies" not in roles.values():
            roles[remaining.pop()] = "pub_rec_bankruptcies"
        old_labels = {c: mapping[c]["variable"] for c in fam_cols}
        if roles and any(old_labels[c] != r for c, r in roles.items()):
            for c in fam_cols:               # clear family labels, then relabel
                if c in roles:
                    mapping[c]["variable"] = roles[c]
                    mapping[c]["evidence"] = mapping[c]["evidence"] + [
                        f"role via companion-column co-occurrence "
                        f"{ {k: round(v, 2) for k, v in aff[c].items()} }"]
                elif mapping[c]["variable"] in ("delinq_2yrs", "pub_rec", "pub_rec_bankruptcies"):
                    del mapping[c]           # label had no evidence to survive
            col_of = {m["variable"]: c for c, m in mapping.items()}
            checks.append(("rare-count roles via mths_since_* co-occurrence (relabeled)", True,
                           " | ".join(f"col{c}={r}" for c, r in roles.items())))
        else:
            checks.append(("rare-count roles via mths_since_* co-occurrence", True,
                           " | ".join(f"col{c}={old_labels[c]} aff={ {k: round(v, 2) for k, v in aff[c].items()} }"
                                      for c in fam_cols)))

    # 3d. pub_rec >= pub_rec_bankruptcies row-wise (bankruptcies are a subset
    # of public records); swap if reversed.
    if all(v in col_of for v in ("pub_rec", "pub_rec_bankruptcies")):
        a, b = num("pub_rec"), num("pub_rec_bankruptcies")
        ok = (a >= b).mean()
        if ok < 0.9 and (b >= a).mean() > 0.98:
            mapping[col_of["pub_rec"]], mapping[col_of["pub_rec_bankruptcies"]] = \
                mapping[col_of["pub_rec_bankruptcies"]], mapping[col_of["pub_rec"]]
            col_of = {m["variable"]: c for c, m in mapping.items()}
            checks.append(("pub_rec >= pub_rec_bankruptcies (auto-swapped)", True, "order corrected"))
        else:
            checks.append(("pub_rec >= pub_rec_bankruptcies row-wise", bool(ok > 0.9), f"{ok:.2%} of rows"))

    # 4. sub_grade starts with grade
    if all(v in col_of for v in ("grade", "sub_grade")):
        ok = (df[col_of["sub_grade"]].astype(str).str[0] == df[col_of["grade"]].astype(str)).mean()
        checks.append(("sub_grade[0] == grade", bool(ok > 0.999), f"{ok:.2%} of rows"))

    # 5. int_rate monotonic in sub_grade rank
    if all(v in col_of for v in ("int_rate", "sub_grade")):
        rank = df[col_of["sub_grade"]].astype(str).map(
            lambda g: (ord(g[0]) - 65) * 5 + int(g[1]) if re.fullmatch(r"[A-G][1-5]", g) else np.nan)
        corr = pd.Series(rank).corr(num("int_rate"), method="spearman")
        checks.append(("int_rate increases with sub_grade rank", bool(corr > 0.9), f"spearman {corr:.3f}"))

    # 6. revol_bal vs tot_cur_bal disambiguation evidence
    for bal in ("revol_bal", "tot_cur_bal"):
        if bal in col_of and "revol_util" in col_of:
            corr = num(bal).corr(num("revol_util"))
            checks.append((f"{bal} vs revol_util correlation", True, f"pearson {corr:.3f} "
                           f"({'revolving-balance-like' if corr > 0.15 else 'NOT revolving-like'})"))
        if bal in col_of and "home_ownership" in col_of:
            ho = df[col_of["home_ownership"]].astype(str).str.upper()
            grp = clean_numeric(df[col_of[bal]]).groupby(ho.values).median()
            if {"MORTGAGE", "RENT"} <= set(grp.index):
                ratio = grp["MORTGAGE"] / max(grp["RENT"], 1)
                checks.append((f"{bal} median MORTGAGE/RENT ratio", True,
                               f"{ratio:.2f}x ({'tot_cur_bal-like (>3x)' if ratio > 3 else 'revol_bal-like (<3x)'})"))

    # 7. loan_to_income sanity
    if all(v in col_of for v in ("loan_amnt", "annual_inc")):
        lti = (num("loan_amnt") / num("annual_inc").replace(0, np.nan)).median()
        checks.append(("median loan_amnt/annual_inc in [0.05, 0.6]", bool(0.05 < lti < 0.6), f"median {lti:.3f}"))
    return checks, mapping


# ============================ feature build ================================

def build_features(df, mapping, resolution):
    col_of = {m["variable"]: c for c, m in mapping.items()}
    out = pd.DataFrame(index=df.index)
    notes = {}

    def raw(v):
        return clean_numeric(df[col_of[v]]) if v in col_of else None

    for feat, r in resolution.items():
        missing = [v for v in r["sources"] if v not in col_of]
        if r["class"] == "UNRESOLVED" or missing:
            out[feat] = np.nan
            notes[feat] = (f"MISSING raw variable(s): {', '.join(missing) or '(unresolved)'} "
                           f"— feature left as NaN")
            continue
        if feat == "int_rate_clean":
            out[feat] = raw("int_rate")
        elif feat == "loan_to_income":
            out[feat] = raw("loan_amnt") / raw("annual_inc").replace(0, np.nan)
        elif feat == "open_acc_ratio":
            out[feat] = raw("open_acc") / raw("total_acc").replace(0, np.nan)
        elif feat == "payment_to_income":
            out[feat] = raw("installment") / (raw("annual_inc").replace(0, np.nan) / 12.0)
        elif feat == "verification_ordinal":
            vmap = {"not verified": 0, "source verified": 1, "verified": 2}
            out[feat] = (df[col_of[r["sources"][0]]].astype(str).str.strip()
                         .str.lower().map(vmap))
        else:  # pass-through / cleaning-only: dti, annual_inc, revol_util, loan_amnt, ...
            out[feat] = raw(r["sources"][0])
        how = TRANSFORM_HOW.get(feat, "clean numeric pass-through")
        notes[feat] = f"[{r['class']}] built from column(s) {[col_of[v] for v in r['sources']]} ({how})"
    if "loan_status" in col_of:  # carry the label through if present (not a feature)
        out["loan_status__label"] = df[col_of["loan_status"]]
        notes["loan_status__label"] = f"target column {col_of['loan_status']} carried through (not a model feature)"
    return out, notes


def describe_profile(p):
    """Human summary for columns no signature could claim."""
    if p.get("empty"):
        return "FULLY EMPTY — no signal; identity cannot be inferred from data"
    if p["card"] == 1:
        return f"constant value {sorted(p['vocab'])} — indistinguishable among constant-valued dictionary variables"
    if p["num_rate"] > 0.95 and p["card_ratio"] > 0.99 and p["min"] > 1000 and p["int_share"] == 1:
        return f"unique sequential integer (range {p['min']:.0f}-{p['max']:.0f}) — id-like (id / member_id / extract index)"
    if p["num_rate"] > 0.9:
        return (f"numeric: med {p['med']:.1f}, max {p['max']:.1f}, {p['zero_share']:.0%} zero, "
                f"{p['missing']:.0%} missing, card {p['card']}")
    return f"text: card {p['card']}, avg len {p['avglen']:.0f}, {p['missing']:.0%} missing"


# ================================ main =====================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dict", default=None)
    ap.add_argument("--derived", default=None,
                    help="feature-engineering lineage CSV (feature,source,rationale)")
    ap.add_argument("--features", default=",".join(TARGET_FEATURES),
                    help="comma-separated final model features to build")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    targets = [f.strip() for f in args.features.split(",") if f.strip()]

    dictionary, dict_note = load_dictionary(args.dict)
    lineage, lineage_note = load_lineage(args.derived)
    resolution = resolve_features(targets, lineage, dictionary)
    required_raw = sorted({v for r in resolution.values() for v in r["sources"]})

    df = pd.read_csv(args.csv, header=None, dtype=str)
    profiles = [profile_column(df[c]) for c in df.columns]
    mapping, score, vars_ = assign(dictionary, profiles)
    checks, mapping = relational_checks(df, mapping)
    features, notes = build_features(df, mapping, resolution)

    # ---- outputs ----
    labeled = df.copy()
    labeled.columns = [mapping.get(c, {}).get("variable", f"UNKNOWN_{c}") for c in df.columns]
    labeled.to_csv(os.path.join(args.out, "dataset2_labeled.csv"), index=False)
    features.to_csv(os.path.join(args.out, "dataset2_features.csv"), index=False)

    json.dump({"dictionary_source": dict_note, "lineage_source": lineage_note,
               "column_mapping": {str(c): mapping.get(c, {"variable": None}) for c in df.columns},
               "relational_checks": [{"check": n, "passed": p, "detail": d} for n, p, d in checks],
               "feature_classification": resolution,
               "feature_notes": notes,
               "required_raw_variables": required_raw},
              open(os.path.join(args.out, "column_mapping.json"), "w"), indent=2)

    # ---- markdown report ----
    assigned_vars = {m["variable"] for m in mapping.values()}
    missing_raw = [v for v in required_raw if v not in assigned_vars]
    col_of = {m["variable"]: c for c, m in mapping.items()}

    lines = ["# Dataset2 headerless column scan", "",
             f"- Input: `{os.path.basename(args.csv)}` — {df.shape[0]} rows x {df.shape[1]} columns (no header)",
             f"- Dictionary: {dict_note}",
             f"- Feature lineage: {lineage_note}",
             f"- Matching: statistical fingerprints (EDA, cardinality, vocab/regex) + optimal assignment; no positional assumptions.", "",
             "## Column mapping", "",
             "| Col # | Variable | Confidence | Evidence |", "|---|---|---|---|"]
    for c in df.columns:
        m = mapping.get(c)
        if m:
            star = " **(raw for model)**" if m["variable"] in required_raw else ""
            lines.append(f"| {c} | `{m['variable']}`{star} | {m['confidence']:.2f} | {'; '.join(m['evidence'])} |")
        else:
            lines.append(f"| {c} | *unassigned* | — | {describe_profile(profiles[c])} |")
    lines += ["", "## Relational verification", ""]
    for n, p, d in checks:
        lines.append(f"- {'PASS' if p else 'INFO/FAIL'} — {n}: {d}")

    lines += ["", f"## Derived vs raw classification ({len(targets)} final features)", "",
              "| Final feature | Class | Raw source(s) | Resolved via | Rationale (from lineage) |",
              "|---|---|---|---|---|"]
    for feat, r in resolution.items():
        src = ", ".join(f"`{s}`" for s in r["sources"]) or "—"
        lines.append(f"| `{feat}` | {r['class']} | {src} | {r['origin']} | {r['rationale'] or '—'} |")

    empty_cols = [c for c in df.columns if profiles[c].get("empty")]
    lines += ["", "## Raw attributes to pick from this dataset", ""]
    for v in required_raw:
        if v in col_of:
            loc = f"column {col_of[v]}"
        elif empty_cols:
            loc = (f"**MISSING/EMPTY — no populated column matches; note {len(empty_cols)} fully-empty "
                   f"column(s) {empty_cols} exist (this variable may be one of them, but an empty "
                   f"column carries no identifying signal)**")
        else:
            loc = "**MISSING — no column matches this variable's profile**"
        lines.append(f"- `{v}` → {loc}")

    if lineage:
        lines += ["", "## Lineage coverage (all Dataset1 derived attributes)", "",
                  "Raw sources referenced by the full feature-engineering lineage and their availability here:", ""]
        all_src = sorted({s for e in lineage.values() for s in e["sources"]})
        for s in all_src:
            lines.append(f"- `{s}`: {'present (col ' + str(col_of[s]) + ')' if s in col_of else 'NOT in this dataset'}")

    ragged = df.index[df.isna().mean(axis=1) > 0.5].tolist()
    lines += ["", "## Data quality", "",
              (f"- {len(ragged)} malformed/truncated row(s) (>50% fields empty) at index(es) "
               f"{ragged[:10]} — emitted with NaNs; drop before scoring."
               if ragged else "- no malformed/truncated rows detected.")]

    lines += ["", "## Feature build log", ""]
    for feat in list(resolution) + (["loan_status__label"] if "loan_status__label" in notes else []):
        lines.append(f"- `{feat}`: {notes.get(feat, '')}")
    lines += ["", f"### Raw variables required: {', '.join(required_raw)}",
              f"### Missing from this dataset: {', '.join(missing_raw) if missing_raw else 'none'}"]
    open(os.path.join(args.out, "mapping_report.md"), "w").write("\n".join(lines) + "\n")

    print(f"dictionary: {dict_note}")
    print(f"lineage: {lineage_note}")
    derived_n = sum(1 for r in resolution.values() if r["class"].startswith("derived"))
    print(f"features: {derived_n} derived / {len(resolution) - derived_n} raw pass-through; "
          f"raw attributes needed: {', '.join(required_raw)}")
    print(f"assigned {len(mapping)}/{df.shape[1]} columns; "
          f"missing raw vars: {missing_raw if missing_raw else 'none'}")
    for n, p, d in checks:
        print(f"  [{'PASS' if p else 'CHECK'}] {n}: {d}")


if __name__ == "__main__":
    main()
