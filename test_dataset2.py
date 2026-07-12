"""
test_dataset2.py — Blind "Dataset 2" robustness harness.

Creates three stressed variants of the development dataset, runs the full
pipeline on each via `python main.py --auto --trials 3 --dataset <variant>`
(each in its own subprocess so a hard crash can't take the harness down),
then reports for every variant: did it complete, the champion model AUC, and
any errors. A summary table is printed at the end.

Run:
    python test_dataset2.py
"""

import os
import sys
import glob
import json
import time
import subprocess

import pandas as pd

ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(ROOT, "data")
SRC    = os.path.join(DATA, "Loan_Data_Development.csv")
TRIALS = "3"
PER_RUN_TIMEOUT = 1800   # seconds, per variant

# Columns feature engineering derives features from (see feature_engineering_agent.py).
# The missing-cols variant drops the first 20 of these to exercise the try/except
# guards on every feature method.
FE_DEPENDENT_COLS = [
    "earliest_cr_line", "issue_d", "grade", "sub_grade", "int_rate", "term",
    "installment", "annual_inc", "loan_amnt", "emp_length", "home_ownership",
    "delinq_2yrs", "mths_since_last_delinq", "pub_rec", "inq_last_6mths",
    "revol_util", "open_acc", "total_acc", "verification_status", "purpose",
    "initial_list_status", "addr_state",
]


# ──────────────────────────────────────────────────────────────────────────
# Variant construction
# ──────────────────────────────────────────────────────────────────────────
def build_variants() -> dict:
    """Create the 3 test CSVs in data/ and return {name: (path, note)}."""
    if not os.path.exists(SRC):
        print(f"ERROR: source dataset not found at {SRC}")
        sys.exit(1)

    print(f"Loading source dataset: {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    variants = {}

    # 1) Headerless — same data, header row stripped.
    p1 = os.path.join(DATA, "test_no_header.csv")
    df.to_csv(p1, header=False, index=False)
    variants["no_header"] = (p1, "header row stripped")
    print(f"  [1] {os.path.basename(p1)} — header row stripped")

    # 2) Renamed key columns.
    p2 = os.path.join(DATA, "test_renamed_cols.csv")
    rename_map = {"loan_status": "status", "annual_inc": "yearly_income", "loan_amnt": "amount"}
    applied = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=applied).to_csv(p2, index=False)
    variants["renamed_cols"] = (p2, "renamed " + ", ".join(f"{k}->{v}" for k, v in applied.items()))
    print(f"  [2] {os.path.basename(p2)} — renamed {applied}")

    # 3) Missing 20 feature-engineering-dependent columns.
    p3 = os.path.join(DATA, "test_missing_cols.csv")
    to_drop = [c for c in FE_DEPENDENT_COLS if c in df.columns][:20]
    df.drop(columns=to_drop).to_csv(p3, index=False)
    variants["missing_cols"] = (p3, f"dropped {len(to_drop)} FE cols")
    print(f"  [3] {os.path.basename(p3)} — dropped {len(to_drop)} FE cols: {to_drop}\n")

    return variants


# ──────────────────────────────────────────────────────────────────────────
# Pipeline execution + result extraction
# ──────────────────────────────────────────────────────────────────────────
def _newest_audit_since(t0: float):
    """Return the newest audit-trail JSON written at/after t0, or None."""
    files = [f for f in glob.glob(os.path.join(ROOT, "outputs", "RUN_*_audit_trail.json"))
             if os.path.getmtime(f) >= t0 - 1]
    return max(files, key=os.path.getmtime) if files else None


def run_pipeline(path: str) -> dict:
    """Run main.py on one variant and return a result dict."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    crashed = False
    stderr_tail = []
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py"),
             "--auto", "--trials", TRIALS, "--dataset", path],
            cwd=ROOT, capture_output=True, text=True, env=env,
            timeout=PER_RUN_TIMEOUT,
        )
        returncode = proc.returncode
        stderr = proc.stderr or ""
        # main.py exits 1 on CONDITIONAL validation (not a crash); a real crash
        # shows a Python traceback on stderr.
        crashed = "Traceback (most recent call last)" in stderr
        stderr_tail = [ln for ln in stderr.strip().splitlines() if ln][-4:]
    except subprocess.TimeoutExpired:
        return {"completed": False, "crashed": True, "elapsed": time.time() - t0,
                "champion": "—", "auc": None, "n_errors": None, "validation": "TIMEOUT",
                "note": f"exceeded {PER_RUN_TIMEOUT}s"}

    elapsed = time.time() - t0
    audit = _newest_audit_since(t0)

    champion, auc, n_errors, validation = "—", None, None, "—"
    if audit:
        try:
            with open(audit, encoding="utf-8") as f:
                d = json.load(f)
            champion = d.get("champion_model_name") or "—"
            auc = d.get("model_metrics", {}).get(champion, {}).get("auc_test")
            n_errors = len(d.get("errors", []))
            validation = "PASS" if d.get("validation_passed") else "CONDITIONAL"
        except Exception as e:
            stderr_tail.append(f"(audit parse failed: {e})")

    # "Completed" = pipeline produced an audit trail with a champion and did not
    # crash with an unhandled exception.
    completed = (audit is not None) and (champion != "—") and (not crashed)
    return {"completed": completed, "crashed": crashed, "elapsed": elapsed,
            "champion": champion, "auc": auc, "n_errors": n_errors,
            "validation": validation, "note": " | ".join(stderr_tail) if crashed else ""}


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "=" * 92)
    print("  DATASET 2 ROBUSTNESS SUMMARY")
    print("=" * 92)
    hdr = f"  {'Variant':<16}{'Completed':<11}{'Champion':<22}{'AUC':<9}{'Errors':<8}{'Validation':<13}{'Time':<8}"
    print(hdr)
    print("  " + "-" * 88)
    for name, r in results.items():
        completed = "YES" if r["completed"] else "NO"
        auc = f"{r['auc']:.4f}" if isinstance(r["auc"], (int, float)) else "—"
        errs = "—" if r["n_errors"] is None else str(r["n_errors"])
        row = (f"  {name:<16}{completed:<11}{str(r['champion'])[:20]:<22}"
               f"{auc:<9}{errs:<8}{r['validation']:<13}{r['elapsed']:.0f}s")
        print(row)
        if r.get("crashed") and r.get("note"):
            print(f"      ↳ crash: {r['note']}")
    print("=" * 92)
    n_ok = sum(1 for r in results.values() if r["completed"])
    print(f"  {n_ok}/{len(results)} variants completed successfully.\n")


def main():
    print("=" * 92)
    print("  BUILDING TEST VARIANTS")
    print("=" * 92)
    variants = build_variants()

    results = {}
    for name, (path, note) in variants.items():
        print("=" * 92)
        print(f"  RUNNING: {name}  ({note})")
        print(f"  dataset: {path}")
        print("=" * 92)
        results[name] = run_pipeline(path)
        r = results[name]
        auc = f"{r['auc']:.4f}" if isinstance(r["auc"], (int, float)) else "—"
        print(f"  -> completed={r['completed']}  champion={r['champion']}  "
              f"AUC={auc}  errors={r['n_errors']}  {r['validation']}  ({r['elapsed']:.0f}s)\n")

    print_summary(results)


if __name__ == "__main__":
    main()
