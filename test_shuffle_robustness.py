#!/usr/bin/env python3
"""
test_shuffle_robustness.py — proves lc_column_scanner.py is column-order
invariant (true data scanning, not a static mapping).

Runs the scanner on the original CSV, then on N random column permutations,
and asserts the variable assigned to each (content-identical) column never
changes. Exit code 0 = PASS, 1 = FAIL.

Usage:
  python3 test_shuffle_robustness.py --csv dataset2_headless.csv \
      [--dict Data_Dictionary.xlsx] [--derived derived_attributes.csv] \
      [--seeds 7,42,123]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.join(HERE, "lc_column_scanner.py")


def run_scanner(csv, out, args):
    cmd = [sys.executable, SCANNER, "--csv", csv, "--out", out]
    if args.dict:
        cmd += ["--dict", args.dict]
    if args.derived:
        cmd += ["--derived", args.derived]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(1)
    m = json.load(open(os.path.join(out, "column_mapping.json")))["column_mapping"]
    return {int(k): v.get("variable") for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dict", default=None)
    ap.add_argument("--derived", default=None)
    ap.add_argument("--seeds", default="7,42,123")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df = pd.read_csv(args.csv, header=None, dtype=str)
    df = df[df.isna().mean(axis=1) <= 0.5]  # drop malformed/truncated rows

    tmp = tempfile.mkdtemp(prefix="shuffle_test_")
    try:
        # Baseline must scan the SAME cleaned rows the shuffles are built from,
        # otherwise base and shuffled runs see different data.
        base_csv = os.path.join(tmp, "base.csv")
        df.to_csv(base_csv, index=False, header=False)
        base = run_scanner(base_csv, os.path.join(tmp, "base"), args)
        ok_all = True
        for seed in seeds:
            perm = np.random.RandomState(seed).permutation(df.shape[1])
            shuf_csv = os.path.join(tmp, f"shuf_{seed}.csv")
            df[perm].to_csv(shuf_csv, index=False, header=False)
            got = run_scanner(shuf_csv, os.path.join(tmp, f"out_{seed}"), args)
            mism = [(j, base[perm[j]], got.get(j))
                    for j in range(df.shape[1]) if got.get(j) != base[perm[j]]]
            ok = not mism
            ok_all &= ok
            print(f"seed {seed}: " + ("IDENTICAL mapping under shuffle" if ok
                                      else f"MISMATCHES {mism}"))
        print("ROBUSTNESS:", "PASS — scanner is order-invariant" if ok_all else "FAIL")
        sys.exit(0 if ok_all else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
