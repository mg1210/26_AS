"""
agents/dqr_agent.py
────────────────────
Phase 2 — Data Quality Review (DQR) Agent

Responsibilities:
  • Missing value assessment (count, %, visualisation data)
  • Outlier detection (IQR, z-score)
  • Duplicate identification
  • Distribution summary + percentile analysis
  • Target rate by variable band
  • Variable stability indicators (IV proxy)
  • LLM-generated DQR narrative and recommendations
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM


class DQRAgent(BaseAgent):

    def __init__(self, missing_threshold: float = 0.40, verbose: bool = True):
        super().__init__("DQRAgent", verbose)
        self.missing_threshold = missing_threshold   # cols above this are flagged

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        df = state.working_df.copy()
        model_cols = (state.numeric_columns + state.categorical_columns
                      + state.date_columns)

        self._log("Checking duplicates …")
        state = self._check_duplicates(state, df)

        self._log("Running missing value analysis …")
        state = self._missing_analysis(state, df, model_cols)

        self._log("Running outlier detection …")
        state = self._outlier_detection(state, df, state.numeric_columns)

        self._log("Running business consistency checks …")
        state = self._consistency_checks(state, df)

        self._log("Computing distribution profiles …")
        state = self._distribution_profiles(state, df, model_cols)

        self._log("Computing target rates by band …")
        state = self._target_rates(state, df)

        self._log("Generating DQR narrative with LLM …")
        state = self._llm_narrative(state)

        # Build structured response
        dup_info   = state.dqr_report.get("duplicates", {})
        miss_cols  = sum(1 for v in state.missing_summary.values()
                         if (v.get("pct_missing", 0) if isinstance(v, dict) else float(v or 0)) > 0)
        state.agent_responses[self.name] = self.build_response(
            summary="Data quality review complete",
            observations=[
                f"Columns with any missing values: {miss_cols}",
                f"High-missing columns (>{self.missing_threshold:.0%}) flagged: {len(state.high_missing_cols)}",
                f"Duplicate loan IDs found: {dup_info.get('duplicate_ids', 0)}",
                f"Duplicate member IDs found: {dup_info.get('duplicate_members', 0)}",
                f"DQR flags raised: {len(state.dqr_flags)}",
                f"Total rows in working dataset: {dup_info.get('total_rows', 'N/A'):,}" if isinstance(dup_info.get('total_rows'), int) else f"Total rows: N/A",
            ],
            reasoning=(
                f"Missing values above {self.missing_threshold:.0%} threshold flagged for exclusion "
                "as they provide insufficient information for reliable imputation. "
                "IQR and z-score methods applied for outlier detection. "
                "Duplicates checked on loan ID and member ID."
            ),
            recommendations=[
                f"Drop or carefully impute {len(state.high_missing_cols)} high-missing columns",
                "Apply median imputation for numeric features with <40% missing",
                "Apply mode imputation for categorical features",
                "Review flagged outliers — clip or winsorise extreme values before modelling",
            ],
        )

        return state

    # ─────────────────────────────────────────────────────────────
    def _check_duplicates(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        total_rows  = len(df)
        dup_records = df.duplicated(subset=["id"]).sum() if "id" in df.columns else 0
        dup_members = df.duplicated(subset=["member_id"]).sum() if "member_id" in df.columns else 0

        state.dqr_report["duplicates"] = {
            "total_rows"       : int(total_rows),
            "duplicate_ids"    : int(dup_records),
            "duplicate_members": int(dup_members),
        }
        if dup_records > 0:
            state.log_warning(self.name, f"{dup_records} duplicate loan IDs found")
            state.dqr_flags.append(f"⚠  {dup_records} duplicate loan IDs — review before modelling")
        self._info(f"Total rows: {total_rows:,} | Dup IDs: {dup_records} | Dup members: {dup_members}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _missing_analysis(self, state: PipelineState,
                          df: pd.DataFrame, cols: list) -> PipelineState:
        missing = {}
        high_missing = []

        for col in cols:
            if col not in df.columns:
                continue
            n_miss  = int(df[col].isna().sum())
            pct     = float(n_miss / len(df))
            missing[col] = {"n_missing": n_miss, "pct_missing": round(pct, 4)}
            if pct > self.missing_threshold:
                high_missing.append(col)
                state.dqr_flags.append(
                    f"⚠  '{col}' is {pct:.0%} missing — consider dropping or imputing carefully"
                )

        state.missing_summary   = missing
        state.high_missing_cols = high_missing
        state.dqr_report["missing"] = missing

        self._info(f"Columns with >{self.missing_threshold:.0%} missing: {len(high_missing)}")
        if high_missing:
            self._info(f"  → {high_missing}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _consistency_checks(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        """Validate business-logic rules against the data (not just missing/outliers).
        Each rule is applied only when its columns exist, so this degrades cleanly
        on a Dataset 2 with a different schema."""
        rules = []

        # Rule 1: funded_amnt <= loan_amnt
        if 'funded_amnt' in df.columns and 'loan_amnt' in df.columns:
            funded = pd.to_numeric(df['funded_amnt'], errors='coerce')
            loan   = pd.to_numeric(df['loan_amnt'], errors='coerce')
            violations = int(((funded > loan) & funded.notna() & loan.notna()).sum())
            rules.append({
                'Rule': 'funded_amnt <= loan_amnt',
                'Description': 'Funded amount cannot exceed loan amount',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 2: int_rate in valid range
        if 'int_rate' in df.columns:
            rate = pd.to_numeric(df['int_rate'].astype(str).str.replace('%', ''), errors='coerce')
            violations = int(((rate < 0) | (rate > 100)).sum())
            rules.append({
                'Rule': 'int_rate ∈ [0, 100]',
                'Description': 'Interest rate must be a valid percentage',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 3: term is a standard value
        if 'term' in df.columns:
            term_clean = df['term'].astype(str).str.extract(r'(\d+)')[0]
            term_num   = pd.to_numeric(term_clean, errors='coerce')
            valid_terms = {36, 60}
            violations = int((~term_num.isin(valid_terms) & term_num.notna()).sum())
            rules.append({
                'Rule': 'term ∈ {36, 60}',
                'Description': 'Only standard loan terms expected',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 4: grade is a valid letter
        if 'grade' in df.columns:
            valid_grades = set('ABCDEFG')
            violations = int((~df['grade'].astype(str).isin(valid_grades) & df['grade'].notna()).sum())
            rules.append({
                'Rule': 'grade ∈ A–G',
                'Description': 'Only valid grade letters',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 5: dti >= 0
        if 'dti' in df.columns:
            dti = pd.to_numeric(df['dti'], errors='coerce')
            violations = int((dti < 0).sum())
            rules.append({
                'Rule': 'dti >= 0',
                'Description': 'Debt-to-income ratio cannot be negative',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 6: annual_inc >= 0
        if 'annual_inc' in df.columns:
            inc = pd.to_numeric(df['annual_inc'], errors='coerce')
            violations = int((inc < 0).sum())
            rules.append({
                'Rule': 'annual_inc >= 0',
                'Description': 'Income cannot be negative',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        # Rule 7: revol_util in a plausible range
        if 'revol_util' in df.columns:
            util = pd.to_numeric(df['revol_util'].astype(str).str.replace('%', ''), errors='coerce')
            violations = int(((util < 0) | (util > 200)).sum())  # >100 possible, >200 suspicious
            rules.append({
                'Rule': 'revol_util ∈ [0, 200]',
                'Description': 'Revolving utilisation must be a plausible percentage',
                'Violations': violations,
                'Status': 'PASS' if violations == 0 else 'FAIL',
            })

        applicable = len(rules)
        passed = sum(1 for r in rules if r['Status'] == 'PASS')
        state.dqr_report['consistency_checks'] = rules
        state.dqr_report['consistency_summary'] = {
            'total_rules': applicable,
            'passed': passed,
            'failed': applicable - passed,
            'note': f'{applicable} of 7 possible rules applicable to this dataset '
                    f'(some columns may not exist in Dataset 2)',
        }

        self._info(f"Consistency checks: {applicable} rules applicable | {passed} passed | {applicable - passed} failed")
        for r in rules:
            self._info(f"  {r['Rule']:<30} {r['Status']:<6} ({r['Violations']} violations)")

        return state

    # ─────────────────────────────────────────────────────────────
    def _outlier_detection(self, state: PipelineState,
                           df: pd.DataFrame, num_cols: list) -> PipelineState:
        outliers = {}
        for col in num_cols:
            if col not in df.columns:
                continue
            if df[col].dtype == object or str(df[col].dtype) == 'large_string':
                continue
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.dropna()
            if len(s) < 10:
                continue
            q1, q3  = s.quantile(0.25), s.quantile(0.75)
            iqr      = q3 - q1
            lo, hi   = q1 - 3 * iqr, q3 + 3 * iqr
            n_iqr    = int(((s < lo) | (s > hi)).sum())
            z        = np.abs(scipy_stats.zscore(s))
            n_z      = int((z > 4).sum())
            outliers[col] = {
                "iqr_outliers"    : n_iqr,
                "zscore_outliers" : n_z,
                "p1"  : round(float(s.quantile(0.01)), 4),
                "p99" : round(float(s.quantile(0.99)), 4),
                "min" : round(float(s.min()), 4),
                "max" : round(float(s.max()), 4),
            }
            if n_iqr > 0.05 * len(s):
                state.dqr_flags.append(
                    f"⚠  '{col}' has {n_iqr:,} IQR outliers ({n_iqr/len(s):.1%} of non-null)"
                )

        state.outlier_summary = outliers
        state.dqr_report["outliers"] = outliers
        self._info(f"Outlier detection complete for {len(outliers)} numeric columns")
        return state

    # ─────────────────────────────────────────────────────────────
    def _distribution_profiles(self, state: PipelineState,
                                df: pd.DataFrame, cols: list) -> PipelineState:
        profiles = {}
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if df[col].dtype in [object]:
                vc = s.value_counts()
                profiles[col] = {
                    "type"        : "categorical",
                    "n_unique"    : int(s.nunique()),
                    "top_5"       : vc.head(5).to_dict(),
                    "missing_pct" : round(float(df[col].isna().mean()), 4),
                }
            else:
                try:
                    s = pd.to_numeric(s, errors='coerce').dropna()
                    profiles[col] = {
                        "type"        : "numeric",
                        "mean"        : round(float(s.mean()), 4),
                        "std"         : round(float(s.std()), 4),
                        "p5"          : round(float(s.quantile(0.05)), 4),
                        "p25"         : round(float(s.quantile(0.25)), 4),
                        "median"      : round(float(s.median()), 4),
                        "p75"         : round(float(s.quantile(0.75)), 4),
                        "p95"         : round(float(s.quantile(0.95)), 4),
                        "skewness"    : round(float(s.skew()), 4),
                        "missing_pct" : round(float(df[col].isna().mean()), 4),
                    }
                except Exception:
                    pass
        state.dqr_report["distributions"] = profiles
        return state

    # ─────────────────────────────────────────────────────────────
    def _target_rates(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        """Compute default rate by quintile band for top numeric features."""
        target_rates = {}
        top_num_cols = [c for c in state.numeric_columns[:10]
                        if c in df.columns and df[c].notna().sum() > 100]

        for col in top_num_cols:
            try:
                df["_band"] = pd.qcut(df[col], q=5, duplicates="drop")
                rates = (df.groupby("_band", observed=True)["target"]
                         .agg(["mean", "count"])
                         .rename(columns={"mean": "default_rate", "count": "n"})
                         .reset_index())
                rates["_band"] = rates["_band"].astype(str)
                target_rates[col] = rates.to_dict(orient="records")
            except Exception:
                pass

        for col in state.categorical_columns[:5]:
            if col not in df.columns:
                continue
            try:
                rates = (df.groupby(col, observed=True)["target"]
                         .agg(["mean", "count"])
                         .rename(columns={"mean": "default_rate", "count": "n"})
                         .reset_index())
                target_rates[col] = rates.to_dict(orient="records")
            except Exception:
                pass

        if "_band" in df.columns:
            df.drop(columns=["_band"], inplace=True)

        state.dqr_report["target_rates"] = target_rates
        self._info(f"Target rates computed for {len(target_rates)} variables")
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_narrative(self, state: PipelineState) -> PipelineState:
        high_miss_info = {
            col: f"{v['pct_missing']:.0%} missing"
            for col, v in state.missing_summary.items()
            if v["pct_missing"] > 0.20
        }
        flags_text = "\n".join(state.dqr_flags[:15]) if state.dqr_flags else "None"
        n_obs = len(state.working_df)
        default_rate = state.working_df["target"].mean() if "target" in state.working_df else "unknown"

        prompt = f"""
You are reviewing a data quality report for a Lending Club consumer credit risk dataset.

KEY FACTS:
- Total observations: {n_obs:,}
- Default rate: {default_rate:.2%}
- High missing columns (>20%): {high_miss_info}
- DQR flags raised: {flags_text}
- Leakage columns identified and removed: {state.leakage_columns}

Write a concise DQR Executive Summary (max 250 words) for a credit risk model validator.
Cover:
1. Overall data quality assessment
2. Key concerns and how they should be addressed
3. Recommendation on whether data is fit for modelling
Be specific — reference actual column names and percentages from the facts above.
"""
        try:
            narrative = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=600)
            state.dqr_report["llm_narrative"] = narrative
            self._info("LLM DQR narrative generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM narrative skipped: {e}")

        return state
