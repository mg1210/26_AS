"""
agents/data_understanding_agent.py
────────────────────────────────────
Phase 1 — Data Understanding Agent

Responsibilities:
  • Load and profile the raw dataset
  • Auto-detect target column (no hardcoded column names)
  • Dynamically classify every unique target value as BAD / GOOD / INDETERMINATE
  • Build binary target series; exclude indeterminate rows
  • Flag post-origination / structural leakage columns
  • Detect behavioral leakage via univariate AUC scan
  • Infer semantic type for every column
  • Detect schema drift vs saved reference
  • Use LLM to generate business-meaning annotations
"""

import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask_json, ask, ask_json_with_usage, CREDIT_RISK_SYSTEM
from core.recommendation import Recommendation
from core.data_loader import smart_read_csv


# ── Known structural leakage — payment/recovery fields recorded after loan outcome ──
KNOWN_LEAKAGE = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
]

# Identifier / admin columns — carry no predictive signal
ID_COLS = [
    "Record_No", "id", "member_id", "url", "desc", "title",
    "zip_code", "emp_title", "pymnt_plan", "policy_code",
]

# Columns that are single-value constants in many standard pulls
ALWAYS_DROP = ["application_type"]

# ── Target value classification patterns (checked case-insensitively, substring) ──
BAD_PATTERNS = [
    "charged off", "default", "write off", "written off", "bad",
    "loss", "npa", "write-off", "defaulted",
]
GOOD_PATTERNS = [
    "fully paid", "paid", "good", "settled", "closed", "repaid", "completed",
]
INDETERMINATE_PATTERNS = [
    "current", "late", "grace", "in progress", "pending", "ongoing",
    "active", "open", "processing", "issued", "in review",
]

# Priority-ordered names to check for target column auto-detection
TARGET_EXACT_NAMES = [
    "loan_status", "status", "target", "default", "is_default", "bad_flag", "outcome",
]
TARGET_PARTIAL_PATTERNS = ["status", "default", "target", "outcome", "bad"]


class DataUnderstandingAgent(BaseAgent):

    def __init__(self, verbose: bool = True, data_dict_path: str = ""):
        super().__init__("DataUnderstandingAgent", verbose)
        self.data_dict_path = data_dict_path

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:

        # 0. Load data dictionary if available
        self._log("Loading data dictionary if available …")
        import glob as _glob
        _dict_files = (
            _glob.glob("data/*.xlsx")
            + _glob.glob("data/*dict*.xlsx")
            + _glob.glob("data/*dictionary*.xlsx")
        )
        if _dict_files:
            self._info(f"Data dictionary found: {_dict_files[0]}")
        else:
            self._info("No data dictionary found in data/ folder — using LLM annotation or name inference")
        state = self._load_data_dictionary(state)

        # 1. Load data (header-aware — headerless blind datasets get col_0, col_1, … names)
        self._log("Loading dataset …")
        df, state.headerless = smart_read_csv(state.dataset_path)
        state.raw_df   = df.copy()
        state.working_df = df.copy()
        self._info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        if state.headerless:
            self._info("No header row detected — columns auto-named col_0, col_1, … for a headerless dataset")
        state.log_audit(self.name, "data_loaded",
                        f"{df.shape[0]} rows, {df.shape[1]} cols, headerless={state.headerless}")

        # 2. Auto-detect and define target variable
        self._log("Auto-detecting target column …")
        state = self._detect_and_define_target(state)

        # 3. Classify columns
        self._log("Classifying columns …")
        state = self._classify_columns(state, df)

        # 4. Build schema profile with semantic types
        self._log("Building schema profile with semantic types …")
        state = self._build_schema_profile(state, df)

        # 5. Behavioral leakage detection (AUC scan)
        self._log("Scanning for behavioral leakage (univariate AUC) …")
        state = self._behavioral_leakage_detection(state)

        # Schema drift disabled — can be re-enabled for Dataset 2 comparison
        # self._log("Checking schema drift …")
        # state = self._check_schema_drift(state, df)

        # 7. LLM annotations
        self._log("Asking LLM to annotate field business meanings …")
        state = self._llm_annotate(state, df)

        # 8. Build structured agent response
        wdf          = state.working_df
        default_rate = wdf["target"].mean() if "target" in wdf.columns else 0.0
        n_leakage_beh = len(state.behavioral_leakage_flags)
        state.agent_responses[self.name] = self.build_response(
            summary="Dataset profiled successfully",
            observations=[
                f"Loaded {wdf.shape[0]:,} rows × {df.shape[1]} columns",
                f"Target column detected: {state.target_column!r} "
                f"({state.schema_drift.get('detection_method', '?')})",
                f"Default rate: {default_rate:.2%}",
                f"Indeterminate rows excluded: {sum(v['count'] for v in state.indeterminate_values):,}",
                f"Structural leakage columns removed: {len(state.leakage_columns)}",
                f"Behavioral leakage columns flagged (AUC ≥ 0.65): {n_leakage_beh}",
                f"Date columns identified: {len(state.date_columns)}",
                f"Categorical columns: {len(state.categorical_columns)}",
                f"Numeric columns: {len(state.numeric_columns)}",
                f"Schema drift: {state.schema_drift.get('status', 'unknown')}",
            ],
            reasoning=(
                f"Target auto-detected as '{state.target_column}'. "
                f"Values classified dynamically: BAD→1, GOOD→0, INDETERMINATE→excluded. "
                f"Behavioral leakage scan ran univariate AUC on all numeric columns; "
                f"{n_leakage_beh} columns scored ≥ 0.65 and should be reviewed."
            ),
            recommendations=[
                "Confirm target definition with business stakeholders before modelling",
                "Review leakage column list — ensure all post-origination fields are excluded",
                f"Investigate {n_leakage_beh} behavioral leakage candidates before feature engineering",
                f"Review {len(state.id_columns)} identifier columns — verify none carry signal",
                "Investigate date columns for credit age feature derivation",
            ],
        )

        state.recommendations.append(Recommendation(
            title="Target Variable Definition",
            recommendation=state.target_definition,
            rationale=(
                f"Target auto-detected as '{state.target_column}'. "
                f"Dynamic value classification applied: "
                + ", ".join(
                    f"{v['value']}→{'BAD(1)' if state.target_mapping.get(v['value']) == 1 else 'GOOD(0)' if state.target_mapping.get(v['value']) == 0 else 'EXCL'}"
                    for v in state.indeterminate_values[:3]
                )
                + (f" + {len(state.indeterminate_values)-3} more excluded" if len(state.indeterminate_values) > 3 else "")
            ),
            confidence=0.90,
            risk="medium",
            requires_human_approval=True,
        ))

        return state

    # ─────────────────────────────────────────────────────────────
    # STEP 1 + 2 + 3 — Target detection, classification, and binary mapping
    # ─────────────────────────────────────────────────────────────

    def _target_likeness_score(self, series: pd.Series):
        """Score how target-like a column's VALUES are, using the credit-outcome
        vocabulary (BAD/GOOD/INDETERMINATE patterns). Returns
        (score, has_bad, has_good). A binary outcome column that carries both a
        BAD side (e.g. 'Charged Off') and a GOOD side (e.g. 'Fully Paid') scores
        highest; unrelated low-cardinality columns (grade, home_ownership) score 0.
        """
        try:
            vals = series.dropna().astype(str).str.strip().str.lower().unique()
        except Exception:
            return 0, False, False
        if not (2 <= len(vals) <= 25):
            return 0, False, False
        has_bad  = any(any(p in v for p in BAD_PATTERNS)  for v in vals)
        has_good = any(any(p in v for p in GOOD_PATTERNS) for v in vals)
        matched  = sum(
            any(p in v for p in (BAD_PATTERNS + GOOD_PATTERNS + INDETERMINATE_PATTERNS))
            for v in vals
        )
        score = matched + (5 if (has_bad and has_good) else 0)
        return score, has_bad, has_good

    def _detect_target_column(self, df: pd.DataFrame):
        """Return (column_name, detection_method). Never hardcodes column names."""
        cols_lower = {c.lower(): c for c in df.columns}

        # Priority 1: exact name match
        for name in TARGET_EXACT_NAMES:
            if name in cols_lower:
                return cols_lower[name], f"exact name match ({name!r})"

        # Priority 2: partial name match
        for col in df.columns:
            col_l = col.lower()
            for pat in TARGET_PARTIAL_PATTERNS:
                if pat in col_l:
                    return col, f"partial name match (pattern={pat!r} in {col!r})"

        # Priority 3: value-based detection. Score every column by how target-like
        # its VALUES are — essential for headerless / blind data where column names
        # carry no signal (prevents picking 'grade' A–G over a real 'Charged Off /
        # Fully Paid' outcome column).
        best_col, best_method, best_score = None, "", 0
        for col in df.columns:
            score, has_bad, has_good = self._target_likeness_score(df[col])
            qualifies = (has_bad and has_good) or ((has_bad or has_good) and score >= 3)
            if qualifies and score > best_score:
                best_col, best_score = col, score
                best_method = f"value-based match ({col!r}: outcome-like values, score={score})"
        if best_col is not None:
            return best_col, best_method

        # Priority 4: any string/object column with low cardinality (last resort)
        # pandas 3.x uses StringDtype (str(dtype)='str'); earlier versions use object
        for col in df.columns:
            dtype_s = str(df[col].dtype)
            is_str  = dtype_s in ("object", "str") or dtype_s.startswith("string")
            if is_str and 2 <= df[col].nunique() <= 20:
                return col, f"low-cardinality string column ({col!r}, {df[col].nunique()} unique values)"

        raise ValueError(
            "Could not auto-detect target column. "
            "Ensure your dataset has a column named: loan_status, status, target, default, "
            "is_default, bad_flag, or outcome — or any string column with ≤20 unique values."
        )

    def _classify_value(self, val: str):
        """Classify a single target value as BAD / GOOD / INDETERMINATE."""
        v = str(val).lower().strip()
        for pat in BAD_PATTERNS:
            if pat in v:
                return "BAD", f"Matches bad pattern: '{pat}'"
        for pat in GOOD_PATTERNS:
            if pat in v:
                return "GOOD", f"Matches good pattern: '{pat}'"
        for pat in INDETERMINATE_PATTERNS:
            if pat in v:
                return "INDETERMINATE", f"Matches indeterminate pattern: '{pat}'"
        return "INDETERMINATE", "Unknown outcome — excluded as ambiguous"

    def _detect_and_define_target(self, state: PipelineState) -> PipelineState:
        df = state.working_df

        # Step 1 — detect column (or use analyst override if pre-set)
        if state.target_column and state.target_column in df.columns:
            target_col       = state.target_column
            detection_method = "analyst_override"
            self._info(f"Using analyst-overridden target column: {target_col!r}")
        else:
            target_col, detection_method = self._detect_target_column(df)
            self._info(f"Target column detected: {target_col!r}  (method: {detection_method})")

        # Step 2 — classify every unique value
        value_counts = df[target_col].value_counts(dropna=False)
        target_mapping    = {}   # raw_value → 0/1/None
        indeterminate_vals = []

        self._info("Value classification:")
        for val, count in value_counts.items():
            classification, reason = self._classify_value(val)
            label = 1 if classification == "BAD" else (0 if classification == "GOOD" else None)
            target_mapping[val] = label

            self._info(f"  {classification:<15} {str(val):<50} {count:>8,} obs")

            if classification == "INDETERMINATE":
                indeterminate_vals.append({
                    "value": val,
                    "count": int(count),
                    "reason": reason,
                })

        # Step 3 — build binary target series
        before = len(df)
        df = df[df[target_col].map(lambda v: target_mapping.get(v) is not None)].copy()
        df["target"] = df[target_col].map(target_mapping).astype(int)
        after = len(df)

        excluded_count = before - after
        excluded_pct   = excluded_count / before if before > 0 else 0.0
        default_rate   = df["target"].mean()

        self._info(f"Final sample: {after:,} | Default rate: {default_rate:.2%}")
        self._info(f"Excluded (indeterminate): {excluded_count:,} ({excluded_pct:.1%} of raw)")

        # Step 4 — build target definition dynamically from actual data
        bad_vals  = [str(v) for v, lbl in target_mapping.items() if lbl == 1]
        good_vals = [str(v) for v, lbl in target_mapping.items() if lbl == 0]
        target_def = (
            f"Binary default flag derived from '{target_col}'. "
            f"BAD (1): {', '.join(bad_vals)}. "
            f"GOOD (0): {', '.join(good_vals)}. "
            f"Default rate: {default_rate:.2%}. "
            f"Excluded indeterminate: {excluded_count:,} rows ({excluded_pct:.1%}). "
            f"Final sample: {after:,} observations."
        )

        # Step 5 — store everything in state
        state.working_df           = df
        state.target_column        = target_col
        state.target_mapping       = target_mapping
        state.indeterminate_values = indeterminate_vals
        state.target_definition    = target_def
        # stash detection method on schema_drift dict for easy access
        state.schema_drift["detection_method"] = detection_method

        state.log_audit(self.name, "target_defined", target_def)
        return state

    # ─────────────────────────────────────────────────────────────
    # Column classification (structural leakage, ids, dates, etc.)
    # ─────────────────────────────────────────────────────────────

    def _looks_like_identifier(self, series: pd.Series, n_rows: int) -> bool:
        """Heuristic identifier detection that does NOT rely on the column name —
        essential for headerless data where columns are named col_0, col_1, … and
        the name-based ID_COLS list cannot match.

        A column is flagged as a likely identifier when either:
          A. its values are essentially unique (cardinality > 98% of all rows) and
             it is an integer-coded or string column. A true ID is ~100% unique;
             the high threshold keeps genuine integer features (e.g. revolving
             balance) — which can exceed 90% uniqueness on a SMALL dataset — from
             being mistaken for IDs; or
          B. its values are a monotonically increasing integer sequence with a
             constant step (e.g. Record_No = 1, 2, 3, …) — a running record number,
             even if not near-unique.
        """
        s = series.dropna()
        n = len(s)
        if n < 20 or n_rows == 0:
            return False

        dtype_s = str(series.dtype)
        is_str  = dtype_s in ("object", "str") or dtype_s.startswith("string")
        num     = pd.to_numeric(s, errors="coerce")
        num_ok  = num.notna().mean() > 0.99
        num_nn  = num.dropna()
        is_int_valued = bool(num_ok and len(num_nn) > 0 and (num_nn == num_nn.round()).all())

        # Rule A — essentially-unique cardinality over the full row count.
        if (is_str or is_int_valued) and s.nunique() / n_rows > 0.98:
            return True

        # Rule B — monotonically increasing integer sequence with a constant step.
        if is_int_valued and len(num_nn) > 2:
            arr = num_nn.to_numpy()
            if (arr[1:] > arr[:-1]).mean() > 0.99:              # strictly increasing
                diffs = arr[1:] - arr[:-1]
                if pd.Series(diffs).nunique() == 1:             # constant step (e.g. +1)
                    return True
        return False

    def _classify_columns(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        target_col = state.target_column
        state.leakage_columns = [c for c in KNOWN_LEAKAGE if c in df.columns]

        name_ids  = [c for c in ID_COLS if c in df.columns]
        # Value/structure-based identifier scan — catches IDs even when the column
        # name gives no hint (headerless data), preventing identifier leakage into
        # the feature set.
        protected = {target_col, "target", "loan_status"}
        n_rows    = len(df)
        value_ids = [
            c for c in df.columns
            if c not in protected and c not in name_ids
            and self._looks_like_identifier(df[c], n_rows)
        ]
        state.id_columns   = list(dict.fromkeys(name_ids + value_ids))
        state.drop_columns = state.leakage_columns + state.id_columns + ALWAYS_DROP
        if value_ids:
            self._info(f"Structure-based identifier columns flagged: {value_ids}")

        remaining = [
            c for c in df.columns
            if c not in state.drop_columns
            and c not in (target_col, "target", "loan_status")
        ]

        date_keywords = ["_d", "earliest_cr", "issue_d", "payment_date"]
        def _is_str(col):
            ds = str(df[col].dtype)
            return ds in ("object", "str") or ds.startswith("string")

        state.date_columns = [
            c for c in remaining
            if any(k in c for k in date_keywords)
            or (_is_str(c) and
                df[c].dropna().head(20).astype(str)
                .str.match(r'^[A-Z][a-z]{2}-\d{2}$').mean() > 0.5)
        ]
        state.categorical_columns = [
            c for c in remaining
            if c not in state.date_columns
            and (_is_str(c) or df[c].nunique() < 15)
        ]
        state.numeric_columns = [
            c for c in remaining
            if c not in state.date_columns
            and c not in state.categorical_columns
        ]

        self._info(f"Structural leakage cols : {len(state.leakage_columns)}")
        self._info(f"ID / admin cols         : {len(state.id_columns)}")
        self._info(f"Date cols               : {len(state.date_columns)}")
        self._info(f"Categorical cols        : {len(state.categorical_columns)}")
        self._info(f"Numeric cols            : {len(state.numeric_columns)}")
        return state

    # ─────────────────────────────────────────────────────────────
    # STEP 7 — Schema profile + semantic type inference
    # ─────────────────────────────────────────────────────────────

    def _infer_semantic_type(self, col: str, series: pd.Series) -> str:
        """Infer semantic type from dtype, cardinality, and column name."""
        n_unique  = series.nunique()
        dtype_str = str(series.dtype)
        col_l     = col.lower()

        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                return "binary_numeric"
            if any(k in col_l for k in ("amount", "loan", "balance", "principal", "income")):
                return "monetary"
            if any(k in col_l for k in ("rate", "ratio", "pct", "percent", "dti")):
                return "ratio"
            if any(k in col_l for k in ("_d", "date", "mths", "month", "year")):
                return "temporal_numeric"
            if any(k in col_l for k in ("id", "number", "no", "record")):
                return "identifier"
            if n_unique < 20:
                return "ordinal"
            return "continuous"

        dtype_s      = str(series.dtype)
        is_str_dtype = dtype_s in ("object", "str") or dtype_s.startswith("string")
        if is_str_dtype:
            if n_unique == 2:
                return "binary_categorical"
            if n_unique <= 20:
                return "low_cardinality_categorical"
            return "high_cardinality_categorical"

        return "other"

    def _load_data_dictionary(self, state: PipelineState) -> PipelineState:
        """Load business meanings from an xlsx data dictionary in the data/ folder."""
        # Use explicit path first, then scan data/ folder
        candidates = []
        if self.data_dict_path and os.path.exists(self.data_dict_path):
            candidates = [self.data_dict_path]
        else:
            data_dir = os.path.dirname(state.dataset_path) or "data"
            if os.path.isdir(data_dir):
                for f in os.listdir(data_dir):
                    if f.lower().endswith(".xlsx"):
                        candidates.append(os.path.join(data_dir, f))

        if not candidates:
            self._info("No data dictionary xlsx found in data/ — business meanings from LLM only")
            return state

        path = candidates[0]
        try:
            xdf = pd.read_excel(path)
            self._info(f"Data dictionary columns found: {xdf.columns.tolist()}")

            col_lower = {c.lower().strip(): c for c in xdf.columns}

            # Field name column — ordered by specificity
            field_col = None
            for candidate in ["loanstatnew", "field", "column", "variable",
                               "name", "feature", "attribute", "col"]:
                if candidate in col_lower:
                    field_col = col_lower[candidate]
                    break
            if field_col is None and len(xdf.columns) >= 1:
                field_col = xdf.columns[0]

            # Description column
            desc_col = None
            for candidate in ["description", "meaning", "definition",
                               "desc", "label", "explanation", "details"]:
                if candidate in col_lower:
                    desc_col = col_lower[candidate]
                    break
            if desc_col is None and len(xdf.columns) >= 2:
                desc_col = xdf.columns[1]

            if field_col and desc_col:
                dd = {
                    str(row[field_col]).strip().lower(): str(row[desc_col]).strip()
                    for _, row in xdf.iterrows()
                    if pd.notna(row[field_col]) and pd.notna(row[desc_col])
                    and str(row[field_col]).strip() not in ("", "nan")
                }
                state.data_dictionary = dd
                self._info(
                    f"Data dictionary loaded: {len(dd)} entries "
                    f"from columns '{field_col}' → '{desc_col}'"
                )
            else:
                self._info(
                    f"Data dictionary xlsx found ({path}) but could not identify "
                    f"field/description columns. Columns found: {list(xdf.columns)}"
                )
        except Exception as e:
            state.log_warning(self.name, f"Could not load data dictionary {path}: {e}")
            self._info(f"Data dictionary load failed: {e}")
        return state

    _ABBREV_PATTERNS = [
        (r'\b(\d+)yrs?\b',  r'in past \1 years'),
        (r'\b(\d+)mths?\b', r'in past \1 months'),
        (r'\binc\b',    'income'),
        (r'\bamt\b',    'amount'),
        (r'\bbal\b',    'balance'),
        (r'\butil\b',   'utilisation'),
        (r'\bpct\b',    'percentage'),
        (r'\bprncp\b',  'principal'),
        (r'\bpymnt\b',  'payment'),
        (r'\bacc\b',    'accounts'),
        (r'\binq\b',    'inquiries'),
        (r'\bdelinq\b', 'delinquencies'),
        (r'\bpub\b',    'public'),
        (r'\brec\b',    'record'),
        (r'\brev\b',    'revolving'),
        (r'\brevol\b',  'revolving'),
        (r'\btot\b',    'total'),
        (r'\bcur\b',    'current'),
        (r'\bhi\b',     'high'),
        (r'\blim\b',    'limit'),
        (r'\bint\b',    'interest'),
        (r'\bverif\b',  'verification'),
        (r'\bdti\b',    'debt-to-income ratio'),
        (r'\bmths\b',   'months'),
        (r'\bsince\b',  'since last'),
        (r'\brcnt\b',   'recent'),
        (r'\bop\b',     'open'),
        (r'\bil\b',     'installment loan'),
        (r'\brv\b',     'revolving'),
        (r'\bbc\b',     'bankcard'),
        (r'\bsat\b',    'satisfactory'),
        (r'\bfi\b',     'finance'),
        (r'\bcu\b',     'credit union'),
        (r'\btl\b',     'tradeline'),
    ]

    def _infer_meaning_from_name(self, col: str) -> str:
        """Generate a human-readable business meaning from a column name."""
        text = col.replace("_", " ").lower()
        for pattern, replacement in self._ABBREV_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.title()

    def _build_schema_profile(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        profile = {}
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            semantic    = self._infer_semantic_type(col, df[col])
            # Business meaning priority: data_dictionary > LLM annotation > name inference
            # data_dictionary keys are stored lowercase; try lowercase then original
            _col_key = col.lower().strip()
            business_meaning = (
                state.data_dictionary.get(_col_key)
                or state.data_dictionary.get(col)
                or state.schema_profile.get(col, {}).get("business_meaning", "")
                or self._infer_meaning_from_name(col)
            )
            profile[col] = {
                "dtype"            : str(df[col].dtype),
                "semantic_type"    : semantic,
                "missing_pct"      : round(float(missing_pct), 4),
                "n_unique"         : int(df[col].nunique()),
                "sample_vals"      : df[col].dropna().head(3).astype(str).tolist(),
                "business_meaning" : business_meaning,
                "role"             : (
                    "leakage"     if col in state.leakage_columns else
                    "identifier"  if col in state.id_columns else
                    "date"        if col in state.date_columns else
                    "categorical" if col in state.categorical_columns else
                    "numeric"     if col in state.numeric_columns else
                    "target"      if col in (state.target_column, "target") else
                    "other"
                ),
            }
        state.schema_profile = profile
        _from_dict = sum(1 for col in df.columns if state.data_dictionary.get(col, ""))
        _from_llm  = sum(
            1 for col in df.columns
            if not state.data_dictionary.get(col, "")
            and state.schema_profile.get(col, {}).get("business_meaning", "")  # pre-existing LLM
        )
        self._info(
            f"Business meanings populated: {_from_dict} from data dictionary, "
            f"{_from_llm} from prior LLM annotation, "
            f"{len(df.columns) - _from_dict - _from_llm} inferred from column name"
        )
        return state

    # ─────────────────────────────────────────────────────────────
    # STEP 6 — Behavioral leakage detection via univariate AUC
    # ─────────────────────────────────────────────────────────────

    def _behavioral_leakage_detection(self, state: PipelineState) -> PipelineState:
        wdf = state.working_df
        if "target" not in wdf.columns:
            return state

        # Structural leakage (set earlier by _classify_columns) is excluded up-front
        # so a column is never flagged by BOTH the structural and behavioral checks.
        structural_leakage_cols = set(state.leakage_columns or [])

        y = wdf["target"]
        numeric_cols = [
            c for c in wdf.select_dtypes(include=[np.number]).columns
            if c != "target"
        ]

        # Sample for speed
        n_sample = min(50_000, len(wdf))
        sample   = wdf.sample(n_sample, random_state=42)
        y_s      = sample["target"]

        flags = {}
        for col in numeric_cols:
            if col in structural_leakage_cols:
                continue   # already caught by structural check — skip to avoid duplicate flagging
            if col == "target":
                continue
            try:
                x = sample[col].fillna(sample[col].median())
                if x.nunique() < 2:
                    continue
                auc = roc_auc_score(y_s, x)
                auc = max(auc, 1.0 - auc)   # normalise direction
                if auc >= 0.65:
                    flags[col] = round(float(auc), 4)
            except Exception:
                pass

        state.behavioral_leakage_flags = flags

        # Confirm the two leakage lists do not overlap.
        overlap = set(flags.keys()) & structural_leakage_cols
        if overlap:
            self._info(f"WARNING: {len(overlap)} columns would have appeared in both lists — "
                       f"excluded from behavioral (already structural): {overlap}")
        else:
            self._info(f"No overlap between structural and behavioral leakage lists — "
                       f"{len(structural_leakage_cols)} structural, {len(flags)} behavioral, 0 shared")

        self._info(f"Behavioral leakage: {len(flags)} columns flagged (AUC ≥ 0.65 on {n_sample:,} sample)")
        for col, auc in sorted(flags.items(), key=lambda x: -x[1])[:10]:
            self._info(f"  {col}: AUC={auc:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    # STEP 8 — Schema drift detection
    # ─────────────────────────────────────────────────────────────

    def _check_schema_drift(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        ref_path = "outputs/reference_schema.json"
        current  = {col: str(df[col].dtype) for col in df.columns}

        # Preserve detection_method already stored on schema_drift
        detection_method = state.schema_drift.get("detection_method", "")

        if not os.path.exists(ref_path):
            os.makedirs("outputs", exist_ok=True)
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
            state.schema_drift = {
                "status":           "first_run",
                "saved_reference":  True,
                "detection_method": detection_method,
            }
            self._info("Schema reference saved (first run — no drift to compare)")
            return state

        with open(ref_path, encoding="utf-8") as f:
            reference = json.load(f)

        new_cols     = [c for c in current   if c not in reference]
        missing_cols = [c for c in reference if c not in current]
        type_changes = {
            c: {"from": reference[c], "to": current[c]}
            for c in current
            if c in reference and current[c] != reference[c]
        }

        has_drift = bool(new_cols or missing_cols or type_changes)
        state.schema_drift = {
            "status":          "drift_detected" if has_drift else "no_drift",
            "new_columns":     new_cols,
            "missing_columns": missing_cols,
            "type_changes":    type_changes,
            "detection_method": detection_method,
        }

        if new_cols:
            self._info(f"Schema drift — {len(new_cols)} new columns: {new_cols[:5]}"
                       + (" …" if len(new_cols) > 5 else ""))
        if missing_cols:
            self._info(f"Schema drift — {len(missing_cols)} missing columns: {missing_cols[:5]}"
                       + (" …" if len(missing_cols) > 5 else ""))
        if type_changes:
            for col, chg in list(type_changes.items())[:5]:
                self._info(f"Schema drift — type change: {col}: {chg['from']} → {chg['to']}")
        if not has_drift:
            self._info("Schema drift: no changes detected")
        return state

    # ─────────────────────────────────────────────────────────────
    # LLM annotation
    # ─────────────────────────────────────────────────────────────

    def _llm_annotate(self, state: PipelineState, df: pd.DataFrame) -> PipelineState:
        candidate_cols = state.numeric_columns[:20] + state.categorical_columns[:10]
        col_list = "\n".join(
            f"- {c} (dtype={df[c].dtype}, missing={df[c].isna().mean():.1%}, "
            f"sample={df[c].dropna().head(2).tolist()})"
            for c in candidate_cols if c in df.columns
        )
        prompt = f"""
You are reviewing a dataset for credit risk modelling.
Here are the candidate modelling columns:

{col_list}

For each column, provide a one-line business interpretation and classify it as one of:
application_info, credit_bureau, loan_terms, behavioural, or other.

Respond ONLY with a JSON object:
{{"column_name": {{"meaning": "...", "category": "..."}} }}
"""
        try:
            annotations, _usage = ask_json_with_usage(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=2000)
            state.llm_token_usage[self.name] = _usage
            for col, info in annotations.items():
                if col in state.schema_profile:
                    state.schema_profile[col]["business_meaning"] = info.get("meaning", "")
                    state.schema_profile[col]["category"]         = info.get("category", "")
            self._info(f"LLM annotated {len(annotations)} columns")
        except Exception as e:
            state.log_warning(self.name, f"LLM annotation skipped: {e}")
            self._info(f"LLM annotation skipped — {e}")
        return state
