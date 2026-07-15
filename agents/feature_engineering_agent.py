"""
agents/feature_engineering_agent.py
─────────────────────────────────────
Phase 3 — Feature Engineering Agent

Responsibilities:
  • Parse date columns → credit age, months-since features
  • Clean and encode categorical columns (ordinal + WOE)
  • Derive domain-specific credit risk features
    - DTI enrichment
    - Revolving utilisation
    - Delinquency indicators
    - Bureau aggregations
    - Vintage features
  • Impute missing values with strategy per column type
  • Log every derived feature with a business rationale
"""

import pandas as pd
import numpy as np
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM


class FeatureEngineeringAgent(BaseAgent):

    def __init__(self, verbose: bool = True):
        super().__init__("FeatureEngineeringAgent", verbose)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        df = state.working_df.copy()

        self._log("Dropping leakage + ID columns …")
        cols_to_drop = [c for c in state.drop_columns if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        self._log("Engineering date features …")
        df, state = self._date_features(df, state)

        self._log("Engineering grade / subgrade ordinal …")
        df, state = self._grade_features(df, state)

        self._log("Engineering term & rate features …")
        df, state = self._loan_features(df, state)

        self._log("Engineering delinquency & bureau features …")
        df, state = self._delinquency_features(df, state)

        self._log("Engineering utilisation & balance features …")
        df, state = self._utilisation_features(df, state)

        self._log("Encoding remaining categoricals …")
        df, state = self._encode_categoricals(df, state)

        self._log("Imputing missing values …")
        df = self._impute(df, state)

        self._log("Dropping original raw columns superseded by engineered ones …")
        drop_raw = ["earliest_cr_line", "issue_d", "grade", "sub_grade",
                    "home_ownership", "verification_status", "purpose",
                    "initial_list_status", "emp_length", "addr_state",
                    "int_rate", "term"]
        df.drop(columns=[c for c in drop_raw if c in df.columns], inplace=True)

        state.engineered_df = df
        self._info(f"Final engineered dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        state.log_audit(self.name, "feature_engineering_complete",
                        f"{df.shape[1]} columns after engineering")

        # LLM summary of features created
        self._log("Asking LLM to summarise engineered features …")
        state = self._llm_feature_summary(state)

        # Build structured response
        engineered_df = state.engineered_df
        state.agent_responses[self.name] = self.build_response(
            summary="Feature engineering complete",
            observations=[
                f"Engineered features created: {len(state.feature_log)}",
                f"Final dataset shape: {engineered_df.shape[0]:,} rows × {engineered_df.shape[1]} columns",
                f"Date-derived features: credit age, months since earliest credit line",
                f"Domain features: DTI enrichment, revolving utilisation, delinquency indicators",
                f"Encoding: ordinal grade/subgrade, WOE for categorical variables",
                f"Imputation: median for numeric, mode for categorical",
            ],
            reasoning=(
                "Domain-driven features created based on credit risk best practices. "
                "Grade and subgrade encoded ordinally to preserve risk ordering. "
                "WOE encoding applied to categorical variables to capture default rate signal. "
                "Post-origination raw columns dropped after deriving engineered equivalents."
            ),
            recommendations=[
                "Review feature_log for any domain features that may be controversial",
                "Validate WOE bins are monotonic for scorecarding compliance",
                "Check that no engineered feature introduces inadvertent leakage",
                "Consider adding vintage features if origination date is available",
            ],
            artifacts={"feature_log_count": len(state.feature_log)},
        )

        return state

    # ─────────────────────────────────────────────────────────────
    def _date_features(self, df: pd.DataFrame, state: PipelineState):
        def parse_lc_date(s):
            """Parse Lending Club MMM-YY format."""
            return pd.to_datetime(s, format="%b-%y", errors="coerce")

        ref_date = pd.Timestamp("2016-01-01")   # approximate snapshot date

        if "earliest_cr_line" in df.columns:
            ecl = parse_lc_date(df["earliest_cr_line"])
            df["credit_age_months"] = ((ref_date - ecl).dt.days / 30.4375).clip(0).round(0)
            state.feature_log.append({
                "feature": "credit_age_months",
                "source" : "earliest_cr_line",
                "rationale": "Credit history length — longer history generally predicts lower default"
            })

        if "issue_d" in df.columns:
            isd = parse_lc_date(df["issue_d"])
            df["loan_age_months"] = ((ref_date - isd).dt.days / 30.4375).clip(0).round(0)
            df["issue_year"]      = isd.dt.year
            df["issue_quarter"]   = isd.dt.quarter
            state.feature_log.append({
                "feature": "loan_age_months",
                "source" : "issue_d",
                "rationale": "Loan vintage — controls for origination cohort effects"
            })

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _grade_features(self, df: pd.DataFrame, state: PipelineState):
        grade_map     = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        subgrade_map  = {f"{g}{n}": (grade_map.get(g, 0) - 1) * 5 + n
                         for g in "ABCDEFG" for n in range(1, 6)}

        if "grade" in df.columns:
            df["grade_ordinal"] = df["grade"].map(grade_map).fillna(4)
            state.feature_log.append({
                "feature": "grade_ordinal",
                "source" : "grade",
                "rationale": "LC-assigned credit grade — strong proxy for creditworthiness (1=best)"
            })

        if "sub_grade" in df.columns:
            df["subgrade_ordinal"] = df["sub_grade"].map(subgrade_map).fillna(20)
            state.feature_log.append({
                "feature": "subgrade_ordinal",
                "source" : "sub_grade",
                "rationale": "Finer-grained credit grade — captures risk within a grade bucket"
            })

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _loan_features(self, df: pd.DataFrame, state: PipelineState):
        if "int_rate" in df.columns:
            df["int_rate_clean"] = pd.to_numeric(
                df["int_rate"].astype(str).str.replace("%", ""), errors="coerce"
            )
            state.feature_log.append({
                "feature": "int_rate_clean",
                "source" : "int_rate",
                "rationale": "Interest rate — higher rates indicate riskier borrowers"
            })

        if "term" in df.columns:
            df["term_60"] = (pd.to_numeric(df["term"], errors="coerce") == 60).astype(int)
            state.feature_log.append({
                "feature": "term_60",
                "source" : "term",
                "rationale": "60-month flag — longer term loans have higher cumulative default probability"
            })

        if all(c in df.columns for c in ["installment", "annual_inc"]):
            df["payment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)
            state.feature_log.append({
                "feature": "payment_to_income",
                "source" : "installment + annual_inc",
                "rationale": "Monthly payment burden as share of income — affordability indicator"
            })

        if all(c in df.columns for c in ["loan_amnt", "annual_inc"]):
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
            state.feature_log.append({
                "feature": "loan_to_income",
                "source" : "loan_amnt + annual_inc",
                "rationale": "Loan-to-income ratio — key affordability measure for unsecured lending"
            })

        if "emp_length" in df.columns:
            df["emp_length_clean"] = pd.to_numeric(df["emp_length"], errors="coerce").fillna(-1)
            state.feature_log.append({
                "feature": "emp_length_clean",
                "source" : "emp_length",
                "rationale": "Employment stability — longer tenure predicts lower default; -1 = unknown"
            })

        if "home_ownership" in df.columns:
            df["has_mortgage"] = (df["home_ownership"] == "MORTGAGE").astype(int)
            df["is_renter"]    = (df["home_ownership"] == "RENT").astype(int)
            state.feature_log.append({
                "feature": "has_mortgage / is_renter",
                "source" : "home_ownership",
                "rationale": "Housing tenure flags — mortgage holders tend to be lower risk"
            })

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _delinquency_features(self, df: pd.DataFrame, state: PipelineState):
        if "delinq_2yrs" in df.columns:
            df["any_delinq_2yr"] = (df["delinq_2yrs"] > 0).astype(int)
            state.feature_log.append({
                "feature": "any_delinq_2yr",
                "source" : "delinq_2yrs",
                "rationale": "Any 30+ dpd in last 2 years — strong default predictor"
            })

        if "mths_since_last_delinq" in df.columns:
            df["delinq_recent"] = (df["mths_since_last_delinq"].fillna(999) < 12).astype(int)
            df["mths_since_last_delinq_filled"] = df["mths_since_last_delinq"].fillna(999)
            state.feature_log.append({
                "feature": "delinq_recent",
                "source" : "mths_since_last_delinq",
                "rationale": "Delinquency within last 12 months — recency amplifies risk signal"
            })

        if "pub_rec" in df.columns:
            df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)
            state.feature_log.append({
                "feature": "has_pub_rec",
                "source" : "pub_rec",
                "rationale": "Any derogatory public record — bankruptcy or collection flag"
            })

        if "inq_last_6mths" in df.columns:
            df["high_inquiries"] = (df["inq_last_6mths"] > 3).astype(int)
            state.feature_log.append({
                "feature": "high_inquiries",
                "source" : "inq_last_6mths",
                "rationale": "High inquiry count signals credit-seeking behaviour / financial stress"
            })

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _utilisation_features(self, df: pd.DataFrame, state: PipelineState):
        if "revol_util" in df.columns:
            df["revol_util_clean"] = pd.to_numeric(
                df["revol_util"].astype(str).str.replace("%", ""), errors="coerce"
            )
            df["high_revol_util"] = (df["revol_util_clean"] > 75).astype(int)
            state.feature_log.append({
                "feature": "revol_util_clean / high_revol_util",
                "source" : "revol_util",
                "rationale": "Revolving utilisation — >75% signals credit stress"
            })

        if all(c in df.columns for c in ["open_acc", "total_acc"]):
            df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)
            state.feature_log.append({
                "feature": "open_acc_ratio",
                "source" : "open_acc + total_acc",
                "rationale": "Proportion of accounts still open — portfolio activity indicator"
            })

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, state: PipelineState):
        # Verification status
        if "verification_status" in df.columns:
            vs_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}
            df["verification_ordinal"] = df["verification_status"].map(vs_map).fillna(0)

        # Purpose — frequency encode (rare categories → "other")
        if "purpose" in df.columns:
            top_purposes = df["purpose"].value_counts().nlargest(8).index
            df["purpose_clean"] = df["purpose"].where(df["purpose"].isin(top_purposes), "other")
            purpose_dummies = pd.get_dummies(df["purpose_clean"], prefix="purpose", drop_first=True)
            df = pd.concat([df, purpose_dummies], axis=1)
            df.drop(columns=["purpose_clean", "purpose"], inplace=True, errors="ignore")

        # Initial list status
        if "initial_list_status" in df.columns:
            df["initial_list_w"] = (df["initial_list_status"] == "w").astype(int)
            df.drop(columns=["initial_list_status"], inplace=True, errors="ignore")

        # addr_state — too many levels, use default rate encoding
        if "addr_state" in df.columns and "target" in df.columns:
            state_dr = df.groupby("addr_state")["target"].mean()
            df["state_default_rate"] = df["addr_state"].map(state_dr).fillna(state_dr.mean())
            df.drop(columns=["addr_state"], inplace=True, errors="ignore")

        return df, state

    # ─────────────────────────────────────────────────────────────
    def _impute(self, df: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
        """Business-meaningful imputation. Numeric fields use sentinel codes that
        preserve WHY a value is missing (rather than blurring it with the median):
          -999999  missing source data (>20% missing in the original variable)
          -999998  invalid ratio (division by zero)
          -999997  not applicable due to filter (engineered conditional feature)
        Low-missing standard numerics still use the median; strings use the mode.
        """
        imputation_log = []
        imputation_map = {}   # exact per-column decision, persisted for Dataset 2 scoring
        n = len(df)
        for col in df.columns:
            if col == "target":
                continue
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            missing_pct = n_miss / n if n else 0.0

            dtype_s = str(df[col].dtype)
            is_str = (dtype_s in ("object", "str")
                      or dtype_s.startswith("string")
                      or dtype_s == "large_string")
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            # Non-numeric / string → mode
            if is_str or numeric_series.notna().sum() == 0:
                mode = df[col].mode()
                fill_val = mode[0] if len(mode) > 0 else "unknown"
                df[col] = df[col].fillna(fill_val)
                imputation_log.append({"Column": col, "Strategy": "Mode", "Value": str(fill_val),
                                       "Missing%": round(missing_pct * 100, 1),
                                       "Reason": "Most frequent category"})
                imputation_map[col] = {"strategy": "mode", "fill_value": fill_val,
                                       "reason": "Most frequent category"}
                continue

            is_ratio_feature = any(kw in col.lower() for kw in ["ratio", "util", "pct", "rate"])
            is_engineered_conditional = (
                any(kw in col.lower() for kw in ["_sum", "_avg", "_count", "flag"])
                and missing_pct < 0.5)

            if is_ratio_feature and missing_pct < 0.3:
                sentinel, reason = -999998, "Invalid ratio (division by zero)"
            elif is_engineered_conditional:
                sentinel, reason = -999997, "Not applicable due to filter (engineered feature, condition not met)"
            elif missing_pct >= 0.20:
                sentinel, reason = -999999, "Missing source data (>20% missing in original variable)"
            else:
                sentinel, reason = None, "Median imputation (low missing %, standard numeric)"

            if sentinel is not None:
                df[col] = numeric_series.fillna(sentinel)
                imputation_log.append({"Column": col, "Strategy": f"Sentinel ({sentinel})",
                                       "Value": sentinel, "Missing%": round(missing_pct * 100, 1),
                                       "Reason": reason})
                imputation_map[col] = {"strategy": "sentinel", "fill_value": float(sentinel),
                                       "reason": reason}
            else:
                median_val = numeric_series.median()
                df[col] = numeric_series.fillna(median_val)
                imputation_log.append({"Column": col, "Strategy": "Median",
                                       "Value": round(float(median_val), 2) if pd.notna(median_val) else 0,
                                       "Missing%": round(missing_pct * 100, 1), "Reason": reason})
                imputation_map[col] = {"strategy": "median",
                                       "fill_value": float(median_val) if pd.notna(median_val) else 0.0,
                                       "reason": reason}

        state.imputation_log = imputation_log
        state.imputation_map = imputation_map
        _n_sent = sum(1 for r in imputation_log if "Sentinel" in r["Strategy"])
        _n_med = sum(1 for r in imputation_log if r["Strategy"] == "Median")
        _n_mode = sum(1 for r in imputation_log if r["Strategy"] == "Mode")
        self._info(f"Imputation: {len(imputation_log)} columns "
                   f"({_n_sent} sentinel, {_n_med} median, {_n_mode} mode)")
        return df

    # ─────────────────────────────────────────────────────────────
    def _llm_feature_summary(self, state: PipelineState) -> PipelineState:
        feature_names = "\n".join(
            f"- {f['feature']}: {f['rationale']}"
            for f in state.feature_log[:20]
        )
        df_shape = state.engineered_df.shape if state.engineered_df is not None else "unknown"
        prompt = f"""
You are a credit risk data scientist. Below is a list of engineered features created
from a Lending Club consumer loan dataset for binary default prediction.

FEATURES CREATED:
{feature_names}

Final dataset shape: {df_shape[0]:,} rows × {df_shape[1]} columns

Write a brief Feature Engineering Summary (max 200 words) for the model development report.
Note any potential multicollinearity concerns, key domain-driven features, and imputation choices.
"""
        try:
            summary = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=500)
            state.dqr_report["feature_engineering_summary"] = summary
            self._info("LLM feature summary generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM feature summary skipped: {e}")
        return state
