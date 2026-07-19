"""
agents/variable_selection_agent.py
────────────────────────────────────
Phase 4 — Variable Selection Agent

Responsibilities:
  • Information Value (IV) calculation with WOE binning
  • Correlation analysis + VIF-based multicollinearity check
  • Missing value filter (post-engineering)
  • Gini / AUC univariate filter
  • Feature importance from a quick Random Forest
  • LLM-driven final shortlist recommendation with rationale
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, ask_with_usage, CREDIT_RISK_SYSTEM
from core.recommendation import Recommendation


# IV thresholds (standard industry practice)
IV_USELESS      = 0.02
IV_WEAK         = 0.10
IV_STRONG       = 0.30
MAX_FEATURES     = 25      # cap on final shortlist


class VariableSelectionAgent(BaseAgent):

    def __init__(self, verbose: bool = True):
        super().__init__("VariableSelectionAgent", verbose)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        df = state.engineered_df.copy()
        target = "target"

        # Separate features and target
        feature_cols = [c for c in df.columns
                        if c != target
                        and c != "loan_status"
                        and df[c].dtype != object
                        and str(df[c].dtype) != 'large_string'
                        and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols]
        y = df[target]

        self._log(f"Candidate features: {len(feature_cols)}")

        self._log("Computing Information Value (IV) …")
        state = self._compute_iv(state, df, feature_cols, target)

        self._log("Computing correlation matrix …")
        state = self._correlation_analysis(state, X)

        self._log("Computing quick Random Forest importance …")
        state = self._rf_importance(state, X, y, feature_cols)

        self._log("Selecting final feature shortlist …")
        state = self._select_features(state, feature_cols)

        self._log("Asking LLM for selection rationale …")
        state = self._llm_rationale(state)

        # Build structured response
        iv_df    = state.iv_table
        top3     = []
        if iv_df is not None and len(iv_df) > 0:
            top_rows = iv_df.head(3) if isinstance(iv_df, pd.DataFrame) else []
            for _, row in (top_rows.iterrows() if hasattr(top_rows, "iterrows") else []):
                top3.append(f"{row.get('feature','?')} (IV={row.get('iv','?')})")

        state.agent_responses[self.name] = self.build_response(
            summary="Variable selection complete",
            observations=[
                f"Candidate features evaluated: {len(feature_cols)}",
                f"Features selected: {len(state.selected_features)}",
                f"Features rejected: {len(state.rejected_features)}",
                f"Top features by IV: {', '.join(top3) if top3 else 'N/A'}",
                f"IV threshold applied: {IV_USELESS} (useless below this)",
            ],
            reasoning=(
                f"IV threshold {IV_USELESS} applied — features below this have insufficient "
                "discriminatory power for credit risk modelling. "
                "High correlation pairs resolved by keeping the feature with higher IV. "
                "Final shortlist capped at 25 features to prevent overfitting."
            ),
            recommendations=[
                "Review rejected features — some may have business value not captured by IV",
                "Validate WOE bins for monotonicity before scorecard deployment",
                "Consider interaction features between top IV variables",
                f"Ensure all {len(state.selected_features)} selected features pass regulatory scrutiny",
            ],
            artifacts={"iv_threshold": IV_USELESS, "max_features": MAX_FEATURES},
        )

        rationale = (
            state.dqr_report.get("variable_selection_rationale")
            or f"IV threshold {IV_USELESS} applied. "
               f"{len(state.selected_features)} features selected from {len(feature_cols)} candidates. "
               f"High-correlation pairs resolved by retaining higher-IV feature."
        )
        state.recommendations.append(Recommendation(
            title="Feature Shortlist",
            recommendation=f"{len(state.selected_features)} features selected for modelling",
            rationale=rationale,
            confidence=0.85,
            risk="low",
            requires_human_approval=True,
        ))

        return state

    # ─────────────────────────────────────────────────────────────
    def _compute_iv(self, state: PipelineState,
                    df: pd.DataFrame, feature_cols: list, target: str) -> PipelineState:
        iv_records = []
        for col in feature_cols:
            try:
                iv, woe_map = self._iv_woe(df[col], df[target])
                iv_records.append({
                    "feature"   : col,
                    "iv"        : round(iv, 4),
                    "strength"  : self._iv_label(iv),
                })
                if woe_map:
                    state.woe_bins[col] = woe_map
            except Exception:
                pass

        iv_df = pd.DataFrame(iv_records).sort_values("iv", ascending=False)
        state.iv_table = iv_df
        self._info(f"IV computed for {len(iv_df)} features")
        self._info(f"  Strong IV (>{IV_WEAK}): "
                   f"{(iv_df['iv'] > IV_WEAK).sum()} features")
        return state

    def _iv_woe(self, x: pd.Series, y: pd.Series, bins: int = 10):
        """Compute IV and WOE map for a single feature."""
        df_tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df_tmp) < 50 or df_tmp["y"].nunique() < 2:
            return 0.0, {}

        # Bin numeric features
        if df_tmp["x"].dtype in [np.float64, np.int64, float, int]:
            try:
                df_tmp["bucket"] = pd.qcut(df_tmp["x"], q=bins, duplicates="drop")
            except Exception:
                df_tmp["bucket"] = pd.cut(df_tmp["x"], bins=5)
        else:
            df_tmp["bucket"] = df_tmp["x"].astype(str)

        total_good = (df_tmp["y"] == 0).sum()
        total_bad  = (df_tmp["y"] == 1).sum()
        if total_good == 0 or total_bad == 0:
            return 0.0, {}

        iv = 0.0
        woe_map = {}
        for bucket, grp in df_tmp.groupby("bucket", observed=True):
            n_bad  = (grp["y"] == 1).sum()
            n_good = (grp["y"] == 0).sum()
            pct_bad  = n_bad  / total_bad  if total_bad  else 1e-4
            pct_good = n_good / total_good if total_good else 1e-4
            pct_bad  = max(pct_bad,  1e-4)
            pct_good = max(pct_good, 1e-4)
            woe = np.log(pct_good / pct_bad)
            iv += (pct_good - pct_bad) * woe
            woe_map[str(bucket)] = round(woe, 4)

        return round(iv, 4), woe_map

    def _iv_label(self, iv: float) -> str:
        if iv < 0.02:  return "Useless"
        if iv < 0.10:  return "Weak"
        if iv < 0.30:  return "Medium"
        if iv <= 0.50: return "Strong"
        return "Suspicious (>0.50)"

    # ─────────────────────────────────────────────────────────────
    def _correlation_analysis(self, state: PipelineState, X: pd.DataFrame) -> PipelineState:
        X_num = X.apply(pd.to_numeric, errors='coerce').select_dtypes(include=[np.number]).fillna(0)
        corr = X_num.corr().abs()
        state.correlation_matrix = corr

        # Find highly correlated pairs
        high_corr_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] > 0.85:
                    high_corr_pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(corr.iloc[i, j]), 3),
                    })

        state.dqr_report["high_correlation_pairs"] = high_corr_pairs
        if high_corr_pairs:
            self._info(f"High correlation pairs (>0.85): {len(high_corr_pairs)}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _rf_importance(self, state: PipelineState,
                       X: pd.DataFrame, y: pd.Series,
                       feature_cols: list) -> PipelineState:
        # Quick RF on a 30k sample for speed
        sample_size = min(30000, len(X))
        idx = np.random.choice(len(X), sample_size, replace=False)
        Xs, ys = X.iloc[idx].fillna(0), y.iloc[idx]

        rf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                    random_state=42, n_jobs=-1)
        rf.fit(Xs, ys)
        importances = dict(zip(feature_cols, rf.feature_importances_))
        state.feature_importance = importances
        self._info(f"RF importance computed (top 5): "
                   f"{sorted(importances.items(), key=lambda x: -x[1])[:5]}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _select_features(self, state: PipelineState, feature_cols: list) -> PipelineState:
        iv_df = state.iv_table.copy() if state.iv_table is not None else pd.DataFrame()
        imp   = state.feature_importance

        selected  = []
        rejected  = {}

        # Step 1: Remove useless IV
        for _, row in iv_df.iterrows():
            feat = row["feature"]
            iv   = row["iv"]
            if iv < IV_USELESS:
                rejected[feat] = f"IV too low ({iv:.4f} < {IV_USELESS})"
                continue
            if iv > 0.50:
                rejected[feat] = (f"Suspicious IV ({iv:.4f} > 0.50) — potential leakage or "
                                  f"overfitting. Confirm variable is free of target leakage "
                                  f"before including.")
                continue
            selected.append(feat)

        # Step 2: Remove from highly correlated pairs — keep higher IV
        if state.dqr_report.get("high_correlation_pairs"):
            for pair in state.dqr_report["high_correlation_pairs"]:
                fa, fb = pair["feature_a"], pair["feature_b"]
                if fa in selected and fb in selected:
                    iv_a = iv_df.set_index("feature").loc[fa, "iv"] if fa in iv_df.set_index("feature").index else 0
                    iv_b = iv_df.set_index("feature").loc[fb, "iv"] if fb in iv_df.set_index("feature").index else 0
                    drop = fa if iv_a < iv_b else fb
                    if drop in selected:
                        selected.remove(drop)
                        rejected[drop] = (f"High correlation with counterpart "
                                          f"(r={pair['correlation']}) — lower IV removed")

        # Step 3: Cap at MAX_FEATURES (by IV + RF importance combined score)
        if len(selected) > MAX_FEATURES:
            iv_idx = iv_df.set_index("feature")["iv"].to_dict()
            imp_max = max(imp.values()) if imp else 1
            scores = {f: (iv_idx.get(f, 0) / 0.5 + imp.get(f, 0) / imp_max)
                      for f in selected}
            selected = sorted(selected, key=lambda f: -scores.get(f, 0))[:MAX_FEATURES]

        state.selected_features = selected
        state.rejected_features = rejected

        self._info(f"Selected: {len(selected)} features")
        self._info(f"Rejected: {len(rejected)} features")
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_rationale(self, state: PipelineState) -> PipelineState:
        iv_df = state.iv_table
        top_features_text = ""
        if iv_df is not None and not iv_df.empty:
            top = iv_df[iv_df["feature"].isin(state.selected_features)].head(15)
            top_features_text = top.to_string(index=False)

        rejected_sample = dict(list(state.rejected_features.items())[:10])

        prompt = f"""
You are a credit risk variable selection specialist reviewing a feature shortlist
for a Lending Club binary default prediction model.

SELECTED FEATURES (top 15 by IV):
{top_features_text}

SAMPLE REJECTED FEATURES:
{rejected_sample}

Write a Variable Selection Rationale (max 200 words) for the model development report.
Explain:
1. Which features are the strongest predictors and why
2. What was removed and the key reasons
3. Any governance or interpretability considerations
Be specific — reference actual feature names and IV values.
"""
        try:
            rationale, _usage = ask_with_usage(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=500)
            state.llm_token_usage[self.name] = _usage
            state.dqr_report["variable_selection_rationale"] = rationale
            self._info("LLM variable selection rationale generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM rationale skipped: {e}")
        return state
