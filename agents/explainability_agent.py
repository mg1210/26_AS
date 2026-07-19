"""
agents/explainability_agent.py
───────────────────────────────
Phase 6 — Explainability & Reasoning Agent

Responsibilities:
  • SHAP values for champion model
  • Global feature importance (mean |SHAP|)
  • Portfolio-level score driver summary
  • Individual prediction explanation (adverse action style)
  • LLM narrative — business-friendly explanation of the model
"""

import pandas as pd
import numpy as np
import shap
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, ask_with_usage, CREDIT_RISK_SYSTEM


class ExplainabilityAgent(BaseAgent):

    def __init__(self, shap_sample: int = 2000, verbose: bool = True):
        super().__init__("ExplainabilityAgent", verbose)
        self.shap_sample = shap_sample

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        model  = state.champion_model
        X_test = state.X_test
        name   = state.champion_model_name

        if model is None:
            state.log_error(self.name, "No champion model found — skipping explainability")
            return state

        self._log(f"Computing SHAP values for {name} …")
        state = self._compute_shap(state, model, X_test, name)

        self._log("Building portfolio-level driver summary …")
        state = self._portfolio_drivers(state, X_test)

        self._log("Building adverse-action reason codes …")
        state = self._adverse_action(state, X_test)

        self._log("Running fairness / bias assessment …")
        state = self._fairness_assessment(state)

        self._log("Generating LLM explanation narrative …")
        state = self._llm_narrative(state)

        # Build structured response
        top5 = sorted(state.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_strs = [f"{f} ({v:.4f})" for f, v in top5]
        state.agent_responses[self.name] = self.build_response(
            summary="Explainability analysis complete",
            observations=[
                f"SHAP values computed for champion model: {name}",
                f"Features with SHAP importance: {len(state.feature_importance)}",
                f"Top 5 features by mean |SHAP|: {', '.join(top5_strs) if top5_strs else 'N/A'}",
                f"Adverse action codes generated for {len(state.adverse_action_codes)} sample applicants",
            ],
            reasoning=(
                "SHAP TreeExplainer used for tree-based models (XGBoost, RandomForest, GradientBoosting); "
                "KernelExplainer fallback used for Logistic Regression. "
                "Mean absolute SHAP value used as global feature importance metric. "
                "Adverse action codes derived from top negative SHAP contributors per applicant."
            ),
            recommendations=[
                "Review top SHAP features for regulatory and fairness concerns",
                "Ensure adverse action codes map to standard industry reason codes",
                "Conduct disparate impact analysis on protected attribute proxies",
                "Document SHAP findings in model risk management documentation",
            ],
            artifacts={"top_features": dict(top5)},
        )

        return state

    # ─────────────────────────────────────────────────────────────
    def _compute_shap(self, state: PipelineState, model, X_test, name: str) -> PipelineState:
        # Sample for speed
        print(f"  [SHAP debug] X_test type={type(X_test)}, len={len(X_test)}, shape={X_test.shape}")
        X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)

        try:
            if name in ("XGBoost", "RandomForest", "LightGBM", "GradientBoosting_AutoML"):
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_sample)
                # shap_values may return list [neg, pos] or 3D array (samples, features, classes)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
            else:
                # Logistic Regression — use linear explainer via background
                explainer = shap.LinearExplainer(
                    model.named_steps["lr"],
                    shap.sample(
                        pd.DataFrame(
                            model.named_steps["scaler"].transform(X_sample),
                            columns=X_sample.columns
                        ), 100
                    )
                )
                X_scaled = pd.DataFrame(
                    model.named_steps["scaler"].transform(X_sample),
                    columns=X_sample.columns
                )
                shap_vals = explainer.shap_values(X_scaled)

            state.shap_values = shap_vals
            mean_abs = np.abs(shap_vals).mean(axis=0)
            feat_names = X_sample.columns.tolist()
            importance = dict(zip(feat_names, mean_abs))
            state.feature_importance = dict(
                sorted(importance.items(), key=lambda x: -x[1])
            )
            self._info(f"SHAP computed on {len(X_sample)} samples")
            self._info(f"Top 5 SHAP features: "
                       f"{list(state.feature_importance.keys())[:5]}")

            # Persist a sample of SHAP + feature values so the UI can draw a beeswarm.
            try:
                import json as _json, os as _os
                _os.makedirs('outputs/models', exist_ok=True)
                _n = min(500, len(X_sample))
                shap_payload = {
                    'feature_names':  feat_names,
                    'shap_values':    np.asarray(shap_vals)[:_n].tolist(),
                    'feature_values': X_sample.iloc[:_n].to_numpy().tolist(),
                }
                with open(f'outputs/models/{state.run_id}_shap_sample.json', 'w') as _f:
                    _json.dump(shap_payload, _f)
                self._info(f"SHAP sample saved for UI beeswarm ({_n} rows)")
            except Exception as _e:
                state.log_warning(self.name, f"Could not save SHAP sample: {_e}")
        except Exception as e:
            state.log_warning(self.name, f"SHAP failed: {e} — falling back to RF importance")
            # Fall back to model feature importances
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(X_test.columns, model.feature_importances_))
                state.feature_importance = dict(sorted(imp.items(), key=lambda x: -x[1]))

        return state

    # ─────────────────────────────────────────────────────────────
    def _fairness_assessment(self, state: PipelineState) -> PipelineState:
        """Per-group predicted-risk parity across proxy protected attributes,
        computed on the test set and persisted to state (so it lands in the audit
        trail and the report, not just the UI)."""
        if state.champion_model is None or state.X_test is None:
            return state
        if state.engineered_df is None:
            return state

        proxy_attrs = ['verification_status', 'home_ownership', 'purpose', 'addr_state']
        y_prob = state.champion_model.predict_proba(state.X_test)[:, 1]

        fairness_results = {}
        for attr in proxy_attrs:
            source_col = attr if (state.raw_df is not None and attr in state.raw_df.columns) else None
            if source_col is None:
                continue
            try:
                attr_vals = (state.raw_df.loc[state.X_test.index, source_col]
                             if hasattr(state.X_test, 'index') else None)
                if attr_vals is None:
                    continue

                df_fair = pd.DataFrame({
                    'group':          attr_vals.values,
                    'predicted_prob': y_prob,
                    'actual':         state.y_test.values,
                })
                overall_mean = df_fair['predicted_prob'].mean()
                group_stats = df_fair.groupby('group').agg(
                    count=('predicted_prob', 'size'),
                    mean_predicted=('predicted_prob', 'mean'),
                    actual_rate=('actual', 'mean'),
                ).reset_index()
                group_stats = group_stats[group_stats['count'] >= 30]
                group_stats['diff_from_avg'] = group_stats['mean_predicted'] - overall_mean
                group_stats['concern_level'] = group_stats['diff_from_avg'].abs().apply(
                    lambda d: 'High' if d > 0.15 else 'Medium' if d > 0.10 else 'Low')

                fairness_results[attr] = {
                    str(row['group']): {
                        'count':          int(row['count']),
                        'mean_predicted': round(float(row['mean_predicted']), 4),
                        'actual_rate':    round(float(row['actual_rate']), 4),
                        'diff_from_avg':  round(float(row['diff_from_avg']), 4),
                        'concern_level':  row['concern_level'],
                    }
                    for _, row in group_stats.iterrows()
                }
                flagged = [k for k, v in fairness_results[attr].items()
                           if v['concern_level'] in ('Medium', 'High')]
                self._info(f"Fairness check — {attr}: {len(group_stats)} groups analysed, "
                           f"{len(flagged)} flagged for review")
            except Exception as e:
                state.log_warning(self.name, f"Fairness check failed for {attr}: {e}")
                continue

        state.fairness_results = fairness_results
        total_flagged = sum(1 for attr_data in fairness_results.values()
                            for v in attr_data.values() if v['concern_level'] in ('Medium', 'High'))
        self._info(f"Fairness assessment complete: {len(fairness_results)} attributes checked, "
                   f"{total_flagged} groups flagged")
        return state

    # ─────────────────────────────────────────────────────────────
    def _portfolio_drivers(self, state: PipelineState, X_test: pd.DataFrame) -> PipelineState:
        if state.shap_values is None or state.feature_importance is None:
            return state

        top5 = list(state.feature_importance.keys())[:5]
        shap_df = pd.DataFrame(
            state.shap_values[:, :len(X_test.columns)],
            columns=X_test.columns
        ) if state.shap_values is not None and hasattr(state.shap_values, '__len__') else pd.DataFrame()

        drivers = {}
        for feat in top5:
            if feat in shap_df.columns:
                pos = int((shap_df[feat] > 0).sum())
                neg = int((shap_df[feat] < 0).sum())
                drivers[feat] = {
                    "mean_abs_shap": round(float(state.feature_importance[feat]), 5),
                    "pct_increasing_risk": round(pos / len(shap_df) * 100, 1) if len(shap_df) else 0,
                    "pct_decreasing_risk": round(neg / len(shap_df) * 100, 1) if len(shap_df) else 0,
                }

        state.dqr_report["portfolio_drivers"] = drivers
        return state

    # ─────────────────────────────────────────────────────────────
    def _adverse_action(self, state: PipelineState, X_test: pd.DataFrame) -> PipelineState:
        """
        For a few high-risk predictions, generate top 3 reason codes
        (adverse action style) using SHAP values.
        """
        if state.shap_values is None:
            return state

        model = state.champion_model
        probs = model.predict_proba(X_test)[:, 1]
        # Take 3 highest-risk predictions
        top_idx = np.argsort(probs)[-3:]

        codes = {}
        shap_arr = state.shap_values
        feat_names = X_test.columns.tolist()

        for i in top_idx:
            if i >= len(shap_arr):
                continue
            row_shap = shap_arr[i]
            # Top 3 features driving risk UP (positive SHAP)
            pos_idx = np.argsort(row_shap)[::-1][:3]
            reasons = []
            for j in pos_idx:
                if j < len(feat_names) and row_shap[j] > 0:
                    reasons.append({
                        "feature"   : feat_names[j],
                        "value"     : round(float(X_test.iloc[i, j]), 3),
                        "shap"      : round(float(row_shap[j]), 4),
                        "reason_code": f"High {feat_names[j]} increases default risk"
                    })
            codes[f"sample_{i}"] = {
                "predicted_prob": round(float(probs[i]), 4),
                "top_reasons"   : reasons,
            }

        state.adverse_action_codes = codes
        self._info(f"Adverse action codes generated for {len(codes)} high-risk samples")
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_narrative(self, state: PipelineState) -> PipelineState:
        top_features = list(state.feature_importance.items())[:10]
        feat_text    = "\n".join(
            f"  {i+1}. {f} — mean |SHAP| = {v:.4f}"
            for i, (f, v) in enumerate(top_features)
        )
        model_name   = state.champion_model_name
        auc          = state.model_metrics.get(model_name, {}).get("auc_test", "N/A")
        ks           = state.model_metrics.get(model_name, {}).get("ks", "N/A")
        gini         = state.model_metrics.get(model_name, {}).get("gini", "N/A")

        prompt = f"""
You are a credit risk explainability specialist writing a section of the model card
for a Lending Club binary default prediction model.

MODEL: {model_name}
Performance: AUC={auc}, KS={ks}, Gini={gini}

TOP 10 MODEL DRIVERS (by mean absolute SHAP value):
{feat_text}

Write a Model Explanation Narrative (max 300 words) for the model development report.
Cover:
1. What the model is predicting and how it makes decisions
2. The top 3-5 key risk drivers and their business interpretation
3. What a high-risk vs low-risk borrower looks like according to the model
4. Any limitations or caveats in interpreting these drivers
Write in plain English suitable for a credit committee or model governance audience.
"""
        try:
            narrative, _usage = ask_with_usage(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=700)
            state.llm_token_usage[self.name] = _usage
            state.shap_summary = narrative
            self._info("LLM explanation narrative generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM narrative skipped: {e}")

        return state
