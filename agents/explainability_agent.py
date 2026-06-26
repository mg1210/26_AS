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
from core.llm import ask, CREDIT_RISK_SYSTEM


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

        self._log("Generating LLM explanation narrative …")
        state = self._llm_narrative(state)

        return state

    # ─────────────────────────────────────────────────────────────
    def _compute_shap(self, state: PipelineState, model, X_test, name: str) -> PipelineState:
        # Sample for speed
        n_rows = int(X_test.shape[0])
        n = min(int(self.shap_sample), n_rows)
        idx = np.random.choice(n_rows, n, replace=False)
        X_sample = X_test.iloc[idx]

        try:
            if name in ("XGBoost", "RandomForest"):
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
            self._info(f"SHAP computed on {n} samples")
            self._info(f"Top 5 SHAP features: "
                       f"{list(state.feature_importance.keys())[:5]}")
        except Exception as e:
            state.log_warning(self.name, f"SHAP failed: {e} — falling back to RF importance")
            # Fall back to model feature importances
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(X_test.columns, model.feature_importances_))
                state.feature_importance = dict(sorted(imp.items(), key=lambda x: -x[1]))

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
            narrative = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=700)
            state.shap_summary = narrative
            self._info("LLM explanation narrative generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM narrative skipped: {e}")

        return state
