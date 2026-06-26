"""
agents/validation_agent.py
────────────────────────────
Phase 7 — Model Validation & Documentation Agent

Responsibilities:
  • KS, AUC, Gini on test set
  • PSI (Population Stability Index) across time splits
  • Score distribution & calibration check
  • Challenger model comparison
  • Generate model development report (text)
  • Compile full audit trail
  • LLM-generated validation summary and recommendations
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM
import json
import os
from datetime import datetime


class ValidationAgent(BaseAgent):

    def __init__(self, output_dir: str = "outputs", verbose: bool = True):
        super().__init__("ValidationAgent", verbose)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:

        self._log("Running discrimination metrics …")
        state = self._discrimination(state)

        self._log("Running PSI analysis …")
        state = self._psi_analysis(state)

        self._log("Running calibration check …")
        state = self._calibration(state)

        self._log("Comparing champion vs challengers …")
        state = self._challenger_comparison(state)

        self._log("Generating validation summary with LLM …")
        state = self._llm_validation_summary(state)

        self._log("Generating model development report …")
        state = self._generate_report(state)

        self._log("Saving audit trail …")
        state = self._save_audit(state)

        return state

    # ─────────────────────────────────────────────────────────────
    def _discrimination(self, state: PipelineState) -> PipelineState:
        model  = state.champion_model
        if model is None:
            return state
        X_te   = state.X_test
        y_te   = state.y_test

        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
        gini   = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        ks     = float(np.max(tpr - fpr))

        # Score decile analysis
        df_tmp = pd.DataFrame({"prob": y_prob, "target": y_te.values})
        df_tmp["decile"] = pd.qcut(df_tmp["prob"], q=10, labels=False, duplicates="drop") + 1
        decile_tbl = (df_tmp.groupby("decile", observed=True)
                      .agg(n=("target","count"),
                           n_bad=("target","sum"),
                           avg_prob=("prob","mean"))
                      .assign(bad_rate=lambda d: d["n_bad"]/d["n"])
                      .reset_index()
                      .to_dict(orient="records"))

        state.validation_metrics = {
            "auc"          : round(auc, 4),
            "gini"         : round(gini, 4),
            "ks"           : round(ks, 4),
            "decile_table" : decile_tbl,
        }
        self._info(f"AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")

        # Pass/fail thresholds (industry minimums)
        state.validation_passed = (auc >= 0.65 and ks >= 0.20)
        if not state.validation_passed:
            state.log_warning(self.name,
                "Model below minimum thresholds (AUC<0.65 or KS<0.20) — review before deployment")
        return state

    # ─────────────────────────────────────────────────────────────
    def _psi_analysis(self, state: PipelineState) -> PipelineState:
        """
        Approximate PSI by splitting test set chronologically if vintage
        info is available, otherwise use random halves.
        """
        df_eng = state.engineered_df
        model  = state.champion_model
        feats  = state.selected_features

        available_feats = [f for f in feats if f in df_eng.columns]
        X_all  = df_eng[available_feats].fillna(0)
        probs  = model.predict_proba(X_all)[:, 1]

        # Split by issue_year if available, else random 50/50
        if "issue_year" in df_eng.columns:
            years  = df_eng["issue_year"].sort_values().unique()
            mid    = years[len(years) // 2]
            mask   = df_eng["issue_year"] <= mid
            label  = f"pre-{mid} vs post-{mid}"
        else:
            mask  = np.random.rand(len(probs)) > 0.5
            label = "random split A vs B"

        psi = self._compute_psi(probs[mask], probs[~mask])
        self._info(f"PSI ({label}) = {psi:.4f}")

        state.psi_results = {
            "psi_score"   : round(psi, 4),
            "split_label" : label,
            "assessment"  : ("Stable" if psi < 0.10 else
                             "Moderate shift" if psi < 0.25 else "Significant shift"),
        }
        return state

    def _compute_psi(self, ref: np.ndarray, sample: np.ndarray, bins: int = 10) -> float:
        if len(ref) == 0 or len(sample) == 0:
            return 0.0
        breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf
        ref_pct    = np.histogram(ref,    bins=breakpoints)[0] / len(ref)
        sample_pct = np.histogram(sample, bins=breakpoints)[0] / len(sample)
        ref_pct    = np.where(ref_pct    == 0, 1e-4, ref_pct)
        sample_pct = np.where(sample_pct == 0, 1e-4, sample_pct)
        return float(np.sum((sample_pct - ref_pct) * np.log(sample_pct / ref_pct)))

    # ─────────────────────────────────────────────────────────────
    def _calibration(self, state: PipelineState) -> PipelineState:
        model  = state.champion_model
        X_te   = state.X_test
        y_te   = state.y_test

        y_prob = model.predict_proba(X_te)[:, 1]
        brier  = brier_score_loss(y_te, y_prob)
        try:
            prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=10)
            cal_error = float(np.mean(np.abs(prob_true - prob_pred)))
        except Exception:
            cal_error = None

        state.validation_metrics["brier_score"] = round(brier, 4)
        state.validation_metrics["calibration_error"] = round(cal_error, 4) if cal_error else None
        self._info(f"Brier score={brier:.4f}  Calibration error={cal_error:.4f}" if cal_error else
                   f"Brier score={brier:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _challenger_comparison(self, state: PipelineState) -> PipelineState:
        rows = []
        for name, m in state.model_metrics.items():
            rows.append({
                "model"     : name,
                "auc_test"  : m.get("auc_test"),
                "ks"        : m.get("ks"),
                "gini"      : m.get("gini"),
                "overfit"   : m.get("overfit"),
                "champion"  : "✓" if name == state.champion_model_name else "",
            })
        state.validation_metrics["challenger_table"] = rows
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_validation_summary(self, state: PipelineState) -> PipelineState:
        vm = state.validation_metrics
        psi = state.psi_results
        prompt = f"""
You are an independent model validator reviewing a credit risk model.

VALIDATION RESULTS:
- Champion model: {state.champion_model_name}
- AUC: {vm.get('auc')}
- Gini: {vm.get('gini')}
- KS: {vm.get('ks')}
- Brier Score: {vm.get('brier_score')}
- Calibration Error: {vm.get('calibration_error')}
- PSI: {psi.get('psi_score')} ({psi.get('assessment')}) [{psi.get('split_label')}]
- Validation passed minimum thresholds: {state.validation_passed}

CHALLENGER COMPARISON:
{state.validation_metrics.get('challenger_table', [])}

Write a Model Validation Summary (max 300 words) for the governance report.
Cover:
1. Overall validation outcome (pass/conditional pass/fail)
2. Discriminatory power assessment
3. Stability assessment
4. Calibration assessment
5. Conditions or recommendations before deployment
Be direct and specific — this is for model governance sign-off.
"""
        try:
            summary = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=700)
            state.validation_summary = summary
            self._info("LLM validation summary generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM summary skipped: {e}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _generate_report(self, state: PipelineState) -> PipelineState:
        run_id  = state.run_id
        report  = []
        sep     = "=" * 70

        def h1(t): report.append(f"\n{sep}\n{t}\n{sep}")
        def h2(t): report.append(f"\n{'─'*50}\n{t}\n{'─'*50}")
        def p(t):  report.append(str(t))

        h1(f"CREDIT RISK MODEL DEVELOPMENT REPORT")
        p(f"Run ID       : {run_id}")
        p(f"Dataset      : {state.dataset_name}")
        p(f"Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        h2("1. EXECUTIVE SUMMARY")
        p(state.dqr_report.get("llm_narrative", "N/A"))

        h2("2. TARGET VARIABLE DEFINITION")
        p(state.target_definition)

        h2("3. DATA QUALITY REVIEW")
        miss = state.missing_summary
        top_miss = sorted(miss.items(), key=lambda x: -x[1]["pct_missing"])[:10]
        p("Top 10 columns by missing rate:")
        for col, v in top_miss:
            p(f"  {col:<40} {v['pct_missing']:.1%}")
        p("\nDQR Flags:")
        for flag in state.dqr_flags[:10]:
            p(f"  {flag}")

        h2("4. FEATURE ENGINEERING SUMMARY")
        p(state.dqr_report.get("feature_engineering_summary", "N/A"))
        p(f"\nEngineered features ({len(state.feature_log)}):")
        for f in state.feature_log[:15]:
            p(f"  • {f['feature']}: {f['rationale']}")

        h2("5. VARIABLE SELECTION")
        p(state.dqr_report.get("variable_selection_rationale", "N/A"))
        p(f"\nSelected features ({len(state.selected_features)}):")
        if state.iv_table is not None:
            sel_iv = state.iv_table[
                state.iv_table["feature"].isin(state.selected_features)
            ].to_string(index=False)
            p(sel_iv)

        h2("6. MODEL DEVELOPMENT")
        p(f"Champion model: {state.champion_model_name}")
        p("\nModel comparison:")
        for name, m in state.model_metrics.items():
            champ = " ← CHAMPION" if name == state.champion_model_name else ""
            p(f"  {name:<25} AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  "
              f"Gini={m['gini']:.4f}  Overfit={m['overfit']:.4f}{champ}")
        p(f"\nModel Selection Rationale:\n{state.model_selection_rationale}")

        h2("7. MODEL EXPLAINABILITY")
        p(state.shap_summary or "N/A")
        p("\nTop feature importances (mean |SHAP|):")
        for feat, val in list(state.feature_importance.items())[:10]:
            p(f"  {feat:<40} {val:.5f}")

        h2("8. MODEL VALIDATION")
        vm = state.validation_metrics
        p(f"AUC     : {vm.get('auc')}")
        p(f"Gini    : {vm.get('gini')}")
        p(f"KS      : {vm.get('ks')}")
        p(f"Brier   : {vm.get('brier_score')}")
        psi = state.psi_results
        p(f"PSI     : {psi.get('psi_score')} — {psi.get('assessment')}")
        p(f"\nValidation Outcome: {'PASS ✓' if state.validation_passed else 'CONDITIONAL ⚠'}")
        p(f"\n{state.validation_summary}")

        h2("9. ASSUMPTIONS, LIMITATIONS & RISKS")
        p("Assumptions:")
        p("  • Binary default definition: Charged Off = 1, Fully Paid = 0")
        p("  • Ambiguous statuses (Current, In Grace Period) excluded from training")
        p("  • Median imputation applied to missing numeric features")
        p("  • WOE encoding computed on training data only (no look-ahead)")
        p("\nLimitations:")
        p("  • Model trained on Lending Club platform data — may not generalise to other portfolios")
        p("  • High missing rates in several bureau features limit completeness")
        p("  • Post-origination fields (payments, recoveries) excluded to prevent leakage")
        p("\nRisks:")
        p("  • Behavioural data (DTI, revolving utilisation) is self-reported — subject to misrepresentation")
        p("  • Model performance should be re-evaluated quarterly using PSI monitoring")
        p("  • Adverse action reason codes must be reviewed before customer-facing deployment")

        h2("10. AUDIT TRAIL")
        for entry in state.audit_log:
            p(f"  {entry['timestamp']}  [{entry['agent']}]  {entry['action']}  {entry.get('detail','')}")

        report_text = "\n".join(report)
        path = os.path.join(self.output_dir, f"{run_id}_model_report.txt")
        with open(path, "w") as f:
            f.write(report_text)

        state.model_report_path = path
        self._info(f"Model report saved → {path}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _save_audit(self, state: PipelineState) -> PipelineState:
        path = os.path.join(self.output_dir, f"{state.run_id}_audit_trail.json")
        summary = state.to_summary_dict()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._info(f"Audit trail saved → {path}")
        return state
