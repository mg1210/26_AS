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
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
try:
    # sklearn ≥ 1.6 — the modern replacement for the removed cv='prefit'.
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover — older sklearn
    FrozenEstimator = None
from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM
from core.recommendation import Recommendation
import json
import os
from datetime import datetime


# ── KPI Methodology framework — RAG thresholds for every validation metric ──
KPI_THRESHOLDS = {
    'gini': {
        'strong_green': (0.45, 0.70), 'acceptable_green': (0.35, 0.45),
        'amber': (0.30, 0.35), 'red': (0, 0.30),
        'formula': 'Gini = 2 × AUROC − 1',
        'basis': 'Dev / OOT'
    },
    'auc': {
        'strong_green': (0.72, 1.0), 'acceptable_green': (0.68, 0.72),
        'amber': (0.65, 0.68), 'red': (0, 0.65),
        'formula': 'AUROC = (Gini + 1) ÷ 2',
        'basis': 'Dev / OOT'
    },
    'ks': {
        'strong_green': (0.35, 0.65), 'acceptable_green': (0.25, 0.35),
        'amber': (0.20, 0.25), 'red': (0, 0.20),
        'formula': 'KS = max|Cum%Bad − Cum%Good| across deciles',
        'basis': 'Dev / OOT'
    },
    'psi': {
        'strong_green': (0, 0.10), 'acceptable_green': (0.10, 0.20),
        'amber': (0.20, 0.30), 'red': (0.30, 999),
        'formula': 'PSI = Σ (%Dev − %OOT) × ln(%Dev ÷ %OOT)',
        'basis': 'Dev vs OOT'
    },
    'csi': {
        'strong_green': (0, 0.10), 'acceptable_green': (0.10, 0.20),
        'amber': (0.20, 0.30), 'red': (0.30, 999),
        'formula': 'CSI = Σ per bin (%Dev − %OOT) × ln(%Dev ÷ %OOT)',
        'basis': 'Per variable'
    },
    'iv': {
        'strong_green': (0.30, 0.50), 'acceptable_green': (0.10, 0.30),
        'amber': (0.02, 0.10), 'red_low': (0, 0.02), 'red_high': (0.50, 999),
        'formula': 'IV = Σ (%Good − %Bad) × WOE per bin',
        'basis': 'Per variable'
    },
    'gini_gap': {
        'strong_green': (0, 0.05), 'acceptable_green': (0.05, 0.10),
        'amber': (0.10, 0.15), 'red': (0.15, 999),
        'formula': '(Gini_Dev − Gini_OOT) ÷ Gini_Dev',
        'basis': 'Dev vs OOT'
    },
    'ks_gap': {
        'strong_green': (0, 0.05), 'acceptable_green': (0.05, 0.10),
        'amber': (0.10, 0.15), 'red': (0.15, 999),
        'formula': '(KS_Dev − KS_OOT) ÷ KS_Dev',
        'basis': 'Dev vs OOT'
    },
}


def get_rag(metric, value):
    """Return (RAG, label) for a metric value against KPI_THRESHOLDS."""
    t = KPI_THRESHOLDS.get(metric, {})
    if metric == 'iv':
        if value < 0.02: return 'RED', 'Useless (<0.02)'
        if value > 0.50: return 'RED', 'Suspicious (>0.50) — potential leakage'
        if value >= 0.30: return 'GREEN', 'Strong'
        if value >= 0.10: return 'GREEN', 'Medium'
        return 'AMBER', 'Weak'
    if metric in ('psi', 'csi', 'gini_gap', 'ks_gap'):
        if value <= t.get('strong_green', (0, 0))[1]: return 'GREEN', 'Strong'
        if value <= t.get('acceptable_green', (0, 0))[1]: return 'GREEN', 'Acceptable'
        if value <= t.get('amber', (0, 0))[1]: return 'AMBER', 'Warning'
        return 'RED', 'Action Required'
    else:
        if t.get('strong_green', (0, 0))[0] <= value <= t.get('strong_green', (0, 999))[1]: return 'GREEN', 'Strong'
        if t.get('acceptable_green', (0, 0))[0] <= value: return 'GREEN', 'Acceptable'
        if t.get('amber', (0, 0))[0] <= value: return 'AMBER', 'Warning'
        return 'RED', 'Action Required'


class ValidationAgent(BaseAgent):

    def __init__(self, output_dir: str = "outputs", verbose: bool = True):
        super().__init__("ValidationAgent", verbose)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:

        self._log("Running discrimination metrics …")
        state = self._discrimination(state)

        self._log("Running calibration analysis...")
        state = self._calibration_analysis(state)

        self._log("Running confusion matrix analysis …")
        state = self._confusion_matrix_analysis(state)

        self._log("Running PSI analysis …")
        state = self._psi_analysis(state)

        self._log("Generating findings register …")
        state = self._generate_findings_register(state)

        self._log("Running calibration check …")
        state = self._calibration(state)

        self._log("Comparing champion vs challengers …")
        state = self._challenger_comparison(state)

        self._log("Generating validation summary with LLM …")
        state = self._llm_validation_summary(state)

        # Build structured response
        vm   = state.validation_metrics
        psi  = state.psi_results
        state.agent_responses[self.name] = self.build_response(
            summary="Model validation complete",
            observations=[
                f"AUC (test): {vm.get('auc', 'N/A')}",
                f"KS: {vm.get('ks', 'N/A')}",
                f"Gini: {vm.get('gini', 'N/A')}",
                f"Brier score: {vm.get('brier_score', 'N/A')}",
                f"PSI score: {psi.get('psi_score', 'N/A')} — {psi.get('assessment', 'N/A')}",
                f"Validation outcome: {'PASS' if state.validation_passed else 'CONDITIONAL'}",
            ],
            reasoning=(
                "Industry-standard thresholds applied: AUC ≥ 0.65, KS ≥ 0.20, Gini ≥ 0.30. "
                "PSI < 0.10 indicates population stability; 0.10–0.25 moderate shift; >0.25 unstable. "
                "Brier score measures calibration quality — lower is better. "
                "Challenger comparison run across all trained candidate models."
            ),
            recommendations=[
                "Schedule quarterly PSI monitoring to detect population drift",
                "Re-calibrate model if Brier score degrades by >0.02 from baseline",
                "Conduct annual model validation with independent out-of-time data",
                "Obtain sign-off from model risk governance before production deployment",
                "Document all validation findings in the Model Risk Management (MRM) report",
            ],
            overall_status="pass" if state.validation_passed else "conditional",
        )

        auc_val = float(vm.get("auc") or 0.0)
        state.recommendations.append(Recommendation(
            title="Deployment Readiness",
            recommendation="PASS" if state.validation_passed else "CONDITIONAL",
            rationale=state.validation_summary or (
                f"AUC={auc_val:.4f}. "
                "Industry thresholds: AUC≥0.65, KS≥0.20, Gini≥0.30. "
                f"Validation outcome: {'PASS' if state.validation_passed else 'CONDITIONAL'}."
            ),
            confidence=auc_val,
            risk="low" if state.validation_passed else "high",
            requires_human_approval=True,
        ))

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

        # ── STEP 3: Rank-order break detection ───────────────────────────────
        # Walk deciles from highest risk to lowest; a break is any decile whose
        # bad rate RISES relative to the adjacent higher-risk decile (bad rate
        # should decline monotonically as risk falls). sorted_deciles holds the
        # same dict objects as decile_tbl, so tagging them tags the table too.
        sorted_deciles = sorted(decile_tbl, key=lambda d: d["decile"], reverse=True)
        prev_bad_rate = None
        for decile in sorted_deciles:
            ro_break = False
            if prev_bad_rate is not None and decile["bad_rate"] > prev_bad_rate:
                ro_break = True
            decile["ro_break"] = ro_break
            prev_bad_rate = decile["bad_rate"]
        n_breaks = sum(1 for d in decile_tbl if d.get("ro_break"))

        state.validation_metrics = {
            "auc"          : round(auc, 4),
            "gini"         : round(gini, 4),
            "ks"           : round(ks, 4),
            "decile_table" : decile_tbl,
        }
        self._info(f"AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}")
        self._info(f"Rank-order breaks: {n_breaks}")

        # ── OOT evaluation (out-of-time holdout, if a time-based split exists) ─
        if state.X_oot is not None and state.y_oot is not None:
            y_prob_oot = model.predict_proba(state.X_oot)[:, 1]
            auc_oot  = roc_auc_score(state.y_oot, y_prob_oot)
            gini_oot = 2 * auc_oot - 1
            fpr_oot, tpr_oot, _ = roc_curve(state.y_oot, y_prob_oot)
            ks_oot   = float(np.max(tpr_oot - fpr_oot))
            # Dataset 1 OOT metrics under dedicated *_d1 keys (never collide with
            # New Data scoring, which uses *_new_data keys in evaluate_oot.py).
            state.validation_metrics['auc_oot_d1']  = round(auc_oot, 4)
            state.validation_metrics['gini_oot_d1'] = round(gini_oot, 4)
            state.validation_metrics['ks_oot_d1']   = round(ks_oot, 4)
            # Keep the legacy keys for backward compatibility.
            state.validation_metrics['auc_oot']  = round(auc_oot, 4)
            state.validation_metrics['gini_oot'] = round(gini_oot, 4)
            state.validation_metrics['ks_oot']   = round(ks_oot, 4)
            self._info(f"OOT performance: AUC={auc_oot:.4f} Gini={gini_oot:.4f} KS={ks_oot:.4f}")

        # ── STEP 2: KPI scoreboard — headline RAG on OOT metrics where available ─
        sb_auc  = state.validation_metrics.get('auc_oot',  auc)
        sb_gini = state.validation_metrics.get('gini_oot', gini)
        sb_ks   = state.validation_metrics.get('ks_oot',   ks)
        auc_rag,  auc_label  = get_rag('auc',  sb_auc)
        ks_rag,   ks_label   = get_rag('ks',   sb_ks)
        gini_rag, gini_label = get_rag('gini', sb_gini)
        state.validation_metrics['kpi_scoreboard'] = [
            {'KPI': 'Gini', 'Formula': 'Gini = 2×AUROC−1', 'Value': round(sb_gini, 4), 'RAG': gini_rag, 'Strength': gini_label, 'Threshold (GREEN)': '≥0.35', 'Threshold (AMBER)': '0.30–0.35', 'Threshold (RED)': '<0.30'},
            {'KPI': 'AUC / AUROC', 'Formula': 'Area under ROC curve', 'Value': round(sb_auc, 4), 'RAG': auc_rag, 'Strength': auc_label, 'Threshold (GREEN)': '>0.68', 'Threshold (AMBER)': '0.65–0.68', 'Threshold (RED)': '<0.65'},
            {'KPI': 'KS Statistic', 'Formula': 'max|Cum%Bad−Cum%Good|', 'Value': round(sb_ks, 4), 'RAG': ks_rag, 'Strength': ks_label, 'Threshold (GREEN)': '0.25–0.65', 'Threshold (AMBER)': '0.20–0.25', 'Threshold (RED)': '<0.20'},
        ]

        # ── STEP 3: Rank-order breaks scoreboard row ─────────────────────────
        ro_rag, ro_label = (('GREEN', 'No breaks') if n_breaks == 0 else
                            ('AMBER', f'{n_breaks} break(s)') if n_breaks <= 2 else
                            ('RED', f'{n_breaks} breaks — action required'))
        state.validation_metrics['kpi_scoreboard'].append({
            'KPI': 'Rank Order Breaks', 'Formula': 'Count deciles where bad rate rises vs adjacent higher-risk decile',
            'Value': n_breaks, 'RAG': ro_rag, 'Strength': ro_label,
            'Threshold (GREEN)': 'No breaks / 1 minor', 'Threshold (AMBER)': '2 below cutoff', 'Threshold (RED)': 'At cutoff or >2'
        })

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

        # Split 70/30: chronological if issue_year available, otherwise positional
        split = int(len(probs) * 0.7)
        if "issue_year" in df_eng.columns:
            order = df_eng["issue_year"].argsort().values
            ref_idx    = order[:split]
            sample_idx = order[split:]
            years      = df_eng["issue_year"].iloc[order]
            label      = f"earliest 70% vs latest 30% by issue_year"
            ref_probs    = probs[ref_idx]
            sample_probs = probs[sample_idx]
        else:
            ref_probs    = probs[:split]
            sample_probs = probs[split:]
            label        = "first 70% rows vs last 30% rows"

        psi = self._compute_psi(ref_probs, sample_probs)
        self._info(f"PSI ({label}) = {psi:.4f}")

        state.psi_results = {
            "psi_score"   : round(psi, 4),
            "split_label" : label,
            "assessment"  : ("Stable" if psi < 0.10 else
                             "Moderate shift" if psi < 0.25 else "Significant shift"),
        }

        # ── CSI: top 5 features by importance ────────────────────────
        csi_results = {}
        imp   = state.feature_importance
        top5  = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5] if imp else []
        for feat, _ in top5:
            if feat not in df_eng.columns:
                continue
            feat_vals = df_eng[feat].fillna(0).values
            if "issue_year" in df_eng.columns:
                ref_feat    = feat_vals[ref_idx]
                sample_feat = feat_vals[sample_idx]
            else:
                ref_feat    = feat_vals[:split]
                sample_feat = feat_vals[split:]
            csi_val = self._compute_psi(ref_feat, sample_feat)
            csi_results[feat] = {
                "csi":        round(csi_val, 4),
                "assessment": ("Stable" if csi_val < 0.10 else
                               "Moderate" if csi_val < 0.25 else "Unstable"),
            }
            self._info(f"CSI({feat}) = {csi_val:.4f}")

        state.validation_metrics["csi_results"] = csi_results

        # ── STEP 4: Gini gap (Dev→OOT) + PSI scoreboard rows ─────────────────
        psi_score = psi
        auc_dev  = state.model_metrics.get(state.champion_model_name, {}).get('auc_train', None)
        gini_dev = (2 * auc_dev - 1) if auc_dev else None
        # Prefer the true out-of-time gini/ks when a time-based split produced one,
        # else fall back to the in-sample test metrics.
        auc_oot  = state.validation_metrics.get('auc_oot')  or state.validation_metrics.get('auc')
        gini_oot = state.validation_metrics.get('gini_oot') or state.validation_metrics.get('gini')
        ks_dev   = state.model_metrics.get(state.champion_model_name, {}).get('ks_train', None)
        ks_oot   = state.validation_metrics.get('ks_oot')   or state.validation_metrics.get('ks')

        # Gini gap (Dev→OOT) is stored for the Performance-tab comparison table;
        # it is intentionally NOT added to the KPI scoreboard to avoid duplication.
        if gini_dev and gini_oot:
            gini_gap = abs(gini_dev - gini_oot) / gini_dev if gini_dev > 0 else 0
            gini_gap_rag, _ = get_rag('gini_gap', gini_gap)
            state.validation_metrics['gini_gap'] = round(gini_gap, 4)
            state.validation_metrics['gini_gap_rag'] = gini_gap_rag

        psi_rag, psi_label = get_rag('psi', psi_score)
        state.validation_metrics.setdefault('kpi_scoreboard', []).append({
            'KPI': 'PSI (Population Stability)', 'Formula': 'Σ (%Dev−%OOT)×ln(%Dev÷%OOT)',
            'Value': round(psi_score, 4), 'RAG': psi_rag, 'Strength': psi_label,
            'Threshold (GREEN)': '<0.10', 'Threshold (AMBER)': '0.10–0.25', 'Threshold (RED)': '>0.25'
        })

        # ── STEP 6: Feature-level KPI consolidated table (IV + CSI + RAG) ─────
        feature_kpi_table = []
        for feat in state.selected_features:
            iv_val = None
            if state.iv_table is not None:
                iv_row = state.iv_table[state.iv_table['feature'] == feat]
                if not iv_row.empty:
                    iv_val = float(iv_row['iv'].iloc[0])
            # CSI is stored per-feature as {'csi': value, 'assessment': ...}
            csi_entry = csi_results.get(feat)
            csi_val = csi_entry.get('csi') if isinstance(csi_entry, dict) else csi_entry
            iv_rag  = get_rag('iv', iv_val)[0] if iv_val is not None else 'N/A'
            csi_rag = get_rag('csi', csi_val)[0] if csi_val is not None else 'N/A'
            feature_kpi_table.append({
                'Feature': feat,
                'IV': round(iv_val, 4) if iv_val else 'N/A',
                'IV RAG': iv_rag,
                'CSI': round(csi_val, 4) if csi_val else 'N/A',
                'CSI RAG': csi_rag,
                'VDI': 'N/A',
                'Notes': 'IV>0.50 — confirm not leakage' if iv_val and iv_val > 0.50 else
                         'IV<0.10 — weak, monitor' if iv_val and iv_val < 0.10 else ''
            })
        state.validation_metrics['feature_kpi_table'] = feature_kpi_table
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
    def _generate_findings_register(self, state: PipelineState) -> PipelineState:
        """Auto-generate a governance Findings Register from KPI RAG ratings,
        IV quality, CSI stability, and any retained behavioral-leakage signals."""
        remediation_map = {
            'Gini': 'Review model on larger/more recent sample; consider recalibration',
            'AUC / AUROC': 'Review feature engineering and model complexity',
            'KS Statistic': 'Review score distribution and feature predictive power',
            'Rank Order Breaks': 'Review score banding; check for monotonicity in WOE bins',
            'Gini Gap (Dev→OOT)': 'Investigate population drift; consider retraining on more recent data',
            'PSI (Population Stability)': 'Investigate population shift; review input feature distributions',
        }
        findings = []
        finding_id = 1

        def _add(category, finding, severity, remediation):
            nonlocal finding_id
            findings.append({
                'Ref': f'F{finding_id:02d}',
                'Category': category,
                'Finding': finding,
                'Severity': severity,
                'Recommended Remediation': remediation,
                'Status': 'Open',
            })
            finding_id += 1

        # 1. KPI scoreboard — one finding per AMBER / RED metric
        for kpi in state.validation_metrics.get('kpi_scoreboard', []):
            if kpi.get('RAG') in ('AMBER', 'RED'):
                category = ('Predictive Power' if kpi['KPI'] in ('Gini', 'AUC / AUROC', 'KS Statistic')
                            else 'Stability' if ('PSI' in kpi['KPI'] or 'Gap' in kpi['KPI'])
                            else 'Rank Order')
                _add(category,
                     f"{kpi['KPI']} = {kpi['Value']} ({kpi['Strength']})",
                     'High' if kpi['RAG'] == 'RED' else 'Medium',
                     remediation_map.get(kpi['KPI'], 'Review and monitor'))

        # 2. IV table — suspicious (>0.50, any candidate = leakage risk) and
        #    weak (<0.10) variables that were nonetheless kept in the model.
        iv_tbl = state.iv_table
        if iv_tbl is not None:
            for _, row in iv_tbl.iterrows():
                feat = row.get('feature')
                iv   = float(row.get('iv') or 0.0)
                if iv > 0.50:
                    _add('Variable Quality',
                         f"{feat}: IV={iv:.4f} suspicious (>0.50) — potential target leakage",
                         'High',
                         'Confirm variable is free of target leakage; exclude if post-outcome')
            for feat in state.selected_features:
                r = iv_tbl[iv_tbl['feature'] == feat]
                if not r.empty:
                    iv = float(r['iv'].iloc[0])
                    if iv < 0.10:
                        _add('Variable Quality',
                             f"{feat}: IV={iv:.4f} weak (<0.10) — limited standalone predictive power",
                             'Low',
                             'Monitor; consider dropping or combining with stronger features')

        # 3. CSI — any variable with characteristic instability (> 0.10)
        for feat, info in state.validation_metrics.get('csi_results', {}).items():
            csi = info.get('csi') if isinstance(info, dict) else info
            if csi is not None and csi > 0.10:
                _add('Stability',
                     f"{feat}: CSI={csi:.4f} (>0.10) — characteristic instability Dev→OOT",
                     'High' if csi > 0.25 else 'Medium',
                     'Investigate feature distribution shift; review data source stability')

        # 4. Behavioral leakage retained in the model (univariate AUC > 0.80)
        for col, auc_v in (state.behavioral_leakage_flags or {}).items():
            try:
                auc_v = float(auc_v)
            except (TypeError, ValueError):
                continue
            if auc_v > 0.80 and any(col == f or col in f for f in state.selected_features):
                _add('Leakage',
                     f"{col}: univariate AUC={auc_v:.3f} (>0.80) retained in model — likely target leakage",
                     'High',
                     'Remove or justify variable; confirm it is not post-outcome information')

        state.validation_metrics['findings_register'] = findings
        n_high = sum(1 for f in findings if f['Severity'] == 'High')
        self._info(f"Findings register: {len(findings)} finding(s), {n_high} High")
        return state

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
    def _calibration_analysis(self, state: PipelineState) -> PipelineState:
        """Full calibration study: uncalibrated vs Platt (sigmoid) vs isotonic,
        with Brier + ECE and reliability tables. Fits calibrators on a clean 25%
        held out from TRAIN (never the test set)."""
        model = state.champion_model
        if model is None:
            return state
        X_test, y_test = state.X_test, state.y_test

        # Uncalibrated probabilities + metrics
        y_prob_uncal = model.predict_proba(X_test)[:, 1]
        brier_uncal = brier_score_loss(y_test, y_prob_uncal)
        ece_uncal = self._compute_ece(y_test, y_prob_uncal, n_bins=10)

        # Clean calibration set from TRAIN (not test), 25% held out, no sample_weight
        from sklearn.model_selection import train_test_split
        X_tr_base, X_cal, y_tr_base, y_cal = train_test_split(
            state.X_train, state.y_train, test_size=0.25,
            random_state=42, stratify=state.y_train
        )

        # Freeze the already-fitted champion so calibrators fit on top without
        # refitting it (modern replacement for the removed cv='prefit').
        base = FrozenEstimator(model) if FrozenEstimator is not None else model
        _prefit_kw = {} if FrozenEstimator is not None else {'cv': 'prefit'}

        # Platt scaling (sigmoid)
        calibrator_sigmoid = CalibratedClassifierCV(base, method='sigmoid', **_prefit_kw)
        calibrator_sigmoid.fit(X_cal, y_cal)
        y_prob_sigmoid = calibrator_sigmoid.predict_proba(X_test)[:, 1]
        brier_sigmoid = brier_score_loss(y_test, y_prob_sigmoid)
        ece_sigmoid = self._compute_ece(y_test, y_prob_sigmoid, n_bins=10)

        # Isotonic
        calibrator_isotonic = CalibratedClassifierCV(base, method='isotonic', **_prefit_kw)
        calibrator_isotonic.fit(X_cal, y_cal)
        y_prob_isotonic = calibrator_isotonic.predict_proba(X_test)[:, 1]
        brier_isotonic = brier_score_loss(y_test, y_prob_isotonic)
        ece_isotonic = self._compute_ece(y_test, y_prob_isotonic, n_bins=10)

        # AUC is invariant under monotonic calibration — verify
        auc_uncal = roc_auc_score(y_test, y_prob_uncal)
        auc_sigmoid = roc_auc_score(y_test, y_prob_sigmoid)

        # Recommend the lower-ECE method
        if ece_sigmoid <= ece_isotonic:
            recommended = 'sigmoid'
            y_prob_final = y_prob_sigmoid
        else:
            recommended = 'isotonic'
            y_prob_final = y_prob_isotonic

        reliability_uncal = self._build_reliability_table(y_test, y_prob_uncal, n_bins=10)
        reliability_cal   = self._build_reliability_table(y_test, y_prob_final, n_bins=10)

        state.calibration_results = {
            'uncalibrated': {'brier': round(brier_uncal, 4), 'ece': round(ece_uncal, 4), 'auc': round(auc_uncal, 4)},
            'sigmoid':      {'brier': round(brier_sigmoid, 4), 'ece': round(ece_sigmoid, 4), 'auc': round(auc_sigmoid, 4)},
            'isotonic':     {'brier': round(brier_isotonic, 4), 'ece': round(ece_isotonic, 4)},
            'recommended_method': recommended,
            'reliability_uncalibrated': reliability_uncal,
            'reliability_calibrated': reliability_cal,
            'ece_threshold': 0.02,
            'ece_status': 'EXCEEDS — needs calibration' if ece_uncal > 0.02 else 'OK',
        }

        self._info(f"Calibration: Uncalibrated ECE={ece_uncal:.4f} (threshold 0.02)")
        self._info(f"  Sigmoid: Brier={brier_sigmoid:.4f} ECE={ece_sigmoid:.4f}")
        self._info(f"  Isotonic: Brier={brier_isotonic:.4f} ECE={ece_isotonic:.4f}")
        self._info(f"  Recommended: {recommended} | AUC unchanged: {auc_uncal:.4f} -> {auc_sigmoid:.4f}")
        return state

    def _confusion_matrix_analysis(self, state: PipelineState) -> PipelineState:
        """Confusion matrix + classification report at a 0.5 reporting threshold,
        computed 3-way (Train / Test / OOT) to mirror the discrimination metrics."""
        model = state.champion_model
        if model is None:
            return state

        def compute_cm(X, y, label):
            if X is None or y is None:
                return None
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            report = classification_report(y, y_pred, output_dict=True,
                                           target_names=['Good (0)', 'Bad (1)'], zero_division=0)
            return {
                'sample': label, 'threshold': 0.5, 'n': int(len(y)),
                'true_positive': int(tp), 'false_positive': int(fp),
                'true_negative': int(tn), 'false_negative': int(fn),
                'accuracy': round(accuracy, 4), 'precision': round(precision, 4),
                'recall': round(recall, 4), 'specificity': round(specificity, 4),
                'f1_score': round(f1, 4), 'classification_report': report,
                'note': '0.5 threshold used for reporting only — NOT the business decision threshold.',
            }

        cm_train = compute_cm(state.X_train, state.y_train, 'Train')
        cm_test  = compute_cm(state.X_test, state.y_test, 'Test')
        cm_oot   = compute_cm(state.X_oot, state.y_oot, 'OOT') if state.X_oot is not None else None

        # Keep the legacy flat keys pointing at Test (backward compatibility).
        state.validation_metrics['confusion_matrix'] = cm_test
        state.validation_metrics['confusion_matrix_train'] = cm_train
        state.validation_metrics['confusion_matrix_test'] = cm_test
        state.validation_metrics['confusion_matrix_oot'] = cm_oot
        if cm_test:
            state.validation_metrics['classification_report'] = cm_test['classification_report']

        for cm, label in [(cm_train, 'Train'), (cm_test, 'Test'), (cm_oot, 'OOT')]:
            if cm:
                self._info(f"Confusion Matrix — {label}: N={cm['n']} "
                           f"Precision={cm['precision']:.3f} Recall={cm['recall']:.3f} F1={cm['f1_score']:.3f}")
        return state

    def _compute_ece(self, y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        y_arr = np.asarray(y_true)
        ece = 0.0
        total = len(y_arr)
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            bin_conf = y_prob[mask].mean()
            bin_acc  = y_arr[mask].mean()
            ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
        return float(ece)

    def _build_reliability_table(self, y_true, y_prob, n_bins=10):
        df = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(y_prob)})
        df['bin'] = pd.qcut(df['p'], n_bins, labels=False, duplicates='drop')
        table = df.groupby('bin').agg(
            count=('y', 'size'),
            mean_predicted=('p', 'mean'),
            actual_rate=('y', 'mean'),
        ).reset_index()
        table['gap'] = table['actual_rate'] - table['mean_predicted']
        return table.round(4).to_dict('records')

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
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)

        state.model_report_path = path
        self._info(f"Model report saved → {path}")

        # Save champion model for LIME what-if analysis in the UI
        try:
            import joblib
            pkl_path = os.path.join(self.output_dir, "champion_model.pkl")
            joblib.dump({
                "model":            state.champion_model,
                "selected_features": state.selected_features,
            }, pkl_path)
            self._info(f"Champion model saved → {pkl_path}")
        except Exception as e:
            self._info(f"Could not save champion model pkl: {e}")

        return state

    # ─────────────────────────────────────────────────────────────
    def _save_audit(self, state: PipelineState) -> PipelineState:
        path = os.path.join(self.output_dir, f"{state.run_id}_audit_trail.json")
        summary = state.to_summary_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        self._info(f"Audit trail saved → {path}")
        return state
