"""
orchestrator.py
────────────────
Master Orchestrator — manages the full Credit Risk Factory pipeline.

Responsibilities:
  • Routes data through each specialist agent in sequence
  • Manages the 3 human-in-the-loop checkpoints
  • Handles errors gracefully (skip phase vs abort)
  • Provides progress updates throughout
  • Supports Dataset 2 plug-and-play reuse via run()
"""

import os
import sys
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

from core.state import PipelineState
from agents.data_understanding_agent import DataUnderstandingAgent
from agents.dqr_agent import DQRAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.variable_selection_agent import VariableSelectionAgent
from agents.model_development_agent import ModelDevelopmentAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.validation_agent import ValidationAgent
from agents.documentation_agent import DocumentationAgent


BANNER = """
╔══════════════════════════════════════════════════════════╗
║        AGENTIC CREDIT RISK FACTORY  —  Dhurin 2026      ║
║          Observe  →  Learn  →  Explain  →  Act          ║
╚══════════════════════════════════════════════════════════╝
"""


class CreditRiskOrchestrator:

    def __init__(self,
                 output_dir: str = "outputs",
                 optuna_trials: int = 30,
                 auto_approve: bool = False,    # True = skip human prompts (demo mode)
                 verbose: bool = True,
                 hyperparam_override: bool = False):
        self.output_dir    = output_dir
        self.optuna_trials = optuna_trials
        self.auto_approve  = auto_approve
        self.verbose       = verbose
        self.hyperparam_override = hyperparam_override
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    def run(self, dataset_path: str, dataset_name: str = "",
            target_col: str = "") -> PipelineState:
        """
        Main entry point.  Call this with a new CSV path and the entire
        pipeline re-runs automatically — Dataset 2 plug-and-play.

        target_col: if non-empty, skips auto-detection and uses this column.
        """
        print(Fore.CYAN + BANNER)

        state = PipelineState(
            dataset_path=dataset_path,
            dataset_name=dataset_name or os.path.basename(dataset_path),
            target_column=target_col,   # pre-set skips auto-detection in Phase 1
        )

        # ── Scan data/ for an xlsx data dictionary ───────────────
        import glob as _glob
        _data_dir = os.path.dirname(dataset_path) or "data"
        _xlsx_files = sorted(_glob.glob(os.path.join(_data_dir, "*.xlsx")))
        _dict_path = _xlsx_files[0] if _xlsx_files else ""
        if _dict_path:
            print(f"  📖 Data dictionary found: {os.path.basename(_dict_path)}")

        # ── Agent registry (ordered) ──────────────────────────────
        phases = [
            ("Phase 1 — Data Understanding",   DataUnderstandingAgent(self.verbose, data_dict_path=_dict_path)),
            ("Phase 2 — Data Quality Review",  DQRAgent(verbose=self.verbose)),
            ("Phase 3 — Feature Engineering",  FeatureEngineeringAgent(self.verbose)),
            ("Phase 4 — Variable Selection",   VariableSelectionAgent(self.verbose)),
            ("Phase 5 — Model Development",    ModelDevelopmentAgent(
                                                   optuna_trials=self.optuna_trials,
                                                   verbose=self.verbose,
                                                   hyperparam_override=self.hyperparam_override,
                                                   output_dir=self.output_dir)),
            ("Phase 6 — Explainability",       ExplainabilityAgent(self.verbose)),
            ("Phase 7 — Validation",           ValidationAgent(self.output_dir, self.verbose)),
            ("Phase 8 — Documentation",        DocumentationAgent(self.output_dir, self.verbose)),
        ]

        for i, (phase_name, agent) in enumerate(phases):
            self._phase_header(phase_name)
            state = agent.execute(state)

            # Human checkpoints
            if i == 0:   # After Data Understanding
                state = self._checkpoint_1(state)
                if not state.checkpoint_1_approved:
                    self._abort("Target definition not approved.")
                    return state

            if i == 3:   # After Variable Selection
                state = self._checkpoint_2(state)
                if not state.checkpoint_2_approved:
                    self._abort("Feature shortlist not approved.")
                    return state

            if i == 6:   # After Validation — interactive mode still needs the prompt
                if not self.auto_approve:
                    state = self._checkpoint_3(state)
                elif self.auto_approve:
                    state.checkpoint_3_approved = True

            # Halt on critical error
            if state.errors:
                last_err = state.errors[-1]
                print(Fore.RED + f"\n  ✗ Error in {last_err['agent']}: {last_err['message']}")
                if not self._ask_continue():
                    return state

        self._print_final_summary(state)
        return state

    # ─────────────────────────────────────────────────────────────
    # Human checkpoint implementations
    # ─────────────────────────────────────────────────────────────

    def _checkpoint_1(self, state: PipelineState) -> PipelineState:
        print(Fore.YELLOW + "\n" + "━" * 60)
        print(Fore.YELLOW + "  ✦  HUMAN CHECKPOINT 1 — Target Definition Review")
        print(Fore.YELLOW + "━" * 60)
        print(f"\n  Target column  : {state.target_column}")
        print(f"  Definition     : {state.target_definition}")
        print(f"\n  Leakage columns flagged ({len(state.leakage_columns)}):")
        for c in state.leakage_columns:
            print(f"    • {c}")
        print(f"\n  DQR flags raised: {len(state.dqr_flags)}")
        for flag in state.dqr_flags[:5]:
            print(f"    {flag}")

        if self.auto_approve:
            print(Fore.GREEN + "\n  [AUTO-APPROVE] Target definition confirmed ✓")
            state.checkpoint_1_approved = True
        else:
            ans = input(Fore.WHITE + "\n  Approve target definition? [y/n]: ").strip().lower()
            if ans == "y":
                state.checkpoint_1_approved = True
                state.log_audit("Orchestrator", "checkpoint_1_approved", "Human confirmed target")
                print(Fore.GREEN + "  ✓ Target definition confirmed")
            else:
                note = input("  Enter correction note: ").strip()
                state.log_audit("Orchestrator", "checkpoint_1_rejected", note)
                print(Fore.RED + "  ✗ Target definition rejected — pipeline aborted")
        return state

    def _checkpoint_2(self, state: PipelineState) -> PipelineState:
        print(Fore.YELLOW + "\n" + "━" * 60)
        print(Fore.YELLOW + "  ✦  HUMAN CHECKPOINT 2 — Feature Shortlist Approval")
        print(Fore.YELLOW + "━" * 60)
        print(f"\n  Selected features ({len(state.selected_features)}):")

        if state.iv_table is not None:
            sel_iv = (state.iv_table[state.iv_table["feature"]
                      .isin(state.selected_features)]
                      .sort_values("iv", ascending=False)
                      .head(15))
            for _, row in sel_iv.iterrows():
                print(f"    {row['feature']:<40} IV={row['iv']:.4f}  [{row['strength']}]")

        print(f"\n  Rejected features: {len(state.rejected_features)}")
        sample_rejected = list(state.rejected_features.items())[:5]
        for feat, reason in sample_rejected:
            print(f"    ✗ {feat}: {reason}")

        if state.dqr_report.get("variable_selection_rationale"):
            print(f"\n  LLM Rationale (excerpt):")
            print("  " + state.dqr_report["variable_selection_rationale"][:300] + " …")

        if self.auto_approve:
            print(Fore.GREEN + "\n  [AUTO-APPROVE] Feature shortlist confirmed ✓")
            state.checkpoint_2_approved = True
        else:
            ans = input(Fore.WHITE + "\n  Approve feature shortlist? [y/n]: ").strip().lower()
            if ans == "y":
                state.checkpoint_2_approved = True
                state.log_audit("Orchestrator", "checkpoint_2_approved", "Human confirmed shortlist")
                print(Fore.GREEN + "  ✓ Feature shortlist approved")
            else:
                note = input("  Enter correction note (e.g. 'remove int_rate_clean'): ").strip()
                # Allow user to remove a specific feature
                if note.strip():
                    to_remove = [f.strip() for f in note.split(",") if f.strip() in state.selected_features]
                    for f in to_remove:
                        state.selected_features.remove(f)
                        state.rejected_features[f] = f"Removed by human reviewer"
                    if to_remove:
                        print(Fore.YELLOW + f"  Removed: {to_remove}")
                state.checkpoint_2_approved = True
                state.log_audit("Orchestrator", "checkpoint_2_modified", note)
                print(Fore.GREEN + "  ✓ Modified shortlist approved")
        return state

    def _checkpoint_3(self, state: PipelineState) -> PipelineState:
        print(Fore.YELLOW + "\n" + "━" * 60)
        print(Fore.YELLOW + "  ✦  HUMAN CHECKPOINT 3 — Model Sign-Off")
        print(Fore.YELLOW + "━" * 60)
        vm = state.validation_metrics
        psi = state.psi_results
        print(f"\n  Champion model : {state.champion_model_name}")
        print(f"  AUC            : {vm.get('auc')}")
        print(f"  Gini           : {vm.get('gini')}")
        print(f"  KS             : {vm.get('ks')}")
        print(f"  Brier Score    : {vm.get('brier_score')}")
        print(f"  PSI            : {psi.get('psi_score')} — {psi.get('assessment')}")
        print(f"  Validation     : {'PASS ✓' if state.validation_passed else 'CONDITIONAL ⚠'}")
        print(f"\n  Validation Summary (excerpt):")
        print("  " + (state.validation_summary or "N/A")[:400] + " …")
        print(f"\n  Report saved   : {state.model_report_path}")

        if self.auto_approve:
            print(Fore.GREEN + "\n  [AUTO-APPROVE] Model signed off ✓")
            state.checkpoint_3_approved = True
        else:
            ans = input(Fore.WHITE + "\n  Sign off model for deployment? [y/n]: ").strip().lower()
            if ans == "y":
                state.checkpoint_3_approved = True
                state.log_audit("Orchestrator", "checkpoint_3_approved", "Model signed off by human")
                print(Fore.GREEN + "  ✓ Model signed off")
            else:
                note = input("  Enter sign-off rejection note: ").strip()
                state.log_audit("Orchestrator", "checkpoint_3_rejected", note)
                print(Fore.YELLOW + "  ⚠ Model sign-off deferred — see audit trail")
        return state

    # ─────────────────────────────────────────────────────────────
    def _phase_header(self, name: str):
        print(Fore.CYAN + f"\n{'━'*60}")
        print(Fore.CYAN + f"  {name}")
        print(Fore.CYAN + f"{'━'*60}")

    def _abort(self, reason: str):
        print(Fore.RED + f"\n  Pipeline aborted: {reason}")

    def _ask_continue(self) -> bool:
        if self.auto_approve:
            return True
        ans = input("  Continue despite error? [y/n]: ").strip().lower()
        return ans == "y"

    def _print_final_summary(self, state: PipelineState):
        print(Fore.GREEN + "\n" + "═" * 60)
        print(Fore.GREEN + "  PIPELINE COMPLETE")
        print(Fore.GREEN + "═" * 60)
        print(f"  Run ID           : {state.run_id}")
        print(f"  Champion model   : {state.champion_model_name}")
        vm = state.validation_metrics
        print(f"  AUC / KS / Gini  : {vm.get('auc')} / {vm.get('ks')} / {vm.get('gini')}")
        print(f"  Features used    : {len(state.selected_features)}")
        print(f"  Validation       : {'PASS ✓' if state.validation_passed else 'CONDITIONAL ⚠'}")
        print(f"  Checkpoints      : "
              f"{'✓' if state.checkpoint_1_approved else '✗'} "
              f"{'✓' if state.checkpoint_2_approved else '✗'} "
              f"{'✓' if state.checkpoint_3_approved else '✗'}")
        print(f"  Report           : {state.model_report_path}")
        print(Fore.GREEN + "═" * 60 + "\n")
