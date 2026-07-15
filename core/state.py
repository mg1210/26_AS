"""
core/state.py
─────────────
Shared pipeline state object passed between every agent.
Acts as the single source of truth for the entire factory run.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import json
import os


@dataclass
class PipelineState:
    # ── Run metadata ──────────────────────────────────────────────
    run_id: str = field(default_factory=lambda: datetime.now().strftime("RUN_%Y%m%d_%H%M%S"))
    dataset_name: str = ""
    dataset_path: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ── Raw data ──────────────────────────────────────────────────
    raw_df: Any = None                  # original DataFrame (never mutated)
    working_df: Any = None              # mutable working copy
    headerless: bool = False            # True if source CSV had no header row (cols auto-named col_0, col_1, …)

    # ── Phase 1: Data understanding ───────────────────────────────
    schema_profile: dict = field(default_factory=dict)
    data_dictionary: dict = field(default_factory=dict)  # {col: description} from uploaded xlsx
    target_column: str = ""
    target_definition: str = ""
    target_mapping: dict = field(default_factory=dict)        # raw_value → 0/1/None
    indeterminate_values: list = field(default_factory=list)  # [{value, count, reason}]
    behavioral_leakage_flags: dict = field(default_factory=dict)  # col → auc
    schema_drift: dict = field(default_factory=dict)
    leakage_columns: list = field(default_factory=list)
    drop_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    date_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)

    # ── Phase 2: DQR ─────────────────────────────────────────────
    dqr_report: dict = field(default_factory=dict)
    missing_summary: dict = field(default_factory=dict)
    outlier_summary: dict = field(default_factory=dict)
    high_missing_cols: list = field(default_factory=list)   # >40% missing
    dqr_flags: list = field(default_factory=list)           # human-readable warnings

    # ── Phase 3: Feature engineering ─────────────────────────────
    engineered_df: Any = None
    feature_log: list = field(default_factory=list)         # what was created and why
    woe_bins: dict = field(default_factory=dict)            # WOE encoding maps
    imputation_log: list = field(default_factory=list)      # per-column imputation strategy + reason
    imputation_map: dict = field(default_factory=dict)      # col -> {strategy, fill_value, reason} for Dataset 2 scoring

    # ── Phase 4: Variable selection ───────────────────────────────
    iv_table: Any = None                # DataFrame with IV scores
    selected_features: list = field(default_factory=list)
    rejected_features: dict = field(default_factory=dict)  # feature -> reason
    correlation_matrix: Any = None

    # ── Phase 5: Model development ───────────────────────────────
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    X_oot: Any = None                                       # out-of-time holdout (time-based split)
    y_oot: Any = None
    split_method: str = 'random'                            # 'time_based' or 'random'
    split_details: dict = field(default_factory=dict)       # sizes + default rates per set
    trained_models: dict = field(default_factory=dict)      # name -> model object
    model_metrics: dict = field(default_factory=dict)       # name -> metrics dict
    champion_model_name: str = ""
    champion_model: Any = None
    champion_model_path: str = ""                           # persisted champion .pkl for scoring new data
    model_selection_rationale: str = ""
    optuna_trials_run: int = 0                              # actual number of Optuna trials executed
    optuna_trials_history: dict = field(default_factory=dict)  # best/worst/top-3 trial summary

    # ── Phase 6: Explainability ───────────────────────────────────
    shap_values: Any = None
    feature_importance: dict = field(default_factory=dict)
    shap_summary: str = ""
    adverse_action_codes: dict = field(default_factory=dict)
    fairness_results: dict = field(default_factory=dict)    # proxy-attribute group parity

    # ── Phase 7: Validation ───────────────────────────────────────
    validation_metrics: dict = field(default_factory=dict)
    psi_results: dict = field(default_factory=dict)
    calibration_results: dict = field(default_factory=dict)   # uncal vs sigmoid vs isotonic
    validation_summary: str = ""
    validation_passed: bool = False

    # ── Phase 8: Documentation ────────────────────────────────────
    model_report_path: str = ""
    chart_paths: dict = field(default_factory=dict)         # key -> PNG path for embedded report charts
    audit_log: list = field(default_factory=list)

    # ── Human checkpoint flags ────────────────────────────────────
    checkpoint_1_approved: bool = False   # target definition confirmed
    checkpoint_2_approved: bool = False   # feature shortlist approved
    checkpoint_3_approved: bool = False   # model sign-off

    # ── Agent structured responses ────────────────────────────────
    agent_responses: dict = field(default_factory=dict)

    # ── Structured recommendations ────────────────────────────────
    recommendations: list = field(default_factory=list)

    # ── Errors and warnings ───────────────────────────────────────
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def log_audit(self, agent: str, action: str, detail: str = ""):
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "detail": detail
        })

    def log_error(self, agent: str, msg: str):
        self.errors.append({"agent": agent, "message": msg, "timestamp": datetime.now().isoformat()})

    def log_warning(self, agent: str, msg: str):
        self.warnings.append({"agent": agent, "message": msg, "timestamp": datetime.now().isoformat()})

    def to_summary_dict(self) -> dict:
        """Return a JSON-serialisable summary (no DataFrames, no numpy types)."""
        import pandas as pd

        def _safe(obj):
            """Round-trip through JSON to strip numpy / non-serialisable types."""
            if obj is None:
                return None
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return None

        def _df_safe(obj):
            """Convert DataFrame → list-of-dicts, then sanitise; pass dicts through _safe."""
            try:
                if isinstance(obj, pd.DataFrame):
                    return _safe(obj.to_dict(orient="records"))
            except Exception:
                pass
            return _safe(obj)

        def _schema_safe(schema):
            """Serialise schema_profile column-by-column so one bad column can't wipe the rest."""
            if not schema:
                return {}
            out = {}
            for col, info in schema.items():
                try:
                    out[col] = _safe(info) or {}
                except Exception:
                    out[col] = {}
            return out

        def _woe_safe(woe_bins):
            """WOE bin values may be DataFrames (one per feature); convert each."""
            if not woe_bins:
                return {}
            out = {}
            for feat, bins in woe_bins.items():
                out[feat] = _df_safe(bins)
            return out

        return {
            # ── Run metadata ─────────────────────────────────────────
            "run_id":            self.run_id,
            "dataset_name":      self.dataset_name,
            "headerless":        self.headerless,
            "target_column":     self.target_column,
            "target_definition": self.target_definition,

            # ── Phase 1: Data understanding ──────────────────────────
            "data_dictionary":    _safe(self.data_dictionary) or {},
            "leakage_columns":    self.leakage_columns,
            "id_columns":         self.id_columns,
            "date_columns":       self.date_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_columns":    self.numeric_columns,
            "high_missing_cols":  self.high_missing_cols,
            "schema_profile":     _schema_safe(self.schema_profile),
            "target_mapping":           _safe(self.target_mapping)             or {},
            "indeterminate_values":     _safe(self.indeterminate_values)       or [],
            "behavioral_leakage_flags": _safe(self.behavioral_leakage_flags)   or {},

            # ── Phase 2: DQR ─────────────────────────────────────────
            "dqr_flags":      self.dqr_flags,
            "dqr_report":     _safe(self.dqr_report)     or {},
            "missing_summary": _safe(self.missing_summary) or {},
            "outlier_summary": _safe(self.outlier_summary) or {},

            # ── Phase 3: Feature engineering ─────────────────────────
            "feature_log": _safe(self.feature_log) or [],
            "woe_bins":    _woe_safe(self.woe_bins),
            "imputation_log": _safe(self.imputation_log) or [],
            "imputation_map": _safe(self.imputation_map) or {},

            # ── Phase 4: Variable selection ──────────────────────────
            "iv_table":         _df_safe(self.iv_table) or [],
            "selected_features": self.selected_features,
            "rejected_features": _safe(self.rejected_features) or {},

            # ── Phase 5: Model development ───────────────────────────
            "champion_model_name":       self.champion_model_name,
            "champion_model_path":       self.champion_model_path,
            "model_selection_rationale": self.model_selection_rationale,
            "model_metrics":             _safe(self.model_metrics) or {},
            "optuna_trials_run":         self.optuna_trials_run,
            "optuna_trials_history":     _safe(self.optuna_trials_history) or {},
            "split_method":              self.split_method,
            "split_details":             _safe(self.split_details) or {},

            # ── Phase 6: Explainability ──────────────────────────────
            "feature_importance":  _safe(self.feature_importance)  or {},
            "shap_summary":        self.shap_summary,
            "adverse_action_codes": _safe(self.adverse_action_codes) or {},
            "fairness_results":    _safe(self.fairness_results) or {},

            # ── Phase 7: Validation ──────────────────────────────────
            "validation_metrics": _safe(self.validation_metrics) or {},
            "kpi_scoreboard":     _safe(self.validation_metrics.get("kpi_scoreboard"))    or [],
            "feature_kpi_table":  _safe(self.validation_metrics.get("feature_kpi_table")) or [],
            "findings_register":  _safe(self.validation_metrics.get("findings_register")) or [],
            "psi_results":        _safe(self.psi_results)        or {},
            "calibration_results": _safe(self.calibration_results) or {},
            "validation_summary": self.validation_summary,
            "validation_passed":  self.validation_passed,

            # ── Checkpoints ──────────────────────────────────────────
            "checkpoints": {
                "target_confirmed":  self.checkpoint_1_approved,
                "features_approved": self.checkpoint_2_approved,
                "model_signed_off":  self.checkpoint_3_approved,
            },

            # ── Agent responses ───────────────────────────────────────
            "agent_responses": _safe(self.agent_responses) or {},

            # ── Recommendations ───────────────────────────────────────
            "recommendations": [r.to_dict() for r in self.recommendations],

            # ── Diagnostics ──────────────────────────────────────────
            "chart_paths": _safe(self.chart_paths) or {},
            "warnings":  self.warnings,
            "errors":    self.errors,
            "audit_log": self.audit_log,
        }
