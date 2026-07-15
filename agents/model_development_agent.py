"""
agents/model_development_agent.py
───────────────────────────────────
Phase 5 — Model Development Agent

Responsibilities:
  • Train-test split (stratified, time-aware if vintage available)
  • Train 4 candidate models: Logistic Regression, Random Forest, XGBoost,
    GradientBoosting_AutoML (GridSearchCV)
  • Hyperparameter optimisation via Optuna (XGBoost) and GridSearchCV (GB)
  • Compute performance metrics: AUC, KS, Gini, Precision, Recall, F1
  • LLM-driven champion selection with written rationale
"""

import os
import json
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_fscore_support, confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from core.base_agent import BaseAgent
from core.state import PipelineState
from core.llm import ask, CREDIT_RISK_SYSTEM
from core.recommendation import Recommendation


class ModelDevelopmentAgent(BaseAgent):

    def __init__(self, test_size: float = 0.25, optuna_trials: int = 30,
                 verbose: bool = True, hyperparam_override: bool = False,
                 output_dir: str = "outputs"):
        super().__init__("ModelDevelopmentAgent", verbose)
        self.test_size     = test_size
        self.optuna_trials = optuna_trials
        self.hyperparam_override = hyperparam_override
        self.output_dir    = output_dir
        self._hp_ov = None            # loaded override (if --hyperparam-override)
        self._applied_override = False

    # ── HITL hyperparameter override helpers ─────────────────────────
    def _load_hp_override(self):
        """Return an active (applied=False) hyperparameter_override dict, else None."""
        if not self.hyperparam_override:
            return None
        try:
            with open(os.path.join(self.output_dir, "checkpoints.json"), encoding="utf-8") as f:
                cps = json.load(f)
            ov = cps.get("hyperparameter_override")
            if ov and not ov.get("applied"):
                return ov
        except Exception:
            pass
        return None

    def _override_params(self, model_name: str, base_params: dict):
        """Merge an active override for this model over base_params. Returns
        (params, overridden_bool)."""
        ov = self._hp_ov
        if ov and ov.get("model") == model_name and ov.get("params"):
            merged = dict(base_params)
            merged.update(ov["params"])
            self._applied_override = True
            self._info(f"HITL hyperparameter override applied to {model_name}: {ov['params']}")
            return merged, True
        return base_params, False

    def _mark_override_applied(self):
        """Persist applied=True so the override is used once, not on every run."""
        if not self._applied_override:
            return
        path = os.path.join(self.output_dir, "checkpoints.json")
        try:
            with open(path, encoding="utf-8") as f:
                cps = json.load(f)
            if cps.get("hyperparameter_override"):
                cps["hyperparameter_override"]["applied"] = True
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cps, f, indent=2, default=str)
                self._info("Hyperparameter override marked applied=True in checkpoints.json")
        except Exception as e:
            self._info(f"Could not mark override applied: {e}")

    # ─────────────────────────────────────────────────────────────
    def run(self, state: PipelineState) -> PipelineState:
        self._hp_ov = self._load_hp_override()
        self._applied_override = False
        df = state.engineered_df.copy()
        feats = [f for f in state.selected_features if f in df.columns]
        self._info(f"Modelling on {len(feats)} selected features")

        X = df[feats].fillna(0)
        y = df["target"]

        self._log("Creating train / test split …")
        state = self._split(state, X, y)

        self._log("Training Logistic Regression …")
        state = self._train_logistic(state)

        self._log("Training Random Forest …")
        state = self._train_rf(state)

        self._log("Training XGBoost with Optuna tuning …")
        state = self._train_xgb(state)

        self._log("Training LightGBM …")
        state = self._train_lightgbm(state)

        self._log("Training GradientBoosting with GridSearchCV …")
        state = self._train_gb_automl(state)

        self._mark_override_applied()

        self._log("Selecting champion model …")
        state = self._select_champion(state)

        self._log("Asking LLM for model selection rationale …")
        state = self._llm_rationale(state)

        # Build structured response
        champ    = state.champion_model_name
        champ_m  = state.model_metrics.get(champ, {})
        state.agent_responses[self.name] = self.build_response(
            summary="Model development complete",
            observations=[
                f"Models trained: {', '.join(state.model_metrics.keys())}",
                f"Champion model: {champ}",
                f"Champion AUC (test): {champ_m.get('auc_test', 'N/A')}",
                f"Champion KS: {champ_m.get('ks', 'N/A')}",
                f"Champion Gini: {champ_m.get('gini', 'N/A')}",
                f"Champion overfit (train-test AUC delta): {champ_m.get('overfit', 'N/A')}",
            ],
            reasoning=state.model_selection_rationale or (
                "Champion selected via AUC on held-out test set with overfit penalty: "
                "score = auc_test - max(0, overfit - 0.03) * 2. "
                "Penalises models with train-test gap > 3pp to favour generalisable models."
            ),
            recommendations=[
                "Validate champion model on an independent out-of-time sample",
                "Run PSI analysis to assess population stability before deployment",
                "Review hyperparameters for regulatory model documentation",
                "Consider ensemble or stacking if AUC is below 0.70",
            ],
            artifacts={"models_trained": list(state.model_metrics.keys())},
        )

        champ_auc    = float(champ_m.get("auc_test") or 0.0)
        champ_overfit = float(champ_m.get("overfit") or 0.0)
        state.recommendations.append(Recommendation(
            title="Champion Model Selection",
            recommendation=f"{champ} selected as champion model",
            rationale=state.model_selection_rationale or (
                f"Champion selected via AUC on held-out test set with overfit penalty. "
                f"AUC={champ_auc:.4f}, overfit delta={champ_overfit:.4f}."
            ),
            confidence=champ_auc,
            risk="medium" if champ_overfit > 0.03 else "low",
            requires_human_approval=True,
        ))

        # ── Persist champion model + feature metadata for scoring new data ────
        try:
            import joblib, json as _json, os as _os
            _os.makedirs('outputs/models', exist_ok=True)
            model_path = f'outputs/models/{state.run_id}_champion_{state.champion_model_name}.pkl'
            joblib.dump(state.champion_model, model_path)
            state.champion_model_path = model_path

            # Persist the exact training imputation decisions so Dataset 2 scoring can
            # fill missing values identically (avoids train/serve skew from fillna(0)).
            imputation_map_path = f'outputs/models/{state.run_id}_imputation_map.json'
            with open(imputation_map_path, 'w') as _f:
                _json.dump(state.imputation_map, _f, indent=2, default=str)
            self._info(f"Imputation map saved ({len(state.imputation_map)} cols) → {imputation_map_path}")

            features_meta = {
                'run_id': state.run_id,
                'champion_model': state.champion_model_name,
                'champion_model_path': model_path,
                'selected_features': state.selected_features,
                'feature_dtypes': {f: str(state.engineered_df[f].dtype)
                                   for f in state.selected_features if f in state.engineered_df.columns},
                'target_column': state.target_column,
                'target_mapping': getattr(state, 'target_mapping', {}),
                'imputation_map_path': imputation_map_path,
            }
            meta_path = f'outputs/models/{state.run_id}_features_meta.json'
            with open(meta_path, 'w') as _f:
                _json.dump(features_meta, _f, indent=2, default=str)
            self._info(f"Champion model saved → {model_path}")
            self._info(f"Features metadata saved → {meta_path}")

            # Save raw-column statistical fingerprints so headerless new data can be
            # matched back to real column names in evaluate_oot.py.
            raw_cols_to_save = ['int_rate', 'loan_amnt', 'annual_inc', 'dti',
                                'revol_util', 'open_acc', 'total_acc', 'tot_cur_bal',
                                'total_rev_hi_lim', 'installment', 'grade', 'sub_grade',
                                'emp_length', 'home_ownership', 'verification_status',
                                'purpose', 'addr_state', 'term', 'delinq_2yrs',
                                'inq_last_6mths', 'pub_rec', 'revol_bal', 'mths_since_last_delinq']
            ref_stats = {}
            for col in raw_cols_to_save:
                if state.raw_df is not None and col in state.raw_df.columns:
                    s = pd.to_numeric(state.raw_df[col], errors='coerce').dropna()
                    if len(s) > 10:
                        ref_stats[col] = {
                            'mean': float(s.mean()),
                            'std':  float(s.std()),
                            'min':  float(s.min()),
                            'max':  float(s.max()),
                            'p25':  float(s.quantile(0.25)),
                            'p50':  float(s.quantile(0.50)),
                            'p75':  float(s.quantile(0.75)),
                        }
            stats_path = f'outputs/models/{state.run_id}_reference_stats.json'
            with open(stats_path, 'w') as _f:
                _json.dump(ref_stats, _f, indent=2)
            self._info(f"Reference stats saved for {len(ref_stats)} columns → {stats_path}")

            # Save the raw Dataset 1 column ORDER so a headerless new dataset (same
            # order, no header) can have these names assigned positionally in
            # evaluate_oot.py. Saved here (training only) — NOT in data_understanding,
            # which also runs during scoring and would overwrite this with new-data cols.
            if state.raw_df is not None:
                col_order_path = 'outputs/models/dataset1_column_order.json'
                with open(col_order_path, 'w') as _f:
                    _json.dump(state.raw_df.columns.tolist(), _f)
                self._info(f"Dataset 1 column order saved: {len(state.raw_df.columns)} columns → {col_order_path}")
        except Exception as e:
            state.log_warning(self.name, f"Could not persist champion model/meta: {e}")

        return state

    # ─────────────────────────────────────────────────────────────
    def _split(self, state: PipelineState, X, y) -> PipelineState:
        df_eng = state.engineered_df
        time_col = None
        if 'issue_year' in df_eng.columns:
            time_col = 'issue_year'
        elif 'loan_age_months' in df_eng.columns:
            time_col = 'loan_age_months'

        if time_col is not None:
            try:
                time_vals = df_eng[time_col].values
                sorted_idx = np.argsort(time_vals)
                n = len(sorted_idx)
                # Step 1: time-based OOT = latest 20% by issue_year (chronological).
                # Step 2: random stratified 70/30 Train/Test within the earliest-80% Dev pool.
                oot_start = int(n * 0.80)
                dev_idx = sorted_idx[:oot_start]
                oot_idx = sorted_idx[oot_start:]
                X_dev = X.iloc[dev_idx]; y_dev = y.iloc[dev_idx]
                X_oot = X.iloc[oot_idx]; y_oot = y.iloc[oot_idx]
                from sklearn.model_selection import train_test_split as tts
                X_train, X_test, y_train, y_test = tts(
                    X_dev, y_dev, test_size=0.30, random_state=42, stratify=y_dev)
                state.X_train = X_train; state.y_train = y_train
                state.X_test  = X_test;  state.y_test  = y_test
                state.X_oot   = X_oot;   state.y_oot   = y_oot
                state.split_method = 'time_based'
                state.split_details = {
                    'method': 'OOT: time-based latest 20% | Train/Test: random 70/30 within Dev 80%',
                    'total': n, 'dev_size': len(dev_idx),
                    'train_size': len(X_train), 'test_size': len(X_test), 'oot_size': len(oot_idx),
                    'dev_pct': 80, 'oot_pct': 20, 'train_pct': 70, 'test_pct': 30,
                    'train_default_rate': float(y_train.mean()),
                    'test_default_rate': float(y_test.mean()),
                    'oot_default_rate': float(y_oot.mean()),
                    'time_col_used': time_col,
                    'note': 'OOT=latest 20% by issue_year. Train/Test=random stratified 70/30 within Dev 80%'
                }
                self._info(f"3-way split: Train={len(X_train):,}(56%) | Test={len(X_test):,}(24%) | OOT={len(oot_idx):,}(20%)")
                self._info(f"Default rates — Train:{y_train.mean():.2%} | Test:{y_test.mean():.2%} | OOT:{y_oot.mean():.2%}")
                return state
            except Exception as e:
                state.log_warning(self.name, f"Time-based split failed: {e} — falling back to random")

        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=42, stratify=y)
        state.X_train = X_train; state.y_train = y_train
        state.X_test = X_test;   state.y_test = y_test
        state.X_oot = None;      state.y_oot = None
        state.split_method = 'random'
        state.split_details = {'method': 'Random fallback 75/25', 'train_size': len(X_train), 'test_size': len(X_test)}
        self._info(f"Random fallback split: Train={len(X_train):,} | Test={len(X_test):,}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _metrics(self, model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
        y_prob_tr = model.predict_proba(X_tr)[:, 1]
        y_prob_te = model.predict_proba(X_te)[:, 1]
        y_pred_te = (y_prob_te >= 0.5).astype(int)

        auc_tr  = roc_auc_score(y_tr, y_prob_tr)
        auc_te  = roc_auc_score(y_te, y_prob_te)
        gini_te = 2 * auc_te - 1

        fpr, tpr, _ = roc_curve(y_te, y_prob_te)
        ks_te = float(np.max(tpr - fpr))

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred_te, average="binary", zero_division=0
        )
        return {
            "model"     : name,
            "auc_train" : round(auc_tr, 4),
            "auc_test"  : round(auc_te, 4),
            "gini"      : round(gini_te, 4),
            "ks"        : round(ks_te, 4),
            "precision" : round(float(prec), 4),
            "recall"    : round(float(rec), 4),
            "f1"        : round(float(f1), 4),
            "overfit"   : round(auc_tr - auc_te, 4),
        }

    # ─────────────────────────────────────────────────────────────
    def _train_logistic(self, state: PipelineState) -> PipelineState:
        lr_params = {"max_iter": 500, "C": 0.1, "class_weight": "balanced", "random_state": 42}
        lr_params, _ = self._override_params("LogisticRegression", lr_params)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_params)),
        ])
        pipe.fit(state.X_train, state.y_train)
        m = self._metrics(pipe, state.X_train, state.X_test,
                          state.y_train, state.y_test, "LogisticRegression")
        m["best_params"] = lr_params
        state.trained_models["LogisticRegression"] = pipe
        state.model_metrics["LogisticRegression"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_rf(self, state: PipelineState) -> PipelineState:
        rf_params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 50,
                     "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
        rf_params, _ = self._override_params("RandomForest", rf_params)
        rf = RandomForestClassifier(**rf_params)
        rf.fit(state.X_train, state.y_train)
        m = self._metrics(rf, state.X_train, state.X_test,
                          state.y_train, state.y_test, "RandomForest")
        m["best_params"] = rf_params
        state.trained_models["RandomForest"] = rf
        state.model_metrics["RandomForest"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_xgb(self, state: PipelineState) -> PipelineState:
        # Deterministic tuning so the champion is stable across runs.
        np.random.seed(42)
        random.seed(42)
        X_tr, y_tr = state.X_train, state.y_train
        X_te, y_te = state.X_test,  state.y_test
        scale_pos  = float((y_tr == 0).sum() / (y_tr == 1).sum())

        # HITL override → skip Optuna search and fit the supplied params directly.
        _xgb_ov, _overridden = self._override_params("XGBoost", {})
        if _overridden:
            best = {"scale_pos_weight": scale_pos, "random_state": 42, "eval_metric": "auc"}
            best.update(_xgb_ov)
            xgb_model = xgb.XGBClassifier(**best, verbosity=0)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            m = self._metrics(xgb_model, X_tr, X_te, y_tr, y_te, "XGBoost")
            m["best_params"] = best
            state.trained_models["XGBoost"] = xgb_model
            state.model_metrics["XGBoost"]  = m
            self._info(f"  (override) AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
            return state

        def objective(trial):
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
                "max_depth"        : trial.suggest_int("max_depth", 3, 7),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight" : trial.suggest_int("min_child_weight", 5, 50),
                "scale_pos_weight" : scale_pos,
                "random_state"     : 42,
                "eval_metric"      : "auc",
                "use_label_encoder": False,
            }
            model = xgb.XGBClassifier(**params, verbosity=0)
            cv_scores = cross_val_score(model, X_tr, y_tr, cv=3,
                                        scoring="roc_auc", n_jobs=-1)
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        # timeout bounds tuning wall-clock (whichever of n_trials / timeout hits first);
        # returns diminish quickly, so 45s keeps AUC within noise while trimming runtime.
        study.optimize(objective, n_trials=self.optuna_trials, timeout=45, show_progress_bar=False)

        # Capture the search history so the report can explain the tuning process.
        _vals = [t.value for t in study.trials if t.value is not None]
        _top = sorted([t for t in study.trials if t.value is not None],
                      key=lambda t: t.value, reverse=True)
        state.optuna_trials_run = len(study.trials)
        state.optuna_trials_history = {
            "n_trials":  len(study.trials),
            "best_auc":  round(study.best_value, 4),
            "best_params": study.best_params,
            "worst_auc": round(min(_vals), 4) if _vals else None,
            "top_3_trials": [{"trial": t.number, "auc": round(t.value, 4), "params": t.params}
                             for t in _top[:3]],
        }

        best = study.best_params
        best["scale_pos_weight"] = scale_pos
        best["random_state"]     = 42
        best["eval_metric"]      = "auc"

        xgb_model = xgb.XGBClassifier(**best, verbosity=0)
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_te, y_te)],
                      verbose=False)

        m = self._metrics(xgb_model, X_tr, X_te, y_tr, y_te, "XGBoost")
        m["best_params"] = best
        state.trained_models["XGBoost"] = xgb_model
        state.model_metrics["XGBoost"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        self._info(f"  Best params: {best}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_lightgbm(self, state: PipelineState) -> PipelineState:
        try:
            import lightgbm as lgb
        except ImportError:
            state.log_warning(self.name, "lightgbm not installed — skipping LightGBM model")
            self._info("LightGBM skipped (not installed — run: pip install lightgbm)")
            return state

        X_tr, y_tr = state.X_train, state.y_train
        X_te, y_te = state.X_test,  state.y_test
        scale_pos  = float((y_tr == 0).sum() / (y_tr == 1).sum())

        lgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
                      "num_leaves": 31, "scale_pos_weight": scale_pos,
                      "random_state": 42, "n_jobs": -1, "verbose": -1}
        lgb_params, _ = self._override_params("LightGBM", lgb_params)
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_tr, y_tr)

        m = self._metrics(model, X_tr, X_te, y_tr, y_te, "LightGBM")
        m["best_params"] = {k: v for k, v in lgb_params.items() if k not in ("n_jobs", "verbose")}
        state.trained_models["LightGBM"] = model
        state.model_metrics["LightGBM"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _train_gb_automl(self, state: PipelineState) -> PipelineState:
        X_tr, y_tr = state.X_train, state.y_train
        X_te, y_te = state.X_test,  state.y_test

        # Fixed, well-regularised configuration. This was previously wrapped in a
        # GridSearchCV over a SINGLE hyperparameter combination, so its 3-fold CV
        # added three extra GradientBoosting fits whose score was never used for
        # champion selection (champion = best held-out test AUC). GradientBoosting
        # is single-threaded and the slowest of the five models, so a direct fit
        # gives identical model quality at roughly a quarter of the training cost.
        best_params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        best_params, _ = self._override_params("GradientBoosting_AutoML", best_params)
        best_model  = GradientBoostingClassifier(random_state=42, **best_params)
        best_model.fit(X_tr, y_tr)

        m = self._metrics(best_model, X_tr, X_te, y_tr, y_te, "GradientBoosting_AutoML")
        m["best_params"] = best_params

        state.trained_models["GradientBoosting_AutoML"] = best_model
        state.model_metrics["GradientBoosting_AutoML"]  = m
        self._info(f"  AUC={m['auc_test']:.4f}  KS={m['ks']:.4f}  Gini={m['gini']:.4f}")
        self._info(f"  Params: {best_params}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _select_champion(self, state: PipelineState) -> PipelineState:
        """
        Champion = highest AUC on test set, with overfit penalty.
        If overfit > 0.03 AUC, penalise that model.
        """
        scores = {}
        for name, m in state.model_metrics.items():
            penalty = max(0, m["overfit"] - 0.03) * 2
            scores[name] = m["auc_test"] - penalty

        champion = max(scores, key=scores.get)
        state.champion_model_name = champion
        state.champion_model      = state.trained_models[champion]

        self._info(f"Champion: {champion}  (adj.score={scores[champion]:.4f})")
        for n, s in scores.items():
            self._info(f"  {n}: adj={s:.4f}  "
                       f"AUC={state.model_metrics[n]['auc_test']:.4f}  "
                       f"overfit={state.model_metrics[n]['overfit']:.4f}")
        return state

    # ─────────────────────────────────────────────────────────────
    def _llm_rationale(self, state: PipelineState) -> PipelineState:
        metrics_text = "\n".join(
            f"  {n}: AUC={m['auc_test']:.4f}, KS={m['ks']:.4f}, "
            f"Gini={m['gini']:.4f}, Overfit={m['overfit']:.4f}"
            for n, m in state.model_metrics.items()
        )
        prompt = f"""
You are a senior credit risk model validator reviewing candidate models for
a Lending Club binary default prediction scorecard.

MODEL COMPARISON:
{metrics_text}

SELECTED CHAMPION: {state.champion_model_name}

Write a Model Selection Rationale (max 200 words) for the model development report.
Address:
1. Why the champion was selected (performance, stability, interpretability)
2. Trade-offs vs the other candidates
3. Any governance or regulatory considerations for this choice
4. Recommended next steps for validation
"""
        try:
            rationale = ask(prompt, system=CREDIT_RISK_SYSTEM, max_tokens=500)
            state.model_selection_rationale = rationale
            self._info("LLM model selection rationale generated")
        except Exception as e:
            state.log_warning(self.name, f"LLM rationale skipped: {e}")
        return state
