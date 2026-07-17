"""
dataset2_pipeline.py — Blind Dataset 2 scoring workflow (7 phases)

Phase 1  Data Understanding      — schema, semantic types, business meanings, scope note
Phase 2  Data Quality Review     — scoped to champion features: missing, cardinality, outliers, dups
Phase 3  Feature Reconstruction  — derive engineered features from raw columns
Phase 4  Feature Stability       — per-feature PSI vs the training reference distribution
Phase 5  Explainability          — SHAP global importance on a sample (champion model)
Phase 6  Prediction              — score every row with the fixed champion + training imputation map
Phase 7  Validation              — metrics ONLY if a target column with valid mapped values is present

IMPORTANT: the target column is NEVER used in Phases 1–6. Phase 7 is the only phase that
looks for a target, and only computes metrics if one is found with valid mapped values.
If the file is headerless, an ``alignment_verified=False`` flag and an UNVERIFIED warning
are propagated through EVERY phase's result.
"""
import pandas as pd
import numpy as np
import json
import os
import glob
import joblib
import warnings
from datetime import datetime


# ── Shared loaders / helpers ────────────────────────────────────────────────

def load_champion_meta(run_id=None):
    """Load the fixed champion model metadata + feature list from Dataset 1's run."""
    if run_id:
        path = f'outputs/models/{run_id}_features_meta.json'
    else:
        files = sorted(glob.glob('outputs/models/*_features_meta.json'), reverse=True)
        if not files:
            files = sorted(glob.glob('sample_results/models/*_features_meta.json'), reverse=True)
        if not files:
            raise FileNotFoundError("No champion model metadata found — run Dataset 1 pipeline first")
        path = files[0]
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_champion_model(meta):
    """Load the champion .pkl (meta path first, then newest matching in outputs/ or sample_results/)."""
    model_path = meta.get('champion_model_path', '')
    if not model_path or not os.path.exists(model_path):
        candidates = sorted(glob.glob(f"outputs/models/*_{meta['champion_model']}.pkl"), reverse=True)
        candidates += sorted(glob.glob(f"sample_results/models/*_{meta['champion_model']}.pkl"), reverse=True)
        if not candidates:
            raise FileNotFoundError("Champion model file not found")
        model_path = candidates[0]
    return joblib.load(model_path)


def load_imputation_map(meta):
    """Load the exact training-time imputation decisions (col -> {strategy, fill_value})
    so missing values are filled identically at scoring time (no train/serve skew)."""
    path = meta.get('imputation_map_path', '')
    if not path or not os.path.exists(path):
        candidates = sorted(glob.glob('outputs/models/*_imputation_map.json'), reverse=True)
        candidates += sorted(glob.glob('sample_results/models/*_imputation_map.json'), reverse=True)
        if candidates:
            path = candidates[0]
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_reference_stats(meta):
    """Load per-raw-column training distribution stats (mean/std/min/p25/p50/p75/max)."""
    run_id = meta.get('run_id')
    path = f"outputs/models/{run_id}_reference_stats.json" if run_id else ''
    if not path or not os.path.exists(path):
        cands = sorted(glob.glob('outputs/models/*_reference_stats.json'), reverse=True)
        cands += sorted(glob.glob('sample_results/models/*_reference_stats.json'), reverse=True)
        if cands:
            path = cands[0]
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_scoring_matrix(df, meta):
    """Build the champion's input matrix X, filling missing cells with the EXACT training
    imputation values (sentinel/median) from the persisted map. Shared by Phase 5
    (explainability) and Phase 6 (prediction) so both see identical inputs.
    Indexed on df.index so scalar (missing→value) and Series (found) assignments both get
    the full row length (an empty frame would backfill early scalar columns to NaN)."""
    expected_features = meta['selected_features']
    imputation_map = load_imputation_map(meta)
    X = pd.DataFrame(index=df.index)
    imputation_applied = []
    for feat in expected_features:
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors='coerce')
            if feat in imputation_map:
                fill_val = imputation_map[feat]['fill_value']
                s = s.fillna(fill_val)
                imputation_applied.append({'feature': feat, 'fill_value': fill_val,
                                           'strategy': imputation_map[feat]['strategy']})
            else:
                s = s.fillna(0)
                imputation_applied.append({'feature': feat, 'fill_value': 0,
                                           'strategy': 'default (no training map entry)'})
            X[feat] = s
        else:
            # Feature genuinely missing entirely — use the training fill value as the best
            # guess (not a blanket 0, which for a sentinel column would misroute the model).
            if feat in imputation_map:
                X[feat] = imputation_map[feat]['fill_value']
                imputation_applied.append({'feature': feat, 'fill_value': imputation_map[feat]['fill_value'],
                                           'strategy': 'entire column missing, used training fill value'})
            else:
                X[feat] = 0
                imputation_applied.append({'feature': feat, 'fill_value': 0,
                                           'strategy': 'entire column missing, no training reference'})
    return X, imputation_applied


def reconstruct_engineered_features(df, expected_features):
    """Derive engineered features from raw columns if they're missing.

    Formulas mirror agents/feature_engineering_agent.py EXACTLY so the reconstructed
    values match the distribution the champion model was trained on. In particular the
    two ratios use a ``(denominator + 1)`` guard (not divide-then-drop-zeros) and the
    verification map fills unmapped values with 0 — both matching the training agent.
    """
    derived_log = []

    # int_rate_clean = numeric int_rate with '%' stripped  (agent._loan_features)
    if 'int_rate_clean' in expected_features and 'int_rate_clean' not in df.columns:
        if 'int_rate' in df.columns:
            df['int_rate_clean'] = pd.to_numeric(
                df['int_rate'].astype(str).str.replace('%', '').str.strip(), errors='coerce'
            )
            derived_log.append({'feature': 'int_rate_clean', 'derived_from': 'int_rate', 'status': 'Reconstructed'})
        else:
            derived_log.append({'feature': 'int_rate_clean', 'derived_from': 'int_rate (NOT FOUND)', 'status': 'Zero-imputed'})

    # loan_to_income = loan_amnt / (annual_inc + 1)   (agent formula: +1 denominator)
    if 'loan_to_income' in expected_features and 'loan_to_income' not in df.columns:
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            loan_amnt = pd.to_numeric(df['loan_amnt'], errors='coerce')
            annual_inc = pd.to_numeric(df['annual_inc'], errors='coerce')
            df['loan_to_income'] = loan_amnt / (annual_inc + 1)
            derived_log.append({'feature': 'loan_to_income', 'derived_from': 'loan_amnt / (annual_inc + 1)', 'status': 'Reconstructed'})
        else:
            derived_log.append({'feature': 'loan_to_income', 'derived_from': 'loan_amnt or annual_inc (NOT FOUND)', 'status': 'Zero-imputed'})

    # open_acc_ratio = open_acc / (total_acc + 1)   (agent formula: +1 denominator)
    if 'open_acc_ratio' in expected_features and 'open_acc_ratio' not in df.columns:
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            open_acc = pd.to_numeric(df['open_acc'], errors='coerce')
            total_acc = pd.to_numeric(df['total_acc'], errors='coerce')
            df['open_acc_ratio'] = open_acc / (total_acc + 1)
            derived_log.append({'feature': 'open_acc_ratio', 'derived_from': 'open_acc / (total_acc + 1)', 'status': 'Reconstructed'})
        else:
            derived_log.append({'feature': 'open_acc_ratio', 'derived_from': 'open_acc or total_acc (NOT FOUND)', 'status': 'Zero-imputed'})

    # verification_ordinal from verification_status  (agent map, .fillna(0))
    if 'verification_ordinal' in expected_features and 'verification_ordinal' not in df.columns:
        if 'verification_status' in df.columns:
            vs_map = {'not verified': 0, 'source verified': 1, 'verified': 2}
            df['verification_ordinal'] = (
                df['verification_status'].astype(str).str.lower().str.strip().map(vs_map).fillna(0)
            )
            derived_log.append({'feature': 'verification_ordinal', 'derived_from': 'verification_status', 'status': 'Reconstructed'})
        else:
            derived_log.append({'feature': 'verification_ordinal', 'derived_from': 'verification_status (NOT FOUND)', 'status': 'Zero-imputed'})

    return df, derived_log


def _alignment_fields(phase1_result):
    """Standard alignment flags propagated into every phase result."""
    return {
        'alignment_verified': phase1_result.get('alignment_verified', True),
        'alignment_warning': phase1_result.get('alignment_warning', ''),
    }


# ── Phase 1 — Data Understanding ────────────────────────────────────────────

def phase1_data_understanding(dataset_path, meta):
    """Profile Dataset 2 — schema, semantic types, business meanings. No target handling."""
    result = {'phase': 'Data Understanding', 'dataset': os.path.basename(dataset_path)}

    # Header detection — same logic as the main pipeline.
    with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    fields = first_line.strip().split(',')
    numeric_count = sum(1 for fld in fields if fld.strip().replace('.', '').replace('-', '').lstrip('-').isdigit())
    has_header = numeric_count < len(fields) * 0.5

    # alignment_verified=True only when columns are matched by real names (headered file).
    # Headerless files fall back to positional assignment, which can silently produce
    # garbage when the file's column order differs from Dataset 1 — so we flag them.
    HEADERLESS_WARNING = (
        'This file appears to have no column headers. Column order cannot be reliably '
        'verified — for accurate scoring, please ensure this file has the same column '
        'names as Dataset 1, or add a header row before uploading.'
    )
    if has_header:
        df = pd.read_csv(dataset_path, low_memory=False)
        result['header_detected'] = True
        result['alignment_verified'] = True
        result['alignment_warning'] = ''
    else:
        # Headerless: best-effort positional assignment, but flagged UNVERIFIED throughout.
        result['header_detected'] = False
        result['alignment_verified'] = False
        result['alignment_warning'] = HEADERLESS_WARNING
        col_order_path = 'outputs/models/dataset1_column_order.json'
        if not os.path.exists(col_order_path):
            col_order_path = 'sample_results/dataset1_column_order.json'
        df = pd.read_csv(dataset_path, header=None, low_memory=False)
        if os.path.exists(col_order_path):
            with open(col_order_path) as f:
                dataset1_cols = json.load(f)
            if len(dataset1_cols) == len(df.columns):
                df.columns = dataset1_cols
                result['columns_assigned_from'] = 'Dataset 1 reference order (UNVERIFIED — positional best-effort)'
            else:
                df.columns = [f'col_{i}' for i in range(len(df.columns))]
                result['columns_assigned_from'] = 'auto-generated (count mismatch with Dataset 1)'
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
            result['columns_assigned_from'] = 'auto-generated (no reference found)'

    result['total_rows'] = len(df)
    result['total_columns'] = df.shape[1]

    # Business meanings: Dataset 2's own dict -> Dataset 1's dict as GLOBAL fallback -> name inference
    data_dict = {}
    own_dict_files = glob.glob('data/*dataset2*dict*.xlsx') + glob.glob('data/*2*dictionary*.xlsx')
    dict_source = None
    if own_dict_files:
        try:
            dd = pd.read_excel(own_dict_files[0])
            cols = dd.columns.tolist()
            data_dict = dict(zip(dd[cols[0]].astype(str).str.lower().str.strip(), dd[cols[1]].astype(str)))
            dict_source = f'Dataset 2 own dictionary ({own_dict_files[0]})'
        except Exception:
            pass
    if not data_dict:
        global_dict_files = glob.glob('data/*ictionary*.xlsx')
        if global_dict_files:
            try:
                dd = pd.read_excel(global_dict_files[0])
                cols = dd.columns.tolist()
                data_dict = dict(zip(dd[cols[0]].astype(str).str.lower().str.strip(), dd[cols[1]].astype(str)))
                dict_source = f'Dataset 1 dictionary used as global fallback ({global_dict_files[0]})'
            except Exception:
                pass
    result['dictionary_source'] = dict_source or 'None found — using name inference'

    # Semantic type inference per column (numeric / categorical / date / identifier)
    semantic_types = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors='coerce').notna().mean() > 0.8:
            n_unique = s.nunique()
            sem_type = 'identifier' if n_unique / max(len(s), 1) > 0.95 else 'numeric'
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pd.to_datetime(s.dropna().head(50), errors='raise')
                sem_type = 'date'
            except Exception:
                sem_type = 'categorical'
        semantic_types.append(sem_type)

    # Schema profile with semantic types
    schema = []
    for col, sem_type in zip(df.columns, semantic_types):
        schema.append({
            'column': col, 'semantic_type': sem_type,
            'dtype': str(df[col].dtype),
            'missing_pct': round(df[col].isna().mean() * 100, 2),
            'n_unique': int(df[col].nunique()),
            'business_meaning': data_dict.get(col.lower(), '—'),
            'is_expected_feature': col in meta['selected_features'],
        })
    result['schema'] = schema

    # Selection bias reminder (informational only for Dataset 2)
    result['scope_note'] = (
        'This dataset is being scored using a model trained on funded/approved loans. '
        'If Dataset 2 includes declined applicants, predictions for that subset carry '
        'additional uncertainty (selection bias) not captured during training.'
    )

    result['df'] = df  # kept in memory for next phase, not serialized
    return result


# ── Phase 2 — Data Quality Review (scoped) ──────────────────────────────────

def phase2_data_quality_scoped(df, meta, phase1_result):
    """DQR scoped ONLY to the features the champion model needs — assessed on the raw
    incoming data (engineered features are reconstructed later, in Phase 3)."""
    expected_features = meta['selected_features']
    result = {'phase': 'Data Quality Review (Scoped)', 'expected_features': expected_features}
    result.update(_alignment_fields(phase1_result))

    available = [f for f in expected_features if f in df.columns]
    missing = [f for f in expected_features if f not in df.columns]
    result['features_found'] = available
    result['features_missing'] = missing
    result['mapping_method'] = 'Exact column name match'

    # Missing value + cardinality + summary stats per needed feature
    quality_rows = []
    for feat in expected_features:
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors='coerce')
            quality_rows.append({
                'feature': feat, 'status': 'Found',
                'missing_pct': round(s.isna().mean() * 100, 2),
                'n_unique': int(df[feat].nunique()),
                'mean': round(float(s.mean()), 4) if s.notna().any() else None,
                'std': round(float(s.std()), 4) if s.notna().any() else None,
                'min': round(float(s.min()), 4) if s.notna().any() else None,
                'max': round(float(s.max()), 4) if s.notna().any() else None,
            })
        else:
            quality_rows.append({'feature': feat, 'status': 'MISSING', 'missing_pct': 100.0,
                                 'n_unique': None, 'mean': None, 'std': None, 'min': None, 'max': None})
    result['quality_table'] = quality_rows

    # Outlier detection (IQR × 3) on needed numeric features that are present
    outlier_rows = []
    for feat in available:
        s = pd.to_numeric(df[feat], errors='coerce').dropna()
        if len(s) < 30:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_outliers = int(((s < lower) | (s > upper)).sum())
        outlier_rows.append({'feature': feat, 'iqr_outliers': n_outliers,
                             'pct': round(n_outliers / len(s) * 100, 2)})
    result['outlier_table'] = outlier_rows

    # Duplicate check if an ID-like column exists
    id_candidates = [c for c in df.columns if c.lower() in ('id', 'member_id', 'record_no', 'loan_id')]
    dup_info = {}
    for idc in id_candidates:
        dup_info[idc] = int(df[idc].duplicated().sum())
    result['duplicate_check'] = dup_info or {'note': 'No obvious ID column found to check duplicates'}

    return result


# ── Phase 3 — Feature Reconstruction ────────────────────────────────────────

def phase3_feature_reconstruction(df, meta, phase1_result):
    """Derive the engineered features the champion needs from raw columns (Phase 2 assessed
    raw availability; this phase builds the missing engineered ones with training-exact math)."""
    result = {'phase': 'Feature Reconstruction'}
    result.update(_alignment_fields(phase1_result))
    df, derived_log = reconstruct_engineered_features(df, meta['selected_features'])
    result['derived_features_log'] = derived_log
    reconstructed = [d['feature'] for d in derived_log if d['status'] == 'Reconstructed']
    result['n_reconstructed'] = len(reconstructed)
    result['reconstructed_features'] = reconstructed
    now_present = [f for f in meta['selected_features'] if f in df.columns]
    result['features_available_after'] = now_present
    result['n_available_after'] = len(now_present)
    result['df'] = df
    return result


# ── Phase 4 — Feature Stability (PSI vs training) ───────────────────────────

def phase4_feature_stability(df, meta, phase1_result):
    """Population Stability Index (PSI) per champion feature — Dataset 2 vs the training
    reference distribution. Mirrors the Dataset 1 validation agent's PSI concept, applied
    to feature distributions to detect covariate/population drift. Target is never used.

    PSI is computed against the training percentile edges saved in reference_stats.json
    (min/p25/p50/p75/max → 4 equal-mass training bins), so PSI = Σ(actual−0.25)·ln(actual/0.25)."""
    result = {'phase': 'Feature Stability'}
    result.update(_alignment_fields(phase1_result))
    ref = load_reference_stats(meta)
    # Engineered champion features map back to the raw column whose reference stats apply.
    ref_key_for = {'int_rate_clean': 'int_rate'}

    stability_rows = []
    for feat in meta['selected_features']:
        ref_key = feat if feat in ref else ref_key_for.get(feat)
        present = feat in df.columns
        d2_mean = None
        if present:
            s_all = pd.to_numeric(df[feat], errors='coerce').dropna()
            if len(s_all):
                d2_mean = round(float(s_all.mean()), 4)
        d1_mean = round(float(ref[ref_key]['mean']), 4) if (ref_key and ref_key in ref) else None
        row = {'feature': feat, 'ref_column': ref_key or '—',
               'd1_mean': d1_mean, 'd2_mean': d2_mean, 'mean_shift_pct': None,
               'psi': None, 'stability_flag': 'NO REFERENCE',
               'assessment': 'No training reference available'}
        if d1_mean is not None and d2_mean is not None and d1_mean != 0:
            row['mean_shift_pct'] = round((d2_mean - d1_mean) / abs(d1_mean) * 100, 2)

        if not ref_key or ref_key not in ref or not present:
            stability_rows.append(row)
            continue
        rs = ref[ref_key]
        edges = sorted(set([rs['min'], rs['p25'], rs['p50'], rs['p75'], rs['max']]))
        if len(edges) < 3:
            row['assessment'] = 'Degenerate reference (near-constant)'
            stability_rows.append(row)
            continue
        s = pd.to_numeric(df[feat], errors='coerce').dropna()
        if len(s) < 30:
            row['assessment'] = 'Too few values to assess'
            stability_rows.append(row)
            continue
        # Inner percentile edges with open ends so out-of-range D2 values still count.
        inner = edges[1:-1]
        bins = [-np.inf] + inner + [np.inf]
        n_bins = len(bins) - 1
        expected = np.full(n_bins, 1.0 / n_bins)          # equal-mass training bins by construction
        actual = np.histogram(s, bins=bins)[0] / len(s)
        expected = np.where(expected == 0, 1e-4, expected)
        actual = np.where(actual == 0, 1e-4, actual)
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        row['psi'] = round(psi, 4)
        if psi < 0.10:
            row['stability_flag'], row['assessment'] = 'STABLE', 'Stable (<0.10)'
        elif psi < 0.25:
            row['stability_flag'], row['assessment'] = 'SHIFTED', 'Moderate shift (0.10–0.25)'
        else:
            row['stability_flag'], row['assessment'] = 'HIGH DRIFT', 'Unstable (>0.25)'
        stability_rows.append(row)
    result['stability_table'] = stability_rows

    assessed = [r for r in stability_rows if r['psi'] is not None]
    if assessed:
        avg = float(np.mean([r['psi'] for r in assessed]))
        result['avg_psi'] = round(avg, 4)
        result['overall_assessment'] = ('Stable' if avg < 0.10
                                        else 'Moderate drift' if avg < 0.25
                                        else 'Unstable — significant population drift')
        result['n_assessed'] = len(assessed)
    else:
        result['avg_psi'] = None
        result['overall_assessment'] = 'No features had training reference stats to assess'
        result['n_assessed'] = 0
    result['psi_note'] = ('Approximate PSI using saved training percentile edges (min/p25/p50/p75/max). '
                          'Engineered ratio features without a raw reference are not assessed.')
    return result


# ── Phase 5 — Explainability (SHAP) ─────────────────────────────────────────

def phase5_explainability(df, meta, phase1_result, sample_size=1000):
    """Global feature importance via SHAP on a Dataset 2 sample using the champion model.
    Mirrors the Dataset 1 explainability agent. Target is never used. Falls back to the
    model's native importances / coefficients if SHAP is unavailable for the model type."""
    result = {'phase': 'Explainability'}
    result.update(_alignment_fields(phase1_result))

    X, _ = build_scoring_matrix(df, meta)
    model = load_champion_model(meta)
    n = min(sample_size, len(X))
    X_sample = X.sample(n=n, random_state=42) if len(X) > n else X

    method = 'unavailable'
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        if isinstance(sv, list):              # some versions return [class0, class1]
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                      # (n, features, classes)
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        method = 'SHAP (TreeExplainer)'
    except Exception as e:
        result['shap_note'] = f'SHAP unavailable ({type(e).__name__}); used model importances'
        if hasattr(model, 'feature_importances_'):
            mean_abs = np.asarray(model.feature_importances_, dtype=float)
            method = 'Model feature_importances_'
        elif hasattr(model, 'coef_'):
            mean_abs = np.abs(np.ravel(model.coef_)).astype(float)
            method = 'Logistic |coef|'
        else:
            mean_abs = np.zeros(X_sample.shape[1])

    feats = list(X_sample.columns)
    total = float(np.sum(mean_abs)) or 1.0
    importance = [{'feature': f, 'mean_abs_shap': round(float(v), 6),
                   'importance_pct': round(100 * float(v) / total, 2)}
                  for f, v in sorted(zip(feats, mean_abs), key=lambda t: t[1], reverse=True)]
    result['method'] = method
    result['sample_size'] = n
    result['feature_importance'] = importance
    result['top_features'] = [r['feature'] for r in importance[:5]]
    return result


# ── Phase 6 — Prediction ────────────────────────────────────────────────────

def phase6_prediction(df, meta, phase1_result=None):
    """Score every row using the fixed champion model + the training imputation map."""
    result = {'phase': 'Prediction'}
    if phase1_result is not None:
        result.update(_alignment_fields(phase1_result))

    X, imputation_applied = build_scoring_matrix(df, meta)
    result['imputation_applied'] = imputation_applied

    model = load_champion_model(meta)
    y_prob = model.predict_proba(X)[:, 1]

    result['n_scored'] = len(y_prob)
    result['champion_model'] = meta['champion_model']
    result['predictions'] = y_prob.tolist()
    result['score_mean'] = round(float(y_prob.mean()), 4)
    result['score_median'] = round(float(np.median(y_prob)), 4)
    result['score_min'] = round(float(y_prob.min()), 4)
    result['score_max'] = round(float(y_prob.max()), 4)
    result['pct_high_risk'] = round(float((y_prob >= 0.5).mean() * 100), 2)

    bands = pd.cut(y_prob, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    result['risk_band_distribution'] = bands.value_counts().to_dict()

    result['y_prob'] = y_prob
    return result


# ── Phase 7 — Validation (conditional) ──────────────────────────────────────

def phase7_validation(df, y_prob, meta):
    """Validate ONLY if a target-like column exists. Never assumed."""
    result = {'phase': 'Validation', 'target_available': False}

    target_col_name = meta.get('target_column', 'loan_status')
    target_mapping = meta.get('target_mapping', {})

    # Try to find a target column — check exact name, then case-insensitive
    found_col = None
    if target_col_name in df.columns:
        found_col = target_col_name
    else:
        col_lower = {c.lower(): c for c in df.columns}
        if target_col_name.lower() in col_lower:
            found_col = col_lower[target_col_name.lower()]

    if found_col is None:
        result['message'] = 'No target column found in Dataset 2 — this is a true blind scoring scenario. Only prediction distribution is available.'
        return result

    # Map values using saved mapping
    mapping_lower = {str(k).lower().strip(): v for k, v in target_mapping.items() if v in (0, 1)}
    y_true_raw = df[found_col].astype(str).str.lower().str.strip().map(mapping_lower)
    valid_mask = y_true_raw.notna()

    if valid_mask.sum() < 10:
        result['message'] = f'Target column "{found_col}" found but fewer than 10 valid mapped values — insufficient for validation metrics.'
        return result

    y_true = y_true_raw[valid_mask].astype(int).values
    y_pred_prob = y_prob[valid_mask.values]

    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss

    auc = roc_auc_score(y_true, y_pred_prob)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_pred_prob)

    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    result['target_available'] = True
    result['target_column_used'] = found_col
    result['n_valid'] = int(valid_mask.sum())
    result['auc'] = round(auc, 4)
    result['gini'] = round(gini, 4)
    result['ks'] = round(ks, 4)
    result['brier_score'] = round(brier, 4)
    result['confusion_matrix'] = {
        'true_positive': int(tp), 'false_positive': int(fp),
        'true_negative': int(tn), 'false_negative': int(fn),
        'precision': round(precision, 4), 'recall': round(recall, 4), 'f1_score': round(f1, 4),
    }
    result['default_rate'] = round(float(y_true.mean()), 4)

    return result


# ── Orchestration ───────────────────────────────────────────────────────────

def run_dataset2_pipeline(dataset_path, run_id=None):
    """Orchestrates all 7 phases and saves combined results."""
    meta = load_champion_meta(run_id)

    p1 = phase1_data_understanding(dataset_path, meta)
    df = p1.pop('df')

    p2 = phase2_data_quality_scoped(df, meta, p1)                 # read-only (raw data)
    p3 = phase3_feature_reconstruction(df, meta, p1)
    df = p3.pop('df')                                            # now has engineered features

    p4 = phase4_feature_stability(df, meta, p1)
    p5 = phase5_explainability(df, meta, p1)
    p6 = phase6_prediction(df, meta, p1)
    y_prob = p6.pop('y_prob')
    p7 = phase7_validation(df, y_prob, meta)

    # Propagate alignment confidence for downstream consumers (UI).
    alignment_verified = p1.get('alignment_verified', True)
    scoring_confidence = ('VERIFIED' if alignment_verified
                          else 'UNVERIFIED — column alignment could not be confirmed')
    p6['prediction_status'] = scoring_confidence
    p7['scoring_confidence'] = scoring_confidence

    combined = {
        'dataset2_run_id': f"D2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'source_dataset1_run': meta.get('run_id'),
        'champion_model': meta['champion_model'],
        'alignment_verified': alignment_verified,
        'alignment_warning': p1.get('alignment_warning', ''),
        'scoring_confidence': scoring_confidence,
        'phase1_data_understanding': p1,
        'phase2_data_quality': p2,
        'phase3_feature_reconstruction': p3,
        'phase4_feature_stability': p4,
        'phase5_explainability': p5,
        'phase6_prediction': {k: v for k, v in p6.items() if k != 'predictions'},  # exclude raw array
        'phase7_validation': p7,
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs('outputs/dataset2', exist_ok=True)
    out_path = f"outputs/dataset2/{combined['dataset2_run_id']}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, default=str)

    pred_path = f"outputs/dataset2/{combined['dataset2_run_id']}_predictions.csv"
    pd.DataFrame({'predicted_probability': y_prob}).to_csv(pred_path, index=False)

    print(f"AUDIT_PATH:{out_path}")
    return combined, out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run_id', default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DATASET 2 — 7-PHASE BLIND SCORING PIPELINE")
    print(f"{'='*60}\n")

    result, path = run_dataset2_pipeline(args.dataset, args.run_id)

    if not result.get('alignment_verified', True):
        print("\n" + "!" * 60)
        print("  ⚠  UNVERIFIED SCORING — COLUMN ALIGNMENT NOT CONFIRMED")
        print("!" * 60)
        print("  " + result.get('alignment_warning', ''))
        print("  All results below are best-effort and should NOT be")
        print("  treated as confident.")
        print("!" * 60)

    p1 = result['phase1_data_understanding']
    print(f"\nPhase 1 — Data Understanding: {p1['total_rows']:,} rows, {p1['total_columns']} columns")
    print(f"  Header detected: {p1['header_detected']} | Alignment: {result.get('scoring_confidence')}")
    print(f"  Dictionary source: {p1['dictionary_source']}")
    sem = {}
    for s in p1['schema']:
        sem[s['semantic_type']] = sem.get(s['semantic_type'], 0) + 1
    print(f"  Semantic types: {sem}")

    p2 = result['phase2_data_quality']
    print(f"\nPhase 2 — Data Quality (Scoped): {len(p2['features_found'])} of "
          f"{len(p2['expected_features'])} champion features present in raw data")
    if p2['features_missing']:
        print(f"  Not in raw data (reconstructed in Phase 3): {p2['features_missing']}")
    n_out = sum(r['iqr_outliers'] for r in p2.get('outlier_table', []))
    print(f"  Outliers (IQR×3) across present features: {n_out:,} | Duplicate check: {p2['duplicate_check']}")

    p3 = result['phase3_feature_reconstruction']
    print(f"\nPhase 3 — Feature Reconstruction: {p3['n_reconstructed']} engineered features rebuilt "
          f"→ {p3['n_available_after']}/{len(p2['expected_features'])} champion features now available")
    for d in p3['derived_features_log']:
        print(f"    {d['feature']:<22} <- {d['derived_from']:<32} [{d['status']}]")

    p4 = result['phase4_feature_stability']
    print(f"\nPhase 4 — Feature Stability (PSI vs training): {p4['overall_assessment']} "
          f"(avg PSI={p4['avg_psi']}, {p4['n_assessed']} features assessed)")
    for r in p4['stability_table']:
        psi_txt = r['psi'] if r['psi'] is not None else '—'
        print(f"    {r['feature']:<22} PSI={str(psi_txt):<8} {r['assessment']}")

    p5 = result['phase5_explainability']
    print(f"\nPhase 5 — Explainability ({p5['method']}, sample={p5['sample_size']:,}):")
    for r in p5['feature_importance'][:5]:
        print(f"    {r['feature']:<22} {r['importance_pct']:>6}%  (mean|SHAP|={r['mean_abs_shap']})")

    p6 = result['phase6_prediction']
    print(f"\nPhase 6 — Prediction: {p6['n_scored']:,} scored with {p6['champion_model']} | "
          f"mean={p6['score_mean']} | median={p6['score_median']} | "
          f"min={p6['score_min']} | max={p6['score_max']} | high-risk={p6['pct_high_risk']}%")
    print(f"    Risk bands: {p6['risk_band_distribution']}")

    p7 = result['phase7_validation']
    print(f"\nPhase 7 — Validation: target_available={p7['target_available']}")
    if p7['target_available']:
        print(f"  Target column used: {p7['target_column_used']} | n_valid={p7['n_valid']:,}")
        print(f"  AUC={p7['auc']} | KS={p7['ks']} | Gini={p7['gini']} | Brier={p7['brier_score']}")
        cm = p7['confusion_matrix']
        print(f"  Precision={cm['precision']} | Recall={cm['recall']} | F1={cm['f1_score']} | "
              f"Default rate={p7['default_rate']}")
    else:
        print(f"  {p7.get('message', '')}")

    print(f"\nResults saved: {path}")
