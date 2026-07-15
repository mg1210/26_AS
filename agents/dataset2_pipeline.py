"""
dataset2_pipeline.py — Blind Dataset 2 scoring workflow
4 phases: Data Understanding → Data Quality Review (scoped) → Prediction → Validation (conditional)
Never uses target column for anything except final validation, and only if present.
"""
import pandas as pd
import numpy as np
import json
import os
import glob
import joblib
from datetime import datetime


def load_champion_meta(run_id=None):
    """Load the fixed champion model + its feature list from Dataset 1's run."""
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


def phase1_data_understanding(dataset_path, meta):
    """Profile Dataset 2 — schema, semantic types, business meanings. No target handling."""
    result = {'phase': 'Data Understanding', 'dataset': os.path.basename(dataset_path)}

    # Handle headerless CSVs — same detection logic as main pipeline
    with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    fields = first_line.strip().split(',')
    numeric_count = sum(1 for fld in fields if fld.strip().replace('.', '').replace('-', '').lstrip('-').isdigit())
    has_header = numeric_count < len(fields) * 0.5

    if has_header:
        df = pd.read_csv(dataset_path, low_memory=False)
        result['header_detected'] = True
    else:
        # Try to map from saved Dataset 1 column order
        col_order_path = 'outputs/models/dataset1_column_order.json'
        if not os.path.exists(col_order_path):
            col_order_path = 'sample_results/dataset1_column_order.json'
        df = pd.read_csv(dataset_path, header=None, low_memory=False)
        if os.path.exists(col_order_path):
            with open(col_order_path) as f:
                dataset1_cols = json.load(f)
            if len(dataset1_cols) == len(df.columns):
                df.columns = dataset1_cols
                result['header_detected'] = False
                result['columns_assigned_from'] = 'Dataset 1 reference order'
            else:
                df.columns = [f'col_{i}' for i in range(len(df.columns))]
                result['header_detected'] = False
                result['columns_assigned_from'] = 'auto-generated (count mismatch with Dataset 1)'
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
            result['header_detected'] = False
            result['columns_assigned_from'] = 'auto-generated (no reference found)'

    result['total_rows'] = len(df)
    result['total_columns'] = df.shape[1]

    # Business meanings: own data dict -> Dataset 1's dict as global fallback -> name inference
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

    # Schema profile
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isna().mean()
        n_unique = df[col].nunique()
        meaning = data_dict.get(col.lower(), '')
        schema.append({
            'column': col, 'dtype': dtype,
            'missing_pct': round(missing_pct * 100, 2),
            'n_unique': int(n_unique),
            'business_meaning': meaning or '—',
            'is_expected_feature': col in meta['selected_features'],
        })
    result['schema'] = schema
    result['df'] = df  # kept in memory for next phase, not serialized

    return result


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


def phase2_data_quality_scoped(df, meta, phase1_result):
    """DQR scoped ONLY to the features the champion model actually needs."""
    expected_features = meta['selected_features']
    # Reconstruct engineered features from raw columns BEFORE the found/missing check,
    # so derived features count as 'found' and are scored with real values (not zeros).
    df, derived_log = reconstruct_engineered_features(df, expected_features)
    result = {'phase': 'Data Quality Review (Scoped)', 'expected_features': expected_features,
              'derived_features_log': derived_log}

    # Map columns — exact name match first (handles shuffled order naturally since we match by name)
    available = [f for f in expected_features if f in df.columns]
    missing = [f for f in expected_features if f not in df.columns]

    result['features_found'] = available
    result['features_missing'] = missing
    result['mapping_method'] = 'Exact column name match'

    # DQR only on the needed features
    quality_rows = []
    for feat in expected_features:
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors='coerce')
            quality_rows.append({
                'feature': feat,
                'status': 'Found',
                'missing_pct': round(s.isna().mean() * 100, 2),
                'mean': round(float(s.mean()), 4) if s.notna().any() else None,
                'std': round(float(s.std()), 4) if s.notna().any() else None,
                'min': round(float(s.min()), 4) if s.notna().any() else None,
                'max': round(float(s.max()), 4) if s.notna().any() else None,
            })
        else:
            quality_rows.append({
                'feature': feat, 'status': 'MISSING — will be imputed with 0',
                'missing_pct': 100.0, 'mean': None, 'std': None, 'min': None, 'max': None,
            })
    result['quality_table'] = quality_rows
    result['df'] = df

    return result


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


def phase3_prediction(df, meta):
    """Score every row using the fixed champion model."""
    result = {'phase': 'Prediction'}
    expected_features = meta['selected_features']
    imputation_map = load_imputation_map(meta)

    # Index on df.index so scalar (missing→value) and Series (found) assignments both
    # get the full row length. Starting from an empty frame lets an early missing
    # feature create a 0-length column that later backfills to NaN when the frame
    # expands — which NaN-intolerant models (e.g. GradientBoosting) then reject.
    # Missing cells are filled with the EXACT training fill value (sentinel/median)
    # from the persisted imputation map, matching how the champion was trained.
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
            # Feature genuinely missing entirely — use the training fill value as the
            # best guess (not a blanket 0, which for a sentinel column would misroute).
            if feat in imputation_map:
                X[feat] = imputation_map[feat]['fill_value']
                imputation_applied.append({'feature': feat, 'fill_value': imputation_map[feat]['fill_value'],
                                           'strategy': 'entire column missing, used training fill value'})
            else:
                X[feat] = 0
                imputation_applied.append({'feature': feat, 'fill_value': 0,
                                           'strategy': 'entire column missing, no training reference'})

    result['imputation_applied'] = imputation_applied

    model_path = meta.get('champion_model_path', '')
    if not model_path or not os.path.exists(model_path):
        candidates = sorted(glob.glob(f"outputs/models/*_{meta['champion_model']}.pkl"), reverse=True)
        candidates += sorted(glob.glob(f"sample_results/models/*_{meta['champion_model']}.pkl"), reverse=True)
        if not candidates:
            raise FileNotFoundError("Champion model file not found")
        model_path = candidates[0]

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X)[:, 1]

    result['n_scored'] = len(y_prob)
    result['champion_model'] = meta['champion_model']
    result['predictions'] = y_prob.tolist()
    result['score_mean'] = round(float(y_prob.mean()), 4)
    result['score_median'] = round(float(np.median(y_prob)), 4)
    result['score_min'] = round(float(y_prob.min()), 4)
    result['score_max'] = round(float(y_prob.max()), 4)
    result['pct_high_risk'] = round(float((y_prob >= 0.5).mean() * 100), 2)

    # Risk bands
    bands = pd.cut(y_prob, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    result['risk_band_distribution'] = bands.value_counts().to_dict()

    result['df'] = df
    result['y_prob'] = y_prob

    return result


def phase4_validation(df, y_prob, meta):
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


def run_dataset2_pipeline(dataset_path, run_id=None):
    """Orchestrates all 4 phases and saves combined results."""
    meta = load_champion_meta(run_id)

    p1 = phase1_data_understanding(dataset_path, meta)
    df = p1.pop('df')

    p2 = phase2_data_quality_scoped(df, meta, p1)
    df = p2.pop('df')

    p3 = phase3_prediction(df, meta)
    df = p3.pop('df')
    y_prob = p3.pop('y_prob')

    p4 = phase4_validation(df, y_prob, meta)

    combined = {
        'dataset2_run_id': f"D2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'source_dataset1_run': meta.get('run_id'),
        'champion_model': meta['champion_model'],
        'phase1_data_understanding': p1,
        'phase2_data_quality': p2,
        'phase3_prediction': {k: v for k, v in p3.items() if k != 'predictions'},  # exclude raw array from summary
        'phase4_validation': p4,
        'timestamp': datetime.now().isoformat(),
    }

    os.makedirs('outputs/dataset2', exist_ok=True)
    out_path = f"outputs/dataset2/{combined['dataset2_run_id']}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, default=str)

    # Save predictions separately (could be large)
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
    print("DATASET 2 — 4-PHASE BLIND SCORING PIPELINE")
    print(f"{'='*60}\n")

    result, path = run_dataset2_pipeline(args.dataset, args.run_id)

    print(f"\nPhase 1 — Data Understanding: {result['phase1_data_understanding']['total_rows']:,} rows, "
          f"{result['phase1_data_understanding']['total_columns']} columns")
    print(f"  Header detected: {result['phase1_data_understanding']['header_detected']}")
    print(f"  Dictionary source: {result['phase1_data_understanding']['dictionary_source']}")

    print(f"\nPhase 2 — Data Quality (Scoped): {len(result['phase2_data_quality']['features_found'])} "
          f"of {len(result['phase2_data_quality']['expected_features'])} features found")
    if result['phase2_data_quality']['features_missing']:
        print(f"  Missing: {result['phase2_data_quality']['features_missing']}")

    print(f"\nPhase 3 — Prediction: {result['phase3_prediction']['n_scored']:,} scored | "
          f"mean={result['phase3_prediction']['score_mean']} | "
          f"high-risk={result['phase3_prediction']['pct_high_risk']}%")

    print(f"\nPhase 4 — Validation: target_available={result['phase4_validation']['target_available']}")
    if result['phase4_validation']['target_available']:
        print(f"  AUC={result['phase4_validation']['auc']} KS={result['phase4_validation']['ks']} "
              f"Gini={result['phase4_validation']['gini']}")
    else:
        print(f"  {result['phase4_validation'].get('message', '')}")

    print(f"\nResults saved: {path}")
