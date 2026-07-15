"""
evaluate_oot.py — Score new/blind dataset using fixed champion model
Usage: python evaluate_oot.py --dataset data/new_data.csv
       python evaluate_oot.py --dataset data/new_data.csv --run_id RUN_20260702_111517
"""
import argparse, json, glob, os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.state import PipelineState
from agents.data_understanding_agent import DataUnderstandingAgent
from agents.dqr_agent import DQRAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent

# Matches dataset2_pipeline.py — headerless files fall back to positional column
# assignment, which can silently produce garbage if the order differs from Dataset 1.
HEADERLESS_WARNING = (
    'This file appears to have no column headers. Column order cannot be reliably '
    'verified — for accurate scoring, please ensure this file has the same column '
    'names as Dataset 1, or add a header row before uploading.'
)

def load_latest_audit():
    files = sorted(glob.glob('outputs/*_audit_trail.json'), reverse=True)
    if not files:
        raise FileNotFoundError("No audit trail found — run main.py first")
    with open(files[0], encoding='utf-8') as f:
        return json.load(f), files[0]

def load_features_meta(run_id=None):
    if run_id:
        path = f'outputs/models/{run_id}_features_meta.json'
    else:
        files = sorted(glob.glob('outputs/models/*_features_meta.json'), reverse=True)
        if not files:
            raise FileNotFoundError("No features metadata found — run main.py first")
        path = files[0]
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def engineer_features_for_new_data(dataset_path, meta):
    """Strategy 3 — run blind data through the pipeline agents (Data Understanding
    → DQR → Feature Engineering) to produce the SAME engineered features used in
    training, then extract the champion's selected features. No feature-mapping
    file is needed: target detection, column classification, headerless handling
    and feature derivation are all done by the agents, so this is truly plug-and-play.
    """
    # Accept absolute or relative path — resolve to absolute so the file can live
    # anywhere on disk (UI temp upload, CLI full path, data/ folder, etc.).
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"  Dataset: {dataset_path}")

    # Headerless data → assume SAME column order as Dataset 1 and assign its names
    # BEFORE feature engineering (which is name-based) so it can derive named features.
    # Positional assignment is deterministic; statistical fingerprint matching was
    # abandoned because credit-feature distributions overlap too much to be unique.
    # By-name alignment (headered files) is confident; headerless positional assignment
    # is best-effort and flagged UNVERIFIED throughout (matches dataset2_pipeline.py).
    alignment_verified = True
    from core.data_loader import detect_headerless, smart_read_csv
    if detect_headerless(dataset_path):
        alignment_verified = False
        print("\n" + "!" * 60)
        print("  ⚠  UNVERIFIED SCORING — COLUMN ALIGNMENT NOT CONFIRMED")
        print("!" * 60)
        print("  " + HEADERLESS_WARNING)
        print("  All predictions below are best-effort and should NOT be")
        print("  treated as confident results.")
        print("!" * 60)
        df_new, _ = smart_read_csv(dataset_path)
        col_order_path = 'outputs/models/dataset1_column_order.json'
        if os.path.exists(col_order_path):
            with open(col_order_path, encoding='utf-8') as f:
                dataset1_cols = json.load(f)
            if len(dataset1_cols) == len(df_new.columns):
                df_new.columns = dataset1_cols
                print(f"Columns mapped: {len(dataset1_cols)} columns assigned from Dataset 1 order")
            else:
                print(f"WARNING: Column count mismatch — Dataset 1 has {len(dataset1_cols)} cols, "
                      f"new data has {len(df_new.columns)} cols")
                print("Cannot map columns — please check the data")
                sys.exit(1)
        else:
            print("WARNING: No Dataset 1 column order found — run main.py first")
            sys.exit(1)
        # Write renamed data to a temp CSV so the pipeline agents load the assigned names.
        import tempfile
        matched_path = os.path.join(tempfile.mkdtemp(), 'matched_' + os.path.basename(dataset_path))
        df_new.to_csv(matched_path, index=False)
        dataset_path = matched_path

    print("\nRunning feature engineering on new data...")

    # Create a minimal pipeline state
    state = PipelineState(
        dataset_path=dataset_path,
        dataset_name=os.path.basename(dataset_path),
        # Pass known leakage and ID columns to exclude
        leakage_columns=meta.get('leakage_columns', []),
    )

    # Phase 1 — Data Understanding (auto-detects target, classifies columns)
    state = DataUnderstandingAgent(verbose=True).execute(state)

    # Phase 2 — DQR (minimal — just for missing value profiles)
    state = DQRAgent(verbose=False).execute(state)

    # Phase 3 — Feature Engineering (creates same derived features)
    state = FeatureEngineeringAgent(verbose=True).execute(state)

    if state.engineered_df is None:
        raise ValueError("Feature engineering failed on new data")

    # Extract only the features the model expects
    selected_features = meta['selected_features']
    available = [f for f in selected_features if f in state.engineered_df.columns]
    missing = [f for f in selected_features if f not in state.engineered_df.columns]

    if missing:
        print(f"  WARNING: {len(missing)} features could not be engineered: {missing}")
        for f in missing:
            state.engineered_df[f] = 0

    X_new = state.engineered_df[selected_features].apply(
        pd.to_numeric, errors='coerce').fillna(0)

    print(f"  Features engineered: {len(available)} found | {len(missing)} imputed")
    print(f"  Final shape: {X_new.shape}")

    # Get target if available
    y_new = None
    if 'target' in state.engineered_df.columns:
        y_new = state.engineered_df['target']
        print(f"  Target detected: {len(y_new):,} obs | Default rate: {y_new.mean():.2%}")

    return X_new, y_new, missing, alignment_verified

def compute_psi(ref_probs, new_probs, bins=10):
    breakpoints = np.percentile(ref_probs, np.linspace(0, 100, bins+1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref_probs, bins=breakpoints)[0] / len(ref_probs)
    new_pct = np.histogram(new_probs, bins=breakpoints)[0] / len(new_probs)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    new_pct = np.where(new_pct == 0, 1e-4, new_pct)
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

def main():
    parser = argparse.ArgumentParser(description='Score new data using fixed champion model')
    parser.add_argument('--dataset', required=True, help='Path to new data CSV')
    parser.add_argument('--run_id', default=None, help='Specific run ID to use')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("NEW DATA SCORING — Credit Risk Factory")
    print(f"{'='*60}")

    # Load audit trail and features metadata
    audit, audit_path = load_latest_audit()
    meta = load_features_meta(args.run_id or audit.get('run_id'))

    print(f"Run ID         : {audit.get('run_id')}")
    print(f"Champion model : {meta['champion_model']}")
    print(f"Features       : {len(meta['selected_features'])}")

    # Engineer features from raw blind data (no mapping file needed)
    X_new, y_new, missing_features, alignment_verified = engineer_features_for_new_data(args.dataset, meta)
    scoring_confidence = ('VERIFIED' if alignment_verified
                          else 'UNVERIFIED — column alignment could not be confirmed')

    # Load champion model
    model_path = meta.get('champion_model_path', '')
    if not model_path or not os.path.exists(model_path):
        model_files = sorted(glob.glob(f"outputs/models/*_{meta['champion_model']}.pkl"), reverse=True)
        if not model_files:
            raise FileNotFoundError(f"Champion model not found. Run main.py first.")
        model_path = model_files[0]
    model = joblib.load(model_path)
    print(f"Champion model loaded: {os.path.basename(model_path)}")

    # Score
    y_prob = model.predict_proba(X_new)[:, 1]
    print(f"\nScoring complete: {len(y_prob):,} predictions")
    print(f"Alignment: {scoring_confidence}")
    print(f"Score distribution: min={y_prob.min():.4f} | mean={y_prob.mean():.4f} | max={y_prob.max():.4f}")

    # Compute metrics if target available
    new_data_metrics = {
        'dataset': os.path.basename(args.dataset),
        'scored_at': datetime.now().isoformat(),
        'total_records': len(y_prob),
        'missing_features': missing_features,
        'alignment_verified': alignment_verified,
        'scoring_confidence': scoring_confidence,
        'alignment_warning': '' if alignment_verified else HEADERLESS_WARNING,
        'score_mean': round(float(y_prob.mean()), 4),
        'score_min': round(float(y_prob.min()), 4),
        'score_max': round(float(y_prob.max()), 4),
    }

    if y_new is not None and y_new.notna().all():
        y_true = y_new.values
        auc  = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks   = float(np.max(tpr - fpr))

        new_data_metrics.update({
            'auc_new_data': round(auc, 4),
            'gini_new_data': round(gini, 4),
            'ks_new_data': round(ks, 4),
            'default_rate': round(float(y_true.mean()), 4),
            'has_metrics': True,
        })

        print(f"\nPerformance Metrics:")
        print(f"  AUC  = {auc:.4f}")
        print(f"  Gini = {gini:.4f}")
        print(f"  KS   = {ks:.4f}")
        print(f"  Default rate = {y_true.mean():.2%}")
    else:
        new_data_metrics['has_metrics'] = False
        print("\nNo target column found — scores generated but no performance metrics")

    # Save scores to CSV
    scores_path = f"outputs/new_data_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame({'predicted_default_probability': y_prob}).to_csv(scores_path, index=False)
    new_data_metrics['scores_path'] = scores_path
    print(f"\nScores saved: {scores_path}")

    # Update audit trail — store New Data metrics under *_new_data keys ONLY.
    # Do NOT overwrite auc_oot / gini_oot / ks_oot — those belong to Dataset 1 OOT.
    audit['new_data_evaluation'] = new_data_metrics
    if new_data_metrics.get('has_metrics'):
        audit['validation_metrics']['auc_new_data']  = new_data_metrics['auc_new_data']
        audit['validation_metrics']['gini_new_data'] = new_data_metrics['gini_new_data']
        audit['validation_metrics']['ks_new_data']   = new_data_metrics['ks_new_data']
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"Audit trail updated: {audit_path}")
    print(f"AUDIT_PATH:{audit_path}")

    if not alignment_verified:
        print("\n" + "!" * 60)
        print("  ⚠  RESULTS ARE UNVERIFIED — column alignment could not be confirmed.")
        print("  " + HEADERLESS_WARNING)
        print("!" * 60)

    print(f"\n{'='*60}")
    print("SCORING COMPLETE — Reload Streamlit UI to see results")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
