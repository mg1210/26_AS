"""
Credit Risk Factory — Streamlit Dashboard
On startup: loads latest outputs/*_audit_trail.json automatically.
Run Pipeline: calls orchestrator.py, then reloads from outputs/.
"""

import sys
import os
import json
import glob
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Credit Risk Factory", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.card { background:#1e1e2e; border:1px solid #313244; border-radius:10px;
        padding:1rem 1.2rem; text-align:center; }
.card-label { color:#a6adc8; font-size:0.75rem; margin-bottom:4px; }
.card-value { color:#cba6f7; font-size:1.9rem; font-weight:700; }
.card-sub   { color:#a6e3a1; font-size:0.7rem; margin-top:2px; }
.pass { background:#a6e3a1; color:#1e1e2e; border-radius:6px;
        padding:2px 12px; font-weight:700; }
.fail { background:#f38ba8; color:#1e1e2e; border-radius:6px;
        padding:2px 12px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Load latest audit trail from outputs/ ─────────────────────────────────────
def load_latest_audit():
    pattern = str(ROOT / "outputs" / "*_audit_trail.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None, None
    with open(files[0], encoding="utf-8") as f:
        return json.load(f), files[0]


def load_latest_report():
    pattern = str(ROOT / "outputs" / "*_model_report.txt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None, None
    with open(files[0], encoding="utf-8", errors="replace") as f:
        return f.read(), files[0]


def card(col, label, value, sub=""):
    if isinstance(value, float):
        v = f"{value:.4f}"
    elif value is None:
        v = "—"
    else:
        v = str(value)
    col.markdown(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{v}</div>'
        f'<div class="card-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Credit Risk Factory")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV to run pipeline", type=["csv"])
    trials   = st.slider("Optuna trials", 5, 100, 10, step=5)
    run_btn  = st.button("▶  Run Pipeline", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Results auto-load from outputs/ on every page open.")


# ── Run pipeline on button click ──────────────────────────────────────────────
if run_btn:
    if uploaded is None:
        st.sidebar.error("Please upload a CSV first.")
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.close()

        with st.spinner("Running pipeline… please wait (2–5 min)"):
            err = None
            try:
                from orchestrator import CreditRiskOrchestrator
                orch = CreditRiskOrchestrator(
                    output_dir=str(ROOT / "outputs"),
                    optuna_trials=trials,
                    auto_approve=True,
                    verbose=False,
                )
                orch.run(dataset_path=tmp.name, dataset_name=uploaded.name)
            except Exception:
                err = traceback.format_exc()
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        if err:
            st.error("Pipeline failed")
            st.code(err)
        else:
            st.success("Pipeline complete — results loaded below.")
            st.rerun()


# ── Load results from disk ────────────────────────────────────────────────────
data, audit_path   = load_latest_audit()
report_txt, rpt_path = load_latest_report()

if data is None:
    st.info("No results yet. Upload a CSV and click **Run Pipeline**.")
    st.stop()

# ── Parse JSON ────────────────────────────────────────────────────────────────
champion   = data.get("champion_model_name", "—")
mm         = data.get("model_metrics", {})
features   = data.get("selected_features", [])
passed     = data.get("validation_passed", False)
chk        = data.get("checkpoints", {})
audit_log  = data.get("audit_log", [])
warnings   = data.get("warnings", [])
errors     = data.get("errors", [])
run_id     = data.get("run_id", "—")
dataset    = data.get("dataset_name", "—")
target_def = data.get("target_definition", "—")

# Champion metrics
champ_m = mm.get(champion, {})
auc   = champ_m.get("auc_test")
gini  = champ_m.get("gini")
ks    = champ_m.get("ks")
prec  = champ_m.get("precision")
rec   = champ_m.get("recall")
f1    = champ_m.get("f1")
ofit  = champ_m.get("overfit")

# ── Header ────────────────────────────────────────────────────────────────────
_cls  = "pass" if passed else "fail"
_txt  = "PASS ✓" if passed else "FAIL ✗"
badge = f'<span class="{_cls}">{_txt}</span>'
st.markdown(f"## 🏆 Champion: **{champion}** &nbsp; {badge}", unsafe_allow_html=True)
st.caption(f"Run: {run_id}  |  Dataset: {dataset}  |  Loaded from: {os.path.basename(audit_path)}")
st.markdown("---")

# ── Pipeline phase tracker ────────────────────────────────────────────────────
PHASES = [
    ("Data Understanding", "DataUnderstandingAgent"),
    ("DQR",               "DQRAgent"),
    ("Feature Eng.",      "FeatureEngineeringAgent"),
    ("Variable Selection","VariableSelectionAgent"),
    ("Model Dev.",        "ModelDevelopmentAgent"),
    ("Explainability",    "ExplainabilityAgent"),
    ("Validation",        "ValidationAgent"),
]

completed_agents = {
    e["agent"]
    for e in audit_log
    if e.get("action") == "completed"
}

phase_cols = st.columns(7)
for col, (label, agent) in zip(phase_cols, PHASES):
    done = agent in completed_agents
    bg   = "#1a3a1a" if done else "#2a2a3a"
    border = "#a6e3a1" if done else "#45475a"
    icon = "✓" if done else "○"
    icon_color = "#a6e3a1" if done else "#6c7086"
    col.markdown(
        f"""<div style="background:{bg};border:1px solid {border};border-radius:8px;
        padding:10px 6px;text-align:center;">
        <div style="font-size:1.3rem;color:{icon_color};font-weight:700;">{icon}</div>
        <div style="font-size:0.7rem;color:#cdd6f4;margin-top:4px;line-height:1.3;">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
card(c1, "AUC",       auc,  "area under ROC")
card(c2, "Gini",      gini, "2×AUC − 1")
card(c3, "KS",        ks,   "max separation")
card(c4, "Precision", prec, "at 0.5 threshold")
card(c5, "Recall",    rec,  "at 0.5 threshold")
card(c6, "Overfit Δ", ofit, "train−test AUC gap")

st.markdown("<br>", unsafe_allow_html=True)

# ── Checkpoints ───────────────────────────────────────────────────────────────
cc1, cc2, cc3 = st.columns(3)
for col, label, key in [
    (cc1, "Checkpoint 1 — Target Definition", "target_confirmed"),
    (cc2, "Checkpoint 2 — Feature Shortlist",  "features_approved"),
    (cc3, "Checkpoint 3 — Model Sign-Off",     "model_signed_off"),
]:
    ok = chk.get(key, False)
    col.markdown(f"{'✅' if ok else '❌'} **{label}**")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Comparison",
    "🔍 Features",
    "📄 Report",
    "🔬 Audit Log",
    "⚠️ Warnings & Errors",
])

# ── Tab 1: Model comparison ───────────────────────────────────────────────────
with tab1:
    rows = []
    for name, m in mm.items():
        rows.append({
            "Model"     : name,
            "AUC"       : m.get("auc_test"),
            "Gini"      : m.get("gini"),
            "KS"        : m.get("ks"),
            "Precision" : m.get("precision"),
            "Recall"    : m.get("recall"),
            "F1"        : m.get("f1"),
            "Overfit Δ" : m.get("overfit"),
            "Champion"  : "✓" if name == champion else "",
        })

    if rows:
        df_mm = pd.DataFrame(rows)
        num_cols = ["AUC", "Gini", "KS", "Precision", "Recall", "F1", "Overfit Δ"]
        present  = [c for c in num_cols if c in df_mm.columns]
        st.dataframe(
            df_mm.style
                .highlight_max(subset=["AUC", "Gini", "KS", "F1"], color="#2d4a2d")
                .highlight_min(subset=["Overfit Δ"], color="#2d4a2d")
                .format({c: "{:.4f}" for c in present}, na_rep="—"),
            use_container_width=True,
        )

        try:
            import plotly.graph_objects as go
            names  = df_mm["Model"].tolist()
            aucs   = df_mm["AUC"].tolist()
            colors = ["#cba6f7" if n == champion else "#6c7086" for n in names]
            fig = go.Figure(go.Bar(
                x=names, y=aucs, marker_color=colors,
                text=[f"{a:.4f}" if a else "—" for a in aucs],
                textposition="outside",
            ))
            fig.update_layout(
                title="AUC by Model",
                yaxis_range=[0.5, max(a for a in aucs if a) * 1.08],
                plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
                font_color="#cdd6f4", height=300, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

    # XGBoost best params
    xgb_params = mm.get("XGBoost", {}).get("best_params")
    if xgb_params:
        with st.expander("XGBoost best hyperparameters"):
            st.json(xgb_params)

# ── Tab 2: Features ───────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"**{len(features)} features selected**")
    if features:
        st.write(", ".join(features))

    st.markdown("---")
    st.markdown("**Target definition**")
    st.info(target_def)

    lc = data.get("leakage_columns", [])
    if lc:
        st.markdown(f"**Leakage columns excluded ({len(lc)})**")
        st.write(", ".join(lc))

# ── Tab 3: Report ─────────────────────────────────────────────────────────────
with tab3:
    if report_txt:
        st.download_button(
            "⬇  Download Model Report",
            report_txt,
            file_name=os.path.basename(rpt_path),
            mime="text/plain",
        )
        st.code(report_txt, language=None)
    else:
        st.info("No model report found in outputs/.")

# ── Tab 4: Audit log ──────────────────────────────────────────────────────────
with tab4:
    if audit_log:
        df_al = pd.DataFrame(audit_log)
        st.dataframe(df_al, use_container_width=True)
    else:
        st.info("Audit log is empty.")

    with st.expander("Raw JSON"):
        st.json(data)

# ── Tab 5: Warnings & errors ──────────────────────────────────────────────────
with tab5:
    if errors:
        st.markdown("### Errors")
        for e in errors:
            st.error(f"[{e.get('agent','?')}] {e.get('message','')}")
    else:
        st.success("No errors.")

    if warnings:
        st.markdown("### Warnings")
        for w in warnings:
            # Don't alarm users about missing API key — it's expected
            msg = w.get("message", "")
            if "401" in msg or "api-key" in msg.lower():
                st.info(f"[{w.get('agent','?')}] LLM step skipped (no API key set)")
            else:
                st.warning(f"[{w.get('agent','?')}] {msg}")
    else:
        st.success("No warnings.")
