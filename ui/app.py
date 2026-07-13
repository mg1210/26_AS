"""
ui/app.py — Credit Risk Factory
Simple Streamlit UI using only default components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import glob
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Cloud-deployment safety: data/ and outputs/ are gitignored, so they may not
# exist on a fresh deploy. Create them up-front so nothing crashes on missing dirs.
for _d in ("data", "outputs", "outputs/models", "outputs/charts"):
    os.makedirs(_d, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# On Streamlit Cloud, live pipeline execution is disabled; we serve a pre-computed
# sample run from sample_results/ instead of outputs/.
IS_CLOUD = os.path.exists('/mount/src')

from core.data_loader import smart_read_csv  # header-aware loader shared with the pipeline

st.set_page_config(page_title="Credit Risk Factory", layout="wide")

st.markdown("""
<style>
.block-container { padding: 1rem 1.5rem 2rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; width: 260px !important; }
section[data-testid="stSidebar"] .stMarkdown p { white-space: nowrap !important; font-size: 11px !important; }
section[data-testid="stSidebar"] .stRadio label { white-space: nowrap !important; font-size: 13px !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
hr { margin: 0.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

# Original grid renderer, captured before we shadow st.dataframe usages with show_df.
_st_dataframe = st.dataframe

# Leading serial/index columns that should be centre-aligned; every other column
# is left-aligned. Matched case-insensitively on the stripped column name.
_SERIAL_COL_NAMES = {
    "#", "s.no", "s.no.", "sno", "s no", "no.", "no", "rank",
    "sr", "sr.", "sl", "sl.", "sr no", "sr. no.", "serial",
}


def _alignment_column_config(df: pd.DataFrame):
    """Build a column_config that centre-aligns a leading serial column
    (S.No / # / Rank) and left-aligns all other columns.

    st.dataframe renders on a canvas grid that ignores CSS ``text-align``
    (from Styler or global CSS), so alignment must go through column_config —
    the only mechanism the grid actually honours.
    """
    try:
        cols = list(df.columns)
    except Exception:
        return None
    if not cols:
        return None
    cfg = {}
    for i, col in enumerate(cols):
        name = str(col).strip().lower()
        align = "center" if (i == 0 and name in _SERIAL_COL_NAMES) else "left"
        cfg[col] = st.column_config.Column(alignment=align)
    return cfg


def show_df(data, **kwargs):
    """Drop-in wrapper for st.dataframe that applies consistent column
    alignment (leading S.No/# centred, all text columns left-aligned).
    Honours an explicit column_config if the caller already supplied one.
    """
    if isinstance(data, pd.DataFrame) and "column_config" not in kwargs:
        cfg = _alignment_column_config(data)
        if cfg:
            kwargs["column_config"] = cfg
    return _st_dataframe(data, **kwargs)


def resolve_dataset_path(data: dict):
    """Locate the CSV actually used for this run.

    Prefers the run's recorded dataset_name over an arbitrary glob so a blind
    Dataset 2 loads the correct file even when several CSVs sit in data/.
    """
    name = (data or {}).get("dataset_name", "") or ""
    for cand in (name, os.path.join("data", name),
                 os.path.join("data", os.path.basename(name)) if name else ""):
        if cand and os.path.exists(cand):
            return cand
    files = sorted(glob.glob("data/*.csv"), reverse=True)
    return files[0] if files else None


def load_run_raw_df(data: dict):
    """Load the run's raw dataset with header-aware parsing (headerless files
    get col_0, col_1, … names — identical to the pipeline). Cached per-file in
    session_state. Returns (df_or_None, headerless_bool).
    """
    path = resolve_dataset_path(data)
    if not path:
        return None, False
    key = f"_raw_df::{path}"
    if key not in st.session_state:
        try:
            st.session_state[key] = smart_read_csv(path)
        except Exception:
            st.session_state[key] = (None, False)
    return st.session_state[key]


def show_headerless_banner(data: dict):
    """Warn when the loaded dataset had no header row, so analysts understand
    why columns are named col_0, col_1, … instead of business names."""
    if (data or {}).get("headerless"):
        st.warning(
            "⚠ **Headerless dataset detected** — the source CSV had no header row, "
            "so columns were auto-named `col_0, col_1, …`. Target detection, business "
            "meanings, and features are all inferred from the data itself."
        )


def show_run_context(data: dict):
    """Small caption identifying which dataset / run the on-screen results belong
    to — so there is never ambiguity about which analysis is being viewed."""
    d = data or {}
    st.caption(f"Results from: {d.get('dataset_name', '—')} | Run: {d.get('run_id', '—')}")


def _rag_cell_style(val):
    """pandas Styler callback — colours a RAG cell green / amber / red."""
    v = str(val).strip().upper()
    if v == "GREEN": return "background-color:#10b98126;color:#10b981;font-weight:600"
    if v == "AMBER": return "background-color:#f59e0b26;color:#f59e0b;font-weight:600"
    if v == "RED":   return "background-color:#ef444426;color:#ef4444;font-weight:600"
    return ""


def _severity_cell_style(val):
    """pandas Styler callback — colours a Severity cell red / amber / green."""
    v = str(val).strip().upper()
    if v == "HIGH":   return "background-color:#ef444426;color:#ef4444;font-weight:600"
    if v == "MEDIUM": return "background-color:#f59e0b26;color:#f59e0b;font-weight:600"
    if v == "LOW":    return "background-color:#10b98126;color:#10b981;font-weight:600"
    return ""


def load_latest_results():
    # Cloud-safe: a missing/empty outputs/ or a corrupt audit file must never crash the page.
    # On Streamlit Cloud we serve the committed sample run instead of a live outputs/ folder.
    _src = "sample_results" if IS_CLOUD else "outputs"
    try:
        files = sorted(glob.glob(f"{_src}/*_audit_trail.json"), reverse=True)
        if not files:
            return None, ""
        with open(files[0], encoding="utf-8") as f:
            audit = json.load(f)
        report_path = files[0].replace("_audit_trail.json", "_model_report.txt")
        report_text = ""
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                report_text = f.read()
        return audit, report_text
    except Exception:
        return None, ""


def load_checkpoints():
    path = ("sample_results/checkpoints.json" if IS_CLOUD else "outputs/checkpoints.json")
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_checkpoint(key, payload):
    path = "outputs/checkpoints.json"
    existing = load_checkpoints()
    existing[key] = {**payload, "timestamp": datetime.now().isoformat()}
    os.makedirs("outputs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def no_data():
    st.info("Run the pipeline first to see results.")


def human_review(page_key: str, reject_panel_fn=None):
    """Render the Human Review approval section at the bottom of a phase page.

    reject_panel_fn: optional callable() that renders page-specific edit widgets
    and returns a dict of changes_made.  When provided, clicking Reject shows
    the inline edit panel instead of immediately recording the rejection.
    """
    rejected_key = f"rejected_{page_key}"

    # ── Inline reject edit panel (shown after analyst clicks Reject) ──────────
    if st.session_state.get(rejected_key):
        st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
        st.markdown("## Human Decision")
        st.error(
            "✗ **Phase Rejected** — make corrections in the edit panel below, "
            "then click 'Apply Changes and Approve'."
        )

        changes: dict = {}
        if reject_panel_fn:
            st.markdown("### Correction Panel")
            changes = reject_panel_fn() or {}

        notes = st.text_area(
            "Analyst notes (required)",
            placeholder="Describe what was changed and why…",
            key=f"reject_notes_{page_key}",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Apply Changes and Approve", type="primary",
                         key=f"apply_reject_{page_key}"):
                if not notes.strip():
                    st.error("Analyst notes are required when applying changes.")
                else:
                    save_checkpoint(page_key, {
                        "original_decision": "rejected",
                        "final_decision":    "modified",
                        "decision":          "approved_with_modifications",
                        "changes_made":      changes,
                        "analyst_notes":     notes,
                    })
                    st.session_state[rejected_key] = False
                    st.success("✓ Changes applied — phase approved with modifications.")
                    st.rerun()
        with c2:
            if st.button("↩ Undo — Go back to Approve", key=f"undo_reject_{page_key}"):
                st.session_state[rejected_key] = False
                st.rerun()

        next_phase = NEXT_PHASE_MAP.get(page_key, "")
        if next_phase:
            st.caption(f"Next: {next_phase}")
        return  # skip normal button row

    # ── Normal human review flow ───────────────────────────────────────────────
    cps  = load_checkpoints()
    prev = cps.get(page_key, {})
    dec  = prev.get("decision", "")
    ts   = prev.get("timestamp", "")
    prev_notes    = prev.get("analyst_notes", "")
    prev_override = prev.get("override_value", "")

    st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
    st.markdown("## Human Decision")
    st.write("Review the agent output above and record your decision before proceeding to the next phase.")

    # ── Governance status banner (always visible on page load) ─────
    if dec == "approved":
        st.success(f"✓ **Approved** — decision recorded in audit trail on {ts}")
        if prev_notes:
            st.caption(f"Analyst notes: {prev_notes}")
    elif dec == "approved_with_modifications":
        st.warning(f"⚠ **Approved with Modifications** — recorded on {ts}")
        if prev_notes:
            st.caption(f"Analyst notes: {prev_notes}")
        changes_made = prev.get("changes_made", {})
        if changes_made:
            with st.expander("Changes recorded"):
                st.json(changes_made)
    elif dec == "rejected":
        st.error(f"✗ **Rejected** — recorded on {ts}")
        if prev_notes:
            st.caption(f"Analyst notes: {prev_notes}")
    elif dec == "overridden":
        st.markdown(
            f"<div style='background:#1e1b4b;border:1px solid #6366f1;border-left:4px solid #818cf8;"
            f"border-radius:6px;padding:12px 16px;margin:8px 0'>"
            f"<span style='color:#a5b4fc;font-weight:700'>↩ OVERRIDDEN</span>"
            f"<span style='color:#c7d2fe;font-size:13px'> — recorded on {ts}</span><br>"
            f"<span style='color:#e0e7ff;font-size:13px'>Override value: <strong>{prev_override}</strong></span>"
            + (f"<br><span style='color:#a5b4fc;font-size:12px'>Notes: {prev_notes}</span>" if prev_notes else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No decision recorded yet for this phase.")

    st.write("")
    notes = st.text_area(
        "Analyst notes",
        placeholder="Add observations, concerns or modifications…",
        key=f"notes_{page_key}",
    )

    # ── 4-button row ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        approve_btn  = st.button("✓ Approve", key=f"approve_{page_key}")
    with col2:
        modify_btn   = st.button("⚠ Approve with Modifications", key=f"modify_{page_key}")
    with col3:
        reject_btn   = st.button("✗ Reject", key=f"reject_{page_key}")
    with col4:
        override_btn = st.button("↩ Override", key=f"override_{page_key}")

    # ── Override input (shown when Override was previously clicked or is pending) ──
    if override_btn or st.session_state.get(f"_override_pending_{page_key}"):
        st.session_state[f"_override_pending_{page_key}"] = True
        override_val = st.text_input(
            "Override value",
            placeholder="Enter your manual override…",
            key=f"override_val_{page_key}",
        )
        if st.button("Confirm Override", key=f"confirm_override_{page_key}"):
            if override_val.strip():
                save_checkpoint(page_key, {
                    "decision":       "overridden",
                    "override_value": override_val.strip(),
                    "analyst_notes":  notes,
                })
                st.session_state[f"_override_pending_{page_key}"] = False
                st.info("Override recorded. This value will be used as the analyst's final decision, superseding the AI recommendation.")
                if notes:
                    st.info(f"Recorded notes: {notes}")
                st.rerun()
            else:
                st.error("Enter an override value before confirming.")

    if approve_btn:
        save_checkpoint(page_key, {"decision": "approved", "analyst_notes": notes})
        st.success("✓ Approved — decision recorded in audit trail with timestamp.")
        if notes:
            st.info(f"Recorded notes: {notes}")
        st.rerun()
    elif modify_btn:
        save_checkpoint(page_key, {"decision": "approved_with_modifications", "analyst_notes": notes})
        st.warning("⚠ Conditional approval recorded. Modifications noted in audit trail — verify changes before next phase.")
        if notes:
            st.info(f"Recorded notes: {notes}")
        st.rerun()
    elif reject_btn:
        save_checkpoint(page_key, {"decision": "rejected", "analyst_notes": notes})
        st.session_state[rejected_key] = True
        st.rerun()

    next_phase = NEXT_PHASE_MAP.get(page_key, "")
    if next_phase:
        st.caption(f"Next: {next_phase}")


def recommendation_cards(recs: list):
    """Render a list of Recommendation dicts as bordered cards."""
    if not recs:
        st.info("No recommendations available — run pipeline to generate.")
        return
    for rec in recs:
        risk  = rec.get("risk", "low")
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
        conf  = rec.get("confidence", 0.0)
        conf  = min(float(conf), 1.0)
        conf_label = "🟢" if conf > 0.8 else ("🟡" if conf >= 0.5 else "🔴")
        with st.container(border=True):
            st.markdown(
                f"**{rec.get('title', '—')}** &nbsp; "
                f"{risk_icon} {risk.upper()} RISK &nbsp; "
                f"{conf_label} {conf:.0%} CONFIDENCE"
            )
            st.write(rec.get("recommendation", ""))
            conf_color = "#10b981" if conf >= 0.8 else "#f59e0b" if conf >= 0.5 else "#ef4444"
            st.markdown(f"""
<div style="background:#1e2436;border-radius:6px;height:8px;margin:8px 0;">
  <div style="background:{conf_color};width:{conf*100:.1f}%;height:8px;border-radius:6px;"></div>
</div>
<p style="font-size:12px;color:#8892a8;margin-top:2px;">Confidence: {conf:.0%}</p>
""", unsafe_allow_html=True)
            with st.expander("Rationale"):
                st.write(rec.get("rationale", ""))
            dec = rec.get("decision")
            if dec:
                dec_icon = {"approved": "✓", "modified": "⚠", "rejected": "✗", "overridden": "↩"}.get(dec, "•")
                st.caption(f"{dec_icon} Decision: {dec.upper()}"
                           + (f" — {rec['decision_notes']}" if rec.get("decision_notes") else ""))
            if rec.get("requires_human_approval") and not dec:
                st.caption("⚠ Requires human approval")


def show_decision_badge(page_key: str):
    """Show a colored status badge for the phase's recorded decision."""
    cps = load_checkpoints()
    prev = cps.get(page_key, {})
    dec  = prev.get("decision", "")
    ts   = prev.get("timestamp", "")[:10] if prev.get("timestamp") else ""
    if dec == "approved":
        st.success(f"✓ Phase decision: **APPROVED** — {ts}")
    elif dec == "approved_with_modifications":
        st.warning(f"⚠ Phase decision: **APPROVED WITH MODIFICATIONS** — {ts}")
    elif dec == "rejected":
        st.error(f"✗ Phase decision: **REJECTED** — {ts}")
    elif dec == "overridden":
        st.info(f"↩ Phase decision: **OVERRIDDEN** — {ts}")


def show_stale_target_banner():
    """Banner + Clear button when a target override is pending. Overrides are NOT
    auto-applied on load; a previous-session override (run_id mismatch with the
    loaded run) is called out so the analyst can clear it."""
    cps = load_checkpoints()
    tov = cps.get("target_override", {})
    if not (tov.get("target_col") and not tov.get("rerun_completed")):
        return
    cur_run = (st.session_state.get("results") or {}).get("run_id", "")
    stale_session = bool(tov.get("run_id") and cur_run and tov.get("run_id") != cur_run)
    if stale_session:
        msg = (f"⚠ **Target override active from a previous session** "
               f"(column `{tov['target_col']}`) — click **Clear** to reset.")
    else:
        msg = (f"⚠ **Target was overridden** (column: `{tov['target_col']}`) — results below "
               f"are from the **previous run**. Go to **Data Understanding** → "
               f"'🔄 Re-run Pipeline', or click **Clear** to discard the override.")
    _bc, _cc = st.columns([6, 1])
    _bc.warning(msg)
    if _cc.button("Clear", key="clear_target_override_banner"):
        existing = load_checkpoints()
        existing.pop("target_override", None)
        with open("outputs/checkpoints.json", "w") as f:
            json.dump(existing, f, indent=2)
        st.rerun()


NEXT_PHASE_MAP = {
    "data_understanding":  "🔍 Data Quality Review",
    "dqr":                 "⚙ Feature Engineering",
    "feature_engineering": "📈 Variable Selection",
    "variable_selection":  "🤖 Model Development",
    "model_development":   "💡 Explainability",
    "explainability":      "✅ Validation",
    "validation":          "📄 Documentation",
}


def show_ai_findings(data: dict, agent_name: str):
    """## AI Findings — summary + observations from the agent response."""
    resp = data.get("agent_responses", {}).get(agent_name, {})
    st.markdown("## AI Findings")
    if not resp:
        st.info("Agent response not available — run pipeline to generate.")
        return
    st.success(f"**Summary:** {resp.get('summary', '—')}")
    obs = resp.get("observations", [])
    if obs:
        st.caption("Observations")
        show_df(
            pd.DataFrame({"#": range(1, len(obs) + 1), "Observation": obs}),
            use_container_width=True, hide_index=True,
        )


def show_ai_recommendation(data: dict, agent_name: str, page_recs: list = None):
    """## AI Recommendation — reasoning text + structured recommendation cards."""
    st.markdown("## AI Recommendation")
    resp = data.get("agent_responses", {}).get(agent_name, {})
    if resp:
        reasoning = resp.get("reasoning", "")
        if reasoning:
            st.info(f"**Reasoning:** {reasoning}")
        recs = resp.get("recommendations", [])
        if recs:
            st.caption("Recommendations")
            show_df(
                pd.DataFrame({"#": range(1, len(recs) + 1), "Recommendation": recs}),
                use_container_width=True, hide_index=True,
            )
    if page_recs:
        recommendation_cards(page_recs)


# ── Session state ──────────────────────────────────────────────────────────────

st.session_state.setdefault("results", None)
st.session_state.setdefault("report", "")
st.session_state.setdefault("has_run", False)

# On Streamlit Cloud, auto-load the committed sample run so the demo shows results immediately.
if IS_CLOUD and not st.session_state.has_run:
    _r, _rpt = load_latest_results()
    if _r:
        st.session_state.results = _r
        st.session_state.report = _rpt
        st.session_state.has_run = True

# Threshold defaults (FIX 8) — set once so sliders render with correct values on first load
st.session_state.setdefault("val_auc_threshold",  0.70)
st.session_state.setdefault("val_ks_threshold",   0.25)
st.session_state.setdefault("val_gini_threshold", 0.40)
st.session_state.setdefault("val_psi_threshold",  0.10)
st.session_state.setdefault("vs_iv_threshold",    0.02)
st.session_state.setdefault("vs_corr_threshold",  0.85)



# ── Sidebar navigation ─────────────────────────────────────────────────────────

st.sidebar.title("Credit Risk Factory")
st.sidebar.caption("Dhurin Hackathon 2026")

page = st.sidebar.radio("Navigate", [
    "Home",
    "📊 Data Understanding",
    "🔍 Data Quality Review",
    "⚙ Feature Engineering",
    "📈 Variable Selection",
    "🤖 Model Development",
    "💡 Explainability",
    "✅ Validation",
    "📄 Documentation",
    "🔁 Audit Trail",
    "👤 HITL Matrix",
    "📋 Model Sign-Off",
])

if st.sidebar.button("Reload Latest Results"):
    r, rpt = load_latest_results()
    if r:
        st.session_state.results = r
        st.session_state.report = rpt
        st.session_state.has_run = True
        st.rerun()
    else:
        st.sidebar.warning("No results found in outputs/")

with st.sidebar.expander("Current Thresholds"):
    st.caption(
        f"AUC ≥ {st.session_state.val_auc_threshold:.2f}  \n"
        f"KS ≥ {st.session_state.val_ks_threshold:.2f}  \n"
        f"Gini ≥ {st.session_state.val_gini_threshold:.2f}  \n"
        f"PSI < {st.session_state.val_psi_threshold:.2f}  \n"
        f"IV ≥ {st.session_state.vs_iv_threshold:.3f}  \n"
        f"Corr ≤ {st.session_state.vs_corr_threshold:.2f}"
    )
    if st.button("Reset all thresholds to defaults", key="sidebar_reset_thr"):
        for k, v in [("val_auc_threshold", 0.70), ("val_ks_threshold", 0.25),
                     ("val_gini_threshold", 0.40), ("val_psi_threshold", 0.10),
                     ("vs_iv_threshold", 0.02),    ("vs_corr_threshold", 0.85)]:
            st.session_state[k] = v
        st.rerun()

data = st.session_state.results or {}


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.title("Credit Risk Factory")
    st.write("Agentic pipeline for autonomous credit risk model development.")
    st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)

    if st.session_state.has_run and data:
        st.success(f"✅ Analysis complete for **{data.get('dataset_name', '—')}**  ·  "
                   f"Run {data.get('run_id', '—')}")

    if data:
        has_new_data = data.get('new_data_evaluation') is not None
        run_id = data.get('run_id', '—')
        scored_at = data.get('new_data_evaluation', {}).get('scored_at', '')[:16] if has_new_data else ''
        if has_new_data:
            st.success(f"Run: {run_id} | New data scored: "
                       f"{data['new_data_evaluation'].get('dataset', '—')} at {scored_at}")
        else:
            st.info(f"Run: {run_id} | New data: not yet scored")

    col_run, col_results = st.columns(2)

    with col_run:
        if IS_CLOUD:
            st.info('This cloud demo displays the full documented pipeline run '
                    '(104,164 rows, XGBoost champion, AUC=0.7071). Live execution is '
                    'available by running locally — see README.')
        api_key = st.text_input("Anthropic API key", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                placeholder="sk-ant-...")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        # ── Section 1 — Dataset 1 (model development) ──────────────────
        st.subheader("Dataset 1 — Model Development")
        st.caption("Train, test and validate the champion model")
        uploaded_dev = st.file_uploader("Upload Dataset 1 (CSV)", type=["csv"], key="dataset1_upload")
        if uploaded_dev:
            if uploaded_dev.name != st.session_state.get("last_dev_file"):
                st.session_state.last_dev_file = uploaded_dev.name
                st.session_state.has_run = False
                st.session_state.results = None
                st.rerun()
            os.makedirs("data", exist_ok=True)
            dev_path = f"data/{uploaded_dev.name}"
            with open(dev_path, "wb") as f:
                f.write(uploaded_dev.getbuffer())
            st.success(f"✓ Dataset 1 ready: {uploaded_dev.name} ({uploaded_dev.size/1e6:.1f} MB)")
            st.session_state.dataset1_path = dev_path
        trials = st.slider("Optuna trials", 5, 100, 20, key="trials_dev")
        run_btn = st.button("▶ Run Pipeline",
                            disabled=IS_CLOUD or not st.session_state.get("dataset1_path"),
                            use_container_width=True)

        if run_btn and st.session_state.get("dataset1_path"):
            _run_t0 = time.time()
            st.caption("Pipeline log")
            log_area = st.empty()
            progress = st.progress(0, text="Starting pipeline…")
            cmd = [sys.executable, os.path.join(PROJECT_ROOT, "main.py"), "--auto",
                   "--trials", str(int(trials)),
                   "--dataset", st.session_state.dataset1_path]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=PROJECT_ROOT,   # run from project root so relative paths resolve (Streamlit Cloud)
                env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
            )
            log_lines = []
            phase_keywords = [
                "DataUnderstanding", "DQR", "FeatureEng", "Variable",
                "ModelDev", "Explainab", "Validation",
            ]
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-50:]) or "Starting…")
                done = sum(1 for kw in phase_keywords
                           if any(kw.lower() in ln.lower() for ln in log_lines))
                progress.progress(min(done / 7, 0.95),
                                  text=f"Phase {done}/7 running… · {time.time() - _run_t0:.0f}s elapsed")
            proc.wait()
            _run_elapsed = time.time() - _run_t0
            progress.progress(1.0, text=f"Complete in {_run_elapsed:.1f}s")

            r, rpt = load_latest_results()
            if r:
                st.session_state.results = r
                st.session_state.report = rpt
                st.session_state.has_run = True
                st.session_state.last_run_seconds = _run_elapsed
                st.success(f"Pipeline complete in {_run_elapsed:.1f}s — results loaded.")
                st.rerun()
            else:
                st.error(f"Pipeline failed (exit code {proc.returncode}) — no results produced. "
                         f"Full log (stdout + stderr):")
                st.code("\n".join(log_lines[-300:]) or "(no output captured)")

        # ── Section 2 — New Data scoring ───────────────────────────────
        st.markdown("---")
        st.subheader("New Data — Model Scoring")
        st.caption("Upload any new dataset to score using the fixed champion model. The system will "
                   "automatically map features and compute performance metrics if a target column is present.")

        model_files = sorted(glob.glob("outputs/models/*_champion_*.pkl"), reverse=True)
        if model_files:
            st.success(f"✓ Champion model available: {os.path.basename(model_files[0])}")
        else:
            st.warning("⚠ No trained model found — run Dataset 1 pipeline first")

        uploaded_new = st.file_uploader("Upload New Data (CSV)", type=["csv"], key="new_data_upload")
        if uploaded_new:
            import tempfile
            # Save to a temp file — the dataset can live anywhere, not just data/.
            # Guarded so a fresh temp dir isn't created on every Streamlit rerun.
            if st.session_state.get("_new_data_name") != uploaded_new.name:
                tmp_dir = tempfile.mkdtemp()
                new_path = os.path.join(tmp_dir, uploaded_new.name)
                with open(new_path, "wb") as f:
                    f.write(uploaded_new.getbuffer())
                st.session_state.new_data_path = new_path
                st.session_state._new_data_name = uploaded_new.name
            st.success(f"✓ New data ready: {uploaded_new.name} ({uploaded_new.size/1e6:.1f} MB)")

        score_btn = st.button("▶ Score New Data",
            disabled=IS_CLOUD or not st.session_state.get("new_data_path") or not model_files,
            use_container_width=True)

        if score_btn and st.session_state.get("new_data_path"):
            with st.spinner("Scoring new data..."):
                result = subprocess.run(
                    [sys.executable, os.path.join(PROJECT_ROOT, "evaluate_oot.py"),
                     "--dataset", st.session_state.new_data_path],
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    cwd=PROJECT_ROOT,
                    env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                )
            if result.returncode == 0:
                st.success("✓ Scoring complete!")
                st.code(result.stdout[-2000:])
                for line in result.stdout.split('\n'):
                    if line.startswith('AUDIT_PATH:'):
                        st.session_state.current_audit_path = line.replace('AUDIT_PATH:', '').strip()
                r, rpt = load_latest_results()
                if r:
                    st.session_state.results = r
                    st.session_state.report = rpt
                    st.session_state.has_run = True
                    st.rerun()
            else:
                st.error(f"Scoring failed (exit code {result.returncode})")
                st.code((result.stdout or "")[-3000:])
                st.code((result.stderr or "")[-3000:])

    with col_results:
        st.subheader("Latest Results")
        if st.session_state.has_run and data:
            mm    = data.get("model_metrics", {})
            champ = data.get("champion_model_name", "—")
            cm    = mm.get(champ, {})
            auc   = cm.get("auc_test")
            ks    = cm.get("ks")
            gini  = cm.get("gini")
            passed = data.get("validation_passed", False)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Champion", champ)
                st.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else "—")
            with r2:
                st.metric("KS",   f"{ks:.4f}"   if isinstance(ks,   float) else "—")
                st.metric("Gini", f"{gini:.4f}" if isinstance(gini, float) else "—")
            with r3:
                st.metric("Validation", "PASS ✓" if passed else "CONDITIONAL ⚠")
                st.metric("Run ID", data.get("run_id", "—"))

            # ── Execution time ────────────────────────────────────────
            resp = data.get("agent_responses", {})
            agent_times = {
                k: v.get("execution_time") for k, v in resp.items()
                if isinstance(v.get("execution_time"), (int, float))
            }
            if agent_times:
                total_s = sum(agent_times.values())
                wall = st.session_state.get("last_run_seconds")
                t1, t2 = st.columns(2)
                t1.metric("Total pipeline time", f"{total_s:.1f}s")
                t2.metric("Wall-clock (last run)",
                          f"{wall:.1f}s" if isinstance(wall, (int, float)) else "—")
                with st.expander("⏱ Time by phase (slowest first)"):
                    tdf = pd.DataFrame(
                        [{"Phase": k.replace("Agent", ""), "Seconds": round(v, 1),
                          "% of total": f"{v / total_s * 100:.0f}%" if total_s else "—"}
                         for k, v in sorted(agent_times.items(), key=lambda x: -x[1])]
                    )
                    show_df(tdf, use_container_width=True, hide_index=True, height=300)

            st.subheader("Recent Runs")
            run_files = sorted(glob.glob("outputs/*_audit_trail.json"), reverse=True)[:5]
            rows = []
            for rf in run_files:
                try:
                    with open(rf, encoding="utf-8") as f:
                        d = json.load(f)
                    ch = d.get("champion_model_name", "—")
                    cm2 = d.get("model_metrics", {}).get(ch, {})
                    rows.append({
                        "Run ID":     d.get("run_id", "—"),
                        "Dataset":    d.get("dataset_name", "—"),
                        "Champion":   ch,
                        "AUC":        cm2.get("auc_test", "—"),
                        "Validation": "PASS" if d.get("validation_passed") else "COND.",
                    })
                except Exception:
                    pass
            if rows:
                show_df(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True, height=150)
        elif st.session_state.get("dataset1_path") and os.path.exists(st.session_state.get("dataset1_path", "")):
            # A dataset is staged but not yet analysed — show no stale results.
            st.info(f"📄 New dataset uploaded — click **Run Pipeline** to analyse "
                    f"**{os.path.basename(st.session_state['dataset1_path'])}**.")
        else:
            st.info("Upload a dataset and click **Run Pipeline** to see results here.")
            st.write("**What this pipeline does:**")
            st.write("1. Data Understanding — schema profiling and leakage detection")
            st.write("2. Data Quality Review — missing values, outliers, duplicates")
            st.write("3. Feature Engineering — WOE encoding and derived features")
            st.write("4. Variable Selection — IV ranking and correlation filtering")
            st.write("5. Model Development — trains 4 models, selects champion")
            st.write("6. Explainability — SHAP values and adverse action codes")
            st.write("7. Validation — discriminatory power, stability, sign-off")

    # ── New Data scoring results (full width) ──────────────────────────
    new_eval = data.get('new_data_evaluation') if data else None
    if new_eval:
        st.markdown("---")
        st.subheader("New Data Scoring Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Dataset", new_eval.get('dataset', '—'))
        c2.metric("Records Scored", f"{new_eval.get('total_records', 0):,}")
        if new_eval.get('has_metrics'):
            c3.metric("AUC (New Data)",  new_eval.get('auc_new_data',  '—'))
            c4.metric("Gini (New Data)", new_eval.get('gini_new_data', '—'))
            c5.metric("KS (New Data)",   new_eval.get('ks_new_data',   '—'))
        else:
            c3.metric("Avg Score", new_eval.get('score_mean', '—'))
            c4.metric("Min Score", new_eval.get('score_min', '—'))
            c5.metric("Max Score", new_eval.get('score_max', '—'))
        if new_eval.get('missing_features'):
            st.warning(f"⚠ {len(new_eval['missing_features'])} features were missing in new data and "
                       f"imputed with 0: {new_eval['missing_features']}")

    # Pipeline progress tracker
    if st.session_state.has_run:
        st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
        st.markdown("**Pipeline Progress**")
        _findings = data.get("findings_register") or data.get("validation_metrics", {}).get("findings_register", [])
        _fh = sum(1 for f in _findings if f.get("Severity") == "High")
        st.metric("Findings", f"{len(_findings)} ({_fh} High)",
                  help="Auto-generated validation findings register — see the Validation page")
        agents = ["DataUnderstandingAgent", "DQRAgent", "FeatureEngineeringAgent",
                  "VariableSelectionAgent", "ModelDevelopmentAgent",
                  "ExplainabilityAgent", "ValidationAgent", "DocumentationAgent"]
        completed = sum(1 for agent in agents if any(
            e.get("agent") == agent and e.get("action") == "completed"
            for e in data.get("audit_log", [])))
        total = len(agents)
        st.progress(completed / total if total else 0,
                    text=f"Pipeline: {completed}/{total} phases complete ({completed/total*100:.0f}%)" if total else "")
        phase_names = ["Data\nUnderstanding", "Data\nQuality", "Feature\nEngineering",
                       "Variable\nSelection", "Model\nDev", "Explainability",
                       "Validation", "Documentation"]
        _pp_cols = st.columns(total)
        for _pp_col, _pp_name, _pp_agent in zip(_pp_cols, phase_names, agents):
            _pp_done = any(e.get("agent") == _pp_agent and e.get("action") == "completed"
                           for e in data.get("audit_log", []))
            with _pp_col:
                st.markdown(
                    f"<div style='text-align:center;font-size:11px;"
                    f"color:{'#10b981' if _pp_done else '#4b5563'}'>"
                    f"{'✓' if _pp_done else '○'}<br>{_pp_name}</div>",
                    unsafe_allow_html=True,
                )

elif page == "📊 Data Understanding":
    st.title("Data Understanding")
    st.write("Schema profiling, target variable definition, and leakage detection.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    if not data:
        no_data()
    else:
        show_headerless_banner(data)
        schema = data.get("schema_profile", {})

        # ── 1. Observations table ─────────────────────────────────
        _du_resp = data.get("agent_responses", {}).get("DataUnderstandingAgent", {})
        _du_obs  = _du_resp.get("observations", [])
        if _du_obs:
            show_df(
                pd.DataFrame({"#": range(1, len(_du_obs) + 1), "Observation": _du_obs}),
                use_container_width=True, hide_index=True, height=220,
            )

        # ── 2. Dataset Overview ───────────────────────────────────
        st.subheader("Dataset Overview")
        n_cols    = len(schema)
        n_numeric = len(data.get("numeric_columns", []))
        n_cat     = len(data.get("categorical_columns", []))
        n_leakage = len(data.get("leakage_columns", []))
        n_date    = len(data.get("date_columns", []))
        target_col_name = data.get("target_column", "—")
        dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
        dc1.metric("Total Columns",   n_cols)
        dc2.metric("Numeric",         n_numeric)
        dc3.metric("Categorical",     n_cat)
        dc4.metric("Date",            n_date)
        dc5.metric("Leakage Removed", n_leakage)
        dc6.metric("Target Column",   target_col_name)

        # ── 2. Target Variable Definition ─────────────────────────
        st.subheader("Target Variable Definition")
        st.write(data.get("target_definition") or "—")

        target_mapping = data.get("target_mapping", {})
        indet_values   = data.get("indeterminate_values", [])
        if target_mapping:
            indet_lookup = {v["value"]: v["count"] for v in indet_values}
            cls_label    = {1: "BAD", 0: "GOOD", None: "INDETERMINATE"}
            cls_icon     = {"BAD": "🔴", "GOOD": "🟢", "INDETERMINATE": "⚪"}
            tmap_rows    = []
            for val, lbl in target_mapping.items():
                classification = cls_label.get(lbl, "INDETERMINATE")
                count          = indet_lookup.get(val, None)
                indet_total    = sum(indet_lookup.values())
                tmap_rows.append({
                    "Value":          str(val),
                    "Classification": f"{cls_icon.get(classification,'')} {classification}",
                    "Count":          count if count is not None else "—",
                    "% of Total":     f"{count/indet_total*100:.1f}%"
                                      if count and indet_total else "—",
                    "Maps To":        "1 (BAD)" if lbl == 1 else ("0 (GOOD)" if lbl == 0 else "Excluded"),
                })
            show_df(pd.DataFrame(tmap_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Target mapping not available — run pipeline to generate.")

        if indet_values:
            with st.expander(f"Excluded indeterminate values ({len(indet_values)})"):
                show_df(
                    pd.DataFrame(indet_values).rename(columns={
                        "value": "Value", "count": "Count", "reason": "Reason"
                    }),
                    use_container_width=True, hide_index=True,
                )

        # ── 3. Schema Profile ─────────────────────────────────────
        st.subheader(f"Schema Profile ({n_cols} columns)")
        if schema:
            # Column type breakdown pie
            role_counts: dict = {}
            for _info in schema.values():
                r = _info.get("role", "other")
                role_counts[r] = role_counts.get(r, 0) + 1
            fig_pie = px.pie(
                names=list(role_counts.keys()),
                values=list(role_counts.values()),
                title="Column Role Breakdown",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textposition="inside", textinfo="label+percent")
            fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

            # Searchable schema table
            search = st.text_input("Search columns", key="schema_search")
            schema_rows = []
            for col, _info in schema.items():
                if search and search.lower() not in col.lower():
                    continue
                schema_rows.append({
                    "Column":           col,
                    "Type":             _info.get("dtype", ""),
                    "Semantic Type":    _info.get("semantic_type", "—"),
                    "Role":             _info.get("role", ""),
                    "Missing %":        f"{float(_info.get('missing_pct', 0))*100:.1f}%",
                    "Unique":           _info.get("n_unique", ""),
                    "Business Meaning": _info.get("business_meaning", "") or "",
                })
            if schema_rows:
                _n_with_meaning = sum(1 for r in schema_rows if r["Business Meaning"].strip())
                if _n_with_meaning < len(schema_rows) * 0.5:
                    st.info(
                        "Business meanings will populate after: "
                        "(1) uploading a Data Dictionary (.xlsx with Field/Description columns) "
                        "to the `data/` folder, or "
                        "(2) running the pipeline with an Anthropic API key set."
                    )
                show_df(pd.DataFrame(schema_rows), use_container_width=True,
                             hide_index=True, height=200)

            # ── Column Distribution Explorer (full dataset) ────────
            st.subheader("Column Distribution Explorer")
            numeric_cols = data.get("numeric_columns", [])
            cat_cols     = data.get("categorical_columns", [])
            explore_cols = numeric_cols + cat_cols
            if explore_cols:
                sel_exp = st.selectbox("Select column to explore",
                                       options=explore_cols, key="col_explore")
                if sel_exp and sel_exp in schema:
                    _info   = schema[sel_exp]
                    n_uniq  = _info.get("n_unique", 0)
                    miss    = _info.get("missing_pct", 0)
                    sem     = _info.get("semantic_type", "—")
                    st.caption(
                        f"Semantic type: **{sem}** · Unique values: **{n_uniq:,}** · "
                        f"Missing: **{miss*100:.1f}%**"
                    )
                    # Load the run's dataset, header-aware (matches pipeline column names)
                    _raw_df, _ = load_run_raw_df(data)

                    if _raw_df is not None and sel_exp in _raw_df.columns:
                        _col_s = _raw_df[sel_exp].dropna()
                        _is_num = pd.api.types.is_numeric_dtype(_col_s)
                        if _is_num:
                            fig_dist = px.histogram(
                                _raw_df, x=sel_exp, nbins=50,
                                title=f"Distribution of {sel_exp} "
                                      f"(full dataset — {len(_raw_df):,} rows)",
                                template="plotly_dark",
                                marginal="box",
                            )
                            fig_dist.update_layout(
                                height=280, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_dist, use_container_width=True)
                            try:
                                st.caption(
                                    f"Showing full dataset: {len(_col_s):,} non-null values · "
                                    f"Skewness: {_col_s.skew():.2f}"
                                )
                            except Exception:
                                pass
                        else:
                            _vc = _col_s.value_counts().head(20).reset_index()
                            _vc.columns = [sel_exp, "Count"]
                            fig_dist = px.bar(
                                _vc, x=sel_exp, y="Count",
                                title=f"Top 20 values — {sel_exp} "
                                      f"(full dataset — {len(_raw_df):,} rows)",
                                template="plotly_dark",
                            )
                            fig_dist.update_layout(
                                height=280, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_dist, use_container_width=True)
                            st.caption(f"Showing top 20 values · {len(_col_s):,} non-null values.")
                    else:
                        # Fallback: sample values from schema
                        samp = _info.get("sample_vals", [])
                        if samp:
                            fig_dist = px.bar(
                                x=[str(v) for v in samp], y=[1] * len(samp),
                                labels={"x": sel_exp, "y": "Count"},
                                title=f"Sample values — {sel_exp}",
                                template="plotly_dark",
                            )
                            fig_dist.update_layout(
                                height=280, margin=dict(l=0, r=0, t=40, b=0),
                                showlegend=False)
                            st.plotly_chart(fig_dist, use_container_width=True)
                            st.caption("Full dataset CSV not found — showing schema sample values only.")
        else:
            st.info("Schema profile not available in this run's output.")

        # ── 4. Leakage Column Definitions ─────────────────────────
        st.subheader("Leakage Column Definitions")
        ldef_c1, ldef_c2 = st.columns(2)
        with ldef_c1:
            st.info(
                "**Structural Leakage** — Columns that are definitionally post-origination. "
                "These fields (e.g. total_pymnt, recoveries, last_pymnt_amnt) are only "
                "populated AFTER a loan outcome is known. Including them would mean the model "
                "learns from the future. They are removed automatically before any modelling."
            )
        with ldef_c2:
            st.warning(
                "**Behavioral Leakage** — Columns detected via univariate AUC analysis. "
                "Any column with AUC ≥ 0.65 against the target is flagged as suspicious — "
                "legitimate predictors rarely exceed this threshold. These are presented for "
                "human review rather than auto-removed, since some (like int_rate) may be "
                "genuine predictors borderline-flagged."
            )

        # ── Safety net: structural / behavioral leakage lists must not overlap ──
        _structural = set(data.get("leakage_columns", []))
        _behavioral = set((data.get("behavioral_leakage_flags", {}) or {}).keys())
        _overlap = _structural & _behavioral
        if _overlap:
            st.error(f"⚠ Data integrity issue: {len(_overlap)} column(s) appear in BOTH the "
                     f"structural and behavioral leakage lists: {sorted(_overlap)}")

        # ── 5. Structural Leakage ─────────────────────────────────
        leakage = data.get("leakage_columns", [])
        st.subheader(f"Structural Leakage Columns Removed ({len(leakage)})")
        if leakage:
            show_df(
                pd.DataFrame({"Column": leakage,
                              "Reason": "Post-origination leakage — removed automatically"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success("No structural leakage columns flagged.")

        # ── 6. Behavioral Leakage ─────────────────────────────────
        beh_flags = data.get("behavioral_leakage_flags", {})
        st.subheader("Behavioral Leakage Scan")

        st.session_state.setdefault("behavioral_leakage_threshold", 0.65)
        bl_threshold = st.slider(
            "Behavioral Leakage AUC Threshold",
            min_value=0.55, max_value=0.90,
            value=st.session_state["behavioral_leakage_threshold"],
            step=0.01,
            key="behavioral_leakage_threshold",
            help=(
                "Columns with univariate AUC above this threshold are flagged as potential leakage. "
                "Lower = stricter. Raise threshold to allow borderline columns like int_rate into modelling."
            ),
        )
        _bl_border = 0.05

        if beh_flags:
            # Load prior analyst decisions
            _bl_cps      = load_checkpoints()
            _bl_prev     = _bl_cps.get("behavioral_leakage_decisions", {})
            _prev_approved = _bl_prev.get("approved_for_modelling", [])

            # Classify all flagged columns at current threshold
            flagged_cols    = sorted(
                [col for col, auc_v in beh_flags.items() if auc_v >= bl_threshold],
                key=lambda c: -beh_flags[c],
            )
            borderline_cols = [c for c in flagged_cols
                               if beh_flags[c] <= bl_threshold + _bl_border]
            clear_cols      = sorted(
                [col for col, auc_v in beh_flags.items() if auc_v < bl_threshold],
                key=lambda c: -beh_flags[c],
            )

            # Summary warning
            if flagged_cols:
                st.warning(
                    f"⚠ {len(flagged_cols)} column(s) flagged (AUC ≥ {bl_threshold:.2f}) — "
                    f"{len(borderline_cols)} BORDERLINE, "
                    f"{len(flagged_cols) - len(borderline_cols)} confirmed LEAKAGE. "
                    "Review before allowing into feature engineering."
                )

            # Build table with all scanned columns
            beh_rows = []
            for col, auc_v in sorted(beh_flags.items(), key=lambda x: -x[1]):
                if auc_v > bl_threshold + _bl_border:
                    flag = "🚫 LEAKAGE"
                elif auc_v >= bl_threshold:
                    flag = "⚠ BORDERLINE"
                else:
                    flag = "✓ CLEAR"
                analyst_dec = (
                    "Approved for modelling" if col in _prev_approved
                    else ("Confirmed leakage" if col in flagged_cols else "—")
                )
                beh_rows.append({
                    "Column":         col,
                    "Univariate AUC": f"{auc_v:.4f}",
                    "Flag":           flag,
                    "Analyst Decision": analyst_dec,
                })

            show_df(pd.DataFrame(beh_rows), use_container_width=True,
                         hide_index=True, height=220)

            # Borderline note
            _bl_borderline_examples = [
                f"{c} (AUC={beh_flags[c]:.4f})" for c in borderline_cols[:3]
            ]
            if _bl_borderline_examples:
                st.caption(
                    f"{', '.join(_bl_borderline_examples)} "
                    f"{'is' if len(_bl_borderline_examples) == 1 else 'are'} flagged as BORDERLINE — "
                    "high univariate AUC but may be legitimate credit risk predictors. "
                    "Analyst should decide whether to treat as leakage or allow into modelling."
                )

            # Analyst override multiselect
            st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
            st.caption("**Analyst Override** — approve flagged columns for modelling")
            approved_for_modelling = st.multiselect(
                "✓ Approve these flagged columns for modelling (overrides leakage flag)",
                options=flagged_cols,
                default=[c for c in _prev_approved if c in flagged_cols],
                help=(
                    "Select columns to KEEP despite high AUC — e.g. int_rate is a legitimate "
                    "predictor. Your decision overrides the AI flag."
                ),
                key="bl_approved_multiselect",
            )

            confirmed_leakage = [c for c in flagged_cols if c not in approved_for_modelling]

            # Dynamic summary
            if confirmed_leakage:
                st.success(
                    f"✓ {len(confirmed_leakage)} column(s) confirmed as leakage — "
                    "will be excluded from modelling: "
                    + ", ".join(f"`{c}`" for c in confirmed_leakage[:5])
                    + (f" + {len(confirmed_leakage)-5} more" if len(confirmed_leakage) > 5 else "")
                )
            if approved_for_modelling:
                st.warning(
                    f"⚠ {len(approved_for_modelling)} column(s) approved for modelling "
                    f"by analyst override: {approved_for_modelling}"
                )

            _bl_analyst = st.text_input("Analyst name", key="bl_analyst_name")
            if st.button("Save behavioral leakage decisions", key="bl_save"):
                if _bl_analyst.strip():
                    save_checkpoint("behavioral_leakage_decisions", {
                        "threshold_used":         bl_threshold,
                        "confirmed_leakage":      confirmed_leakage,
                        "approved_for_modelling": approved_for_modelling,
                        "analyst":                _bl_analyst.strip(),
                        "timestamp":              datetime.now().isoformat(),
                    })
                    st.success("Behavioral leakage decisions saved.")
                else:
                    st.error("Enter analyst name before saving.")
        else:
            st.success(f"No behavioral leakage detected (no column scored AUC ≥ {bl_threshold:.2f}).")

        # ── Override Target Definition ────────────────────────────
        st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
        st.subheader("Override Target Definition")
        cps = load_checkpoints()
        tov = cps.get("target_override", {})

        # Show current override + warning + re-run button
        if tov.get("target_col"):
            st.info(
                f"**Analyst Override Active** — Column: `{tov['target_col']}` · "
                f"BAD: `{tov.get('bad_val', '?')}` · GOOD: `{tov.get('good_val', '?')}`  \n"
                f"*Recorded: {tov.get('timestamp', '')}*"
            )
            if not tov.get("rerun_completed"):
                st.warning(
                    "⚠ **Target definition overridden.** All downstream results "
                    "(DQR, features, models, validation) were based on the **previous** target. "
                    "You must re-run the pipeline for changes to take effect."
                )
                ds_name = data.get("dataset_name", "") if data else ""
                dataset_path_rerun = os.path.join("data", ds_name) if ds_name else ""
                if dataset_path_rerun and os.path.exists(dataset_path_rerun):
                    if st.button("🔄 Re-run Pipeline with New Target", type="primary",
                                 key="rerun_with_target"):
                        _run_t0 = time.time()
                        st.caption("Pipeline log")
                        log_area = st.empty()
                        progress = st.progress(0, text="Starting pipeline with overridden target…")
                        cmd = [
                            sys.executable, os.path.join(PROJECT_ROOT, "main.py"), "--auto",
                            "--dataset", dataset_path_rerun,
                            "--target-col", tov["target_col"],
                        ]
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=PROJECT_ROOT,
                            env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                        )
                        lines: list = []
                        phase_keywords = [
                            "DataUnderstanding", "DQR", "FeatureEng", "Variable",
                            "ModelDev", "Explainab", "Validation",
                        ]
                        for line in proc.stdout:
                            lines.append(line.rstrip())
                            log_area.code("\n".join(lines[-50:]) or "Starting…")
                            done = sum(
                                1 for kw in phase_keywords
                                if any(kw.lower() in ln.lower() for ln in lines)
                            )
                            progress.progress(min(done / 7, 0.95),
                                              text=f"Phase {done}/7 running… · {time.time() - _run_t0:.0f}s elapsed")
                        proc.wait()
                        _run_elapsed = time.time() - _run_t0
                        progress.progress(1.0, text=f"Complete in {_run_elapsed:.1f}s")
                        existing = load_checkpoints()
                        if "target_override" in existing:
                            existing["target_override"]["rerun_completed"] = True
                            with open("outputs/checkpoints.json", "w") as f:
                                json.dump(existing, f, indent=2)
                        r, rpt = load_latest_results()
                        if r:
                            st.session_state.results = r
                            st.session_state.report  = rpt
                            st.session_state.has_run = True
                            st.session_state.last_run_seconds = _run_elapsed
                            st.success(f"Pipeline re-run complete in {_run_elapsed:.1f}s — results updated.")
                            st.rerun()
                        else:
                            st.error(f"Pipeline failed (exit code {proc.returncode}) — no results. "
                                     f"Full log (stdout + stderr):")
                            st.code("\n".join(lines[-300:]) or "(no output captured)")
                else:
                    st.error(
                        f"Dataset file `{dataset_path_rerun}` not found. "
                        "Upload the dataset on the Home page and run the pipeline first."
                    )
            else:
                st.success("✓ Pipeline has been re-run with the overridden target.")

        with st.expander("Edit override"):
            all_cols = list(schema.keys()) if schema else []
            str_cols = [
                c for c, info in schema.items()
                if info.get("semantic_type", "") in
                   ("binary_categorical", "low_cardinality_categorical")
            ] if schema else []
            col_opts = str_cols or all_cols

            sel_col = st.selectbox(
                "Select target column",
                options=col_opts,
                index=col_opts.index(tov.get("target_col")) if tov.get("target_col") in col_opts else 0,
                key="tov_col",
            )
            # Get unique sample values from schema_profile for the chosen column
            col_vals = []
            if sel_col and schema.get(sel_col):
                col_vals = schema[sel_col].get("sample_vals", [])
            # Add any previously-saved values so they remain selectable
            for v in [tov.get("bad_val"), tov.get("good_val")]:
                if v and v not in col_vals:
                    col_vals.append(v)
            col_vals = [v for v in col_vals if v]

            bad_idx  = col_vals.index(tov.get("bad_val"))  if tov.get("bad_val")  in col_vals else 0
            good_idx = col_vals.index(tov.get("good_val")) if tov.get("good_val") in col_vals else min(1, len(col_vals)-1)
            sel_bad  = st.selectbox("Select BAD value (1 = default)", options=col_vals,
                                    index=bad_idx, key="tov_bad")
            sel_good = st.selectbox("Select GOOD value (0 = non-default)", options=col_vals,
                                    index=good_idx, key="tov_good")
            just = st.text_input("Justification", key="tov_just")

            col_sv, col_clr = st.columns(2)
            with col_sv:
                if st.button("Save target override"):
                    if sel_col and sel_bad and sel_good and sel_bad != sel_good:
                        save_checkpoint("target_override", {
                            "target_col":     sel_col,
                            "bad_val":        sel_bad,
                            "good_val":       sel_good,
                            "justification":  just,
                            "original":       data.get("target_definition", ""),
                            "rerun_completed": False,
                            "run_id":         data.get("run_id", ""),
                        })
                        st.rerun()
                    else:
                        st.error("Select a column and distinct BAD / GOOD values.")
            with col_clr:
                if tov and st.button("Clear target override"):
                    existing = load_checkpoints()
                    existing.pop("target_override", None)
                    with open("outputs/checkpoints.json", "w") as f:
                        json.dump(existing, f, indent=2)
                    st.rerun()

    # ── Checkpoint 1 — Target Definition Sign-Off ──────────────────────────────
    if st.session_state.has_run and data:
        st.markdown('---')
        st.subheader('Checkpoint 1 — Target Definition Sign-Off')
        _cp1 = load_checkpoints().get('cp1', {})
        if _cp1.get('decision'):
            (st.success if _cp1['decision'] == 'approved' else st.error)(
                f"Checkpoint 1 {_cp1['decision']} — {_cp1.get('timestamp', '')}")
        cp1_notes = st.text_area('Notes', value=_cp1.get('analyst_notes', ''), key='cp1_notes')
        _cp1c1, _cp1c2 = st.columns(2)
        if _cp1c1.button('✓ Approve Target Definition', key='cp1_approve', use_container_width=True):
            save_checkpoint('cp1', {'decision': 'approved', 'analyst_notes': cp1_notes,
                                    'target_definition': data.get('target_definition', '')})
            st.success('Checkpoint 1 approved.')
            st.rerun()
        if _cp1c2.button('✗ Reject', key='cp1_reject', use_container_width=True):
            save_checkpoint('cp1', {'decision': 'rejected', 'analyst_notes': cp1_notes})
            st.error('Checkpoint 1 rejected.')
            st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY REVIEW
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Data Quality Review":
    st.title("Data Quality Review")
    st.write("Missing values, outliers, duplicates, and data quality flags.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("dqr")
    show_ai_findings(data, "DQRAgent")

    if not data:
        no_data()
    else:
        dqr = data.get("dqr_report", {})
        _dqr_tabs = st.tabs(["Missing Values", "Outliers", "Duplicates", "Flags",
                             "Distributions", "Consistency Checks"])

        with _dqr_tabs[2]:  # Duplicates
            dup = dqr.get("duplicates", {})
            if dup:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rows",         f"{dup.get('total_rows', 0):,}")
                c2.metric("Duplicate IDs",      dup.get("duplicate_ids", 0))
                c3.metric("Duplicate Members",  dup.get("duplicate_members", 0))
            else:
                st.info("Duplicate check data not available.")
            st.caption(dqr.get("llm_narrative") or "No LLM narrative — run with API key.")

        with _dqr_tabs[3]:  # Flags
            flags = data.get("dqr_flags", [])
            if flags:
                show_df(pd.DataFrame({"Flag": flags}), use_container_width=True,
                             hide_index=True, height=200)
            else:
                st.success("No data quality flags raised.")

        with _dqr_tabs[1]:  # Outliers
            outliers = dqr.get("outliers", {})
            if outliers:
                rows_out = [{"Column": k, "IQR Outliers": v.get("iqr_outliers", 0),
                             "Z-Score Outliers": v.get("zscore_outliers", 0),
                             "Min": v.get("min", ""), "Max": v.get("max", "")}
                            for k, v in outliers.items()]
                show_df(pd.DataFrame(rows_out).sort_values("IQR Outliers", ascending=False),
                             use_container_width=True, hide_index=True, height=200)
            else:
                st.info("Outlier data not available.")

        with _dqr_tabs[4]:  # Distributions (placeholder)
            st.info("Select a column in 📊 Data → Column Distribution Explorer to view full distributions.")

        with _dqr_tabs[5]:  # Consistency Checks (business-logic rules)
            cc = data.get('dqr_report', {}).get('consistency_checks', [])
            cc_summary = data.get('dqr_report', {}).get('consistency_summary', {})
            if cc:
                st.metric('Rules Applicable', cc_summary.get('total_rules', 0))
                _c1, _c2 = st.columns(2)
                _c1.metric('Passed', cc_summary.get('passed', 0))
                _c2.metric('Failed', cc_summary.get('failed', 0))
                _cc_df = pd.DataFrame(cc)
                _sev = [c for c in ['Status'] if c in _cc_df.columns]
                st.dataframe(
                    _cc_df.style.map(lambda v: 'background-color:#10b98126;color:#10b981;font-weight:600'
                                     if str(v) == 'PASS' else
                                     'background-color:#ef444426;color:#ef4444;font-weight:600'
                                     if str(v) == 'FAIL' else '', subset=_sev) if _sev else _cc_df,
                    use_container_width=True, hide_index=True,
                )
                st.caption(cc_summary.get('note', ''))
            else:
                st.info('Consistency checks not available — run pipeline to compute')

        with _dqr_tabs[0]:  # Missing Values (main tab — first shown)
            st.subheader("Missing Value Report")
        miss = data.get("missing_summary", {})
        if miss:
            miss_rows = []
            for k, v in miss.items():
                if isinstance(v, dict):
                    n   = v.get("n_missing", 0)
                    pct = v.get("pct_missing", 0)
                else:
                    n, pct = 0, float(v) if v else 0
                if n > 0:
                    miss_rows.append({"Column": k,
                                      "Missing Count": n,
                                      "Missing %": round(pct * 100, 1)})
            if miss_rows:
                miss_df = pd.DataFrame(miss_rows).sort_values("Missing %", ascending=False)
                show_df(miss_df, use_container_width=True, hide_index=True, height=220)

                # Missing-% chart — severity colours + 20%/50% threshold lines + hover count (top 20)
                top20 = miss_df.head(20).sort_values("Missing %", ascending=True)
                _mcolors = ["#ef4444" if p > 50 else "#f59e0b" if p > 20 else "#10b981"
                            for p in top20["Missing %"]]
                fig_miss = go.Figure(go.Bar(
                    x=top20["Missing %"], y=top20["Column"], orientation="h",
                    marker_color=_mcolors,
                    text=[f"{p:.1f}%" for p in top20["Missing %"]], textposition="outside",
                    customdata=top20["Missing Count"],
                    hovertemplate="%{y}<br>Missing: %{x:.1f}%<br>Count: %{customdata:,}<extra></extra>",
                ))
                fig_miss.add_vline(x=20, line_dash="dash", line_color="#f59e0b", annotation_text="20%")
                fig_miss.add_vline(x=50, line_dash="dash", line_color="#ef4444", annotation_text="50%")
                fig_miss.update_layout(title="Missing % by Column (Top 20)", template="plotly_dark",
                                       height=500, xaxis_title="Missing %",
                                       margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("No missing values found.")
        else:
            st.info("Missing value summary not available.")

        # ── FIX 1: Missing Value & Cardinality Review + Imputation HITL ──────────
        st.subheader("Missing Value & Cardinality Review")
        schema_p = data.get("schema_profile", {})
        if miss and schema_p:
            total_rows = dqr.get("duplicates", {}).get("total_rows", 0) or sum(
                v.get("n_missing", 0) + (v.get("n_total", 0) - v.get("n_missing", 0))
                if isinstance(v, dict) else 0 for v in miss.values()
            )
            # Sparse bureau fields are expected to be missing
            sparse_keywords = ["bureau", "mths_since", "revol", "inq_", "pub_rec", "tax_liens"]
            review_rows = []
            for col, v in sorted(miss.items(), key=lambda x: -(x[1].get("pct_missing", 0) if isinstance(x[1], dict) else 0)):
                n_miss = v.get("n_missing", 0) if isinstance(v, dict) else 0
                if isinstance(v, dict):
                    pct = float(v.get("pct_missing", v.get("missing_pct", 0)) or 0)
                elif isinstance(v, (int, float)):
                    pct = float(v)
                else:
                    pct = 0.0
                pct_d  = round(pct * 100, 1) if pct <= 1 else round(pct, 1)
                col_info = schema_p.get(col, {})
                dtype_s  = str(col_info.get("dtype", "unknown"))
                cardinality = col_info.get("n_unique", "—")
                is_sparse = any(kw in col.lower() for kw in sparse_keywords)
                # Single categorical classification (replaces separate Expected/Unexpected columns)
                if is_sparse:
                    classification = "Expected (sparse field)"
                elif pct_d > 5:
                    classification = "Unexpected — investigate"
                else:
                    classification = "Normal"
                strat = "Median" if dtype_s not in ("object","str") and not dtype_s.startswith("string") else "Mode"
                if is_sparse:
                    strat = "Sentinel (-999999)"
                review_rows.append({
                    "Column":                col,
                    "Dtype":                 dtype_s,
                    "Missing Count":         n_miss,
                    "Missing %":             pct_d,
                    "Cardinality":           cardinality,
                    "Missing Classification": classification,
                    "Current Strategy":      strat,
                })
            if review_rows:
                rv_df = pd.DataFrame(review_rows)

                def _cls_color(v):
                    v = str(v)
                    if v == "Normal":
                        return "background-color:#10b98126;color:#10b981;font-weight:600"
                    if v.startswith("Expected"):
                        return "background-color:#3b82f626;color:#60a5fa;font-weight:600"
                    if v.startswith("Unexpected"):
                        return "background-color:#ef444426;color:#ef4444;font-weight:600"
                    return ""
                st.dataframe(
                    rv_df.style.map(_cls_color, subset=["Missing Classification"]),
                    use_container_width=True, hide_index=True,
                )

                with st.expander("⚙ Override Imputation Strategies"):
                    st.caption("Override the default imputation strategy for any column. Saved to checkpoints.json.")
                    impute_opts = ["Median (default)", "Mean", "Mode", "Constant = 0",
                                   "Constant = -1", "Drop column", "Flag as special value -999"]
                    imp_overrides = dict(st.session_state.get("imputation_overrides", {}))
                    cols_with_missing = [r["Column"] for r in review_rows if r["Missing Count"] > 0]
                    if cols_with_missing:
                        ov_cols = st.multiselect("Columns to override", cols_with_missing,
                                                 key="imp_ov_cols")
                        for oc in ov_cols:
                            cur_strat = next((r["Current Strategy"] for r in review_rows if r["Column"] == oc), "Median (default)")
                            opt_idx   = next((i for i, o in enumerate(impute_opts) if cur_strat in o), 0)
                            chosen = st.selectbox(f"`{oc}` strategy", impute_opts,
                                                  index=opt_idx, key=f"imp_ov_{oc}")
                            imp_overrides[oc] = chosen
                        ov_analyst = st.text_input("Analyst name", key="imp_ov_analyst")
                        ov_just    = st.text_input("Justification", key="imp_ov_just")
                        if st.button("Save imputation overrides", key="imp_ov_save"):
                            if ov_analyst.strip():
                                st.session_state["imputation_overrides"] = imp_overrides
                                save_checkpoint("imputation_overrides", {
                                    "overrides":   imp_overrides,
                                    "analyst_name": ov_analyst.strip(),
                                    "justification": ov_just.strip(),
                                })
                                st.success(f"Saved {len(imp_overrides)} imputation override(s).")
                            else:
                                st.error("Enter analyst name before saving.")

                        if imp_overrides:
                            st.caption("Current overrides:")
                            show_df(
                                pd.DataFrame([{"Column": k, "Strategy": v}
                                              for k, v in imp_overrides.items()]),
                                use_container_width=True, hide_index=True,
                            )
                    else:
                        st.success("No columns with missing values — no overrides needed.")
        else:
            st.info("Missing value or schema data not available.")

        # ── Missing Value Imputation Strategy (sentinel codes) ──────────────
        st.subheader("Missing Value Imputation Strategy")
        sentinel_defs = pd.DataFrame([
            {"Imputed Value": "-999999", "Meaning": "Missing source data",
             "When Used": "Original variable has missing values (typically >20% missing observations). "
                          "Information was not available in source data."},
            {"Imputed Value": "-999997", "Meaning": "Not applicable due to filter",
             "When Used": "Engineered variable with conditional calculation (e.g. sum of balance across "
                          "active accounts) where the underlying condition does not apply (e.g. no active accounts)."},
            {"Imputed Value": "-999998", "Meaning": "Invalid ratio (division by zero)",
             "When Used": "Ratio-based features where the denominator equals zero, making the ratio "
                          "mathematically undefined."},
        ])
        show_df(sentinel_defs, use_container_width=True, hide_index=True)

        imp_log = data.get("imputation_log", [])
        if imp_log:
            st.caption("Per-column imputation applied this run:")
            show_df(pd.DataFrame(imp_log), use_container_width=True, hide_index=True, height=300)
        else:
            st.caption("Imputation log not available — re-run the pipeline to populate it.")

    show_ai_recommendation(data, "DQRAgent")

    def _dqr_reject_panel():
        st.write("Specify remediation actions for flagged columns.")
        miss = data.get("missing_summary", {})
        flagged_cols = [k for k, v in miss.items()
                        if (v.get("pct_missing", 0) if isinstance(v, dict) else float(v or 0)) > 0][:40]
        selected_cols = st.multiselect(
            "Select columns to action",
            options=flagged_cols, key="dqr_reject_cols",
        )
        col_actions: dict = {}
        for col in selected_cols:
            action = st.selectbox(
                f"Action for `{col}`", ["Impute", "Drop", "Keep"],
                key=f"dqr_action_{col}",
            )
            col_actions[col] = action
        return {"column_actions": col_actions}

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙ Feature Engineering":
    st.title("Feature Engineering")
    st.write("Derived variables, WOE encoding, imputation, and transformations.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("feature_engineering")
    show_ai_findings(data, "FeatureEngineeringAgent")

    st.markdown("## Evidence")
    if not data:
        no_data()
    else:
        dqr = data.get("dqr_report", {})

        st.subheader("Feature Engineering Summary")
        st.write(dqr.get("feature_engineering_summary") or "No summary — run with API key.")

        feature_log = data.get("feature_log", [])
        st.subheader(f"Engineered Features ({len(feature_log)})")
        if feature_log:
            show_df(pd.DataFrame(feature_log), use_container_width=True, hide_index=True)
        else:
            st.info("Feature log not available.")

    show_ai_recommendation(data, "FeatureEngineeringAgent")

    def _fe_reject_panel():
        st.write("Remove unwanted engineered features or add a custom formula.")
        feat_log = data.get("feature_log", [])
        feat_names = [f.get("feature", "") for f in feat_log if f.get("feature")]
        to_remove = st.multiselect(
            "Remove engineered features from the pipeline",
            options=feat_names, key="fe_reject_remove",
        )
        custom_formula = st.text_input(
            "Add custom feature formula (e.g. `loan_amnt / annual_inc`)",
            placeholder="Leave blank to skip", key="fe_reject_formula",
        )
        return {"features_to_remove": to_remove,
                "custom_formula": custom_formula.strip()}

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.


# ══════════════════════════════════════════════════════════════════════════════
# VARIABLE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Variable Selection":
    st.title("Variable Selection")
    st.write("Information Value, correlation analysis, and feature shortlisting.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("variable_selection")
    show_ai_findings(data, "VariableSelectionAgent")

    # ── IV Threshold Slider ───────────────────────────────────────
    with st.expander("⚙ Adjust IV Selection Threshold"):
        st.caption(
            "Features with Information Value below this threshold are rejected. "
            "Change with HITL authorisation and document justification."
        )
        iv_threshold = st.slider(
            "Minimum IV threshold for selection",
            min_value=0.01, max_value=0.10,
            value=float(st.session_state.get("vs_iv_threshold", 0.02)),
            step=0.005, format="%.3f",
            key="vs_iv_threshold",
        )
        iv_thr_name = st.text_input("Analyst name", key="vs_iv_thr_name")
        iv_thr_just = st.text_input("Justification for IV threshold change", key="vs_iv_thr_just")
        if st.button("Save IV threshold", key="vs_save_iv_thr"):
            if iv_thr_name.strip():
                save_checkpoint("iv_threshold", {
                    "iv_threshold":  st.session_state.get("vs_iv_threshold", 0.02),
                    "analyst_name":  iv_thr_name.strip(),
                    "justification": iv_thr_just.strip(),
                })
                st.success(f"IV threshold saved: {st.session_state.get('vs_iv_threshold', 0.02):.3f}")
            else:
                st.error("Enter your analyst name to save.")

    iv_threshold = float(st.session_state.get("vs_iv_threshold", 0.02))

    # FIX 3 — Correlation threshold HITL
    with st.expander("⚙ Adjust Correlation Threshold"):
        st.caption(
            "Feature pairs with absolute correlation above this threshold are flagged; "
            "the lower-IV feature is dropped. Reduce threshold to be more aggressive, "
            "increase to allow more correlated features through."
        )
        corr_threshold = st.slider(
            "Correlation threshold (r)",
            min_value=0.70, max_value=0.95,
            value=float(st.session_state.get("vs_corr_threshold", 0.85)),
            step=0.01, format="%.2f",
            key="vs_corr_threshold",
        )
        corr_name = st.text_input("Analyst name", key="vs_corr_name")
        corr_just = st.text_input("Justification for correlation threshold change", key="vs_corr_just")
        if st.button("Save correlation threshold", key="vs_save_corr"):
            if corr_name.strip():
                save_checkpoint("correlation_threshold", {
                    "corr_threshold":  corr_threshold,
                    "analyst_name":    corr_name.strip(),
                    "justification":   corr_just.strip(),
                })
                st.success(f"Correlation threshold saved: {corr_threshold:.2f}")
            else:
                st.error("Enter your analyst name to save.")

    corr_threshold = float(st.session_state.get("vs_corr_threshold", 0.85))

    st.markdown("## Evidence")
    if not data:
        no_data()
    else:
        iv_table_raw = data.get("iv_table", [])
        # Single source of truth for counts — always from pipeline output
        selected = data.get("selected_features", [])
        rejected = data.get("rejected_features", {})
        dqr      = data.get("dqr_report", {})

        _n_sel = len(selected)
        _n_rej = len(rejected)
        _n_tot = _n_sel + _n_rej

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected", _n_sel)
        c2.metric("Rejected", _n_rej)
        c3.metric("Selection Rate",
                  f"{_n_sel / _n_tot * 100:.0f}%" if _n_tot else "—")

        st.subheader("Selection Rationale")
        st.write(dqr.get("variable_selection_rationale") or "No rationale — run with API key.")

        # ── Feature Reduction Journey (FIX 8) ───────────────────────
        schema = data.get('schema_profile', {})
        total_raw = len(schema)
        structural_leakage = len(data.get('leakage_columns', []))
        id_cols   = len([c for c, v in schema.items() if v.get('role') == 'identifier'])
        date_cols = len([c for c, v in schema.items() if v.get('role') == 'date'])
        admin_cols = len([c for c, v in schema.items() if v.get('role') == 'other'])
        after_removal = total_raw - structural_leakage - id_cols - date_cols - admin_cols
        engineered_added = len(data.get('feature_log', []))
        candidates = after_removal + engineered_added
        iv_rejected = len([v for v in data.get('rejected_features', {}).values()
                           if 'IV' in v or 'Useless' in v or 'Weak' in v or 'Suspicious' in v])
        corr_rejected = len([v for v in data.get('rejected_features', {}).values()
                             if 'correlation' in v.lower()])
        final_selected = len(data.get('selected_features', []))

        journey_df = pd.DataFrame([
            {'Stage': '1. Raw Dataset', 'Columns': total_raw, 'Change': '', 'Reason': 'Original dataset columns'},
            {'Stage': '2. Remove Structural Leakage', 'Columns': total_raw - structural_leakage, 'Change': f'-{structural_leakage}', 'Reason': f'{structural_leakage} post-origination columns removed'},
            {'Stage': '3. Remove Identifiers', 'Columns': total_raw - structural_leakage - id_cols, 'Change': f'-{id_cols}', 'Reason': f'{id_cols} ID/admin columns removed (id, member_id, url etc)'},
            {'Stage': '4. Remove Date Columns', 'Columns': total_raw - structural_leakage - id_cols - date_cols, 'Change': f'-{date_cols}', 'Reason': f'{date_cols} raw date columns removed (used to create age features instead)'},
            {'Stage': '5. Feature Engineering', 'Columns': after_removal + engineered_added, 'Change': f'+{engineered_added}', 'Reason': f'{engineered_added} new features created (credit age, ratios, flags, encodings)'},
            {'Stage': '6. Numeric Candidates for IV', 'Columns': candidates, 'Change': '', 'Reason': 'Non-numeric columns excluded from IV scoring'},
            {'Stage': '7. IV Filter (remove IV < threshold)', 'Columns': candidates - iv_rejected, 'Change': f'-{iv_rejected}', 'Reason': f'{iv_rejected} features rejected: IV below threshold or suspicious (>0.50)'},
            {'Stage': '8. Correlation Filter (r > threshold)', 'Columns': candidates - iv_rejected - corr_rejected, 'Change': f'-{corr_rejected}', 'Reason': f'{corr_rejected} features removed: high correlation with stronger feature'},
            {'Stage': '9. Final Selected Features', 'Columns': final_selected, 'Change': '', 'Reason': 'Features approved for model training'},
        ])

        st.subheader("Feature Reduction Journey")
        st.caption("How features reduce from raw dataset to final model inputs — every step explained")
        show_df(journey_df, use_container_width=True, hide_index=True)

        fig_journey = go.Figure(go.Waterfall(
            name='Features', orientation='v',
            measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'absolute', 'relative', 'relative', 'absolute'],
            x=journey_df['Stage'].tolist(),
            y=[total_raw, -structural_leakage, -id_cols, -date_cols, engineered_added,
               candidates, -iv_rejected, -corr_rejected, final_selected],
            text=journey_df['Columns'].tolist(), textposition='outside',
            decreasing=dict(marker=dict(color='#ef4444')),
            increasing=dict(marker=dict(color='#10b981')),
            totals=dict(marker=dict(color='#3b82f6')),
        ))
        fig_journey.update_layout(title='Feature Reduction Waterfall', template='plotly_dark',
                                  height=400, xaxis_tickangle=-45, showlegend=False,
                                  margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_journey, use_container_width=True)

        # ── FIX 2: Feature Selection Waterfall Chart ────────────────
        st.subheader("Feature Selection Waterfall")
        if selected or rejected:
            n_total   = len(selected) + len(rejected)
            iv_rej    = sum(1 for r in rejected.values()
                            if "IV" in r and ("< threshold" in r or "too low" in r or "Suspicious" in r))
            corr_rej  = sum(1 for r in rejected.values() if "correlation" in r.lower())
            other_rej = len(rejected) - iv_rej - corr_rej

            # Schema note: schema col count approximated from iv_table total.
            # Waterfall shows the funnel via relative deltas only; the sole
            # "selected" count comes from len(selected) (data['selected_features']).
            schema_total = n_total

            wf_labels  = ["Raw candidates", "IV filter", "Correlation filter"]
            wf_measure = ["absolute",       "relative",  "relative"]
            wf_x       = [schema_total,     -iv_rej,     -corr_rej]
            wf_text    = [f"{schema_total}", f"−{iv_rej}", f"−{corr_rej}"]
            wf_colors  = ["#4f6ef7",        "#ef4444",   "#f59e0b"]

            if other_rej > 0:
                wf_labels.append("Other filters")
                wf_measure.append("relative")
                wf_x.append(-other_rej)
                wf_text.append(f"−{other_rej}")
                wf_colors.append("#f59e0b")

            wf_labels.append("Final selected")
            wf_measure.append("total")
            wf_x.append(len(selected))
            wf_text.append(f"{len(selected)}")
            wf_colors.append("#10b981")

            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=wf_measure,
                x=wf_labels,
                y=wf_x,
                text=wf_text,
                textposition="outside",
                connector={"line": {"color": "#4b5563", "width": 1, "dash": "dot"}},
                increasing={"marker": {"color": "#10b981"}},
                decreasing={"marker": {"color": "#ef4444"}},
                totals={"marker":    {"color": "#4f6ef7"}},
            ))
            fig_wf.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e7ff",
                yaxis_title="Feature count",
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            if rejected:
                with st.expander(f"Rejected features by category ({len(rejected)} total)"):
                    rej_rows = []
                    for feat, reason in rejected.items():
                        if "IV" in reason and ("< threshold" in reason or "too low" in reason or "Suspicious" in reason):
                            cat = "IV Filter"
                        elif "correlation" in reason.lower():
                            cat = "Correlation Filter"
                        else:
                            cat = "Other"
                        rej_rows.append({"Category": cat, "Feature": feat, "Reason": reason})
                    rej_df = pd.DataFrame(rej_rows).sort_values("Category")
                    for cat_name, cat_df in rej_df.groupby("Category"):
                        st.write(f"**{cat_name}** ({len(cat_df)})")
                        show_df(cat_df[["Feature", "Reason"]],
                                     use_container_width=True, hide_index=True)
        else:
            st.info("No variable selection data available.")

        # ── IV Bar Chart ──────────────────────────────────────────
        iv_table = data.get("iv_table", [])
        iv_lookup = {r.get("feature"): r for r in iv_table} if iv_table else {}
        iv_sel = [r for r in iv_table if r.get("feature") in set(selected)] if iv_table else []
        if iv_sel:
            iv_plot = pd.DataFrame(iv_sel).sort_values("iv", ascending=True)
            strength_color = {
                "Strong":     "#10b981",
                "Medium":     "#4f6ef7",
                "Weak":       "#6b7280",
                "Suspicious": "#f59e0b",
                "Useless":    "#ef4444",
            }
            fig_iv = px.bar(
                iv_plot, x="iv", y="feature", orientation="h",
                color="strength", color_discrete_map=strength_color,
                title="IV Scores — Selected Features",
                labels={"iv": "Information Value", "feature": "Feature", "strength": "Strength"},
                template="plotly_dark",
            )
            fig_iv.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_iv, use_container_width=True)

        st.subheader("Selected Features")
        if selected:
            sel_rows = []
            for i, feat in enumerate(selected):
                iv_info = iv_lookup.get(feat, {})
                sel_rows.append({
                    "Rank":     i + 1,
                    "Feature":  feat,
                    "IV":       iv_info.get("iv", "—"),
                    "Strength": iv_info.get("strength", "—"),
                })
            show_df(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True, height=200)
        else:
            st.info("No selected features.")

        st.subheader("Rejected Features")
        if rejected:
            rej_full = [
                {"Feature": k,
                 "IV":      iv_lookup.get(k, {}).get("iv", "—"),
                 "Reason":  v}
                for k, v in list(rejected.items())[:50]
            ]
            show_df(pd.DataFrame(rej_full), use_container_width=True, hide_index=True, height=200)
        else:
            st.info("No rejected features recorded.")

        st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
        st.subheader("Feature Overrides")
        cps = load_checkpoints()
        fov = cps.get("feature_overrides", {})
        added_ov   = fov.get("added", [])
        removed_ov = fov.get("removed", [])
        if added_ov or removed_ov:
            msg = []
            if added_ov:   msg.append(f"Added: {', '.join(added_ov)}")
            if removed_ov: msg.append(f"Removed: {', '.join(removed_ov)}")
            st.warning("**Feature Override Active** — " + " · ".join(msg) +
                       f"\n\n*Recorded: {fov.get('timestamp','')}*")

        col_a, col_r = st.columns(2)
        with col_a:
            st.write("**Add feature manually**")
            new_feat = st.text_input("Feature name", key="add_feat_input")
            if st.button("Add to shortlist"):
                if new_feat.strip():
                    save_checkpoint("feature_overrides", {
                        "added":   list(dict.fromkeys(added_ov + [new_feat.strip()])),
                        "removed": removed_ov,
                    })
                    st.success(f"Added '{new_feat.strip()}'.")
                    st.rerun()
                else:
                    st.error("Enter a feature name.")

        with col_r:
            st.write("**Remove features**")
            effective = [f for f in (selected + added_ov) if f not in removed_ov]
            to_remove = st.multiselect("Select features to remove", options=effective)
            if st.button("Remove selected"):
                if to_remove:
                    save_checkpoint("feature_overrides", {
                        "added":   [f for f in added_ov if f not in to_remove],
                        "removed": list(dict.fromkeys(removed_ov + to_remove)),
                    })
                    st.success(f"Removed: {', '.join(to_remove)}")
                    st.rerun()
                else:
                    st.warning("Select at least one feature.")

        if fov and st.button("Clear all feature overrides"):
            existing = load_checkpoints()
            existing.pop("feature_overrides", None)
            with open("outputs/checkpoints.json", "w") as f:
                json.dump(existing, f, indent=2)
            st.rerun()

    page_recs = [r for r in data.get("recommendations", []) if r.get("title") == "Feature Shortlist"]
    show_ai_recommendation(data, "VariableSelectionAgent", page_recs)

    def _vs_reject_panel():
        st.write("Adjust the IV threshold and force-include / force-exclude specific features.")
        iv_tbl = data.get("iv_table", [])
        all_feats_vs = [r.get("feature") for r in iv_tbl if r.get("feature")] if iv_tbl else []
        selected_vs  = data.get("selected_features", [])
        iv_thresh = st.slider(
            "IV threshold (features below this are rejected)",
            min_value=0.01, max_value=0.10, value=0.02, step=0.005,
            key="vs_reject_iv",
        )
        force_in = st.multiselect(
            "Force-include features (even if below IV threshold)",
            options=[f for f in all_feats_vs if f not in selected_vs],
            key="vs_reject_force_in",
        )
        force_out = st.multiselect(
            "Force-exclude features (remove from shortlist)",
            options=selected_vs, key="vs_reject_force_out",
        )
        return {
            "iv_threshold":  iv_thresh,
            "force_include": force_in,
            "force_exclude": force_out,
        }

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.

    # ── Checkpoint 2 — Feature Shortlist Sign-Off ──────────────────────────────
    if st.session_state.has_run and data:
        st.markdown('---')
        st.subheader('Checkpoint 2 — Feature Shortlist Sign-Off')
        _sel = data.get('selected_features', [])
        st.caption(f"{len(_sel)} features shortlisted for modelling.")
        _cp2 = load_checkpoints().get('cp2', {})
        if _cp2.get('decision'):
            (st.success if _cp2['decision'] == 'approved' else st.error)(
                f"Checkpoint 2 {_cp2['decision']} — {_cp2.get('timestamp', '')}")
        cp2_notes = st.text_area('Notes', value=_cp2.get('analyst_notes', ''), key='cp2_notes')
        _cp2c1, _cp2c2 = st.columns(2)
        if _cp2c1.button('✓ Approve Feature Shortlist', key='cp2_approve', use_container_width=True):
            save_checkpoint('cp2', {'decision': 'approved', 'analyst_notes': cp2_notes,
                                    'approved_features': _sel})
            st.success('Checkpoint 2 approved.')
            st.rerun()
        if _cp2c2.button('✗ Reject', key='cp2_reject', use_container_width=True):
            save_checkpoint('cp2', {'decision': 'rejected', 'analyst_notes': cp2_notes})
            st.error('Checkpoint 2 rejected.')
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Development":
    st.title("Model Development")
    st.write("Candidate model training, hyperparameter optimisation, and champion selection.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("model_development")
    show_ai_findings(data, "ModelDevelopmentAgent")

    if not data:
        no_data()
    else:
        mm    = data.get("model_metrics", {})
        champ = data.get("champion_model_name", "")
        cps   = load_checkpoints()
        cov   = cps.get("champion_override", {})
        effective_champ = cov.get("model", champ) if cov else champ
        if cov.get("model") and cov["model"] != champ:
            st.warning(f"**Champion overridden:** {cov['model']} — {cov.get('justification','—')}")

        _md_tabs = st.tabs(["Sample Split", "Model Comparison", "Champion Detail", "Hyperparameters"])

        with _md_tabs[0]:  # Sample Split
            sd = data.get("split_details", {})
            sm = data.get("split_method", "random")
            if sd:
                st.caption(f"**Split method:** {sd.get('method', sm)}")

                def _dr(v): return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "—"
                def _sz(v): return f"{v:,}"        if isinstance(v, (int, float)) else "—"

                split_rows = []
                if sd.get("total") is not None:
                    split_rows.append({"Set": "Total", "Size": _sz(sd.get("total")), "Default Rate": "—"})
                if sd.get("dev_size") is not None:
                    split_rows.append({"Set": "Dev (earliest 60%)", "Size": _sz(sd.get("dev_size")),
                                       "Default Rate": _dr(sd.get("dev_default_rate"))})
                _indent = "  " if sd.get("dev_size") is not None else ""
                split_rows.append({"Set": f"{_indent}Train", "Size": _sz(sd.get("train_size")),
                                   "Default Rate": _dr(sd.get("train_default_rate"))})
                split_rows.append({"Set": f"{_indent}Test", "Size": _sz(sd.get("test_size")),
                                   "Default Rate": _dr(sd.get("test_default_rate"))})
                if sd.get("oot_size") is not None:
                    split_rows.append({"Set": "OOT (latest 40%)", "Size": _sz(sd.get("oot_size")),
                                       "Default Rate": _dr(sd.get("oot_default_rate"))})
                show_df(pd.DataFrame(split_rows), use_container_width=True, hide_index=True, height=230)

                if sm == "time_based":
                    st.caption("3-way split — OOT = latest 20% of loans by origination date (chronological "
                               "holdout); Train/Test = random stratified 70/30 within the earliest-80% Dev "
                               "pool. Blind Dataset 2 can overwrite the OOT metrics via evaluate_oot.py.")
                else:
                    st.caption("No time column available — fell back to a random stratified split.")

                dist_rows = []
                for label, key in [("Train", "train_default_rate"), ("Test", "test_default_rate"),
                                   ("OOT", "oot_default_rate")]:
                    dr = sd.get(key)
                    if isinstance(dr, (int, float)):
                        dist_rows.append({"Split": label, "Class": "Default (1)",     "Rate": dr})
                        dist_rows.append({"Split": label, "Class": "Non-Default (0)", "Rate": 1 - dr})
                if dist_rows:
                    fig_split = px.bar(pd.DataFrame(dist_rows), x="Split", y="Rate", color="Class",
                                       barmode="stack", title="Class Distribution by Split",
                                       template="plotly_dark",
                                       color_discrete_map={"Default (1)": "#ef4444", "Non-Default (0)": "#10b981"})
                    fig_split.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), yaxis_tickformat=".0%")
                    st.plotly_chart(fig_split, use_container_width=True)
            else:
                st.info("Split details not available — re-run the pipeline.")

        with _md_tabs[1]:  # Model Comparison
            if mm:
                sorted_models = sorted(mm.items(), key=lambda x: x[1].get("auc_test", 0), reverse=True)
                second_best   = sorted_models[1][0] if len(sorted_models) > 1 else None
                rows = []
                for name, m in mm.items():
                    rec_label = "✅ Champion" if name == effective_champ else ("⭐ Runner-up" if name == second_best else "—")
                    rows.append({"Model": name, "AUC": m.get("auc_test"), "KS": m.get("ks"),
                                 "Gini": m.get("gini"), "F1": m.get("f1"),
                                 "Overfit Δ": m.get("overfit"),
                                 "Champion": "✓" if name == effective_champ else "",
                                 "Recommended": rec_label})
                show_df(pd.DataFrame(rows).sort_values("AUC", ascending=False),
                             use_container_width=True, hide_index=True, height=150)
                chart_rows = [{"Model": n, "Metric": met, "Value": float(v)}
                              for n, m in mm.items()
                              for met, v in [("AUC", m.get("auc_test")), ("KS", m.get("ks")), ("Gini", m.get("gini"))]
                              if v is not None]
                if chart_rows:
                    fig_models = px.bar(pd.DataFrame(chart_rows), x="Model", y="Value",
                                        color="Metric", barmode="group",
                                        title="Model Comparison — AUC / KS / Gini",
                                        template="plotly_dark",
                                        color_discrete_map={"AUC": "#4f6ef7", "KS": "#10b981", "Gini": "#f59e0b"})
                    fig_models.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_models, use_container_width=True)
                st.write(data.get("model_selection_rationale") or "No rationale — run with API key.")
                model_names = list(mm.keys())
                idx = model_names.index(effective_champ) if effective_champ in model_names else 0
                sel = st.selectbox("Override champion model", model_names, index=idx)
                jst = st.text_input("Justification", key="champ_just")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Apply champion override"):
                        save_checkpoint("champion_override", {"model": sel, "justification": jst,
                                                              "algorithm_champion": champ})
                        st.success(f"Champion overridden to '{sel}'.")
                        st.rerun()
                with col2:
                    if cov and st.button("Clear champion override"):
                        existing = load_checkpoints()
                        existing.pop("champion_override", None)
                        with open("outputs/checkpoints.json", "w") as f:
                            json.dump(existing, f, indent=2)
                        st.rerun()
            else:
                st.info("Model metrics not available.")

        with _md_tabs[2]:  # Champion Detail
            if effective_champ and effective_champ in mm:
                cm = mm[effective_champ]
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Model", effective_champ)
                c2.metric("AUC",   f"{cm.get('auc_test','—')}")
                c3.metric("KS",    f"{cm.get('ks','—')}")
                c4.metric("Gini",  f"{cm.get('gini','—')}")
                c5.metric("F1",    f"{cm.get('f1','—')}")
                c6.metric("Overfit Δ", f"{cm.get('overfit','—')}")
            else:
                st.info("No champion selected.")

        with _md_tabs[3]:  # Hyperparameters
            champion = effective_champ
            champ_metrics = mm.get(champion, {}) if champion else {}
            # FIX 1 — robust read across possible key names
            hyperparams = (champ_metrics.get("best_params")
                           or champ_metrics.get("params")
                           or champ_metrics.get("hyperparameters"))

            if not hyperparams and champion:
                st.warning(f"No tuned hyperparameters recorded for **{champion}** — this model uses "
                           f"fixed/default parameters (not Optuna-tuned). Showing the default "
                           f"configuration used:")
                default_params_map = {
                    "LightGBM": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.05,
                                 "scale_pos_weight": "auto-computed"},
                    "RandomForest": {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"},
                    "LogisticRegression": {"max_iter": 500, "C": 0.1, "class_weight": "balanced"},
                    "GradientBoosting_AutoML": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
                }
                hyperparams = default_params_map.get(champion, {})

            if hyperparams:
                show_df(pd.DataFrame(list(hyperparams.items()), columns=["Parameter", "Value"]),
                        use_container_width=True, hide_index=True)
            elif champion:
                st.info(f"No hyperparameter configuration available for {champion}.")
            else:
                st.info("No champion selected.")

            # FIX 2 — Manual Hyperparameter Tuning (HITL)
            if champion:
                st.markdown("---")
                st.subheader("⚙ Manual Hyperparameter Tuning")
                st.caption("Adjust parameters and re-train the champion. Changes take effect on the "
                           "next pipeline run launched with `--hyperparam-override`.")
                with st.expander("Override Champion Hyperparameters", expanded=False):
                    hp = hyperparams or {}
                    override_params = {}

                    def _iget(k, d):  # safe int default from possibly-string values
                        try: return int(float(hp.get(k, d)))
                        except Exception: return d
                    def _fget(k, d):
                        try: return float(hp.get(k, d))
                        except Exception: return d

                    if champion == "XGBoost":
                        override_params["n_estimators"] = st.number_input("n_estimators", 50, 1000, _iget("n_estimators", 200), step=10)
                        override_params["max_depth"] = st.number_input("max_depth", 2, 10, _iget("max_depth", 5))
                        override_params["learning_rate"] = st.number_input("learning_rate", 0.001, 0.5, _fget("learning_rate", 0.05), format="%.4f")
                        override_params["subsample"] = st.slider("subsample", 0.5, 1.0, _fget("subsample", 0.8))
                        override_params["min_child_weight"] = st.number_input("min_child_weight", 1, 100, _iget("min_child_weight", 10))
                    elif champion == "LightGBM":
                        override_params["n_estimators"] = st.number_input("n_estimators", 50, 1000, _iget("n_estimators", 100), step=10)
                        override_params["num_leaves"] = st.number_input("num_leaves", 5, 100, _iget("num_leaves", 31))
                        override_params["learning_rate"] = st.number_input("learning_rate", 0.001, 0.5, _fget("learning_rate", 0.05), format="%.4f")
                    elif champion == "RandomForest":
                        override_params["n_estimators"] = st.number_input("n_estimators", 50, 500, _iget("n_estimators", 200), step=10)
                        override_params["max_depth"] = st.number_input("max_depth", 2, 20, _iget("max_depth", 8))
                    else:
                        st.info(f"Manual tuning UI not configured for {champion}")

                    hp_tune_notes = st.text_area("Reason for hyperparameter change", key="hp_tune_notes")
                    if st.button("💾 Save Hyperparameter Override") and override_params:
                        from datetime import datetime as _dt
                        cps_path = "outputs/checkpoints.json"
                        _cps = {}
                        if os.path.exists(cps_path):
                            with open(cps_path, encoding="utf-8") as _f:
                                _cps = json.load(_f)
                        _cps["hyperparameter_override"] = {
                            "model": champion,
                            "params": override_params,
                            "analyst_notes": hp_tune_notes,
                            "timestamp": _dt.now().isoformat(),
                            "applied": False,
                        }
                        with open(cps_path, "w", encoding="utf-8") as _f:
                            json.dump(_cps, _f, indent=2, default=str)
                        st.success("✓ Hyperparameter override saved. Re-run the pipeline with "
                                   "`--hyperparam-override` to apply it (it is used once, then marked applied).")

    page_recs = [r for r in data.get("recommendations", []) if r.get("title") == "Champion Model Selection"]
    show_ai_recommendation(data, "ModelDevelopmentAgent", page_recs)

    def _md_reject_panel():
        st.write("Override champion selection and adjust the overfit penalty threshold.")
        mm_rej = data.get("model_metrics", {})
        model_names_rej = list(mm_rej.keys()) if mm_rej else []
        manual_champ = st.selectbox(
            "Manually select champion model",
            options=model_names_rej, key="md_reject_champ",
        )
        overfit_penalty = st.number_input(
            "Overfit penalty threshold (train-test AUC gap allowed)",
            value=0.03, min_value=0.0, max_value=0.20, step=0.01,
            key="md_reject_penalty",
        )
        return {
            "manual_champion":            manual_champ,
            "overfit_penalty_threshold":  overfit_penalty,
        }

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡 Explainability":
    st.title("Explainability")
    st.write("SHAP feature importance, score drivers, and adverse action codes.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("explainability")
    show_ai_findings(data, "ExplainabilityAgent")

    if not data:
        no_data()
    else:
        _expl_tabs = st.tabs(["Feature Importance", "SHAP Summary", "SHAP Beeswarm",
                              "Adverse Action", "What-If Analysis (LIME)", "Fairness & Bias Check"])

        with _expl_tabs[1]:  # SHAP Summary
            st.write(data.get("shap_summary") or "No SHAP narrative — run with API key.")

        with _expl_tabs[2]:  # SHAP Beeswarm
            run_id = data.get("run_id", "")
            _shap_files = sorted(glob.glob("outputs/models/*_shap_sample.json"), reverse=True)
            _shap_path = next((sf for sf in _shap_files if run_id and run_id in os.path.basename(sf)), None)
            if _shap_path is None and _shap_files:
                _shap_path = _shap_files[0]
            if _shap_path:
                try:
                    with open(_shap_path, encoding="utf-8") as f:
                        _sp = json.load(f)
                    _names = _sp["feature_names"]
                    _shap = np.array(_sp["shap_values"], dtype=float)
                    _feat = np.array(_sp["feature_values"], dtype=float)
                    _order = np.argsort(np.abs(_shap).mean(axis=0))[::-1][:10]
                    _top_names = [_names[i] for i in _order]
                    fig_bee = go.Figure()
                    for _row, _fi in enumerate(reversed(list(_order))):
                        _sv = _shap[:, _fi]
                        _xv = _feat[:, _fi]
                        _norm = (_xv - np.nanmin(_xv)) / (np.nanmax(_xv) - np.nanmin(_xv) + 1e-8)
                        _yj = _row + np.random.uniform(-0.3, 0.3, len(_sv))
                        fig_bee.add_trace(go.Scatter(
                            x=_sv, y=_yj, mode="markers",
                            marker=dict(size=4, color=_norm, colorscale="RdBu_r", opacity=0.6),
                            showlegend=False,
                        ))
                    fig_bee.update_layout(
                        title="SHAP Summary Plot — Feature Impact on Default Prediction",
                        xaxis_title="SHAP value (impact on model output)",
                        yaxis=dict(tickvals=list(range(len(_top_names))),
                                   ticktext=list(reversed(_top_names))),
                        template="plotly_dark", height=500, margin=dict(l=0, r=0, t=40, b=0),
                        shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(_top_names) - 0.5,
                                     line=dict(color="white", width=1, dash="dash"))],
                    )
                    st.plotly_chart(fig_bee, use_container_width=True)
                    st.caption("Red = high feature value, Blue = low. Points right of centre increase "
                               "default risk, left decrease it.")
                except Exception as e:
                    st.warning(f"Could not render SHAP beeswarm: {e}")
            else:
                st.info("SHAP sample not available — re-run the pipeline to generate the beeswarm plot.")

        with _expl_tabs[3]:  # Adverse Action
            adverse = data.get("adverse_action_codes", {})
            st.subheader(f"Adverse Action Codes ({len(adverse)} samples)")
            if adverse:
                for sample_id, sample_data in adverse.items():
                    with st.expander(f"{sample_id} — {sample_data.get('predicted_prob', 0):.2%}"):
                        reasons = sample_data.get("top_reasons", [])
                        if reasons:
                            show_df(pd.DataFrame(reasons), use_container_width=True,
                                         hide_index=True, height=150)
            else:
                st.info("Adverse action codes not available.")

        with _expl_tabs[0]:  # Feature Importance (default tab)
            feat_imp = data.get("feature_importance", {})
            if feat_imp:
                imp_df = (
                    pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Mean |SHAP|"])
                    .sort_values("Mean |SHAP|", ascending=False)
                )
                top_imp = imp_df.head(25).sort_values("Mean |SHAP|", ascending=True)
                fig_shap = px.bar(
                    top_imp, x="Mean |SHAP|", y="Feature", orientation="h",
                    title="SHAP Feature Importance (top 25)",
                    template="plotly_dark",
                    color="Mean |SHAP|",
                    color_continuous_scale=["#4f6ef7", "#10b981"],
                )
                fig_shap.update_layout(
                    height=280, margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # FIX 5 — SHAP beeswarm scatter ───────────────────────────
                shap_detail = data.get("shap_detail", {})
                if shap_detail:
                    bswarm_rows = []
                    top10 = list(imp_df["Feature"].head(10))
                    for feat in top10:
                        feat_data = shap_detail.get(feat, {})
                        shap_vals  = feat_data.get("shap_values", [])
                        feat_vals  = feat_data.get("feature_values", [])
                        if shap_vals:
                            for sv, fv in zip(shap_vals, feat_vals if feat_vals else [None]*len(shap_vals)):
                                bswarm_rows.append({
                                    "Feature":       feat,
                                    "SHAP value":    float(sv),
                                    "Feature value": float(fv) if fv is not None else 0.0,
                                })
                    if bswarm_rows:
                        bs_df = pd.DataFrame(bswarm_rows)
                        fv_min = bs_df["Feature value"].min()
                        fv_max = bs_df["Feature value"].max()
                        bs_df["feat_norm"] = (bs_df["Feature value"] - fv_min) / max(fv_max - fv_min, 1e-9)
                        fig_bee = px.scatter(
                            bs_df, x="SHAP value", y="Feature", color="feat_norm",
                            title="SHAP Beeswarm — top 10 features",
                            template="plotly_dark",
                            color_continuous_scale="RdBu_r",
                            opacity=0.6,
                            labels={"feat_norm": "Feature value (norm)"},
                        )
                        fig_bee.update_layout(
                            height=280, margin=dict(l=0, r=0, t=30, b=0),
                            coloraxis_colorbar=dict(title="Feature<br>value", len=0.5),
                        )
                        st.plotly_chart(fig_bee, use_container_width=True)
                        st.caption("Red = high feature value, Blue = low. Right of zero → increases risk.")
                else:
                    st.caption("Run pipeline with detailed SHAP output for beeswarm chart.")

                show_df(imp_df, use_container_width=True, hide_index=True, height=220)
            else:
                st.info("Feature importance not available.")

        with _expl_tabs[4]:  # What-If Analysis (LIME)
            st.caption(
                "LIME-style local explanation — adjust feature values to see how each factor "
                "changes this borrower's predicted default risk. Helps validate model direction "
                "and explain decisions to stakeholders."
            )
            # Load champion model pkl
            _pkl_path = "outputs/champion_model.pkl"
            _lime_model = None
            _lime_feats = []
            if os.path.exists(_pkl_path):
                try:
                    import joblib as _jl
                    _pkl = _jl.load(_pkl_path)
                    _lime_model = _pkl.get("model")
                    _lime_feats = _pkl.get("selected_features", [])
                except Exception as _e:
                    st.warning(f"Could not load champion model: {_e}")
            else:
                st.info("Champion model not saved yet — run the pipeline first. "
                        "(outputs/champion_model.pkl is created by the Validation agent.)")

            if _lime_model and _lime_feats:
                # Load dataset to get feature ranges (header-aware, matches pipeline names)
                _lime_df, _ = load_run_raw_df(data)

                top10_feats = _lime_feats[:10]
                obs = {}
                slider_cols = st.columns(2)
                for idx_f, feat in enumerate(top10_feats):
                    with slider_cols[idx_f % 2]:
                        if _lime_df is not None and feat in _lime_df.columns:
                            col_s = _lime_df[feat].dropna()
                            if pd.api.types.is_numeric_dtype(col_s) and len(col_s) > 0:
                                f_min  = float(col_s.quantile(0.01))
                                f_max  = float(col_s.quantile(0.99))
                                f_med  = float(col_s.median())
                                f_step = max((f_max - f_min) / 100, 1e-6)
                                obs[feat] = st.slider(
                                    feat, min_value=f_min, max_value=f_max,
                                    value=f_med, step=f_step,
                                    key=f"lime_sl_{feat}",
                                )
                            else:
                                obs[feat] = 0.0
                                st.caption(f"`{feat}` — non-numeric, using 0")
                        else:
                            obs[feat] = st.number_input(
                                feat, value=0.0, key=f"lime_ni_{feat}")

                # Build feature vector for all selected features (fill missing with 0)
                _feat_vec = pd.DataFrame([{f: obs.get(f, 0.0) for f in _lime_feats}]).fillna(0)
                try:
                    _prob = float(_lime_model.predict_proba(_feat_vec)[0, 1])
                    # Gauge display
                    _gauge_color = "#ef4444" if _prob > 0.5 else ("#f59e0b" if _prob > 0.25 else "#10b981")
                    st.markdown(
                        f"<div style='background:{_gauge_color}22;border-left:4px solid {_gauge_color};"
                        f"border-radius:6px;padding:12px 20px;margin:12px 0;text-align:center'>"
                        f"<div style='font-size:13px;color:#9ca3af'>Predicted Default Probability</div>"
                        f"<div style='font-size:36px;font-weight:700;color:{_gauge_color}'>{_prob:.1%}</div>"
                        f"<div style='background:#1f2937;border-radius:4px;height:12px;margin:8px 0'>"
                        f"<div style='background:{_gauge_color};width:{_prob*100:.1f}%;height:12px;border-radius:4px'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                    # Baseline = median of each feature; feature contributions = prob_with_feature - baseline
                    if _lime_df is not None:
                        _baseline_vec = pd.DataFrame([{
                            f: float(_lime_df[f].median()) if f in _lime_df.columns and pd.api.types.is_numeric_dtype(_lime_df[f]) else 0.0
                            for f in _lime_feats
                        }]).fillna(0)
                        _base_prob = float(_lime_model.predict_proba(_baseline_vec)[0, 1])
                        contribs = []
                        for feat in top10_feats:
                            _v = _baseline_vec.copy()
                            _v[feat] = obs.get(feat, 0.0)
                            _p = float(_lime_model.predict_proba(_v)[0, 1])
                            contribs.append({"Feature": feat, "Contribution": round(_p - _base_prob, 4)})
                        _contrib_df = pd.DataFrame(contribs).sort_values("Contribution", key=abs, ascending=False)
                        _colors = ["#ef4444" if c > 0 else "#10b981" for c in _contrib_df["Contribution"]]
                        fig_lime = go.Figure(go.Bar(
                            x=_contrib_df["Contribution"], y=_contrib_df["Feature"],
                            orientation="h", marker_color=_colors,
                            text=[f"{v:+.4f}" for v in _contrib_df["Contribution"]],
                            textposition="outside",
                        ))
                        fig_lime.update_layout(
                            title=f"Feature Contributions (vs. median baseline {_base_prob:.1%})",
                            height=280, margin=dict(l=0, r=0, t=40, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e7ff", xaxis_title="Δ Predicted Probability",
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)
                        st.caption("Red = increases default risk vs. median borrower · Green = decreases risk. "
                                   "Each bar shows the isolated impact of that feature at its current slider value.")
                except Exception as _e:
                    st.error(f"Prediction error: {_e}")

        with _expl_tabs[5]:  # Fairness & Bias Check
            st.caption(
                "Fairness check per SR 11-7 / fair-lending requirements. Computed in the pipeline "
                "(Explainability agent) on the test set and saved to the audit trail — this view "
                "matches exactly what appears in the Model Development Document."
            )
            fairness = data.get("fairness_results", {}) if data else {}
            if not fairness:
                st.info("Fairness results not available — run the pipeline to compute them.")
            else:
                _fair_rows = []
                for attr, groups in fairness.items():
                    if not groups:
                        continue
                    st.subheader(attr)
                    _g = pd.DataFrame([
                        {"Category": k,
                         "Mean Predicted": v.get("mean_predicted"),
                         "Actual Rate": v.get("actual_rate"),
                         "Diff from Avg": v.get("diff_from_avg"),
                         "Count": v.get("count"),
                         "Concern": v.get("concern_level")}
                        for k, v in groups.items()
                    ]).sort_values("Mean Predicted", ascending=False)
                    # overall mean = mean_predicted − diff_from_avg (constant across groups)
                    _overall = float((_g["Mean Predicted"] - _g["Diff from Avg"]).iloc[0]) if len(_g) else 0.0
                    _colors = ["#ef4444" if c == "High" else "#f59e0b" if c == "Medium" else "#10b981"
                               for c in _g["Concern"]]
                    fig_fair = go.Figure(go.Bar(
                        x=_g["Category"].astype(str), y=_g["Mean Predicted"], marker_color=_colors,
                        text=[f"{v:.1%}" for v in _g["Mean Predicted"]], textposition="outside"))
                    fig_fair.add_hline(y=_overall, line_dash="dash", line_color="#9ca3af",
                                       annotation_text=f"Overall avg {_overall:.1%}",
                                       annotation_position="top right")
                    fig_fair.update_layout(
                        title=f"Mean Predicted Default Probability by {attr}",
                        height=300, margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e0e7ff", yaxis_tickformat=".0%", yaxis_title="Mean Predicted Prob")
                    st.plotly_chart(fig_fair, use_container_width=True)
                    _flagged = _g[_g["Concern"].isin(["Medium", "High"])]
                    if not _flagged.empty:
                        st.warning(f"⚠ {len(_flagged)} group(s) in `{attr}` flagged (>10% deviation): "
                                   + ", ".join(f"`{r}`" for r in _flagged["Category"].tolist()))
                    for _, r in _g.iterrows():
                        _fair_rows.append({"Attribute": attr, **r.to_dict()})

                if _fair_rows:
                    st.subheader("Fairness Summary Table")
                    show_df(pd.DataFrame(_fair_rows), use_container_width=True, hide_index=True, height=240)

                # HITL fairness review
                st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
                st.subheader("Fairness Review Decision")
                cps_fair = load_checkpoints()
                prev_fair = cps_fair.get("fairness_review", {})
                fair_notes = st.text_area(
                    "Document fairness review findings", value=prev_fair.get("notes", ""),
                    placeholder="Describe any disparate impact findings, mitigations, and compliance status…",
                    height=80, key="fairness_notes")
                _fopts = ["No Concern", "Flagged for Review", "Escalate to Compliance"]
                fair_decision = st.selectbox(
                    "Fairness review decision", _fopts,
                    index=_fopts.index(prev_fair.get("decision")) if prev_fair.get("decision") in _fopts else 0,
                    key="fairness_decision")
                fair_analyst = st.text_input("Analyst name", key="fairness_analyst",
                                             value=prev_fair.get("analyst", ""))
                if st.button("Save fairness review", key="fairness_save"):
                    if fair_analyst.strip():
                        save_checkpoint("fairness_review", {"decision": fair_decision,
                            "notes": fair_notes.strip(), "analyst": fair_analyst.strip()})
                        st.success(f"Fairness review saved: {fair_decision}")
                    else:
                        st.error("Enter analyst name before saving.")

    show_ai_recommendation(data, "ExplainabilityAgent")

    def _expl_reject_panel():
        st.write("Exclude specific features from the SHAP explanation output.")
        feat_imp_rej = data.get("feature_importance", {})
        top_feats = list(feat_imp_rej.keys())[:40] if feat_imp_rej else []
        excl_feats = st.multiselect(
            "Exclude features from explanation",
            options=top_feats, key="expl_reject_excl",
        )
        return {"features_to_exclude": excl_feats}

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "✅ Validation":
    st.title("Model Validation")
    st.write("Discriminatory power, stability, calibration, and challenger comparison.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    show_stale_target_banner()
    show_decision_badge("validation")
    show_ai_findings(data, "ValidationAgent")

    # ── Editable Thresholds ──────────────────────────────────────────────────
    with st.expander("⚙ Adjust Validation Thresholds"):
        st.caption(
            "Default thresholds follow industry standard minimums. "
            "Adjust with HOTL/HITL authorisation and document justification."
        )
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            auc_threshold = st.slider(
                "Minimum AUC threshold", 0.50, 0.85,
                float(st.session_state.get("val_auc_threshold", 0.70)), 0.01,
                key="val_auc_threshold",
            )
        with tc2:
            ks_threshold = st.slider(
                "Minimum KS threshold", 0.10, 0.50,
                float(st.session_state.get("val_ks_threshold", 0.25)), 0.01,
                key="val_ks_threshold",
            )
        with tc3:
            gini_threshold = st.slider(
                "Minimum Gini threshold", 0.20, 0.70,
                float(st.session_state.get("val_gini_threshold", 0.40)), 0.01,
                key="val_gini_threshold",
            )
        with tc4:
            psi_threshold = st.slider(
                "Maximum PSI threshold", 0.05, 0.50,
                float(st.session_state.get("val_psi_threshold", 0.10)), 0.01,
                key="val_psi_threshold",
            )
        thr_name = st.text_input("Analyst name (for threshold change log)", key="val_thr_name")
        thr_just = st.text_input("Justification for threshold change", key="val_thr_just")
        if st.button("Save threshold changes", key="val_save_thr"):
            if thr_name.strip():
                save_checkpoint("validation_thresholds", {
                    "auc_threshold":  st.session_state.get("val_auc_threshold",  0.70),
                    "ks_threshold":   st.session_state.get("val_ks_threshold",   0.25),
                    "gini_threshold": st.session_state.get("val_gini_threshold", 0.40),
                    "psi_threshold":  st.session_state.get("val_psi_threshold",  0.10),
                    "analyst_name":   thr_name.strip(),
                    "justification":  thr_just.strip(),
                })
                st.success("Threshold changes saved to checkpoints.")
            else:
                st.error("Enter your analyst name to save threshold changes.")

    # Read current thresholds from session_state (sliders persist within session)
    auc_threshold  = float(st.session_state.get("val_auc_threshold",  0.70))
    ks_threshold   = float(st.session_state.get("val_ks_threshold",   0.25))
    gini_threshold = float(st.session_state.get("val_gini_threshold", 0.40))
    psi_threshold  = float(st.session_state.get("val_psi_threshold",  0.10))

    # Derive AMBER thresholds as 0.05 below GREEN (standard industry practice)
    amber_auc  = round(auc_threshold  - 0.05, 2)
    amber_ks   = round(ks_threshold   - 0.05, 2)
    amber_psi  = round(psi_threshold  * 2.5,  2)

    if not data:
        no_data()
    else:
        vm      = data.get("validation_metrics", {})
        psi     = data.get("psi_results", {})
        passed  = data.get("validation_passed", False)
        mm      = data.get("model_metrics", {})
        champ   = data.get("champion_model_name", "")
        champ_m = mm.get(champ, {})

        cps = load_checkpoints()
        vov = cps.get("validation_override", {})
        if vov.get("decision"):
            st.warning(f"**Validation Decision Overridden: {vov['decision']}** — "
                       f"{vov.get('justification','—')} *(Recorded: {vov.get('timestamp','')})*")
            effective_verdict = vov["decision"]
        else:
            effective_verdict = "Pass" if passed else "Conditional"

        auc   = float(champ_m.get("auc_test", vm.get("auc")) or 0.0)
        ks    = float(champ_m.get("ks",       vm.get("ks"))  or 0.0)
        gini  = float(champ_m.get("gini",     vm.get("gini")) or 0.0)
        brier = vm.get("brier_score")
        psi_v = float(psi.get("psi_score") or 1.0)

        green_met = (auc >= auc_threshold and ks >= ks_threshold and psi_v < psi_threshold)
        amber_met = (auc >= amber_auc     and ks >= amber_ks     and psi_v < amber_psi)

        if green_met:
            risk_rating, rating_color = "GREEN", "#10b981"
            rating_text = "LOW RISK — All thresholds met"
        elif amber_met:
            risk_rating, rating_color = "AMBER", "#f59e0b"
            rating_text = "MEDIUM RISK — Minimum thresholds met"
        else:
            risk_rating, rating_color = "RED", "#ef4444"
            rating_text = "HIGH RISK — Threshold(s) not met"

        st.markdown(
            f"<div style='background:{rating_color}22;border-left:4px solid {rating_color};"
            f"border-radius:4px;padding:6px 12px;margin:4px 0'>"
            f"<span style='color:{rating_color};font-weight:700'>🚦 {risk_rating}</span>"
            f" — {rating_text}</div>",
            unsafe_allow_html=True,
        )

        _val_tabs = st.tabs(["KPI Scoreboard", "Performance", "Stability",
                             "Calibration", "Challenger", "CSI", "Findings Register",
                             "Confusion Matrix"])

        with _val_tabs[0]:  # KPI Scoreboard
            scoreboard = data.get("kpi_scoreboard") or vm.get("kpi_scoreboard", [])
            # Gini Gap lives on the Performance tab's Dev→OOT table — drop it here
            # (also filters older audit trails that still carry the row).
            scoreboard = [r for r in scoreboard if r.get("KPI") != "Gini Gap (Dev→OOT)"]
            if scoreboard:
                rags = [str(r.get("RAG", "")).upper() for r in scoreboard]
                n_green, n_amber, n_red = rags.count("GREEN"), rags.count("AMBER"), rags.count("RED")
                total = len(rags)
                _sc_color = "#10b981" if n_red == 0 and n_amber == 0 else "#f59e0b" if n_red == 0 else "#ef4444"
                st.markdown(
                    f"<div style='font-size:15px;font-weight:600'>"
                    f"<span style='color:{_sc_color}'>{n_green} of {total} KPIs GREEN</span> · "
                    f"<span style='color:#f59e0b'>{n_amber} AMBER</span> · "
                    f"<span style='color:#ef4444'>{n_red} RED</span></div>",
                    unsafe_allow_html=True,
                )
                sb_df = pd.DataFrame(scoreboard)
                # Format Value as strings so integer KPIs (Rank Order Breaks) show as
                # "0" rather than "0.000000" (pandas coerces a mixed column to float).
                if "Value" in sb_df.columns and "KPI" in sb_df.columns:
                    def _fmt_sb_val(r):
                        v = r["Value"]
                        if r["KPI"] == "Rank Order Breaks":
                            try: return str(int(float(v)))
                            except (TypeError, ValueError): return str(v)
                        try: return f"{float(v):.4f}"
                        except (TypeError, ValueError): return str(v)
                    sb_df["Value"] = sb_df.apply(_fmt_sb_val, axis=1)
                # New Data column (blind Dataset 2 metrics) where available
                _new_eval_sb = data.get("new_data_evaluation", {})
                if _new_eval_sb.get("has_metrics") and "KPI" in sb_df.columns:
                    _new_map = {
                        "Gini":         _new_eval_sb.get("gini_new_data"),
                        "AUC / AUROC":  _new_eval_sb.get("auc_new_data"),
                        "KS Statistic": _new_eval_sb.get("ks_new_data"),
                    }
                    sb_df["New Data"] = sb_df["KPI"].map(
                        lambda k: f"{_new_map[k]:.4f}" if isinstance(_new_map.get(k), (int, float)) else "—")
                _rag_sub = [c for c in ["RAG"] if c in sb_df.columns]
                st.dataframe(
                    sb_df.style.map(_rag_cell_style, subset=_rag_sub) if _rag_sub else sb_df,
                    use_container_width=True, hide_index=True, height=290,
                )
            else:
                st.info("KPI scoreboard not available — re-run the pipeline to generate it.")

            st.subheader("Feature-Level KPIs (IV + CSI)")
            feat_kpi = data.get("feature_kpi_table") or vm.get("feature_kpi_table", [])
            if feat_kpi:
                fk_df = pd.DataFrame(feat_kpi)
                _fk_sub = [c for c in ["IV RAG", "CSI RAG"] if c in fk_df.columns]
                st.dataframe(
                    fk_df.style.map(_rag_cell_style, subset=_fk_sub) if _fk_sub else fk_df,
                    use_container_width=True, hide_index=True, height=300,
                )
            else:
                st.caption("No feature-level KPI table available.")

        with _val_tabs[1]:  # Performance
            # 3-way comparison: Development (Test) vs OOT (Dataset 1) vs New Data (blind Dataset 2).
            new_eval = data.get('new_data_evaluation', {})
            comparison_rows = []
            for metric, dev_key, oot_key, new_key, threshold in [
                ('AUC / AUROC', 'auc',  'auc_oot_d1',  'auc_new_data',  '≥0.68 GREEN'),
                ('Gini',        'gini', 'gini_oot_d1', 'gini_new_data', '≥0.35 GREEN'),
                ('KS Statistic','ks',   'ks_oot_d1',   'ks_new_data',   '0.25–0.65 GREEN'),
            ]:
                dev_val = vm.get(dev_key)
                oot_val = vm.get(oot_key)
                new_val = new_eval.get(new_key) if new_eval.get('has_metrics') else None

                def _gap(a, b):
                    if a and b:
                        g = abs(a - b) / a * 100
                        rag = '🟢' if g <= 10 else '🟡' if g <= 15 else '🔴'
                        return f"{g:.1f}% {rag}"
                    return 'N/A'

                comparison_rows.append({
                    'Metric': metric,
                    'Development': dev_val,
                    'OOT (Dataset 1)': oot_val,
                    'New Data': new_val if new_val else '—',
                    'Dev→OOT Gap': _gap(dev_val, oot_val),
                    'Dev→New Gap': _gap(dev_val, new_val) if new_val else '—',
                    'Threshold (GREEN)': threshold,
                })

            st.subheader("Performance: Development vs OOT vs New Data")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
            if not new_eval.get('has_metrics'):
                st.caption("New Data column shows — because either no target column was found in new data, "
                           "or new data has not been scored yet.")
            else:
                st.caption(f"New Data scored: {new_eval.get('dataset','—')} | "
                           f"{new_eval.get('total_records',0):,} records | "
                           f"Default rate: {new_eval.get('default_rate',0):.2%}")

            crit_rows = [
                {"Criterion": f"AUC ≥ {auc_threshold:.2f}", "Value": f"{auc:.4f}",  "Tier": "GREEN",
                 "Status": "✓ Pass" if auc >= auc_threshold else "✗ Fail"},
                {"Criterion": f"KS ≥ {ks_threshold:.2f}",   "Value": f"{ks:.4f}",   "Tier": "GREEN",
                 "Status": "✓ Pass" if ks >= ks_threshold   else "✗ Fail"},
                {"Criterion": f"PSI < {psi_threshold:.2f}",  "Value": f"{psi_v:.4f}", "Tier": "GREEN",
                 "Status": "✓ Pass" if psi_v < psi_threshold else "✗ Fail"},
                {"Criterion": f"AUC ≥ {amber_auc:.2f}",  "Value": f"{auc:.4f}",  "Tier": "AMBER",
                 "Status": "✓ Pass" if auc >= amber_auc  else "✗ Fail"},
                {"Criterion": f"KS ≥ {amber_ks:.2f}",   "Value": f"{ks:.4f}",   "Tier": "AMBER",
                 "Status": "✓ Pass" if ks >= amber_ks   else "✗ Fail"},
                {"Criterion": f"PSI < {amber_psi:.2f}",  "Value": f"{psi_v:.4f}", "Tier": "AMBER",
                 "Status": "✓ Pass" if psi_v < amber_psi else "✗ Fail"},
            ]
            show_df(pd.DataFrame(crit_rows), use_container_width=True,
                         hide_index=True, height=200)

            deciles = vm.get("decile_table", [])
            if deciles:
                dec_df = pd.DataFrame(deciles)
                show_df(dec_df, use_container_width=True, hide_index=True, height=220)
                dr_col  = next((c for c in dec_df.columns if "default" in c.lower() and "rate" in c.lower()), None)
                dec_col = next((c for c in dec_df.columns if "decile" in c.lower() or "score" in c.lower()), None)
                if dr_col and dec_col:
                    fig_dec = px.bar(dec_df, x=dec_col, y=dr_col,
                                     title="Default Rate by Score Decile", template="plotly_dark",
                                     color=dr_col, color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                                     labels={dec_col: "Decile", dr_col: "Default Rate"})
                    fig_dec.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                                          coloraxis_showscale=False)
                    st.plotly_chart(fig_dec, use_container_width=True)

            st.write(data.get("validation_summary") or "No summary — run with API key.")

            st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)
            opts    = ["Pass", "Conditional", "Fail"]
            cur_idx = opts.index(vov.get("decision", "Conditional")) if vov.get("decision") in opts else 1
            new_dec = st.radio("Override decision", opts, index=cur_idx, horizontal=True)
            jst     = st.text_area("Justification (required)", value=vov.get("justification", ""), height=60)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save validation override"):
                    if jst.strip():
                        save_checkpoint("validation_override", {
                            "decision": new_dec, "justification": jst.strip(),
                            "pipeline_decision": "Pass" if passed else "Conditional",
                        })
                        st.success(f"Overridden to '{new_dec}'.")
                        st.rerun()
                    else:
                        st.error("Justification is required.")
            with col2:
                if vov and st.button("Clear validation override"):
                    existing = load_checkpoints()
                    existing.pop("validation_override", None)
                    with open("outputs/checkpoints.json", "w") as f:
                        json.dump(existing, f, indent=2)
                    st.rerun()

        with _val_tabs[2]:  # Stability
            if psi:
                st.metric("PSI", f"{psi_v:.4f}", delta=psi.get("assessment", ""))
                st.caption(f"Split: {psi.get('split_label', '')}")

        with _val_tabs[3]:  # Calibration
            cal = vm.get("calibration", {})
            if cal:
                st.json(cal)
            else:
                st.caption(f"Brier score: {brier:.4f}" if isinstance(brier, float) else "No calibration data.")

        with _val_tabs[4]:  # Challenger
            challengers = vm.get("challenger_table", [])
            if challengers:
                show_df(pd.DataFrame(challengers), use_container_width=True,
                             hide_index=True, height=180)
            else:
                st.info("No challenger comparison available.")

        with _val_tabs[5]:  # CSI
            csi_results = vm.get("csi_results", {})
            if csi_results:
                st.caption("< 0.10 = Stable · 0.10–0.25 = Moderate · > 0.25 = Unstable")
                csi_rows = [
                    {"Feature": feat, "CSI": v["csi"], "Assessment": v["assessment"],
                     "Status": "✓ Stable" if v["csi"] < 0.10 else ("⚠ Moderate" if v["csi"] < 0.25 else "✗ Unstable")}
                    for feat, v in csi_results.items()
                ]
                csi_df = pd.DataFrame(csi_rows).sort_values("CSI", ascending=False)
                show_df(csi_df, use_container_width=True, hide_index=True, height=220)
                csi_colors = ["#10b981" if r["CSI"] < 0.10 else "#f59e0b" if r["CSI"] < 0.25 else "#ef4444"
                              for _, r in csi_df.iterrows()]
                fig_csi = go.Figure(go.Bar(x=csi_df["Feature"], y=csi_df["CSI"],
                                           marker_color=csi_colors,
                                           text=[f"{v:.4f}" for v in csi_df["CSI"]],
                                           textposition="outside"))
                fig_csi.add_hline(y=0.10, line_dash="dot", line_color="#f59e0b",
                                  annotation_text="0.10", annotation_position="top left")
                fig_csi.add_hline(y=0.25, line_dash="dot", line_color="#ef4444",
                                  annotation_text="0.25", annotation_position="top left")
                fig_csi.update_layout(title="CSI — Top Features", height=280,
                                      margin=dict(l=0, r=0, t=30, b=0),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="#e0e7ff", yaxis_title="CSI")
                st.plotly_chart(fig_csi, use_container_width=True)
            else:
                st.info("CSI not available — run pipeline to generate.")

        with _val_tabs[6]:  # Findings Register
            findings = data.get("findings_register") or vm.get("findings_register", [])
            n_high = sum(1 for f in findings if f.get("Severity") == "High")
            n_med  = sum(1 for f in findings if f.get("Severity") == "Medium")
            n_low  = sum(1 for f in findings if f.get("Severity") == "Low")
            if not findings:
                st.success("No findings raised — all KPIs within acceptable thresholds.")
            else:
                st.markdown(
                    f"<div style='font-size:15px;font-weight:600'>{len(findings)} findings total · "
                    f"<span style='color:#ef4444'>{n_high} High</span> · "
                    f"<span style='color:#f59e0b'>{n_med} Medium</span> · "
                    f"<span style='color:#10b981'>{n_low} Low</span></div>",
                    unsafe_allow_html=True,
                )
                fnd_df = pd.DataFrame(findings)
                _order = [c for c in ["Ref", "Category", "Finding", "Severity",
                                      "Recommended Remediation", "Status"] if c in fnd_df.columns]
                fnd_df = fnd_df[_order]
                _sev_sub = [c for c in ["Severity"] if c in fnd_df.columns]
                st.dataframe(
                    fnd_df.style.map(_severity_cell_style, subset=_sev_sub) if _sev_sub else fnd_df,
                    use_container_width=True, hide_index=True, height=340,
                )

        with _val_tabs[7]:  # Confusion Matrix
            cm = data.get('validation_metrics', {}).get('confusion_matrix', {})
            if cm:
                z = [[cm['true_negative'], cm['false_positive']],
                     [cm['false_negative'], cm['true_positive']]]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=z, x=['Predicted Good', 'Predicted Bad'],
                    y=['Actual Good', 'Actual Bad'],
                    text=z, texttemplate='%{text}', textfont={'size': 20},
                    colorscale='Blues'))
                fig_cm.update_layout(title=f"Confusion Matrix @ {cm['threshold']} threshold",
                                     template='plotly_dark', height=350,
                                     margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_cm, use_container_width=True)

                _cmcols = st.columns(5)
                _cmcols[0].metric('Accuracy', cm['accuracy'])
                _cmcols[1].metric('Precision', cm['precision'])
                _cmcols[2].metric('Recall', cm['recall'])
                _cmcols[3].metric('Specificity', cm['specificity'])
                _cmcols[4].metric('F1 Score', cm['f1_score'])
                st.warning(cm['note'])
            else:
                st.info('Confusion matrix not available — run pipeline to compute')

    page_recs = [r for r in data.get("recommendations", []) if r.get("title") == "Deployment Readiness"]
    show_ai_recommendation(data, "ValidationAgent", page_recs)

    def _val_reject_panel():
        st.write("Override the automated risk rating and record your justification.")
        risk_override = st.radio(
            "Override risk rating",
            ["GREEN", "AMBER", "RED"], horizontal=True, key="val_reject_risk",
        )
        analyst_name_rej = st.text_input(
            "Analyst name", placeholder="Your full name", key="val_reject_analyst",
        )
        justification_rej = st.text_area(
            "Justification (required)",
            placeholder="Explain why the rating is being overridden…",
            key="val_reject_just",
        )
        return {
            "risk_rating_override": risk_override,
            "analyst_name":         analyst_name_rej,
            "justification":        justification_rej,
        }

    # Human Decision section removed (FIX 4) — analyst sign-off lives on Model Sign-Off page.


# ══════════════════════════════════════════════════════════════════════════════
# HITL MATRIX
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👤 HITL Matrix":
    st.title("Human-in-the-Loop (HITL) Reference Matrix")
    st.write("Every phase requires a human decision before the pipeline proceeds. "
             "This table summarises what the AI does and what the analyst must do in each phase.")
    st.markdown('<hr style="margin:4px 0;border-color:#1e2436">', unsafe_allow_html=True)

    hitl_rows = [
        {
            "Phase":        "Data Understanding",
            "AI Action":    "Detect target, leakage columns, and schema profile",
            "Human Action": "Approve or override target definition",
        },
        {
            "Phase":        "Data Quality Review",
            "AI Action":    "Profile data quality, flag missing values and outliers",
            "Human Action": "Approve or override remediation decisions",
        },
        {
            "Phase":        "Feature Engineering",
            "AI Action":    "Generate domain-driven derived features and WOE encoding",
            "Human Action": "Approve or modify the engineered feature list",
        },
        {
            "Phase":        "Variable Selection",
            "AI Action":    "Recommend feature shortlist via IV ranking and correlation filtering",
            "Human Action": "Lock shortlist or restore manually removed variables",
        },
        {
            "Phase":        "Model Development",
            "AI Action":    "Train candidate models and recommend champion via AUC with overfit penalty",
            "Human Action": "Select final production model (may override AI champion)",
        },
        {
            "Phase":        "Explainability",
            "AI Action":    "Generate SHAP-based feature importance and adverse action codes",
            "Human Action": "Review model interpretation and flag any concerns",
        },
        {
            "Phase":        "Validation",
            "AI Action":    "Compute risk rating (GREEN/AMBER/RED) and deployment recommendation",
            "Human Action": "Approve, reject, or request retraining before sign-off",
        },
    ]

    show_df(
        pd.DataFrame(hitl_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Phase":        st.column_config.TextColumn("Phase", width="medium"),
            "AI Action":    st.column_config.TextColumn("AI Action", width="large"),
            "Human Action": st.column_config.TextColumn("Human Action", width="large"),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Documentation":
    st.title("Documentation & Governance")
    st.write("Model development report, assumptions, limitations, and risks.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    if not data:
        no_data()
    else:
        run_id = data.get("run_id", "report")
        report = st.session_state.report

        # ── Model Development Document package (Word + 3 Excel companions) ──
        st.subheader("Model Development Document Package")
        _docmap = [
            (f"{run_id}_Model_Development_Document.docx", "📄 Model Dev Document (Word)",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            (f"{run_id}_DQR_Full_Report.xlsx", "📊 DQR Full Report",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            (f"{run_id}_Feature_Selection_Detail.xlsx", "📊 Feature Selection",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            (f"{run_id}_Model_Comparison.xlsx", "📊 Model Comparison",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ]
        _dcols = st.columns(4)
        for _col, (_fname, _label, _mime) in zip(_dcols, _docmap):
            _fpath = os.path.join("outputs", _fname)
            with _col:
                if os.path.exists(_fpath):
                    with open(_fpath, "rb") as _fh:
                        st.download_button(_label, data=_fh.read(), file_name=_fname, mime=_mime,
                                           use_container_width=True, key=f"dl_{_fname}")
                else:
                    st.caption(f"⏳ {_label} — re-run pipeline")
        st.markdown('<hr style="margin:6px 0;border-color:#1e2436">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if report:
                st.download_button("Download Model Report (.txt)", data=report,
                                   file_name=f"{run_id}_model_report.txt",
                                   mime="text/plain")
        with col2:
            st.download_button("Download Audit Trail (.json)",
                               data=json.dumps(data, indent=2, default=str),
                               file_name=f"{run_id}_audit_trail.json",
                               mime="application/json")
        with col3:
            cps = load_checkpoints()
            if cps:
                st.download_button("Download Checkpoints (.json)",
                                   data=json.dumps(cps, indent=2),
                                   file_name="checkpoints.json",
                                   mime="application/json")

        st.subheader("Assumptions")
        st.write("""
- Binary default: Charged Off = 1, Fully Paid = 0
- Ambiguous statuses excluded from training
- Median imputation for missing numeric features
- WOE encoding computed on training data only
- Post-origination fields excluded to prevent leakage
        """)

        st.subheader("Limitations")
        st.write("""
- Trained on Lending Club data — may not generalise to all portfolios
- High missing rates in bureau features limit completeness
- Payment/recovery fields excluded — reduces available information
- Model not calibrated for probability estimation
        """)

        st.subheader("Risks & Caveats")
        st.write("""
- DTI and income are self-reported — subject to misrepresentation
- Model performance should be re-evaluated quarterly via PSI
- Adverse action codes must be reviewed before customer deployment
- Grade/subgrade may reflect platform bias
- Regulatory review required before production deployment
        """)

        if report:
            st.subheader("Full Model Report")
            with st.expander("View report"):
                st.text(report)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔁 Audit Trail":
    st.title("Audit Trail")
    st.write("Complete log of all agent actions, decisions, warnings, and errors.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    if not data:
        no_data()
    else:
        st.subheader("Run Metadata")
        c1, c2, c3 = st.columns(3)
        c1.metric("Run ID",    data.get("run_id", "—"))
        c2.metric("Dataset",   data.get("dataset_name", "—"))
        c3.metric("Champion",  data.get("champion_model_name", "—"))

        audit_log = data.get("audit_log", [])
        st.subheader(f"Agent Actions ({len(audit_log)} entries)")
        if audit_log:
            adf = pd.DataFrame(audit_log)
            agents = sorted(adf["agent"].unique().tolist()) if "agent" in adf.columns else []
            sel = st.multiselect("Filter by agent", agents)
            if sel:
                adf = adf[adf["agent"].isin(sel)]
            show_df(adf, use_container_width=True, hide_index=True, height=350)
        else:
            st.info("No audit log entries.")

        warnings = data.get("warnings", [])
        if warnings:
            st.subheader("Warnings")
            for w in warnings:
                st.warning(f"[{w.get('agent','')}] {w.get('message','')}")

        errors = data.get("errors", [])
        st.subheader("Errors")
        if errors:
            for e in errors:
                st.error(f"[{e.get('agent','')}] {e.get('message','')}")
        else:
            st.success("No errors recorded.")

        cps = load_checkpoints()
        if cps.get("override_audit"):
            st.subheader("Override Audit Log")
            show_df(pd.DataFrame(cps["override_audit"]),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Checkpoint 1":
    st.title("Checkpoint 1 — Target Definition Review")
    st.warning("**Human Review Required** — Confirm the target variable definition "
               "before feature engineering and modelling begins.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    target_def = data.get("target_definition", "Binary default flag: 1 = Charged Off, 0 = Fully Paid")
    leakage    = data.get("leakage_columns", [])

    st.subheader("Current Target Definition")
    st.write(target_def)

    if leakage:
        st.subheader("Leakage Columns Flagged")
        st.write(", ".join(leakage))

    cps = load_checkpoints()
    cp1 = cps.get("cp1", {})
    if cp1.get("decision"):
        if cp1["decision"] == "approved":
            st.success(f"Already approved — {cp1.get('timestamp','')}")
        else:
            st.error(f"Rejected — {cp1.get('timestamp','')}")

    st.subheader("Analyst Input")
    notes = st.text_area("Analyst notes",
                         value=cp1.get("analyst_notes", ""),
                         placeholder="Add observations or concerns about the target definition…",
                         height=100)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Approve", type="primary"):
            save_checkpoint("cp1", {"decision": "approved", "analyst_notes": notes,
                                    "target_definition": target_def})
            st.success("Checkpoint 1 approved.")
            st.rerun()
    with col2:
        if st.button("✗ Reject"):
            save_checkpoint("cp1", {"decision": "rejected", "analyst_notes": notes})
            st.error("Checkpoint 1 rejected.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 2
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Checkpoint 2":
    st.title("Checkpoint 2 — Feature Shortlist Approval")
    st.warning("**Human Review Required** — Review the selected features before "
               "model training begins.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    selected = data.get("selected_features", [])
    dqr      = data.get("dqr_report", {})

    st.subheader("Selection Rationale")
    st.write(dqr.get("variable_selection_rationale") or "No rationale available.")

    st.subheader(f"Feature Shortlist ({len(selected)} features)")
    if selected:
        removed = []
        for feat in selected:
            if not st.checkbox(feat, value=True, key=f"cp2_{feat}"):
                removed.append(feat)
    else:
        st.info("No selected features available.")
        removed = []

    cps = load_checkpoints()
    cp2 = cps.get("cp2", {})
    if cp2.get("decision"):
        if cp2["decision"] == "approved":
            st.success(f"Already approved — {cp2.get('timestamp','')}")
        else:
            st.error(f"Rejected — {cp2.get('timestamp','')}")

    notes = st.text_area("Analyst notes",
                         value=cp2.get("analyst_notes", ""),
                         placeholder="Add observations about feature selection…",
                         height=80)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Approve", type="primary"):
            approved = [f for f in selected if f not in removed]
            save_checkpoint("cp2", {
                "decision":          "approved",
                "analyst_notes":     notes,
                "approved_features": approved,
                "removed_features":  removed,
            })
            st.success(f"Approved {len(approved)} features. Removed: {removed or 'none'}")
            st.rerun()
    with col2:
        if st.button("✗ Reject"):
            save_checkpoint("cp2", {"decision": "rejected", "analyst_notes": notes})
            st.error("Checkpoint 2 rejected.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT 3
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Model Sign-Off":
    st.title("Model Sign-Off")
    st.warning("**Final Governance Gate** — Review validation results and provide "
               "your deployment decision. This decision is recorded in the audit trail.")

    if not st.session_state.has_run:
        st.info("Run the pipeline first to see results.")
        st.stop()

    show_run_context(data)

    mm      = data.get("model_metrics", {})
    champ   = data.get("champion_model_name", "—")
    champ_m = mm.get(champ, {})
    psi     = data.get("psi_results", {})
    passed  = data.get("validation_passed", False)

    st.subheader("Validation Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Champion",   champ)
    c2.metric("AUC",        champ_m.get("auc_test", "—"))
    c3.metric("KS",         champ_m.get("ks", "—"))
    c4.metric("Gini",       champ_m.get("gini", "—"))
    c5.metric("PSI",        psi.get("psi_score", "—"))

    st.subheader("Validation Summary")
    st.write(data.get("validation_summary") or "No summary available.")

    cps = load_checkpoints()
    cp3 = cps.get("cp3", {})
    if cp3.get("decision"):
        st.info(f"Previous decision: **{cp3['decision']}** by {cp3.get('analyst_name','')} "
                f"on {cp3.get('timestamp','')}")

    st.subheader("Governance Decision")
    analyst_name = st.text_input("Analyst name",
                                 value=cp3.get("analyst_name", ""),
                                 placeholder="Your full name")
    decision     = st.radio("Deployment decision",
                            ["Approve for Deployment", "Request Changes", "Reject"],
                            horizontal=True)
    notes        = st.text_area("Sign-off notes",
                                value=cp3.get("sign_off_notes", ""),
                                placeholder="Document your reasoning, conditions, or concerns…",
                                height=100)

    if st.button("Submit Governance Decision", type="primary"):
        if not analyst_name.strip():
            st.error("Enter your name before submitting.")
        elif not notes.strip():
            st.error("Sign-off notes are required.")
        else:
            save_checkpoint("cp3", {
                "decision":          decision,
                "analyst_name":      analyst_name.strip(),
                "sign_off_notes":    notes.strip(),
                "champion_model":    champ,
                "auc":               champ_m.get("auc_test"),
                "ks":                champ_m.get("ks"),
                "gini":              champ_m.get("gini"),
                "validation_passed": passed,
            })
            if decision == "Approve for Deployment":
                st.success(f"Model approved for deployment by {analyst_name}.")
            elif decision == "Request Changes":
                st.warning(f"Changes requested by {analyst_name}.")
            else:
                st.error(f"Model rejected by {analyst_name}.")
            st.rerun()
