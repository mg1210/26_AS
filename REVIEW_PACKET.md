# Credit Risk Factory — External Review Packet
> Self-contained context for an external LLM review. You do not need any prior conversation.
> Generated 2026-07-06.

---

## 0. What we want from you (reviewer)
This is a hackathon project: an **Agentic Credit Risk Factory** that autonomously develops, validates, explains, and documents a credit-risk model for any given dataset, including a **blind evaluation dataset (Dataset 2)** that may be **headerless**. We'd like feedback on:

1. **Generalization / robustness** — will this actually work on an unseen dataset with different columns, values, or no header row? Where will it break?
2. **Correctness** — any logic bugs in the agents (target detection, WOE/IV, leakage handling, PSI/CSI, champion selection)?
3. **ML soundness** — is the modelling/validation methodology defensible for a credit scorecard (leakage control, overfit penalty, stability metrics, fairness)?
4. **Architecture** — is the agent/orchestrator/state design clean and extensible? Anything over- or under-engineered?
5. **Governance / HITL** — are the human-in-the-loop checkpoints and audit trail adequate for a regulated credit-model context?
6. **Priorities** — given limited time, what are the highest-value fixes/improvements?

Open questions we already know about are in **§7 Known Limitations**.

---

## 1. Project overview
- **Goal:** given a new dataset, a pipeline of specialist agents autonomously runs Observe → Learn → Explain → Act and produces a validated, documented credit-risk model.
- **Stack:** Python 3.14 (Windows), Streamlit UI, XGBoost / LightGBM / scikit-learn, SHAP, Optuna, Plotly, pandas 3.0, python-docx, Anthropic API (optional, for narrative text — pipeline degrades gracefully without a key).
- **Evaluation criteria (hackathon):** autonomy, explainability, human-AI collaboration, governance, adaptability to a blind Dataset 2.

## 2. Architecture — orchestrator + 8 specialist agents
A single `PipelineState` dataclass (`core/state.py`) is passed through every agent and is the single source of truth. Each agent subclasses `BaseAgent` (`core/base_agent.py`) and returns a structured response via `build_response()`.

| Phase | Agent | File | Responsibility |
|---|---|---|---|
| — | Orchestrator | `orchestrator.py` | Runs agents in sequence, 3 HITL checkpoints |
| 1 | Data Understanding | `agents/data_understanding_agent.py` | Load (header-aware), auto-detect target, classify columns, structural + behavioral leakage scan, semantic types, data-dictionary/LLM/name meaning inference |
| 2 | Data Quality Review | `agents/dqr_agent.py` | Missing values, outliers, duplicates, DQR flags, distribution profiles |
| 3 | Feature Engineering | `agents/feature_engineering_agent.py` | 20+ derived features, WOE encoding, imputation (pandas CoW-safe) |
| 4 | Variable Selection | `agents/variable_selection_agent.py` | IV/WOE, correlation filter, RF importance, shortlist |
| 5 | Model Development | `agents/model_development_agent.py` | 5 models: LR, RF, XGBoost+Optuna, LightGBM, GradientBoosting; champion selection |
| 6 | Explainability | `agents/explainability_agent.py` | SHAP, adverse-action codes, what-if, fairness |
| 7 | Validation | `agents/validation_agent.py` | AUC/KS/Gini/PSI/CSI, GREEN/AMBER/RED rating |
| 8 | Documentation | `agents/documentation_agent.py` | Word (.docx) + text report, audit-trail JSON |

**UI:** `ui/app.py` (single-file Streamlit app, ~2930 lines) — dark theme, multi-page, Plotly charts, per-page HITL controls, 3 formal checkpoint pages, audit trail.

**Core:** `core/state.py` (PipelineState), `core/base_agent.py`, `core/llm.py` (Anthropic wrapper, reads `ANTHROPIC_API_KEY`), `core/recommendation.py`, `core/data_loader.py` (header-aware CSV loader — NEW).

## 3. How to run
```powershell
cd credit_risk_factory
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # optional; narratives blank without it
python main.py --auto --trials 5                       # full pipeline, auto-approve checkpoints
python main.py --auto --trials 5 --dataset data/x.csv  # plug-and-play new dataset
streamlit run ui/app.py                                # UI
```

## 4. Current performance (full dev dataset — Lending Club)
- 104,164 obs after filtering, default rate 19.61%, binary classification.
- Champion: **XGBoost**, AUC **0.7068**, KS **0.2997**, Gini **0.4136**, PSI 0.0098 → **GREEN PASS**.
- 10 of 49 candidate features used (IV threshold 0.02).
- End-to-end wall clock ≈ **187s** (ModelDevelopment ≈ 146s of that).

## 5. Governance / HITL
- **3 HITL checkpoints** that halt the pipeline: after Phase 1 (target definition), Phase 4 (feature shortlist), Phase 7 (validation sign-off).
- Per-page review controls: Approve / Approve-with-modifications / Reject / Override, persisted to `outputs/checkpoints.json`.
- **HOTL** (human-over-the-loop) documented as a governance concept (kill-switch) — not a runtime control in this build.
- Every agent action is logged to an audit trail (`outputs/RUN_*_audit_trail.json`).

## 6. Recent changes (2026-07-06 session)
1. **UI table alignment** — replaced dead global CSS (targeted `td`/`th`, which the canvas-based `st.dataframe` grid never renders) with canvas-native `column_config` alignment, applied centrally via a `show_df()` wrapper to all ~31 tables. Leading S.No/#/Rank columns centred, text columns left-aligned.
2. **Variable-selection count consistency** — verified Evidence metric and Observations both read `selected_features`; removed dead code that computed a phantom intermediate count.
3. **Headerless / blind-data support (NEW `core/data_loader.py`)**:
   - `smart_read_csv()` detects headerless CSVs (dtype-fraction heuristic robust to sparse data + `csv.Sniffer` fallback) and names columns `col_0, col_1, …`.
   - Shared by the pipeline **and** the UI so column names always match.
   - `PipelineState.headerless` flag serialized to the audit trail; UI shows a warning banner.
   - **Value-based target detection** (new Priority 3): scores columns by credit-outcome vocabulary so it picks the real target (e.g. loan_status) over lookalikes (e.g. grade) when there are no header names. HITL override remains the safety net.
   - UI fairness proxies now derived dynamically from `categorical_columns` (was a hardcoded Lending-Club column list).
   - Verified: full 8-agent pipeline runs end-to-end on a headerless CSV, target correctly resolved by value.
4. **Streamlit duplicate-widget crash fix** — the live pipeline-log used `text_area` inside a loop (widget IDs collide); switched to `st.code` (display element).
5. **Execution timing** — Home page shows total pipeline time, wall-clock, and per-phase breakdown; progress bar shows live elapsed seconds.
6. **Performance** — `GradientBoosting_AutoML` was wrapped in `GridSearchCV(cv=3)` over a single hyperparameter combo (3 wasted fits whose CV score was never used for champion selection); replaced with a direct fit. XGBoost Optuna timeout 60s→45s. Model quality unchanged (still GREEN), ModelDevelopment ~161–225s → ~146s.

## 7. Known limitations / open questions (please scrutinize)
1. **Target auto-detection on blind data** is heuristic. Value-based scoring helps, but a headerless dataset with a numeric 0/1 target and no name hints would still rely on the HITL override. Is the vocabulary/scoring robust enough?
2. **ID-column leakage on headerless data** — the ID-column filter matches by NAME (`Record_No`, `id`, `member_id`), so on headerless data identifier columns (now `col_0`, `col_2`, …) can slip into the feature set. Needs a value/structure-based check (near-unique or monotonic sequential). Currently partly mitigated by the behavioral-leakage AUC scan + HITL.
3. **ModelDevelopment still dominates runtime** (~146s / ~187s). Further speedups (lighter RF, shorter XGBoost tuning, subsampled training, dropping the redundant GradientBoosting model) trade some quality/robustness.
4. **Stale duplicate files** — `agents/state.py` and `agents/llm.py` appear to be old duplicates of `core/state.py` and `core/llm.py`; the agents import from `core.*`. Should be removed to avoid confusion.
5. **Selection bias** — dataset is funded-loans only (survivorship); documented as a caveat, not corrected.
6. **`ui/app.py` is a single ~2930-line file** — maintainable for a hackathon but a candidate for modularization.
7. **Champion selection** uses `auc_test − max(0, overfit − 0.03) × 2`. Is this penalty defensible, or should it use CV / an OOT fold?

## 8. File map (source only; data/ and outputs/ excluded from the bundle)
```
credit_risk_factory/
├── main.py                     (69)    CLI entry point
├── orchestrator.py             (264)   sequence + 3 HITL checkpoints
├── requirements.txt
├── PLAN.md                             project plan / context
├── REVIEW_PACKET.md                    this file
├── core/
│   ├── state.py                (231)   PipelineState + JSON serializer
│   ├── base_agent.py           (74)    BaseAgent, build_response()
│   ├── llm.py                  (66)    Anthropic wrapper
│   ├── recommendation.py       (39)
│   └── data_loader.py          (105)   header-aware smart_read_csv (NEW)
├── agents/
│   ├── data_understanding_agent.py (703)
│   ├── dqr_agent.py            (274)
│   ├── feature_engineering_agent.py (352)
│   ├── variable_selection_agent.py (307)
│   ├── model_development_agent.py (339)
│   ├── explainability_agent.py (233)
│   ├── validation_agent.py     (406)
│   ├── documentation_agent.py  (446)
│   ├── state.py                (210)   STALE duplicate — see §7.4
│   └── llm.py                  (85)    STALE duplicate — see §7.4
└── ui/
    └── app.py                  (2930)  Streamlit UI
```

## 9. Design constraints (self-imposed, for context)
- Never hardcode field names, target values, or column meanings — must work on any dataset.
- pandas 3.0 / Copy-on-Write safe — no `inplace=True`; use `df[col] = df[col].fillna(...)`.
- Windows paths via `os.path.join`.
- Cast with `pd.to_numeric(errors="coerce")` before math.
- All logs/metrics grounded in actual computed data — no placeholders.
