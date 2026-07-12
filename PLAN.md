# Agentic Credit Risk Factory — Project Context & Plan
> Dhurin Hackathon 2026 | Observe → Learn → Explain → Act

---

## 1. Project Overview

Multi-agent pipeline that autonomously develops, validates, explains, and documents a credit risk model when given a new dataset. Built on Python 3.14, Windows, using Streamlit UI.

**Challenge:** Design and build an Agentic Credit Risk Factory that can autonomously develop, validate, explain, and document a credit risk model when presented with a new dataset.

**Key evaluation criteria:** Autonomy, explainability, human-AI collaboration, governance, adaptability to Dataset 2 (blind evaluation dataset).

---

## 2. Architecture — 8 Specialist Agents + Orchestrator

| Agent | File | Responsibility |
|---|---|---|
| Orchestrator | `orchestrator.py` | Master controller, 3 HITL checkpoints, runs agents in sequence |
| Agent 1 | `data_understanding_agent.py` | Auto-detect target, classify columns, behavioral leakage scan, semantic types, data dictionary lookup |
| Agent 2 | `dqr_agent.py` | Missing values, outliers, duplicates, DQR flags, distribution profiles |
| Agent 3 | `feature_engineering_agent.py` | 20+ derived features, WOE encoding, imputation (CoW-safe) |
| Agent 4 | `variable_selection_agent.py` | IV/WOE, correlation filter, RF importance, feature shortlist |
| Agent 5 | `model_development_agent.py` | 5 models: LR, RF, XGBoost+Optuna, LightGBM, GBM AutoML |
| Agent 6 | `explainability_agent.py` | SHAP, adverse action codes, What-If analysis, Fairness checks |
| Agent 7 | `validation_agent.py` | AUC/KS/Gini/PSI/CSI, GREEN/AMBER/RED rating |
| Agent 8 | `documentation_agent.py` | Word report (.docx), audit trail (.json), checkpoint log |

---

## 3. Folder Structure

```
credit_risk_factory/
├── main.py                          ← entry point (--auto --trials N --dataset path)
├── orchestrator.py                  ← master controller + 3 HITL checkpoints
├── requirements.txt
├── core/
│   ├── state.py                     ← PipelineState dataclass (shared between all agents)
│   ├── base_agent.py                ← BaseAgent abstract class with build_response()
│   ├── llm.py                       ← Anthropic API wrapper (reads ANTHROPIC_API_KEY env var)
│   └── recommendation.py           ← Recommendation dataclass (title/rationale/confidence/risk)
├── agents/
│   ├── data_understanding_agent.py
│   ├── dqr_agent.py
│   ├── feature_engineering_agent.py
│   ├── variable_selection_agent.py
│   ├── model_development_agent.py
│   ├── explainability_agent.py
│   ├── validation_agent.py
│   └── documentation_agent.py
├── ui/
│   └── app.py                       ← Streamlit UI (dark theme, multi-page, Plotly charts)
├── data/                            ← datasets + Data Dictionary.xlsx
└── outputs/                         ← audit trail JSON, model reports, checkpoints.json
```

---

## 4. How to Run

```powershell
# Navigate to project
cd C:\Users\astha\Downloads\credit_risk_factory

# Set API key (required for LLM narratives)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# Run pipeline
python main.py --auto --trials 5

# Run UI (separate terminal)
streamlit run ui/app.py

# Run with specific dataset
python main.py --auto --trials 5 --dataset data/your_dataset.csv
```

---

## 5. Current Pipeline Performance

| Metric | Value | Status |
|---|---|---|
| Dataset | Lending Club loans | 104,164 obs after filtering |
| Default Rate | 19.61% | Binary classification |
| Champion Model | XGBoost | Selected from 5 candidates |
| AUC | 0.7071 | ✅ GREEN (≥0.70) |
| KS | 0.3011 | ✅ GREEN (≥0.25) |
| Gini | 0.4143 | ✅ GREEN |
| PSI | 0.0098 | ✅ Stable (<0.10) |
| Features Used | 10 of 49 candidates | IV threshold = 0.02 |
| Validation | GREEN PASS | All thresholds met |

---

## 6. Key Design Principles

1. **Generalizability over convenience** — nothing hardcoded, all logic dynamic
2. **HITL (Human-in-the-Loop)** — 3 formal checkpoints + review panel on every page
3. **HOTL (Human-over-the-Loop)** — passive oversight concept, Kill Switch documented
4. **Fix bugs before adding features** — stability first
5. **All logs/metrics grounded in actual computed data** — no placeholders
6. **Plug-and-play for Dataset 2** — pipeline reruns automatically on any new CSV

---

## 7. HITL & HOTL Framework

### HITL Checkpoints (pipeline HALTS until approved)
| Checkpoint | Trigger | What analyst reviews |
|---|---|---|
| Checkpoint 1 | After Phase 1 | Target column, GOOD/BAD/INDET mapping, leakage columns |
| Checkpoint 2 | After Phase 4 | Feature shortlist (IV-ranked), remove/add features |
| Checkpoint 3 | After Phase 7 | Model validation results, formal sign-off |

### Per-Page Human Review (every phase page)
- ✓ Approve
- ⚠ Approve with Modifications
- ✗ Reject (opens inline edit panel)
- ↩ Override AI decision

### HOTL (Human Over the Loop)
- Passive supervisor — does not block pipeline
- Can view dashboard at any time
- Kill Switch concept: would halt pipeline between phases in production
- For hackathon: documented as governance concept, not runtime control

---

## 8. Streamlit UI Structure

### Sidebar Navigation
```
Home
Data Understanding
Data Quality Review
Feature Engineering
Variable Selection
Model Development
Explainability
Validation
Documentation
Audit Trail
HITL Matrix
Checkpoint 1
Checkpoint 2
Checkpoint 3
Reload Latest Results
```

### Each Phase Page Structure
1. **Page title + one-line description**
2. **Observations table** (from agent_responses) — always first
3. **Evidence** (tables + Plotly charts)
4. **HITL controls** where applicable (threshold sliders, overrides)
5. **Human Decision** (Approve/Modify/Reject/Override buttons)

### UI Technical Details
- Dark theme (background #07090e)
- Plotly charts with `template="plotly_dark"`
- Tabs on complex pages: DQR, Model Development, Validation, Explainability
- All dataframes: `height=220`, charts: `height=280`
- Reads results from latest `outputs/*_audit_trail.json` on load
- Results only shown after pipeline run (not auto-loaded on startup)

---

## 9. Features Implemented (Complete List)

### Pipeline / Agents
- [x] Auto target detection — dynamic GOOD/BAD/INDETERMINATE from actual values
- [x] Behavioral leakage detection — univariate AUC ≥ 0.65 flagged
- [x] Structural leakage — 13 post-origination columns auto-removed
- [x] Data dictionary auto-loading from `data/` folder (reads any xlsx)
- [x] Business meanings: data dict → LLM annotation → column name inference
- [x] Semantic type inference per column
- [x] Schema drift detection vs reference (informational only)
- [x] Headerless CSV detection (auto-names col_0, col_1 etc)
- [x] Robust feature engineering (all methods wrapped in try/except)
- [x] Graceful variable selection (auto-lowers IV threshold if too few features)
- [x] 5 candidate models: LR, RF, XGBoost+Optuna, LightGBM, GBM AutoML
- [x] Champion selection: AUC − max(0, overfit−0.03)×2
- [x] SHAP TreeExplainer (2,000 samples)
- [x] Adverse action reason codes
- [x] What-If analysis (LIME-style sliders)
- [x] Fairness & Bias Check per categorical variable
- [x] CSI (Characteristic Stability Index) per feature
- [x] PSI split: earliest 70% vs latest 30% by issue_year
- [x] GREEN/AMBER/RED risk rating with criteria breakdown
- [x] Word report (.docx) + text (.txt) + audit trail (.json)
- [x] Structured Recommendation objects (confidence/risk/requires_approval)
- [x] Selection bias warning (funded-only dataset)

### UI Features
- [x] Pipeline progress bar on Home page
- [x] Feature selection waterfall chart (go.Waterfall)
- [x] IV horizontal bar chart (colored by strength)
- [x] Model comparison grouped bar chart
- [x] SHAP importance bar chart
- [x] Gauge charts for AUC/KS/Gini on Validation page
- [x] Score decile bar chart
- [x] CSI bar chart with color coding
- [x] Dev/OOT split grouped bar chart
- [x] Column distribution explorer (histogram/bar on full dataset)
- [x] Column role breakdown pie chart
- [x] Editable thresholds: AUC, KS, Gini, PSI, IV, correlation (sliders)
- [x] Behavioral leakage HITL: AUC threshold slider + BORDERLINE/LEAKAGE flags + analyst override multiselect
- [x] Missing value cardinality table + imputation override per column
- [x] Hyperparameter review table with assessment + next best action
- [x] Leakage workflow table (Discovery → DQR → Feature Eng → Model Dev)
- [x] HITL Matrix reference page
- [x] Checkpoint pages 1, 2, 3 with full analyst sign-off flow
- [x] Audit trail filterable by agent
- [x] Download: Word report, audit trail JSON, checkpoints JSON
- [x] Recent runs table (last 5 runs)
- [x] Reload Latest Results button

---

## 10. Known Issues (To Fix)

| Issue | Location | Priority |
|---|---|---|
| Variable Selection shows 17 selected in Evidence but 10 in Observations | `ui/app.py` Variable Selection page | High |
| Table alignment: S.No should be center, text should be left | All pages | Medium |
| Agent execution time shown on pages (developer debug info) | All phase pages | Low |
| LLM narratives blank when API key not set | All pages | Low (by design) |

---

## 11. Remaining Hackathon Deliverables

| Deliverable | Status |
|---|---|
| Agentic AI Solution | ✅ Complete |
| Human-in-the-Loop Interface | ✅ Complete |
| Model Development Report | ✅ Auto-generated |
| Dataset 2 Demonstration | 🔲 Not tested yet |
| Solution Architecture Document | 🔲 Needs update |
| Final Presentation | 🔲 Not started |

---

## 12. Dataset 2 Readiness Checklist

| Scenario | Handling | Status |
|---|---|---|
| Different column names | Target auto-detected from name patterns | ✅ |
| Different target values | GOOD/BAD/INDET classified dynamically | ✅ |
| Headerless CSV | Auto-names col_0, col_1 etc + UI warning | ✅ |
| No data dictionary | Falls back to LLM → name inference | ✅ |
| Missing columns (feature eng fails) | try/except on every feature method | ✅ |
| Too few features pass IV | Auto-lowers threshold up to 3 times | ✅ |
| Empty feature list | Model dev guard — logs error, stops gracefully | ✅ |
| Schema drift vs Dataset 1 | Compares vs reference_schema.json | ✅ |
| Different missing patterns | DQR handles dynamically | ✅ |
| Different data types | Imputation handles per dtype | ✅ |

---

## 13. Tech Stack

```
Python 3.14 (Windows)
Streamlit >= 1.28
XGBoost >= 2.0
LightGBM
scikit-learn >= 1.3
SHAP >= 0.43
Optuna >= 3.4
Plotly >= 5.17
pandas >= 2.0
numpy >= 1.24
anthropic >= 0.20
python-docx >= 1.1
joblib >= 1.3
colorama >= 0.4.6
openpyxl >= 3.1
```

---

## 14. Next Steps (Priority Order)

1. **Fix known UI bugs** (Variable Selection count discrepancy, table alignment)
2. **Test Dataset 2** — run pipeline on a new unseen dataset to verify plug-and-play
3. **Update Solution Architecture Document** — reflect everything built
4. **Build Final Presentation deck**
5. **Deploy to Streamlit Cloud or ngrok** for public demo URL
6. **Continue UI improvements page by page** (DQR, Feature Engineering, Model Dev, Explainability, Validation)

---

## 15. Important Notes for Claude Code

- **Never hardcode field names, target values, or column meanings** — must work on any dataset
- **Python 3.14 / pandas CoW** — use `df[col] = df[col].fillna(value)` not `inplace=True`
- **Windows paths** — use `os.path.join()` not hardcoded `/` separators
- **All numeric operations** — always cast with `pd.to_numeric(errors='coerce')` before math
- **Streamlit session state** — always use `st.session_state.setdefault()` for threshold defaults
- **One fix at a time** — confirm each fix works before adding the next
