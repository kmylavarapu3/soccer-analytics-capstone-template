# Soccer Analytics Capstone — Trilemma Foundation

**Georgia Tech MSA Spring 2026 Practicum**  
**Author**: Karthik Mylavarapu (kmylavarapu3)

## Project Overview
Goal: build an open-source soccer analytics pipeline that combines StatsBomb event data and Polymarket market data to support an interactive dashboard and a systematic pre-game / halftime prediction workflow.

**Track 2: Soccer Analytics Dashboard**

> [!IMPORTANT]
> **License Notice**: The code in this repository is licensed under MIT. However, the data sources (StatsBomb and Polymarket) are not covered by the MIT license and have their own licensing terms. See the [Data Licensing](#data-licensing) section below.

## For reviewers (submission bundle)

- **Written report (PDF):** `Reports/Final_Report_kmylavarapu3.pdf` (duplicate build: `Reports/Final_Report_kmylavarapu3_refreshed.pdf`; editable source: `Reports/FINAL_REPORT_SOURCE.md`).
- **Slides:** `Final_Presentation_kmylavarapu3.pptx` (light text on dark theme; run `python format_final_presentation.py` after manual edits if colors need resetting).
- **Public / industry deliverable:** `template/dashboard.py` — six tabs including **Prediction Lab**; requires `eda/output/match_features.parquet` from `python eda/predictive_modeling.py` first.
- **Verification:** `python -m pytest tests/` (6 tests).

## Key Results
- **Pre-match ELO baseline:** `R² = 0.242`
- **Full-match xG baseline:** `R² = 0.477`
- **Full-match tactical model:** `R² = 0.582`
- **Halftime score baseline:** `R² = 0.542`
- **Halftime live model:** `R² = 0.610`
- **Halftime leader failure rate:** `22.4%`
- **Balanced halftime upset tree:** `0.651` balanced accuracy, `0.902` collapse recall
- **Market linkage:** `2,781` candidate matches, model-only Brier score `0.165`

## Repository Structure
```
├── data/
│   ├── Statsbomb/          # StatsBomb match, event, lineup data (Parquet)
│   ├── Polymarket/         # Polymarket market, trade, odds data (Parquet)
│   └── download_data.py    # Data download script
├── eda/
│   ├── EDA.ipynb                 # Technical EDA notebook
│   ├── EDA_Executive.ipynb       # Executive summary notebook
│   ├── eda_starter_template.py   # Starter EDA script
│   ├── entity_resolution.py      # StatsBomb ↔ Polymarket team mapping
│   ├── feature_engineering.py    # Advanced metrics (PPDA, xT, Field Tilt)
│   ├── predictive_modeling.py    # Multi-feature OLS and outcome prediction
│   ├── market_comparison.py      # xG vs Polymarket odds analysis
│   ├── market_analysis.py        # Market efficiency and volume analysis
│   └── render_report_pdf.py      # Render Reports/FINAL_REPORT_SOURCE.md to PDF
├── Reports/
│   ├── Final_Report_kmylavarapu3.pdf
│   ├── Final_Report_kmylavarapu3_refreshed.pdf
│   ├── FINAL_REPORT_SOURCE.md
│   └── FINAL_REPORT_REFRESH.md
├── template/
│   ├── dashboard.py              # Enhanced interactive dashboard (6 tabs incl. Prediction Lab)
│   └── dashboard_template.py     # Original template dashboard
├── tests/
│   ├── conftest.py
│   └── test_timestamps.py
├── Final_Presentation_kmylavarapu3.pptx
├── format_final_presentation.py   # Re-apply slide colors/layout (optional, after manual PPT edits)
├── requirements.txt
└── README.md
```

## Data Sources
- **StatsBomb Open Data**: 3,464 matches, 12.2M events across La Liga, Premier League, Serie A, Ligue 1, Bundesliga (CC BY-NC 4.0)
- **Polymarket**: 8,549 soccer betting markets, 1.1M trades, $3B+ total volume (Apr 2025–Jan 2026)

## Analytics Modules

### Entity Resolution (`eda/entity_resolution.py`)
Purpose: map StatsBomb and Polymarket team names using normalization, regex extraction, Jaccard similarity, and manual overrides.

### Feature Engineering (`eda/feature_engineering.py`)
Core metrics:
- **Possession Chains**: Segment matches into possession sequences
- **PPDA** (Passes Per Defensive Action): Pressure intensity metric
- **Field Tilt**: Ball position distribution analysis
- **Expected Threat (xT)**: Progressive passing value estimation
- **Team Style Classification**: Tactical categorization

### Predictive Modeling (`eda/predictive_modeling.py`)
Model ladder:
- **Pre-match ELO baseline**: `R² = 0.242`
- **Baseline full-match model** (`xg_diff` only): `R² = 0.477`
- **Full-match tactical model**: `R² = 0.582`
- **Halftime score baseline**: `R² = 0.542`
- **Halftime live model** (score + first-half momentum metrics): `R² = 0.610`
- Forward feature selection with forward-chaining temporal validation
- Halftime edge analysis for score vs xG disagreement
- Balanced decision tree for comeback / upset-state analysis

### Market Comparison (`eda/market_comparison.py`)
Market analysis:
- Poisson-based probability estimation from xG
- Model vs market calibration curves
- Value bet identification
- Market efficiency metrics by competition and time period
- Candidate match-to-market linkage for 2,781 matches

### Interactive Dashboard (`template/dashboard.py`)
Plotly Dash application with 6 analysis tabs:
1. **Overview**: Competition summaries, xG distributions, top teams/players
2. **Match Analysis**: Game-by-game breakdown with xG flow timelines and shot maps
3. **Team Analytics**: Radar charts, performance vs xG, tactical metrics
4. **Player Analytics**: Individual stats, efficiency, style classification
5. **Tactical Metrics**: PPDA/Field Tilt distributions, possession chain analysis
6. **Prediction Lab**: Model ladder comparison, halftime score/xG diagnostics, and lead-collapse analysis

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/kmylavarapu3/soccer-analytics-capstone-template.git
   cd soccer-analytics-capstone-template
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data**:
   ```bash
   python data/download_data.py
   ```

4. **Run exploratory analysis**:
   ```bash
   python eda/eda_starter_template.py
   ```

5. **Run predictive modeling**:
   ```bash
   python eda/predictive_modeling.py
   ```
   This generates the saved match-level feature file and figures used by the dashboard's Prediction Lab tab.

6. **Launch the interactive dashboard**:
   ```bash
   python template/dashboard.py
   ```
   Then open `http://127.0.0.1:8050` in your browser. Run `predictive_modeling.py` first so `eda/output/match_features.parquet` exists; the **Prediction Lab** tab reloads that file automatically (no need to restart the server after generating it).

7. **Analyze market efficiency**:
   ```bash
   python eda/market_comparison.py
   ```

## Requirements
- Python 3.9+
- Core dependencies: `polars`, `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `plotly`, `dash`
- Data handling: `pyarrow`, `psutil`
- See `requirements.txt` for complete dependency list

## Project Deliverables
- **MIT-licensed GitHub repository** with complete source code
- **Interactive dashboard** with 6 analytical tabs and dynamic filtering
- **Dashboard as the public-facing deliverable**, separate from the formal program report
- **Executive summary notebook** (EDA_Executive.ipynb) for stakeholder communication
- **Technical EDA notebook** (EDA.ipynb) with detailed exploratory analysis
- **Final Report** (PDF) in `Reports/`
- **Refreshed final report PDF** (`Reports/Final_Report_kmylavarapu3_refreshed.pdf`) generated from the editable markdown source
- **Final report source** (`Reports/FINAL_REPORT_SOURCE.md`) for editable, source-backed narrative updates
- **Final Presentation** (PPTX) updated to reflect validated model results

## Key Findings
- xG differential is the strongest single full-match feature
- ELO works as a clean pre-match baseline
- First-half state becomes more informative than pre-match strength once the match begins
- Halftime score and halftime xG disagreement is a useful upset signal
- Polymarket comparison is feasible, but liquid markets remain difficult to beat consistently

## Data Licensing
This project uses data from multiple sources, each with their own licensing terms:

### StatsBomb Data
- **License**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Creative Commons Attribution-NonCommercial 4.0 International)
- **Usage**: Non-commercial use only, attribution required
- **Citation**: "StatsBomb Open Data"
- **Source**: Publicly available match event data

### Polymarket Data
- **Copyright**: © 2026 Polymarket
- **Usage**: Subject to [Polymarket Terms of Service](https://polymarket.com/terms)
- **Restrictions**: For analytical and research purposes only; users responsible for compliance with local laws and regulations
- **Source**: Historical prediction market data provided through Polymarket APIs

> [!WARNING]
> The data in this project is **not covered by the MIT license**. Users must comply with the licensing terms of each respective data provider when using the data for their own projects or analyses.
