# Final Report: Delivering Elite European Football Analytics

**Soccer Analytics Capstone — Trilemma Foundation**  
**Author:** Karthik Mylavarapu (`kmylavarapu3`)  
**GT ID:** 903754308  
**Program:** Georgia Tech Master of Science in Analytics  
**Semester:** Spring 2026 Practicum  

---

## Project Overview

**Goal:** Build an open-source soccer analytics pipeline that ingests public match event data, supports interactive player and team dashboards, and adds a systematic match-outcome prediction workflow (pre-game through halftime).

**Deliverables:** A formal program report (this document) and a public-facing interactive dashboard with a dedicated Prediction Lab.

**Data sources:**

- **StatsBomb:** 3,464 matches and 12.2M events across major European leagues (La Liga, Premier League, Serie A, Ligue 1, Bundesliga) and international competitions.
- **Polymarket:** 8,549 soccer markets with $3B+ total trading volume and 1.1M individual trades in the project window.

**Key challenge:** Team identifiers are not normalized across datasets — StatsBomb team IDs do not map to Polymarket market slugs. Entity resolution is a critical early task, and match–market linkage remains heuristic.

**Track selected:** Track 2 — Soccer Analytics Dashboard (extended with prediction and market comparison modules).

---

## Work Completed: Infrastructure and Data Pipeline

**Repository setup**

- Forked the official template repository and installed dependencies.
- Downloaded and organized StatsBomb and Polymarket datasets; verified integrity with the starter EDA workflow.

**Custom analytics modules (representative scope)**

- `entity_resolution.py` — Fuzzy team-name matching between StatsBomb and Polymarket (normalization, regex extraction from market text, token overlap, manual overrides for major clubs).
- `feature_engineering.py` — Possession chains, xG flow, PPDA, field tilt, expected threat (xT), and team-style heuristics used as inputs to match-level features.
- `market_analysis.py` — Market efficiency metrics, volume analysis, and competition-level summaries.
- `predictive_modeling.py` — Match feature construction, ELO-style team strength, halftime “live state” features, OLS baselines, forward-chaining temporal validation, and a recall-oriented upset tree for halftime leaders.
- `market_comparison.py` — Poisson-style probabilities from xG, linkage to Polymarket markets, and calibration-style comparison where coverage exists.

**Dashboard**

- Extended the Dash application to six tabs, including a Prediction Lab that surfaces the model ladder, halftime score and xG scatter views, and lead-collapse rates.

---

## Work Completed: Exploratory Analysis (Key Findings)

**StatsBomb (event-level)**

- Pass completion rate: 77.7% overall — broadly stable across competitions in the sample.
- 88,023 shots with xG; mean xG per shot = 0.10.
- Average 2.85 goals per match; La Liga shows the highest scoring among the five leagues emphasized in the sample.
- Coverage note: descriptive player-level leaders (for example cumulative xG leaders) appear in the technical EDA notebook and dashboard views.

**Polymarket**

- Champions League–related markets show very large volume concentration (on the order of $1B+ in the broader soccer sample).
- Trading peaks during European evening hours (17:00–22:00 UTC).
- Only about 38% of markets show any trade activity; liquidity is uneven and concentrated.

---

## Work Completed: Predictive Modeling and Validation

**Research questions**

- How much can public event data explain about full-time goal differential and outcomes?
- How much lift comes from tactical metrics beyond a simple xG differential?
- How predictive is the first half (score and advanced metrics) for the final 90-minute result?
- When do halftime leads look unstable relative to underlying performance (xG and momentum proxies)?

**Model ladder (systematic traversal)**

1. Pre-match ELO baseline (team strength from chronological match history).
2. Full-match xG baseline (goal differential ~ xG differential).
3. Full-match tactical OLS (xG plus PPDA, field tilt, pressure, carries, pass completion, shots on target, possession share).
4. Forward-selection OLS (parsimonious tactical subset).
5. Halftime score baseline.
6. Halftime xG baseline.
7. Halftime “live” OLS (halftime score and xG differentials plus shots on target, pass volume, possession, pressure, carries into the final third, pass completion).

**Upset-oriented model**

- Balanced decision tree on matches with a halftime leader, targeting recall for cases where the leader fails to win (unsteady leads).

**Validation**

- Forward-chaining temporal cross-validation and season-to-season stability checks — matches are not treated as exchangeable i.i.d. rows.

**Results (goal differential prediction, aggregated reporting run)**

| Model | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| ELO baseline | 0.2420 | 1.8154 | 1.4013 |
| xG baseline | 0.4770 | 1.5080 | 1.1821 |
| Full-match tactical OLS | 0.5817 | 1.3487 | 1.0542 |
| Forward selection OLS | 0.5800 | 1.3514 | 1.0578 |
| Halftime score baseline | 0.5420 | 1.4112 | 1.0883 |
| Halftime xG baseline | 0.3409 | 1.6929 | 1.3106 |
| Halftime live OLS | 0.6096 | 1.3030 | 1.0123 |

**Interpretation:** xG remains the strongest single explanatory anchor. Tactical metrics add measurable lift beyond xG alone. Halftime state is materially informative for full-time outcomes; the halftime live model is the strongest overall in this ladder. ELO is a credible pre-match baseline but is dominated by in-match information once the first half is observed.

**Halftime edge (illustrative instability rates)**

- Halftime leader failed to win: 458 / 2046 = 22.4%.
- Halftime xG leader failed to win: 1498 / 3464 = 43.2%.
- Halftime score and halftime xG disagreed on who was “ahead” on leader definitions: 391 / 3464 = 11.3%.

These disagreement cases are the most direct operationalization of “unsteady states” in this project: the scoreboard and underlying chance creation do not line up.

**Upset tree (halftime leader subset)**

- Balanced accuracy about 0.65 with recall-oriented weighting; collapse recall about 0.90 in the reported configuration.
- Most important split: absolute halftime lead (`abs_halftime_lead`).

The tree is intentionally not a pure accuracy play; it is a diagnostic for unstable leads.

---

## Work Completed: Market Comparison (Where Coverage Exists)

**Approach**

- Convert match xG into simple Poisson win/draw/loss probabilities.
- Link StatsBomb matches to candidate Polymarket markets and compare implied probabilities.

**Scale**

- Linked candidate matches: 2,781 (heuristic linkage; not ground-truth identity).

**Calibration-style summary**

- Model-only Brier score: 0.165 (reporting run).

**Interpretation:** Event-derived probabilities are informative, but liquid markets are hard to beat consistently. The comparison is most useful as a calibration and coverage diagnostic — especially where markets are thin.

---

## Plan: Remaining Work and Future Extensions

**Near-term analytics**

- Player-level xT contribution rollups and pass-network views tied to the dashboard.
- Stronger market linkage stratified by liquidity buckets.

**Modeling**

- Richer pre-match priors beyond ELO; optional league-specific calibration.
- Live updating features if streaming event feeds become available.

**Product**

- Deployment packaging for the dashboard (configuration, data paths, and a short operator README).

---

## Reproducibility

**Generated artifacts (examples)**

- `eda/output/match_features.parquet`
- `eda/output/05_model_comparison.png`
- `eda/output/06_temporal_stability.png`
- `eda/output/07_upset_tree_importance.png`
- `eda/output/08_upset_tree_rules.txt`
- `figures/market_efficiency_report.json`

**Workflow**

1. Run `eda/predictive_modeling.py` to regenerate features, figures, and console metrics.
2. Launch `template/dashboard.py` after `match_features.parquet` exists.
3. Render this PDF via `eda/render_report_pdf.py`.

---

## Closing

This project demonstrates that open event data can support both an interactive analytics surface and a defensible modeling ladder. The strongest practical edge in this sample comes from halftime state — especially when score and advanced metrics disagree — which matches the course guidance to treat matches as evolving processes rather than static rows.
