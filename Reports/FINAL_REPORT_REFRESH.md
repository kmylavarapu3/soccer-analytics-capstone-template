# Final Report Refresh Notes

This file captures the validated results from the current codebase and should be treated as the authoritative source when refreshing `Reports/Final_Report_kmylavarapu3.pdf`.

## Executive Summary Replacement

This project delivers an open-source soccer analytics pipeline that supports both descriptive dashboard exploration and systematic match-outcome prediction before kickoff and at halftime. The pipeline integrates StatsBomb public match-event data (3,464 matches, 12.2M events) with Polymarket soccer market data (8,549 markets, 1.1M trades, $3B+ volume) and now emphasizes a model ladder that moves from simple pre-match baselines toward richer live-state prediction.

Validated headline results:

- Pre-match ELO baseline: `R² = 0.242`
- Full-match xG baseline: `R² = 0.477`
- Full-match tactical model: `R² = 0.582`
- Halftime score baseline: `R² = 0.542`
- Halftime live model: `R² = 0.610`
- Halftime leaders fail to win `22.4%` of the time
- Halftime score and halftime xG disagree on the leader in `11.3%` of matches
- Balanced halftime upset tree: `0.651` balanced accuracy with `0.902` recall on collapses
- Market-linkage pipeline connects `2,781` candidate matches with a model-only Brier score of approximately `0.165`

## Methodology Revisions

### Predictive Modeling

Replace older generic wording with the following framing:

1. Pre-match baseline:
   Use a chronological ELO model to assign team-strength ratings before each match. This captures league/tournament momentum in a lightweight and interpretable way.

2. Full-match explanatory models:
   Use xG differential as the baseline explanatory feature, then extend with tactical context such as PPDA, field tilt, possession share, shots on target, carries into the final third, pressure events, and pass completion.

3. Halftime / live-state modeling:
   Treat halftime as a separate predictive state rather than merely another feature row. Use halftime score, halftime xG, halftime shots on target, halftime pass volume, halftime possession share, halftime pressure, halftime pass completion, and halftime carries into the final third to estimate the final 90-minute result.

4. Upset-state analysis:
   Fit a balanced decision tree on halftime leaders to identify when apparently safe leads are unstable. This directly follows the office-hours recommendation to subset games by lead state and explain upsets / comebacks.

### Validation

Use the following wording in place of any ambiguous CV language:

- Forward-chaining temporal cross-validation was used for the regression models so that training folds only use earlier matches and evaluation folds use later matches.
- Separate season-to-season stability checks were run to understand how xG-based relationships vary across competitions and seasons.

## Results Revisions

### Model Comparison Table

| Model | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| ELO baseline | 0.2420 | 1.8154 | 1.4013 |
| xG baseline | 0.4770 | 1.5080 | 1.1821 |
| Full-match tactical OLS | 0.5817 | 1.3487 | 1.0542 |
| Forward selection OLS | 0.5800 | 1.3514 | 1.0578 |
| Halftime score baseline | 0.5420 | 1.4112 | 1.0883 |
| Halftime xG baseline | 0.3409 | 1.6929 | 1.3106 |
| Halftime live OLS | 0.6096 | 1.3030 | 1.0123 |

### Halftime Edge Findings

- Halftime leader failed to win: `458 / 2046 = 22.4%`
- Halftime xG leader failed to win: `1498 / 3464 = 43.2%`
- Halftime score and halftime xG disagreed on the leader: `391 / 3464 = 11.3%`

### Upset Tree Findings

- Holdout accuracy: `0.5122`
- Balanced accuracy: `0.6508`
- Collapse recall: `0.9022`
- Non-collapse recall: `0.3994`
- Dominant split variable: `abs_halftime_lead`

Interpretation:
The tree is not useful as a plain accuracy maximizer because collapse cases are relatively rare, but it is useful as a recall-oriented diagnostic tool for identifying likely leader-collapse scenarios.

### Market Comparison Findings

- Candidate linked matches: `2781`
- Model-only Brier score: `0.165`

Recommended wording:
Polymarket comparison is now a real pipeline rather than only a conceptual extension, but the linkage remains heuristic and should be presented as candidate match-market alignment rather than perfect ground truth.

## Dashboard Revisions

Replace any references to a 5-tab dashboard with:

- The dashboard now has 6 tabs, including a dedicated `Prediction Lab`.
- `Prediction Lab` surfaces the model ladder, halftime score vs final-result diagnostics, halftime xG vs final-result diagnostics, and halftime lead-collapse analysis.
- The dashboard is the public-facing deliverable, distinct from the formal program report.

## Claims To Avoid

Do not claim any of the following unless the static PDF is regenerated from an updated source:

- Team-style classification is fully K-means-based if the implementation still uses threshold logic.
- Market odds are fully linked at the exact match level without caveat.
- Temporal cross-validation if the wording is not explicitly forward-chaining.
- Old model ranges like `~0.55–0.60` when exact validated values are available.

## Supporting Artifacts

The following generated artifacts now exist and can be cited in an updated report:

- `eda/output/match_features.parquet`
- `eda/output/05_model_comparison.png`
- `eda/output/06_temporal_stability.png`
- `eda/output/07_upset_tree_importance.png`
- `eda/output/08_upset_tree_rules.txt`
- `figures/market_efficiency_report.json`
