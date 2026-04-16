from template.dashboard import filter_prediction_features, compute_prediction_r2, HALFTIME_LIVE_FEATURES, compute_halftime_failure_rate, all_competitions, all_seasons

for c in all_competitions:
    df = filter_prediction_features(c, "all")
    r2_xg = compute_prediction_r2(df, ["xg_diff"])
    if r2_xg is not None and abs(r2_xg - 0.311) < 0.005:
        print(f"Match competition '{c}': xG {r2_xg:.3f}")

for s in all_seasons:
    df = filter_prediction_features("all", s)
    r2_xg = compute_prediction_r2(df, ["xg_diff"])
    if r2_xg is not None and abs(r2_xg - 0.311) < 0.005:
        print(f"Match season '{s}': xG {r2_xg:.3f}")
