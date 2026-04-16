from template.dashboard import update_halftime_score_scatter, update_prediction_model_chart
fig1, text1 = update_halftime_score_scatter("UEFA Euro", "all")
print("Fig1 data length:", len(fig1.data) if hasattr(fig1, 'data') else "No data")
if hasattr(fig1, 'data'):
    for trace in fig1.data:
        print(f"  Trace: {trace.name}, points: {len(trace.x) if trace.x is not None else 0}")
fig2 = update_prediction_model_chart("UEFA Euro", "all")
print("Fig2 data length:", len(fig2.data) if hasattr(fig2, 'data') else "No data")
if hasattr(fig2, 'data'):
    for trace in fig2.data:
        print(f"  Trace: {trace.name}, points: {len(trace.x) if trace.x is not None else 0}")
