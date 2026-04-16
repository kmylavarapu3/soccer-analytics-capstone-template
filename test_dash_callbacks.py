import json
import plotly
from template.dashboard import update_prediction_summary_cards, update_prediction_model_chart, get_prediction_features_df

print("Testing with None, None:")
print(update_prediction_summary_cards(None, None))
print("Testing with 'all', 'all':")
print(update_prediction_summary_cards("all", "all"))
print("Testing with 'La Liga', 'all':")
print(update_prediction_summary_cards("La Liga", "all"))

fig = update_prediction_model_chart("all", "all")
if isinstance(fig, plotly.graph_objs.Figure):
    print("Chart created successfully")
else:
    print("Chart failed?", fig)
