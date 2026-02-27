"""
Soccer Analytics Dashboard Template
Interactive Plotly Dash dashboard for visualizing StatsBomb soccer data
"""

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, dcc, html

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"

# Load data
print("Loading data...")
matches_df = pl.read_parquet(STATSBOMB_DIR / "matches.parquet")
events_df = pl.read_parquet(STATSBOMB_DIR / "events.parquet")
lineups_df = pl.read_parquet(STATSBOMB_DIR / "lineups.parquet")

# Prepare data
print("Processing data...")

# Match results
matches_with_results = matches_df.with_columns(
    pl.when(pl.col("home_score") > pl.col("away_score"))
    .then(pl.lit("Home Win"))
    .when(pl.col("away_score") > pl.col("home_score"))
    .then(pl.lit("Away Win"))
    .otherwise(pl.lit("Draw"))
    .alias("result")
)

# Event type counts
event_counts = (
    events_df.group_by("type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(15)
)

# xG data (shots only)
shots_df = events_df.filter(
    (pl.col("type") == "Shot") & (pl.col("shot_statsbomb_xg").is_not_null())
)

# Goals per match
matches_with_goals = matches_df.with_columns(
    (pl.col("home_score") + pl.col("away_score")).alias("total_goals")
)

# Top players by xG
top_players_xg = (
    shots_df.filter(pl.col("player").is_not_null())
    .group_by("player")
    .agg([
        pl.col("shot_statsbomb_xg").sum().alias("total_xg"),
        pl.len().alias("shots"),
    ])
    .sort("total_xg", descending=True)
    .head(10)
)

# Top players by events
top_players_events = (
    events_df.filter(pl.col("player").is_not_null())
    .group_by("player")
    .agg(pl.len().alias("event_count"))
    .sort("event_count", descending=True)
    .head(10)
)

# Get unique teams for filter
all_teams = sorted(
    set(
        matches_df["home_team"].unique().to_list()
        + matches_df["away_team"].unique().to_list()
    )
)

# Initialize Dash app
app = Dash(__name__)

# Centralized THEME configuration
THEME = {
    "colors": {
        "background": "#1A1B3A",  # Dark Indigo - main background
        "card": "#25264A",  # Slightly lighter indigo for cards
        "card_hover": "#2E2F5A",  # Hover state for cards
        "border": "#3A3B5C",  # Subtle border color
        "text": "#FFFFFF",  # Pure White - primary text
        "text_muted": "#BFC0C5",  # Cool Silver - secondary text
        "accent": "#F49D52",  # Tangerine Orange - primary action
        "accent_secondary": "#759ACE",  # Soft Cornflower - secondary action
        "success": "#10b981",  # emerald-500 (kept for consistency)
        "warning": "#f59e0b",  # amber-500 (kept for consistency)
        "danger": "#ef4444",  # red-500 (kept for consistency)
        "grid": "#3A3B5C",  # Grid lines matching border
    },
    "spacing": {
        "xs": "8px",
        "sm": "12px",
        "md": "16px",
        "lg": "24px",
        "xl": "32px",
    },
    "shadows": {
        "sm": "0 2px 4px rgba(0, 0, 0, 0.3)",
        "md": "0 4px 6px rgba(0, 0, 0, 0.4)",
        "lg": "0 8px 16px rgba(0, 0, 0, 0.5)",
    },
    "transitions": "all 0.2s ease",
}

# Custom Plotly template
plotly_template = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor=THEME["colors"]["card"],
        paper_bgcolor=THEME["colors"]["card"],
        font=dict(color=THEME["colors"]["text"], family="DM Sans, sans-serif"),
        xaxis=dict(
            gridcolor=THEME["colors"]["grid"],
            linecolor=THEME["colors"]["border"],
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=THEME["colors"]["grid"],
            linecolor=THEME["colors"]["border"],
            zeroline=False,
        ),
        colorway=[
            THEME["colors"]["accent"],
            THEME["colors"]["accent_secondary"],
            THEME["colors"]["success"],
            THEME["colors"]["warning"],
            THEME["colors"]["danger"],
        ],
    )
)

# Reusable style dictionaries
CARD_STYLE = {
    "backgroundColor": THEME["colors"]["card"],
    "padding": THEME["spacing"]["lg"],
    "borderRadius": "12px",
    "boxShadow": THEME["shadows"]["md"],
    "border": f"1px solid {THEME['colors']['border']}",
    "transition": THEME["transitions"],
}

CARD_STYLE_HOVER = {
    **CARD_STYLE,
    "borderColor": THEME["colors"]["accent"],
    "boxShadow": THEME["shadows"]["lg"],
}

STATS_CARD_STYLE = {
    **CARD_STYLE,
    "className": "stats-card dashboard-card",
}

HEADER_STYLE = {
    "color": THEME["colors"]["text"],
    "marginBottom": THEME["spacing"]["md"],
    "fontSize": "20px",
    "fontWeight": "600",
    "fontFamily": "DM Sans, sans-serif",
}

LABEL_STYLE = {
    "color": THEME["colors"]["text"],
    "marginBottom": THEME["spacing"]["xs"],
    "display": "block",
    "fontWeight": "500",
    "fontFamily": "DM Sans, sans-serif",
}

# App layout
app.layout = html.Div(
    style={
        "backgroundColor": THEME["colors"]["background"],
        "padding": THEME["spacing"]["lg"],
        "minHeight": "100vh",
    },
    children=[
        # Google Fonts
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap",
        ),
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto"},
            children=[
                # Header
                html.Div(
                    style={
                        **CARD_STYLE,
                        "padding": THEME["spacing"]["xl"],
                        "marginBottom": THEME["spacing"]["xl"],
                        "background": f"linear-gradient(135deg, {THEME['colors']['card']} 0%, {THEME['colors']['card_hover']} 100%)",
                    },
                    className="dashboard-header",
                    children=[
                        html.H1(
                            "⚽ Soccer Analytics Dashboard",
                            style={
                                "color": THEME["colors"]["text"],
                                "margin": f"0 0 {THEME['spacing']['sm']} 0",
                                "fontSize": "36px",
                                "fontWeight": "700",
                                "fontFamily": "DM Sans, sans-serif",
                            },
                        ),
                        html.P(
                            "Interactive StatsBomb Data Visualization",
                            style={
                                "color": THEME["colors"]["text_muted"],
                                "margin": "0",
                                "fontSize": "16px",
                                "fontFamily": "DM Sans, sans-serif",
                            },
                        ),
                    ],
                ),
                # Stats Cards
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                        "gap": THEME["spacing"]["lg"],
                        "marginBottom": THEME["spacing"]["xl"],
                    },
                    className="stats-grid",
                    children=[
                        html.Div(
                            style={
                                **STATS_CARD_STYLE,
                                "borderLeft": f"4px solid {THEME['colors']['accent']}",
                            },
                            children=[
                                html.Div(
                                    "Total Matches",
                                    style={
                                        "color": THEME["colors"]["text_muted"],
                                        "fontSize": "14px",
                                        "marginBottom": THEME["spacing"]["xs"],
                                        "fontFamily": "DM Sans, sans-serif",
                                    },
                                ),
                                html.Div(
                                    id="total-matches-stat",
                                    style={
                                        "color": THEME["colors"]["text"],
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "fontFamily": "JetBrains Mono, monospace",
                                    },
                                    className="stats-number",
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                **STATS_CARD_STYLE,
                                "borderLeft": f"4px solid {THEME['colors']['accent_secondary']}",
                            },
                            children=[
                                html.Div(
                                    "Total Events",
                                    style={
                                        "color": THEME["colors"]["text_muted"],
                                        "fontSize": "14px",
                                        "marginBottom": THEME["spacing"]["xs"],
                                        "fontFamily": "DM Sans, sans-serif",
                                    },
                                ),
                                html.Div(
                                    id="total-events-stat",
                                    style={
                                        "color": THEME["colors"]["text"],
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "fontFamily": "JetBrains Mono, monospace",
                                    },
                                    className="stats-number",
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                **STATS_CARD_STYLE,
                                "borderLeft": f"4px solid {THEME['colors']['warning']}",
                            },
                            children=[
                                html.Div(
                                    "Unique Players",
                                    style={
                                        "color": THEME["colors"]["text_muted"],
                                        "fontSize": "14px",
                                        "marginBottom": THEME["spacing"]["xs"],
                                        "fontFamily": "DM Sans, sans-serif",
                                    },
                                ),
                                html.Div(
                                    id="unique-players-stat",
                                    style={
                                        "color": THEME["colors"]["text"],
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "fontFamily": "JetBrains Mono, monospace",
                                    },
                                    className="stats-number",
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                **STATS_CARD_STYLE,
                                "borderLeft": f"4px solid {THEME['colors']['danger']}",
                            },
                            children=[
                                html.Div(
                                    "Avg Goals/Match",
                                    style={
                                        "color": THEME["colors"]["text_muted"],
                                        "fontSize": "14px",
                                        "marginBottom": THEME["spacing"]["xs"],
                                        "fontFamily": "DM Sans, sans-serif",
                                    },
                                ),
                                html.Div(
                                    id="avg-goals-stat",
                                    style={
                                        "color": THEME["colors"]["text"],
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "fontFamily": "JetBrains Mono, monospace",
                                    },
                                    className="stats-number",
                                ),
                            ],
                        ),
                    ],
                ),
                # Filters
                html.Div(
                    style={
                        **CARD_STYLE,
                        "marginBottom": THEME["spacing"]["xl"],
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                                "gap": THEME["spacing"]["lg"],
                            },
                            className="filter-grid",
                            children=[
                                html.Div(
                                    [
                                        html.Label("Competition", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="competition-filter",
                                            options=[
                                                {
                                                    "label": "All Competitions",
                                                    "value": "all",
                                                }
                                            ]
                                            + [
                                                {"label": comp, "value": comp}
                                                for comp in sorted(
                                                    matches_df["competition_name"]
                                                    .unique()
                                                    .to_list()
                                                )
                                            ],
                                            value="all",
                                            searchable=True,
                                            placeholder="Search or select competition...",
                                            clearable=False,
                                            className="custom-dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Season", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="season-filter",
                                            options=[
                                                {"label": "All Seasons", "value": "all"}
                                            ]
                                            + [
                                                {"label": season, "value": season}
                                                for season in sorted(
                                                    matches_df["season_name"]
                                                    .unique()
                                                    .to_list()
                                                )
                                            ],
                                            value="all",
                                            searchable=True,
                                            placeholder="Search or select season...",
                                            clearable=False,
                                            className="custom-dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Team", style=LABEL_STYLE),
                                        dcc.Dropdown(
                                            id="team-filter",
                                            options=[
                                                {"label": "All Teams", "value": "all"}
                                            ]
                                            + [
                                                {"label": team, "value": team}
                                                for team in all_teams
                                            ],
                                            value="all",
                                            searchable=True,
                                            placeholder="Search or select team...",
                                            clearable=False,
                                            className="custom-dropdown",
                                        ),
                                    ]
                                ),
                            ],
                        )
                    ],
                ),
                # Charts Grid
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
                        "gap": THEME["spacing"]["xl"],
                        "marginBottom": THEME["spacing"]["xl"],
                    },
                    className="chart-grid",
                    children=[
                        # Events Distribution
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3("Event Type Distribution", style=HEADER_STYLE),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            figure=px.bar(
                                                event_counts.to_pandas(),
                                                x="type",
                                                y="count",
                                                labels={
                                                    "type": "Event Type",
                                                    "count": "Count",
                                                },
                                                color="count",
                                                color_continuous_scale="Blues",
                                                height=400,
                                            )
                                            .update_layout(
                                                template=plotly_template,
                                                showlegend=False,
                                                margin=dict(l=60, r=20, t=40, b=100),
                                                xaxis_tickangle=-45,
                                            )
                                            .update_traces(
                                                marker=dict(
                                                    line=dict(width=0),
                                                )
                                            ),
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Match Results
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3(
                                    "Match Results Distribution", style=HEADER_STYLE
                                ),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            id="results-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # Second Row Charts
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
                        "gap": THEME["spacing"]["xl"],
                        "marginBottom": THEME["spacing"]["xl"],
                    },
                    className="chart-grid",
                    children=[
                        # xG Distribution
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3("xG Distribution (Shots)", style=HEADER_STYLE),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            figure=px.histogram(
                                                shots_df.to_pandas(),
                                                x="shot_statsbomb_xg",
                                                nbins=30,
                                                labels={
                                                    "shot_statsbomb_xg": "Expected Goals (xG)"
                                                },
                                                color_discrete_sequence=[
                                                    THEME["colors"]["accent"]
                                                ],
                                                height=400,
                                            )
                                            .update_layout(
                                                template=plotly_template,
                                                showlegend=False,
                                                margin=dict(l=60, r=20, t=40, b=70),
                                                xaxis=dict(automargin=True),
                                                yaxis=dict(automargin=True),
                                            )
                                            .update_traces(
                                                marker=dict(
                                                    line=dict(width=0),
                                                )
                                            ),
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Goals by Competition
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3("Goals by Competition", style=HEADER_STYLE),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            id="goals-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # Third Row - Player Analytics
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(500px, 1fr))",
                        "gap": THEME["spacing"]["xl"],
                        "marginBottom": THEME["spacing"]["xl"],
                    },
                    className="chart-grid",
                    children=[
                        # Top Players by xG
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3("Top Players by Expected Goals (xG)", style=HEADER_STYLE),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            figure=px.bar(
                                                top_players_xg.to_pandas(),
                                                x="total_xg",
                                                y="player",
                                                orientation="h",
                                                labels={
                                                    "total_xg": "Total xG",
                                                    "player": "Player",
                                                },
                                                color="total_xg",
                                                color_continuous_scale="Oranges",
                                                height=400,
                                            )
                                            .update_layout(
                                                template=plotly_template,
                                                showlegend=False,
                                                margin=dict(l=200, r=40, t=40, b=60),
                                                yaxis=dict(autorange="reversed"),
                                            ),
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Top Players by Events
                        html.Div(
                            style={
                                **CARD_STYLE,
                                "height": "500px",
                            },
                            className="dashboard-card",
                            children=[
                                html.H3("Most Active Players (Total Events)", style=HEADER_STYLE),
                                dcc.Loading(
                                    type="default",
                                    color=THEME["colors"]["accent"],
                                    children=[
                                        dcc.Graph(
                                            figure=px.bar(
                                                top_players_events.to_pandas(),
                                                x="event_count",
                                                y="player",
                                                orientation="h",
                                                labels={
                                                    "event_count": "Total Events",
                                                    "player": "Player",
                                                },
                                                color="event_count",
                                                color_continuous_scale="Blues",
                                                height=400,
                                            )
                                            .update_layout(
                                                template=plotly_template,
                                                showlegend=False,
                                                margin=dict(l=200, r=40, t=40, b=60),
                                                yaxis=dict(autorange="reversed"),
                                            ),
                                            config={"displayModeBar": False},
                                            style={"height": "400px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # Footer
                html.Div(
                    style={
                        "textAlign": "center",
                        "color": THEME["colors"]["text_muted"],
                        "marginTop": THEME["spacing"]["xl"],
                        "paddingTop": THEME["spacing"]["lg"],
                        "borderTop": f"1px solid {THEME['colors']['border']}",
                        "fontFamily": "DM Sans, sans-serif",
                    },
                    children=[
                        html.P(
                            "© 2026 Trilemma Foundation - Soccer Analytics Capstone | Georgia Tech MSA",
                            style={"margin": "0"},
                        )
                    ],
                ),
            ],
        ),
    ],
)


# Callbacks
@app.callback(
    [
        Output("total-matches-stat", "children"),
        Output("total-events-stat", "children"),
        Output("unique-players-stat", "children"),
        Output("avg-goals-stat", "children"),
    ],
    [
        Input("competition-filter", "value"),
        Input("season-filter", "value"),
        Input("team-filter", "value"),
    ],
)
def update_stats_cards(competition, season, team):
    # Filter matches based on selections
    filtered_matches = matches_df

    if competition != "all":
        filtered_matches = filtered_matches.filter(
            pl.col("competition_name") == competition
        )
    if season != "all":
        filtered_matches = filtered_matches.filter(pl.col("season_name") == season)
    if team != "all":
        filtered_matches = filtered_matches.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        )

    # Total Matches
    total_matches = len(filtered_matches)
    matches_text = f"{total_matches:,}"

    # Total Events (filter events by match_ids)
    if total_matches > 0:
        filtered_match_ids = filtered_matches["match_id"].to_list()
        filtered_events = events_df.filter(pl.col("match_id").is_in(filtered_match_ids))
        total_events = len(filtered_events)
        events_text = f"{total_events:,}"
    else:
        events_text = "0"

    # Unique Players (filter lineups by match_ids)
    if total_matches > 0:
        filtered_match_ids = filtered_matches["match_id"].to_list()
        filtered_lineups = lineups_df.filter(
            pl.col("match_id").is_in(filtered_match_ids)
        )
        unique_players = filtered_lineups["player_name"].n_unique()
        players_text = f"{unique_players:,}"
    else:
        players_text = "0"

    # Avg Goals/Match
    if total_matches > 0:
        filtered_matches_with_goals = filtered_matches.with_columns(
            (pl.col("home_score") + pl.col("away_score")).alias("total_goals")
        )
        avg_goals = filtered_matches_with_goals["total_goals"].mean()
        goals_text = f"{avg_goals:.2f}"
    else:
        goals_text = "0.00"

    return matches_text, events_text, players_text, goals_text


@app.callback(
    Output("results-chart", "figure"),
    [
        Input("competition-filter", "value"),
        Input("season-filter", "value"),
        Input("team-filter", "value"),
    ],
)
def update_results_chart(competition, season, team):
    filtered_df = matches_with_results

    if competition != "all":
        filtered_df = filtered_df.filter(pl.col("competition_name") == competition)
    if season != "all":
        filtered_df = filtered_df.filter(pl.col("season_name") == season)
    if team != "all":
        filtered_df = filtered_df.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        )

    result_counts = filtered_df.group_by("result").agg(pl.len().alias("count"))

    fig = px.pie(
        result_counts.to_pandas(),
        values="count",
        names="result",
        color="result",
        color_discrete_map={
            "Home Win": THEME["colors"]["success"],
            "Away Win": THEME["colors"]["danger"],
            "Draw": THEME["colors"]["warning"],
        },
        height=400,
    )

    fig.update_layout(
        template=plotly_template,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


@app.callback(
    Output("goals-chart", "figure"),
    [
        Input("competition-filter", "value"),
        Input("season-filter", "value"),
        Input("team-filter", "value"),
    ],
)
def update_goals_chart(competition, season, team):
    filtered_df = matches_with_goals

    if competition != "all":
        filtered_df = filtered_df.filter(pl.col("competition_name") == competition)
    if season != "all":
        filtered_df = filtered_df.filter(pl.col("season_name") == season)
    if team != "all":
        filtered_df = filtered_df.filter(
            (pl.col("home_team") == team) | (pl.col("away_team") == team)
        )

    goals_by_comp = (
        filtered_df.group_by("competition_name")
        .agg(pl.col("total_goals").mean().alias("avg_goals"))
        .sort("avg_goals", descending=True)
    )

    fig = px.bar(
        goals_by_comp.to_pandas(),
        x="competition_name",
        y="avg_goals",
        labels={
            "competition_name": "Competition",
            "avg_goals": "Average Goals per Match",
        },
        color="avg_goals",
        color_continuous_scale="Viridis",
        height=400,
    )

    fig.update_layout(
        template=plotly_template,
        showlegend=False,
        margin=dict(l=60, r=80, t=40, b=180),
        xaxis_tickangle=-45,
        xaxis=dict(
            tickmode='linear',
            automargin=True,
        ),
    )

    return fig


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Soccer Analytics Dashboard")
    print("  Starting server at http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
