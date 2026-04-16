"""
Market Comparison Analysis: xG-Based Model vs Polymarket Odds

This module links StatsBomb match event data with Polymarket betting odds to:
- Estimate win/draw/loss probabilities from xG differentials
- Compare model-implied probabilities against market prices
- Identify market inefficiencies and value bets
- Analyze odds reactions to key match events
- Compute calibration metrics for model and market

The core insight: xG-derived probabilities should diverge from market odds in
systematic ways that traders can exploit, especially in early-season or
lower-liquidity markets.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional

# Configure a local matplotlib cache before importing pyplot.
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"
POLYMARKET_DIR = DATA_DIR / "Polymarket"

# Ensure figures directory exists
FIGURES_DIR.mkdir(exist_ok=True)


class XGProbabilityEstimator:
    """
    Convert xG differentials to win/draw/loss probabilities using Poisson approximation.

    Given home and away expected goals, compute the probability distribution over
    final match outcomes using the Poisson probability mass function (PMF).

    The Poisson distribution is a natural fit for soccer goals:
    - Goals are rare events that occur randomly in time
    - The rate parameter (lambda) is the expected goals (xG)
    - P(k goals) = (lambda^k * e^-lambda) / k!
    """

    def __init__(self, max_goals: int = 10):
        """
        Initialize estimator.

        Args:
            max_goals: Maximum number of goals to consider in PMF (0 to max_goals).
                      Higher values increase accuracy but computational cost.
        """
        self.max_goals = max_goals

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """
        Compute Poisson PMF: P(X = k | lambda).

        Args:
            k: Number of successes (goals)
            lam: Rate parameter (expected goals)

        Returns:
            Probability of observing exactly k goals given lambda
        """
        if lam <= 0 or k < 0:
            return 0.0
        try:
            # P(k) = (lambda^k * e^-lambda) / k!
            numerator = np.exp(-lam) * (lam ** k)
            denominator = math.factorial(k)
            return numerator / denominator
        except (OverflowError, ValueError):
            return 0.0

    def estimate_probabilities(
        self, home_xg: float, away_xg: float
    ) -> dict[str, float]:
        """
        Estimate P(home_win), P(draw), P(away_win) from xG values.

        Computes the joint distribution over home and away goal counts,
        then aggregates to match outcomes.

        Args:
            home_xg: Home team expected goals
            away_xg: Away team expected goals

        Returns:
            Dictionary with keys:
                - home_win: Probability home team wins
                - draw: Probability of draw
                - away_win: Probability away team wins
                - distribution: Full goal distribution {(h_goals, a_goals): prob}
        """
        # Clamp xG to reasonable bounds to avoid numerical issues
        home_xg = max(0.01, min(home_xg, 20.0))
        away_xg = max(0.01, min(away_xg, 20.0))

        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        goal_dist = {}

        # Enumerate all possible goal combinations
        for home_goals in range(self.max_goals + 1):
            for away_goals in range(self.max_goals + 1):
                # Joint probability (independence assumption)
                joint_prob = self._poisson_pmf(
                    home_goals, home_xg
                ) * self._poisson_pmf(away_goals, away_xg)

                goal_dist[(home_goals, away_goals)] = joint_prob

                # Accumulate outcome probabilities
                if home_goals > away_goals:
                    home_win_prob += joint_prob
                elif home_goals == away_goals:
                    draw_prob += joint_prob
                else:
                    away_win_prob += joint_prob

        return {
            "home_win": home_win_prob,
            "draw": draw_prob,
            "away_win": away_win_prob,
            "distribution": goal_dist,
        }

    def compute_implied_odds(
        self, home_xg: float, away_xg: float
    ) -> dict[str, float]:
        """
        Convert probabilities to decimal odds (1 / probability).

        Args:
            home_xg: Home team expected goals
            away_xg: Away team expected goals

        Returns:
            Dictionary with decimal odds for each outcome
        """
        probs = self.estimate_probabilities(home_xg, away_xg)

        return {
            "home_win_odds": 1.0 / probs["home_win"] if probs["home_win"] > 0 else float("inf"),
            "draw_odds": 1.0 / probs["draw"] if probs["draw"] > 0 else float("inf"),
            "away_win_odds": 1.0 / probs["away_win"] if probs["away_win"] > 0 else float("inf"),
        }


class MarketMatchLinker:
    """
    Link StatsBomb matches with Polymarket markets.

    Uses entity resolution to find matching markets, extracts pre-match odds,
    and compares model-implied probabilities vs market prices.
    """

    def __init__(self):
        """Initialize with data loaders."""
        self._matches: Optional[pl.DataFrame] = None
        self._events: Optional[pl.DataFrame] = None
        self._markets: Optional[pl.DataFrame] = None
        self._odds_history: Optional[pl.DataFrame] = None
        self._tokens: Optional[pl.DataFrame] = None
        self.estimator = XGProbabilityEstimator()

    @property
    def matches(self) -> pl.DataFrame:
        """Load StatsBomb matches (cached)."""
        if self._matches is None:
            self._matches = pl.read_parquet(STATSBOMB_DIR / "matches.parquet")
        return self._matches

    @property
    def events(self) -> pl.DataFrame:
        """Load StatsBomb events (cached)."""
        if self._events is None:
            self._events = pl.read_parquet(STATSBOMB_DIR / "events.parquet")
        return self._events

    @property
    def markets(self) -> pl.DataFrame:
        """Load Polymarket markets (cached)."""
        if self._markets is None:
            self._markets = pl.read_parquet(POLYMARKET_DIR / "soccer_markets.parquet")
        return self._markets

    @property
    def odds_history(self) -> pl.DataFrame:
        """Load Polymarket odds history (cached)."""
        if self._odds_history is None:
            oh = pl.read_parquet(POLYMARKET_DIR / "soccer_odds_history.parquet")
            self._odds_history = oh.with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
            )
        return self._odds_history

    @property
    def tokens(self) -> pl.DataFrame:
        """Load Polymarket tokens (cached)."""
        if self._tokens is None:
            self._tokens = pl.read_parquet(POLYMARKET_DIR / "soccer_tokens.parquet")
        return self._tokens

    def extract_match_xg(self, match_id: str) -> dict[str, float]:
        """
        Extract total xG for home and away teams for a match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            Dict with 'home_xg' and 'away_xg' keys
        """
        match_info = self.matches.filter(pl.col("match_id") == match_id)
        if match_info.is_empty():
            return {"home_xg": 0.0, "away_xg": 0.0}

        home_team = match_info["home_team"][0]
        away_team = match_info["away_team"][0]

        events = self.events.filter(
            (pl.col("match_id") == match_id)
            & (pl.col("type") == "Shot")
        )

        home_shots = events.filter(
            pl.col("team") == home_team
        )
        away_shots = events.filter(
            pl.col("team") == away_team
        )

        home_xg = home_shots["shot_statsbomb_xg"].fill_null(0).sum()
        away_xg = away_shots["shot_statsbomb_xg"].fill_null(0).sum()

        return {
            "home_xg": float(home_xg) if home_xg is not None else 0.0,
            "away_xg": float(away_xg) if away_xg is not None else 0.0,
        }

    def get_pre_match_odds(
        self, market_id: str, match_date: str, hours_before: float = 24.0
    ) -> dict[str, float]:
        """
        Extract odds closest to match start (but before it starts).

        Args:
            market_id: Polymarket market ID
            match_date: Match date (ISO format)
            hours_before: How many hours before match to look for odds

        Returns:
            Dict with 'price_home_win', 'price_draw', 'price_away_win' (or empty if not found)
        """
        oh = self.odds_history.filter(pl.col("market_id") == market_id)

        if oh.is_empty():
            return {}

        # Get token information
        tokens = self.tokens.filter(pl.col("market_id") == market_id)
        if tokens.is_empty():
            return {}

        # Parse match_date
        try:
            match_dt = pl.datetime_range(
                match_date, match_date, interval="1d", eager=True
            )[0]
        except Exception:
            return {}

        # Find odds just before match
        cutoff = match_dt.replace(hour=0, minute=0, second=0)

        # Get latest price before match
        pre_match = oh.filter(pl.col("timestamp") < cutoff)
        if pre_match.is_empty():
            return {}

        latest = pre_match.sort("timestamp").tail(1)

        if latest.is_empty():
            return {}

        result = {}
        for token_row in tokens.iter_rows(named=True):
            token_id = token_row["token_id"]
            outcome = token_row["outcome"]

            token_price = latest.filter(
                pl.col("token_id") == token_id
            )["price"].first()

            if token_price is not None:
                result[f"price_{outcome}"] = float(token_price)

        return result

    def link_matches_to_markets(self) -> pl.DataFrame:
        """
        Link each StatsBomb match to corresponding Polymarket markets.

        Returns:
            DataFrame with match data, xG, and pre-match odds
        """
        from entity_resolution import EntityResolver

        resolver = EntityResolver()
        mapping = resolver.build_team_mapping()
        pm_teams = resolver.load_polymarket_teams()

        matches = self.matches
        results = []

        for match_row in matches.iter_rows(named=True):
            match_id = match_row["match_id"]
            match_date = match_row["match_date"]
            home_team = match_row["home_team"]
            away_team = match_row["away_team"]

            # Extract xG
            xg = self.extract_match_xg(match_id)

            # Find markets that mention both teams.
            home_matches = mapping.filter(pl.col("sb_team") == home_team)
            away_matches = mapping.filter(pl.col("sb_team") == away_team)

            if home_matches.is_empty() or away_matches.is_empty():
                continue

            home_pm_names = home_matches["pm_team"].unique().to_list()
            away_pm_names = away_matches["pm_team"].unique().to_list()

            home_market_ids = set(
                pm_teams.filter(pl.col("extracted_team").is_in(home_pm_names))["market_id"].to_list()
            )
            away_market_ids = set(
                pm_teams.filter(pl.col("extracted_team").is_in(away_pm_names))["market_id"].to_list()
            )

            shared_market_ids = sorted(home_market_ids & away_market_ids)
            if not shared_market_ids:
                continue

            # Keep a single best candidate market per match using highest volume.
            candidate_markets = self.markets.filter(
                pl.col("market_id").is_in(shared_market_ids)
            ).sort("volume", descending=True)
            if candidate_markets.is_empty():
                continue

            market_id = candidate_markets["market_id"][0]
            market_question = candidate_markets["question"][0]
            model_probs = self.estimator.estimate_probabilities(
                xg["home_xg"], xg["away_xg"]
            )
            pre_match_odds = self.get_pre_match_odds(market_id, str(match_date))

            result_row = {
                "match_id": match_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "market_id": market_id,
                "market_question": market_question,
                "home_xg": xg["home_xg"],
                "away_xg": xg["away_xg"],
                "home_score": match_row["home_score"],
                "away_score": match_row["away_score"],
                "model_home_win_prob": model_probs["home_win"],
                "model_draw_prob": model_probs["draw"],
                "model_away_win_prob": model_probs["away_win"],
                "competition": match_row.get("competition_name", "Unknown"),
            }
            result_row.update(pre_match_odds)
            results.append(result_row)

        if results:
            return pl.DataFrame(results)

        return pl.DataFrame(
            schema={
                "match_id": pl.Utf8,
                "match_date": pl.Utf8,
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
                "market_id": pl.Utf8,
                "market_question": pl.Utf8,
                "home_xg": pl.Float64,
                "away_xg": pl.Float64,
                "home_score": pl.Int64,
                "away_score": pl.Int64,
                "model_home_win_prob": pl.Float64,
                "model_draw_prob": pl.Float64,
                "model_away_win_prob": pl.Float64,
                "competition": pl.Utf8,
            }
        )


class MarketEfficiencyReport:
    """
    Generate comprehensive market efficiency analysis and insights.

    Compares xG-implied model probabilities to market prices, identifies
    value bets, analyzes odds reactions, and computes calibration metrics.
    """

    def __init__(self, comparison_df: Optional[pl.DataFrame] = None):
        """
        Initialize report generator.

        Args:
            comparison_df: DataFrame from MarketMatchLinker.link_matches_to_markets()
        """
        self.comparison_df = comparison_df
        self.estimator = XGProbabilityEstimator()

    def compute_model_vs_market_comparison(
        self, market_odds_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Compare model-implied probabilities vs market prices.

        Args:
            market_odds_df: DataFrame with market odds (must have columns:
                           match_id, price_home_win, price_draw, price_away_win)

        Returns:
            DataFrame with model probs, market probs, and divergence metrics
        """
        if self.comparison_df is None:
            return pl.DataFrame()

        comparison = self.comparison_df.clone()

        # If market odds provided, merge them
        if market_odds_df is not None and not market_odds_df.is_empty():
            comparison = comparison.join(
                market_odds_df,
                on="match_id",
                how="left"
            )

            # Convert market odds to probabilities (1/odds)
            comparison = comparison.with_columns([
                (1.0 / pl.col("price_home_win")).alias("market_home_win_prob"),
                (1.0 / pl.col("price_draw")).alias("market_draw_prob"),
                (1.0 / pl.col("price_away_win")).alias("market_away_win_prob"),
            ])

            # Compute divergence metrics
            comparison = comparison.with_columns([
                (
                    pl.col("model_home_win_prob") - pl.col("market_home_win_prob")
                ).alias("home_win_divergence"),
                (
                    pl.col("model_draw_prob") - pl.col("market_draw_prob")
                ).alias("draw_divergence"),
                (
                    pl.col("model_away_win_prob") - pl.col("market_away_win_prob")
                ).alias("away_win_divergence"),
            ])

        return comparison

    def find_value_bets(
        self, comparison_df: Optional[pl.DataFrame] = None, threshold: float = 0.05
    ) -> pl.DataFrame:
        """
        Identify bets where model diverges significantly from market.

        A value bet exists when:
        - Model probability significantly > market probability (bet against market)
        - Model probability significantly < market probability (don't bet, or bet other outcome)

        Args:
            comparison_df: DataFrame from compute_model_vs_market_comparison()
            threshold: Minimum divergence (in probability) to flag as value

        Returns:
            DataFrame of identified value bets
        """
        if comparison_df is None:
            comparison_df = self.compute_model_vs_market_comparison()

        if comparison_df.is_empty() or "home_win_divergence" not in comparison_df.columns:
            return pl.DataFrame()

        # Find significant divergences
        value_bets = comparison_df.filter(
            (pl.col("home_win_divergence").abs() > threshold)
            | (pl.col("draw_divergence").abs() > threshold)
            | (pl.col("away_win_divergence").abs() > threshold)
        )

        # Compute value magnitude
        if not value_bets.is_empty():
            value_bets = value_bets.with_columns([
                (
                    pl.col("home_win_divergence").abs()
                    + pl.col("draw_divergence").abs()
                    + pl.col("away_win_divergence").abs()
                ).alias("total_divergence"),
            ])

        return value_bets.sort("total_divergence", descending=True)

    def analyze_odds_reaction_to_goals(self) -> pl.DataFrame:
        """
        Analyze how odds change after goals (where data available).

        Returns:
            Summary of odds movements relative to goal events
        """
        if self.comparison_df is None or self.comparison_df.is_empty():
            return pl.DataFrame()

        # Note: This requires integrating with match event data
        # For now, return structure showing which matches had different xG vs final score

        analysis = self.comparison_df.with_columns([
            (pl.col("home_xg") - pl.col("home_score").cast(pl.Float64)).alias("home_xg_diff"),
            (pl.col("away_xg") - pl.col("away_score").cast(pl.Float64)).alias("away_xg_diff"),
        ])

        # Flag matches where xG didn't match actual goals (potential market mispricings)
        analysis = analysis.with_columns([
            (
                (pl.col("home_xg_diff").abs() + pl.col("away_xg_diff").abs()) > 1.0
            ).alias("significant_xg_goal_mismatch"),
        ])

        return analysis

    def compute_brier_scores(self) -> dict[str, float]:
        """
        Compute Brier score calibration metric.

        Brier Score = mean((predicted_probability - actual_outcome)^2)
        Lower is better. 0 = perfect calibration, 0.25 = random guessing.

        For each outcome, actual_outcome is 1 if it occurred, 0 otherwise.

        Returns:
            Dictionary with Brier scores for model, market, and comparison
        """
        if self.comparison_df is None or self.comparison_df.is_empty():
            return {}

        df = self.comparison_df

        # Determine actual outcomes
        df_scored = df.with_columns([
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("home_win_actual"),
            ((pl.col("home_score") == pl.col("away_score")).cast(pl.Int32)).alias("draw_actual"),
            (pl.col("away_score") > pl.col("home_score")).cast(pl.Int32).alias("away_win_actual"),
        ])

        # Compute Brier scores for model
        model_brier_home = (
            ((df_scored["model_home_win_prob"] - df_scored["home_win_actual"]) ** 2).mean()
        )
        model_brier_draw = (
            ((df_scored["model_draw_prob"] - df_scored["draw_actual"]) ** 2).mean()
        )
        model_brier_away = (
            ((df_scored["model_away_win_prob"] - df_scored["away_win_actual"]) ** 2).mean()
        )

        model_brier_overall = (model_brier_home + model_brier_draw + model_brier_away) / 3

        result = {
            "model_brier_home_win": float(model_brier_home),
            "model_brier_draw": float(model_brier_draw),
            "model_brier_away_win": float(model_brier_away),
            "model_brier_overall": float(model_brier_overall),
        }

        # If market odds available, compute market Brier
        if "market_home_win_prob" in df.columns:
            market_brier_home = (
                ((df_scored["market_home_win_prob"] - df_scored["home_win_actual"]) ** 2).mean()
            )
            market_brier_draw = (
                ((df_scored["market_draw_prob"] - df_scored["draw_actual"]) ** 2).mean()
            )
            market_brier_away = (
                ((df_scored["market_away_win_prob"] - df_scored["away_win_actual"]) ** 2).mean()
            )

            market_brier_overall = (
                (market_brier_home + market_brier_draw + market_brier_away) / 3
            )

            result.update({
                "market_brier_home_win": float(market_brier_home),
                "market_brier_draw": float(market_brier_draw),
                "market_brier_away_win": float(market_brier_away),
                "market_brier_overall": float(market_brier_overall),
            })

        return result

    def generate_summary_report(self) -> dict:
        """
        Generate comprehensive summary report of all findings.

        Returns:
            Dictionary with all analysis results
        """
        if self.comparison_df is None or self.comparison_df.is_empty():
            return {"error": "No comparison data available"}

        comparison = self.compute_model_vs_market_comparison()
        value_bets = self.find_value_bets(comparison)
        odds_reactions = self.analyze_odds_reaction_to_goals()
        brier_scores = self.compute_brier_scores()

        report = {
            "summary": {
                "total_matches": len(self.comparison_df),
                "competitions": self.comparison_df["competition"].unique().to_list(),
            },
            "model_performance": brier_scores,
            "value_bet_analysis": {
                "value_bets_found": len(value_bets),
                "pct_of_matches": (
                    len(value_bets) / len(self.comparison_df) * 100
                    if len(self.comparison_df) > 0
                    else 0
                ),
                "avg_divergence": (
                    float(value_bets.select("total_divergence").mean()[0, 0])
                    if not value_bets.is_empty()
                    else 0.0
                ),
            },
            "xg_mismatch_analysis": {
                "matches_with_significant_mismatch": int(
                    odds_reactions.filter(
                        pl.col("significant_xg_goal_mismatch")
                    ).height
                ),
                "pct_of_matches": (
                    odds_reactions.filter(
                        pl.col("significant_xg_goal_mismatch")
                    ).height
                    / len(odds_reactions)
                    * 100
                    if len(odds_reactions) > 0
                    else 0
                ),
            },
        }

        return report


def plot_calibration_curve(
    probs: list[float],
    outcomes: list[int],
    title: str = "Model Calibration Curve",
    bins: int = 10,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot calibration curve: predicted probability vs actual frequency.

    Args:
        probs: List of predicted probabilities
        outcomes: List of actual outcomes (0 or 1)
        title: Plot title
        bins: Number of bins for calibration plot
        figsize: Figure size
    """
    if len(probs) == 0:
        return

    # Bin probabilities
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_outcomes = []
    bin_counts = []

    for i in range(bins):
        mask = (np.array(probs) >= bin_edges[i]) & (np.array(probs) < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_outcomes.append(np.array(outcomes)[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_outcomes.append(0)
            bin_counts.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Calibration curve
    ax.plot(bin_centers, bin_outcomes, "o-", label="Model", linewidth=2, markersize=8)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_vs_market(
    model_probs: list[float],
    market_probs: list[float],
    title: str = "Model vs Market Implied Probabilities",
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot scatter of model probs vs market probs.

    Args:
        model_probs: List of model-implied probabilities
        market_probs: List of market-implied probabilities
        title: Plot title
        figsize: Figure size
    """
    if len(model_probs) == 0:
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(model_probs, market_probs, alpha=0.5, s=50)

    # 45-degree line (perfect agreement)
    ax.plot([0, 1], [0, 1], "r--", label="Perfect Agreement", linewidth=2)

    ax.set_xlabel("Model Probability", fontsize=12)
    ax.set_ylabel("Market Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_vs_market.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_value_bet_distribution(
    value_bets_df: pl.DataFrame,
    title: str = "Value Bet Distribution by Competition",
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot distribution of value bets across competitions.

    Args:
        value_bets_df: DataFrame from MarketEfficiencyReport.find_value_bets()
        title: Plot title
        figsize: Figure size
    """
    if value_bets_df.is_empty():
        return

    # Count by competition
    by_comp = (
        value_bets_df.group_by("competition")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    fig, ax = plt.subplots(figsize=figsize)

    comps = by_comp["competition"].to_list()
    counts = by_comp["count"].to_list()

    ax.barh(comps, counts, color="steelblue")
    ax.set_xlabel("Number of Value Bets", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "value_bet_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_market_efficiency_by_competition(
    comparison_df: pl.DataFrame,
    title: str = "Market Efficiency by Competition",
    figsize: tuple = (12, 7),
) -> None:
    """
    Plot Brier scores and other efficiency metrics by competition.

    Args:
        comparison_df: DataFrame from MarketMatchLinker.link_matches_to_markets()
        title: Plot title
        figsize: Figure size
    """
    if comparison_df.is_empty():
        return

    # Compute Brier by competition
    brier_by_comp = []

    for comp in comparison_df["competition"].unique().to_list():
        comp_df = comparison_df.filter(pl.col("competition") == comp)

        comp_df_scored = comp_df.with_columns([
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("home_win_actual"),
        ])

        model_brier = (
            ((comp_df_scored["model_home_win_prob"] - comp_df_scored["home_win_actual"]) ** 2).mean()
        )

        brier_by_comp.append({
            "competition": comp,
            "brier_score": float(model_brier),
            "match_count": len(comp_df),
        })

    if not brier_by_comp:
        return

    brier_df = pl.DataFrame(brier_by_comp).sort("brier_score")

    fig, ax = plt.subplots(figsize=figsize)

    comps = brier_df["competition"].to_list()
    briers = brier_df["brier_score"].to_list()

    colors = ["green" if b < 0.2 else "orange" if b < 0.25 else "red" for b in briers]
    ax.barh(comps, briers, color=colors)

    ax.set_xlabel("Brier Score (lower = better)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=0.25, color="gray", linestyle="--", label="Random Guessing (0.25)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "market_efficiency_by_competition.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("  Market Comparison Analysis: xG Model vs Polymarket Odds")
    print("=" * 70)

    # Example 1: Test XGProbabilityEstimator
    print("\n--- XG Probability Estimation ---")
    estimator = XGProbabilityEstimator()

    test_cases = [
        (1.5, 0.8, "Home favored (1.5 vs 0.8 xG)"),
        (2.0, 2.0, "Balanced (2.0 vs 2.0 xG)"),
        (0.5, 2.5, "Away favored (0.5 vs 2.5 xG)"),
    ]

    for home_xg, away_xg, description in test_cases:
        probs = estimator.estimate_probabilities(home_xg, away_xg)
        print(f"\n{description}:")
        print(f"  Home Win Probability: {probs['home_win']:.4f}")
        print(f"  Draw Probability:     {probs['draw']:.4f}")
        print(f"  Away Win Probability: {probs['away_win']:.4f}")

    # Example 2: Test MarketMatchLinker
    print("\n--- Market-Match Linking ---")
    try:
        linker = MarketMatchLinker()
        print(f"Loaded {len(linker.matches)} matches")
        print(f"Loaded {len(linker.markets)} Polymarket markets")

        # Try to link matches to markets
        linked = linker.link_matches_to_markets()
        if not linked.is_empty():
            print(f"\nSuccessfully linked {len(linked)} matches")
            print("\nSample linked matches:")
            print(linked.head(3))
        else:
            print("No matches linked (this may be expected if entity mapping is incomplete)")

        # Example 3: Generate efficiency report
        if not linked.is_empty():
            print("\n--- Market Efficiency Report ---")
            report = MarketEfficiencyReport(linked)
            summary = report.generate_summary_report()

            print("\nReport Summary:")
            print(f"  Total Matches: {summary['summary']['total_matches']}")
            print(
                f"  Model Brier Score: {summary['model_performance'].get('model_brier_overall', 'N/A')}"
            )
            print(
                f"  Value Bets Found: {summary['value_bet_analysis']['value_bets_found']}"
            )

            # Save summary to file
            import json

            with open(FIGURES_DIR / "market_efficiency_report.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("\n✓ Report saved to figures/market_efficiency_report.json")

    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("This is expected if data hasn't been downloaded yet.")
    except Exception as e:
        print(f"Error during analysis: {e}")

    print("\n" + "=" * 70)
    print("  Analysis complete!")
    print("=" * 70)
