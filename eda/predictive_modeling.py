"""
Predictive Modeling Module for Soccer Analytics Capstone
Georgia Tech Master of Science in Analytics

Comprehensive module for building and evaluating predictive models of match outcomes
using advanced soccer analytics features (xG, PPDA, field tilt, pressure, etc.).

Features:
- Match feature engineering and aggregation
- Multiple baseline and advanced regression models
- Forward feature selection with cross-validation
- Match outcome classification (Win/Draw/Loss)
- Temporal stability analysis and season-based validation
- Comprehensive evaluation metrics and visualization tools

Uses Polars for data processing and NumPy for linear algebra.
"""

from __future__ import annotations

import os
import warnings
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
from sklearn.tree import DecisionTreeClassifier, export_text

from feature_engineering import MatchMetrics, PossessionChainBuilder, TeamStyleClassifier

warnings.filterwarnings("ignore")

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"
OUTPUT_DIR = Path(__file__).parent / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


class MatchFeatureBuilder:
    """
    Compute per-match features for predictive modeling.

    Builds a feature matrix from raw StatsBomb events, aggregating team-level
    metrics per match (xG, PPDA, field tilt, possession, pressure, etc.).

    Attributes:
        events_df: Polars DataFrame of all match events
        matches_df: Polars DataFrame of match metadata
        features_df: Computed per-match feature matrix
    """

    def __init__(self, events_df: pl.DataFrame, matches_df: pl.DataFrame):
        """
        Initialize feature builder.

        Args:
            events_df: Events from StatsBomb (columns: match_id, type, team, location_x,
                      shot_statsbomb_xg, pass_outcome, etc.)
            matches_df: Match metadata (columns: match_id, match_date, home_team, away_team,
                       home_score, away_score, season_name, etc.)
        """
        self.events = events_df
        self.matches = matches_df
        self.metrics = MatchMetrics(events_df, matches_df)
        self.features_df = None
        self.feature_cols = []
        self.fulltime_feature_cols = []
        self.halftime_feature_cols = []
        self.elo_lookup = {}

    def build_features(self) -> pl.DataFrame:
        """
        Build complete feature matrix for all matches.

        Returns:
            DataFrame with one row per match and columns for:
            - match_id, match_date, home_team, away_team
            - home_score, away_score, goal_diff
            - xg_diff, ppda_diff, field_tilt_diff, possession_share
            - shots_on_target_diff, carries_final_third_diff
            - pressure_events_diff, pass_completion_diff
            - season_name, is_home

        The DataFrame is sorted by match_date.
        """
        features_list = []
        self._build_elo_lookup()
        match_ids = (
            self.matches.sort(["match_date", "kickoff", "match_id"])["match_id"]
            .to_list()
        )

        for match_id in match_ids:
            try:
                feature_row = self._compute_match_features(match_id)
                if feature_row:
                    features_list.append(feature_row)
            except Exception as e:
                print(f"Warning: Failed to compute features for match {match_id}: {e}")
                continue

        if not features_list:
            raise ValueError("No valid features computed from any matches")

        self.features_df = pl.DataFrame(features_list).sort("match_date")
        self._initialize_feature_groups()
        return self.features_df

    def _build_elo_lookup(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 20.0,
        home_advantage: float = 60.0,
    ) -> None:
        """
        Build pre-match ELO features for each match in chronological order.

        Ratings are updated after each match, so the stored values reflect
        only the information available before kickoff.
        """
        ratings = {}
        elo_lookup = {}

        ordered_matches = self.matches.sort(["match_date", "kickoff", "match_id"])
        for row in ordered_matches.iter_rows(named=True):
            match_id = row["match_id"]
            home_team = row["home_team"]
            away_team = row["away_team"]
            home_rating = float(ratings.get(home_team, initial_rating))
            away_rating = float(ratings.get(away_team, initial_rating))

            rating_gap = (home_rating + home_advantage) - away_rating
            expected_home = 1.0 / (1.0 + 10.0 ** (-rating_gap / 400.0))
            expected_away = 1.0 - expected_home

            elo_lookup[match_id] = {
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_diff": home_rating - away_rating,
                "elo_home_expected_score": expected_home,
                "elo_away_expected_score": expected_away,
            }

            goal_diff = row["home_score"] - row["away_score"]
            if goal_diff > 0:
                actual_home = 1.0
            elif goal_diff < 0:
                actual_home = 0.0
            else:
                actual_home = 0.5

            margin_multiplier = (
                1.0
                if goal_diff == 0
                else np.log(abs(goal_diff) + 1.0)
                * (2.2 / ((abs(rating_gap) * 0.001) + 2.2))
            )
            delta = k_factor * margin_multiplier * (actual_home - expected_home)

            ratings[home_team] = home_rating + delta
            ratings[away_team] = away_rating - delta

        self.elo_lookup = elo_lookup

    def _initialize_feature_groups(self) -> None:
        """Group feature columns by full-match vs halftime/live availability."""
        if self.features_df is None:
            return

        exclude_cols = {
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "competition_name",
            "season_name",
            "home_score",
            "away_score",
            "goal_diff",
        }

        self.feature_cols = [
            col for col in self.features_df.columns if col not in exclude_cols
        ]
        self.fulltime_feature_cols = [
            col for col in self.feature_cols if not col.startswith("halftime_")
        ]
        self.halftime_feature_cols = [
            col for col in self.feature_cols if col.startswith("halftime_")
        ]

    def _compute_match_features(self, match_id: int) -> dict:
        """
        Compute all features for a single match.

        Args:
            match_id: Unique match identifier

        Returns:
            Dictionary with per-match features
        """
        match_info = self.matches.filter(pl.col("match_id") == match_id)
        if match_info.is_empty():
            return None

        match_info = match_info.row(0, named=True)
        match_events = self.events.filter(pl.col("match_id") == match_id)
        first_half_events = match_events.filter(pl.col("period") == 1)

        if match_events.is_empty():
            return None

        home_team = match_info["home_team"]
        away_team = match_info["away_team"]
        home_score = match_info["home_score"]
        away_score = match_info["away_score"]

        # Extract xG
        home_xg = self._compute_team_xg(match_events, home_team)
        away_xg = self._compute_team_xg(match_events, away_team)
        elo_features = self.elo_lookup.get(
            match_id,
            {
                "home_elo": 1500.0,
                "away_elo": 1500.0,
                "elo_diff": 0.0,
                "elo_home_expected_score": 0.5,
                "elo_away_expected_score": 0.5,
            },
        )

        # Extract PPDA
        ppda_dict = self.metrics.compute_ppda(match_id)
        home_ppda = ppda_dict.get("home_ppda", 15.0)
        away_ppda = ppda_dict.get("away_ppda", 15.0)

        # Extract field tilt
        tilt_dict = self.metrics.compute_field_tilt(match_id)
        home_tilt = tilt_dict.get("home_field_tilt", 50.0)
        away_tilt = tilt_dict.get("away_field_tilt", 50.0)

        # Extract possession share
        home_poss, away_poss = self._compute_possession_share(
            match_events, home_team, away_team
        )
        halftime_home_poss, halftime_away_poss = self._compute_possession_share(
            first_half_events, home_team, away_team
        )

        # Extract shots on target
        home_sot = self._compute_shots_on_target(match_events, home_team)
        away_sot = self._compute_shots_on_target(match_events, away_team)
        halftime_home_sot = self._compute_shots_on_target(first_half_events, home_team)
        halftime_away_sot = self._compute_shots_on_target(first_half_events, away_team)

        # Extract carries into final third
        home_carries = self._compute_carries_into_final_third(match_events, home_team)
        away_carries = self._compute_carries_into_final_third(match_events, away_team)
        halftime_home_carries = self._compute_carries_into_final_third(
            first_half_events, home_team
        )
        halftime_away_carries = self._compute_carries_into_final_third(
            first_half_events, away_team
        )

        # Extract pressure events
        home_pressure = self._count_event_type(match_events, home_team, "Pressure")
        away_pressure = self._count_event_type(match_events, away_team, "Pressure")
        halftime_home_pressure = self._count_event_type(
            first_half_events, home_team, "Pressure"
        )
        halftime_away_pressure = self._count_event_type(
            first_half_events, away_team, "Pressure"
        )

        # Extract pass completion rate
        home_pass_completion = self._compute_pass_completion(match_events, home_team)
        away_pass_completion = self._compute_pass_completion(match_events, away_team)
        halftime_home_pass_completion = self._compute_pass_completion(
            first_half_events, home_team
        )
        halftime_away_pass_completion = self._compute_pass_completion(
            first_half_events, away_team
        )

        # Halftime score and xG states
        halftime_home_goals = self._compute_goals(first_half_events, home_team)
        halftime_away_goals = self._compute_goals(first_half_events, away_team)
        halftime_home_xg = self._compute_team_xg(first_half_events, home_team)
        halftime_away_xg = self._compute_team_xg(first_half_events, away_team)

        # Halftime passing volume
        halftime_home_passes = self._count_event_type(first_half_events, home_team, "Pass")
        halftime_away_passes = self._count_event_type(first_half_events, away_team, "Pass")

        return {
            "match_id": match_id,
            "match_date": match_info.get("match_date"),
            "home_team": home_team,
            "away_team": away_team,
            "competition_name": match_info.get("competition_name", "Unknown"),
            "season_name": match_info.get("season_name", "Unknown"),
            "home_score": home_score,
            "away_score": away_score,
            "goal_diff": home_score - away_score,
            "home_elo": elo_features["home_elo"],
            "away_elo": elo_features["away_elo"],
            "elo_diff": elo_features["elo_diff"],
            "elo_home_expected_score": elo_features["elo_home_expected_score"],
            "elo_away_expected_score": elo_features["elo_away_expected_score"],
            "xg_diff": home_xg - away_xg,
            "ppda_diff": away_ppda - home_ppda,  # Away PPDA - Home PPDA (negative = home pressing harder)
            "field_tilt_diff": home_tilt - away_tilt,
            "possession_share": home_poss / (home_poss + away_poss) if (home_poss + away_poss) > 0 else 0.5,
            "shots_on_target_diff": home_sot - away_sot,
            "carries_final_third_diff": home_carries - away_carries,
            "pressure_events_diff": home_pressure - away_pressure,
            "pass_completion_diff": home_pass_completion - away_pass_completion,
            "halftime_home_goals": halftime_home_goals,
            "halftime_away_goals": halftime_away_goals,
            "halftime_goal_diff": halftime_home_goals - halftime_away_goals,
            "halftime_home_xg": halftime_home_xg,
            "halftime_away_xg": halftime_away_xg,
            "halftime_xg_diff": halftime_home_xg - halftime_away_xg,
            "halftime_shots_on_target_diff": halftime_home_sot - halftime_away_sot,
            "halftime_carries_final_third_diff": halftime_home_carries - halftime_away_carries,
            "halftime_pressure_events_diff": halftime_home_pressure - halftime_away_pressure,
            "halftime_pass_volume_diff": halftime_home_passes - halftime_away_passes,
            "halftime_pass_completion_diff": halftime_home_pass_completion - halftime_away_pass_completion,
            "halftime_possession_share": (
                halftime_home_poss / (halftime_home_poss + halftime_away_poss)
                if (halftime_home_poss + halftime_away_poss) > 0
                else 0.5
            ),
        }

    def _compute_team_xg(self, match_events: pl.DataFrame, team_name: str) -> float:
        """Compute total expected goals for a team in a match."""
        team_shots = match_events.filter(
            (pl.col("team") == team_name) &
            (pl.col("type") == "Shot")
        )
        return team_shots["shot_statsbomb_xg"].fill_null(0).sum() or 0.0

    def _compute_goals(self, match_events: pl.DataFrame, team_name: str) -> int:
        """Count goals scored by a team from shot outcomes."""
        return match_events.filter(
            (pl.col("team") == team_name)
            & (pl.col("type") == "Shot")
            & (pl.col("shot_outcome") == "Goal")
        ).height

    def _compute_shots_on_target(self, match_events: pl.DataFrame, team_name: str) -> int:
        """Count shots on target (including goals) for a team."""
        team_shots = match_events.filter(
            (pl.col("team") == team_name) &
            (pl.col("type") == "Shot")
        )
        return team_shots.filter(
            pl.col("shot_outcome").is_in(["Goal", "Saved", "Saved to Post"])
        ).height

    def _compute_carries_into_final_third(self, match_events: pl.DataFrame, team_name: str) -> int:
        """Count carries that progress into the opponent's final third."""
        team_carries = match_events.filter(
            (pl.col("team") == team_name) &
            (pl.col("type") == "Carry")
        )
        return team_carries.filter(
            pl.col("carry_end_location_x").fill_null(0) > 80
        ).height

    def _compute_possession_share(
        self, match_events: pl.DataFrame, home_team: str, away_team: str
    ) -> tuple[float, float]:
        """Compute possession share (passes) for both teams."""
        passes = match_events.filter(pl.col("type") == "Pass")
        if passes.is_empty():
            return 0.0, 0.0

        home_passes = passes.filter(pl.col("team") == home_team).height
        away_passes = passes.filter(pl.col("team") == away_team).height
        return home_passes, away_passes

    def _count_event_type(self, match_events: pl.DataFrame, team_name: str, event_type: str) -> int:
        """Count events of a specific type for a team."""
        return match_events.filter(
            (pl.col("team") == team_name) &
            (pl.col("type") == event_type)
        ).height

    def _compute_pass_completion(self, match_events: pl.DataFrame, team_name: str) -> float:
        """Compute pass completion rate for a team."""
        team_passes = match_events.filter(
            (pl.col("team") == team_name) &
            (pl.col("type") == "Pass")
        )
        if team_passes.is_empty():
            return 0.5

        total = team_passes.height
        completed = team_passes.filter(pl.col("pass_outcome").is_null()).height
        return completed / total if total > 0 else 0.5


class MatchOutcomeModel:
    """
    Multiple predictive models for match outcomes (goal difference).

    Implements from-scratch OLS regression with proper cross-validation,
    feature selection, and evaluation metrics without external ML libraries.

    Attributes:
        features_df: Feature matrix with target variable
        target_col: Name of target variable (default: 'goal_diff')
        feature_cols: List of feature column names
        models: Dictionary of fitted model information
    """

    def __init__(self, features_df: pl.DataFrame, target_col: str = "goal_diff"):
        """
        Initialize model.

        Args:
            features_df: DataFrame with features and target variable
            target_col: Name of the target column to predict
        """
        self.features_df = features_df
        self.target_col = target_col
        self.feature_cols = None
        self.fulltime_feature_cols = None
        self.halftime_feature_cols = None
        self.models = {}
        self._initialize_feature_cols()

    def _initialize_feature_cols(self):
        """Extract feature column names (exclude ID, team, date, season columns)."""
        exclude_cols = {
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "competition_name",
            "season_name",
            "home_score",
            "away_score",
            self.target_col,
            "outcome_label",
        }

        self.feature_cols = [
            col for col in self.features_df.columns if col not in exclude_cols
        ]
        self.fulltime_feature_cols = [
            col for col in self.feature_cols if not col.startswith("halftime_")
        ]
        self.halftime_feature_cols = [
            col for col in self.feature_cols if col.startswith("halftime_")
        ]

    def _fit_named_model(self, model_name: str, feature_cols: list[str]) -> dict:
        """Fit and store an OLS model for a specific feature set."""
        X, y = self._prepare_data(feature_cols)
        coeffs, r2, rmse, mae = self._fit_ols(X, y)

        feature_dict = {
            feature_cols[i]: coeffs[i + 1] for i in range(len(feature_cols))
        }

        self.models[model_name] = {
            "features": feature_cols,
            "coefficients": coeffs,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "intercept": coeffs[0],
            "feature_coefficients": feature_dict,
        }
        return self.models[model_name]

    def fit_baseline_ols(self) -> dict:
        """
        Fit baseline OLS model: goal_diff ~ xg_diff

        Returns:
            Dictionary with model coefficients, R², RMSE, MAE
        """
        baseline = self._fit_named_model("baseline_ols", ["xg_diff"])
        baseline["xg_coeff"] = baseline["coefficients"][1]
        return baseline

    def fit_elo_baseline(self) -> dict:
        """Pre-match baseline using only ELO strength differential."""
        return self._fit_named_model("elo_baseline", ["elo_diff"])

    def fit_multifeature_ols(self) -> dict:
        """
        Fit multi-feature OLS model with all available features.

        Returns:
            Dictionary with model coefficients, R², RMSE, MAE
        """
        return self._fit_named_model("multifeature_ols", self.fulltime_feature_cols)

    def fit_halftime_score_baseline(self) -> dict:
        """Predict full-time goal difference from halftime score only."""
        return self._fit_named_model(
            "halftime_score_baseline", ["halftime_goal_diff"]
        )

    def fit_halftime_xg_baseline(self) -> dict:
        """Predict full-time goal difference from halftime xG only."""
        return self._fit_named_model("halftime_xg_baseline", ["halftime_xg_diff"])

    def fit_halftime_live_ols(self) -> dict:
        """
        Predict final goal difference from halftime score plus momentum indicators.

        This is the model that best matches the office-hours guidance around
        in-game prediction and identifying unstable halftime states.
        """
        live_features = [
            "halftime_goal_diff",
            "halftime_xg_diff",
            "halftime_shots_on_target_diff",
            "halftime_pass_volume_diff",
            "halftime_possession_share",
            "halftime_pressure_events_diff",
            "halftime_pass_completion_diff",
            "halftime_carries_final_third_diff",
        ]
        live_features = [
            feature for feature in live_features if feature in self.features_df.columns
        ]
        return self._fit_named_model("halftime_live_ols", live_features)

    def forward_feature_selection(
        self,
        k_best: int = 5,
        feature_pool: Optional[list[str]] = None,
        model_name: str = "forward_selection",
    ) -> dict:
        """
        Forward feature selection to identify most informative features.

        Starts with no features and iteratively adds the feature that most
        improves R² until k_best features are selected.

        Args:
            k_best: Maximum number of features to select

        Returns:
            Dictionary with selected features and model performance
        """
        selected_features = []
        remaining_features = (feature_pool or self.fulltime_feature_cols).copy()

        selection_history = []

        while remaining_features and len(selected_features) < k_best:
            best_feature = None
            best_r2 = -float("inf")

            # Try adding each remaining feature
            for feature in remaining_features:
                test_features = selected_features + [feature]
                X, y = self._prepare_data(test_features)
                _, r2, _, _ = self._fit_ols(X, y)

                if r2 > best_r2:
                    best_r2 = r2
                    best_feature = feature

            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)

                X, y = self._prepare_data(selected_features)
                coeffs, r2, rmse, mae = self._fit_ols(X, y)

                selection_history.append({
                    "step": len(selected_features),
                    "added_feature": best_feature,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                })

        X, y = self._prepare_data(selected_features)
        coeffs, r2, rmse, mae = self._fit_ols(X, y)

        feature_dict = {
            selected_features[i]: coeffs[i + 1] for i in range(len(selected_features))
        }

        self.models[model_name] = {
            "features": selected_features,
            "coefficients": coeffs,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "intercept": coeffs[0],
            "feature_coefficients": feature_dict,
            "selection_history": selection_history,
        }

        return self.models[model_name]

    def fit_outcome_classifier(self) -> dict:
        """
        Fit a match outcome classifier (Win/Draw/Loss for home team).

        Uses logistic regression-like approach: computes class probabilities
        based on xG difference as primary predictor.

        Returns:
            Dictionary with classification accuracy and class boundaries
        """
        # Create outcome labels: 1 = Home Win, 0 = Draw, -1 = Away Win
        outcomes = []
        for row in self.features_df.iter_rows(named=True):
            diff = row.get(self.target_col, 0)
            if diff > 0.5:
                outcomes.append(1)
            elif diff < -0.5:
                outcomes.append(-1)
            else:
                outcomes.append(0)

        self.features_df = self.features_df.with_columns(
            pl.Series("outcome_label", outcomes)
        )

        # Simple classification based on xG difference
        X, y = self._prepare_data(["xg_diff"])
        y_outcomes = np.array(self.features_df["outcome_label"].to_list())

        # Compute class boundaries
        predictions = []
        for pred in X[:, 1]:  # Skip intercept column
            if pred > 0.5:
                predictions.append(1)
            elif pred < -0.5:
                predictions.append(-1)
            else:
                predictions.append(0)

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_outcomes)

        self.models["outcome_classifier"] = {
            "primary_feature": "xg_diff",
            "win_threshold": 0.5,
            "loss_threshold": -0.5,
            "accuracy": accuracy,
            "total_predictions": len(predictions),
        }

        return self.models["outcome_classifier"]

    def cross_validate(self, model_name: str, k_folds: int = 5) -> dict:
        """
        Perform k-fold cross-validation on a model.

        Args:
            model_name: Name of model to validate (e.g., 'baseline_ols', 'multifeature_ols')
            k_folds: Number of folds

        Returns:
            Dictionary with mean/std of R², RMSE, MAE across folds
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Fit model first.")

        model_info = self.models[model_name]
        feature_cols = model_info["features"]

        X, y = self._prepare_data(feature_cols)
        n_samples = X.shape[0]
        fold_indices = np.array_split(np.arange(n_samples), k_folds)

        fold_results = {"r2": [], "rmse": [], "mae": []}

        # Forward-chaining validation: train only on earlier matches, test on later matches.
        for fold in range(1, len(fold_indices)):
            train_idx = np.concatenate(fold_indices[:fold])
            test_idx = fold_indices[fold]

            if len(train_idx) <= X.shape[1] or len(test_idx) == 0:
                continue

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            coeffs, _, _, _ = self._fit_ols(X_train, y_train)

            y_pred = X_test @ coeffs
            r2 = self._compute_r2(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mae = np.mean(np.abs(y_test - y_pred))

            fold_results["r2"].append(r2)
            fold_results["rmse"].append(rmse)
            fold_results["mae"].append(mae)

        if not fold_results["r2"]:
            raise ValueError("Temporal cross-validation produced no valid folds.")

        cv_summary = {
            "model": model_name,
            "k_folds": len(fold_results["r2"]),
            "strategy": "forward_chaining_temporal",
            "r2_mean": np.mean(fold_results["r2"]),
            "r2_std": np.std(fold_results["r2"]),
            "rmse_mean": np.mean(fold_results["rmse"]),
            "rmse_std": np.std(fold_results["rmse"]),
            "mae_mean": np.mean(fold_results["mae"]),
            "mae_std": np.std(fold_results["mae"]),
        }

        if "cv_results" not in self.models[model_name]:
            self.models[model_name]["cv_results"] = {}
        self.models[model_name]["cv_results"] = cv_summary

        return cv_summary

    def temporal_stability_test(
        self, feature_cols: Optional[list[str]] = None
    ) -> dict:
        """
        Test temporal stability by training on earlier seasons and testing on later ones.

        Returns:
            Dictionary with R² by test season
        """
        feature_cols = feature_cols or ["xg_diff"]

        # Get unique seasons and sort
        seasons = sorted(self.features_df["season_name"].unique().to_list())

        temporal_results = {"train_season": [], "test_season": [], "r2": []}

        for test_idx in range(1, len(seasons)):
            train_season = seasons[test_idx - 1]
            test_season = seasons[test_idx]

            # Train on earlier season
            train_data = self.features_df.filter(pl.col("season_name") == train_season)
            X_train, y_train = self._prepare_data_from_df(train_data, feature_cols)

            # Test on later season
            test_data = self.features_df.filter(pl.col("season_name") == test_season)
            X_test, y_test = self._prepare_data_from_df(test_data, feature_cols)

            # Fit and evaluate
            if X_train.shape[0] > 2 and X_test.shape[0] > 2:
                coeffs, _, _, _ = self._fit_ols(X_train, y_train)
                y_pred = X_test @ coeffs
                r2 = self._compute_r2(y_test, y_pred)

                temporal_results["train_season"].append(train_season)
                temporal_results["test_season"].append(test_season)
                temporal_results["r2"].append(r2)

        return temporal_results

    def get_feature_importance(self, model_name: str) -> dict:
        """
        Get feature importance (absolute coefficient values) for a model.

        Args:
            model_name: Name of fitted model

        Returns:
            Dictionary with features ranked by importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        model = self.models[model_name]
        feature_coeffs = model.get("feature_coefficients", {})

        # Rank by absolute coefficient value
        ranked = sorted(
            feature_coeffs.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            "model": model_name,
            "features_ranked": ranked,
            "feature_count": len(ranked),
        }

    def predict(self, model_name: str, features_df: Optional[pl.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using a fitted model.

        Args:
            model_name: Name of fitted model
            features_df: DataFrame with features (uses self.features_df if None)

        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        if features_df is None:
            features_df = self.features_df

        model = self.models[model_name]
        feature_cols = model["features"]
        coeffs = model["coefficients"]

        X, _ = self._prepare_data_from_df(features_df, feature_cols)

        return X @ coeffs

    def evaluate(self, model_name: str, y_pred: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate model performance.

        Args:
            model_name: Name of fitted model
            y_pred: Predicted values (computes if None)

        Returns:
            Dictionary with R², RMSE, MAE, residual stats
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        y = np.array(self.features_df[self.target_col].to_list())

        if y_pred is None:
            y_pred = self.predict(model_name)

        r2 = self._compute_r2(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        residuals = y - y_pred

        return {
            "model": model_name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
        }

    # ===== Private helper methods =====

    def _prepare_data(self, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector for modeling."""
        return self._prepare_data_from_df(self.features_df, feature_cols)

    def _prepare_data_from_df(
        self, df: pl.DataFrame, feature_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y from a DataFrame.

        Adds intercept column (column of ones) to X.
        Removes rows with missing values.
        """
        # Remove rows with nulls
        df_clean = df.select(feature_cols + [self.target_col]).drop_nulls()

        X = df_clean.select(feature_cols).to_numpy()
        y = df_clean[self.target_col].to_numpy()

        # Add intercept column
        X = np.column_stack([np.ones(X.shape[0]), X])

        return X, y

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """
        Fit OLS regression using normal equations.

        Args:
            X: Feature matrix (with intercept as first column)
            y: Target vector

        Returns:
            Tuple of (coefficients, R², RMSE, MAE)
        """
        # Normal equations: (X'X)^-1 X'y
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            coeffs = np.zeros(X.shape[1])
            coeffs[0] = np.mean(y)

        y_pred = X @ coeffs

        r2 = self._compute_r2(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))

        return coeffs, r2, rmse, mae

    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - (ss_res / ss_tot)


def plot_xg_vs_goals(model: MatchOutcomeModel, output_path: Optional[Path] = None):
    """
    Create scatter plot of xG differential vs actual goal differential.

    Args:
        model: Fitted MatchOutcomeModel
        output_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    xg_diff = model.features_df["xg_diff"].to_numpy()
    goal_diff = model.features_df["goal_diff"].to_numpy()

    ax.scatter(xg_diff, goal_diff, alpha=0.5, s=30)

    # Add regression line
    z = np.polyfit(xg_diff, goal_diff, 1)
    p = np.poly1d(z)
    ax.plot(xg_diff, p(xg_diff), "r--", linewidth=2, label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}")

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    ax.set_xlabel("xG Differential (Home - Away)", fontsize=11)
    ax.set_ylabel("Goal Differential (Home - Away)", fontsize=11)
    ax.set_title("Expected Goals vs Actual Goals", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def plot_residuals(model: MatchOutcomeModel, model_name: str = "baseline_ols",
                  output_path: Optional[Path] = None):
    """
    Create residual distribution plot.

    Args:
        model: Fitted MatchOutcomeModel
        model_name: Name of model to plot residuals for
        output_path: Path to save figure (optional)
    """
    y = np.array(model.features_df[model.target_col].to_list())
    y_pred = model.predict(model_name)
    residuals = y - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax1.set_xlabel("Residuals", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Q-Q plot approximation
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted Values", fontsize=11)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.set_title("Residuals vs Fitted Values", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, (ax1, ax2)


def plot_feature_importance(model: MatchOutcomeModel, model_name: str = "multifeature_ols",
                           top_k: int = 8, output_path: Optional[Path] = None):
    """
    Create feature importance bar chart.

    Args:
        model: Fitted MatchOutcomeModel
        model_name: Name of model
        top_k: Number of top features to display
        output_path: Path to save figure (optional)
    """
    importance = model.get_feature_importance(model_name)
    ranked = importance["features_ranked"][:top_k]

    features = [f[0] for f in ranked]
    coeffs = [f[1] for f in ranked]
    colors = ["green" if c > 0 else "red" for c in coeffs]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(features, coeffs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Coefficient Value", fontsize=11)
    ax.set_title(f"Top {top_k} Feature Importance ({model_name})", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def plot_model_comparison(results: dict, output_path: Optional[Path] = None):
    """
    Create model comparison bar chart (R², RMSE, MAE).

    Args:
        results: Dictionary with model evaluation results
        output_path: Path to save figure (optional)
    """
    models = list(results.keys())
    r2_scores = [results[m].get("r2", 0) for m in models]
    rmse_scores = [results[m].get("rmse", 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    ax1.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("R² Score", fontsize=11)
    ax1.set_title("Model Comparison: R²", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # RMSE comparison
    ax2.bar(models, rmse_scores, color=colors, alpha=0.8, edgecolor="black")
    ax2.set_ylabel("RMSE", fontsize=11)
    ax2.set_title("Model Comparison: RMSE", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, (ax1, ax2)


def plot_temporal_stability(temporal_results: dict, output_path: Optional[Path] = None):
    """
    Create temporal stability line chart (R² by season).

    Args:
        temporal_results: Dictionary with temporal test results
        output_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    seasons = temporal_results["test_season"]
    r2_scores = temporal_results["r2"]

    ax.plot(seasons, r2_scores, marker="o", linewidth=2, markersize=8, color="steelblue")
    ax.fill_between(range(len(seasons)), r2_scores, alpha=0.3, color="steelblue")

    ax.set_xlabel("Test Season", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("Temporal Stability: OLS Model Performance by Season", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def analyze_halftime_edge(features_df: pl.DataFrame) -> dict:
    """
    Quantify how informative halftime state is for the final result.

    This mirrors the office-hours guidance to focus on cases where the halftime
    score is misleading relative to underlying first-half performance.
    """
    scored = features_df.with_columns(
        [
            pl.when(pl.col("halftime_goal_diff") > 0)
            .then(pl.lit("home"))
            .when(pl.col("halftime_goal_diff") < 0)
            .then(pl.lit("away"))
            .otherwise(pl.lit("draw"))
            .alias("halftime_score_state"),
            pl.when(pl.col("halftime_xg_diff") > 0)
            .then(pl.lit("home"))
            .when(pl.col("halftime_xg_diff") < 0)
            .then(pl.lit("away"))
            .otherwise(pl.lit("draw"))
            .alias("halftime_xg_state"),
            pl.when(pl.col("goal_diff") > 0)
            .then(pl.lit("home"))
            .when(pl.col("goal_diff") < 0)
            .then(pl.lit("away"))
            .otherwise(pl.lit("draw"))
            .alias("fulltime_state"),
        ]
    )

    halftime_leads = scored.filter(pl.col("halftime_score_state") != "draw")
    halftime_leader_failed = halftime_leads.filter(
        pl.col("halftime_score_state") != pl.col("fulltime_state")
    )
    halftime_xg_leads = scored.filter(pl.col("halftime_xg_state") != "draw")
    halftime_xg_leader_failed = halftime_xg_leads.filter(
        pl.col("halftime_xg_state") != pl.col("fulltime_state")
    )
    score_xg_disagreements = scored.filter(
        (pl.col("halftime_score_state") != "draw")
        & (pl.col("halftime_xg_state") != "draw")
        & (pl.col("halftime_score_state") != pl.col("halftime_xg_state"))
    )

    return {
        "matches": len(scored),
        "halftime_leads": len(halftime_leads),
        "halftime_leader_failed_to_win": len(halftime_leader_failed),
        "halftime_leader_failure_rate": (
            len(halftime_leader_failed) / len(halftime_leads)
            if len(halftime_leads) > 0
            else 0.0
        ),
        "halftime_xg_leads": len(halftime_xg_leads),
        "halftime_xg_leader_failed_to_win": len(halftime_xg_leader_failed),
        "halftime_xg_leader_failure_rate": (
            len(halftime_xg_leader_failed) / len(halftime_xg_leads)
            if len(halftime_xg_leads) > 0
            else 0.0
        ),
        "score_xg_disagreements": len(score_xg_disagreements),
        "score_xg_disagreement_rate": (
            len(score_xg_disagreements) / len(scored) if len(scored) > 0 else 0.0
        ),
    }


def fit_halftime_upset_tree(
    features_df: pl.DataFrame,
    max_depth: int = 3,
    min_samples_leaf: int = 40,
) -> dict:
    """
    Fit a decision tree to explain when halftime leaders fail to win.

    This follows the office-hours suggestion to subset matches by halftime lead
    and identify unstable game states rather than treating all games uniformly.
    """
    halftime_leads = features_df.filter(pl.col("halftime_goal_diff") != 0).sort(
        "match_date"
    )
    if halftime_leads.is_empty():
        return {"error": "No halftime leads available for tree analysis"}

    modeling_df = halftime_leads.with_columns(
        [
            (
                ((pl.col("halftime_goal_diff") > 0) & (pl.col("goal_diff") <= 0))
                | ((pl.col("halftime_goal_diff") < 0) & (pl.col("goal_diff") >= 0))
            )
            .cast(pl.Int32)
            .alias("leader_failed"),
            pl.col("halftime_goal_diff").abs().alias("abs_halftime_lead"),
        ]
    )

    feature_cols = [
        "abs_halftime_lead",
        "halftime_xg_diff",
        "halftime_shots_on_target_diff",
        "halftime_pass_volume_diff",
        "halftime_possession_share",
        "halftime_pressure_events_diff",
        "halftime_pass_completion_diff",
        "halftime_carries_final_third_diff",
    ]
    clean = modeling_df.select(feature_cols + ["leader_failed"]).drop_nulls()
    if clean.height < 200:
        return {"error": "Not enough clean halftime-lead rows for tree analysis"}

    X = clean.select(feature_cols).to_numpy()
    y = clean["leader_failed"].to_numpy()

    split_idx = max(int(len(clean) * 0.8), max_depth + min_samples_leaf)
    split_idx = min(split_idx, len(clean) - min_samples_leaf)
    if split_idx <= 0 or split_idx >= len(clean):
        split_idx = int(len(clean) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if len(X_test) == 0:
        return {"error": "Temporal holdout split for upset tree is empty"}

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced",
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    positive_rate = float(np.mean(y_test))
    baseline_accuracy = float(max(positive_rate, 1.0 - positive_rate))
    collapse_recall = float(
        np.mean(y_pred[y_test == 1] == 1) if np.any(y_test == 1) else 0.0
    )
    noncollapse_recall = float(
        np.mean(y_pred[y_test == 0] == 0) if np.any(y_test == 0) else 0.0
    )
    balanced_accuracy = float((collapse_recall + noncollapse_recall) / 2.0)

    feature_importance = sorted(
        [
            (feature_cols[idx], float(tree.feature_importances_[idx]))
            for idx in range(len(feature_cols))
            if tree.feature_importances_[idx] > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    lead_summary = (
        modeling_df.with_columns(
            pl.when(pl.col("abs_halftime_lead") >= 2)
            .then(pl.lit("2+ goals"))
            .otherwise(pl.lit("1 goal"))
            .alias("lead_bucket")
        )
        .group_by("lead_bucket")
        .agg(
            [
                pl.len().alias("matches"),
                pl.col("leader_failed").mean().alias("failure_rate"),
            ]
        )
        .sort("lead_bucket")
        .to_dicts()
    )

    return {
        "matches": int(len(clean)),
        "holdout_matches": int(len(y_test)),
        "tree_depth": int(tree.get_depth()),
        "leaf_count": int(tree.get_n_leaves()),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "positive_rate": positive_rate,
        "collapse_recall": collapse_recall,
        "noncollapse_recall": noncollapse_recall,
        "balanced_accuracy": balanced_accuracy,
        "feature_importance": feature_importance,
        "rules": export_text(tree, feature_names=feature_cols, max_depth=max_depth),
        "lead_summary": lead_summary,
    }


def plot_upset_tree_feature_importance(
    tree_results: dict, output_path: Optional[Path] = None
):
    """Plot feature importance for the halftime upset tree."""
    importance = tree_results.get("feature_importance", [])
    if not importance:
        return None, None

    features = [item[0] for item in importance]
    values = [item[1] for item in importance]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, values, color="indianred", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(
        "Halftime Upset Tree Feature Importance",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def run_full_analysis() -> dict:
    """
    Execute full modeling analysis pipeline.

    Loads data, computes features, fits all models, evaluates performance,
    and generates comprehensive results and visualizations.

    Returns:
        Dictionary with all results and model objects
    """
    print("=" * 80)
    print("  SOCCER ANALYTICS CAPSTONE: PREDICTIVE MODELING ANALYSIS")
    print("=" * 80)

    # ===== Data Loading =====
    print("\n[1/7] Loading data...")
    try:
        events_df = pl.read_parquet(STATSBOMB_DIR / "events.parquet")
        matches_df = pl.read_parquet(STATSBOMB_DIR / "matches.parquet")
        print(f"  ✓ Events: {len(events_df):,} records")
        print(f"  ✓ Matches: {len(matches_df):,} records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    # ===== Feature Engineering =====
    print("\n[2/7] Building match features...")
    builder = MatchFeatureBuilder(events_df, matches_df)
    features_df = builder.build_features()
    print(f"  ✓ Features computed for {len(features_df):,} matches")
    print(f"  ✓ Full-match feature columns: {', '.join(builder.fulltime_feature_cols)}")
    print(f"  ✓ Halftime feature columns: {', '.join(builder.halftime_feature_cols)}")
    features_output_path = OUTPUT_DIR / "match_features.parquet"
    features_df.write_parquet(features_output_path)
    print(f"  ✓ Saved match-level features to {features_output_path}")

    # ===== Model Fitting =====
    print("\n[3/7] Fitting predictive models...")
    model = MatchOutcomeModel(features_df, target_col="goal_diff")

    print("  Pre-match ELO baseline...")
    elo_results = model.fit_elo_baseline()
    print(f"    R² = {elo_results['r2']:.4f}, RMSE = {elo_results['rmse']:.4f}")

    print("  Baseline OLS (xG differential only)...")
    baseline_results = model.fit_baseline_ols()
    print(f"    R² = {baseline_results['r2']:.4f}, RMSE = {baseline_results['rmse']:.4f}")

    print("  Multi-feature OLS (all features)...")
    multifeature_results = model.fit_multifeature_ols()
    print(f"    R² = {multifeature_results['r2']:.4f}, RMSE = {multifeature_results['rmse']:.4f}")

    print("  Forward feature selection (top 5 features)...")
    selection_results = model.forward_feature_selection(k_best=5)
    print(f"    R² = {selection_results['r2']:.4f}, RMSE = {selection_results['rmse']:.4f}")
    print(f"    Selected features: {', '.join(selection_results['features'])}")

    print("  Halftime score baseline...")
    halftime_score_results = model.fit_halftime_score_baseline()
    print(f"    R² = {halftime_score_results['r2']:.4f}, RMSE = {halftime_score_results['rmse']:.4f}")

    print("  Halftime xG baseline...")
    halftime_xg_results = model.fit_halftime_xg_baseline()
    print(f"    R² = {halftime_xg_results['r2']:.4f}, RMSE = {halftime_xg_results['rmse']:.4f}")

    print("  Halftime live model (score + momentum metrics)...")
    halftime_live_results = model.fit_halftime_live_ols()
    print(f"    R² = {halftime_live_results['r2']:.4f}, RMSE = {halftime_live_results['rmse']:.4f}")

    print("  Match outcome classifier...")
    classifier_results = model.fit_outcome_classifier()
    print(f"    Accuracy = {classifier_results['accuracy']:.4f}")

    # ===== Cross-Validation =====
    print("\n[4/7] Cross-validation (5-fold)...")
    cv_elo = model.cross_validate("elo_baseline", k_folds=5)
    print(f"  ELO baseline: R² = {cv_elo['r2_mean']:.4f} ± {cv_elo['r2_std']:.4f}")

    cv_baseline = model.cross_validate("baseline_ols", k_folds=5)
    print(f"  Baseline OLS: R² = {cv_baseline['r2_mean']:.4f} ± {cv_baseline['r2_std']:.4f}")

    cv_multifeature = model.cross_validate("multifeature_ols", k_folds=5)
    print(f"  Multi-feature OLS: R² = {cv_multifeature['r2_mean']:.4f} ± {cv_multifeature['r2_std']:.4f}")

    cv_selection = model.cross_validate("forward_selection", k_folds=5)
    print(f"  Forward selection: R² = {cv_selection['r2_mean']:.4f} ± {cv_selection['r2_std']:.4f}")

    cv_halftime_live = model.cross_validate("halftime_live_ols", k_folds=5)
    print(f"  Halftime live OLS: R² = {cv_halftime_live['r2_mean']:.4f} ± {cv_halftime_live['r2_std']:.4f}")

    # ===== Temporal Stability =====
    print("\n[5/7] Temporal stability test (season-to-season)...")
    temporal_results = model.temporal_stability_test(feature_cols=["xg_diff"])
    if temporal_results["r2"]:
        avg_temporal_r2 = np.mean(temporal_results["r2"])
        print(f"  Average R² across seasons: {avg_temporal_r2:.4f}")
        for i, test_season in enumerate(temporal_results["test_season"]):
            print(f"    {temporal_results['train_season'][i]} → {test_season}: R² = {temporal_results['r2'][i]:.4f}")

    # ===== Evaluation =====
    print("\n[6/7] Model evaluation...")
    evaluations = {}
    for model_name in [
        "elo_baseline",
        "baseline_ols",
        "multifeature_ols",
        "forward_selection",
        "halftime_score_baseline",
        "halftime_xg_baseline",
        "halftime_live_ols",
    ]:
        eval_result = model.evaluate(model_name)
        evaluations[model_name] = eval_result
        print(f"\n  {model_name}:")
        print(f"    R² = {eval_result['r2']:.4f}")
        print(f"    RMSE = {eval_result['rmse']:.4f}")
        print(f"    MAE = {eval_result['mae']:.4f}")

    halftime_edge = analyze_halftime_edge(features_df)
    upset_tree_results = fit_halftime_upset_tree(features_df)
    print("\n[6b/7] Halftime edge analysis...")
    print(
        "  Halftime leader failed to win: "
        f"{halftime_edge['halftime_leader_failed_to_win']:,} / "
        f"{halftime_edge['halftime_leads']:,} "
        f"({halftime_edge['halftime_leader_failure_rate']:.1%})"
    )
    print(
        "  Halftime xG leader failed to win: "
        f"{halftime_edge['halftime_xg_leader_failed_to_win']:,} / "
        f"{halftime_edge['halftime_xg_leads']:,} "
        f"({halftime_edge['halftime_xg_leader_failure_rate']:.1%})"
    )
    print(
        "  Matches where halftime score and xG disagree on the leader: "
        f"{halftime_edge['score_xg_disagreements']:,} "
        f"({halftime_edge['score_xg_disagreement_rate']:.1%})"
    )
    if "error" not in upset_tree_results:
        print("\n[6c/7] Halftime upset tree...")
        print(
            "  Holdout accuracy / balanced accuracy: "
            f"{upset_tree_results['accuracy']:.4f} / "
            f"{upset_tree_results['balanced_accuracy']:.4f}"
        )
        print(
            "  Collapse recall / non-collapse recall: "
            f"{upset_tree_results['collapse_recall']:.4f} / "
            f"{upset_tree_results['noncollapse_recall']:.4f}"
        )
        print(
            "  Tree depth / leaves: "
            f"{upset_tree_results['tree_depth']} / {upset_tree_results['leaf_count']}"
        )
        if upset_tree_results["feature_importance"]:
            top_feature, top_value = upset_tree_results["feature_importance"][0]
            print(f"  Most important halftime split: {top_feature} ({top_value:.3f})")

    # ===== Visualizations =====
    print("\n[7/7] Generating visualizations...")

    plot_xg_vs_goals(model, OUTPUT_DIR / "01_xg_vs_goals.png")
    plot_residuals(model, "baseline_ols", OUTPUT_DIR / "02_residuals_baseline.png")
    plot_residuals(model, "multifeature_ols", OUTPUT_DIR / "03_residuals_multifeature.png")
    plot_feature_importance(model, "multifeature_ols", top_k=8,
                          output_path=OUTPUT_DIR / "04_feature_importance.png")
    plot_model_comparison(evaluations, OUTPUT_DIR / "05_model_comparison.png")
    plot_temporal_stability(temporal_results, OUTPUT_DIR / "06_temporal_stability.png")
    if "error" not in upset_tree_results:
        plot_upset_tree_feature_importance(
            upset_tree_results, OUTPUT_DIR / "07_upset_tree_importance.png"
        )
        (OUTPUT_DIR / "08_upset_tree_rules.txt").write_text(
            upset_tree_results["rules"]
        )
        print(f"Saved: {OUTPUT_DIR / '08_upset_tree_rules.txt'}")

    # ===== Summary Report =====
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<25} {'R²':<12} {'RMSE':<12} {'MAE':<12}")
    print("-" * 60)

    for model_name in [
        "elo_baseline",
        "baseline_ols",
        "multifeature_ols",
        "forward_selection",
        "halftime_score_baseline",
        "halftime_xg_baseline",
        "halftime_live_ols",
    ]:
        eval_res = evaluations[model_name]
        print(f"{model_name:<25} {eval_res['r2']:<12.4f} {eval_res['rmse']:<12.4f} {eval_res['mae']:<12.4f}")

    print("\n" + "=" * 80)
    print("  FEATURE IMPORTANCE (Multi-Feature Model)")
    print("=" * 80)

    importance = model.get_feature_importance("multifeature_ols")
    for feature, coeff in importance["features_ranked"][:10]:
        print(f"  {feature:<30} {coeff:>10.4f}")

    print("\n" + "=" * 80)
    print(f"  Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

    # Return comprehensive results
    return {
        "features_df": features_df,
        "model": model,
        "elo_results": elo_results,
        "baseline_results": baseline_results,
        "multifeature_results": multifeature_results,
        "selection_results": selection_results,
        "halftime_score_results": halftime_score_results,
        "halftime_xg_results": halftime_xg_results,
        "halftime_live_results": halftime_live_results,
        "classifier_results": classifier_results,
        "cv_results": {
            "elo": cv_elo,
            "baseline": cv_baseline,
            "multifeature": cv_multifeature,
            "selection": cv_selection,
            "halftime_live": cv_halftime_live,
        },
        "temporal_results": temporal_results,
        "halftime_edge": halftime_edge,
        "upset_tree_results": upset_tree_results,
        "evaluations": evaluations,
        "output_dir": OUTPUT_DIR,
    }


if __name__ == "__main__":
    results = run_full_analysis()
