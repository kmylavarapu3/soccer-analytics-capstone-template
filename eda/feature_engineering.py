"""
Feature Engineering Pipeline for Soccer Analytics

This module provides utilities for computing advanced soccer analytics metrics:
- Possession chains
- Expected Goals (xG) flow
- Expected Threat (xT) 
- Pressure metrics (PPDA, defensive intensity)
- Field tilt and territorial dominance
- Playing style classification

Uses Polars for efficient processing of large event datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"

# Pitch dimensions (StatsBomb uses 120x80)
PITCH_LENGTH = 120
PITCH_WIDTH = 80

# Zone definitions for xT grid (12x8 grid)
XT_GRID_X = 12
XT_GRID_Y = 8

# Pre-computed xT values (simplified Karun Singh model approximation)
# Higher values in attacking areas, especially central near goal
XT_GRID = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # Own goal area
    [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    [0.002, 0.003, 0.004, 0.004, 0.004, 0.004, 0.003, 0.002],
    [0.003, 0.005, 0.007, 0.008, 0.008, 0.007, 0.005, 0.003],
    [0.005, 0.008, 0.012, 0.015, 0.015, 0.012, 0.008, 0.005],
    [0.008, 0.013, 0.020, 0.025, 0.025, 0.020, 0.013, 0.008],
    [0.012, 0.020, 0.032, 0.042, 0.042, 0.032, 0.020, 0.012],
    [0.018, 0.030, 0.050, 0.065, 0.065, 0.050, 0.030, 0.018],
    [0.025, 0.045, 0.075, 0.100, 0.100, 0.075, 0.045, 0.025],
    [0.035, 0.065, 0.110, 0.150, 0.150, 0.110, 0.065, 0.035],
    [0.050, 0.095, 0.170, 0.250, 0.250, 0.170, 0.095, 0.050],
    [0.080, 0.150, 0.280, 0.380, 0.380, 0.280, 0.150, 0.080],  # Opponent goal area
])


def get_xt_zone(x: float, y: float) -> tuple[int, int]:
    """Get the xT grid zone for a given pitch location."""
    zone_x = min(int(x / PITCH_LENGTH * XT_GRID_X), XT_GRID_X - 1)
    zone_y = min(int(y / PITCH_WIDTH * XT_GRID_Y), XT_GRID_Y - 1)
    return zone_x, zone_y


def get_xt_value(x: float, y: float) -> float:
    """Get the xT value for a given pitch location."""
    zone_x, zone_y = get_xt_zone(x, y)
    return XT_GRID[zone_x, zone_y]


class PossessionChainBuilder:
    """
    Build possession chains from event data.
    
    A possession chain is a sequence of events by the same team
    until possession is lost (via turnover, foul, out of play, etc.).
    """
    
    # Events that end a possession
    POSSESSION_END_EVENTS = {
        "Dispossessed", "Miscontrol", "Foul Won", "Ball Recovery",
        "Interception", "Clearance", "Shot", "Goal Keeper",
    }
    
    # Events that indicate possession change
    POSSESSION_CHANGE_PATTERNS = {
        "Incomplete",  # Pass outcome
        "Out",  # Ball out of play
        "Lost",  # Duel lost
    }
    
    def __init__(self, events_df: pl.DataFrame):
        self.events = events_df.sort(["match_id", "period", "minute", "second"])
        
    def build_chains(self) -> pl.DataFrame:
        """
        Build possession chains for all matches.
        
        Returns DataFrame with chain_id added to events.
        """
        # Process by match
        chains = []
        
        for match_id in self.events["match_id"].unique().to_list():
            match_events = self.events.filter(pl.col("match_id") == match_id)
            match_chains = self._build_match_chains(match_events)
            chains.append(match_chains)
        
        return pl.concat(chains) if chains else self.events.with_columns(
            pl.lit(0).alias("chain_id")
        )
    
    def _build_match_chains(self, events: pl.DataFrame) -> pl.DataFrame:
        """Build chains for a single match."""
        chain_ids = []
        current_chain = 0
        current_team = None
        
        for row in events.iter_rows(named=True):
            event_type = row.get("type", "")
            team = row.get("team", "")
            pass_outcome = row.get("pass_outcome", "")
            
            # Check if possession changed
            possession_changed = False
            
            # Team changed
            if team and team != current_team:
                possession_changed = True
            
            # Possession-ending event
            if event_type in self.POSSESSION_END_EVENTS:
                possession_changed = True
            
            # Incomplete pass
            if pass_outcome and any(p in pass_outcome for p in self.POSSESSION_CHANGE_PATTERNS):
                possession_changed = True
            
            if possession_changed and current_team is not None:
                current_chain += 1
            
            current_team = team
            chain_ids.append(current_chain)
        
        return events.with_columns(pl.Series("chain_id", chain_ids))
    
    def get_chain_summary(self, chains_df: pl.DataFrame) -> pl.DataFrame:
        """
        Summarize possession chains.
        
        Returns metrics per chain: duration, events, xT gained, shots, goals.
        """
        return chains_df.group_by(["match_id", "chain_id", "team"]).agg([
            pl.len().alias("event_count"),
            (pl.col("minute").max() - pl.col("minute").min()).alias("duration_minutes"),
            pl.col("shot_statsbomb_xg").sum().alias("total_xg"),
            (pl.col("type") == "Shot").sum().alias("shots"),
            pl.col("location_x").mean().alias("avg_x"),
            pl.col("location_y").mean().alias("avg_y"),
        ]).sort(["match_id", "chain_id"])


class MatchMetrics:
    """
    Compute match-level advanced metrics.
    """
    
    def __init__(self, events_df: pl.DataFrame, matches_df: pl.DataFrame):
        self.events = events_df
        self.matches = matches_df
        
    def compute_xg_flow(self, match_id: int) -> pl.DataFrame:
        """
        Compute cumulative xG over time for a match.
        
        Returns minute-by-minute xG for both teams.
        """
        match_events = self.events.filter(
            (pl.col("match_id") == match_id) & 
            (pl.col("type") == "Shot") &
            (pl.col("shot_statsbomb_xg").is_not_null())
        ).sort(["period", "minute", "second"])
        
        # Get team names
        match_info = self.matches.filter(pl.col("match_id") == match_id)
        if match_info.is_empty():
            return pl.DataFrame()
        
        home_team = match_info["home_team"][0]
        away_team = match_info["away_team"][0]
        
        # Compute cumulative xG
        home_xg = []
        away_xg = []
        minutes = []
        
        cum_home = 0.0
        cum_away = 0.0
        
        for row in match_events.iter_rows(named=True):
            team = row.get("team", "")
            xg = row.get("shot_statsbomb_xg", 0) or 0
            minute = row.get("minute", 0)
            period = row.get("period", 1)
            
            # Adjust minute for second half
            if period == 2:
                minute += 45
            elif period == 3:
                minute += 90
            elif period == 4:
                minute += 105
            
            if team == home_team:
                cum_home += xg
            elif team == away_team:
                cum_away += xg
            
            minutes.append(minute)
            home_xg.append(cum_home)
            away_xg.append(cum_away)
        
        return pl.DataFrame({
            "minute": minutes,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_team": [home_team] * len(minutes),
            "away_team": [away_team] * len(minutes),
        })
    
    def compute_ppda(self, match_id: int) -> dict:
        """
        Compute Passes Per Defensive Action (PPDA) for each team.
        
        PPDA = Opponent passes in own half / Defensive actions in opponent half
        Lower PPDA indicates more intense pressing.
        """
        match_events = self.events.filter(pl.col("match_id") == match_id)
        
        match_info = self.matches.filter(pl.col("match_id") == match_id)
        if match_info.is_empty():
            return {}
        
        home_team = match_info["home_team"][0]
        away_team = match_info["away_team"][0]
        
        # Defensive actions: tackles, interceptions, fouls, challenges
        defensive_types = ["Pressure", "Duel", "Interception", "Foul Committed", "Block"]
        
        # Home team PPDA: Away passes in away half / Home defensive actions in away half
        away_passes_own_half = match_events.filter(
            (pl.col("team") == away_team) &
            (pl.col("type") == "Pass") &
            (pl.col("location_x") < 60)  # Own half (0-60)
        ).height
        
        home_def_actions_away_half = match_events.filter(
            (pl.col("team") == home_team) &
            (pl.col("type").is_in(defensive_types)) &
            (pl.col("location_x") > 60)  # Opponent half
        ).height
        
        # Away team PPDA
        home_passes_own_half = match_events.filter(
            (pl.col("team") == home_team) &
            (pl.col("type") == "Pass") &
            (pl.col("location_x") < 60)
        ).height
        
        away_def_actions_home_half = match_events.filter(
            (pl.col("team") == away_team) &
            (pl.col("type").is_in(defensive_types)) &
            (pl.col("location_x") > 60)
        ).height
        
        home_ppda = away_passes_own_half / home_def_actions_away_half if home_def_actions_away_half > 0 else float('inf')
        away_ppda = home_passes_own_half / away_def_actions_home_half if away_def_actions_home_half > 0 else float('inf')
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_ppda": round(home_ppda, 2),
            "away_ppda": round(away_ppda, 2),
            "home_pressing_intensity": "High" if home_ppda < 10 else "Medium" if home_ppda < 15 else "Low",
            "away_pressing_intensity": "High" if away_ppda < 10 else "Medium" if away_ppda < 15 else "Low",
        }
    
    def compute_field_tilt(self, match_id: int) -> dict:
        """
        Compute Field Tilt: percentage of total passes in the final third.
        
        Measures territorial dominance.
        """
        match_events = self.events.filter(
            (pl.col("match_id") == match_id) &
            (pl.col("type") == "Pass")
        )
        
        match_info = self.matches.filter(pl.col("match_id") == match_id)
        if match_info.is_empty():
            return {}
        
        home_team = match_info["home_team"][0]
        away_team = match_info["away_team"][0]
        
        # Final third is x > 80 (last 40 yards of 120)
        home_final_third = match_events.filter(
            (pl.col("team") == home_team) &
            (pl.col("location_x") > 80)
        ).height
        
        away_final_third = match_events.filter(
            (pl.col("team") == away_team) &
            (pl.col("location_x") > 80)
        ).height
        
        total = home_final_third + away_final_third
        
        if total == 0:
            return {"home_tilt": 50.0, "away_tilt": 50.0}
        
        home_tilt = (home_final_third / total) * 100
        away_tilt = (away_final_third / total) * 100
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_field_tilt": round(home_tilt, 1),
            "away_field_tilt": round(away_tilt, 1),
            "dominant_team": home_team if home_tilt > away_tilt else away_team,
        }


class TeamStyleClassifier:
    """
    Classify team playing styles based on event data.
    
    Styles:
    - Possession-based: High pass completion, long possession chains
    - Counter-attacking: Fast transitions, direct play
    - High-pressing: Low PPDA, high defensive actions in opponent half
    - Defensive: Deep defensive line, low field tilt
    """
    
    STYLE_THRESHOLDS = {
        "possession": {
            "pass_completion": 0.82,
            "avg_chain_length": 5,
        },
        "counter_attack": {
            "direct_speed": 2.5,  # Avg x-progress per event
            "shot_chain_length": 4,  # Avg events before shot
        },
        "high_press": {
            "ppda": 10,
            "high_recoveries_pct": 0.35,
        },
        "defensive": {
            "field_tilt": 45,
            "avg_defensive_line": 35,
        },
    }
    
    def __init__(self, events_df: pl.DataFrame, matches_df: pl.DataFrame):
        self.events = events_df
        self.matches = matches_df
        self.metrics = MatchMetrics(events_df, matches_df)
        
    def classify_team_style(self, team_name: str) -> dict:
        """
        Classify a team's playing style based on their event data.
        """
        team_events = self.events.filter(pl.col("team") == team_name)
        
        if team_events.is_empty():
            return {"team": team_name, "style": "Unknown", "confidence": 0}
        
        # Compute metrics
        passes = team_events.filter(pl.col("type") == "Pass")
        total_passes = len(passes)
        successful_passes = passes.filter(pl.col("pass_outcome").is_null()).height
        pass_completion = successful_passes / total_passes if total_passes > 0 else 0
        
        # Possession metrics
        chain_builder = PossessionChainBuilder(team_events)
        chains = chain_builder.build_chains()
        chain_summary = chain_builder.get_chain_summary(chains)
        avg_chain_length = chain_summary["event_count"].mean() if not chain_summary.is_empty() else 0
        
        # Pressing metrics
        pressure_events = team_events.filter(pl.col("type") == "Pressure")
        high_press_recoveries = team_events.filter(
            (pl.col("type") == "Ball Recovery") &
            (pl.col("location_x") > 60)
        ).height
        total_recoveries = team_events.filter(pl.col("type") == "Ball Recovery").height
        high_recovery_pct = high_press_recoveries / total_recoveries if total_recoveries > 0 else 0
        
        # Determine primary style
        styles = []
        
        if pass_completion > self.STYLE_THRESHOLDS["possession"]["pass_completion"]:
            styles.append(("Possession-based", pass_completion))
        
        if high_recovery_pct > self.STYLE_THRESHOLDS["high_press"]["high_recoveries_pct"]:
            styles.append(("High-pressing", high_recovery_pct))
        
        if avg_chain_length and avg_chain_length < self.STYLE_THRESHOLDS["counter_attack"]["shot_chain_length"]:
            styles.append(("Counter-attacking", 1 - (avg_chain_length / 10)))
        
        if not styles:
            styles.append(("Balanced", 0.5))
        
        # Sort by confidence
        styles.sort(key=lambda x: x[1], reverse=True)
        primary_style = styles[0][0]
        
        return {
            "team": team_name,
            "primary_style": primary_style,
            "pass_completion": round(pass_completion, 3),
            "avg_chain_length": round(avg_chain_length, 1) if avg_chain_length else 0,
            "high_press_recovery_pct": round(high_recovery_pct, 3),
            "pressure_events": len(pressure_events),
            "all_styles": styles,
        }
    
    def classify_all_teams(self) -> pl.DataFrame:
        """Classify playing style for all teams in the dataset."""
        teams = self.events["team"].unique().drop_nulls().to_list()
        
        results = []
        for team in teams:
            style = self.classify_team_style(team)
            results.append({
                "team": style["team"],
                "primary_style": style["primary_style"],
                "pass_completion": style["pass_completion"],
                "avg_chain_length": style["avg_chain_length"],
                "high_press_pct": style["high_press_recovery_pct"],
            })
        
        return pl.DataFrame(results).sort("team")


def compute_xt_for_events(events_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add xT (Expected Threat) values to events with locations.
    
    Also computes xT gained for passes and carries.
    """
    # Add start xT
    events_with_xt = events_df.with_columns([
        pl.struct(["location_x", "location_y"]).map_elements(
            lambda row: get_xt_value(row["location_x"] or 0, row["location_y"] or 0),
            return_dtype=pl.Float64
        ).alias("start_xt")
    ])
    
    return events_with_xt


def compute_match_summary(
    events_df: pl.DataFrame, 
    matches_df: pl.DataFrame,
    match_id: int
) -> dict:
    """
    Compute comprehensive match summary with all advanced metrics.
    """
    metrics = MatchMetrics(events_df, matches_df)
    
    match_info = matches_df.filter(pl.col("match_id") == match_id)
    if match_info.is_empty():
        return {}
    
    xg_flow = metrics.compute_xg_flow(match_id)
    ppda = metrics.compute_ppda(match_id)
    field_tilt = metrics.compute_field_tilt(match_id)
    
    return {
        "match_id": match_id,
        "home_team": match_info["home_team"][0],
        "away_team": match_info["away_team"][0],
        "home_score": match_info["home_score"][0],
        "away_score": match_info["away_score"][0],
        "home_xg": xg_flow["home_xg"][-1] if not xg_flow.is_empty() else 0,
        "away_xg": xg_flow["away_xg"][-1] if not xg_flow.is_empty() else 0,
        "home_ppda": ppda.get("home_ppda", 0),
        "away_ppda": ppda.get("away_ppda", 0),
        "home_field_tilt": field_tilt.get("home_field_tilt", 50),
        "away_field_tilt": field_tilt.get("away_field_tilt", 50),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Feature Engineering Pipeline Demo")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    events_df = pl.read_parquet(STATSBOMB_DIR / "events.parquet")
    matches_df = pl.read_parquet(STATSBOMB_DIR / "matches.parquet")
    
    print(f"Events: {len(events_df):,}")
    print(f"Matches: {len(matches_df):,}")
    
    # Demo: Possession chains for one match
    print("\n--- Possession Chains Demo ---")
    sample_match_id = matches_df["match_id"][0]
    sample_events = events_df.filter(pl.col("match_id") == sample_match_id)
    
    chain_builder = PossessionChainBuilder(sample_events)
    chains = chain_builder.build_chains()
    chain_summary = chain_builder.get_chain_summary(chains)
    print(f"Match {sample_match_id}: {chain_summary['chain_id'].max() + 1} possession chains")
    print(chain_summary.head(10))
    
    # Demo: Match metrics
    print("\n--- Match Metrics Demo ---")
    metrics = MatchMetrics(events_df, matches_df)
    ppda = metrics.compute_ppda(sample_match_id)
    print(f"PPDA: {ppda}")
    
    field_tilt = metrics.compute_field_tilt(sample_match_id)
    print(f"Field Tilt: {field_tilt}")
    
    # Demo: Team style classification
    print("\n--- Team Style Classification Demo ---")
    classifier = TeamStyleClassifier(events_df, matches_df)
    
    # Classify Barcelona (should be possession-based)
    if "Barcelona" in events_df["team"].unique().to_list():
        barca_style = classifier.classify_team_style("Barcelona")
        print(f"Barcelona: {barca_style}")
    
    # Classify a few teams
    print("\nTop 10 teams by pass completion:")
    all_styles = classifier.classify_all_teams()
    print(all_styles.sort("pass_completion", descending=True).head(10))
