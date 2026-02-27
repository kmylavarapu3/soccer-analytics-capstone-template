"""
Entity Resolution Layer: StatsBomb ↔ Polymarket Mapping

This module provides utilities for mapping entities (teams, players, competitions)
between StatsBomb match/event data and Polymarket betting markets.

Key Challenge: IDs are NOT normalized across datasets. This module creates
the mapping layers needed to join betting interest with match events.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import polars as pl

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
POLYMARKET_DIR = DATA_DIR / "Polymarket"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name for fuzzy matching.
    
    Removes common suffixes, normalizes spacing, and lowercases.
    
    Examples:
        "Arsenal FC" -> "arsenal"
        "Manchester United" -> "manchester united"
        "FC Barcelona" -> "barcelona"
        "Paris Saint-Germain" -> "paris saint germain"
    """
    if not name:
        return ""
    
    # Lowercase
    name = name.lower().strip()
    
    # Remove common suffixes/prefixes
    suffixes = [
        r'\s+fc$', r'\s+cf$', r'\s+sc$', r'\s+ac$',
        r'^fc\s+', r'^ac\s+', r'^as\s+', r'^ss\s+',
        r'\s+football\s+club$', r'\s+f\.c\.$', r'\s+afc$',
        r'\s+united$', r'\s+city$',  # Keep these for now, they're identifying
    ]
    
    for suffix in suffixes[:8]:  # Only remove FC/AC type suffixes
        name = re.sub(suffix, '', name)
    
    # Normalize punctuation and spacing
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def extract_team_from_polymarket_question(question: str) -> list[str]:
    """
    Extract team name(s) from a Polymarket market question.
    
    Examples:
        "Will Arsenal win the Premier League?" -> ["arsenal"]
        "Manchester City vs Liverpool - Winner?" -> ["manchester city", "liverpool"]
        "Will Real Madrid win the Champions League?" -> ["real madrid"]
    """
    if not question:
        return []
    
    teams = []
    question_lower = question.lower()
    
    # Pattern 1: "Will [Team] win..."
    match = re.search(r'will\s+([a-z\s]+?)\s+win', question_lower)
    if match:
        teams.append(normalize_team_name(match.group(1)))
    
    # Pattern 2: "[Team] wins the..."
    match = re.search(r'^([a-z\s]+?)\s+wins\s+the', question_lower)
    if match:
        teams.append(normalize_team_name(match.group(1)))
    
    # Pattern 3: "[Team] vs [Team]" or "[Team] - [Team]"
    match = re.search(r'([a-z\s]+?)\s+(?:vs\.?|v\.?|-)\s+([a-z\s]+)', question_lower)
    if match:
        teams.extend([normalize_team_name(match.group(1)), normalize_team_name(match.group(2))])
    
    return list(set(filter(None, teams)))


def extract_competition_from_polymarket(question: str, slug: str) -> Optional[str]:
    """
    Extract competition name from Polymarket data.
    
    Returns standardized competition name or None.
    """
    text = f"{question} {slug}".lower()
    
    competitions = {
        'premier league': ['premier league', 'epl', 'premier-league'],
        'la liga': ['la liga', 'laliga', 'la-liga'],
        'champions league': ['champions league', 'ucl', 'champions-league'],
        'serie a': ['serie a', 'serie-a', 'seriea'],
        'bundesliga': ['bundesliga', '1. bundesliga'],
        'ligue 1': ['ligue 1', 'ligue-1', 'ligue1'],
        'europa league': ['europa league', 'europa-league', 'uel'],
        'world cup': ['world cup', 'fifa world cup', 'worldcup'],
        'euro': ['euro 2024', 'euro 2025', 'uefa euro', 'euros'],
    }
    
    for standard_name, patterns in competitions.items():
        for pattern in patterns:
            if pattern in text:
                return standard_name
    
    return None


class EntityResolver:
    """
    Main class for resolving entities between StatsBomb and Polymarket datasets.
    """
    
    def __init__(self):
        self._sb_teams: Optional[pl.DataFrame] = None
        self._pm_teams: Optional[pl.DataFrame] = None
        self._team_mapping: Optional[dict] = None
        
    def load_statsbomb_teams(self) -> pl.DataFrame:
        """Load unique teams from StatsBomb matches."""
        if self._sb_teams is not None:
            return self._sb_teams
        
        matches = pl.read_parquet(STATSBOMB_DIR / "matches.parquet")
        
        # Get unique teams from both home and away
        home_teams = matches.select(
            pl.col("home_team").alias("team_name")
        ).unique()
        
        away_teams = matches.select(
            pl.col("away_team").alias("team_name")
        ).unique()
        
        teams = pl.concat([home_teams, away_teams]).unique()
        
        # Add normalized name
        teams = teams.with_columns(
            pl.col("team_name").map_elements(
                normalize_team_name, return_dtype=pl.Utf8
            ).alias("normalized_name")
        )
        
        self._sb_teams = teams.sort("team_name")
        return self._sb_teams
    
    def load_polymarket_teams(self) -> pl.DataFrame:
        """Extract team mentions from Polymarket markets."""
        if self._pm_teams is not None:
            return self._pm_teams
        
        markets = pl.read_parquet(POLYMARKET_DIR / "soccer_markets.parquet")
        
        # Extract teams from questions
        teams_list = []
        for row in markets.iter_rows(named=True):
            question = row.get("question", "")
            slug = row.get("slug", "")
            market_id = row.get("market_id", "")
            
            extracted = extract_team_from_polymarket_question(question)
            for team in extracted:
                if team:
                    teams_list.append({
                        "extracted_team": team,
                        "market_id": market_id,
                        "question": question,
                        "slug": slug,
                    })
        
        if teams_list:
            self._pm_teams = pl.DataFrame(teams_list)
        else:
            self._pm_teams = pl.DataFrame(schema={
                "extracted_team": pl.Utf8,
                "market_id": pl.Utf8,
                "question": pl.Utf8,
                "slug": pl.Utf8,
            })
        
        return self._pm_teams
    
    def build_team_mapping(self, similarity_threshold: float = 0.8) -> pl.DataFrame:
        """
        Build a mapping between StatsBomb teams and Polymarket team mentions.
        
        Uses fuzzy matching based on normalized names.
        
        Returns a DataFrame with columns:
            - sb_team: StatsBomb team name
            - pm_team: Polymarket extracted team
            - confidence: Match confidence (0-1)
        """
        sb_teams = self.load_statsbomb_teams()
        pm_teams = self.load_polymarket_teams()
        
        if pm_teams.is_empty():
            return pl.DataFrame(schema={
                "sb_team": pl.Utf8,
                "pm_team": pl.Utf8,
                "confidence": pl.Float64,
            })
        
        # Get unique PM teams
        pm_unique = pm_teams.select("extracted_team").unique()
        
        mappings = []
        for sb_row in sb_teams.iter_rows(named=True):
            sb_name = sb_row["team_name"]
            sb_norm = sb_row["normalized_name"]
            
            for pm_row in pm_unique.iter_rows(named=True):
                pm_name = pm_row["extracted_team"]
                
                # Calculate similarity
                confidence = self._string_similarity(sb_norm, pm_name)
                
                if confidence >= similarity_threshold:
                    mappings.append({
                        "sb_team": sb_name,
                        "pm_team": pm_name,
                        "confidence": confidence,
                    })
        
        if mappings:
            return pl.DataFrame(mappings).sort("confidence", descending=True)
        
        return pl.DataFrame(schema={
            "sb_team": pl.Utf8,
            "pm_team": pl.Utf8,
            "confidence": pl.Float64,
        })
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using a simple approach.
        
        Combines exact substring matching with token overlap.
        """
        if not s1 or not s2:
            return 0.0
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # One contains the other
        if s1 in s2 or s2 in s1:
            return 0.9
        
        # Token overlap (Jaccard similarity)
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_matched_markets_for_team(self, team_name: str) -> pl.DataFrame:
        """
        Get all Polymarket markets that mention a specific StatsBomb team.
        """
        mapping = self.build_team_mapping()
        pm_teams = self.load_polymarket_teams()
        
        # Find PM team names that match this SB team
        matched = mapping.filter(pl.col("sb_team") == team_name)
        
        if matched.is_empty():
            return pl.DataFrame()
        
        pm_names = matched["pm_team"].to_list()
        
        # Get all markets mentioning these teams
        return pm_teams.filter(pl.col("extracted_team").is_in(pm_names))
    
    def get_team_market_summary(self) -> pl.DataFrame:
        """
        Generate a summary of market coverage for each StatsBomb team.
        """
        sb_teams = self.load_statsbomb_teams()
        mapping = self.build_team_mapping()
        pm_teams = self.load_polymarket_teams()
        
        summaries = []
        for row in sb_teams.iter_rows(named=True):
            team = row["team_name"]
            
            # Get matched PM teams
            matched = mapping.filter(pl.col("sb_team") == team)
            
            if matched.is_empty():
                summaries.append({
                    "team": team,
                    "has_polymarket_data": False,
                    "market_count": 0,
                    "avg_confidence": 0.0,
                })
            else:
                pm_names = matched["pm_team"].to_list()
                market_count = pm_teams.filter(
                    pl.col("extracted_team").is_in(pm_names)
                )["market_id"].n_unique()
                
                summaries.append({
                    "team": team,
                    "has_polymarket_data": True,
                    "market_count": market_count,
                    "avg_confidence": matched["confidence"].mean(),
                })
        
        return pl.DataFrame(summaries).sort("market_count", descending=True)


# Convenience functions
def get_team_mapping() -> pl.DataFrame:
    """Get the team mapping between StatsBomb and Polymarket."""
    resolver = EntityResolver()
    return resolver.build_team_mapping()


def get_market_coverage_summary() -> pl.DataFrame:
    """Get a summary of Polymarket coverage for each team."""
    resolver = EntityResolver()
    return resolver.get_team_market_summary()


# Manual overrides for known mappings that fuzzy matching might miss
MANUAL_TEAM_MAPPINGS = {
    # StatsBomb name -> Polymarket normalized name
    "Arsenal": "arsenal",
    "Manchester City": "manchester city",
    "Manchester United": "manchester united",
    "Liverpool": "liverpool",
    "Chelsea": "chelsea",
    "Tottenham Hotspur": "tottenham",
    "Barcelona": "barcelona",
    "Real Madrid": "real madrid",
    "Bayern Munich": "bayern munich",
    "Paris Saint-Germain": "paris saint germain",
    "Juventus": "juventus",
    "Inter Milan": "inter milan",
    "AC Milan": "ac milan",
    "Borussia Dortmund": "borussia dortmund",
    "Atlético Madrid": "atletico madrid",
}


if __name__ == "__main__":
    print("=" * 60)
    print("  Entity Resolution: StatsBomb ↔ Polymarket")
    print("=" * 60)
    
    resolver = EntityResolver()
    
    print("\n--- StatsBomb Teams ---")
    sb_teams = resolver.load_statsbomb_teams()
    print(f"Total unique teams: {len(sb_teams)}")
    print(sb_teams.head(10))
    
    print("\n--- Polymarket Team Mentions ---")
    pm_teams = resolver.load_polymarket_teams()
    print(f"Total team mentions: {len(pm_teams)}")
    unique_pm = pm_teams.select("extracted_team").unique()
    print(f"Unique teams extracted: {len(unique_pm)}")
    print(unique_pm.head(10))
    
    print("\n--- Team Mapping (Fuzzy Match) ---")
    mapping = resolver.build_team_mapping()
    print(f"Total mappings found: {len(mapping)}")
    print(mapping.head(20))
    
    print("\n--- Market Coverage Summary ---")
    summary = resolver.get_team_market_summary()
    covered = summary.filter(pl.col("has_polymarket_data"))
    print(f"Teams with Polymarket coverage: {len(covered)} / {len(summary)}")
    print(covered.head(20))
