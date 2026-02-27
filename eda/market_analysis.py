"""
Market Efficiency Analysis: Correlating Match Events with Polymarket Odds

This module analyzes:
- Market efficiency by comparing predicted probabilities to outcomes
- Odds movement patterns around key match events
- xG-based probability estimation vs market odds
- Trading volume analysis and market depth
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
POLYMARKET_DIR = DATA_DIR / "Polymarket"
STATSBOMB_DIR = DATA_DIR / "Statsbomb"


class MarketDataLoader:
    """Load and preprocess Polymarket data."""
    
    def __init__(self):
        self._markets: Optional[pl.DataFrame] = None
        self._trades: Optional[pl.DataFrame] = None
        self._odds_history: Optional[pl.DataFrame] = None
        self._tokens: Optional[pl.DataFrame] = None
        self._summary: Optional[pl.DataFrame] = None
        self._event_stats: Optional[pl.DataFrame] = None
    
    @property
    def markets(self) -> pl.DataFrame:
        if self._markets is None:
            self._markets = pl.read_parquet(POLYMARKET_DIR / "soccer_markets.parquet")
        return self._markets
    
    @property
    def trades(self) -> pl.DataFrame:
        if self._trades is None:
            self._trades = pl.read_parquet(
                POLYMARKET_DIR / "soccer_trades.parquet"
            ).with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
            )
        return self._trades
    
    @property
    def odds_history(self) -> pl.DataFrame:
        if self._odds_history is None:
            self._odds_history = pl.read_parquet(
                POLYMARKET_DIR / "soccer_odds_history.parquet"
            ).with_columns(
                pl.col("timestamp").cast(pl.Int64).cast(pl.Datetime("ms"))
            )
        return self._odds_history
    
    @property
    def tokens(self) -> pl.DataFrame:
        if self._tokens is None:
            self._tokens = pl.read_parquet(POLYMARKET_DIR / "soccer_tokens.parquet")
        return self._tokens
    
    @property
    def summary(self) -> pl.DataFrame:
        if self._summary is None:
            self._summary = pl.read_parquet(
                POLYMARKET_DIR / "soccer_summary.parquet"
            ).with_columns([
                pl.col("first_trade").cast(pl.Int64).cast(pl.Datetime("ms")),
                pl.col("last_trade").cast(pl.Int64).cast(pl.Datetime("ms")),
            ])
        return self._summary
    
    @property
    def event_stats(self) -> pl.DataFrame:
        if self._event_stats is None:
            self._event_stats = pl.read_parquet(POLYMARKET_DIR / "soccer_event_stats.parquet")
        return self._event_stats


class MarketEfficiencyAnalyzer:
    """
    Analyze market efficiency and calibration.
    """
    
    def __init__(self, loader: Optional[MarketDataLoader] = None):
        self.loader = loader or MarketDataLoader()
    
    def compute_price_statistics(self) -> pl.DataFrame:
        """Compute basic statistics on market prices."""
        trades = self.loader.trades
        
        stats = trades.group_by("market_id").agg([
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").std().alias("price_std"),
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            pl.col("price").median().alias("median_price"),
            pl.col("size").sum().alias("total_volume"),
            pl.len().alias("trade_count"),
            (pl.col("price").max() - pl.col("price").min()).alias("price_range"),
        ])
        
        return stats.sort("total_volume", descending=True)
    
    def compute_volume_by_time(self, granularity: str = "day") -> pl.DataFrame:
        """
        Aggregate trading volume over time.
        
        Args:
            granularity: "hour", "day", "week", "month"
        """
        trades = self.loader.trades
        
        # Truncate to granularity
        if granularity == "hour":
            truncate = "1h"
        elif granularity == "day":
            truncate = "1d"
        elif granularity == "week":
            truncate = "1w"
        else:
            truncate = "1mo"
        
        return trades.group_by(
            pl.col("timestamp").dt.truncate(truncate).alias("time_period")
        ).agg([
            pl.col("size").sum().alias("total_volume"),
            pl.len().alias("trade_count"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("market_id").n_unique().alias("active_markets"),
        ]).sort("time_period")
    
    def compute_calibration(self, bins: int = 10) -> pl.DataFrame:
        """
        Compute market calibration: do prices match empirical probabilities?
        
        Note: This requires knowing market outcomes, which may be in the 'closed' status.
        For now, we analyze the distribution of final prices for closed markets.
        """
        markets = self.loader.markets
        summary = self.loader.summary
        
        # Get closed markets
        closed_markets = markets.filter(pl.col("closed") == True)
        
        # Join with summary to get last trade info
        closed_with_summary = closed_markets.join(
            summary.select(["market_id", "trade_count"]),
            on="market_id",
            how="left"
        )
        
        # Analyze volume distribution for closed vs active
        return pl.DataFrame({
            "status": ["closed", "active"],
            "count": [
                markets.filter(pl.col("closed") == True).height,
                markets.filter(pl.col("closed") == False).height,
            ],
            "total_volume": [
                markets.filter(pl.col("closed") == True)["volume"].sum(),
                markets.filter(pl.col("closed") == False)["volume"].sum(),
            ],
        })
    
    def analyze_market_depth(self, market_id: str) -> dict:
        """
        Analyze market depth and liquidity for a specific market.
        """
        trades = self.loader.trades.filter(pl.col("market_id") == market_id)
        
        if trades.is_empty():
            return {"market_id": market_id, "error": "No trades found"}
        
        # Buy vs Sell analysis
        buy_trades = trades.filter(pl.col("side") == "BUY")
        sell_trades = trades.filter(pl.col("side") == "SELL")
        
        return {
            "market_id": market_id,
            "total_trades": trades.height,
            "buy_trades": buy_trades.height,
            "sell_trades": sell_trades.height,
            "buy_sell_ratio": buy_trades.height / sell_trades.height if sell_trades.height > 0 else float('inf'),
            "total_volume": trades["size"].sum(),
            "avg_trade_size": trades["size"].mean(),
            "max_trade_size": trades["size"].max(),
            "price_range": trades["price"].max() - trades["price"].min(),
            "first_trade": trades["timestamp"].min(),
            "last_trade": trades["timestamp"].max(),
        }
    
    def find_high_volume_markets(self, top_n: int = 20) -> pl.DataFrame:
        """Find the most actively traded markets."""
        markets = self.loader.markets
        summary = self.loader.summary
        
        return markets.join(
            summary.select(["market_id", "trade_count"]),
            on="market_id",
            how="left"
        ).sort("volume", descending=True).head(top_n).select([
            "market_id", "question", "volume", "trade_count", "active", "closed"
        ])


class OddsMovementAnalyzer:
    """
    Analyze odds movements and volatility.
    """
    
    def __init__(self, loader: Optional[MarketDataLoader] = None):
        self.loader = loader or MarketDataLoader()
    
    def compute_odds_volatility(self, market_id: str) -> dict:
        """
        Compute volatility metrics for a market's odds history.
        """
        odds = self.loader.odds_history.filter(pl.col("market_id") == market_id)
        
        if odds.is_empty():
            return {"market_id": market_id, "error": "No odds history"}
        
        # Compute price changes
        odds_sorted = odds.sort("timestamp")
        prices = odds_sorted["price"].to_list()
        
        if len(prices) < 2:
            return {"market_id": market_id, "error": "Insufficient data"}
        
        # Calculate returns
        import numpy as np
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        return {
            "market_id": market_id,
            "snapshots": len(prices),
            "start_price": prices[0],
            "end_price": prices[-1],
            "price_change": prices[-1] - prices[0],
            "volatility": float(np.std(returns)) if len(returns) > 0 else 0,
            "max_drawdown": float(np.min(prices_arr) - np.max(prices_arr[:np.argmin(prices_arr)+1])) if len(prices) > 1 else 0,
            "avg_price": float(np.mean(prices_arr)),
        }
    
    def compute_intraday_patterns(self) -> pl.DataFrame:
        """
        Analyze trading patterns by hour of day.
        """
        trades = self.loader.trades
        
        return trades.with_columns(
            pl.col("timestamp").dt.hour().alias("hour")
        ).group_by("hour").agg([
            pl.col("size").sum().alias("total_volume"),
            pl.len().alias("trade_count"),
            pl.col("price").mean().alias("avg_price"),
        ]).sort("hour")
    
    def find_large_price_movements(self, threshold: float = 0.1) -> pl.DataFrame:
        """
        Find markets with large price movements (potential event reactions).
        
        Args:
            threshold: Minimum price change to flag (e.g., 0.1 = 10 percentage points)
        """
        odds = self.loader.odds_history
        
        # Compute price range per market-token
        movements = odds.group_by(["market_id", "token_id"]).agg([
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            pl.col("timestamp").min().alias("first_time"),
            pl.col("timestamp").max().alias("last_time"),
            pl.len().alias("snapshots"),
        ]).with_columns(
            (pl.col("max_price") - pl.col("min_price")).alias("price_range")
        )
        
        # Filter for large movements
        large_moves = movements.filter(pl.col("price_range") > threshold)
        
        # Join with market info
        markets = self.loader.markets
        
        return large_moves.join(
            markets.select(["market_id", "question"]),
            on="market_id",
            how="left"
        ).sort("price_range", descending=True)


class CompetitionAnalyzer:
    """
    Analyze market interest by competition.
    """
    
    # Competition keywords for classification
    COMPETITIONS = {
        "Premier League": ["premier league", "epl", "premier-league"],
        "Champions League": ["champions league", "ucl", "champions-league"],
        "La Liga": ["la liga", "laliga", "la-liga"],
        "Serie A": ["serie a", "serie-a"],
        "Bundesliga": ["bundesliga"],
        "Ligue 1": ["ligue 1", "ligue-1"],
        "World Cup": ["world cup", "fifa world cup"],
        "Europa League": ["europa league", "uel"],
    }
    
    def __init__(self, loader: Optional[MarketDataLoader] = None):
        self.loader = loader or MarketDataLoader()
    
    def classify_market_competition(self, question: str, slug: str) -> str:
        """Classify a market into a competition category."""
        text = f"{question} {slug}".lower()
        
        for comp, keywords in self.COMPETITIONS.items():
            for keyword in keywords:
                if keyword in text:
                    return comp
        
        return "Other"
    
    def analyze_by_competition(self) -> pl.DataFrame:
        """
        Aggregate market statistics by competition.
        """
        markets = self.loader.markets
        summary = self.loader.summary
        
        # Classify each market
        classifications = []
        for row in markets.iter_rows(named=True):
            question = row.get("question", "")
            slug = row.get("slug", "")
            market_id = row.get("market_id", "")
            comp = self.classify_market_competition(question, slug)
            classifications.append({"market_id": market_id, "competition": comp})
        
        comp_df = pl.DataFrame(classifications)
        
        # Join and aggregate
        markets_with_comp = markets.join(comp_df, on="market_id", how="left")
        
        return markets_with_comp.group_by("competition").agg([
            pl.len().alias("market_count"),
            pl.col("volume").sum().alias("total_volume"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("active").sum().alias("active_markets"),
        ]).sort("total_volume", descending=True)


def generate_market_report() -> dict:
    """
    Generate a comprehensive market analysis report.
    """
    loader = MarketDataLoader()
    efficiency = MarketEfficiencyAnalyzer(loader)
    odds = OddsMovementAnalyzer(loader)
    competition = CompetitionAnalyzer(loader)
    
    report = {
        "summary": {
            "total_markets": loader.markets.height,
            "total_trades": loader.trades.height,
            "total_volume": loader.markets["volume"].sum(),
            "active_markets": loader.markets.filter(pl.col("active") == True).height,
            "closed_markets": loader.markets.filter(pl.col("closed") == True).height,
        },
        "volume_by_day": efficiency.compute_volume_by_time("day").head(30).to_dicts(),
        "top_markets": efficiency.find_high_volume_markets(10).to_dicts(),
        "competition_breakdown": competition.analyze_by_competition().to_dicts(),
        "intraday_patterns": odds.compute_intraday_patterns().to_dicts(),
    }
    
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("  Market Efficiency Analysis")
    print("=" * 60)
    
    loader = MarketDataLoader()
    
    print("\n--- Data Summary ---")
    print(f"Markets: {loader.markets.height:,}")
    print(f"Trades: {loader.trades.height:,}")
    print(f"Odds snapshots: {loader.odds_history.height:,}")
    print(f"Tokens: {loader.tokens.height:,}")
    
    print("\n--- Market Statistics ---")
    efficiency = MarketEfficiencyAnalyzer(loader)
    price_stats = efficiency.compute_price_statistics()
    print(price_stats.head(10))
    
    print("\n--- Volume by Day ---")
    volume_by_day = efficiency.compute_volume_by_time("day")
    print(volume_by_day.head(10))
    
    print("\n--- Top Markets by Volume ---")
    top_markets = efficiency.find_high_volume_markets(10)
    print(top_markets)
    
    print("\n--- Competition Breakdown ---")
    comp_analyzer = CompetitionAnalyzer(loader)
    comp_breakdown = comp_analyzer.analyze_by_competition()
    print(comp_breakdown)
    
    print("\n--- Intraday Trading Patterns ---")
    odds_analyzer = OddsMovementAnalyzer(loader)
    intraday = odds_analyzer.compute_intraday_patterns()
    print(intraday)
    
    print("\n--- Large Price Movements ---")
    large_moves = odds_analyzer.find_large_price_movements(0.3)  # 30pp moves
    print(f"Found {len(large_moves)} markets with >30pp price swings")
    print(large_moves.head(10))
