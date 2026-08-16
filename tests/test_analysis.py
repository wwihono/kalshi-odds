"""Runnable assertion-based tests for the Kalshi NBA analysis pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_data import (  # noqa: E402
    canonical_market_team,
    canonical_team,
    prepare_markets,
)
from analyze_data import build_features  # noqa: E402


def test_team_normalization() -> None:
    """Check that Kalshi city labels map to Basketball Reference names.

    Inputs:
        None. Uses fixed representative team labels.
    Returns:
        None. Raises AssertionError if normalization is incorrect.
    """
    assert canonical_team("LA Lakers") == "Los Angeles Lakers"
    assert canonical_team("Oklahoma City") == "Oklahoma City Thunder"
    assert canonical_team("San Antonio Spurs") == "San Antonio Spurs"
    assert canonical_team("Los Angeles C") == "Los Angeles Clippers"
    assert canonical_team("Los Angeles L") == "Los Angeles Lakers"


def test_market_team_normalization() -> None:
    """Check ticker-based normalization of ambiguous Los Angeles labels.

    Inputs:
        None. Uses the LA label and ticker formats observed in the raw data.
    Returns:
        None. Raises AssertionError if either LA team is misidentified.
    """
    assert (
        canonical_market_team("LA", "KXNBAGAME-25NOV06LACPHX-LAC")
        == "Los Angeles Clippers"
    )
    assert (
        canonical_market_team("Los Angeles", "KXNBAGAME-25NOV03LALPOR-LAL")
        == "Los Angeles Lakers"
    )


def test_market_cleaning() -> None:
    """Check that only settled in-season markets survive cleaning.

    Inputs:
        None. Builds a small raw market list inside the test.
    Returns:
        None. Raises AssertionError if filtering or normalization fails.
    """
    markets = [
        {
            "ticker": "valid",
            "event_ticker": "event",
            "yes_sub_title": "LA Lakers",
            "occurrence_datetime": "2025-11-01T02:00:00Z",
            "open_time": "2025-10-30T00:00:00Z",
            "result": "yes",
            "volume_fp": "12",
            "_api_source": "historical",
        },
        {
            "ticker": "unsettled",
            "occurrence_datetime": "2025-11-02T02:00:00Z",
            "result": "",
        },
    ]
    cleaned = prepare_markets(markets)
    assert cleaned["ticker"].tolist() == ["valid"]
    assert cleaned.loc[cleaned.index[0], "yes_team"] == "Los Angeles Lakers"


def test_rolling_features_use_only_prior_games() -> None:
    """Check that rolling features use only previously completed games.

    Inputs:
        None. Builds a two-game chronological schedule inside the test.
    Returns:
        None. Raises AssertionError if current or future results leak.
    """
    schedule = pd.DataFrame(
        [
            {
                "game_date": "2025-10-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 110,
                "away_score": 100,
                "home_win": 1,
            },
            {
                "game_date": "2025-10-03",
                "home_team": "B",
                "away_team": "A",
                "home_score": 120,
                "away_score": 90,
                "home_win": 1,
            },
        ]
    )
    features = build_features(schedule)
    assert np.isnan(features.loc[0, "home_win_pct_5"])
    assert features.loc[1, "home_win_pct_5"] == 0
    assert features.loc[1, "away_win_pct_5"] == 1
    assert features.loc[1, "home_rest_days"] == 1


def test_processed_dataset() -> None:
    """Validate keys, prices, timing, settlement, and the Brier score.

    Inputs:
        None. Reads the generated EDA CSV and summary JSON.
    Returns:
        None. Raises AssertionError when an output invariant is violated.
    """
    data_path = ROOT / "data" / "processed" / "kalshi_nba_eda.csv"
    summary_path = ROOT / "data" / "processed" / "eda_summary.json"
    data = pd.read_csv(data_path)
    with summary_path.open(encoding="utf-8") as source:
        summary = json.load(source)
    assert not data.empty
    assert data["game_id"].is_unique
    assert data["ticker"].is_unique
    assert data["kalshi_prob"].between(0, 1, inclusive="both").all()
    assert data["minutes_before_tip"].ge(15).all()
    assert data["settlement_agrees"].all()
    assert data[["home_team", "away_team", "home_win"]].notna().all().all()
    recalculated = ((data["kalshi_prob"] - data["home_win"]) ** 2).mean()
    assert abs(recalculated - summary["overall"]["brier_score"]) < 1e-12
    assert summary["dimensions"]["rows"] == len(data)


def main() -> None:
    """Run all EDA tests without requiring a separate test framework.

    Inputs:
        None.
    Returns:
        None. Prints a success message or propagates an assertion failure.
    """
    test_team_normalization()
    test_market_team_normalization()
    test_market_cleaning()
    test_rolling_features_use_only_prior_games()
    test_processed_dataset()
    print("All analysis tests passed.")


if __name__ == "__main__":
    main()
