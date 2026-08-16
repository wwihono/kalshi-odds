"""Clean and combine cached NBA schedule, market, and quote data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EASTERN = ZoneInfo("America/New_York")
SEASON_START = pd.Timestamp("2025-10-01", tz="UTC")
SEASON_END = pd.Timestamp("2026-07-01", tz="UTC")

TEAM_CITIES = {
    "atlanta": "Atlanta Hawks",
    "boston": "Boston Celtics",
    "brooklyn": "Brooklyn Nets",
    "charlotte": "Charlotte Hornets",
    "chicago": "Chicago Bulls",
    "cleveland": "Cleveland Cavaliers",
    "dallas": "Dallas Mavericks",
    "denver": "Denver Nuggets",
    "detroit": "Detroit Pistons",
    "golden state": "Golden State Warriors",
    "houston": "Houston Rockets",
    "indiana": "Indiana Pacers",
    "la clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
    "los angeles c": "Los Angeles Clippers",
    "los angeles l": "Los Angeles Lakers",
    "memphis": "Memphis Grizzlies",
    "miami": "Miami Heat",
    "milwaukee": "Milwaukee Bucks",
    "minnesota": "Minnesota Timberwolves",
    "new orleans": "New Orleans Pelicans",
    "new york": "New York Knicks",
    "oklahoma city": "Oklahoma City Thunder",
    "orlando": "Orlando Magic",
    "philadelphia": "Philadelphia 76ers",
    "phoenix": "Phoenix Suns",
    "portland": "Portland Trail Blazers",
    "sacramento": "Sacramento Kings",
    "san antonio": "San Antonio Spurs",
    "toronto": "Toronto Raptors",
    "utah": "Utah Jazz",
    "washington": "Washington Wizards",
}
TEAM_NAMES = TEAM_CITIES | {name.lower(): name for name in TEAM_CITIES.values()}


def canonical_team(value: object) -> str | None:
    """Normalize Kalshi and Basketball Reference team labels.

    Inputs:
        value: A team label or a missing value from either source.
    Returns:
        The canonical full NBA team name, or None for a missing input.
    """
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).strip().lower().split())
    return TEAM_NAMES.get(text, str(value).strip())


def canonical_market_team(value: object, ticker: object) -> str | None:
    """Normalize a Kalshi team label, using its ticker when LA is ambiguous.

    Inputs:
        value: The team label from a Kalshi Yes contract.
        ticker: The contract ticker whose final segment identifies the team.
    Returns:
        The canonical NBA team name, or the generic normalized label when the
        ticker does not resolve an ambiguous Los Angeles value.
    """
    normalized = canonical_team(value)
    if normalized is None:
        return None
    if normalized.lower() not in {"la", "los angeles"}:
        return normalized
    team_code = str(ticker).strip().upper().rsplit("-", maxsplit=1)[-1]
    if team_code == "LAC":
        return "Los Angeles Clippers"
    if team_code == "LAL":
        return "Los Angeles Lakers"
    return normalized


def parse_market_time(market: dict[str, Any]) -> pd.Timestamp | None:
    """Return the best valid scheduled game time exposed by Kalshi.

    Inputs:
        market: One raw Kalshi market record.
    Returns:
        A timezone-aware UTC timestamp, or None when no valid time exists.
    """
    raw_time = market.get("occurrence_datetime")
    raw_time = raw_time or market.get("expected_expiration_time")
    if not raw_time:
        return None
    timestamp = pd.to_datetime(raw_time, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


def clean_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Normalize a combined Basketball Reference schedule table.

    Inputs:
        schedule: Raw monthly schedule rows from Basketball Reference.
    Returns:
        Completed games with normalized teams, UTC tipoffs, scores, and outcomes.
    """
    schedule = schedule[schedule["Date"].astype(str) != "Date"].copy()
    schedule["game_date"] = pd.to_datetime(schedule["Date"], errors="coerce")
    start_text = schedule["Start (ET)"].astype(str).str.strip()
    start_text = start_text.str.replace(r"([ap])$", r"\1m", regex=True).str.upper()
    local_tipoff = pd.to_datetime(
        schedule["game_date"].dt.strftime("%Y-%m-%d") + " " + start_text,
        format="%Y-%m-%d %I:%M%p",
        errors="coerce",
    )
    schedule["tipoff_time_utc"] = (
        local_tipoff.dt.tz_localize(
            EASTERN, ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    )
    schedule["away_team"] = schedule["Visitor/Neutral"].map(canonical_team)
    schedule["home_team"] = schedule["Home/Neutral"].map(canonical_team)
    schedule["away_score"] = pd.to_numeric(schedule["PTS"], errors="coerce")
    schedule["home_score"] = pd.to_numeric(schedule["PTS.1"], errors="coerce")
    required = [
        "game_date",
        "tipoff_time_utc",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
    ]
    schedule = schedule.dropna(subset=required)
    schedule["home_win"] = (
        schedule["home_score"] > schedule["away_score"]
    ).astype(int)
    schedule["point_diff"] = schedule["home_score"] - schedule["away_score"]
    keep = required + ["home_win", "point_diff", "source_url"]
    return (
        schedule[keep]
        .drop_duplicates()
        .sort_values("game_date")
        .reset_index(drop=True)
    )


def prepare_markets(markets: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten settled season markets and normalize their fields.

    Inputs:
        markets: Raw Kalshi market dictionaries from live and historical APIs.
    Returns:
        One normalized row per unique settled in-season contract.
    """
    rows = []
    for market in markets:
        game_time = parse_market_time(market)
        if game_time is None or not (SEASON_START <= game_time < SEASON_END):
            continue
        result = market.get("result")
        if result not in {"yes", "no"}:
            continue
        rows.append(
            {
                "ticker": market.get("ticker"),
                "event_ticker": market.get("event_ticker"),
                "yes_team": canonical_market_team(
                    market.get("yes_sub_title"), market.get("ticker")
                ),
                "game_time_utc": game_time,
                "game_date": game_time.tz_convert(EASTERN)
                .tz_localize(None)
                .normalize(),
                "open_time": pd.to_datetime(
                    market.get("open_time"), utc=True, errors="coerce"
                ),
                "result": result,
                "settlement_value": float(
                    market.get("settlement_value_dollars") or (result == "yes")
                ),
                "total_volume": float(market.get("volume_fp") or 0),
                "api_source": market.get("_api_source"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("game_time_utc").drop_duplicates("ticker", keep="last")


def match_home_markets(
    schedule: pd.DataFrame, markets: pd.DataFrame
) -> pd.DataFrame:
    """Match each completed game to its home-team Kalshi contract.

    Inputs:
        schedule: Cleaned NBA schedule and result rows.
        markets: Normalized settled Kalshi contract rows.
    Returns:
        Schedule rows joined to unique home-team contracts within eight hours.
    """
    event_teams = (
        markets.groupby("event_ticker")["yes_team"]
        .agg(lambda values: frozenset(values.dropna()))
        .to_dict()
    )
    candidates = markets[markets["yes_team"].notna()].copy()
    matched = []
    for game_index, game in schedule.iterrows():
        tipoff = pd.to_datetime(game["tipoff_time_utc"], utc=True)
        time_gap = (candidates["game_time_utc"] - tipoff).abs()
        mask = (
            (candidates["yes_team"] == game["home_team"])
            & (time_gap <= pd.Timedelta(hours=8))
            & candidates["event_ticker"].map(
                lambda event: game["away_team"] in event_teams.get(event, set())
            )
        )
        choices = candidates.loc[mask].copy()
        if choices.empty:
            continue
        choices["time_gap_hours"] = (
            time_gap.loc[choices.index].dt.total_seconds() / 3600
        )
        choice = choices.sort_values(["time_gap_hours", "game_time_utc"]).iloc[0]
        row = game.to_dict() | choice.to_dict()
        row["kalshi_event_date"] = choice["game_date"]
        row["game_date"] = game["game_date"]
        row["tipoff_time_utc"] = game["tipoff_time_utc"]
        row["schedule_row"] = int(game_index)
        matched.append(row)
    result = pd.DataFrame(matched)
    if result.empty:
        return result
    return (
        result.sort_values(["time_gap_hours", "schedule_row"])
        .drop_duplicates("ticker", keep="first")
        .sort_values("schedule_row")
        .reset_index(drop=True)
    )


def combine_matches_and_quotes(
    matched: pd.DataFrame, quotes: pd.DataFrame
) -> pd.DataFrame:
    """Merge matched games with quotes and derive validated identifiers.

    Inputs:
        matched: Games already matched to home-team Kalshi contracts.
        quotes: One pregame quote record per contract ticker.
    Returns:
        Game-level rows with quote fields, settlement checks, and unique IDs.
    """
    merged = matched.merge(quotes, on="ticker", how="left", validate="one_to_one")
    merged["market_home_win"] = merged["settlement_value"].round().astype(int)
    merged["settlement_agrees"] = merged["market_home_win"] == merged["home_win"]
    merged["game_id"] = (
        merged["game_date"].dt.strftime("%Y-%m-%d")
        + "_"
        + merged["away_team"].str.replace(" ", "_", regex=False)
        + "_at_"
        + merged["home_team"].str.replace(" ", "_", regex=False)
    )
    return merged.drop(columns=["time_gap_hours", "result"], errors="ignore")


def main() -> None:
    """Rebuild the cleaned matched dataset from cached raw extracts.

    Inputs:
        None. Cached files are read from data/raw.
    Returns:
        None. Writes data/raw/matched_games_with_quotes.csv and prints its size.
    """
    schedule = pd.read_csv(
        RAW_DIR / "basketball_reference_schedule.csv",
        parse_dates=["game_date", "tipoff_time_utc"],
    )
    with (RAW_DIR / "kalshi_market_responses.json").open(encoding="utf-8") as source:
        markets = json.load(source)
    with (RAW_DIR / "pregame_quote_snapshots.json").open(encoding="utf-8") as source:
        quotes = pd.DataFrame(json.load(source))
    matched = match_home_markets(schedule, prepare_markets(markets))
    cleaned = combine_matches_and_quotes(matched, quotes)
    output = RAW_DIR / "matched_games_with_quotes.csv"
    cleaned.to_csv(output, index=False)
    print(f"Wrote {len(cleaned)} cleaned games to {output}")


if __name__ == "__main__":
    main()
