"""Download and cache public Kalshi and NBA schedule data."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from clean_data import (
    clean_schedule,
    combine_matches_and_quotes,
    match_home_markets,
    parse_market_time,
    prepare_markets,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
BR_BASE = "https://www.basketball-reference.com"
BR_INDEX = f"{BR_BASE}/leagues/NBA_2026_games.html"
USER_AGENT = "CSE163-Kalshi-NBA-EDA/1.0 (educational project)"
SEASON_START = pd.Timestamp("2025-10-01", tz="UTC")


def request_bytes(url: str, attempts: int = 4, timeout: int = 90) -> bytes:
    """Return URL content with bounded retries for transient failures.

    Inputs:
        url: Public endpoint to request.
        attempts: Maximum number of request attempts.
        timeout: Per-request timeout in seconds.
    Returns:
        The downloaded response body as bytes.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable retry state")


def request_json(url: str) -> dict[str, Any]:
    """Download a JSON object from a public endpoint.

    Inputs:
        url: Public JSON endpoint to request.
    Returns:
        The decoded top-level JSON object.
    """
    return json.loads(request_bytes(url).decode("utf-8"))


def fetch_paginated_markets(path: str, source: str) -> list[dict[str, Any]]:
    """Fetch KXNBAGAME pages until the requested season has been passed.

    Inputs:
        path: Kalshi API path for live or historical markets.
        source: Label stored on each row to identify its API tier.
    Returns:
        Raw market dictionaries collected across all relevant pages.
    """
    markets = []
    cursor = ""
    for _ in range(12):
        query = {"series_ticker": "KXNBAGAME", "limit": 1000}
        if cursor:
            query["cursor"] = cursor
        url = f"{API_BASE}/{path}?{urllib.parse.urlencode(query)}"
        payload = request_json(url)
        page = payload.get("markets", [])
        for market in page:
            market["_api_source"] = source
            markets.append(market)
        cursor = payload.get("cursor", "")
        dated = [parse_market_time(item) for item in page]
        dated = [item for item in dated if item is not None]
        if source == "historical" and dated and min(dated) < SEASON_START:
            break
        if not cursor or not page:
            break
    return markets


def fetch_schedule() -> pd.DataFrame:
    """Download the monthly Basketball Reference schedule tables.

    Inputs:
        None. Uses the configured 2025-26 Basketball Reference URLs.
    Returns:
        A cleaned DataFrame containing completed season games.
    """
    request_bytes(BR_INDEX)
    months = [
        "october",
        "november",
        "december",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
    ]
    frames = []
    for month in months:
        url = f"{BR_BASE}/leagues/NBA_2026_games-{month}.html"
        html = request_bytes(url).decode("utf-8")
        table = pd.read_html(io.StringIO(html), attrs={"id": "schedule"})[0]
        table["source_url"] = url
        frames.append(table)
        time.sleep(0.25)
    return clean_schedule(pd.concat(frames, ignore_index=True))


def _close_value(block: object) -> float | None:
    """Read a numeric close value from a candlestick price block.

    Inputs:
        block: A possible nested Kalshi price dictionary.
    Returns:
        The close price as a float, or None when it is absent or invalid.
    """
    if not isinstance(block, dict):
        return None
    raw = block.get("close_dollars", block.get("close"))
    try:
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return None


def fetch_pregame_quote(row: dict[str, Any]) -> dict[str, Any]:
    """Fetch the latest quote ending at least 15 minutes before tipoff.

    Inputs:
        row: A matched game/market record with ticker, tipoff, and API source.
    Returns:
        A quote record containing the selected probability and pregame metadata,
        or a record whose status marks the quote as missing.
    """
    ticker = row["ticker"]
    tipoff = pd.to_datetime(row["tipoff_time_utc"], utc=True)
    end_ts = int(tipoff.timestamp()) - 15 * 60
    start_ts = end_ts - int(timedelta(hours=24).total_seconds())
    if row["api_source"] == "historical":
        path = f"historical/markets/{ticker}/candlesticks"
    else:
        path = f"series/KXNBAGAME/markets/{ticker}/candlesticks"
    query = urllib.parse.urlencode(
        {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1}
    )
    url = f"{API_BASE}/{path}?{query}"
    try:
        candles = request_json(url).get("candlesticks", [])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        candles = []
    candles = [item for item in candles if int(item["end_period_ts"]) <= end_ts]
    candles.sort(key=lambda item: int(item["end_period_ts"]))
    selected = None
    bid = ask = trade = None
    for item in reversed(candles):
        candidate_bid = _close_value(item.get("yes_bid"))
        candidate_ask = _close_value(item.get("yes_ask"))
        candidate_trade = _close_value(item.get("price"))
        quote_ok = (
            candidate_bid is not None
            and candidate_ask is not None
            and 0 <= candidate_bid <= candidate_ask <= 1
        )
        if quote_ok or candidate_trade is not None:
            selected = item
            bid, ask, trade = candidate_bid, candidate_ask, candidate_trade
            break
    if selected is None:
        return {"ticker": ticker, "quote_status": "missing"}
    probability = (bid + ask) / 2 if bid is not None and ask is not None else trade
    price_source = (
        "bid_ask_midpoint"
        if bid is not None and ask is not None
        else "last_trade"
    )
    volumes = []
    for item in candles:
        try:
            volumes.append(float(item.get("volume_fp", item.get("volume", 0)) or 0))
        except (TypeError, ValueError):
            volumes.append(0.0)
    quote_time = int(selected["end_period_ts"])
    return {
        "ticker": ticker,
        "quote_status": "available",
        "kalshi_prob": probability,
        "yes_bid": bid,
        "yes_ask": ask,
        "last_trade": trade,
        "bid_ask_spread": None if bid is None or ask is None else ask - bid,
        "price_source": price_source,
        "quote_time_utc": datetime.fromtimestamp(
            quote_time, tz=timezone.utc
        ).isoformat(),
        "minutes_before_tip": (int(tipoff.timestamp()) - quote_time) / 60,
        "pregame_volume": sum(volumes),
        "candlestick_count": len(candles),
        "request_url": url,
    }


def collect_quotes(matches: pd.DataFrame) -> pd.DataFrame:
    """Download pregame quotes concurrently while retaining failures.

    Inputs:
        matches: Matched game/market rows requiring candlestick quotes.
    Returns:
        One available-or-missing pregame quote record per contract ticker.
    """
    records = matches.to_dict("records")
    cache_path = RAW_DIR / "pregame_quote_partial.jsonl"
    quotes = []
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as source:
            quotes = [json.loads(line) for line in source if line.strip()]
        quotes = list({row["ticker"]: row for row in quotes}.values())
    completed = {row["ticker"] for row in quotes}
    remaining = [row for row in records if row["ticker"] not in completed]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_pregame_quote, row): row for row in remaining
        }
        for count, future in enumerate(as_completed(futures), start=1):
            quote = future.result()
            quotes.append(quote)
            with cache_path.open("a", encoding="utf-8") as cache:
                cache.write(json.dumps(quote) + "\n")
            if count % 100 == 0:
                print(f"Fetched {count}/{len(remaining)} quotes", flush=True)
    return pd.DataFrame(quotes)


def load_or_fetch_sources() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load cached sources, downloading each source when it is absent.

    Inputs:
        None. Uses the configured paths under data/raw.
    Returns:
        The NBA schedule DataFrame and list of raw Kalshi market dictionaries.
    """
    schedule_path = RAW_DIR / "basketball_reference_schedule.csv"
    market_path = RAW_DIR / "kalshi_market_responses.json"
    if schedule_path.exists():
        schedule = pd.read_csv(
            schedule_path, parse_dates=["game_date", "tipoff_time_utc"]
        )
    else:
        schedule = fetch_schedule()
        schedule.to_csv(schedule_path, index=False)
    if market_path.exists():
        with market_path.open(encoding="utf-8") as source:
            markets = json.load(source)
    else:
        markets = fetch_paginated_markets("historical/markets", "historical")
        markets += fetch_paginated_markets("markets", "live")
        with market_path.open("w", encoding="utf-8") as output:
            json.dump(markets, output, indent=2)
    return schedule, markets


def main() -> None:
    """Collect sources and quotes, then write the cleaned matched extract.

    Inputs:
        None. Reads caches or public endpoints using module constants.
    Returns:
        None. Writes raw caches, the matched CSV, and a collection audit.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    schedule, markets = load_or_fetch_sources()
    market_frame = prepare_markets(markets)
    matched = match_home_markets(schedule, market_frame)
    quotes = collect_quotes(matched)
    quote_records = quotes.where(pd.notna(quotes), None).to_dict("records")
    with (RAW_DIR / "pregame_quote_snapshots.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(quote_records, output, indent=2)
    merged = combine_matches_and_quotes(matched, quotes)
    merged.to_csv(RAW_DIR / "matched_games_with_quotes.csv", index=False)
    audit = {
        "schedule_games": int(len(schedule)),
        "season_markets": int(len(market_frame)),
        "matched_games": int(len(matched)),
        "available_quotes": int((merged["quote_status"] == "available").sum()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (RAW_DIR / "collection_audit.json").open("w", encoding="utf-8") as output:
        json.dump(audit, output, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
