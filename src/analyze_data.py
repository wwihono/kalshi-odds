"""Create pregame features, summaries, and visuals for Kalshi NBA analysis."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "matched_games_with_quotes.csv"
SCHEDULE_PATH = ROOT / "data" / "raw" / "basketball_reference_schedule.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "figures"
WIDTH = 1400
HEIGHT = 850
NAVY = "#15324A"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
GRID = "#D9E2E8"
TEXT = "#17242E"
MUTED = "#5C6B76"
BACKGROUND = "#F7FAFC"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a readable Windows font for report-quality raster charts."""
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def safe_float(value: Any) -> float | None:
    """Convert finite numeric values to plain Python floats."""
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def team_snapshot(history: list[dict[str, Any]], games: int) -> tuple[float, float]:
    """Return prior-game win rate and point differential over a window."""
    recent = history[-games:]
    if not recent:
        return np.nan, np.nan
    return (
        float(np.mean([item["win"] for item in recent])),
        float(np.mean([item["point_diff"] for item in recent])),
    )


def build_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe rolling, rest, and Elo features chronologically."""
    games = schedule.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date").reset_index(drop=True)
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ratings: dict[str, float] = defaultdict(lambda: 1500.0)
    last_played: dict[str, pd.Timestamp] = {}
    rows = []
    for _, game in games.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        date = game["game_date"]
        row = {
            "game_date": date,
            "home_team": home,
            "away_team": away,
        }
        for window in (5, 10):
            home_win, home_diff = team_snapshot(histories[home], window)
            away_win, away_diff = team_snapshot(histories[away], window)
            row[f"home_win_pct_{window}"] = home_win
            row[f"away_win_pct_{window}"] = away_win
            row[f"recent_win_pct_{window}_diff"] = home_win - away_win
            row[f"home_point_diff_{window}"] = home_diff
            row[f"away_point_diff_{window}"] = away_diff
            row[f"recent_point_diff_{window}_diff"] = home_diff - away_diff
        for side, team in (("home", home), ("away", away)):
            previous = last_played.get(team)
            if previous is None:
                rest_days = np.nan
            else:
                rest_days = max((date - previous).days - 1, 0)
            row[f"{side}_rest_days"] = rest_days
            row[f"{side}_back_to_back"] = int(rest_days == 0) if not pd.isna(rest_days) else 0
        row["rest_diff"] = row["home_rest_days"] - row["away_rest_days"]
        row["home_elo"] = ratings[home]
        row["away_elo"] = ratings[away]
        row["elo_diff"] = ratings[home] - ratings[away]
        rows.append(row)

        expected_home = 1 / (1 + 10 ** (-(ratings[home] + 100 - ratings[away]) / 400))
        change = 20 * (int(game["home_win"]) - expected_home)
        ratings[home] += change
        ratings[away] -= change
        home_diff = float(game["home_score"] - game["away_score"])
        histories[home].append({"win": int(game["home_win"]), "point_diff": home_diff})
        histories[away].append(
            {"win": 1 - int(game["home_win"]), "point_diff": -home_diff}
        )
        last_played[home] = date
        last_played[away] = date
    return pd.DataFrame(rows)


def seven_number(series: pd.Series) -> dict[str, float | int | None]:
    """Return count plus the required seven-number quantitative summary."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": safe_float(values.mean()),
        "std": safe_float(values.std()),
        "min": safe_float(values.min()),
        "q1": safe_float(values.quantile(0.25)),
        "median": safe_float(values.median()),
        "q3": safe_float(values.quantile(0.75)),
        "max": safe_float(values.max()),
    }


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def chart_base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), title, fill=NAVY, font=font(40, True))
    draw.text((70, 95), subtitle, fill=MUTED, font=font(22))
    return image, draw


def draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = bounds
    draw.line((left, top, left, bottom), fill=TEXT, width=3)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=3)
    label_font = font(21, True)
    label_width = draw.textlength(x_label, font=label_font)
    draw.text(
        (left + (right - left - label_width) / 2, bottom + 55),
        x_label,
        fill=TEXT,
        font=label_font,
    )
    draw.text((left, top - 36), y_label, fill=TEXT, font=label_font)


def plot_calibration(table: pd.DataFrame, output: Path) -> None:
    """Create an observed-versus-implied calibration chart."""
    image, draw = chart_base(
        "Kalshi NBA calibration before tipoff",
        "Circle size represents the number of matched games in each probability bin.",
    )
    bounds = (150, 160, 1290, 700)
    draw_axes(draw, bounds, "Average Kalshi home-win probability", "Observed home-win rate")
    left, top, right, bottom = bounds
    for tick in np.linspace(0, 1, 6):
        x = left + tick * (right - left)
        y = bottom - tick * (bottom - top)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{tick:.1f}"
        draw.text((x - 18, bottom + 12), label, fill=MUTED, font=font(18))
        draw.text((left - 55, y - 10), label, fill=MUTED, font=font(18))
    draw.line((left, bottom, right, top), fill=ORANGE, width=4)
    max_count = max(int(table["count"].max()), 1)
    for _, row in table.dropna(subset=["mean_probability", "observed_win_rate"]).iterrows():
        x = left + float(row["mean_probability"]) * (right - left)
        y = bottom - float(row["observed_win_rate"]) * (bottom - top)
        radius = 12 + 22 * math.sqrt(float(row["count"]) / max_count)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=BLUE, outline=NAVY, width=3)
        draw.text((x + radius + 6, y - 12), str(int(row["count"])), fill=TEXT, font=font(17, True))
    draw.line((1010, 137, 1060, 137), fill=ORANGE, width=4)
    draw.text((1070, 125), "Perfect calibration", fill=MUTED, font=font(18))
    image.save(output, quality=95)


def plot_probability_distribution(data: pd.DataFrame, output: Path) -> None:
    """Create a stacked histogram of pregame probabilities by outcome."""
    image, draw = chart_base(
        "Distribution of pregame home-team probabilities",
        "Most observations cluster away from the extremes; color indicates the realized winner.",
    )
    bounds = (150, 170, 1290, 700)
    draw_axes(draw, bounds, "Kalshi probability bin", "Number of games")
    left, top, right, bottom = bounds
    labels = [f"{start}-{start + 10}%" for start in range(0, 100, 10)]
    category = pd.cut(
        data["kalshi_prob"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        labels=labels,
    )
    counts = pd.crosstab(category, data["home_win"]).reindex(labels, fill_value=0)
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    max_count = max(int(counts.sum(axis=1).max()), 1)
    gap = 14
    bar_width = ((right - left) - gap * 9) / 10
    for tick in range(0, max_count + 1, max(1, math.ceil(max_count / 5))):
        y = bottom - tick / max_count * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 55, y - 10), str(tick), fill=MUTED, font=font(18))
    for index, label in enumerate(labels):
        x0 = left + index * (bar_width + gap)
        away_count = int(counts.iloc[index, 0])
        home_count = int(counts.iloc[index, 1])
        away_height = away_count / max_count * (bottom - top)
        home_height = home_count / max_count * (bottom - top)
        draw.rectangle((x0, bottom - away_height, x0 + bar_width, bottom), fill=ORANGE)
        draw.rectangle(
            (x0, bottom - away_height - home_height, x0 + bar_width, bottom - away_height),
            fill=TEAL,
        )
        draw.text((x0 + 2, bottom + 12), label, fill=MUTED, font=font(14))
    draw.rectangle((940, 130, 970, 155), fill=TEAL)
    draw.text((980, 130), "Home team won", fill=MUTED, font=font(18))
    draw.rectangle((1120, 130, 1150, 155), fill=ORANGE)
    draw.text((1160, 130), "Away team won", fill=MUTED, font=font(18))
    image.save(output, quality=95)


def plot_volume_accuracy(table: pd.DataFrame, output: Path) -> None:
    """Create side-by-side bars for volume quartile Brier score and accuracy."""
    image, draw = chart_base(
        "Market accuracy by pregame volume quartile",
        "Lower Brier scores are better; accuracy uses a 50% probability threshold.",
    )
    left, top, right, bottom = (170, 170, 1260, 700)
    draw_axes(draw, (left, top, right, bottom), "Pregame volume quartile", "Metric value")
    for tick in np.linspace(0, 1, 6):
        y = bottom - tick * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 55, y - 10), f"{tick:.1f}", fill=MUTED, font=font(18))
    groups = table["volume_quartile"].tolist()
    centers = np.linspace(left + 120, right - 120, len(groups))
    width = 65
    for x, (_, row) in zip(centers, table.iterrows()):
        brier = float(row["brier_score"])
        accuracy = float(row["accuracy"])
        draw.rectangle((x - width - 5, bottom - brier * (bottom - top), x - 5, bottom), fill=BLUE)
        draw.rectangle((x + 5, bottom - accuracy * (bottom - top), x + width + 5, bottom), fill=GOLD)
        label = str(row["volume_quartile"])
        label_width, _ = text_size(draw, label, font(18, True))
        draw.text((x - label_width / 2, bottom + 18), label, fill=TEXT, font=font(18, True))
        draw.text((x - 56, bottom - brier * (bottom - top) - 30), f"{brier:.3f}", fill=NAVY, font=font(16, True))
        draw.text((x + 13, bottom - accuracy * (bottom - top) - 30), f"{accuracy:.2f}", fill=TEXT, font=font(16, True))
    draw.rectangle((940, 130, 970, 155), fill=BLUE)
    draw.text((980, 130), "Brier score", fill=MUTED, font=font(18))
    draw.rectangle((1110, 130, 1140, 155), fill=GOLD)
    draw.text((1150, 130), "Accuracy", fill=MUTED, font=font(18))
    image.save(output, quality=95)


def plot_form_relationship(data: pd.DataFrame, output: Path) -> None:
    """Plot recent point-differential advantage against market probability."""
    frame = data.dropna(subset=["recent_point_diff_5_diff", "kalshi_prob"])
    image, draw = chart_base(
        "Recent form and Kalshi probability",
        "Each point is a game; the fitted line summarizes the market's response to recent form.",
    )
    bounds = (160, 170, 1280, 700)
    draw_axes(draw, bounds, "Home minus away recent point differential (last 5)", "Kalshi home-win probability")
    left, top, right, bottom = bounds
    if frame.empty:
        draw.text((500, 400), "No complete observations", fill=MUTED, font=font(30, True))
        image.save(output)
        return
    x_values = frame["recent_point_diff_5_diff"].to_numpy(dtype=float)
    y_values = frame["kalshi_prob"].to_numpy(dtype=float)
    limit = max(10.0, float(np.nanquantile(np.abs(x_values), 0.98)))
    for tick in np.linspace(-limit, limit, 5):
        x = left + (tick + limit) / (2 * limit) * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x - 20, bottom + 12), f"{tick:.0f}", fill=MUTED, font=font(18))
    for tick in np.linspace(0, 1, 6):
        y = bottom - tick * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 55, y - 10), f"{tick:.1f}", fill=MUTED, font=font(18))
    for _, row in frame.iterrows():
        x_raw = max(-limit, min(limit, float(row["recent_point_diff_5_diff"])))
        x = left + (x_raw + limit) / (2 * limit) * (right - left)
        y = bottom - float(row["kalshi_prob"]) * (bottom - top)
        color = TEAL if int(row["home_win"]) else ORANGE
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    y_start = intercept + slope * (-limit)
    y_end = intercept + slope * limit
    draw.line(
        (left, bottom - y_start * (bottom - top), right, bottom - y_end * (bottom - top)),
        fill=NAVY,
        width=5,
    )
    draw.rectangle((930, 130, 960, 155), fill=TEAL)
    draw.text((970, 130), "Home win", fill=MUTED, font=font(18))
    draw.rectangle((1070, 130, 1100, 155), fill=ORANGE)
    draw.text((1110, 130), "Away win", fill=MUTED, font=font(18))
    image.save(output, quality=95)


def make_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Calculate all tables and scalar findings used by the report."""
    clipped = data["kalshi_prob"].clip(1e-6, 1 - 1e-6)
    data["brier"] = (data["kalshi_prob"] - data["home_win"]) ** 2
    data["log_loss"] = -(
        data["home_win"] * np.log(clipped)
        + (1 - data["home_win"]) * np.log(1 - clipped)
    )
    data["correct_prediction"] = (
        (data["kalshi_prob"] >= 0.5).astype(int) == data["home_win"]
    )
    data["absolute_error"] = (data["kalshi_prob"] - data["home_win"]).abs()
    probability_breaks = np.linspace(0, 1, 11)
    data["probability_bin"] = pd.cut(
        data["kalshi_prob"], bins=probability_breaks, include_lowest=True
    ).astype(str)
    try:
        data["volume_quartile"] = pd.qcut(
            data["pregame_volume"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
        ).astype(str)
    except ValueError:
        data["volume_quartile"] = "Insufficient variation"

    calibration = (
        data.groupby("probability_bin", observed=False)
        .agg(
            count=("home_win", "size"),
            mean_probability=("kalshi_prob", "mean"),
            observed_win_rate=("home_win", "mean"),
            brier_score=("brier", "mean"),
        )
        .reset_index()
    )
    calibration["calibration_gap"] = (
        calibration["observed_win_rate"] - calibration["mean_probability"]
    )
    volume = (
        data.groupby("volume_quartile", observed=False)
        .agg(
            count=("home_win", "size"),
            pregame_volume_median=("pregame_volume", "median"),
            brier_score=("brier", "mean"),
            accuracy=("correct_prediction", "mean"),
        )
        .reset_index()
    )
    desired_order = {"Q1 lowest": 0, "Q2": 1, "Q3": 2, "Q4 highest": 3}
    volume["_order"] = volume["volume_quartile"].map(desired_order).fillna(99)
    volume = volume.sort_values("_order").drop(columns="_order")

    quantitative = [
        "kalshi_prob",
        "pregame_volume",
        "bid_ask_spread",
        "recent_win_pct_5_diff",
        "recent_point_diff_5_diff",
        "rest_diff",
        "elo_diff",
        "brier",
    ]
    categorical = ["home_win", "price_source", "probability_bin", "volume_quartile"]
    numeric_summaries = {name: seven_number(data[name]) for name in quantitative}
    category_counts = {
        name: {str(key): int(value) for key, value in data[name].value_counts(dropna=False).items()}
        for name in categorical
    }
    missing = {
        name: {
            "count": int(data[name].isna().sum()),
            "percent": float(data[name].isna().mean() * 100),
        }
        for name in data.columns
    }
    anomaly_columns = [
        "game_date",
        "away_team",
        "home_team",
        "kalshi_prob",
        "home_win",
        "pregame_volume",
        "absolute_error",
    ]
    anomalies = data.nlargest(6, "absolute_error")[anomaly_columns].copy()
    anomalies["game_date"] = anomalies["game_date"].dt.strftime("%Y-%m-%d")
    ece = float(
        (
            calibration["count"]
            / calibration["count"].sum()
            * calibration["calibration_gap"].abs()
        ).sum()
    )
    return {
        "dimensions": {"rows": int(len(data)), "columns": int(data.shape[1])},
        "row_definition": "One completed 2025-26 NBA game matched to the Kalshi contract for the home team.",
        "overall": {
            "accuracy": float(data["correct_prediction"].mean()),
            "brier_score": float(data["brier"].mean()),
            "log_loss": float(data["log_loss"].mean()),
            "mean_probability": float(data["kalshi_prob"].mean()),
            "observed_home_win_rate": float(data["home_win"].mean()),
            "expected_calibration_error": ece,
            "median_minutes_before_tip": float(data["minutes_before_tip"].median()),
            "settlement_disagreements": int((~data["settlement_agrees"]).sum()),
        },
        "missingness": missing,
        "quantitative_summaries": numeric_summaries,
        "categorical_counts": category_counts,
        "calibration": calibration.where(pd.notna(calibration), None).to_dict("records"),
        "volume_accuracy": volume.where(pd.notna(volume), None).to_dict("records"),
        "largest_errors": anomalies.where(pd.notna(anomalies), None).to_dict("records"),
    }


def main() -> None:
    """Run the analysis and write processed data, figures, and summaries."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(RAW_PATH, parse_dates=["game_date", "game_time_utc", "open_time"])
    schedule = pd.read_csv(SCHEDULE_PATH, parse_dates=["game_date"])
    features = build_features(schedule)
    join_keys = ["game_date", "home_team", "away_team"]
    data = matched.merge(features, on=join_keys, how="left", validate="one_to_one")
    data = data[
        data["quote_status"].eq("available")
        & data["kalshi_prob"].between(0, 1, inclusive="both")
        & data["minutes_before_tip"].ge(15)
        & data["settlement_agrees"]
    ].copy()
    summary = make_summary(data)
    data.to_csv(PROCESSED_DIR / "kalshi_nba_eda.csv", index=False)
    with (PROCESSED_DIR / "eda_summary.json").open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)

    calibration = pd.DataFrame(summary["calibration"])
    volume = pd.DataFrame(summary["volume_accuracy"])
    plot_calibration(calibration, FIGURE_DIR / "calibration.png")
    plot_probability_distribution(data, FIGURE_DIR / "probability_distribution.png")
    plot_volume_accuracy(volume, FIGURE_DIR / "volume_accuracy.png")
    plot_form_relationship(data, FIGURE_DIR / "recent_form.png")
    print(json.dumps(summary["dimensions"] | summary["overall"], indent=2))


if __name__ == "__main__":
    main()
