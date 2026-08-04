# Kalshi NBA Market Analysis

This repository is the foundation for **Can Data Beat the Market? Predicting NBA
Games and Evaluating Kalshi Odds**.

## Run

```bash
python src/collect_data.py
python src/clean_data.py
python src/analyze_data.py
python tests/test_analysis.py
```

The collector downloads and caches public Kalshi market/candlestick data and the
2025-26 Basketball Reference schedule. The cleaning module normalizes teams and
times, filters markets, matches contracts to games, and combines quote snapshots.
The analysis module creates features, the processed game-level CSV, summary JSON,
and figures.

## Visual styling libraries

The project creates its visuals directly instead of using Matplotlib or Seaborn.
The following libraries and resources control their appearance:

- **Pillow (`PIL`)** creates the four PNG charts. `Image` provides each
  1400-by-850-pixel canvas, `ImageDraw` draws axes, grid lines, bars, points,
  labels, and legends, and `ImageFont` supplies readable type. Drawing the charts
  directly makes the navy, blue, teal, orange, and gold palette consistent across
  every figure.
- **Arial**, loaded from the Windows system font directory through Pillow, is used
  for chart text in regular and bold weights.
- **NumPy** supports the visual calculations: evenly spaced axes and bins,
  quantile-based chart limits, and the fitted trend line in the recent-form plot.
  It does not render graphics itself.
- **pandas** prepares the values displayed in the visuals by binning probabilities,
  building cross-tabulations and grouped summaries, and filtering missing values.
  Like NumPy, it supports the charts but does not control their rendering style.

This approach keeps the standalone figures visually consistent without relying on
an external plotting theme or stylesheet.

## Project structure

- `src/collect_data.py`: downloads and caches source data without post-tipoff leakage.
- `src/clean_data.py`: cleans cached sources and builds the matched game dataset.
- `src/analyze_data.py`: constructs pregame features, summaries, and visualizations.
- `tests/test_analysis.py`: assertion-based checks for matching, probabilities, chronology, and feature leakage.
- `data/raw/`: shareable API and schedule source extracts.
- `data/processed/`: cleaned analytical dataset.
- `figures/`: report-ready visualizations.
- `reports/eda.pdf`: archived final version of the exploratory data analysis report.
