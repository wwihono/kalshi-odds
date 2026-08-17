# Kalshi NBA Market Analysis

This repository supports **Can Data Beat the Market? Predicting NBA Games and
Evaluating Kalshi Odds**. It collects and cleans 2025-26 NBA/Kalshi data,
constructs pregame-only team features, trains logistic-regression, random-forest,
and gradient-boosting models, compares them with Kalshi on a later test period,
and simulates threshold trades with the validation-selected model.

## Requirements

- Python 3.11 or newer
- Internet access only if the raw sources are refreshed
- Packages in `requirements.txt`: pandas, NumPy, scikit-learn, Pillow, lxml, and
  flake8

The three classifiers use scikit-learn, one of the data-science libraries taught
in CSE 163. pandas and NumPy handle preprocessing and evaluation calculations.

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the analysis

The repository includes cached raw extracts, so the normal reproducibility path
does not need to call either public website.

1. Rebuild the matched game/quote table from the cached schedule, markets, and
   quote snapshots:

   ```bash
   python src/clean_data.py
   ```

2. Rebuild leakage-safe rolling features, the EDA table, summary JSON, and four
   exploratory figures:

   ```bash
   python src/analyze_data.py
   ```

3. Clean the model inputs, create the 60/20/20 chronological periods, select the
   hyperparameters and winning family on validation data, retrain each family's
   best candidate on the first 80%, and evaluate once on the final period:

   ```bash
   python src/model_data.py
   ```

4. Run the assertion-based test programs and the required style check:

   ```bash
   python tests/test_analysis.py
   python tests/test_model_data.py
   python -m flake8 src tests
   ```

To refresh all public source data instead of using the cached files, run
`python src/collect_data.py` before step 1. Collection paginates the Kalshi API
and downloads per-market candles, so it is much slower and depends on the
continued availability and format of both external sources.

## Model and cleaning decisions

- Rows are sorted chronologically, and game dates are never split between two
  periods. The normalized clean sample contains 1,069 games: 650 train, 206
  validation, and 213 test.
- Kalshi's shortened `Los Angeles C`, `Los Angeles L`, `LA`, and `Los Angeles`
  labels are resolved to the Clippers or Lakers, using the contract ticker when
  the displayed label is ambiguous. All 30 NBA teams are represented.
- One `$0.00/$1.00` quote is excluded because its full-dollar bid/ask spread is
  non-informative. The cleaning audit records every exclusion.
- Missing early-season rolling values are median-imputed using development data
  only. Scaling values are also fitted without using the test period.
- Features contain only prior-game recent form, rest/back-to-back differences,
  and Elo differences. Kalshi prices and current-game outcomes are not model
  features. Because every target is the home-team outcome, the model intercept
  represents the common home-court baseline.
- The planned logistic-regression, random-forest, and gradient-boosting families
  are tuned separately. The family with the lowest validation Brier score is
  selected before test scoring and supplies the trade-simulation probabilities.
- Simulations buy one contract at the displayed ask when model edge reaches 5%,
  10%, or 15%. Taker fees use `ceil(0.07 * contracts * price * (1-price))`
  rounded to cents, consistent with [Kalshi's February 2026 general fee
  schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf). This is a
  historical simulation, not a trading recommendation.

## Generated outputs

- `data/processed/kalshi_nba_eda.csv`: game-level EDA table.
- `data/processed/eda_summary.json`: EDA summaries and Kalshi calibration metrics.
- `data/processed/model_ready_games.csv`: cleaned rows, chronological labels, and
  test-only model probabilities.
- `data/processed/model_results.json`: cleaning audit, split dates, validation
  tuning for all three families, selected hyperparameters, test metrics, and
  trade summaries.
- `data/processed/simulated_trades.csv`: position-level prices, fees, profit, and
  cumulative profit for each edge threshold.
- `figures/`: programmatically generated EDA charts.
- `reports/eda.pdf`: archived exploratory data analysis report.

## Python files

- `src/collect_data.py`: downloads and caches public Kalshi markets/candles and
  Basketball Reference schedules while enforcing the pre-tipoff cutoff.
- `src/clean_data.py`: normalizes teams and times, validates results, matches each
  game to its home-team contract, and combines matches with quote snapshots.
- `src/analyze_data.py`: constructs leakage-safe rolling, rest, and Elo features;
  computes EDA summaries; and generates figures.
- `src/model_data.py`: cleans model rows, makes chronological splits, trains and
  validates logistic regression, random forest, and gradient boosting; evaluates
  the held-out test period; and simulates threshold trades.
- `tests/test_analysis.py`: tests team normalization, market preparation, rolling
  feature chronology, processed keys, quote timing, settlement, and Brier-score
  reproduction.
- `tests/test_model_data.py`: tests every model-pipeline function, including
  cleaning, splitting, preprocessing, training, prediction, metrics, model
  selection, fees, trading, summaries, and the complete in-memory workflow.

## Visualization libraries

Pillow draws the four PNG charts directly. NumPy supplies bins, axes, quantiles,
and fitted lines, while pandas prepares grouped values. Arial is used when it is
available on Windows. The models use scikit-learn's course-standard classifiers.
