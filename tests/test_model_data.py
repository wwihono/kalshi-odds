"""Runnable assertion-based tests for model cleaning and evaluation."""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model_data  # noqa: E402
from model_data import (  # noqa: E402
    FEATURE_NAMES,
    chronological_split,
    classification_metrics,
    clean_model_data,
    estimate_taker_fee,
    fit_preprocessor,
    predict_gradient_boosting,
    predict_model_family,
    predict_probabilities,
    predict_random_forest,
    run_analysis,
    select_model_families,
    select_regularization,
    simulate_trades,
    summarize_fitted_model,
    summarize_trades,
    train_gradient_boosting,
    train_logistic_regression,
    train_model_family,
    train_random_forest,
    transform_features,
)


def example_games(row_count: int = 40) -> pd.DataFrame:
    """Create deterministic, chronologically ordered synthetic game rows.

    Inputs:
        row_count: Number of synthetic games to create.
    Returns:
        A valid game-level DataFrame suitable for model-pipeline tests.
    """
    rows = []
    signals = [-2.0, -1.0, 1.0, 2.0]
    for index in range(row_count):
        signal = signals[index % len(signals)]
        probability = {-2.0: 0.30, -1.0: 0.40, 1.0: 0.60, 2.0: 0.70}[signal]
        rows.append(
            {
                "game_id": f"game-{index:03d}",
                "ticker": f"ticker-{index:03d}",
                "game_date": pd.Timestamp("2025-10-01") + pd.Timedelta(days=index),
                "tipoff_time_utc": (
                    pd.Timestamp("2025-10-01 23:00", tz="UTC")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "home_win": int(signal > 0),
                "quote_status": "available",
                "kalshi_prob": probability,
                "yes_bid": probability - 0.01,
                "yes_ask": probability + 0.01,
                "minutes_before_tip": 15.0,
                "settlement_agrees": True,
                "home_back_to_back": index % 2,
                "away_back_to_back": (index // 2) % 2,
                "recent_win_pct_5_diff": signal / 4,
                "recent_win_pct_10_diff": signal / 5,
                "recent_point_diff_5_diff": signal * 3,
                "recent_point_diff_10_diff": signal * 2,
                "rest_diff": float((index % 3) - 1),
                "elo_diff": signal * 50,
            }
        )
    return pd.DataFrame(rows)


def test_clean_model_data() -> None:
    """Check audits for unusable quotes, timing failures, and duplicates.

    Inputs:
        None. Modifies a synthetic game table inside the test.
    Returns:
        None. Raises AssertionError if cleaning or audit counts are wrong.
    """
    data = example_games(8)
    data.loc[1, ["yes_bid", "yes_ask"]] = [0.0, 1.0]
    data.loc[2, "minutes_before_tip"] = 14.0
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    cleaned, audit = clean_model_data(data)
    assert len(cleaned) == 6
    assert audit["wide_spread_rows"] == 1
    assert audit["invalid_timing_rows"] == 1
    assert audit["duplicate_rows"] == 1
    assert "back_to_back_diff" in cleaned.columns


def test_chronological_split() -> None:
    """Check that chronological periods are ordered and date-disjoint.

    Inputs:
        None. Uses a cleaned synthetic game table.
    Returns:
        None. Raises AssertionError if split ordering is invalid.
    """
    data, _ = clean_model_data(example_games())
    training, validation, test = chronological_split(data)
    assert len(training) + len(validation) + len(test) == len(data)
    assert training["game_date"].max() < validation["game_date"].min()
    assert validation["game_date"].max() < test["game_date"].min()


def test_fit_preprocessor() -> None:
    """Check preprocessing parameters fitted from supplied training data.

    Inputs:
        None. Builds a numeric training table inside the test.
    Returns:
        None. Raises AssertionError if medians, means, or scales are wrong.
    """
    frame = pd.DataFrame({"a": [1.0, np.nan, 5.0], "b": [4.0, 4.0, 4.0]})
    preprocessor = fit_preprocessor(frame, ["a", "b"])
    assert preprocessor["medians"]["a"] == 3.0
    assert preprocessor["means"]["a"] == 3.0
    assert preprocessor["scales"]["b"] == 1.0


def test_transform_features() -> None:
    """Check that transformation imputes missing values and stays finite.

    Inputs:
        None. Builds fitted parameters and scoring rows inside the test.
    Returns:
        None. Raises AssertionError if the transformed matrix is incorrect.
    """
    training = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [2.0, 2.0, 2.0]})
    preprocessor = fit_preprocessor(training, ["a", "b"])
    transformed = transform_features(
        pd.DataFrame({"a": [np.nan, 5.0], "b": [2.0, 2.0]}), preprocessor
    )
    assert transformed.shape == (2, 2)
    assert transformed[0, 0] == 0
    assert np.isfinite(transformed).all()


def test_train_logistic_regression() -> None:
    """Check that gradient descent learns a simple separating coefficient.

    Inputs:
        None. Uses a four-row linearly separable feature matrix.
    Returns:
        None. Raises AssertionError if model weight shape or direction is wrong.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    target = np.array([0, 0, 1, 1])
    weights = train_logistic_regression(
        features, target, l2_penalty=0.0, epochs=1000
    )
    assert weights.coef_.shape == (1, 1)
    assert weights.coef_[0, 0] > 0


def test_predict_probabilities() -> None:
    """Check prediction ordering and probability bounds.

    Inputs:
        None. Uses fixed weights and three ordered feature rows.
    Returns:
        None. Raises AssertionError if probabilities are invalid.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    model = train_logistic_regression(
        features, np.array([0, 0, 1, 1]), l2_penalty=0.0
    )
    probabilities = predict_probabilities(
        model, np.array([[-2.0], [0.0], [2.0]])
    )
    assert np.all(np.diff(probabilities) > 0)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_train_random_forest() -> None:
    """Check forest construction and bounded probability predictions.

    Inputs:
        None. Fits a small forest to a deterministic binary pattern.
    Returns:
        None. Raises AssertionError if tree count or probabilities are invalid.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 4)
    target = np.array([0, 0, 1, 1] * 4)
    model = train_random_forest(
        features,
        target,
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=1,
    )
    probabilities = predict_random_forest(model, np.array([[-1.5], [1.5]]))
    assert len(model.estimators_) == 12
    assert 0 <= probabilities[0] < probabilities[1] <= 1


def test_predict_random_forest() -> None:
    """Check that forest prediction returns one bounded value per row.

    Inputs:
        None. Fits a small forest and scores two rows.
    Returns:
        None. Raises AssertionError if prediction shape or bounds are wrong.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 3)
    target = np.array([0, 0, 1, 1] * 3)
    model = train_random_forest(
        features, target, n_estimators=4, max_depth=2, min_samples_leaf=1
    )
    probabilities = predict_random_forest(model, np.array([[-1.0], [1.0]]))
    assert probabilities.shape == (2,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_train_gradient_boosting() -> None:
    """Check boosting learns higher probability for a positive signal.

    Inputs:
        None. Fits a small boosted ensemble to a repeated binary pattern.
    Returns:
        None. Raises AssertionError if its fitted probabilities are unordered.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 4)
    target = np.array([0, 0, 1, 1] * 4)
    model = train_gradient_boosting(
        features,
        target,
        n_estimators=20,
        max_depth=1,
        learning_rate=0.2,
        min_samples_leaf=1,
    )
    probabilities = predict_gradient_boosting(
        model, np.array([[-1.5], [1.5]])
    )
    assert len(model.estimators_) == 20
    assert probabilities[0] < probabilities[1]


def test_predict_gradient_boosting() -> None:
    """Check boosting prediction returns one bounded value per row.

    Inputs:
        None. Fits a small boosting model and scores two rows.
    Returns:
        None. Raises AssertionError if prediction shape or bounds are wrong.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 3)
    target = np.array([0, 0, 1, 1] * 3)
    model = train_gradient_boosting(
        features,
        target,
        n_estimators=5,
        max_depth=1,
        learning_rate=0.2,
        min_samples_leaf=1,
    )
    probabilities = predict_gradient_boosting(
        model, np.array([[-1.0], [1.0]])
    )
    assert probabilities.shape == (2,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_classification_metrics() -> None:
    """Check metric formulas against small hand-calculated values.

    Inputs:
        None. Uses two fixed outcomes and probabilities.
    Returns:
        None. Raises AssertionError if any metric differs from expectation.
    """
    metrics = classification_metrics([0, 1], [0.1, 0.8])
    assert metrics["accuracy"] == 1.0
    assert abs(metrics["brier_score"] - 0.025) < 1e-12
    expected_log_loss = -(np.log(0.9) + np.log(0.8)) / 2
    assert abs(metrics["log_loss"] - expected_log_loss) < 1e-12


def test_select_regularization() -> None:
    """Check that model selection returns a tested regularization candidate.

    Inputs:
        None. Fits two candidates on synthetic chronological periods.
    Returns:
        None. Raises AssertionError if selection output is incomplete.
    """
    data, _ = clean_model_data(example_games())
    training, validation, _ = chronological_split(data)
    candidates = [0.0, 0.1]
    selected, tuning = select_regularization(
        training,
        validation,
        feature_names=FEATURE_NAMES,
        l2_values=candidates,
        epochs=500,
    )
    assert selected in candidates
    assert len(tuning) == len(candidates)
    assert all("brier_score" in row for row in tuning)


def test_train_model_family() -> None:
    """Check the common trainer dispatches to all three model families.

    Inputs:
        None. Fits each family to a small deterministic feature matrix.
    Returns:
        None. Raises AssertionError if any family cannot be fitted and scored.
    """
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 3)
    target = np.array([0, 0, 1, 1] * 3)
    parameters = {
        "logistic_regression": {"l2_penalty": 0.0},
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        "gradient_boosting": {
            "n_estimators": 5,
            "max_depth": 1,
            "learning_rate": 0.2,
            "min_samples_leaf": 1,
        },
    }
    for family, family_parameters in parameters.items():
        model = train_model_family(
            family,
            features,
            target,
            family_parameters,
            logistic_epochs=200,
        )
        probabilities = predict_model_family(family, model, features)
        assert probabilities.shape == target.shape
        assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_predict_model_family() -> None:
    """Check the common predictor rejects unsupported family names.

    Inputs:
        None. Calls the predictor with a deliberately invalid family name.
    Returns:
        None. Raises AssertionError unless the expected ValueError occurs.
    """
    try:
        predict_model_family("unsupported", None, np.array([[0.0]]))
    except ValueError:
        return
    raise AssertionError("unsupported model families must raise ValueError")


def test_select_model_families() -> None:
    """Check validation tuning compares every planned model family.

    Inputs:
        None. Tunes one compact candidate per family on synthetic periods.
    Returns:
        None. Raises AssertionError if selection metadata is incomplete.
    """
    data, _ = clean_model_data(example_games())
    training, validation, _ = chronological_split(data)
    forest_grid = [
        {"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 1}
    ]
    boosting_grid = [
        {
            "n_estimators": 5,
            "max_depth": 1,
            "learning_rate": 0.2,
            "min_samples_leaf": 1,
        }
    ]
    selected, best_parameters, tuning = select_model_families(
        training,
        validation,
        l2_values=[0.0],
        random_forest_grid=forest_grid,
        gradient_boosting_grid=boosting_grid,
        logistic_epochs=200,
    )
    expected = {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert selected in expected
    assert set(best_parameters) == expected
    assert set(tuning) == expected
    assert all(len(rows) == 1 for rows in tuning.values())


def test_summarize_fitted_model() -> None:
    """Check fitted-model metadata is compact and JSON-compatible.

    Inputs:
        None. Summarizes fixed logistic weights and feature names.
    Returns:
        None. Raises AssertionError if coefficients or row count are lost.
    """
    model = train_logistic_regression(
        np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        np.array([0, 0, 1, 1]),
        l2_penalty=0.1,
    )
    summary = summarize_fitted_model(
        "logistic_regression",
        model,
        {"l2_penalty": 0.1},
        ["signal"],
        20,
    )
    assert summary["training_rows"] == 20
    assert summary["coefficients"]["signal"] > 0


def test_estimate_taker_fee() -> None:
    """Check that taker fees round a single-contract charge up to cents.

    Inputs:
        None. Uses fixed contract prices.
    Returns:
        None. Raises AssertionError if fee calculations are incorrect.
    """
    assert estimate_taker_fee(0.50) == 0.02
    assert estimate_taker_fee(0.00) == 0.00


def test_simulate_trades() -> None:
    """Check that simulation buys the correctly favored contract at the ask.

    Inputs:
        None. Uses two synthetic games and fixed model probabilities.
    Returns:
        None. Raises AssertionError if side, price, or profit is incorrect.
    """
    games = example_games(2)
    trades = simulate_trades(games, [0.90, 0.10], thresholds=[0.10])
    assert trades["side"].tolist() == ["home_yes", "home_no"]
    assert trades["purchase_price"].tolist() == [0.31, 0.61]
    assert np.allclose(
        trades["profit"], trades["payout"] - trades["purchase_price"] - trades["fee"]
    )


def test_summarize_trades() -> None:
    """Check trade-summary position counts, profit, and ROI.

    Inputs:
        None. Summarizes a fixed two-position simulation.
    Returns:
        None. Raises AssertionError if an aggregate is incorrect.
    """
    games = example_games(2)
    trades = simulate_trades(games, [0.90, 0.10], thresholds=[0.10])
    summary = summarize_trades(trades, thresholds=[0.10])[0]
    assert summary["positions"] == 2
    assert summary["wins"] == 1
    assert summary["roi"] == summary["total_profit"] / summary["total_cost"]


def test_run_analysis() -> None:
    """Check that the full workflow produces test-only model predictions.

    Inputs:
        None. Runs the analysis on deterministic synthetic games.
    Returns:
        None. Raises AssertionError if outputs or split safeguards are wrong.
    """
    forest_grid = [
        {"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 1}
    ]
    boosting_grid = [
        {
            "n_estimators": 5,
            "max_depth": 1,
            "learning_rate": 0.2,
            "min_samples_leaf": 1,
        }
    ]
    results, model_ready, trades = run_analysis(
        example_games(),
        l2_values=[0.0],
        random_forest_grid=forest_grid,
        gradient_boosting_grid=boosting_grid,
        logistic_epochs=200,
    )
    test_rows = results["split"]["test"]["rows"]
    assert results["cleaning"]["output_rows"] == 40
    assert model_ready.loc[model_ready["split"] == "test", "model_probability"].notna().sum() == test_rows
    assert model_ready.loc[model_ready["split"] != "test", "model_probability"].isna().all()
    families = {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert results["selected_family"] in families
    assert set(results["final_models"]) == families
    assert families.issubset(results["test_metrics"])
    for family in families:
        column = f"{family}_probability"
        assert model_ready.loc[model_ready["split"] == "test", column].notna().sum() == test_rows
        assert model_ready.loc[model_ready["split"] != "test", column].isna().all()
    assert set(trades.columns).issuperset({"threshold", "profit", "cumulative_profit"})


def test_main() -> None:
    """Check that the runnable entry point writes all three output files.

    Inputs:
        None. Redirects module input and output paths to a temporary folder.
    Returns:
        None. Raises AssertionError if an expected output file is not written.
    """
    fake_results = {"test_metrics": {"model": {"accuracy": 1.0}}}
    fake_games = pd.DataFrame({"game_id": ["game-001"]})
    fake_trades = pd.DataFrame({"profit": [0.25]})
    original_input = model_data.INPUT_PATH
    original_output = model_data.OUTPUT_DIR
    original_run_analysis = model_data.run_analysis
    with tempfile.TemporaryDirectory() as temp_directory:
        temporary_path = Path(temp_directory)
        input_path = temporary_path / "input.csv"
        example_games(3).to_csv(input_path, index=False)
        try:
            model_data.INPUT_PATH = input_path
            model_data.OUTPUT_DIR = temporary_path
            model_data.run_analysis = lambda raw_data: (
                fake_results,
                fake_games,
                fake_trades,
            )
            with redirect_stdout(io.StringIO()):
                model_data.main()
        finally:
            model_data.INPUT_PATH = original_input
            model_data.OUTPUT_DIR = original_output
            model_data.run_analysis = original_run_analysis
        assert (temporary_path / "model_ready_games.csv").exists()
        assert (temporary_path / "model_results.json").exists()
        assert (temporary_path / "simulated_trades.csv").exists()


def main() -> None:
    """Run all model tests without requiring a separate test framework.

    Inputs:
        None.
    Returns:
        None. Prints a success message or propagates an assertion failure.
    """
    test_clean_model_data()
    test_chronological_split()
    test_fit_preprocessor()
    test_transform_features()
    test_train_logistic_regression()
    test_predict_probabilities()
    test_train_random_forest()
    test_predict_random_forest()
    test_train_gradient_boosting()
    test_predict_gradient_boosting()
    test_classification_metrics()
    test_select_regularization()
    test_train_model_family()
    test_predict_model_family()
    test_select_model_families()
    test_summarize_fitted_model()
    test_estimate_taker_fee()
    test_simulate_trades()
    test_summarize_trades()
    test_run_analysis()
    test_main()
    print("All model tests passed.")


if __name__ == "__main__":
    main()
