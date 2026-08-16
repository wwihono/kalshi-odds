"""Clean inputs, compare three model families, and simulate test trades."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "kalshi_nba_eda.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
FEATURE_NAMES = [
    "recent_win_pct_5_diff",
    "recent_win_pct_10_diff",
    "recent_point_diff_5_diff",
    "recent_point_diff_10_diff",
    "rest_diff",
    "back_to_back_diff",
    "elo_diff",
]
DEFAULT_L2_VALUES = (0.0, 0.01, 0.1, 1.0, 10.0)
DEFAULT_RANDOM_FOREST_GRID = (
    {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 10},
    {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 10},
    {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 8},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 8},
)
DEFAULT_GRADIENT_BOOSTING_GRID = (
    {
        "n_estimators": 50,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 100,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.10,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.10,
        "min_samples_leaf": 10,
    },
)
MODEL_FAMILIES = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15)


def clean_model_data(
    data: pd.DataFrame, max_bid_ask_spread: float = 0.25
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Validate and filter game rows before chronological modeling.

    Inputs:
        data: Processed game rows containing quotes, outcomes, and features.
        max_bid_ask_spread: Largest quote spread treated as informative.
    Returns:
        The cleaned chronological rows and a dictionary of exclusion counts.
    """
    required = {
        "game_id",
        "ticker",
        "game_date",
        "tipoff_time_utc",
        "home_win",
        "quote_status",
        "kalshi_prob",
        "yes_bid",
        "yes_ask",
        "minutes_before_tip",
        "settlement_agrees",
        "home_back_to_back",
        "away_back_to_back",
    } | (set(FEATURE_NAMES) - {"back_to_back_diff"})
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")
    if not 0 <= max_bid_ask_spread <= 1:
        raise ValueError("max_bid_ask_spread must be between 0 and 1")

    cleaned = data.copy()
    input_rows = len(cleaned)
    cleaned["game_date"] = pd.to_datetime(cleaned["game_date"], errors="coerce")
    numeric = [
        "home_win",
        "kalshi_prob",
        "yes_bid",
        "yes_ask",
        "minutes_before_tip",
        "home_back_to_back",
        "away_back_to_back",
    ] + [name for name in FEATURE_NAMES if name != "back_to_back_diff"]
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    settlement_agrees = cleaned["settlement_agrees"]
    if settlement_agrees.dtype != bool:
        settlement_agrees = settlement_agrees.astype(str).str.lower().eq("true")
    cleaned["settlement_agrees"] = settlement_agrees

    valid_core = (
        cleaned["game_date"].notna()
        & cleaned["game_id"].notna()
        & cleaned["ticker"].notna()
        & cleaned["home_win"].isin([0, 1])
    )
    invalid_core_rows = int((~valid_core).sum())
    cleaned = cleaned.loc[valid_core].copy()

    valid_quote = (
        cleaned["quote_status"].eq("available")
        & cleaned["kalshi_prob"].between(0, 1, inclusive="both")
        & cleaned["yes_bid"].between(0, 1, inclusive="both")
        & cleaned["yes_ask"].between(0, 1, inclusive="both")
        & cleaned["yes_bid"].le(cleaned["yes_ask"])
    )
    invalid_quote_rows = int((~valid_quote).sum())
    cleaned = cleaned.loc[valid_quote].copy()
    cleaned["bid_ask_spread"] = cleaned["yes_ask"] - cleaned["yes_bid"]

    informative_quote = cleaned["bid_ask_spread"].le(max_bid_ask_spread)
    wide_spread_rows = int((~informative_quote).sum())
    cleaned = cleaned.loc[informative_quote].copy()

    valid_timing = cleaned["minutes_before_tip"].ge(15)
    invalid_timing_rows = int((~valid_timing).sum())
    cleaned = cleaned.loc[valid_timing].copy()

    valid_outcome = cleaned["settlement_agrees"]
    invalid_outcome_rows = int((~valid_outcome).sum())
    cleaned = cleaned.loc[valid_outcome].copy()
    cleaned = cleaned.sort_values(
        ["game_date", "tipoff_time_utc", "game_id"]
    ).reset_index(drop=True)
    duplicate_rows = int(
        (
            cleaned.duplicated("game_id", keep="first")
            | cleaned.duplicated("ticker", keep="first")
        ).sum()
    )
    cleaned = cleaned.drop_duplicates("game_id", keep="first")
    cleaned = cleaned.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    cleaned["home_win"] = cleaned["home_win"].astype(int)
    cleaned["back_to_back_diff"] = (
        cleaned["home_back_to_back"] - cleaned["away_back_to_back"]
    )
    audit: dict[str, int | float] = {
        "input_rows": int(input_rows),
        "invalid_core_rows": invalid_core_rows,
        "invalid_quote_rows": invalid_quote_rows,
        "wide_spread_rows": wide_spread_rows,
        "invalid_timing_rows": invalid_timing_rows,
        "settlement_disagreement_rows": invalid_outcome_rows,
        "duplicate_rows": duplicate_rows,
        "output_rows": int(len(cleaned)),
        "max_bid_ask_spread": float(max_bid_ask_spread),
    }
    return cleaned, audit


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test periods without splitting a game date.

    Inputs:
        data: Clean game rows containing dates and stable identifiers.
        train_fraction: Approximate earliest share assigned to training.
        validation_fraction: Approximate following share assigned to validation.
    Returns:
        Chronologically ordered training, validation, and test DataFrames.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test period")
    if len(data) < 3:
        raise ValueError("at least three rows are required for a chronological split")
    ordered = data.copy()
    ordered["game_date"] = pd.to_datetime(ordered["game_date"], errors="raise")
    ordered = ordered.sort_values(
        ["game_date", "tipoff_time_utc", "game_id"]
    ).reset_index(drop=True)
    row_count = len(ordered)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    for name, boundary in (("train", train_end), ("validation", validation_end)):
        boundary_date = ordered.loc[boundary - 1, "game_date"]
        while boundary < row_count and ordered.loc[boundary, "game_date"] == boundary_date:
            boundary += 1
        if name == "train":
            train_end = boundary
        else:
            validation_end = boundary
    if not 0 < train_end < validation_end < row_count:
        raise ValueError("date grouping left an empty chronological period")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )


def fit_preprocessor(
    training_data: pd.DataFrame, feature_names: Sequence[str]
) -> dict[str, Any]:
    """Fit median imputation and standardization using training rows only.

    Inputs:
        training_data: Rows allowed to determine preprocessing values.
        feature_names: Ordered names of numeric model features.
    Returns:
        Feature order plus fitted medians, means, and standard deviations.
    """
    features = list(feature_names)
    numeric = training_data[features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    if medians.isna().any():
        missing = medians[medians.isna()].index.tolist()
        raise ValueError(f"Training features contain no values: {missing}")
    imputed = numeric.fillna(medians)
    means = imputed.mean()
    scales = imputed.std(ddof=0).replace(0, 1.0)
    return {
        "feature_names": features,
        "medians": {name: float(medians[name]) for name in features},
        "means": {name: float(means[name]) for name in features},
        "scales": {name: float(scales[name]) for name in features},
    }


def transform_features(
    data: pd.DataFrame, preprocessor: dict[str, Any]
) -> np.ndarray:
    """Apply a fitted training-only preprocessor to feature rows.

    Inputs:
        data: Rows whose model features should be transformed.
        preprocessor: Training-fitted feature order and numeric parameters.
    Returns:
        A finite, standardized NumPy feature matrix.
    """
    features = preprocessor["feature_names"]
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    medians = pd.Series(preprocessor["medians"])
    means = pd.Series(preprocessor["means"])
    scales = pd.Series(preprocessor["scales"])
    transformed = (numeric.fillna(medians) - means) / scales
    values = transformed.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Transformed model features must be finite")
    return values


def train_logistic_regression(
    features: np.ndarray,
    target: np.ndarray,
    l2_penalty: float,
    epochs: int = 4000,
) -> LogisticRegression:
    """Fit scikit-learn logistic regression with L2 regularization.

    Inputs:
        features: Finite numeric matrix with one row per observation.
        target: Binary outcome vector aligned to features.
        l2_penalty: Nonnegative coefficient-regularization strength.
        epochs: Maximum solver iterations before convergence is stopped.
    Returns:
        A fitted scikit-learn LogisticRegression classifier.
    """
    x_values = np.asarray(features, dtype=float)
    y_values = np.asarray(target, dtype=float)
    if x_values.ndim != 2 or len(x_values) != len(y_values) or len(y_values) == 0:
        raise ValueError("features and target must contain matching nonempty rows")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("training inputs must be finite")
    if not np.isin(y_values, [0, 1]).all():
        raise ValueError("logistic target values must be 0 or 1")
    if l2_penalty < 0 or epochs <= 0:
        raise ValueError("epochs must be positive and l2 may not be negative")
    inverse_regularization = 1e12 if l2_penalty == 0 else 1 / l2_penalty
    model = LogisticRegression(
        C=inverse_regularization,
        solver="lbfgs",
        max_iter=epochs,
        random_state=163,
    )
    model.fit(x_values, y_values.astype(int))
    return model


def predict_probabilities(
    model: LogisticRegression, features: np.ndarray
) -> np.ndarray:
    """Return positive-class probabilities for a fitted logistic model.

    Inputs:
        model: Fitted scikit-learn logistic-regression classifier.
        features: Standardized numeric rows to score.
    Returns:
        One home-win probability per input row.
    """
    x_values = np.asarray(features, dtype=float)
    if x_values.ndim != 2 or not hasattr(model, "predict_proba"):
        raise ValueError("a fitted logistic model and feature matrix are required")
    return model.predict_proba(x_values)[:, 1]


def train_random_forest(
    features: np.ndarray,
    target: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 8,
    random_seed: int = 163,
) -> RandomForestClassifier:
    """Fit a scikit-learn random-forest probability classifier.

    Inputs:
        features: Finite numeric matrix with one row per observation.
        target: Binary outcome vector aligned to features.
        n_estimators: Number of bootstrapped trees in the ensemble.
        max_depth: Largest number of split levels in each tree.
        min_samples_leaf: Smallest number of rows allowed in a leaf.
        random_seed: Seed controlling bootstrap and feature sampling.
    Returns:
        A fitted scikit-learn RandomForestClassifier.
    """
    x_values = np.asarray(features, dtype=float)
    y_values = np.asarray(target, dtype=float)
    if x_values.ndim != 2 or len(x_values) != len(y_values) or len(y_values) == 0:
        raise ValueError("features and target must contain matching nonempty rows")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("random-forest training inputs must be finite")
    if not np.isin(y_values, [0, 1]).all():
        raise ValueError("random-forest target values must be 0 or 1")
    if n_estimators <= 0 or max_depth <= 0 or min_samples_leaf <= 0:
        raise ValueError("random-forest parameters must be positive")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(x_values, y_values.astype(int))
    return model


def predict_random_forest(
    model: RandomForestClassifier, features: np.ndarray
) -> np.ndarray:
    """Return probabilities from a fitted random-forest classifier.

    Inputs:
        model: Fitted classifier returned by train_random_forest.
        features: Finite numeric feature rows to score.
    Returns:
        One home-win probability per input row.
    """
    x_values = np.asarray(features, dtype=float)
    if x_values.ndim != 2 or not hasattr(model, "predict_proba"):
        raise ValueError("a fitted random forest and feature matrix are required")
    return model.predict_proba(x_values)[:, 1]


def train_gradient_boosting(
    features: np.ndarray,
    target: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 2,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 10,
    random_seed: int = 163,
) -> GradientBoostingClassifier:
    """Fit a scikit-learn gradient-boosting probability classifier.

    Inputs:
        features: Finite numeric matrix with one row per observation.
        target: Binary outcome vector aligned to features.
        n_estimators: Number of sequential boosting stages.
        max_depth: Largest number of split levels in each weak learner.
        learning_rate: Shrinkage applied to each boosting stage.
        min_samples_leaf: Smallest number of rows allowed in a leaf.
        random_seed: Seed used to make model fitting reproducible.
    Returns:
        A fitted scikit-learn GradientBoostingClassifier.
    """
    x_values = np.asarray(features, dtype=float)
    y_values = np.asarray(target, dtype=float)
    if x_values.ndim != 2 or len(x_values) != len(y_values) or len(y_values) == 0:
        raise ValueError("features and target must contain matching nonempty rows")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("gradient-boosting training inputs must be finite")
    if not np.isin(y_values, [0, 1]).all():
        raise ValueError("gradient-boosting target values must be 0 or 1")
    if (
        n_estimators <= 0
        or max_depth <= 0
        or learning_rate <= 0
        or min_samples_leaf <= 0
    ):
        raise ValueError("gradient-boosting parameters must be positive")
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
    )
    model.fit(x_values, y_values.astype(int))
    return model


def predict_gradient_boosting(
    model: GradientBoostingClassifier, features: np.ndarray
) -> np.ndarray:
    """Return probabilities from a fitted gradient-boosting classifier.

    Inputs:
        model: Fitted classifier returned by train_gradient_boosting.
        features: Finite numeric feature rows to score.
    Returns:
        One home-win probability per input row.
    """
    x_values = np.asarray(features, dtype=float)
    if x_values.ndim != 2 or not hasattr(model, "predict_proba"):
        raise ValueError("a fitted boosting model and feature matrix are required")
    return model.predict_proba(x_values)[:, 1]


def classification_metrics(target: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Calculate accuracy, Brier score, and binary log loss.

    Inputs:
        target: Observed binary outcomes.
        probabilities: Predicted positive-class probabilities.
    Returns:
        A dictionary containing accuracy, Brier score, and log loss.
    """
    y_values = np.asarray(target, dtype=float)
    predictions = np.asarray(probabilities, dtype=float)
    if y_values.shape != predictions.shape or y_values.size == 0:
        raise ValueError("target and probabilities must have matching nonempty shapes")
    if not np.isin(y_values, [0, 1]).all() or not (
        (0 <= predictions) & (predictions <= 1)
    ).all():
        raise ValueError("targets must be binary and probabilities must be in [0, 1]")
    clipped = np.clip(predictions, 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(np.mean((predictions >= 0.5) == y_values)),
        "brier_score": float(np.mean((predictions - y_values) ** 2)),
        "log_loss": float(
            np.mean(-(y_values * np.log(clipped) + (1 - y_values) * np.log(1 - clipped)))
        ),
    }


def select_regularization(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
    l2_values: Sequence[float] = DEFAULT_L2_VALUES,
    epochs: int = 4000,
) -> tuple[float, list[dict[str, float]]]:
    """Choose the L2 value with the lowest validation-period Brier score.

    Inputs:
        training_data: Earliest rows used to fit each candidate model.
        validation_data: Following rows used only for candidate comparison.
        feature_names: Ordered numeric predictors.
        l2_values: Candidate regularization strengths.
        epochs: Maximum solver iterations used for every candidate.
    Returns:
        The selected L2 value and validation metrics for every candidate.
    """
    if not l2_values:
        raise ValueError("at least one L2 value is required")
    preprocessor = fit_preprocessor(training_data, feature_names)
    train_features = transform_features(training_data, preprocessor)
    validation_features = transform_features(validation_data, preprocessor)
    train_target = training_data["home_win"].to_numpy(dtype=int)
    validation_target = validation_data["home_win"].to_numpy(dtype=int)
    rows = []
    for l2_penalty in l2_values:
        model = train_logistic_regression(
            train_features,
            train_target,
            float(l2_penalty),
            epochs,
        )
        probabilities = predict_probabilities(model, validation_features)
        row = {"l2_penalty": float(l2_penalty)}
        row.update(classification_metrics(validation_target, probabilities))
        rows.append(row)
    best = min(rows, key=lambda row: (row["brier_score"], row["log_loss"]))
    return best["l2_penalty"], rows


def train_model_family(
    family: str,
    features: np.ndarray,
    target: np.ndarray,
    parameters: dict[str, Any],
    logistic_epochs: int = 4000,
) -> Any:
    """Fit one supported model family with supplied hyperparameters.

    Inputs:
        family: Logistic-regression, random-forest, or boosting family name.
        features: Finite, preprocessed training feature matrix.
        target: Binary outcome vector aligned to the feature rows.
        parameters: Hyperparameters for the requested family.
        logistic_epochs: Maximum solver iterations for logistic regression.
    Returns:
        A fitted scikit-learn classifier for the requested family.
    """
    if family == "logistic_regression":
        return train_logistic_regression(
            features,
            target,
            l2_penalty=float(parameters["l2_penalty"]),
            epochs=logistic_epochs,
        )
    if family == "random_forest":
        return train_random_forest(
            features,
            target,
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            random_seed=int(parameters.get("random_seed", 163)),
        )
    if family == "gradient_boosting":
        return train_gradient_boosting(
            features,
            target,
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            learning_rate=float(parameters["learning_rate"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            random_seed=int(parameters.get("random_seed", 163)),
        )
    raise ValueError(f"Unsupported model family: {family}")


def predict_model_family(
    family: str, model: Any, features: np.ndarray
) -> np.ndarray:
    """Score feature rows using one supported fitted model family.

    Inputs:
        family: Logistic-regression, random-forest, or boosting family name.
        model: Fitted model returned by train_model_family.
        features: Finite, preprocessed feature matrix to score.
    Returns:
        One home-win probability per feature row.
    """
    if family == "logistic_regression":
        return predict_probabilities(model, features)
    if family == "random_forest":
        return predict_random_forest(model, features)
    if family == "gradient_boosting":
        return predict_gradient_boosting(model, features)
    raise ValueError(f"Unsupported model family: {family}")


def select_model_families(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
    l2_values: Sequence[float] = DEFAULT_L2_VALUES,
    random_forest_grid: Sequence[dict[str, Any]] = DEFAULT_RANDOM_FOREST_GRID,
    gradient_boosting_grid: Sequence[
        dict[str, Any]
    ] = DEFAULT_GRADIENT_BOOSTING_GRID,
    logistic_epochs: int = 4000,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Tune three families and choose the lowest validation Brier score.

    Inputs:
        training_data: Earliest rows used to fit every candidate.
        validation_data: Following rows used only for candidate comparison.
        feature_names: Ordered numeric predictors shared by all families.
        l2_values: Logistic-regression regularization candidates.
        random_forest_grid: Forest parameter dictionaries to compare.
        gradient_boosting_grid: Boosting parameter dictionaries to compare.
        logistic_epochs: Maximum solver iterations per logistic candidate.
    Returns:
        Selected family, best parameters by family, and all tuning metrics.
    """
    if not l2_values or not random_forest_grid or not gradient_boosting_grid:
        raise ValueError("every model family needs at least one candidate")
    preprocessor = fit_preprocessor(training_data, feature_names)
    train_features = transform_features(training_data, preprocessor)
    validation_features = transform_features(validation_data, preprocessor)
    train_target = training_data["home_win"].to_numpy(dtype=int)
    validation_target = validation_data["home_win"].to_numpy(dtype=int)
    grids: dict[str, Sequence[dict[str, Any]]] = {
        "logistic_regression": [
            {"l2_penalty": float(value)}
            for value in l2_values
        ],
        "random_forest": random_forest_grid,
        "gradient_boosting": gradient_boosting_grid,
    }
    tuning: dict[str, list[dict[str, Any]]] = {}
    best_parameters: dict[str, dict[str, Any]] = {}
    family_scores = []
    for family in MODEL_FAMILIES:
        rows = []
        for candidate in grids[family]:
            parameters = dict(candidate)
            model = train_model_family(
                family,
                train_features,
                train_target,
                parameters,
                logistic_epochs=logistic_epochs,
            )
            probabilities = predict_model_family(
                family, model, validation_features
            )
            metrics = classification_metrics(validation_target, probabilities)
            rows.append({"parameters": parameters, **metrics})
        best = min(
            rows,
            key=lambda row: (row["brier_score"], row["log_loss"]),
        )
        tuning[family] = rows
        best_parameters[family] = dict(best["parameters"])
        family_scores.append(
            {
                "family": family,
                "brier_score": best["brier_score"],
                "log_loss": best["log_loss"],
            }
        )
    selected = min(
        family_scores,
        key=lambda row: (row["brier_score"], row["log_loss"]),
    )
    return str(selected["family"]), best_parameters, tuning


def summarize_fitted_model(
    family: str,
    model: Any,
    parameters: dict[str, Any],
    feature_names: Sequence[str],
    training_rows: int,
) -> dict[str, Any]:
    """Create a JSON-safe description of a fitted final model.

    Inputs:
        family: Name of the fitted model family.
        model: Fitted scikit-learn classifier.
        parameters: Selected validation-stage hyperparameters.
        feature_names: Ordered predictor names used by the fitted model.
        training_rows: Number of development-period rows used for fitting.
    Returns:
        A compact dictionary describing parameters and fitted model size.
    """
    summary: dict[str, Any] = {
        "family": family,
        "training_rows": int(training_rows),
        "hyperparameters": dict(parameters),
    }
    if family == "logistic_regression":
        summary["intercept"] = float(model.intercept_[0])
        summary["coefficients"] = {
            name: float(value)
            for name, value in zip(feature_names, model.coef_[0])
        }
    else:
        summary["tree_count"] = int(len(model.estimators_))
        if family == "random_forest":
            summary["features_sampled_per_split"] = "sqrt"
    return summary


def estimate_taker_fee(price: float, contracts: int = 1, fee_rate: float = 0.07) -> float:
    """Estimate Kalshi taker fees, rounded up to the next cent.

    Inputs:
        price: Contract purchase price in dollars from zero to one.
        contracts: Positive number of contracts purchased.
        fee_rate: Nonnegative multiplier in the general fee formula.
    Returns:
        The estimated fee in dollars after cent rounding.
    """
    if not 0 <= price <= 1 or contracts <= 0 or fee_rate < 0:
        raise ValueError("price, contracts, and fee_rate are outside valid bounds")
    raw_fee = fee_rate * contracts * price * (1 - price)
    return math.ceil(raw_fee * 100 - 1e-12) / 100


def simulate_trades(
    test_data: pd.DataFrame,
    model_probabilities: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    fee_rate: float = 0.07,
) -> pd.DataFrame:
    """Simulate one taker contract when a model edge reaches a threshold.

    Inputs:
        test_data: Held-out games with contract quotes and realized outcomes.
        model_probabilities: Home-win probabilities aligned to test_data.
        thresholds: Minimum model-versus-purchase-price advantages to trade.
        fee_rate: Multiplier used by the taker-fee calculation.
    Returns:
        Position-level prices, fees, payouts, profit, and cumulative profit.
    """
    probabilities = np.asarray(model_probabilities, dtype=float)
    if len(probabilities) != len(test_data):
        raise ValueError("model probabilities must align with test rows")
    if not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("model probabilities must be in [0, 1]")
    records = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("trade thresholds must be in [0, 1]")
        for (_, game), model_probability in zip(test_data.iterrows(), probabilities):
            home_edge = model_probability - float(game["yes_ask"])
            away_edge = float(game["yes_bid"]) - model_probability
            if max(home_edge, away_edge) < threshold:
                continue
            if home_edge >= away_edge:
                side = "home_yes"
                purchase_price = float(game["yes_ask"])
                contract_probability = float(model_probability)
                payout = int(game["home_win"])
                edge = home_edge
            else:
                side = "home_no"
                purchase_price = 1 - float(game["yes_bid"])
                contract_probability = 1 - float(model_probability)
                payout = 1 - int(game["home_win"])
                edge = away_edge
            fee = estimate_taker_fee(purchase_price, fee_rate=fee_rate)
            records.append(
                {
                    "threshold": float(threshold),
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "side": side,
                    "model_probability": contract_probability,
                    "purchase_price": purchase_price,
                    "model_edge": float(edge),
                    "fee": fee,
                    "payout": payout,
                    "profit": float(payout - purchase_price - fee),
                }
            )
    columns = [
        "threshold",
        "game_id",
        "game_date",
        "side",
        "model_probability",
        "purchase_price",
        "model_edge",
        "fee",
        "payout",
        "profit",
        "cumulative_profit",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    trades = pd.DataFrame(records).sort_values(
        ["threshold", "game_date", "game_id"]
    )
    trades["cumulative_profit"] = trades.groupby("threshold")["profit"].cumsum()
    return trades[columns].reset_index(drop=True)


def summarize_trades(
    trades: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> list[dict[str, float | int | None]]:
    """Summarize count, profit, win rate, and ROI for each threshold.

    Inputs:
        trades: Position-level output from simulate_trades.
        thresholds: Thresholds to include, including those with no positions.
    Returns:
        One summary dictionary per requested threshold.
    """
    summaries = []
    for threshold in thresholds:
        selected = trades[np.isclose(trades["threshold"], threshold)].copy()
        positions = len(selected)
        total_cost = float((selected["purchase_price"] + selected["fee"]).sum())
        total_profit = float(selected["profit"].sum())
        summaries.append(
            {
                "threshold": float(threshold),
                "positions": int(positions),
                "wins": int(selected["payout"].sum()),
                "win_rate": None if positions == 0 else float(selected["payout"].mean()),
                "total_cost": total_cost,
                "total_profit": total_profit,
                "roi": None if total_cost == 0 else total_profit / total_cost,
            }
        )
    return summaries


def run_analysis(
    raw_data: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    l2_values: Sequence[float] = DEFAULT_L2_VALUES,
    random_forest_grid: Sequence[dict[str, Any]] = DEFAULT_RANDOM_FOREST_GRID,
    gradient_boosting_grid: Sequence[
        dict[str, Any]
    ] = DEFAULT_GRADIENT_BOOSTING_GRID,
    logistic_epochs: int = 4000,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compare three families, evaluate their final fits, and simulate trades.

    Inputs:
        raw_data: Processed EDA rows to clean for modeling.
        feature_names: Ordered predictors shared by all model families.
        thresholds: Model-edge cutoffs used by the trade simulation.
        l2_values: Logistic-regression regularization candidates.
        random_forest_grid: Forest parameter dictionaries to compare.
        gradient_boosting_grid: Boosting parameter dictionaries to compare.
        logistic_epochs: Maximum solver iterations for logistic regression.
    Returns:
        Results metadata, model-ready game rows, and simulated positions.
    """
    cleaned, cleaning_audit = clean_model_data(raw_data)
    training, validation, test = chronological_split(cleaned)
    selected_family, best_parameters, tuning = select_model_families(
        training,
        validation,
        feature_names=feature_names,
        l2_values=l2_values,
        random_forest_grid=random_forest_grid,
        gradient_boosting_grid=gradient_boosting_grid,
        logistic_epochs=logistic_epochs,
    )
    development = pd.concat([training, validation], ignore_index=True)
    preprocessor = fit_preprocessor(development, feature_names)
    development_features = transform_features(development, preprocessor)
    test_features = transform_features(test, preprocessor)
    development_target = development["home_win"].to_numpy(dtype=int)
    family_models = {}
    family_probabilities = {}
    final_models = {}
    family_metrics = {}
    for family in MODEL_FAMILIES:
        model = train_model_family(
            family,
            development_features,
            development_target,
            best_parameters[family],
            logistic_epochs=logistic_epochs,
        )
        probabilities = predict_model_family(family, model, test_features)
        family_models[family] = model
        family_probabilities[family] = probabilities
        family_metrics[family] = classification_metrics(
            test["home_win"], probabilities
        )
        final_models[family] = summarize_fitted_model(
            family,
            model,
            best_parameters[family],
            feature_names,
            len(development),
        )
    selected_probabilities = family_probabilities[selected_family]
    selected_metrics = family_metrics[selected_family]
    market_metrics = classification_metrics(test["home_win"], test["kalshi_prob"])
    trades = simulate_trades(
        test, selected_probabilities, thresholds=thresholds
    )

    model_ready = cleaned.copy()
    model_ready["split"] = ""
    model_ready.loc[model_ready["game_id"].isin(training["game_id"]), "split"] = "train"
    model_ready.loc[model_ready["game_id"].isin(validation["game_id"]), "split"] = "validation"
    model_ready.loc[model_ready["game_id"].isin(test["game_id"]), "split"] = "test"
    test_mask = model_ready["split"].eq("test")
    for family in MODEL_FAMILIES:
        column = f"{family}_probability"
        model_ready[column] = np.nan
        probability_by_game = pd.Series(
            family_probabilities[family], index=test["game_id"]
        )
        model_ready.loc[test_mask, column] = model_ready.loc[
            test_mask, "game_id"
        ].map(probability_by_game)
    model_ready["selected_model_probability"] = model_ready[
        f"{selected_family}_probability"
    ]
    model_ready["model_probability"] = model_ready[
        "selected_model_probability"
    ]

    split_summary = {}
    for name, frame in (("train", training), ("validation", validation), ("test", test)):
        split_summary[name] = {
            "rows": int(len(frame)),
            "start_date": frame["game_date"].min().strftime("%Y-%m-%d"),
            "end_date": frame["game_date"].max().strftime("%Y-%m-%d"),
        }
    results = {
        "cleaning": cleaning_audit,
        "feature_names": list(feature_names),
        "split": split_summary,
        "model_families": list(MODEL_FAMILIES),
        "selection_metric": "validation Brier score",
        "selected_family": selected_family,
        "selected_hyperparameters": best_parameters[selected_family],
        "best_hyperparameters_by_family": best_parameters,
        "validation_tuning": tuning,
        "preprocessor": preprocessor,
        "final_models": final_models,
        "final_model": final_models[selected_family],
        "test_metrics": {
            **family_metrics,
            "kalshi": market_metrics,
            "selected_model": selected_metrics,
            "selected_model_minus_kalshi": {
                name: selected_metrics[name] - market_metrics[name]
                for name in selected_metrics
            },
            "model": selected_metrics,
            "model_minus_kalshi": {
                name: selected_metrics[name] - market_metrics[name]
                for name in selected_metrics
            },
        },
        "simulated_trading": {
            "model_family": selected_family,
            "fee_rate": 0.07,
            "fee_assumption": (
                "One immediately matched contract; fee is rounded up to the next cent."
            ),
            "threshold_results": summarize_trades(trades, thresholds),
        },
    }
    return results, model_ready, trades


def main() -> None:
    """Train the model from the processed EDA table and write outputs.

    Inputs:
        None. Reads data/processed/kalshi_nba_eda.csv.
    Returns:
        None. Writes model-ready games, results JSON, and simulated trades.
    """
    raw_data = pd.read_csv(INPUT_PATH)
    results, model_ready, trades = run_analysis(raw_data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(OUTPUT_DIR / "model_ready_games.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "simulated_trades.csv", index=False)
    with (OUTPUT_DIR / "model_results.json").open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, allow_nan=False)
    print(json.dumps(results["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
