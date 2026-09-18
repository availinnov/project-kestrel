"""Deterministic cross-validated models for predicting manual audio gain."""

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kestrel.audio_calibration import finite_number, numeric_distribution, percentile
from kestrel.project.parser import ProjectParseError

FOLD_COUNT = 10
FOLD_SEED = "project-kestrel-audio-gain-v1"
OUTLIER_LIMIT = 20
RAW_BANDS = (
    ("below_-55", None, -55.0),
    ("-55_to_-45", -55.0, -45.0),
    ("-45_to_-35", -45.0, -35.0),
    ("-35_to_-25", -35.0, -25.0),
    ("at_least_-25", -25.0, None),
)


@dataclass(frozen=True)
class ModelSpec:
    """One deterministic model family and transformation configuration."""

    name: str
    family: str
    deadband_db: float | None = None
    minimum_gain_db: float | None = None
    maximum_gain_db: float | None = None
    breakpoint_quantiles: tuple[float, ...] = field(default_factory=tuple)


def model_specs() -> list[ModelSpec]:
    """Return the complete, intentionally small candidate set."""
    specs = [
        ModelSpec("zero_gain", "zero"),
        ModelSpec("constant_target", "constant_target"),
        ModelSpec("ols_linear", "linear"),
        ModelSpec("robust_linear", "robust_linear"),
    ]
    specs.extend(
        ModelSpec(f"linear_deadband_{str(value).replace('.', '_')}", "deadband", value)
        for value in (0.5, 1.0, 1.5, 2.0, 3.0)
    )
    specs.extend(
        ModelSpec(f"linear_clamp_{limit:g}", "clamp", None, -limit, limit)
        for limit in (6.0, 9.0, 12.0, 15.0, 18.0)
    )
    specs.append(ModelSpec("linear_clamp_filmora", "clamp", None, -18.35, 18.0))
    specs.extend(
        [
            ModelSpec("linear_deadband_1_clamp_12", "deadband_clamp", 1.0, -12, 12),
            ModelSpec("linear_deadband_2_clamp_12", "deadband_clamp", 2.0, -12, 12),
            ModelSpec("linear_deadband_1_clamp_18", "deadband_clamp", 1.0, -18, 18),
            ModelSpec("linear_deadband_2_clamp_18", "deadband_clamp", 2.0, -18, 18),
            ModelSpec(
                "linear_deadband_1_clamp_filmora",
                "deadband_clamp",
                1.0,
                -18.35,
                18.0,
            ),
        ]
    )
    specs.extend(
        [
            ModelSpec("piecewise_2_p33", "piecewise", breakpoint_quantiles=(1 / 3,)),
            ModelSpec("piecewise_2_p50", "piecewise", breakpoint_quantiles=(0.5,)),
            ModelSpec("piecewise_2_p67", "piecewise", breakpoint_quantiles=(2 / 3,)),
            ModelSpec(
                "piecewise_3_p25_p75",
                "piecewise",
                breakpoint_quantiles=(0.25, 0.75),
            ),
        ]
    )
    return specs


def grouped_folds(
    observations: list[dict[str, Any]], count: int = FOLD_COUNT
) -> list[list[int]]:
    """Create deterministic folds grouped by catalog ID."""
    if count < 2:
        raise ValueError("Cross-validation requires at least two folds")
    by_source: dict[str, list[int]] = {}
    for index, observation in enumerate(observations):
        catalog_id = observation["catalog_id"]
        by_source.setdefault(catalog_id, []).append(index)
    if len(by_source) < 2:
        raise ValueError("Cross-validation requires at least two sources")
    ordered_sources = sorted(
        by_source,
        key=lambda value: hashlib.sha256(f"{FOLD_SEED}\0{value}".encode()).hexdigest(),
    )
    folds: list[list[int]] = [[] for _ in range(min(count, len(ordered_sources)))]
    for source_index, catalog_id in enumerate(ordered_sources):
        folds[source_index % len(folds)].extend(by_source[catalog_id])
    return folds


def _fit_weighted_linear(
    raw: list[float], gains: list[float], weights: list[float]
) -> tuple[float, float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Linear fit has no positive observation weights")
    mean_raw = sum(w * x for w, x in zip(weights, raw, strict=True)) / total
    mean_gain = sum(w * y for w, y in zip(weights, gains, strict=True)) / total
    variance = sum(w * (x - mean_raw) ** 2 for w, x in zip(weights, raw, strict=True))
    if variance <= 0:
        raise ValueError("Linear fit requires varying raw RMS")
    covariance = sum(
        w * (x - mean_raw) * (y - mean_gain)
        for w, x, y in zip(weights, raw, gains, strict=True)
    )
    slope = covariance / variance
    return mean_gain - slope * mean_raw, slope


def fit_ols(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit ordinary least-squares gain against raw RMS."""
    raw = [item["raw_rms_dbfs"] for item in observations]
    gains = [item["manual_gain_db"] for item in observations]
    intercept, slope = _fit_weighted_linear(raw, gains, [1.0] * len(raw))
    return {"intercept": intercept, "slope": slope}


def _huber_weights(
    observations: list[dict[str, Any]], intercept: float, slope: float
) -> tuple[list[float], float]:
    residuals = [
        item["manual_gain_db"] - (intercept + slope * item["raw_rms_dbfs"])
        for item in observations
    ]
    deviations = sorted(abs(value) for value in residuals)
    mad = percentile(deviations, 0.5)
    assert mad is not None
    scale = mad / 0.6744897501960817
    if scale <= 1e-12:
        return [1.0] * len(residuals), scale
    cutoff = 1.345 * scale
    return [
        min(1.0, cutoff / abs(value)) if value else 1.0 for value in residuals
    ], scale


def fit_huber(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit a deterministic Huber IRLS line without external ML dependencies."""
    raw = [item["raw_rms_dbfs"] for item in observations]
    gains = [item["manual_gain_db"] for item in observations]
    parameters = fit_ols(observations)
    iterations = 0
    scale = 0.0
    for iteration in range(1, 51):
        iterations = iteration
        weights, scale = _huber_weights(
            observations, parameters["intercept"], parameters["slope"]
        )
        intercept, slope = _fit_weighted_linear(raw, gains, weights)
        change = max(
            abs(intercept - parameters["intercept"]),
            abs(slope - parameters["slope"]),
        )
        parameters = {"intercept": intercept, "slope": slope}
        if change < 1e-9:
            break
    weights, scale = _huber_weights(
        observations, parameters["intercept"], parameters["slope"]
    )
    reduced = []
    for observation, weight in zip(observations, weights, strict=True):
        if weight < 0.999999:
            predicted = (
                parameters["intercept"]
                + parameters["slope"] * observation["raw_rms_dbfs"]
            )
            reduced.append(
                {
                    "catalog_id": observation["catalog_id"],
                    "fragment_id": observation["fragment_id"],
                    "filename": observation["filename"],
                    "residual_db": predicted - observation["manual_gain_db"],
                    "weight": weight,
                }
            )
    return {
        **parameters,
        "method": "Huber IRLS",
        "huber_epsilon": 1.345,
        "iterations": iterations,
        "robust_scale_db": scale,
        "reduced_influence_count": len(reduced),
        "reduced_influence_observations": reduced,
    }


def _solve(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small dense linear system with partial pivoting."""
    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise ValueError("Piecewise fit is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def fit_piecewise(
    observations: list[dict[str, Any]], quantiles: tuple[float, ...]
) -> dict[str, Any]:
    """Fit a continuous hinge-basis piecewise line."""
    raw = sorted(item["raw_rms_dbfs"] for item in observations)
    breakpoints = [percentile(raw, value) for value in quantiles]
    if any(value is None for value in breakpoints):
        raise ValueError("Piecewise breakpoints are unavailable")
    knots = [float(value) for value in breakpoints if value is not None]
    rows = [
        [1.0, item["raw_rms_dbfs"]]
        + [max(0.0, item["raw_rms_dbfs"] - knot) for knot in knots]
        for item in observations
    ]
    gains = [item["manual_gain_db"] for item in observations]
    size = len(rows[0])
    normal = [
        [sum(row[i] * row[j] for row in rows) for j in range(size)] for i in range(size)
    ]
    right = [
        sum(row[i] * gain for row, gain in zip(rows, gains, strict=True))
        for i in range(size)
    ]
    coefficients = _solve(normal, right)
    slopes = []
    intercepts = []
    for segment in range(len(knots) + 1):
        slopes.append(coefficients[1] + sum(coefficients[2 : 2 + segment]))
        intercepts.append(
            coefficients[0]
            - sum(coefficients[2 + index] * knots[index] for index in range(segment))
        )
    return {
        "breakpoint_quantiles": list(quantiles),
        "breakpoints_dbfs": knots,
        "coefficients": coefficients,
        "segment_slopes": slopes,
        "segment_intercepts": intercepts,
        "continuity": "continuous hinge basis",
    }


def fit_model(spec: ModelSpec, observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit one model specification."""
    if spec.family == "zero":
        return {}
    if spec.family == "constant_target":
        effective = sorted(
            item["raw_rms_dbfs"] + item["manual_gain_db"] for item in observations
        )
        target = percentile(effective, 0.5)
        assert target is not None
        return {"target_effective_rms_dbfs": target}
    if spec.family == "robust_linear":
        return fit_huber(observations)
    if spec.family == "piecewise":
        return fit_piecewise(observations, spec.breakpoint_quantiles)
    return fit_ols(observations)


def predict_gain(spec: ModelSpec, parameters: dict[str, Any], raw: float) -> float:
    """Predict gain and apply the configured deterministic transformations."""
    if spec.family == "zero":
        prediction = 0.0
    elif spec.family == "constant_target":
        prediction = parameters["target_effective_rms_dbfs"] - raw
    elif spec.family == "piecewise":
        row = [1.0, raw] + [
            max(0.0, raw - knot) for knot in parameters["breakpoints_dbfs"]
        ]
        prediction = sum(
            coefficient * value
            for coefficient, value in zip(parameters["coefficients"], row, strict=True)
        )
    else:
        prediction = parameters["intercept"] + parameters["slope"] * raw
    if spec.deadband_db is not None and abs(prediction) < spec.deadband_db:
        prediction = 0.0
    if spec.minimum_gain_db is not None:
        prediction = max(spec.minimum_gain_db, prediction)
    if spec.maximum_gain_db is not None:
        prediction = min(spec.maximum_gain_db, prediction)
    return prediction


def prediction_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    """Calculate signed and absolute prediction-error metrics."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Metrics require equal non-empty observations")
    errors = [guess - truth for truth, guess in zip(actual, predicted, strict=True)]
    absolute = sorted(abs(value) for value in errors)
    return {
        "count": len(errors),
        "mae_db": statistics.fmean(absolute),
        "median_absolute_error_db": percentile(absolute, 0.5),
        "rmse_db": math.sqrt(statistics.fmean(value * value for value in errors)),
        "p90_absolute_error_db": percentile(absolute, 0.9),
        "residual_stddev_db": statistics.pstdev(errors),
        "bias_db": statistics.fmean(errors),
        **{
            f"within_{limit}_db_percent": 100
            * sum(value <= limit for value in absolute)
            / len(absolute)
            for limit in (1, 2, 3, 6)
        },
    }


def band_metrics(
    observations: list[dict[str, Any]], predicted: list[float]
) -> dict[str, Any]:
    """Report prediction behavior across fixed raw-RMS bands."""
    result: dict[str, dict[str, Any]] = {}
    for name, lower, upper in RAW_BANDS:
        indices = [
            index
            for index, item in enumerate(observations)
            if (lower is None or item["raw_rms_dbfs"] >= lower)
            and (upper is None or item["raw_rms_dbfs"] < upper)
        ]
        actual = [observations[index]["manual_gain_db"] for index in indices]
        guesses = [predicted[index] for index in indices]
        if not indices:
            result[name] = {
                "count": 0,
                "manual_gain_median_db": None,
                "predicted_gain_median_db": None,
                "mae_db": None,
                "bias_db": None,
            }
            continue
        errors = [guess - truth for truth, guess in zip(actual, guesses, strict=True)]
        result[name] = {
            "count": len(indices),
            "manual_gain_median_db": percentile(sorted(actual), 0.5),
            "predicted_gain_median_db": percentile(sorted(guesses), 0.5),
            "mae_db": statistics.fmean(abs(value) for value in errors),
            "bias_db": statistics.fmean(errors),
        }
    return result


def predicted_distribution(values: list[float]) -> dict[str, Any]:
    """Summarize predicted gain levels and practical threshold counts."""
    distribution = numeric_distribution(values)
    return {
        "count": distribution["count"],
        "min": distribution["min"],
        "p10": distribution["p10"],
        "median": distribution["median"],
        "p90": distribution["p90"],
        "max": distribution["max"],
        "gain_at_least_positive_3_db": sum(value >= 3 for value in values),
        "gain_at_least_positive_6_db": sum(value >= 6 for value in values),
        "gain_at_least_positive_12_db": sum(value >= 12 for value in values),
        "gain_at_most_negative_3_db": sum(value <= -3 for value in values),
        "gain_at_most_negative_6_db": sum(value <= -6 for value in values),
        "gain_at_most_negative_12_db": sum(value <= -12 for value in values),
    }


def _model_complexity(spec: ModelSpec) -> dict[str, Any]:
    parameter_count = {
        "zero": 0,
        "constant_target": 1,
        "linear": 2,
        "robust_linear": 2,
        "deadband": 3,
        "clamp": 4,
        "deadband_clamp": 5,
        "piecewise": 2 + 2 * len(spec.breakpoint_quantiles),
    }[spec.family]
    behavior = {
        "zero": "always zero",
        "constant_target": "slope -1; unbounded extremes",
        "linear": "two-parameter unbounded line",
        "robust_linear": "Huber-weighted unbounded line",
        "deadband": "linear with zero region",
        "clamp": "linear with bounded extremes",
        "deadband_clamp": "zero region and bounded extremes",
        "piecewise": "continuous fitted slope changes",
    }[spec.family]
    return {"parameter_count": parameter_count, "extreme_behavior": behavior}


def _validate_observations(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    if type(calibration.get("version")) is not int or calibration["version"] != 1:
        raise ValueError("Audio gain evaluation requires calibration version 1")
    values = calibration.get("fragment_observations")
    if not isinstance(values, list) or not values:
        raise ValueError("Calibration requires fragment observations")
    observations = []
    seen_fragments: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid calibration observation at index {index}")
        catalog_id = value.get("catalog_id")
        fragment_id = value.get("fragment_id")
        raw = value.get("raw_rms_dbfs")
        gain = value.get("manual_gain_db")
        if (
            not isinstance(catalog_id, str)
            or not catalog_id
            or not isinstance(fragment_id, str)
            or not fragment_id
            or fragment_id in seen_fragments
            or not isinstance(value.get("filename"), str)
            or not finite_number(raw)
            or not finite_number(gain)
        ):
            raise ValueError(f"Invalid calibration observation at index {index}")
        seen_fragments.add(fragment_id)
        observations.append(value)
    return observations


def _load_calibration(path: Path) -> dict[str, Any]:
    def unique_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate calibration JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_fields
    )
    if not isinstance(value, dict):
        raise ValueError("Calibration must be a JSON object")
    return value


def audio_gain_evaluate(
    calibration_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Cross-validate deterministic gain models over saved calibration JSON."""
    started = time.perf_counter()
    calibration_path = Path(calibration_path)
    output_path = Path(output_path)
    if output_path.resolve() == calibration_path.resolve():
        raise ProjectParseError("Model output must be distinct from calibration input")
    if output_path.exists():
        raise ProjectParseError("Model output already exists; refusing to overwrite")
    try:
        calibration = _load_calibration(calibration_path)
        observations = _validate_observations(calibration)
        folds = grouped_folds(observations)
        specs = model_specs()
        model_results: dict[str, Any] = {}
        cv_predictions: dict[str, list[float]] = {}
        actual = [item["manual_gain_db"] for item in observations]

        for spec in specs:
            predictions: list[float | None] = [None] * len(observations)
            for validation_indices in folds:
                validation_set = set(validation_indices)
                train = [
                    item
                    for index, item in enumerate(observations)
                    if index not in validation_set
                ]
                parameters = fit_model(spec, train)
                for index in validation_indices:
                    predictions[index] = predict_gain(
                        spec, parameters, observations[index]["raw_rms_dbfs"]
                    )
            if any(value is None for value in predictions):
                raise ValueError(f"Incomplete cross-validation for {spec.name}")
            concrete = [float(value) for value in predictions if value is not None]
            full_parameters = fit_model(spec, observations)
            cv_predictions[spec.name] = concrete
            model_results[spec.name] = {
                "family": spec.family,
                "configuration": {
                    "deadband_db": spec.deadband_db,
                    "minimum_gain_db": spec.minimum_gain_db,
                    "maximum_gain_db": spec.maximum_gain_db,
                    "breakpoint_quantiles": list(spec.breakpoint_quantiles),
                },
                "complexity": _model_complexity(spec),
                "full_dataset_fit": full_parameters,
                "cross_validation": {
                    "metrics": prediction_metrics(actual, concrete),
                    "raw_level_bands": band_metrics(observations, concrete),
                    "predicted_gain_distribution": predicted_distribution(concrete),
                },
            }

        families = {
            "deadband": [spec.name for spec in specs if spec.family == "deadband"],
            "clamp": [spec.name for spec in specs if spec.family == "clamp"],
            "deadband_clamp": [
                spec.name for spec in specs if spec.family == "deadband_clamp"
            ],
            "piecewise": [spec.name for spec in specs if spec.family == "piecewise"],
        }
        best_tested = {
            family: min(
                names,
                key=lambda name: (
                    model_results[name]["cross_validation"]["metrics"]["mae_db"],
                    name,
                ),
            )
            for family, names in families.items()
        }
        comparison_names = [
            "zero_gain",
            "constant_target",
            "ols_linear",
            "robust_linear",
            best_tested["deadband"],
            best_tested["clamp"],
            best_tested["deadband_clamp"],
            best_tested["piecewise"],
        ]
        comparison_table = []
        for name in comparison_names:
            model = model_results[name]
            comparison_table.append(
                {
                    "model": name,
                    "family": model["family"],
                    **model["complexity"],
                    **model["cross_validation"]["metrics"],
                }
            )

        for name, cv_values in cv_predictions.items():
            residuals = [
                prediction - item["manual_gain_db"]
                for item, prediction in zip(observations, cv_values, strict=True)
            ]
            indices = sorted(
                range(len(observations)),
                key=lambda index: (
                    -abs(residuals[index]),
                    observations[index]["fragment_id"],
                ),
            )[:OUTLIER_LIMIT]
            model_results[name]["cross_validation"]["largest_residuals"] = [
                {
                    "filename": observations[index]["filename"],
                    "catalog_id": observations[index]["catalog_id"],
                    "fragment_id": observations[index]["fragment_id"],
                    "raw_rms_dbfs": observations[index]["raw_rms_dbfs"],
                    "manual_gain_db": observations[index]["manual_gain_db"],
                    "predicted_gain_db": cv_values[index],
                    "error_db": residuals[index],
                }
                for index in indices
            ]

        reasonable = [
            "ols_linear",
            "robust_linear",
            best_tested["deadband"],
            best_tested["clamp"],
            best_tested["deadband_clamp"],
            best_tested["piecewise"],
        ]
        common_failures = []
        for index, observation in enumerate(observations):
            errors = {
                name: cv_predictions[name][index] - observation["manual_gain_db"]
                for name in reasonable
            }
            if all(abs(value) >= 6 for value in errors.values()):
                common_failures.append(
                    {
                        "filename": observation["filename"],
                        "catalog_id": observation["catalog_id"],
                        "fragment_id": observation["fragment_id"],
                        "raw_rms_dbfs": observation["raw_rms_dbfs"],
                        "manual_gain_db": observation["manual_gain_db"],
                        "model_errors_db": errors,
                        "minimum_absolute_error_db": min(
                            abs(v) for v in errors.values()
                        ),
                    }
                )
        common_failures.sort(
            key=lambda item: (-item["minimum_absolute_error_db"], item["fragment_id"])
        )
        fold_by_observation = {
            observation_index: fold_index
            for fold_index, indices in enumerate(folds)
            for observation_index in indices
        }
        evaluation_observations = [
            {
                "filename": observation["filename"],
                "catalog_id": observation["catalog_id"],
                "fragment_id": observation["fragment_id"],
                "raw_rms_dbfs": observation["raw_rms_dbfs"],
                "manual_gain_db": observation["manual_gain_db"],
                "validation_fold": fold_by_observation[index],
                "cross_validated_predictions": {
                    name: {
                        "predicted_gain_db": predictions[index],
                        "error_db": predictions[index] - observation["manual_gain_db"],
                    }
                    for name, predictions in cv_predictions.items()
                },
            }
            for index, observation in enumerate(observations)
        ]
        source_ids = {item["catalog_id"] for item in observations}
        result = {
            "version": 1,
            "input": {
                "calibration_path": str(calibration_path),
                "calibration_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
                "calibration_bytes": calibration_path.stat().st_size,
            },
            "observation_count": len(observations),
            "source_count": len(source_ids),
            "cross_validation": {
                "method": "deterministic source-grouped folds",
                "fold_count": len(folds),
                "ordering_seed": FOLD_SEED,
                "folds": [
                    {
                        "fold": index,
                        "observation_count": len(indices),
                        "source_ids": sorted(
                            {observations[item]["catalog_id"] for item in indices}
                        ),
                    }
                    for index, indices in enumerate(folds)
                ],
            },
            "evaluation_observations": evaluation_observations,
            "manual_gain_distribution": predicted_distribution(actual),
            "tested_model_count": len(specs),
            "models": model_results,
            "best_tested_by_family": best_tested,
            "comparison_table": comparison_table,
            "common_large_residuals": {
                "definition": (
                    "absolute CV error >= 6 dB for every listed reasonable model"
                ),
                "models": reasonable,
                "count": len(common_failures),
                "observations": common_failures[:OUTLIER_LIMIT],
            },
            "selection_note": (
                "Lowest CV MAE identifies family representatives only; no production "
                "model is selected"
            ),
        }
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return {
            "output": str(output_path),
            "observation_count": len(observations),
            "source_count": len(source_ids),
            "fold_count": len(folds),
            "tested_model_count": len(specs),
            "best_tested_by_family": best_tested,
            "comparison_table": comparison_table,
            "common_large_residual_count": len(common_failures),
            "runtime_seconds": time.perf_counter() - started,
            "output_bytes": output_path.stat().st_size,
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        raise ProjectParseError(str(error)) from error
