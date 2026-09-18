"""Calibrate raw source RMS against fragment-level manual gain edits."""

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kestrel.media_analysis import source_basename
from kestrel.project.parser import ProjectParseError

NEAR_ZERO_GAIN_DB = 0.5
OUTLIER_LIMIT = 20


def finite_number(value: Any) -> bool:
    """Return whether value is a finite JSON number, excluding booleans."""
    return type(value) in (int, float) and math.isfinite(value)


def percentile(ordered: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile from sorted finite values."""
    if not ordered:
        return None
    location = (len(ordered) - 1) * fraction
    index = int(location)
    return ordered[index] + (
        ordered[min(index + 1, len(ordered) - 1)] - ordered[index]
    ) * (location - index)


def numeric_distribution(values: list[float]) -> dict[str, Any]:
    """Summarize a numeric population with deterministic percentiles."""
    if not values:
        return {
            key: 0 if key == "count" else None
            for key in (
                "count",
                "min",
                "p10",
                "p25",
                "median",
                "p75",
                "p90",
                "p95",
                "max",
                "mean",
                "stddev",
            )
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "stddev": statistics.pstdev(ordered),
    }


def linear_regression(raw: list[float], gains: list[float]) -> dict[str, Any]:
    """Fit manual gain = intercept + slope * raw RMS using ordinary least squares."""
    if len(raw) != len(gains) or not raw:
        raise ValueError("Regression requires equal non-empty observations")
    mean_raw = statistics.fmean(raw)
    mean_gain = statistics.fmean(gains)
    centered_raw = [value - mean_raw for value in raw]
    centered_gain = [value - mean_gain for value in gains]
    sum_xx = sum(value * value for value in centered_raw)
    sum_yy = sum(value * value for value in centered_gain)
    sum_xy = sum(x * y for x, y in zip(centered_raw, centered_gain, strict=True))
    if sum_xx == 0:
        return {
            "model": "manual_gain_db = intercept + slope * raw_rms_dbfs",
            "count": len(raw),
            "slope": None,
            "intercept": None,
            "r_squared": None,
            "correlation": None,
            "residual_stddev": None,
        }
    slope = sum_xy / sum_xx
    intercept = mean_gain - slope * mean_raw
    residuals = [
        gain - (intercept + slope * level)
        for level, gain in zip(raw, gains, strict=True)
    ]
    residual_sum_squares = sum(value * value for value in residuals)
    return {
        "model": "manual_gain_db = intercept + slope * raw_rms_dbfs",
        "count": len(raw),
        "slope": slope,
        "intercept": intercept,
        "r_squared": (1 - residual_sum_squares / sum_yy if sum_yy > 0 else None),
        "correlation": (sum_xy / math.sqrt(sum_xx * sum_yy) if sum_yy > 0 else None),
        "residual_stddev": statistics.pstdev(residuals),
    }


def target_simulation(
    raw: list[float], gains: list[float], target: float
) -> dict[str, Any]:
    """Compare constant-target predicted gains with fragment-level manual gains."""
    if len(raw) != len(gains) or not raw:
        raise ValueError("Target simulation requires equal non-empty observations")
    errors = [target - level - gain for level, gain in zip(raw, gains, strict=True)]
    absolute = sorted(abs(value) for value in errors)
    return {
        "target_effective_rms_dbfs": target,
        "count": len(errors),
        "mae_db": statistics.fmean(absolute),
        "median_absolute_error_db": percentile(absolute, 0.5),
        "rmse_db": math.sqrt(statistics.fmean(value * value for value in errors)),
        "p90_absolute_error_db": percentile(absolute, 0.9),
        **{
            f"within_{limit}_db_percent": 100
            * sum(value <= limit for value in absolute)
            / len(absolute)
            for limit in (1, 2, 3, 6)
        },
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    def unique_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate {description} JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_fields
    )
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _indexed_records(
    value: Any, key: str, description: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(value):
        if not isinstance(record, dict) or not isinstance(record.get(key), str):
            raise ValueError(f"Invalid {description} record at index {index}")
        identity = record[key]
        if not identity or identity in result:
            raise ValueError(f"Duplicate or empty {description} identity: {identity}")
        result[identity] = record
    return result


def _outlier_record(observation: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "filename",
        "catalog_id",
        "fragment_id",
        "raw_rms_dbfs",
        "manual_gain_db",
        "effective_rms_dbfs",
        "median_target_residual_db",
    )
    return {field: observation[field] for field in fields}


def _exclusion_record(
    fragment: dict[str, Any],
    filename: str,
    reason: str,
    raw_rms_dbfs: Any = None,
) -> dict[str, Any]:
    return {
        "reason": reason,
        "catalog_id": fragment.get("catalog_id"),
        "filename": filename,
        "fragment_id": fragment.get("fragment_id"),
        "sequence_index": fragment.get("sequence_index"),
        "timeline_begin": fragment.get("timeline_begin"),
        "timeline_end": fragment.get("timeline_end"),
        "audio_pair_status": fragment.get("audio_pair_status"),
        "manual_gain_db": fragment.get("audio_gain_db"),
        "raw_rms_dbfs": raw_rms_dbfs,
    }


def audio_calibrate(
    dataset_path: str | Path,
    analysis_path: str | Path,
    output_path: str | Path,
    *,
    exclude_filenames: list[str] | None = None,
) -> dict[str, Any]:
    """Join fragment gains to source RMS and write empirical calibration data."""
    started = time.perf_counter()
    dataset_path = Path(dataset_path)
    analysis_path = Path(analysis_path)
    output_path = Path(output_path)
    resolved_output = output_path.resolve()
    if resolved_output in {dataset_path.resolve(), analysis_path.resolve()}:
        raise ProjectParseError(
            "Calibration output must be distinct from dataset and analysis inputs"
        )
    if output_path.exists():
        raise ProjectParseError(
            "Calibration output already exists; refusing to overwrite"
        )
    try:
        dataset = _load_json(dataset_path, "dataset")
        analysis = _load_json(analysis_path, "analysis")
        if (
            type(dataset.get("version")) is not int
            or dataset["version"] != 1
            or type(analysis.get("version")) is not int
            or analysis["version"] != 1
        ):
            raise ValueError("Calibration requires Dataset v1 and Media Analysis v1")
        dataset_sources = _indexed_records(
            dataset.get("sources"), "catalog_id", "sources"
        )
        analyzed_sources = _indexed_records(
            analysis.get("sources"), "catalog_id", "analysis sources"
        )
        fragments = dataset.get("timeline_fragments")
        fragment_index = _indexed_records(fragments, "fragment_id", "fragments")
        requested_exclusions = {
            name.casefold(): name for name in exclude_filenames or []
        }
        available_names = {
            source_basename(source["filename"]).casefold()
            for source in dataset_sources.values()
            if isinstance(source.get("filename"), str)
        }
        missing_exclusions = [
            name
            for key, name in requested_exclusions.items()
            if key not in available_names
        ]
        if missing_exclusions:
            raise ValueError(
                "Excluded filenames were not found: " + ", ".join(missing_exclusions)
            )
        excluded_names = set(requested_exclusions)
        excluded: Counter[str] = Counter()
        excluded_fragments: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        catalog_joined = 0

        for fragment in fragment_index.values():
            catalog_id = fragment.get("catalog_id")
            if not isinstance(catalog_id, str) or catalog_id not in dataset_sources:
                raise ValueError(f"Fragment has unknown catalog ID: {catalog_id}")
            source = dataset_sources[catalog_id]
            filename = source.get("filename")
            if not isinstance(filename, str):
                raise ValueError(f"Source {catalog_id} has invalid filename")
            analyzed = analyzed_sources.get(catalog_id)
            if analyzed is not None:
                catalog_joined += 1
            if source_basename(filename).casefold() in excluded_names:
                reason = "explicit_filename_exclusion"
                excluded[reason] += 1
                excluded_fragments.append(_exclusion_record(fragment, filename, reason))
                continue
            pair_status = fragment.get("audio_pair_status")
            if pair_status != "matched":
                reason = f"audio_pair_{pair_status or 'unknown'}"
                excluded[reason] += 1
                excluded_fragments.append(_exclusion_record(fragment, filename, reason))
                continue
            gain = fragment.get("audio_gain_db")
            if gain is None:
                reason = "manual_gain_unavailable"
                excluded[reason] += 1
                excluded_fragments.append(_exclusion_record(fragment, filename, reason))
                continue
            if not finite_number(gain):
                raise ValueError(f"Fragment {fragment['fragment_id']} has invalid gain")
            if analyzed is None:
                reason = "analysis_source_unavailable"
                excluded[reason] += 1
                excluded_fragments.append(_exclusion_record(fragment, filename, reason))
                continue
            signals = analyzed.get("signals")
            audio = signals.get("audio") if isinstance(signals, dict) else None
            raw = audio.get("rms_dbfs") if isinstance(audio, dict) else None
            if raw is None:
                reason = "raw_rms_unavailable"
                excluded[reason] += 1
                excluded_fragments.append(_exclusion_record(fragment, filename, reason))
                continue
            if not finite_number(raw):
                raise ValueError(f"Source {catalog_id} has invalid raw RMS")
            observation = {
                "catalog_id": catalog_id,
                "filename": filename,
                "fragment_id": fragment["fragment_id"],
                "sequence_index": fragment.get("sequence_index"),
                "clip_id": fragment.get("clip_id"),
                "track_id": fragment.get("track_id"),
                "track_index": fragment.get("track_index"),
                "source_in": fragment.get("source_in"),
                "source_out": fragment.get("source_out"),
                "source_duration": fragment.get("source_duration"),
                "timeline_begin": fragment.get("timeline_begin"),
                "timeline_end": fragment.get("timeline_end"),
                "timeline_duration": fragment.get("timeline_duration"),
                "audio_pair_status": pair_status,
                "source_fragment_count": source.get("fragment_count"),
                "source_retained_ratio": source.get("retained_ratio"),
                "source_overlapping_retained_ranges": source.get(
                    "overlapping_retained_ranges"
                ),
                "raw_rms_dbfs": raw,
                "manual_gain_db": gain,
                "effective_rms_dbfs": raw + gain,
            }
            observations.append(observation)

        if not observations:
            raise ValueError("No eligible calibration observations")
        raw_values = [item["raw_rms_dbfs"] for item in observations]
        gains = [item["manual_gain_db"] for item in observations]
        effective = [item["effective_rms_dbfs"] for item in observations]
        raw_distribution = numeric_distribution(raw_values)
        gain_distribution = numeric_distribution(gains)
        effective_distribution = numeric_distribution(effective)
        absolute_gain_distribution = numeric_distribution(
            [abs(value) for value in gains]
        )
        target_values = {
            "median_effective_rms": effective_distribution["median"],
            "mean_effective_rms": effective_distribution["mean"],
            "p25_effective_rms": effective_distribution["p25"],
            "p75_effective_rms": effective_distribution["p75"],
        }
        simulations = {
            name: target_simulation(raw_values, gains, target)
            for name, target in target_values.items()
            if finite_number(target)
        }
        median_target = effective_distribution["median"]
        assert isinstance(median_target, float)
        for observation in observations:
            observation["median_target_residual_db"] = (
                median_target - observation["effective_rms_dbfs"]
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            grouped[observation["catalog_id"]].append(observation)
        source_aggregates = []
        inconsistencies = []
        for catalog_id, items in sorted(grouped.items()):
            source_gains = [item["manual_gain_db"] for item in items]
            source_effective = [item["effective_rms_dbfs"] for item in items]
            aggregate = {
                "catalog_id": catalog_id,
                "filename": items[0]["filename"],
                "observation_count": len(items),
                "fragment_ids": [item["fragment_id"] for item in items],
                "raw_rms_dbfs": items[0]["raw_rms_dbfs"],
                "manual_gain_distribution": numeric_distribution(source_gains),
                "effective_rms_distribution": numeric_distribution(source_effective),
                "manual_gain_range_db": max(source_gains) - min(source_gains),
            }
            source_aggregates.append(aggregate)
            if len(items) > 1 and aggregate["manual_gain_range_db"] > 1e-9:
                inconsistencies.append(
                    {
                        "catalog_id": catalog_id,
                        "filename": items[0]["filename"],
                        "manual_gain_range_db": aggregate["manual_gain_range_db"],
                        "fragments": [
                            {
                                "fragment_id": item["fragment_id"],
                                "timeline_begin": item["timeline_begin"],
                                "timeline_end": item["timeline_end"],
                                "manual_gain_db": item["manual_gain_db"],
                            }
                            for item in items
                        ],
                    }
                )

        strongest_positive = sorted(
            (item for item in observations if item["manual_gain_db"] > 0),
            key=lambda item: (-item["manual_gain_db"], item["fragment_id"]),
        )[:OUTLIER_LIMIT]
        strongest_negative = sorted(
            (item for item in observations if item["manual_gain_db"] < 0),
            key=lambda item: (item["manual_gain_db"], item["fragment_id"]),
        )[:OUTLIER_LIMIT]
        largest_residual = sorted(
            observations,
            key=lambda item: (
                -abs(item["median_target_residual_db"]),
                item["fragment_id"],
            ),
        )[:OUTLIER_LIMIT]
        median_simulation = simulations["median_effective_rms"]
        result = {
            "version": 1,
            "input": {
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    "bytes": dataset_path.stat().st_size,
                    "version": dataset["version"],
                },
                "media_analysis": {
                    "path": str(analysis_path),
                    "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                    "bytes": analysis_path.stat().st_size,
                    "version": analysis["version"],
                },
                "excluded_filenames": sorted(exclude_filenames or []),
            },
            "counts": {
                "dataset_fragment_count": len(fragment_index),
                "catalog_joined_fragment_count": catalog_joined,
                "eligible_fragment_count": len(observations),
                "excluded_fragment_count": len(fragment_index) - len(observations),
                "excluded_counts": dict(excluded),
                "eligible_source_count": len(grouped),
            },
            "semantics": {
                "join_key": "catalog_id",
                "required_audio_pair_status": "matched",
                "near_zero_gain_db": f"abs(manual_gain_db) < {NEAR_ZERO_GAIN_DB}",
                "stddev": "population standard deviation",
                "median_target_residual_db": (
                    "predicted median-target gain minus manual gain"
                ),
            },
            "fragment_observations": observations,
            "excluded_fragments": excluded_fragments,
            "source_aggregates": source_aggregates,
            "distributions": {
                "raw_rms_dbfs": raw_distribution,
                "manual_gain_db": gain_distribution,
                "effective_rms_dbfs": effective_distribution,
                "absolute_gain_magnitude_db": absolute_gain_distribution,
            },
            "gain_counts": {
                "positive": sum(value >= NEAR_ZERO_GAIN_DB for value in gains),
                "negative": sum(value <= -NEAR_ZERO_GAIN_DB for value in gains),
                "zero_or_near_zero": sum(
                    abs(value) < NEAR_ZERO_GAIN_DB for value in gains
                ),
                "gain_at_least_positive_3_db": sum(value >= 3 for value in gains),
                "gain_at_least_positive_6_db": sum(value >= 6 for value in gains),
                "gain_at_least_positive_12_db": sum(value >= 12 for value in gains),
                "gain_at_most_negative_3_db": sum(value <= -3 for value in gains),
                "gain_at_most_negative_6_db": sum(value <= -6 for value in gains),
                "gain_at_most_negative_12_db": sum(value <= -12 for value in gains),
            },
            "regression": linear_regression(raw_values, gains),
            "implied_target_effective_rms_dbfs": median_target,
            "target_simulations": simulations,
            "constant_target_diagnostics": {
                "effective_rms_iqr_db": (
                    effective_distribution["p75"] - effective_distribution["p25"]
                ),
                "effective_rms_p90_p10_range_db": (
                    effective_distribution["p90"] - effective_distribution["p10"]
                ),
                "median_target_mae_db": median_simulation["mae_db"],
                "median_target_within_3_db_percent": median_simulation[
                    "within_3_db_percent"
                ],
                "interpretation": (
                    "Descriptive diagnostics only; no production target is selected"
                ),
            },
            "outliers": {
                "limit_per_group": OUTLIER_LIMIT,
                "strongest_positive_manual_gain": [
                    _outlier_record(item) for item in strongest_positive
                ],
                "strongest_negative_manual_gain": [
                    _outlier_record(item) for item in strongest_negative
                ],
                "largest_absolute_median_target_residual": [
                    _outlier_record(item) for item in largest_residual
                ],
            },
            "repeated_source_gain_inconsistencies": inconsistencies,
            "limitations": [
                (
                    "Raw RMS is measured over the full source while gain is attached "
                    "to a retained fragment placement"
                ),
                (
                    "Manual gains may encode creative intent and are not treated as "
                    "an automatic normalization policy"
                ),
                (
                    "Explicitly excluded disruptive-event sources do not contribute "
                    "to distributions, regression, or target simulations"
                ),
            ],
        }
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return {
            "output": str(output_path),
            "dataset_fragment_count": len(fragment_index),
            "catalog_joined_fragment_count": catalog_joined,
            "eligible_fragment_count": len(observations),
            "excluded_fragment_count": len(fragment_index) - len(observations),
            "excluded_counts": dict(excluded),
            "eligible_source_count": len(grouped),
            "effective_rms_dbfs": effective_distribution,
            "regression": result["regression"],
            "repeated_source_gain_inconsistency_count": len(inconsistencies),
            "runtime_seconds": time.perf_counter() - started,
            "output_bytes": output_path.stat().st_size,
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        raise ProjectParseError(str(error)) from error
