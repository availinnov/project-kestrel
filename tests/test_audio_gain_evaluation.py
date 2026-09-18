"""Deterministic cross-validation and audio gain model tests."""

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from kestrel.audio_gain_evaluation import (
    ModelSpec,
    audio_gain_evaluate,
    band_metrics,
    fit_huber,
    fit_ols,
    fit_piecewise,
    grouped_folds,
    predict_gain,
    prediction_metrics,
)
from kestrel.cli import main


def observation(
    catalog_id: str, raw: float, gain: float, fragment: str | None = None
) -> dict[str, object]:
    return {
        "catalog_id": catalog_id,
        "fragment_id": fragment or f"fragment-{catalog_id}",
        "filename": f"C:/media/{catalog_id}.mp4",
        "raw_rms_dbfs": raw,
        "manual_gain_db": gain,
    }


def test_grouped_folds_are_deterministic_without_source_leakage() -> None:
    values = [
        observation("a", -50, 5, "a1"),
        observation("a", -50, 6, "a2"),
        observation("b", -40, 0),
        observation("c", -30, -5),
        observation("d", -20, -10),
    ]
    first = grouped_folds(values, 3)
    assert first == grouped_folds(values, 3)
    source_fold: dict[str, int] = {}
    for fold_index, indices in enumerate(first):
        for index in indices:
            source = str(values[index]["catalog_id"])
            assert source_fold.setdefault(source, fold_index) == fold_index


def test_ols_and_robust_regression() -> None:
    clean = [
        observation(str(index), raw, 2 - 0.5 * raw)
        for index, raw in enumerate((-60.0, -50.0, -40.0, -30.0, -20.0))
    ]
    fitted = fit_ols(clean)
    assert fitted["slope"] == pytest.approx(-0.5)
    assert fitted["intercept"] == pytest.approx(2)

    contaminated = clean + [observation("outlier", -40, 40)]
    ordinary = fit_ols(contaminated)
    robust = fit_huber(contaminated)
    expected = 22.0
    ordinary_at_outlier = ordinary["intercept"] + ordinary["slope"] * -40
    robust_at_outlier = robust["intercept"] + robust["slope"] * -40
    assert abs(robust_at_outlier - expected) < abs(ordinary_at_outlier - expected)
    assert robust["reduced_influence_count"] >= 1


def test_deadband_and_clamps() -> None:
    parameters = {"intercept": 0.0, "slope": 1.0}
    deadband = ModelSpec("deadband", "deadband", deadband_db=1.0)
    assert predict_gain(deadband, parameters, 0.75) == 0
    assert predict_gain(deadband, parameters, 1.0) == 1.0

    symmetric = ModelSpec("clamp", "clamp", minimum_gain_db=-6, maximum_gain_db=6)
    assert predict_gain(symmetric, parameters, -10) == -6
    assert predict_gain(symmetric, parameters, 10) == 6

    asymmetric = ModelSpec(
        "asymmetric", "clamp", minimum_gain_db=-18.35, maximum_gain_db=18
    )
    assert predict_gain(asymmetric, parameters, -30) == -18.35
    assert predict_gain(asymmetric, parameters, 30) == 18


def test_continuous_piecewise_prediction() -> None:
    values = [
        observation(str(index), raw, gain)
        for index, (raw, gain) in enumerate(
            [(-60, 20), (-50, 10), (-40, 0), (-30, -5), (-20, -10)]
        )
    ]
    parameters = fit_piecewise(values, (0.5,))
    spec = ModelSpec("piecewise", "piecewise", breakpoint_quantiles=(0.5,))
    assert parameters["breakpoints_dbfs"] == [-40.0]
    assert parameters["segment_slopes"] == pytest.approx([-1.0, -0.5])
    assert predict_gain(spec, parameters, -35) == pytest.approx(-2.5)


def test_metrics_and_raw_level_bands() -> None:
    metrics = prediction_metrics([0, 2], [1, 0])
    assert metrics["mae_db"] == 1.5
    assert metrics["median_absolute_error_db"] == 1.5
    assert metrics["rmse_db"] == pytest.approx(math.sqrt(2.5))
    assert metrics["bias_db"] == -0.5

    values = [
        observation("a", -60, 10),
        observation("b", -50, 5),
        observation("c", -40, 0),
        observation("d", -30, -5),
        observation("e", -20, -10),
    ]
    bands = band_metrics(values, [8, 4, 0, -4, -8])
    assert all(value["count"] == 1 for value in bands.values())
    assert bands["below_-55"]["mae_db"] == 2
    assert bands["at_least_-25"]["bias_db"] == 2


def _calibration_file(tmp_path: Path) -> Path:
    values = [
        observation(
            f"source-{index}",
            -65.0 + index * 4,
            12.0 - index * 1.8 + (0.5 if index % 2 else -0.5),
        )
        for index in range(12)
    ]
    calibration = {"version": 1, "fragment_observations": values}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    return path


def test_evaluation_input_integrity_and_output_safety(tmp_path: Path) -> None:
    calibration = _calibration_file(tmp_path)
    before = hashlib.sha256(calibration.read_bytes()).hexdigest()
    output = tmp_path / "models.json"
    summary = audio_gain_evaluate(calibration, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary["observation_count"] == 12
    assert summary["source_count"] == 12
    assert summary["fold_count"] == 10
    assert result["tested_model_count"] >= 20
    assert result["models"]["ols_linear"]["full_dataset_fit"]["slope"] < 0
    assert len(result["evaluation_observations"]) == 12
    first = result["evaluation_observations"][0]
    assert first["fragment_id"] == "fragment-source-0"
    assert set(first["cross_validated_predictions"]) == set(result["models"])
    assert hashlib.sha256(calibration.read_bytes()).hexdigest() == before

    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        audio_gain_evaluate(calibration, calibration)
    with pytest.raises(ValueError, match="already exists"):
        audio_gain_evaluate(calibration, existing)
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_audio_gain_evaluate_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _calibration_file(tmp_path)
    output = tmp_path / "models.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["kestrel", "audio-gain-evaluate", str(calibration), str(output)],
    )
    main()
    assert json.loads(output.read_text(encoding="utf-8"))["observation_count"] == 12
