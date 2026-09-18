"""Fragment-level calibration of source RMS against manual gain edits."""

import hashlib
import json
from pathlib import Path

import pytest

from kestrel.audio_calibration import (
    audio_calibrate,
    linear_regression,
    target_simulation,
)
from kestrel.cli import main


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    sources = [
        {
            "catalog_id": "a",
            "filename": "C:/media/a.mp4",
            "fragment_count": 2,
            "retained_ratio": 0.5,
            "overlapping_retained_ranges": False,
        },
        {
            "catalog_id": "b",
            "filename": "C:/media/b.mp4",
            "fragment_count": 3,
            "retained_ratio": 0.75,
            "overlapping_retained_ranges": False,
        },
        {
            "catalog_id": "c",
            "filename": "C:/media/c.mp4",
            "fragment_count": 1,
            "retained_ratio": 1.0,
            "overlapping_retained_ranges": False,
        },
        {
            "catalog_id": "d",
            "filename": "C:/media/d.mp4",
            "fragment_count": 1,
            "retained_ratio": 1.0,
            "overlapping_retained_ranges": False,
        },
    ]

    def fragment(
        fragment_id: str,
        catalog_id: str,
        gain: float | None,
        status: str = "matched",
        position: int = 0,
    ) -> dict[str, object]:
        return {
            "fragment_id": fragment_id,
            "catalog_id": catalog_id,
            "clip_id": f"clip-{fragment_id}",
            "track_id": "track",
            "track_index": 1,
            "source_in": 0,
            "source_out": 10,
            "source_duration": 10,
            "timeline_begin": position,
            "timeline_end": position + 10,
            "timeline_duration": 10,
            "audio_gain_db": gain,
            "audio_pair_status": status,
            "sequence_index": position,
        }

    dataset = {
        "version": 1,
        "project": {"source_count": 4},
        "sources": sources,
        "timeline_fragments": [
            fragment("f1", "a", 10, position=0),
            fragment("f2", "a", 12, position=10),
            fragment("f3", "b", 0, position=20),
            fragment("f4", "c", 3, position=30),
            fragment("f5", "d", 4, position=40),
            fragment("f6", "b", None, position=50),
            fragment("f7", "b", 1, "ambiguous", position=60),
        ],
    }

    def analyzed(catalog_id: str, rms: float | None) -> dict[str, object]:
        return {
            "catalog_id": catalog_id,
            "filename": f"different/{catalog_id}.mp4",
            "source_duration_ticks": 10,
            "signals": {"audio": {"rms_dbfs": rms}},
            "analysis_status": "ok",
            "actions": [],
        }

    analysis = {
        "version": 1,
        "project": {"source_count": 3},
        "detector_config": {},
        "measurement_notes": {},
        "sources": [analyzed("a", -30), analyzed("b", -20), analyzed("c", None)],
        "summary": {},
    }
    dataset_path = tmp_path / "dataset.json"
    analysis_path = tmp_path / "analysis.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    return dataset_path, analysis_path


def test_join_exclusions_observations_and_source_aggregates(tmp_path: Path) -> None:
    dataset, analysis = _write_inputs(tmp_path)
    output = tmp_path / "calibration.json"
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    analysis_hash = hashlib.sha256(analysis.read_bytes()).hexdigest()

    summary = audio_calibrate(dataset, analysis, output)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert summary["eligible_fragment_count"] == 3
    assert result["counts"]["excluded_counts"] == {
        "raw_rms_unavailable": 1,
        "analysis_source_unavailable": 1,
        "manual_gain_unavailable": 1,
        "audio_pair_ambiguous": 1,
    }
    assert len(result["excluded_fragments"]) == 4
    assert {item["reason"] for item in result["excluded_fragments"]} == set(
        result["counts"]["excluded_counts"]
    )
    observations = result["fragment_observations"]
    assert [item["fragment_id"] for item in observations] == ["f1", "f2", "f3"]
    assert [item["effective_rms_dbfs"] for item in observations] == [-20, -18, -20]
    assert all(
        item["filename"].endswith(f"{item['catalog_id']}.mp4") for item in observations
    )
    source_a = next(
        item for item in result["source_aggregates"] if item["catalog_id"] == "a"
    )
    assert source_a["observation_count"] == 2
    assert source_a["manual_gain_distribution"]["mean"] == 11
    assert source_a["manual_gain_range_db"] == 2
    assert len(result["repeated_source_gain_inconsistencies"]) == 1
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == dataset_hash
    assert hashlib.sha256(analysis.read_bytes()).hexdigest() == analysis_hash


def test_regression_and_target_simulation() -> None:
    regression = linear_regression([-30, -20, -10], [17, 12, 7])
    assert regression["slope"] == pytest.approx(-0.5)
    assert regression["intercept"] == pytest.approx(2)
    assert regression["r_squared"] == pytest.approx(1)
    assert regression["residual_stddev"] == pytest.approx(0)

    simulation = target_simulation([-30, -20], [10, 0], -20)
    assert simulation["mae_db"] == 0
    assert simulation["rmse_db"] == 0
    assert simulation["within_1_db_percent"] == 100


def test_explicit_exclusion_and_output_safety(tmp_path: Path) -> None:
    dataset, analysis = _write_inputs(tmp_path)
    output = tmp_path / "excluded.json"
    summary = audio_calibrate(dataset, analysis, output, exclude_filenames=["a.mp4"])
    assert summary["eligible_fragment_count"] == 1
    assert summary["excluded_counts"]["explicit_filename_exclusion"] == 2

    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    for unsafe in (dataset, analysis):
        with pytest.raises(ValueError, match="distinct"):
            audio_calibrate(dataset, analysis, unsafe)
    with pytest.raises(ValueError, match="already exists"):
        audio_calibrate(dataset, analysis, existing)
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_cli_audio_calibrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, analysis = _write_inputs(tmp_path)
    output = tmp_path / "cli.json"
    monkeypatch.setattr(
        "sys.argv",
        ["kestrel", "audio-calibrate", str(dataset), str(analysis), str(output)],
    )
    main()
    assert output.is_file()
    assert "eligible_fragment_count: 3" in capsys.readouterr().out
