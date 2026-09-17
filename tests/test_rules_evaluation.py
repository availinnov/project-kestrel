"""Offline rule re-evaluation over existing analysis signals."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from kestrel.cli import main
from kestrel.rules_evaluation import rules_evaluate


def _rule(
    rule_id: str,
    signal: str,
    op: str,
    value: Any,
    action: dict[str, Any],
    *,
    priority: int = 10,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "enabled": enabled,
        "mode": "auto",
        "priority": priority,
        "when": {"signal": signal, "op": op, "value": value},
        "then": action,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    analysis = {
        "version": 1,
        "project": {"source_count": 2},
        "detector_config": {"max_frames": 24},
        "measurement_notes": {"audio": "measured once"},
        "sources": [
            {
                "catalog_id": "a",
                "filename": "file:/a.mp4",
                "source_duration_ticks": 10_000_000,
                "signals": {
                    "duration_seconds": 1.0,
                    "video": {"black_frame_ratio": 0.0},
                    "audio": {"rms_dbfs": None},
                },
                "analysis_status": "partial",
                "error": "audio unavailable",
                "detector_notes": {"sample": "preserve"},
                "actions": [{"type": "warning", "rule_id": "stale"}],
            },
            {
                "catalog_id": "b",
                "filename": "file:/b.mp4",
                "source_duration_ticks": 30_000_000,
                "signals": {
                    "duration_seconds": 3.0,
                    "video": None,
                    "audio": {"rms_dbfs": -10.0},
                },
                "analysis_status": "ok",
                "actions": [{"type": "drop", "rule_id": "wrong"}],
            },
        ],
        "summary": {
            "source_count": 2,
            "total_runtime_seconds": 123.0,
            "rule_match_counts": {"stale": 2},
            "proposed_quiet_normalization_count": 99,
            "proposed_loud_normalization_count": 99,
            "proposed_short_peak_review_count": 99,
        },
    }
    rules = {
        "version": 1,
        "rules": [
            _rule(
                "missing",
                "audio.short_relative_loud_event_count",
                "!=",
                -18,
                {"warning": "missing"},
                priority=40,
            ),
            _rule(
                "disabled",
                "duration_seconds",
                ">",
                0,
                {"warning": "disabled"},
                priority=30,
                enabled=False,
            ),
            _rule(
                "short_warning",
                "duration_seconds",
                "<",
                2,
                {"warning": "short"},
                priority=20,
            ),
            _rule(
                "short_drop",
                "duration_seconds",
                "<",
                2,
                {"drop": True},
                priority=10,
            ),
        ],
    }
    analysis_path = tmp_path / "analysis.json"
    rules_path = tmp_path / "rules.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    return analysis_path, rules_path, analysis


def test_recomputes_actions_and_preserves_measurements(tmp_path: Path) -> None:
    analysis_path, rules_path, original = _write_inputs(tmp_path)
    output = tmp_path / "output.json"
    analysis_hash = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    rules_hash = hashlib.sha256(rules_path.read_bytes()).hexdigest()

    report = rules_evaluate(analysis_path, rules_path, output)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert [action["rule_id"] for action in result["sources"][0]["actions"]] == [
        "short_warning",
        "short_drop",
    ]
    assert result["sources"][1]["actions"] == []
    assert report["source_count"] == report["evaluated_source_count"] == 2
    assert report["rule_count"] == 4
    assert report["rule_match_counts"] == {"short_warning": 1, "short_drop": 1}
    assert report["action_type_counts"] == {
        "drop": 1,
        "normalize_audio": 0,
        "color_tag": 0,
        "warning": 1,
    }

    for before, after in zip(original["sources"], result["sources"], strict=True):
        expected = copy.deepcopy(before)
        expected.pop("actions")
        actual = copy.deepcopy(after)
        actual.pop("actions")
        assert actual == expected
    assert result["detector_config"] == original["detector_config"]
    assert result["measurement_notes"] == original["measurement_notes"]
    assert result["summary"]["total_runtime_seconds"] == 123.0
    assert result["summary"]["rule_match_counts"] == report["rule_match_counts"]
    assert hashlib.sha256(analysis_path.read_bytes()).hexdigest() == analysis_hash
    assert hashlib.sha256(rules_path.read_bytes()).hexdigest() == rules_hash


def test_output_file_safety(tmp_path: Path) -> None:
    analysis_path, rules_path, _ = _write_inputs(tmp_path)
    existing = tmp_path / "existing.json"
    existing.write_text("do not replace", encoding="utf-8")

    for output in (analysis_path, rules_path):
        with pytest.raises(ValueError, match="new file distinct"):
            rules_evaluate(analysis_path, rules_path, output)
    with pytest.raises(ValueError, match="already exists"):
        rules_evaluate(analysis_path, rules_path, existing)
    assert existing.read_text(encoding="utf-8") == "do not replace"


def test_cli_rules_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_path, rules_path, _ = _write_inputs(tmp_path)
    output = tmp_path / "cli-output.json"
    monkeypatch.setattr(
        "sys.argv",
        ["kestrel", "rules-evaluate", str(analysis_path), str(rules_path), str(output)],
    )
    main()
    assert output.is_file()
    assert "evaluated_source_count: 2" in capsys.readouterr().out


@pytest.mark.parametrize(
    "analysis",
    [
        {},
        {"version": 2, "sources": []},
        {"version": 1, "sources": {}},
        {"version": 1, "sources": [{}]},
    ],
)
def test_analysis_schema_validation(tmp_path: Path, analysis: dict[str, Any]) -> None:
    analysis_path, rules_path, _ = _write_inputs(tmp_path)
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(ValueError):
        rules_evaluate(analysis_path, rules_path, tmp_path / "output.json")
