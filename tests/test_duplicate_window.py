"""Time-window candidates and incremental, canonical review label preservation."""

import json
from pathlib import Path
from typing import Any

import pytest

from kestrel.duplicate_analysis import (
    capture_window_pairs,
    describe_frame,
    duplicate_analyze,
)
from kestrel.duplicate_calibration import duplicate_calibrate
from kestrel.duplicate_evaluation import evaluate_controls, write_review_queue
from kestrel.project.parser import ProjectParseError
from tests.test_dataset import dataset_fixture
from tests.test_duplicate_analysis import image
from tests.test_duplicate_calibration import fixture, pair
from tests.test_sequence import rewrite_fixture


def test_window_inclusive_unrestricted_and_deterministic() -> None:
    sources = [dict(capture_time=t) for t in (100, 110, 120, 130, 280, 281)]
    pairs = capture_window_pairs(sources)
    assert (0, 3) in pairs and (0, 4) in pairs
    assert (0, 5) not in pairs
    assert pairs == capture_window_pairs(sources)
    assert capture_window_pairs(sources, 0) == []
    assert capture_window_pairs([dict(capture_time=None)]) == []
    with pytest.raises(ValueError, match="chronologically"):
        capture_window_pairs(list(reversed(sources)))


def test_sparse_window_early_stop() -> None:
    # A quadratic scan would make this deliberately large sparse case expensive.
    assert (
        capture_window_pairs([dict(capture_time=100 + i * 181) for i in range(10000)])
        == []
    )


def test_analyzer_defaults_to_window(tmp_path: Path, monkeypatch: Any) -> None:
    source = dataset_fixture(tmp_path)

    def change(documents: Any) -> None:
        for i, timestamp in enumerate((100, 280, 281)):
            documents[f"ProjectFolder/Medias/m{i}/media.json"]["sourceInfo"][
                "basicInfo"
            ]["createDate"] = timestamp

    rewrite_fixture(source, change)
    monkeypatch.setattr(
        "kestrel.duplicate_analysis.sample_source",
        lambda *args: [describe_frame(image(1))],
    )

    def forbidden(*args: Any) -> Any:
        raise AssertionError("Default must not use the old neighbor gate")

    monkeypatch.setattr("kestrel.duplicate_analysis.neighbor_pairs", forbidden)
    output = tmp_path / "window.json"
    result = duplicate_analyze(source, output)
    assert result["pair_count"] == 2
    data = json.loads(output.read_text())
    assert data["config"]["candidate_policy"] == "capture_time_window"
    assert [p["capture_time_gap_seconds"] for p in data["pairs"]] == [180, 1]


def test_queue_only_new_unlabeled_and_reversed_labels(tmp_path: Path) -> None:
    pairs = [pair("a", name) for name in "bcdef"]
    for p in pairs:
        p["signals"].update(
            orb_best_inlier_ratio=1,
            orb_best_good_match_ratio=0.1,
            hash_best_match_median=10,
        )
    report = evaluate_controls(dict(summary={}, pairs=pairs), [["e", "a"]])
    old = dict(pairs=[pair("a", "b"), pair("a", "c")])
    reviewed = [dict(pair("d", "a"), manual_same_scene=False)]
    before = json.dumps(reviewed)
    output = tmp_path / "new.json"
    write_review_queue(report, output, previous_analysis=old, reviewed_pairs=reviewed)
    queue = json.loads(output.read_text())
    assert [p["right"]["filename"] for p in queue["pairs"]] == ["f"]
    assert queue["pairs"][0]["manual_same_scene"] is None
    assert json.dumps(reviewed) == before
    second = tmp_path / "second.json"
    write_review_queue(report, second, previous_analysis=old, reviewed_pairs=reviewed)
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        write_review_queue(report, output, previous_analysis=old)


def test_calibration_merges_queues_without_inferred_labels(tmp_path: Path) -> None:
    analysis, review = fixture()
    a, q, extra, out = (
        tmp_path / name for name in ("a.json", "q.json", "extra.json", "out.json")
    )
    a.write_text(json.dumps(analysis))
    q.write_text(json.dumps(review))
    extra.write_text(
        json.dumps(
            dict(
                pairs=[
                    dict(pair("d", "a"), manual_same_scene=False),
                    dict(pair("a", "b"), manual_same_scene=True),
                ]
            )
        )
    )
    before = [p.read_bytes() for p in (a, q, extra)]
    duplicate_calibrate(a, q, out, additional_review_paths=[extra])
    result = json.loads(out.read_text())
    assert result["label_dataset_summary"]["total_manual_positives"] == 1
    assert result["label_dataset_summary"]["total_manual_negatives"] == 2
    assert len(result["labeled_dataset"]) == 2
    assert [p.read_bytes() for p in (a, q, extra)] == before
    with pytest.raises(ProjectParseError):
        duplicate_calibrate(a, q, extra, additional_review_paths=[extra])
    extra.write_text(
        json.dumps(dict(pairs=[dict(pair("b", "a"), manual_same_scene=False)]))
    )
    with pytest.raises(ProjectParseError, match="Conflicting"):
        duplicate_calibrate(
            a, q, tmp_path / "conflict.json", additional_review_paths=[extra]
        )
