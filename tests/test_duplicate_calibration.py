"""Offline label merging, time gates, feature math, and candidate evaluation."""

import json
from pathlib import Path
from typing import Any

import pytest

from kestrel.duplicate_calibration import (
    calibrate_data,
    candidate_rules,
    canonical_pair,
    derive_features,
    duplicate_calibrate,
    evaluate_rule,
    merge_labels,
    missing_pair_coverage,
    predict,
    time_eligible,
)
from kestrel.project.parser import ProjectParseError


def pair(left: str = "a", right: str = "b", gap: int = 10) -> dict[str, Any]:
    return dict(
        left=dict(filename=left),
        right=dict(filename=right),
        capture_time_gap_seconds=gap,
        analysis_status="ok",
        signals=dict(
            orb_best_inlier_count=8,
            orb_best_good_match_count=10,
            image_similarity_median=0.8,
        ),
    )


def match(
    i: int, j: int, count: int, direction: str = "left_to_right"
) -> dict[str, Any]:
    return dict(
        query_frame_index=i,
        target_frame_index=j,
        direction=direction,
        orb_keypoints_left=10,
        orb_keypoints_right=20,
        orb_inlier_count=count,
        orb_good_match_count=count + 2,
    )


def test_canonical_and_label_merge() -> None:
    assert canonical_pair("C:/media/A.mp4", "b.mp4") == canonical_pair("B.mp4", "a.mp4")
    review = [
        dict(pair(), manual_same_scene=True),
        dict(pair("b", "a"), manual_same_scene=True),
        dict(pair("b", "c"), manual_same_scene=False),
        dict(pair("a", "d"), manual_same_scene=None),
    ]
    labels, counts = merge_labels([["a", "b"], ["b", "a"]], review)
    assert len(labels) == 2
    assert labels[("a", "b")]["origins"] == ["known_positive", "manual"]
    assert counts["total_known_positives"] == 1
    assert counts["total_manual_positives"] == 1
    assert counts["total_manual_negatives"] == 1


def test_conflicts_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        merge_labels([["a", "b"]], [dict(pair("b", "a"), manual_same_scene=False)])
    with pytest.raises(ValueError, match="Conflicting"):
        merge_labels(
            [],
            [
                dict(pair(), manual_same_scene=True),
                dict(pair("b", "a"), manual_same_scene=False),
            ],
        )


@pytest.mark.parametrize(
    ("gap", "allowed"),
    [(0, True), (180, True), (181, False), (-1, False), (None, False), (True, False)],
)
def test_inclusive_time_gate(gap: Any, allowed: bool) -> None:
    assert time_eligible(gap) is allowed


def test_features_physical_pairs_and_coverage() -> None:
    p = pair()
    p["signals"]["orb_frame_matches"] = [
        match(0, 1, 8),
        match(1, 0, 8, "right_to_left"),
        match(1, 2, 12),
    ]
    features, missing = derive_features(p, 5, 5)
    assert missing == []
    assert features["orb_best_inliers_per_min_keypoints"] == 1
    assert features["orb_best_good_matches_per_min_keypoints"] == 1
    assert features["orb_frame_pair_count_with_inliers_ge_4"] == 2
    assert features["orb_frame_pair_count_with_inliers_ge_8"] == 2
    assert features["orb_frame_pair_count_with_inliers_ge_12"] == 1
    assert features["orb_source_frame_coverage"] == 0.4
    assert features["orb_best_inlier_count"] == 8  # retain original raw summary
    p["signals"]["orb_frame_matches"] = [match(0, 0, 4)]
    assert derive_features(p, 5, 5)[0]["orb_best_inliers_per_min_keypoints"] == 0.4


def test_missing_features_are_not_invented() -> None:
    features, missing = derive_features(pair(), 5, 5)
    assert features["orb_source_frame_coverage"] is None
    assert missing
    rule = next(
        r for r in candidate_rules() if r["rule_id"] == "inliers_4_coverage_0.4"
    )
    assert predict(rule, dict(features=features, capture_time_gap_seconds=10)) is None


def test_confusion_and_determinism() -> None:
    rule = candidate_rules()[0]
    rows = [
        dict(
            same_scene=label,
            features=dict(orb_best_inlier_count=count),
            capture_time_gap_seconds=10,
        )
        for label, count in [(True, 8), (True, 0), (False, 8), (False, 0)]
    ]
    result = evaluate_rule(rule, rows)
    assert result == evaluate_rule(rule, rows)
    assert [
        result[k]
        for k in (
            "true_positives",
            "false_negatives",
            "true_negatives",
            "false_positives",
        )
    ] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert len(result["mismatched_positive_pairs"]) == 1
    assert len(result["mismatched_negative_pairs"]) == 1
    assert candidate_rules() == candidate_rules()
    rows[0]["capture_time_gap_seconds"] = 181
    assert predict(rule, rows[0]) is False


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = dict(
        sources=[
            dict(filename=s, capture_time=100 + i * 60, sampled_frame_count=5)
            for i, s in enumerate("abcd")
        ],
        pairs=[pair(gap=180), pair("b", "c", 181), pair("a", "d", 10)],
    )
    review = dict(
        pairs=[
            dict(pair("b", "a"), manual_same_scene=True),
            dict(pair("b", "c"), manual_same_scene=False),
        ]
    )
    return analysis, review


def test_filter_before_calibration_and_unlabeled_exclusion() -> None:
    analysis, review = fixture()
    result = calibrate_data(analysis, review, [["a", "b"]])
    assert len(result["labeled_dataset"]) == 1
    assert result["time_filter_summary"]["remaining_known_positives"] == 1
    assert result["time_filter_summary"]["remaining_manual_negatives"] == 0
    assert len(result["time_filter_summary"]["excluded_labeled_pairs"]) == 1
    assert all(
        c["false_positives"] == c["true_negatives"] == 0
        for c in result["candidate_rules"]
    )
    assert result["production_rule"] is None


def test_missing_neighbor_coverage() -> None:
    sources = [
        dict(filename=s, capture_time=100 + i * 20) for i, s in enumerate("abcd")
    ]
    existing = {
        canonical_pair("a", "b"),
        canonical_pair("a", "c"),
        canonical_pair("b", "c"),
        canonical_pair("b", "d"),
        canonical_pair("c", "d"),
    }
    result = missing_pair_coverage(sources, existing, 2)
    assert result["potential_short_gap_pair_count"] == 6
    assert result["missing_short_gap_pair_count"] == 1
    assert result["missing_due_to_old_neighbor_limit"] == 1


def test_json_unchanged_no_decoding_and_output_safety(
    tmp_path: Path, monkeypatch: Any
) -> None:
    analysis, review = fixture()
    a, q, output = (
        tmp_path / name for name in ("analysis.json", "review.json", "out.json")
    )
    a.write_text(json.dumps(analysis))
    q.write_text(json.dumps(review))
    before = [a.read_bytes(), q.read_bytes()]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Calibration must not decode media")

    monkeypatch.setattr("subprocess.run", forbidden)
    duplicate_calibrate(a, q, output)
    assert [a.read_bytes(), q.read_bytes()] == before
    assert json.loads(output.read_text())["production_rule"] is None
    for target in (a, q, output):
        with pytest.raises(ProjectParseError):
            duplicate_calibrate(a, q, target)
