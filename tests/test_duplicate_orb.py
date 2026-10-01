"""Generated image transformations and ORB geometric failure controls."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from kestrel.duplicate_evaluation import write_review_queue
from kestrel.duplicate_orb import OrbFrame, compare_orb, describe_orb, match_orb


def textured(seed: int = 1) -> Any:
    rng = np.random.default_rng(seed)
    image = np.full((180, 320), 90, dtype=np.uint8)
    for _ in range(160):
        center = tuple(int(v) for v in rng.integers([35, 35], [285, 145]))
        cv2.circle(
            image, center, int(rng.integers(2, 7)), int(rng.integers(0, 256)), -1
        )
    return image


@pytest.mark.parametrize(
    "transform", ["identity", "translation", "scale", "rotation", "brightness"]
)
def test_transformations(transform: str) -> None:
    image = textured()
    altered = image.copy()
    if transform == "translation":
        altered = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, 5]]), (320, 180))
    elif transform in ("scale", "rotation"):
        matrix = cv2.getRotationMatrix2D(
            (160, 90),
            7 if transform == "rotation" else 0,
            1.12 if transform == "scale" else 1,
        )
        altered = cv2.warpAffine(image, matrix, (320, 180))
    elif transform == "brightness":
        altered = np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    result = match_orb(describe_orb(image), describe_orb(altered))
    assert result["orb_good_match_count"] >= 30
    assert result["orb_inlier_ratio"] > 0.7
    if transform == "identity":
        assert result["orb_good_match_ratio"] == 1
        assert result["orb_inlier_ratio"] == 1


def test_unrelated_and_flat() -> None:
    a, b = describe_orb(textured(1)), describe_orb(textured(55))
    result = match_orb(a, b)
    assert result["orb_good_match_ratio"] < 0.1
    flat = describe_orb(np.zeros((180, 320), dtype=np.uint8))
    for left, right in ((flat, flat), (a, flat), (flat, a)):
        result = match_orb(left, right)
        assert result["orb_good_match_count"] == 0
        assert result["orb_inlier_count"] == 0
        assert result["orb_inlier_ratio"] == 0


def test_insufficient_and_failed_homography(monkeypatch: Any) -> None:
    a = describe_orb(textured())
    tiny = OrbFrame(a.points[:3], a.descriptors[:3])
    result = match_orb(tiny, tiny)
    assert result["orb_good_match_count"] == 3
    assert result["orb_inlier_count"] == 0
    monkeypatch.setattr(
        "kestrel.duplicate_orb.cv2.findHomography", lambda *args, **kwargs: (None, None)
    )
    result = match_orb(a, a)
    assert result["orb_good_match_count"] > 4
    assert result["orb_inlier_ratio"] == 0


def test_deterministic_and_frame_shift() -> None:
    a, b = describe_orb(textured(1)), describe_orb(textured(55))
    first = compare_orb([a, b], [b, a])
    assert first == compare_orb([a, b], [b, a])
    assert first["orb_match_ratio_median"] == 1
    assert first["orb_inlier_ratio_median"] == 1
    assert len(first["orb_frame_matches"]) == 8
    again = describe_orb(textured(1))
    assert np.array_equal(a.descriptors, again.descriptors)
    assert np.array_equal(a.points, again.points)


def test_review_queue_preserves_manual_labels(tmp_path: Path) -> None:
    output = tmp_path / "queue.json"
    controls = [
        dict(
            left=dict(filename="a"),
            right=dict(filename="b"),
            signals=dict(
                orb_best_inlier_count=8,
                orb_best_good_match_ratio=0.1,
                orb_best_inlier_ratio=0.8,
            ),
        )
    ]
    report = dict(unlabeled_controls=controls)
    write_review_queue(report, output)
    data = json.loads(output.read_text())
    assert data["pairs"][0]["manual_same_scene"] is None
    data["pairs"][0]["manual_same_scene"] = True
    output.write_text(json.dumps(data))
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_review_queue(report, output)
    assert output.read_bytes() == before
