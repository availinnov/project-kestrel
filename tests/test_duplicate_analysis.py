"""Synthetic visual controls and read-only bounded pipeline checks."""

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kestrel.duplicate_analysis import (
    HEIGHT,
    WIDTH,
    compare_frames,
    describe_frame,
    duplicate_analyze,
    image_similarity,
    neighbor_pairs,
    sample_source,
)
from kestrel.duplicate_evaluation import evaluate_controls
from kestrel.project.parser import ProjectParseError
from tests.test_dataset import dataset_fixture


def image(seed: int) -> bytes:
    rng = random.Random(seed)
    cells = [rng.randrange(20, 230) for _ in range(32 * 18)]
    return bytes(
        cells[(y // 5) * 32 + x // 5] for y in range(HEIGHT) for x in range(WIDTH)
    )


def test_visual_controls_and_shift() -> None:
    raw = image(1)
    a = describe_frame(raw)
    modified = describe_frame(bytes(min(255, p + 8) for p in raw))
    unrelated = describe_frame(image(2))
    assert image_similarity(a, a) == pytest.approx(1)
    assert image_similarity(a, modified) > 0.99
    assert image_similarity(a, unrelated) < 0.2
    result = compare_frames([a, unrelated], [unrelated, a])
    assert result["hash_best_match_median"] == 0
    assert result["image_similarity_median"] == pytest.approx(1)
    assert compare_frames([a], [unrelated]) == compare_frames([unrelated], [a])
    black = describe_frame(bytes(WIDTH * HEIGHT))
    white = describe_frame(bytes([255]) * WIDTH * HEIGHT)
    assert image_similarity(black, black) == 1
    assert image_similarity(black, white) == 0


def test_neighbors() -> None:
    assert neighbor_pairs(4, 1) == [(0, 1), (1, 2), (2, 3)]
    assert neighbor_pairs(4, 2) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    assert len(neighbor_pairs(10000, 2)) == 19997
    assert len(neighbor_pairs(41, 2)) == 79
    assert neighbor_pairs(0, 2) == []


def test_single_process_seek_strategy(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"fixture")
    calls = []

    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        small = image(1)
        packed = b"".join(
            small[y * WIDTH : (y + 1) * WIDTH] + bytes(160) for y in range(HEIGHT)
        ) + bytes(320 * 180)
        return subprocess.CompletedProcess(command, 0, packed * 5, b"")

    monkeypatch.setattr("kestrel.duplicate_analysis.subprocess.run", run)
    sampled = sample_source(str(source), 10, "ffmpeg", 5)
    assert len(sampled) == 5
    assert all(frame.orb is not None for frame in sampled)
    assert len(calls) == 1
    command = calls[0]
    assert [command[i + 1] for i, v in enumerate(command) if v == "-ss"] == [
        "1.000000000",
        "3.000000000",
        "5.000000000",
        "7.000000000",
        "9.000000000",
    ]


@pytest.mark.parametrize("failure", [False, True])
def test_pipeline_schema_order_and_preservation(
    tmp_path: Path, monkeypatch: Any, failure: bool
) -> None:
    source = dataset_fixture(tmp_path)
    before = source.read_bytes()
    calls = []

    def sample(filename: str, *args: Any) -> Any:
        calls.append(filename)
        if failure and len(calls) == 2:
            raise subprocess.TimeoutExpired("ffmpeg", 60)
        return [describe_frame(image(1))]

    monkeypatch.setattr("kestrel.duplicate_analysis.sample_source", sample)
    output = tmp_path / "analysis.json"
    result = duplicate_analyze(source, output)
    data = json.loads(output.read_text())
    assert source.read_bytes() == before
    assert len(calls) == 3
    assert [s["catalog_id"] for s in data["sources"]] == ["m0", "m1", "m2"]
    assert result["pair_count"] == 3
    assert result["measured_pair_count"] == (1 if failure else 3)
    assert data["schema_version"] == 1
    assert data["descriptor_definitions"]["hash"]
    for pair in data["pairs"]:
        assert set(pair["left"]) == {"catalog_id", "filename"}
        assert "duplicate" not in pair
        if pair["analysis_status"] == "ok":
            assert pair["signals"]["image_similarity_median"] == pytest.approx(1)
    with pytest.raises(ProjectParseError):
        duplicate_analyze(source, source)
    with pytest.raises(ProjectParseError):
        duplicate_analyze(source, output)


def test_actual_missing_media(tmp_path: Path) -> None:
    source = dataset_fixture(tmp_path)
    result = duplicate_analyze(source, tmp_path / "missing.json")
    assert result["failed_source_count"] == 3
    assert result["measured_pair_count"] == 0


def test_failed_decode(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "bad.mp4"
    source.write_bytes(b"invalid")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr("kestrel.duplicate_analysis.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        sample_source(str(source), 1, "ffmpeg", 5)


def test_offline_evaluation_counts() -> None:
    pairs = [
        dict(
            left=dict(filename="a"),
            right=dict(filename=name),
            analysis_status="ok",
            signals=dict(
                image_similarity_median=similarity, hash_best_match_median=distance
            ),
        )
        for name, similarity, distance in [("b", 0.8, 10), ("c", 0.9, 5)]
    ]
    result = evaluate_controls(dict(summary={}, pairs=pairs), [["b", "a"]])
    threshold = next(
        t
        for t in result["threshold_exploration"]
        if t["signal"] == "image_similarity_median" and t["threshold"] == 0.85
    )
    assert threshold["positive_controls_matched"] == 0
    assert threshold["unlabeled_pairs_matched"] == 1
    assert threshold["known_positives_missed"] == 1
    assert "negative_controls_matched" not in threshold
    assert "provisional_false_positive_count" not in threshold
    assert len(threshold["missed_positive_pairs"]) == 1
    assert result["ungenerated_positive_pairs"] == []


def test_duplicate_cli(tmp_path: Path) -> None:
    source = dataset_fixture(tmp_path)
    output = tmp_path / "cli.json"
    result = subprocess.run(
        [
            "kestrel",
            "duplicate-analyze",
            str(source),
            str(output),
            "--max-neighbor-distance",
            "1",
            "--frame-samples",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert json.loads(output.read_text())["summary"]["pair_count"] == 2
