"""Synthetic measurements and pipeline failures without external media."""

import array
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kestrel.media_analysis import analyze_source, media_analyze
from kestrel.media_detectors import (
    AudioAccumulator,
    DetectorConfig,
    audio_measurements,
    duration_signal,
    local_media_path,
    sample_times,
    video_measurements,
)
from tests.test_dataset import dataset_fixture
from tests.test_sequence import rewrite_fixture


def test_duration_and_paths() -> None:
    assert duration_signal(120750000) == 12.075
    assert duration_signal(True) is None
    assert duration_signal(-1) is None
    assert str(local_media_path("file:/C:/media/a%20b.mp4")).endswith("a b.mp4")
    for value in ("https://example.com/a.mp4", "file://server/a.mp4", "relative.mp4"):
        with pytest.raises(ValueError):
            local_media_path(value)


@pytest.mark.parametrize("fraction", [0.0, 0.6, 1.0])
def test_synthetic_video_samples(
    tmp_path: Path, monkeypatch: Any, fraction: float
) -> None:
    calls = []

    def decode(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        pixel = 0 if len(calls) <= int(10 * fraction) else 128
        return subprocess.CompletedProcess(command, 0, bytes([pixel]) * 1024, b"")

    monkeypatch.setattr("kestrel.media_detectors.subprocess.run", decode)
    result = video_measurements(
        tmp_path / "synthetic.mp4", 10, "ffmpeg", DetectorConfig(max_frames=10)
    )
    assert result["sampled_frame_count"] == 10
    assert result["black_frame_ratio"] == fraction
    assert len(sample_times(10000, DetectorConfig())) == 24
    assert len(sample_times(0.1, DetectorConfig())) == 1
    assert all("-ss" in c and "-frames:v" in c for c in calls)


@pytest.mark.parametrize("amplitude", [0, 1, 3000, 30000])
def test_audio_tones(amplitude: int) -> None:
    accumulator = AudioAccumulator(DetectorConfig())
    values = [
        int(amplitude * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)
    ]
    for i in range(0, len(values), 1600):
        accumulator.add(array.array("h", values[i : i + 1600]).tobytes())
    result = accumulator.result()
    assert result["integrated_lufs"] is None
    assert result["window_count"] == 10
    if amplitude == 0:
        assert result["silent"] and result["rms_dbfs"] is None
    elif amplitude > 1:
        assert result["rms_dbfs"] == pytest.approx(
            20 * math.log10(amplitude / 32768 / math.sqrt(2)), abs=0.02
        )


def test_short_spike() -> None:
    stats = AudioAccumulator(DetectorConfig())
    for i in range(100):
        stats.add(array.array("h", [30000 if i == 40 else 3000] * 1600).tobytes())
    result = stats.result()
    assert result["high_peak_count"] == 1 and result["high_peak_ratio"] == 0.01
    assert result["peak_dbfs"] > -3


def test_source_failures(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "media.mp4"
    path.write_bytes(b"synthetic")
    assert (
        analyze_source(str(path), 10000000, True, None, DetectorConfig())[
            "analysis_status"
        ]
        == "decode_dependency_missing"
    )
    assert (
        analyze_source(
            str(tmp_path / "missing"), 10000000, True, None, DetectorConfig()
        )["analysis_status"]
        == "missing_media"
    )
    monkeypatch.setattr(
        "kestrel.media_analysis.video_measurements",
        lambda *a: {"sampled_frame_count": 1, "black_frame_ratio": 0, "mean_luma": 0.5},
    )

    def fail(*args: Any) -> Any:
        raise ValueError("bad audio")

    monkeypatch.setattr("kestrel.media_analysis.audio_measurements", fail)
    result = analyze_source(str(path), 10000000, True, "ffmpeg", DetectorConfig())
    assert result["analysis_status"] == "partial" and result["signals"]["video"]
    result = analyze_source(str(path), 10000000, False, "ffmpeg", DetectorConfig())
    assert (
        result["analysis_status"] == "ok"
        and result["signals"]["audio"]["available"] is False
    )


def test_pipeline(tmp_path: Path, monkeypatch: Any) -> None:
    source = dataset_fixture(tmp_path)

    def change(d: Any) -> None:
        items = d["ProjectFolder/Medias/medias_info.json"]["media_items"]
        for i in range(3):
            file = tmp_path / f"{i}.mp4"
            if i != 2:
                file.write_bytes(b"synthetic")
            items[f"m{i}"]["download_url"] = str(file)

    rewrite_fixture(source, change)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr("kestrel.media_analysis.shutil.which", lambda x: "ffmpeg")
    monkeypatch.setattr(
        "kestrel.media_analysis.video_measurements",
        lambda path, *a: dict(
            sampled_frame_count=10,
            black_frame_ratio=1 if path.stem == "1" else 0,
            mean_luma=0.2,
        ),
    )
    monkeypatch.setattr(
        "kestrel.media_analysis.audio_measurements",
        lambda *a: dict(
            available=True,
            integrated_lufs=None,
            rms_dbfs=-20,
            peak_dbfs=-6,
            high_peak_count=0,
            high_peak_ratio=0,
        ),
    )
    output = tmp_path / "analysis.json"
    report = media_analyze(source, output, "rules/default_rules_v1.json")
    data = json.loads(output.read_text())
    assert (
        report["source_count"] == 3
        and report["missing_count"] == 1
        and report["analyzed_count"] == 2
    )
    assert data["sources"][0]["actions"][0]["rule_id"] == "drop_short_clip"
    assert data["sources"][1]["actions"][0]["rule_id"] == "drop_mostly_black"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError):
        media_analyze(source, source)


def test_cli_analysis_only(tmp_path: Path) -> None:
    source = dataset_fixture(tmp_path)
    result = subprocess.run(
        [
            "kestrel",
            "media-analyze",
            str(source),
            str(tmp_path / "cli.json"),
            "--ffmpeg",
            "missing-decoder",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "cli.json").read_text())
    assert all(not s["actions"] for s in data["sources"])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="External ffmpeg unavailable"
)
def test_real_decoder_generated_media(tmp_path: Path) -> None:
    executable = shutil.which("ffmpeg")
    assert executable
    path = tmp_path / "generated.mkv"
    subprocess.run(
        [
            executable,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    config = DetectorConfig(max_frames=2)
    assert video_measurements(path, 1, executable, config)["black_frame_ratio"] == 1
    assert audio_measurements(path, executable, config)["rms_dbfs"] is not None
