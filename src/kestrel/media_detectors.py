"""Bounded measurements with external decoding and no editing policy."""

import array
import math
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class DetectorConfig:
    max_frames: int = 24
    sample_rate: float = 1.0
    black_luma_threshold: float = 0.05
    audio_rate: int = 16000
    window_seconds: float = 0.1
    high_peak_dbfs: float = -3.0
    relative_loud_window_db: float = 12.0
    short_loud_event_max_seconds: float = 1.0
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if type(self.max_frames) is not int or not 1 <= self.max_frames <= 128:
            raise ValueError("max_frames must be between 1 and 128")
        for value in (
            self.sample_rate,
            self.window_seconds,
            self.short_loud_event_max_seconds,
            self.timeout_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Rates and timeouts must be positive and finite")
        if (
            not 0 <= self.black_luma_threshold <= 1
            or not math.isfinite(self.high_peak_dbfs)
            or self.high_peak_dbfs > 0
            or not math.isfinite(self.relative_loud_window_db)
            or self.relative_loud_window_db < 0
        ):
            raise ValueError("Invalid detector thresholds")
        if self.audio_rate != 16000:
            raise ValueError("v1 audio rate must be 16000")
        if int(self.audio_rate * self.window_seconds) < 1:
            raise ValueError("Audio window must contain at least one sample")


def local_media_path(value: str) -> Path:
    if value.startswith("file:"):
        parsed = urlsplit(value)
        if parsed.netloc:
            raise ValueError("Remote media URLs are not supported")
        value = unquote(parsed.path)
        if len(value) > 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    if "://" in value or value.startswith(("\\\\", "//")):
        raise ValueError("Remote media paths are not supported")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Source media path must be absolute")
    return path


def duration_signal(ticks: Any) -> float | None:
    return (
        ticks / 10_000_000
        if type(ticks) in (int, float) and math.isfinite(ticks) and ticks > 0
        else None
    )


def sample_times(duration: float, config: DetectorConfig) -> list[float]:
    count = min(config.max_frames, max(1, math.ceil(duration * config.sample_rate)))
    return [(i + 0.5) * duration / count for i in range(count)]


def video_measurements(
    path: Path, duration: float, executable: str, config: DetectorConfig
) -> dict[str, Any]:
    means = []
    for point in sample_times(duration, config):
        result = subprocess.run(
            [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-ss",
                str(point),
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                "scale=32:32,format=gray",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            capture_output=True,
            timeout=config.timeout_seconds,
            check=True,
        )
        if len(result.stdout) != 1024:
            raise ValueError("Video sample did not decode to one 32x32 frame")
        means.append(sum(result.stdout) / (1024 * 255))
    return dict(
        sampled_frame_count=len(means),
        black_frame_ratio=sum(v < config.black_luma_threshold for v in means)
        / len(means),
        mean_luma=sum(means) / len(means),
    )


class AudioAccumulator:
    """Streaming mono sample statistics over deterministic fixed windows."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.samples = 0
        self.energy = 0.0
        self.peak = 0.0
        self.windows = 0
        self.high = 0
        self.pending = bytearray()
        self.window_rms: list[float | None] = []
        self.window_sample_counts: list[int] = []

    def add(self, data: bytes) -> None:
        self.pending.extend(data)
        window_bytes = int(self.config.audio_rate * self.config.window_seconds) * 2
        while len(self.pending) >= window_bytes:
            self._add_window(bytes(self.pending[:window_bytes]))
            del self.pending[:window_bytes]

    def _add_window(self, data: bytes) -> None:
        values = array.array("h")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        if not values:
            return
        peak = max(abs(v) for v in values) / 32768
        energy = sum((v / 32768) ** 2 for v in values)
        rms = math.sqrt(energy / len(values))
        self.samples += len(values)
        self.energy += energy
        self.peak = max(self.peak, peak)
        self.windows += 1
        self.high += peak >= 10 ** (self.config.high_peak_dbfs / 20)
        self.window_rms.append(20 * math.log10(rms) if rms else None)
        self.window_sample_counts.append(len(values))

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        location = (len(ordered) - 1) * fraction
        index = int(location)
        return ordered[index] + (
            ordered[min(index + 1, len(ordered) - 1)] - ordered[index]
        ) * (location - index)

    def result(self) -> dict[str, Any]:
        if self.pending:
            if len(self.pending) % 2:
                raise ValueError("Audio stream ended with an incomplete PCM sample")
            self._add_window(bytes(self.pending))
            self.pending.clear()
        if not self.samples:
            raise ValueError("Audio stream decoded no samples")
        rms = math.sqrt(self.energy / self.samples)
        finite_windows = [value for value in self.window_rms if value is not None]
        median = self._percentile(finite_windows, 0.5)
        p90 = self._percentile(finite_windows, 0.9)
        p95 = self._percentile(finite_windows, 0.95)
        maximum = max(finite_windows) if finite_windows else None
        relative = [
            value is not None
            and median is not None
            and value >= median + self.config.relative_loud_window_db
            for value in self.window_rms
        ]
        events: list[int] = []
        current_samples = 0
        for is_loud, sample_count in zip(
            relative, self.window_sample_counts, strict=True
        ):
            if is_loud:
                current_samples += sample_count
            elif current_samples:
                events.append(current_samples)
                current_samples = 0
        if current_samples:
            events.append(current_samples)
        short_limit = self.config.short_loud_event_max_seconds * self.config.audio_rate
        short_events = [event for event in events if event <= short_limit]
        relative_count = sum(relative)
        return dict(
            available=True,
            integrated_lufs=None,
            loudness_method="mono_16k_pcm_rms",
            rms_dbfs=20 * math.log10(rms) if rms else None,
            peak_dbfs=20 * math.log10(self.peak) if self.peak else None,
            silent=rms == 0,
            high_peak_count=self.high,
            high_peak_ratio=self.high / self.windows,
            window_count=self.windows,
            window_rms_median_dbfs=median,
            window_rms_p90_dbfs=p90,
            window_rms_p95_dbfs=p95,
            window_rms_max_dbfs=maximum,
            max_rms_above_median_db=(
                maximum - median if maximum is not None and median is not None else None
            ),
            relative_loud_window_count=relative_count,
            relative_loud_window_ratio=relative_count / self.windows,
            relative_loud_event_count=len(events),
            max_relative_loud_event_seconds=(
                max(events) / self.config.audio_rate if events else 0.0
            ),
            short_relative_loud_event_count=len(short_events),
            short_relative_loud_event_ratio=sum(short_events) / self.samples,
        )


def empty_audio(available: bool | None) -> dict[str, Any]:
    return dict(
        available=available,
        integrated_lufs=None,
        loudness_method=None,
        rms_dbfs=None,
        peak_dbfs=None,
        silent=None,
        high_peak_count=None,
        high_peak_ratio=None,
        window_count=None,
        window_rms_median_dbfs=None,
        window_rms_p90_dbfs=None,
        window_rms_p95_dbfs=None,
        window_rms_max_dbfs=None,
        max_rms_above_median_db=None,
        relative_loud_window_count=None,
        relative_loud_window_ratio=None,
        relative_loud_event_count=None,
        max_relative_loud_event_seconds=None,
        short_relative_loud_event_count=None,
        short_relative_loud_event_ratio=None,
    )


def audio_measurements(
    path: Path, executable: str, config: DetectorConfig
) -> dict[str, Any]:
    stats = AudioAccumulator(config)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            [
                executable,
                "-nostdin",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(config.audio_rate),
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        timer = threading.Timer(config.timeout_seconds, process.kill)
        timer.start()
        try:
            assert process.stdout is not None
            with process.stdout:
                while data := process.stdout.read(
                    max(2, int(config.audio_rate * config.window_seconds) * 2)
                ):
                    stats.add(data)
            if process.wait() != 0:
                raise ValueError("Audio decoding failed or timed out")
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
    return stats.result()
