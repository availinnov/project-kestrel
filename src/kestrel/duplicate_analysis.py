"""Bounded, read-only neighboring-source visual measurements; no edit policy."""

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kestrel.audio_calibration import percentile
from kestrel.dataset import read_dataset_project
from kestrel.duplicate_orb import (
    ORB_DEFINITIONS,
    OrbFrame,
    compare_orb,
    describe_orb,
)
from kestrel.media_analysis import source_basename
from kestrel.media_detectors import duration_signal, local_media_path
from kestrel.project.models import RawObject
from kestrel.project.parser import ProjectParseError
from kestrel.project.sequence import capture_time, source_order_key

WIDTH, HEIGHT = 160, 90
DEFINITIONS = {
    "sampling": "(i+0.5)/frame_samples; input seeks, one frame per position",
    "geometry": "160x90 grayscale, aspect-preserving scale and black padding",
    "hash": "64-bit dHash: 9x8 area means, left > right; Hamming 0..64, lower better",
    "image": "32x18 area means; max(0, Pearson correlation), 0..1 higher better; "
    "if either frame is flat, 1 - mean absolute difference / 255",
    "matching": "Independent nearest frame for each signal, both A->B and B->A; "
    "pool directional best matches; linearly interpolated percentiles",
    "duration_ratio": "min(duration)/max(duration), 0..1",
    "capture_time_gap_seconds": "absolute difference of catalog createDate seconds",
}


def neighbor_pairs(count: int, distance: int) -> list[tuple[int, int]]:
    """Generate only a bounded chronological neighborhood."""
    if count < 0 or distance < 1:
        raise ValueError("Count must be nonnegative and distance positive")
    return [
        (i, j) for i in range(count) for j in range(i + 1, min(count, i + distance + 1))
    ]


def area_means(frame: bytes, width: int, height: int) -> tuple[float, ...]:
    """Deterministic disjoint integer cells covering the entire grayscale frame."""
    if len(frame) != WIDTH * HEIGHT:
        raise ValueError("Invalid grayscale frame size")
    values = []
    for y in range(height):
        y0, y1 = y * HEIGHT // height, (y + 1) * HEIGHT // height
        for x in range(width):
            x0, x1 = x * WIDTH // width, (x + 1) * WIDTH // width
            values.append(
                sum(
                    sum(frame[row * WIDTH + x0 : row * WIDTH + x1])
                    for row in range(y0, y1)
                )
                / ((x1 - x0) * (y1 - y0))
            )
    return tuple(values)


@dataclass(frozen=True)
class FrameDescriptor:
    hash: int
    pixels: tuple[float, ...]
    centered: tuple[float, ...]
    norm: float
    orb: OrbFrame | None = None


def describe_frame(frame: bytes) -> FrameDescriptor:
    grid = area_means(frame, 9, 8)
    hashed = sum(
        int(grid[y * 9 + x] > grid[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8)
        for x in range(8)
    )
    pixels = area_means(frame, 32, 18)
    mean = sum(pixels) / len(pixels)
    centered = tuple(p - mean for p in pixels)
    return FrameDescriptor(
        hashed, pixels, centered, math.sqrt(sum(p * p for p in centered))
    )


def image_similarity(a: FrameDescriptor, b: FrameDescriptor) -> float:
    if a.norm < 1e-9 or b.norm < 1e-9:
        return 1 - sum(abs(x - y) for x, y in zip(a.pixels, b.pixels, strict=True)) / (
            len(a.pixels) * 255
        )
    return max(
        0.0,
        min(
            1.0,
            sum(x * y for x, y in zip(a.centered, b.centered, strict=True))
            / (a.norm * b.norm),
        ),
    )


def compare_frames(
    left: list[FrameDescriptor], right: list[FrameDescriptor]
) -> dict[str, Any]:
    if not left or not right:
        raise ValueError("Both sources require frames")
    hashes = [[float((a.hash ^ b.hash).bit_count()) for b in right] for a in left]
    images = [[image_similarity(a, b) for b in right] for a in left]
    distances = sorted(
        [min(row) for row in hashes] + [min(col) for col in zip(*hashes, strict=True)]
    )
    similarities = sorted(
        [max(row) for row in images] + [max(col) for col in zip(*images, strict=True)]
    )
    return dict(
        hash_best_match_median=percentile(distances, 0.5),
        hash_best_match_p90=percentile(distances, 0.9),
        hash_best_match_min=min(distances),
        hash_best_match_max=max(distances),
        image_similarity_median=percentile(similarities, 0.5),
        image_similarity_p10=percentile(similarities, 0.1),
        image_similarity_max=max(similarities),
    )


def sample_source(
    filename: str, duration: float | None, executable: str | None, count: int
) -> list[FrameDescriptor]:
    path = local_media_path(filename)
    if not path.is_file():
        raise FileNotFoundError("Source media unavailable")
    if executable is None:
        raise ValueError("ffmpeg unavailable")
    if duration is None:
        raise ValueError("Positive source duration unavailable")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-filter_complex_threads",
        "1",
    ]
    filters = []
    for i in range(count):
        command += [
            "-threads",
            "1",
            "-ss",
            f"{duration * (i + 0.5) / count:.9f}",
            "-t",
            "1",
            "-i",
            str(path),
        ]
        filters.append(
            f"[{i}:v:0]trim=end_frame=1,setpts=PTS-STARTPTS,"
            f"split=2[small{i}][large{i}];[small{i}]"
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,format=gray,pad=320:90:0:0[s{i}];"
            f"[large{i}]scale=320:180:force_original_aspect_ratio=decrease,"
            f"pad=320:180:(ow-iw)/2:(oh-ih)/2,setsar=1,format=gray[l{i}];"
            f"[s{i}][l{i}]vstack=inputs=2[f{i}]"
        )
    filters.append(
        "".join(f"[f{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[out]"
    )
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-frames:v",
        str(count),
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    size = 320 * 270
    if len(result.stdout) != count * size:
        raise ValueError("Incomplete sampled frames")
    frames = []
    for i in range(count):
        packed = np.frombuffer(
            result.stdout[i * size : (i + 1) * size], dtype=np.uint8
        ).reshape(270, 320)
        frame = describe_frame(packed[:90, :160].tobytes())
        frames.append(
            FrameDescriptor(
                frame.hash,
                frame.pixels,
                frame.centered,
                frame.norm,
                describe_orb(packed[90:, :]),
            )
        )
    return frames


def duplicate_analyze(
    source: str | Path,
    output: str | Path,
    *,
    max_neighbor_distance: int = 2,
    frame_samples: int = 5,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    started = time.perf_counter()
    source, output = Path(source), Path(output)
    if not 1 <= frame_samples <= 16 or max_neighbor_distance < 1:
        raise ProjectParseError(
            "frame-samples must be 1..16; neighbor distance positive"
        )
    if output.exists() or source.resolve() == output.resolve():
        raise ProjectParseError("Output must be a new file distinct from input")
    try:
        project, _ = read_dataset_project(source)
        documents = project.raw_documents
        catalog = documents["ProjectFolder/Medias/medias_info.json"]["media_items"]
        if isinstance(catalog, RawObject) and len(catalog.pairs) != len(catalog):
            raise ValueError("Duplicate catalog IDs")
        sources = []
        cache = []
        executable = shutil.which(ffmpeg)
        for identifier, media in sorted(
            catalog.items(), key=lambda item: source_order_key(documents, *item)
        ):
            if media.get("media_type") != 8:
                continue
            if media.get("id") != identifier:
                raise ValueError("Catalog identity mismatch")
            info = documents.get(
                f"ProjectFolder/Medias/{identifier}/media.json", {}
            ).get("sourceInfo", {})
            if not info.get("vidStreamInfos"):
                continue
            filename = media.get("download_url")
            if not isinstance(filename, str):
                continue
            duration = duration_signal(
                info.get("basicInfo", {}).get("mediaLength", media.get("media_length"))
            )
            item = dict(
                catalog_id=identifier,
                filename=source_basename(filename),
                source_path=filename,
                duration_seconds=duration,
                capture_time=capture_time(documents, identifier),
                analysis_status="ok",
                sampled_frame_count=0,
            )
            frames = []
            try:
                frames = sample_source(filename, duration, executable, frame_samples)
                item["sampled_frame_count"] = len(frames)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                item.update(analysis_status="unavailable", error=str(error)[:500])
            sources.append(item)
            cache.append(frames)
        pairs = []
        for i, j in neighbor_pairs(len(sources), max_neighbor_distance):
            a, b = sources[i], sources[j]
            signals = (
                compare_frames(cache[i], cache[j]) if cache[i] and cache[j] else {}
            )
            if cache[i] and cache[j]:
                left_orb = [f.orb for f in cache[i] if f.orb is not None]
                right_orb = [f.orb for f in cache[j] if f.orb is not None]
                if left_orb and right_orb:
                    signals.update(compare_orb(left_orb, right_orb))
            da, db = a["duration_seconds"], b["duration_seconds"]
            signals["duration_ratio"] = min(da, db) / max(da, db) if da and db else None
            ta, tb = a["capture_time"], b["capture_time"]
            pairs.append(
                dict(
                    left={k: a[k] for k in ("catalog_id", "filename")},
                    right={k: b[k] for k in ("catalog_id", "filename")},
                    source_distance=j - i,
                    capture_time_gap_seconds=abs(tb - ta) if ta and tb else None,
                    analysis_status="ok" if cache[i] and cache[j] else "unavailable",
                    signals=signals,
                )
            )
        summary = dict(
            source_count=len(sources),
            pair_count=len(pairs),
            measured_pair_count=sum(p["analysis_status"] == "ok" for p in pairs),
            failed_source_count=sum(not f for f in cache),
            runtime_seconds=time.perf_counter() - started,
        )
        data = dict(
            schema_version=1,
            input=str(source),
            descriptor_definitions=dict(DEFINITIONS, **ORB_DEFINITIONS),
            config=dict(
                max_neighbor_distance=max_neighbor_distance, frame_samples=frame_samples
            ),
            summary=summary,
            sources=sources,
            pairs=pairs,
        )
        output.write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return dict(output=str(output), **summary)
    except (ValueError, KeyError) as error:
        raise ProjectParseError(str(error)) from error
