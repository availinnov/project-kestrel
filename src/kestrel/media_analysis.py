"""Source measurements and proposed actions without project modification."""

import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kestrel.dataset import read_dataset_project
from kestrel.media_detectors import (
    DetectorConfig,
    audio_measurements,
    duration_signal,
    empty_audio,
    local_media_path,
    video_measurements,
)
from kestrel.project.models import RawObject
from kestrel.project.parser import ProjectParseError
from kestrel.rules import (
    action_statistics,
    evaluate_source_rules,
    lookup,
    parse_rules,
)


def analyze_source(
    filename: str,
    ticks: Any,
    has_audio: bool,
    executable: str | None,
    config: DetectorConfig,
) -> dict[str, Any]:
    duration = duration_signal(ticks)
    signals: dict[str, Any] = dict(
        duration_seconds=duration,
        video=None,
        audio=empty_audio(None if has_audio else False),
    )
    result: dict[str, Any] = dict(signals=signals, analysis_status="ok")
    try:
        path = local_media_path(filename)
    except ValueError as error:
        return dict(result, analysis_status="missing_media", error=str(error))
    if not path.is_file():
        return dict(
            result,
            analysis_status="missing_media",
            error="Source media file is unavailable",
        )
    if executable is None:
        return dict(
            result,
            analysis_status="decode_dependency_missing",
            error="ffmpeg executable was not found; provide --ffmpeg or add it to PATH",
        )
    failures = []
    if duration is None:
        failures.append("video: positive metadata duration unavailable")
    else:
        try:
            signals["video"] = video_measurements(path, duration, executable, config)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append(f"video: {type(error).__name__}: decoding failed")
    if has_audio:
        try:
            signals["audio"] = audio_measurements(path, executable, config)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append(f"audio: {type(error).__name__}: decoding failed")
    if failures:
        result["error"] = "; ".join(failures)
        result["analysis_status"] = (
            "partial"
            if signals["video"] is not None or signals["audio"]["available"] is True
            else "video_decode_failed"
        )
    return result


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)

    def percentile(p: float) -> float | None:
        if not values:
            return None
        location = (len(values) - 1) * p
        index = int(location)
        return values[index] + (
            values[min(index + 1, len(values) - 1)] - values[index]
        ) * (location - index)

    return dict(
        count=len(values),
        min=values[0] if values else None,
        p10=percentile(0.1),
        p50=percentile(0.5),
        median=percentile(0.5),
        p90=percentile(0.9),
        p95=percentile(0.95),
        p99=percentile(0.99),
        max=values[-1] if values else None,
    )


def media_analyze(
    source: str | Path,
    output: str | Path,
    rules_path: str | Path | None = None,
    *,
    ffmpeg: str = "ffmpeg",
    config: DetectorConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or DetectorConfig()
    source, output = Path(source), Path(output)
    if output.exists() or output.resolve() == source.resolve():
        raise ProjectParseError(
            "Analysis output must be a new file distinct from input"
        )
    try:
        rules = (
            parse_rules(Path(rules_path).read_text(encoding="utf-8-sig"))
            if rules_path
            else []
        )
        project, _ = read_dataset_project(source)
        catalog = project.raw_documents["ProjectFolder/Medias/medias_info.json"][
            "media_items"
        ]
        if isinstance(catalog, RawObject) and len(catalog.pairs) != len(catalog):
            raise ValueError("Duplicate catalog IDs")
        executable = shutil.which(ffmpeg)
        sources = []
        for identifier, media in sorted(catalog.items()):
            if media.get("media_type") != 8:
                continue
            if media.get("id") != identifier:
                raise ValueError("Catalog identity mismatch")
            metadata = project.raw_documents.get(
                f"ProjectFolder/Medias/{identifier}/media.json", {}
            )
            info = metadata.get("sourceInfo", {})
            if not info.get("vidStreamInfos"):
                continue
            filename = media.get("download_url")
            if not isinstance(filename, str):
                continue
            ticks = info.get("basicInfo", {}).get(
                "mediaLength", media.get("media_length")
            )
            analysis = analyze_source(
                filename, ticks, bool(info.get("audStreamInfos")), executable, config
            )
            sources.append(
                dict(
                    catalog_id=identifier,
                    filename=filename,
                    source_duration_ticks=ticks,
                    **analysis,
                    actions=[],
                )
            )
        evaluate_source_rules(sources, rules)
        statuses = Counter(s["analysis_status"] for s in sources)
        measured = sum(
            s["signals"]["video"] is not None
            or s["signals"]["audio"]["available"] is True
            for s in sources
        )
        summary = dict(
            source_count=len(sources),
            analyzed_count=measured,
            missing_count=statuses["missing_media"],
            failed_count=sum(
                v for k, v in statuses.items() if k not in ("ok", "missing_media")
            ),
            status_counts=dict(statuses),
        )
        distributions = {}
        for path in (
            "duration_seconds",
            "video.black_frame_ratio",
            "audio.rms_dbfs",
            "audio.integrated_lufs",
            "audio.peak_dbfs",
        ):
            values = [lookup(s["signals"], path) for s in sources]
            distributions[path] = distribution(
                [v for v in values if type(v) in (int, float)]
            )
        counts = action_statistics(sources)["rule_match_counts"]
        black = [lookup(s["signals"], "video.black_frame_ratio") for s in sources]
        duration_values = [s["signals"]["duration_seconds"] for s in sources]
        report = dict(
            output=str(output),
            **summary,
            total_runtime_seconds=time.perf_counter() - started,
            duration_under_2s_count=sum(
                v is not None and v < 2 for v in duration_values
            ),
            long_clip_count=sum(v is not None and v >= 20 for v in duration_values),
            black_ratio_counts={
                str(t): sum(v is not None and v > t for v in black)
                for t in (0.25, 0.5, 0.75)
            },
            audio_analyzed_count=sum(
                s["signals"]["audio"]["available"] is True for s in sources
            ),
            proposed_quiet_normalization_count=counts.get(
                "normalize_quiet_audio", 0
            ),
            proposed_loud_normalization_count=counts.get(
                "normalize_loud_audio", 0
            ),
            proposed_short_peak_review_count=counts.get("mark_audio_peaks", 0),
            rule_match_counts=dict(counts),
            distributions=distributions,
            decoder_available=executable is not None,
        )
        dataset = dict(
            version=1,
            project=dict(source_project=str(source), **summary),
            detector_config=asdict(config),
            measurement_notes=dict(
                integrated_lufs="Unavailable in v1; RMS is not LUFS",
                audio=(
                    "Mono 16kHz sample peak and RMS; peaks may differ "
                    "from native multichannel true peak"
                ),
                high_peak=(
                    "Number and fraction of fixed windows whose sample peak "
                    "reaches configured dBFS threshold"
                ),
            ),
            sources=sources,
            summary=report,
        )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(dataset, stream, indent=2, ensure_ascii=False, allow_nan=False)
        report["output_bytes"] = output.stat().st_size
        return report
    except (ValueError, KeyError, TypeError) as error:
        raise ProjectParseError(str(error)) from error
