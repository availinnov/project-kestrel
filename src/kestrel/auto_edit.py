"""Auto Edit v1 orchestration: inventory -> signals -> policy -> existing writer."""

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

from kestrel.auto_edit_policy import (
    MAX_CAPTURE_GAP_SECONDS,
    POLICY,
    RELATIVE_LOUD_THRESHOLD_DB,
    finalize_plan,
    hard_quality_plan,
)
from kestrel.duplicate_analysis import (
    capture_window_pairs,
    compare_frames,
    sample_source,
)
from kestrel.duplicate_orb import ORB_DEFINITIONS, compare_orb
from kestrel.media_analysis import analyze_source, source_basename
from kestrel.media_detectors import DetectorConfig, duration_signal
from kestrel.project.models import RawObject
from kestrel.project.parser import parse_project
from kestrel.project.sequence import (
    capture_time,
    initial_source_range,
    materialize_plan,
    resource_from_media,
    source_order_key,
    supported_template,
)
from kestrel.project.writer import ProjectWriteError


def inventory(source: Path) -> tuple[list[dict[str, Any]], Fraction]:
    project = parse_project(source)
    timeline = project.active_timeline
    if timeline is None:
        raise ProjectWriteError("An unambiguous active timeline is required")
    templates = [
        c
        for t in timeline.tracks
        for c in t.clips
        if c.type == 1 and supported_template(c)
    ]
    if len(templates) != 1 or templates[0].resource is None:
        raise ProjectWriteError(
            "One ordinary video template with a resolved resource is required"
        )
    template = templates[0].resource
    catalogs = [
        d["media_items"]
        for d in project.raw_documents.values()
        if isinstance(d, dict) and isinstance(d.get("media_items"), dict)
    ]
    if len(catalogs) != 1:
        raise ProjectWriteError("An unambiguous source catalog is required")
    catalog = catalogs[0]
    if isinstance(catalog, RawObject) and len(catalog.pairs) != len(catalog):
        raise ProjectWriteError("Duplicate catalog IDs")
    fps_raw = timeline.raw.get("frameRate")
    initial_source_range(template, fps_raw)
    assert isinstance(fps_raw, dict)
    fps = Fraction(fps_raw["num"], fps_raw["den"])
    sources = []
    seen_paths = set()
    for identifier, media in sorted(
        catalog.items(), key=lambda item: source_order_key(project.raw_documents, *item)
    ):
        if media.get("media_type") != 8:
            continue
        if media.get("id") != identifier:
            raise ProjectWriteError("Catalog identity mismatch")
        path = media.get("download_url")
        if not isinstance(path, str) or not path or path in seen_paths:
            raise ProjectWriteError("Ambiguous ordinary source path")
        matches = [
            r
            for r in project.resources
            if r.document == timeline.document and r.filename == "file:/" + path
        ]
        metadata = project.raw_documents.get(
            f"ProjectFolder/Medias/{identifier}/media.json", {}
        )
        info = metadata.get("sourceInfo", {})
        if not matches and not info.get("vidStreamInfos"):
            continue
        if len(matches) > 1:
            raise ProjectWriteError("Ambiguous source resource")
        resource = (
            matches[0]
            if matches
            else resource_from_media(template, metadata, media, "pending:" + identifier)
        )
        if (
            resource.raw.get("streamType") != 2
            or len(resource.raw.get("vidStreamInfo", [])) != 1
            or len(resource.raw.get("audStreamInfo", [])) != 1
        ):
            raise ProjectWriteError(
                "Auto Edit requires ordinary single-stream AV sources"
            )
        full_out = initial_source_range(resource, fps_raw)[1]
        ticks = resource.duration
        sources.append(
            dict(
                catalog_id=identifier,
                filename=source_basename(path),
                source_path=path,
                capture_time=capture_time(project.raw_documents, identifier),
                original_duration=duration_signal(ticks),
                source_duration_ticks=ticks,
                full_out_ticks=full_out,
                has_audio=True,
            )
        )
        seen_paths.add(path)
    if not sources:
        raise ProjectWriteError("No ordinary AV source media found")
    return sources, fps


def same_scene_measurements(
    retained: list[dict[str, Any]], executable: str | None
) -> list[dict[str, Any]]:
    candidates = capture_window_pairs(retained, MAX_CAPTURE_GAP_SECONDS)
    participating = {i for pair in candidates for i in pair}
    cache: dict[int, Any] = {}
    failures: dict[int, str] = {}
    for i in sorted(participating):
        source = retained[i]
        try:
            cache[i] = sample_source(
                source["source_path"], source["original_duration"], executable, 5
            )
        except (OSError, ValueError, subprocess.SubprocessError, cv2.error) as error:
            failures[i] = (
                f"Duplicate source sampling failed: {type(error).__name__}: {error}"
            )
            source["warnings"].append(failures[i][:500])
    relations = []
    for i, j in candidates:
        a, b = retained[i], retained[j]
        relation = dict(
            left={k: a[k] for k in ("catalog_id", "filename")},
            right={k: b[k] for k in ("catalog_id", "filename")},
            capture_time_gap_seconds=abs(b["capture_time"] - a["capture_time"]),
            orb_best_inlier_count=None,
            tagged=False,
            signals={},
        )
        try:
            if i in failures or j in failures:
                raise ValueError("Source descriptors unavailable")
            signals = compare_frames(cache[i], cache[j])
            left = [f.orb for f in cache[i] if f.orb is not None]
            right = [f.orb for f in cache[j] if f.orb is not None]
            if not left or not right:
                raise ValueError("ORB descriptors unavailable")
            signals.update(compare_orb(left, right))
            relation.update(
                signals=signals, orb_best_inlier_count=signals["orb_best_inlier_count"]
            )
        except (OSError, ValueError, subprocess.SubprocessError, cv2.error) as error:
            relation["warning"] = f"No review tag from unavailable pair: {error}"[:500]
        relations.append(relation)
    return relations


def publish_new_file(staged: Path, destination: Path) -> None:
    """Publish validated bytes exclusively, inheriting the destination folder ACL."""
    created = False
    try:
        with destination.open("xb") as target:
            created = True
            with staged.open("rb") as source:
                shutil.copyfileobj(source, target)
    except OSError:
        if created:
            destination.unlink(missing_ok=True)
        raise


def project_auto_edit(
    source: str | Path, output: str | Path, *, report_path: str | Path | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    source, output = Path(source), Path(output)
    report = (
        Path(report_path)
        if report_path is not None
        else output.with_name(output.stem + "_report.json")
    )
    if (
        output.exists()
        or report.exists()
        or len({source.resolve(), output.resolve(), report.resolve()}) != 3
    ):
        raise ProjectWriteError(
            "Output WFP/report must be new files distinct from input and each other"
        )
    timings = {}
    stage = time.perf_counter()
    try:
        sources, fps = inventory(source)
        timings["inventory"] = time.perf_counter() - stage
        executable = shutil.which("ffmpeg")
        config = DetectorConfig(
            max_frames=5, relative_loud_window_db=RELATIVE_LOUD_THRESHOLD_DB
        )
        stage = time.perf_counter()
        for item in sources:
            item["analysis"] = analyze_source(
                item["source_path"],
                item["source_duration_ticks"],
                item["has_audio"],
                executable,
                config,
            )
        timings["media_analysis"] = time.perf_counter() - stage
        stage = time.perf_counter()
        decisions = hard_quality_plan(sources)
        timings["hard_rule_planning"] = time.perf_counter() - stage
        stage = time.perf_counter()
        relations = same_scene_measurements(
            [s for s in decisions if s["keep"]], executable
        )
        timings["duplicate_analysis"] = time.perf_counter() - stage
        stage = time.perf_counter()
        plan, decisions = finalize_plan(decisions, relations, fps)
        timings["final_planning"] = time.perf_counter() - stage
        stage = time.perf_counter()
        with tempfile.TemporaryDirectory(
            prefix=".kestrel-auto-", dir=output.parent
        ) as temp:
            staged = Path(temp) / "output.wfp"
            materialization = materialize_plan(source, staged, plan)
            updated = parse_project(staged)
            assert updated.active_timeline is not None
            actual = {c.id: c for t in updated.active_timeline.tracks for c in t.clips}
            selected = {s["catalog_id"]: s for s in materialization["selected_sources"]}
            for item in decisions:
                applied = selected.get(item["catalog_id"])
                item.update(trim_start=0.0, trim_end=0.0)
                if applied is not None:
                    item.update(
                        trim_start=applied["in_point"] / 10_000_000,
                        trim_end=(applied["full_out_point"] - applied["out_point"])
                        / 10_000_000,
                        applied_gain_db=actual[applied["audio_clip_id"]].audio_gain_db,
                    )
            timings["wfp_materialization"] = time.perf_counter() - stage
            materialization["output"] = str(output)
            summary = dict(
                dropped_short=sum(
                    "duration_under_3s" in s["drop_reasons"] for s in decisions
                ),
                dropped_black=sum(
                    "black_screen" in s["drop_reasons"] for s in decisions
                ),
                tagged_same_scene=sum(s["color_tag"] == 1 for s in decisions),
                tagged_long=sum(s["color_tag"] == 2 for s in decisions),
                tagged_short_loud_event=sum(s["color_tag"] == 7 for s in decisions),
                gain_adjusted=sum(
                    s["keep"] and s["planned_gain_db"] is not None for s in decisions
                ),
                analysis_failures=sum(
                    s["analysis"]["analysis_status"] != "ok" for s in decisions
                ),
                duplicate_pair_failures=sum("warning" in p for p in relations),
            )
            result = dict(
                version=1,
                input=str(source),
                output=str(output),
                report=str(report),
                experimental_until_manual_filmora_validation=True,
                source_count=len(decisions),
                kept_count=len(selected),
                dropped_count=len(decisions) - len(selected),
                final_timeline_duration=materialization["timeline_duration"]
                / 10_000_000,
                final_timeline_duration_ticks=materialization["timeline_duration"],
                policy=POLICY,
                detector_config=asdict(config),
                orb_definitions=ORB_DEFINITIONS,
                analysis_reuse_note="Quality detector uses bounded 32x32 samples. "
                "Retained candidate sources get a separate ORB sampling pass; "
                "descriptors are cached once per source, never decoded per pair. "
                "Detector implementations are unchanged.",
                timings_seconds=timings,
                runtime_seconds=time.perf_counter() - started,
                summary=summary,
                sources=decisions,
                same_scene_relations=relations,
                materialization=materialization,
            )
            staged_report = Path(temp) / "report.json"
            staged_report.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            publish_new_file(staged, output)
            try:
                publish_new_file(staged_report, report)
            except OSError:
                output.unlink()
                raise
        return {
            k: result[k]
            for k in (
                "output",
                "report",
                "source_count",
                "kept_count",
                "dropped_count",
                "final_timeline_duration",
                "runtime_seconds",
            )
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProjectWriteError(f"Unsupported Auto Edit input: {error}") from error
