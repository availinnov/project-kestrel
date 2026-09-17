"""Read-only extraction of source usage and retained timeline fragments."""

import base64
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kestrel.formats.inspect import parse_json, safe_entry
from kestrel.project.models import Clip, RawObject
from kestrel.project.parser import ProjectParseError, build_project


def read_dataset_project(path: Path) -> tuple[Any, int]:
    """Read only catalog/source metadata and the main timeline document."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = {i.filename for i in infos}
        if len(names) != len(infos) or any(not safe_entry(n) for n in names):
            raise ValueError("Duplicate or unsafe archive entry")
        entries: dict[str, bytes] = {}
        total = 0

        def read(name: str) -> bytes:
            nonlocal total
            info = archive.getinfo(name)
            limit = 128 * 1024 * 1024
            if info.file_size > limit or total + info.file_size > 256 * 1024 * 1024:
                raise ValueError("Selected JSON exceeds dataset memory limit")
            with archive.open(info) as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Selected JSON exceeds dataset entry limit")
            total += len(data)
            entries[name] = data
            return data

        info_name = "ProjectFolder/project_info.json"
        info = parse_json(read(info_name))
        main = info.get("timeline_mediaId")
        if not isinstance(main, str) or not main:
            raise ValueError("Main timeline media identity is missing")
        prefix = f"ProjectFolder/Medias/{main}/"
        read("ProjectFolder/Medias/medias_info.json")
        timeline_name = prefix + "timeline.wesproj"
        timeline_data = parse_json(read(timeline_name))
        # Legacy catalog-linked clips have no timeline Resource table.
        if timeline_data.get("resources") is None:
            timeline_data["resources"] = []
            entries[timeline_name] = json.dumps(timeline_data).encode()
        del timeline_data
        if prefix + "extra.json" in names:
            read(prefix + "extra.json")
        for name in sorted(names):
            if name.startswith("ProjectFolder/Medias/") and name.endswith(
                "/media.json"
            ):
                read(name)
    project = build_project(entries)
    project.raw_entries.clear()
    ignored = sum(
        not i.is_dir() and not i.filename.endswith((".json", ".wesproj")) for i in infos
    )
    return project, ignored


def user_text(clip: Clip, key: int) -> str | None:
    entries = [
        e
        for e in clip.raw.get("userData", [])
        if isinstance(e, dict) and e.get("key") == key
    ]
    if not entries:
        return None
    if len(entries) != 1:
        raise ValueError(f"Clip {clip.id}: ambiguous userData key {key}")
    entry = entries[0]
    data = base64.b64decode(entry["data"], validate=True)
    if len(data) != entry.get("size"):
        raise ValueError(f"Clip {clip.id}: invalid userData key {key} size")
    return data.rstrip(b"\0").decode("utf-8")


def effects(clip: Clip) -> tuple[Any, list[dict[str, Any]]]:
    adjustments: dict[str, Any] = {}
    unknown = []
    for chain in clip.raw.get("effectChainList", []):
        for effect in chain.get("effectList", []):
            params = effect.get("paramList", [])
            if not isinstance(params, list):
                params = []
            recognized = []
            for p in params:
                if not isinstance(p, dict):
                    continue
                name = p.get("name")
                fx = p.get("fxParam", {})
                value = fx.get("unValue") if isinstance(fx, dict) else None
                if (
                    name
                    in ("Brightness", "Contrast", "Saturation", "Temperature", "Tint")
                    and fx.get("paramType") == 2
                    and type(value) in (int, float)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and effect.get("enable", True) is True
                ):
                    recognized.append((name.lower(), value))
            # Duplicate controls cannot safely be collapsed into one value.
            if (
                recognized
                and len(recognized) == len(params)
                and all(k not in adjustments for k, _ in recognized)
                and len({k for k, _ in recognized}) == len(recognized)
            ):
                adjustments.update(recognized)
            else:
                unknown.append(
                    dict(
                        effect_id=effect.get("id", "<unknown>"),
                        param_names=[
                            p.get("name")
                            for p in params
                            if isinstance(p, dict) and isinstance(p.get("name"), str)
                        ],
                    )
                )
    return adjustments or None, unknown


def extract_dataset(source: str | Path, output: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ProjectParseError("Dataset output must be a new file distinct from input")
    try:
        project, ignored = read_dataset_project(source)
        timeline = project.active_timeline
        if timeline is None:
            raise ValueError("Active timeline is absent or ambiguous")
        catalog = project.raw_documents["ProjectFolder/Medias/medias_info.json"][
            "media_items"
        ]
        if isinstance(catalog, RawObject) and len(catalog.pairs) != len(catalog):
            raise ValueError("Duplicate catalog ID")
        sources: dict[str, Any] = {}
        excluded: Counter[str] = Counter()
        paths: dict[str, list[str]] = defaultdict(list)
        for identifier, media in catalog.items():
            if media.get("id") != identifier:
                raise ValueError(f"Catalog key/id mismatch: {identifier}")
            path = media.get("download_url")
            if (
                media.get("media_type") != 8
                or not isinstance(path, str)
                or not path
                or "://" in path
            ):
                excluded["catalog_nonordinary"] += 1
                continue
            metadata = project.raw_documents.get(
                f"ProjectFolder/Medias/{identifier}/media.json", {}
            )
            info = metadata.get("sourceInfo", {})
            videos = info.get("vidStreamInfos", [])
            if not videos:
                excluded["catalog_no_video_metadata"] += 1
                continue
            video = videos[0]
            basic = info.get("basicInfo", {})
            capture = basic.get("createDate")
            if type(capture) is not int or capture <= 0:
                capture = None
            sources[identifier] = dict(
                catalog_id=identifier,
                filename=path,
                display_name=media.get("name"),
                capture_time=capture,
                capture_time_available=capture is not None,
                source_duration_ticks=basic.get(
                    "mediaLength", media.get("media_length")
                ),
                video_codec=video.get("fourCC"),
                width=video.get("width"),
                height=video.get("height"),
                fps=video.get("frameRate"),
                has_audio=bool(info.get("audStreamInfos")),
                fragments=[],
            )
            paths[path].append(identifier)
        audio: dict[tuple[Any, ...], list[Clip]] = defaultdict(list)

        def geometry(c: Clip) -> tuple[Any, ...]:
            return (
                user_text(c, 10) or c.source_id,
                c.in_point,
                c.out_point,
                c.begin,
                c.end,
            )

        for track in timeline.tracks:
            for clip in track.clips:
                if clip.type == 2 and clip.nested_timeline_id is None:
                    audio[geometry(clip)].append(clip)
        fragments = []
        clip_map = {}
        unparsed: Counter[str] = Counter()
        for track_index, track in enumerate(timeline.tracks):
            for clip in track.clips:
                if clip.type != 1 or clip.nested_timeline_id is not None:
                    excluded[f"clip_type_{clip.type}"] += 1
                    continue
                identifier = user_text(clip, 10)
                if identifier is None:
                    path = (
                        clip.resource.filename
                        if clip.resource
                        else clip.raw.get("filename", "")
                    ) or ""
                    matches = paths.get(path.removeprefix("file:/"), [])
                    if len(matches) != 1:
                        raise ValueError(f"Clip {clip.id}: unresolved catalog linkage")
                    identifier = matches[0]
                if identifier not in sources:
                    if identifier in catalog:
                        excluded["clip_nonordinary_catalog"] += 1
                        continue
                    raise ValueError(f"Clip {clip.id}: unknown catalog ID {identifier}")
                values = (clip.in_point, clip.out_point, clip.begin, clip.end)
                if any(type(v) not in (int, float) for v in values):
                    raise ValueError(f"Clip {clip.id}: incomplete geometry")
                start, end, begin, finish = values
                if not start < end or not begin < finish:
                    raise ValueError(f"Clip {clip.id}: non-positive geometry")
                paired = audio.get(geometry(clip), [])
                linkage = user_text(clip, 3)
                if linkage is not None:
                    try:
                        paired = [
                            a for a in paired if user_text(a, 3) in (None, linkage)
                        ]
                    except (ValueError, KeyError, TypeError):
                        paired = []
                status = (
                    "matched"
                    if len(paired) == 1
                    else "ambiguous"
                    if paired
                    else "missing"
                )
                adjustments, unknown = effects(clip)
                unparsed.update(str(e["effect_id"]) for e in unknown)
                fid = f"{timeline.id}:{clip.id}"
                fragment = dict(
                    fragment_id=fid,
                    catalog_id=identifier,
                    source_uuid=clip.source_id,
                    clip_id=clip.id,
                    track_id=track.id,
                    track_index=track_index,
                    source_in=start,
                    source_out=end,
                    source_duration=end - start,
                    timeline_begin=begin,
                    timeline_end=finish,
                    timeline_duration=finish - begin,
                    audio_gain_db=paired[0].audio_gain_db if len(paired) == 1 else None,
                    audio_pair_status=status,
                    image_adjustments=adjustments,
                    unparsed_video_effects=unknown,
                )
                fragments.append(fragment)
                clip_map[fid] = clip
                sources[identifier]["fragments"].append(fid)
        fragments.sort(
            key=lambda f: (f["timeline_begin"], f["track_index"], str(f["clip_id"]))
        )
        by_id = {f["fragment_id"]: f for f in fragments}
        if len(by_id) != len(fragments):
            raise ValueError("Duplicate fragment ID")
        for index, f in enumerate(fragments):
            f["sequence_index"] = index
        transitions = []
        for f in fragments:
            for transition in clip_map[f["fragment_id"]].transitions:
                raw = transition.raw
                if not raw:
                    continue
                neighbors = [
                    n
                    for n in fragments
                    if n["track_index"] == f["track_index"]
                    and n["fragment_id"] != f["fragment_id"]
                    and (
                        n["timeline_end"] == f["timeline_begin"]
                        if transition.position == "preTransition"
                        else n["timeline_begin"] == f["timeline_end"]
                    )
                ]
                neighbor = neighbors[0]["fragment_id"] if len(neighbors) == 1 else None
                transitions.append(
                    dict(
                        transition_id=transition.id,
                        type=raw.get("id", transition.type),
                        name=raw.get("display", raw.get("name")),
                        duration_ticks=transition.end - transition.begin
                        if transition.end is not None and transition.begin is not None
                        else raw.get("duration"),
                        timeline_begin=transition.begin,
                        timeline_end=transition.end,
                        left_fragment_id=neighbor
                        if transition.position == "preTransition"
                        else f["fragment_id"],
                        right_fragment_id=f["fragment_id"]
                        if transition.position == "preTransition"
                        else neighbor,
                    )
                )
        for record in sources.values():
            retained = [by_id[i] for i in record["fragments"]]
            total = sum(f["source_duration"] for f in retained)
            duration = record["source_duration_ticks"]
            ranges = sorted((f["source_in"], f["source_out"]) for f in retained)
            record.update(
                used=bool(retained),
                fragment_count=len(retained),
                retained_source_ticks=total,
                retained_ratio=total / duration
                if type(duration) in (int, float) and duration > 0
                else None,
                overlapping_retained_ranges=any(
                    b[0] < a[1] for a, b in zip(ranges, ranges[1:], strict=False)
                ),
            )
        used = sum(s["used"] for s in sources.values())
        dataset = dict(
            version=1,
            project=dict(
                source_project=str(source),
                timeline_id=timeline.id,
                timeline_fps=timeline.raw.get("frameRate"),
                timeline_duration=timeline.duration,
                source_count=len(sources),
                used_source_count=used,
                dropped_source_count=len(sources) - used,
            ),
            sources=list(sources.values()),
            timeline_fragments=fragments,
            transitions=transitions,
            unparsed_effects_summary=dict(unparsed),
            excluded_counts=dict(excluded),
            warnings=project.warnings,
        )
        report = dict(
            output=str(output),
            validated=True,
            imported_video_sources=len(sources),
            used_sources=used,
            dropped_sources=len(sources) - used,
            retained_fragments=len(fragments),
            sources_with_multiple_fragments=sum(
                s["fragment_count"] > 1 for s in sources.values()
            ),
            fragments_with_audio_gain=sum(
                f["audio_gain_db"] is not None for f in fragments
            ),
            fragments_with_transitions=len(
                {
                    i
                    for t in transitions
                    for i in (t["left_fragment_id"], t["right_fragment_id"])
                    if i
                }
            ),
            fragments_with_image_adjustments=sum(
                f["image_adjustments"] is not None for f in fragments
            ),
            unparsed_effect_count=sum(unparsed.values()),
            excluded_counts=dict(excluded),
            ignored_binary_entries=ignored,
            transition_type_counts=dict(Counter(str(t["type"]) for t in transitions)),
            top_unparsed_effect_ids=unparsed.most_common(20),
        )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(dataset, stream, ensure_ascii=False, indent=2, allow_nan=False)
        report.update(
            output_bytes=output.stat().st_size,
            runtime_seconds=time.perf_counter() - started,
        )
        return report
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise ProjectParseError(str(error)) from error
