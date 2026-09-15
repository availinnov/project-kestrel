"""Conservative sequence layout from existing imported resources."""

import base64
import copy
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from kestrel.project.models import Clip, Raw, Resource
from kestrel.project.parser import objects, parse_project
from kestrel.project.writer import (
    TICKS_PER_SECOND,
    Edit,
    JsonAppend,
    ProjectWriteError,
    identity_allocator,
    object_path,
    validate_ordinary_speed_parameter,
    write_copy,
)


def half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def initial_source_range(
    resource_duration: int | Resource, timeline_fps: Any
) -> tuple[int, int]:
    if not isinstance(timeline_fps, dict) or any(
        type(timeline_fps.get(k)) is not int or timeline_fps[k] <= 0
        for k in ("num", "den")
    ):
        raise ProjectWriteError("Timeline FPS requires positive integer num/den")
    duration = (
        resource_duration.duration
        if isinstance(resource_duration, Resource)
        else resource_duration
    )
    if type(duration) is not int or duration <= 0:
        raise ProjectWriteError("Resource requires a positive integer duration")
    fps = Fraction(timeline_fps["num"], timeline_fps["den"])
    frames = half_up(Fraction(duration, TICKS_PER_SECOND) * fps)
    end = half_up(Fraction(frames * TICKS_PER_SECOND) / fps)
    if end <= 0:
        raise ProjectWriteError("Resource rounds to an empty frame range")
    return 0, end


SPEED_MD5 = "dcac65f27685081eb443af69a4267e8f"


def build_constant_speed_param(out_point_ticks: int) -> str:
    if type(out_point_ticks) is not int or out_point_ticks <= 0:
        raise ProjectWriteError("Speed span requires positive integer ticks")
    seconds = (
        f"{out_point_ticks // TICKS_PER_SECOND}."
        f"{out_point_ticks % TICKS_PER_SECOND:07d}"
    )
    return (
        '{"Version":3,"ParameterType":0,"keyframeSets":['
        '{"_time":0.0,"Interpolation":6,"_value":1.0},'
        '{"_time":' + seconds + ',"Interpolation":6,"_value":1.0}],'
        '"MD5":"' + SPEED_MD5 + '","_totalTime":' + seconds + "}"
    )


def resource_from_media(
    template: Resource, metadata: Raw, media: Raw, identifier: str
) -> Resource:
    info = metadata["sourceInfo"]
    basic = info["basicInfo"]
    if (
        metadata["file_name"] != media["download_url"]
        or basic["mediaLength"] != media["media_length"]
    ):
        raise ProjectWriteError("Imported source metadata disagrees with catalog")
    raw = copy.deepcopy(template.raw)
    for key in raw:
        if key in basic:
            raw[key] = copy.deepcopy(basic[key])
        elif key == "modifyDate":
            raw[key] = basic["createDate"]
        elif key not in (
            "sourceUuid",
            "filename",
            "vidStreamInfo",
            "audStreamInfo",
            "AIGCMeta",
            "alphaPremultiply",
        ):
            raise ProjectWriteError(f"Unresolved Resource field: {key}")
    for target, source_key in (
        ("vidStreamInfo", "vidStreamInfos"),
        ("audStreamInfo", "audStreamInfos"),
    ):
        source_streams = info[source_key]
        if len(source_streams) != 1 or len(raw[target]) != 1:
            raise ProjectWriteError("Only one video and one audio stream are supported")
        stream = raw[target][0]
        for key in stream:
            if key in source_streams[0]:
                stream[key] = copy.deepcopy(source_streams[0][key])
            elif stream[key] not in (0, 0.0, "", None, False):
                raise ProjectWriteError(f"Unresolved source stream field: {key}")
    raw.update(sourceUuid=identifier, filename="file:/" + media["download_url"])
    return Resource(
        identifier, template.document, raw["filename"], basic["mediaLength"], raw
    )


@dataclass(frozen=True)
class SequenceItem:
    resource: Resource
    source_in: int
    source_out: int
    timeline_begin: int
    timeline_end: int


def sequence_layout(resources: list[Resource], fps: Any) -> list[SequenceItem]:
    result = []
    begin = 0
    for resource in resources:
        start, end = initial_source_range(resource, fps)
        result.append(SequenceItem(resource, start, end, begin, begin + end))
        begin += end
    return result


def supported_template(clip: Clip) -> bool:
    if (
        clip.type not in (1, 2)
        or clip.resource is None
        or clip.transitions
        or clip.nested_timeline_id is not None
        or clip.raw.get("enable", True) is not True
        or clip.audio_gain_db not in (None, 0)
    ):
        return False
    if any(
        type(v) is not int
        for v in (clip.in_point, clip.out_point, clip.begin, clip.end)
    ):
        return False
    assert isinstance(clip.in_point, int) and isinstance(clip.out_point, int)
    assert isinstance(clip.begin, int) and isinstance(clip.end, int)
    if (
        clip.out_point <= clip.in_point
        or clip.end - clip.begin != clip.out_point - clip.in_point
    ):
        return False
    speed = clip.raw.get("speed")
    if not isinstance(speed, dict) or speed.get("reverse") is not False:
        return False
    if speed.get("offset") != clip.in_point / TICKS_PER_SECOND:
        return False
    # A source-bound video endpoint can differ from rounded clip geometry.
    if speed.get("offsetEnd") not in (
        clip.out_point / TICKS_PER_SECOND,
        (clip.resource.duration or 0) / TICKS_PER_SECOND,
    ):
        return False
    try:
        validate_ordinary_speed_parameter(speed, clip.out_point / TICKS_PER_SECOND)
        for chain in objects(clip.raw.get("effectChainList", []), "effectChainList"):
            for effect in objects(chain.get("effectList", []), "effectList"):
                if effect.get("id") not in {
                    "video/effect/crop-pan-zoom",
                    "video/effect/transform",
                    "audio/effect/clip_volume",
                    "audio/effect/change_channel",
                    "audio/effect/volume",
                    "audio/effect/ducking",
                    "audio/effect/fade",
                }:
                    return False
                if any(
                    p.get("name") not in ("dwValue", "EnableTransform")
                    for p in objects(effect.get("paramList", []), "paramList")
                ):
                    return False
        for field in ("volumeKeyframe", "audioDuckingframe"):
            if field in clip.raw:
                if json.loads(clip.raw[field]["parameter"]).get("keyframeSets"):
                    return False
    except (ValueError, KeyError, TypeError):
        return False
    return True


def _build_sequence(
    source: str | Path, output: str | Path, count: int
) -> dict[str, Any]:
    if type(count) is not int or count < 1:
        raise ProjectWriteError("Count must be a positive integer")
    project = parse_project(source)
    timeline = project.active_timeline
    if timeline is None:
        raise ProjectWriteError("An unambiguous active timeline is required")
    fps = timeline.raw.get("frameRate")
    catalog_docs = [
        (n, d)
        for n, d in project.raw_documents.items()
        if isinstance(d, dict) and isinstance(d.get("media_items"), dict)
    ]
    if len(catalog_docs) != 1:
        raise ProjectWriteError("An unambiguous media catalog is required")
    catalog_name, catalog = catalog_docs[0]
    candidates = [c for t in timeline.tracks for c in t.clips if supported_template(c)]
    pairs = [
        (v, a)
        for v in candidates
        if v.type == 1
        for a in candidates
        if a.type == 2
        and (v.source_id, v.in_point, v.out_point, v.begin, v.end)
        == (a.source_id, a.in_point, a.out_point, a.begin, a.end)
    ]
    if not pairs:
        raise ProjectWriteError("No supported ordinary AV template pair exists")
    video_template, audio_template = pairs[0]
    template_linkage = None
    for template in (video_template, audio_template):
        assert template.resource is not None and template.resource.filename is not None
        if template.raw.get("filename") != template.resource.filename:
            raise ProjectWriteError("Template path does not match its resource")
        entries = objects(template.raw.get("userData", []), "template userData")
        for required in (3, 10):
            if len([e for e in entries if e.get("key") == required]) != 1:
                raise ProjectWriteError(
                    "Template requires unique key3 and catalog references"
                )
        entry = next(e for e in entries if e["key"] == 3)
        decoded = base64.b64decode(entry["data"], validate=True)
        identifier = decoded.rstrip(b"\0").decode("ascii")
        if (
            len(identifier) != 38
            or len(decoded) != entry["size"]
            or decoded != identifier.encode() + bytes(entry["size"] - 38)
        ):
            raise ProjectWriteError("Unsupported template linkage payload")
        if template_linkage is not None and template_linkage != identifier:
            raise ProjectWriteError("Template AV linkage mismatch")
        template_linkage = identifier
    fresh, identities = identity_allocator(project)
    eligible: list[Resource] = []
    new_resources: list[Raw] = []
    items: dict[Any, tuple[str, Raw]] = {}
    rejected_sources = []
    seen_paths: set[str] = set()
    assert video_template.resource is not None
    for catalog_id, media in catalog["media_items"].items():
        if media.get("media_type") != 8 or media.get("id") != catalog_id:
            continue
        source_path = media.get("download_url")
        if (
            not isinstance(source_path, str)
            or not source_path
            or source_path in seen_paths
        ):
            continue
        matches = [
            r
            for r in project.resources
            if r.document == timeline.document and r.filename == "file:/" + source_path
        ]
        if len(matches) > 1:
            continue
        if matches:
            resource = matches[0]
        else:
            metadata = project.raw_documents.get(
                f"ProjectFolder/Medias/{catalog_id}/media.json"
            )
            if not isinstance(metadata, dict):
                continue
            try:
                resource = resource_from_media(
                    video_template.resource, metadata, media, "pending"
                )
            except (KeyError, TypeError, ProjectWriteError) as error:
                rejected_sources.append(f"{catalog_id}: {error}")
                continue
        video = resource.raw.get("vidStreamInfo", [])
        audio = resource.raw.get("audStreamInfo", [])
        if (
            resource.raw.get("streamType") != 2
            or len(video) != 1
            or len(audio) != 1
            or any(
                type(v) is not int
                for v in (video[0].get("vidStreamId"), audio[0].get("audStreamId"))
            )
        ):
            continue
        try:
            initial_source_range(resource, fps)
        except ProjectWriteError:
            continue
        if not matches:
            resource.id = fresh(None, "resource.sourceUuid")
            resource.raw["sourceUuid"] = resource.id
            new_resources.append(resource.raw)
        eligible.append(resource)
        items[resource.id] = (catalog_id, media)
        seen_paths.add(source_path)
        if len(eligible) == count:
            break
    if len(eligible) < count:
        raise ProjectWriteError(
            f"Requested {count} sources; only {len(eligible)} are eligible"
            + ("; " + "; ".join(rejected_sources) if rejected_sources else "")
        )
    layout = sequence_layout(eligible[:count], fps)
    duration = layout[-1].timeline_end
    destinations = []
    for kind in (1, 2):
        compatible = []
        for track in timeline.tracks:
            if track.type != kind:
                continue
            if all(
                type(c.begin) is int
                and type(c.end) is int
                and c.begin >= 0
                and c.end > c.begin
                and not (c.begin < duration and c.end > 0)
                for c in track.clips
            ):
                compatible.append(track)
        if not compatible:
            raise ProjectWriteError(
                "Existing timeline content overlaps sequence placement"
            )
        destinations.append(compatible[0])
    if timeline.duration is None or timeline.duration > duration:
        raise ProjectWriteError(
            "Preserved timeline content extends beyond the sequence end"
        )
    extra_name = timeline.document.replace("timeline.wesproj", "extra.json")
    extra = project.raw_documents.get(extra_name)
    edits: list[Edit] = []
    if new_resources:
        edits.append((timeline.document, ("resources",), JsonAppend(new_resources)))
    generated: list[list[Raw]] = [[], []]
    selected = []
    for item in layout:
        assert item.resource.filename is not None
        catalog_id, media = items[item.resource.id]
        linkage: str | None = None
        for index, template in enumerate((video_template, audio_template)):
            assert (
                template.resource is not None and template.resource.filename is not None
            )
            clip = copy.deepcopy(template.raw)
            clip.update(
                thisUId=fresh(template.id, "clip.thisUId"),
                enable=True,
                sourceUuid=item.resource.id,
                filename=item.resource.filename,
                inPoint=0,
                outPoint=item.source_out,
                tlBegin=item.timeline_begin,
                tlEnd=item.timeline_end,
                streamId=item.resource.raw[
                    "vidStreamInfo" if index == 0 else "audStreamInfo"
                ][0]["vidStreamId" if index == 0 else "audStreamId"],
            )
            clip["speed"]["speedParam"] = build_constant_speed_param(item.source_out)
            clip["speed"].update(
                offset=0.0, offsetEnd=item.source_out / TICKS_PER_SECOND, reverse=False
            )
            if index == 0:
                # Prefer the selected source's established full-range representation.
                examples = [
                    c
                    for t in timeline.tracks
                    for c in t.clips
                    if c.type == 1
                    and c.source_id == item.resource.id
                    and c.in_point == 0
                    and c.out_point == item.source_out
                    and supported_template(c)
                ]
                representation = examples[0] if examples else template
                if (
                    representation.resource is not None
                    and representation.resource.duration is not None
                    and representation.out_point is not None
                    and representation.raw["speed"]["offsetEnd"]
                    == representation.resource.duration / TICKS_PER_SECOND
                    and representation.raw["speed"]["offsetEnd"]
                    != representation.out_point / TICKS_PER_SECOND
                ):
                    assert item.resource.duration is not None
                    clip["speed"]["offsetEnd"] = (
                        item.resource.duration / TICKS_PER_SECOND
                    )
            for chain in clip.get("effectChainList", []):
                for effect in chain.get("effectList", []):
                    if "thisUId" in effect:
                        effect["thisUId"] = fresh(effect["thisUId"], "effect.thisUId")
            user = objects(clip.get("userData", []), "userData")
            if len({e.get("key") for e in user}) != len(user):
                raise ProjectWriteError("Ambiguous template userData")
            for entry in user:
                key = entry.get("key")
                if key not in (1, 2, 3, 6, 10, 50, 73, 74, 102, 103):
                    raise ProjectWriteError(
                        "Unsupported media-specific template userData"
                    )
                if key in (1, 2, 102, 103):
                    continue
                raw = base64.b64decode(entry["data"], validate=True)
                if key == 10:
                    template_items = [
                        k
                        for k, v in catalog["media_items"].items()
                        if v.get("download_url")
                        == template.resource.filename.removeprefix("file:/")
                    ]
                    if len(template_items) != 1 or raw != template_items[0].encode(
                        "ascii"
                    ):
                        raise ProjectWriteError("Template catalog reference mismatch")
                    raw = catalog_id.encode("ascii")
                elif key == 50:
                    raw = media["name"].encode("utf-8")
                elif key == 6:
                    if type(timeline.id) is not int:
                        raise ProjectWriteError("Numeric timeline identity required")
                    raw = timeline.id.to_bytes(4, "little")
                elif key == 3:
                    old = raw.rstrip(b"\0").decode("ascii")
                    if extra is None or old not in extra.get("mediaClipsMapInfo", {}):
                        raise ProjectWriteError("Template linkage metadata is missing")
                    if linkage is None:
                        linkage = "{" + fresh(old, "clip.linkage") + "}"
                        for parent in (
                            ("mediaClipsMapInfo",),
                            ("allMarkersInfo", "beatDetectInfo"),
                        ):
                            obj = extra
                            for part in parent:
                                obj = obj[part]
                            value = copy.deepcopy(obj[old])
                            if parent == ("mediaClipsMapInfo",):
                                value["mediaId"] = catalog_id
                                if value.get("subClips"):
                                    raise ProjectWriteError(
                                        "Template subclips are unsupported"
                                    )
                            edits.append(
                                (extra_name, parent, JsonAppend({linkage: value}))
                            )
                    elif (
                        extra["mediaClipsMapInfo"][old]["mediaId"]
                        != extra["mediaClipsMapInfo"][
                            base64.b64decode(
                                next(
                                    e["data"]
                                    for e in video_template.raw["userData"]
                                    if e["key"] == 3
                                )
                            )
                            .rstrip(b"\0")
                            .decode("ascii")
                        ]["mediaId"]
                    ):
                        raise ProjectWriteError("Template AV linkage mismatch")
                    raw = linkage.encode("ascii") + bytes(entry["size"] - 38)
                else:
                    data = json.loads(raw)
                    for effect in objects(
                        data.get("effectList"), "embedded effectList"
                    ):
                        if effect.get("enable") is not False:
                            raise ProjectWriteError(
                                "Enabled embedded effects unsupported"
                            )
                        for parameter in objects(effect.get("paramList"), "paramList"):
                            if parameter.get("name") != "OriginPath":
                                raise ProjectWriteError(
                                    "Unknown embedded resource parameter"
                                )
                            if parameter["fxParam"].get(
                                "unValue"
                            ) != template.resource.filename.removeprefix("file:/"):
                                raise ProjectWriteError(
                                    "Embedded template source path mismatch"
                                )
                            parameter["fxParam"]["unValue"] = (
                                item.resource.filename.removeprefix("file:/")
                            )
                    raw = json.dumps(data, separators=(",", ":")).encode()
                entry.update(data=base64.b64encode(raw).decode("ascii"), size=len(raw))
            generated[index].append(clip)
        selected.append(
            dict(
                filename=item.resource.filename,
                resource_id=catalog_id,
                catalog_id=catalog_id,
                key3=linkage,
                source_uuid=item.resource.id,
                resource_duration=item.resource.duration,
                frame_aligned_out_point=item.source_out,
                tl_begin=item.timeline_begin,
                tl_end=item.timeline_end,
                video_clip_id=generated[0][-1]["thisUId"],
                audio_clip_id=generated[1][-1]["thisUId"],
            )
        )
    for track, clips in zip(destinations, generated, strict=True):
        path = object_path(project.raw_documents[timeline.document], track.raw)
        edits.append((timeline.document, (*path, "clipList"), JsonAppend(clips)))
    # Merge insertions at the same object boundary into one patch.
    merged: dict[tuple[str, tuple[str | int, ...]], dict[str, Any]] = {}
    other = []
    for name, path, value in edits:
        if isinstance(value, JsonAppend) and isinstance(value.value, dict):
            merged.setdefault((name, path), {}).update(value.value)
        else:
            other.append((name, path, value))
    edits = other + [
        (name, path, JsonAppend(value)) for (name, path), value in merged.items()
    ]
    main_id = project.metadata.get("timeline_mediaId")
    if not isinstance(main_id, (str, int)):
        raise ProjectWriteError("Main timeline identity is missing")
    if (
        main_id not in catalog["media_items"]
        or "duration" not in catalog["media_items"][main_id]
    ):
        raise ProjectWriteError("Main timeline duration metadata is missing")
    for duration_field in ("duration", "media_length"):
        if duration_field in catalog["media_items"][main_id]:
            edits.append(
                (catalog_name, ("media_items", main_id, duration_field), duration)
            )
    metadata_name = next(
        n for n, d in project.raw_documents.items() if d is project.metadata
    )
    if "project_timeline_duration" not in project.metadata:
        raise ProjectWriteError("Project timeline duration is missing")
    edits.append((metadata_name, ("project_timeline_duration",), duration))
    expected = copy.deepcopy(project.raw_documents)
    for name, path, value in edits:
        obj = expected[name]
        for path_part in path[:-1]:
            obj = obj[path_part]
        if isinstance(value, JsonAppend):
            target = obj[path[-1]]
            if isinstance(target, list):
                target.extend(value.value)
            else:
                target.update(value.value)
        else:
            obj[path[-1]] = value

    def validate(updated: Any) -> None:
        if (
            updated.active_timeline is None
            or updated.raw_documents != expected
            or updated.active_timeline.duration != duration
        ):
            raise ProjectWriteError("Post-write sequence structure validation failed")
        if [r.raw for r in updated.resources] != [
            r.raw for r in project.resources
        ] + new_resources:
            raise ProjectWriteError("Resources changed")
        for pair in selected:
            found_pair = []
            for field in ("video_clip_id", "audio_clip_id"):
                matches = [
                    c
                    for t in updated.active_timeline.tracks
                    for c in t.clips
                    if c.id == pair[field]
                ]
                if (
                    len(matches) != 1
                    or matches[0].resource is None
                    or matches[0].source_id != pair["source_uuid"]
                ):
                    raise ProjectWriteError("Generated clip resource resolution failed")
                found_pair.append(matches[0])
            for c in found_pair:
                if (c.in_point, c.out_point, c.begin, c.end) != (
                    0,
                    pair["frame_aligned_out_point"],
                    pair["tl_begin"],
                    pair["tl_end"],
                ):
                    raise ProjectWriteError("Post-write sequence geometry mismatch")
                if initial_source_range(c.resource, fps)[1] != c.out_point:
                    raise ProjectWriteError("Post-write endpoint mismatch")
                if c.raw["speed"]["speedParam"] != build_constant_speed_param(
                    c.out_point
                ):
                    raise ProjectWriteError(
                        "Post-write constant-speed payload mismatch"
                    )
            if (
                updated.raw_documents[extra_name]["mediaClipsMapInfo"][pair["key3"]][
                    "mediaId"
                ]
                != pair["catalog_id"]
            ):
                raise ProjectWriteError("Post-write placement linkage mismatch")
        values = [i["generated"] for i in identities]
        if len(values) != len(set(values)):
            raise ProjectWriteError("Generated identities are not unique")

    report = write_copy(Path(source), Path(output), project, edits, validate)
    report.update(
        requested_count=count,
        generated_source_count=count,
        generated_video_clip_count=count,
        generated_audio_clip_count=count,
        timeline_duration=duration,
        video_track_id=destinations[0].id,
        audio_track_id=destinations[1].id,
        selected_sources=selected,
        generated_identity_fields=identities,
    )
    return report


def build_sequence(
    source: str | Path, output: str | Path, count: int
) -> dict[str, Any]:
    try:
        return _build_sequence(source, output, count)
    except ProjectWriteError:
        raise
    except (KeyError, TypeError, UnicodeError, ValueError, OverflowError) as error:
        raise ProjectWriteError(
            "Unsupported or incomplete sequence structure"
        ) from error
