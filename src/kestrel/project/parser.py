"""Discover structured documents in ZIP containers without extracting files."""

import base64
import json
import math
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from kestrel.formats.inspect import (
    MAX_ARCHIVE_SIZE,
    parse_json,
    read_entry,
    safe_entry,
    zip_entries,
)
from kestrel.project.models import (
    COLOR_TAG_PALETTE_ORDER,
    Clip,
    Identifier,
    Project,
    Raw,
    RawObject,
    Resource,
    Timeline,
    TimeValue,
    Track,
    Transition,
)


class ProjectParseError(ValueError):
    """The container is invalid or has no supported structured documents."""


def object_value(value: Any, location: str) -> Raw:
    if not isinstance(value, dict):
        raise ProjectParseError(f"{location}: expected an object")
    return value


def objects(value: Any, location: str) -> list[Raw]:
    if not isinstance(value, list):
        raise ProjectParseError(f"{location}: expected an array")
    return [object_value(item, f"{location}[{i}]") for i, item in enumerate(value)]


def identifier(value: Any, location: str) -> Identifier | None:
    if value is None:
        return None
    if type(value) not in (str, int) or value == "":
        raise ProjectParseError(f"{location}: expected a string or integer identifier")
    assert isinstance(value, (str, int))
    return value


def required_id(value: Any, location: str) -> Identifier:
    result = identifier(value, location)
    if result is None:
        raise ProjectParseError(f"{location}: missing identifier")
    return result


def time_value(raw: Raw, key: str) -> TimeValue | None:
    value = raw.get(key)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ProjectParseError(f"{key}: expected a numeric time value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectParseError(f"{key}: expected a finite time value")
    assert isinstance(value, (int, float))
    return value


def decode_color_tag(raw: Raw, warnings: list[str]) -> int | None:
    """Decode a little-endian int32 tag without altering its original user data."""
    entries = raw.get("userData", [])
    label = f"Clip {raw.get('thisUId')}: color tag"
    if not isinstance(entries, list):
        warnings.append(f"{label}: invalid userData retained")
        return None
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and type(entry.get("key")) is int
        and entry["key"] == 13000
    ]
    if not matches:
        return None
    if len(matches) != 1:
        warnings.append(f"{label}: duplicate entries retained; value is ambiguous")
        return None
    entry = matches[0]
    try:
        if type(entry.get("size")) is not int or entry["size"] != 4:
            raise ValueError("expected size 4")
        if not isinstance(entry.get("data"), str):
            raise ValueError("expected Base64 text")
        data = base64.b64decode(entry["data"], validate=True)
        if len(data) != 4:
            raise ValueError("expected four decoded bytes")
    except ValueError as error:
        warnings.append(f"{label}: {error}; raw entry retained")
        return None
    value = int.from_bytes(data, byteorder="little", signed=True)
    if value not in COLOR_TAG_PALETTE_ORDER:
        warnings.append(f"{label}: unknown numeric value {value} retained")
    return value


def decode_audio_gain(raw: Raw, warnings: list[str]) -> float | None:
    """Locate VolumeGain by name across effects, retaining all original fields."""
    matches: list[tuple[str, Raw]] = []
    label = f"Clip {raw.get('thisUId')}: audio gain"
    for chain in objects(raw.get("effectChainList", []), "effectChainList"):
        for effect in objects(chain.get("effectList", []), "effectList"):
            identity = str(
                effect.get("id") or effect.get("display") or "unnamed effect"
            )
            parameters = effect.get("paramList", [])
            if not isinstance(parameters, list):
                warnings.append(
                    f"{label}: invalid paramList in {identity}; raw retained"
                )
                continue
            for parameter in parameters:
                if (
                    isinstance(parameter, dict)
                    and parameter.get("name") == "VolumeGain"
                ):
                    matches.append((identity, parameter))
    if not matches:
        return None
    if len(matches) != 1:
        identities = ", ".join(identity for identity, _ in matches)
        warnings.append(f"{label}: ambiguous VolumeGain in {identities}; raw retained")
        return None
    identity, parameter = matches[0]
    fx = parameter.get("fxParam")
    try:
        if (
            not isinstance(fx, dict)
            or type(fx.get("paramType")) is not int
            or fx["paramType"] != 2
        ):
            raise ValueError("expected fxParam.paramType 2")
        value = fx.get("unValue")
        if type(value) not in (int, float):
            raise ValueError("expected numeric unValue")
        assert isinstance(value, (int, float))
        gain = float(value)
        if not math.isfinite(gain):
            raise ValueError("expected finite unValue")
    except (ValueError, OverflowError) as error:
        warnings.append(f"{label} in {identity}: {error}; raw retained")
        return None
    return gain


def parse_clip(raw: Raw, warnings: list[str]) -> Clip:
    clip_id = required_id(raw.get("thisUId"), "clip.thisUId")
    transitions = []
    for position in ("preTransition", "postTransition"):
        if raw.get(position) is not None:
            item = object_value(raw[position], position)
            transitions.append(
                Transition(
                    identifier(item.get("thisUId"), position),
                    position,
                    identifier(item.get("type"), f"{position}.type"),
                    time_value(item, "tlBegin"),
                    time_value(item, "tlEnd"),
                    item,
                )
            )
    volume: Raw = {}
    for key in ("volume", "volumeKeyframe", "audioDuckingframe"):
        if key in raw:
            volume[key] = raw[key]
            value = raw[key]
            if isinstance(value, dict) and isinstance(value.get("parameter"), str):
                try:
                    volume[f"{key}_decoded"] = parse_json(value["parameter"].encode())
                except (ValueError, RecursionError):
                    warnings.append(f"Clip {clip_id}: invalid {key} parameter retained")
    effects = []
    for chain in objects(raw.get("effectChainList", []), "effectChainList"):
        for effect in objects(chain.get("effectList", []), "effectList"):
            effect_id = effect.get("id", "")
            if isinstance(effect_id, str) and effect_id.rsplit("/", 1)[-1] in {
                "volume",
                "clip_volume",
                "fade",
                "ducking",
            }:
                effects.append(effect)
    if effects:
        volume["effects"] = effects
    clip = Clip(
        clip_id,
        identifier(raw.get("type"), "clip.type"),
        identifier(raw.get("sourceUuid"), "clip.sourceUuid"),
        time_value(raw, "inPoint"),
        time_value(raw, "outPoint"),
        time_value(raw, "tlBegin"),
        time_value(raw, "tlEnd"),
        identifier(raw.get("timelineId"), "clip.timelineId"),
        volume,
        transitions,
        raw,
        color_tag=decode_color_tag(raw, warnings),
        audio_gain_db=decode_audio_gain(raw, warnings),
    )
    for begin, end, label in (
        (clip.in_point, clip.out_point, "source range"),
        (clip.begin, clip.end, "timeline range"),
    ):
        if begin is None or end is None:
            warnings.append(f"Clip {clip.id}: incomplete {label}")
        elif end < begin:
            raise ProjectParseError(f"Clip {clip.id}: reversed {label}")
    return clip


def build_project(entries: dict[str, bytes]) -> Project:
    documents: dict[str, Any] = {}
    warnings: list[str] = []
    for name, data in entries.items():
        if data.lstrip(b"\xef\xbb\xbf \r\n\t").startswith((b"{", b"[")):
            repeated = False

            def pairs(items: list[tuple[str, Any]]) -> RawObject:
                nonlocal repeated
                result = RawObject(items)
                repeated |= len(items) != len(result)
                return result

            def constant(value: str) -> Any:
                raise ValueError(f"Invalid JSON constant: {value}")

            def number(value: str) -> float:
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError("JSON number exceeds finite range")
                return result

            try:
                documents[name] = json.loads(
                    data,
                    object_pairs_hook=pairs,
                    parse_constant=constant,
                    parse_float=number,
                )
            except (ValueError, RecursionError) as error:
                raise ProjectParseError(
                    f"Invalid structured document: {name}"
                ) from error
            if repeated:
                doc = documents[name]
                if isinstance(doc, dict) and (
                    "timelineInfos" in doc
                    or "project_guid" in doc
                    or "project_timeline_duration" in doc
                ):
                    raise ProjectParseError(f"Ambiguous repeated keys: {name}")
                warnings.append(f"Repeated keys retained as raw object pairs: {name}")
    structured = {
        name: doc
        for name, doc in documents.items()
        if isinstance(doc, dict) and "timelineInfos" in doc
    }
    if not structured:
        raise ProjectParseError("No supported timeline documents found")
    metadata_docs = {
        name: doc
        for name, doc in documents.items()
        if isinstance(doc, dict)
        and ("project_guid" in doc or "project_timeline_duration" in doc)
    }
    if len(metadata_docs) > 1:
        raise ProjectParseError("Multiple project metadata documents are ambiguous")
    metadata = next(iter(metadata_docs.values()), {})
    resources: list[Resource] = []
    timelines: list[Timeline] = []
    resource_map: dict[tuple[str, Identifier], Resource] = {}
    timeline_map: dict[tuple[str, Identifier], Timeline] = {}
    active_candidates: list[Timeline] = []
    for name, doc in structured.items():
        for raw in objects(doc.get("resources", []), "resources"):
            resource_id = required_id(raw.get("sourceUuid"), "resource.sourceUuid")
            if (name, resource_id) in resource_map:
                raise ProjectParseError(f"Duplicate resource identifier: {resource_id}")
            filename = raw.get("filename")
            if filename is not None and not isinstance(filename, str):
                raise ProjectParseError("resource.filename: expected text")
            resource = Resource(
                resource_id, name, filename, time_value(raw, "mediaLength"), raw
            )
            resources.append(resource)
            resource_map[name, resource_id] = resource
        for raw in objects(doc["timelineInfos"], "timelineInfos"):
            timeline_id = required_id(raw.get("timelineId"), "timeline.timelineId")
            if (name, timeline_id) in timeline_map:
                raise ProjectParseError(f"Duplicate timeline identifier: {timeline_id}")
            tracks = []
            track_ids: set[Identifier] = set()
            clip_ids: set[Identifier] = set()
            for track_raw in objects(raw.get("trackInfos", []), "trackInfos"):
                track_id = required_id(track_raw.get("uuid"), "track.uuid")
                if track_id in track_ids:
                    raise ProjectParseError(f"Duplicate track identifier: {track_id}")
                track_ids.add(track_id)
                clips = [
                    parse_clip(c, warnings)
                    for c in objects(track_raw.get("clipList", []), "clipList")
                ]
                for clip in clips:
                    if clip.id in clip_ids:
                        raise ProjectParseError(f"Duplicate clip identifier: {clip.id}")
                    clip_ids.add(clip.id)
                tracks.append(
                    Track(
                        track_id,
                        identifier(track_raw.get("trackType"), "trackType"),
                        clips,
                        track_raw,
                    )
                )
            timeline = Timeline(timeline_id, name, tracks, raw)
            timelines.append(timeline)
            timeline_map[name, timeline_id] = timeline
            if timeline_id == identifier(
                doc.get("currentTimelineId"), "currentTimelineId"
            ):
                active_candidates.append(timeline)
    for timeline in timelines:
        for track in timeline.tracks:
            for clip in track.clips:
                if clip.source_id is not None:
                    clip.resource = resource_map.get(
                        (timeline.document, clip.source_id)
                    )
                    if clip.resource is None:
                        warnings.append(
                            f"Clip {clip.id}: unresolved source {clip.source_id}"
                        )
                if clip.nested_timeline_id is not None:
                    clip.nested_timeline = timeline_map.get(
                        (timeline.document, clip.nested_timeline_id)
                    )
                    if clip.nested_timeline is None:
                        warnings.append(f"Clip {clip.id}: unresolved nested timeline")
    primary_id = metadata.get("timeline_mediaId")
    if primary_id is not None:
        active_candidates = [
            t
            for t in active_candidates
            if PurePosixPath(t.document).parent.name == primary_id
        ]
    active = active_candidates[0] if len(active_candidates) == 1 else None
    if active is None:
        warnings.append(
            "Active timeline is absent or ambiguous; total duration is unknown"
        )
    return Project(metadata, resources, timelines, active, documents, entries, warnings)


def parse_project(path: str | Path) -> Project:
    """Read a bounded ZIP snapshot; never follow source paths or nested references."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = zip_entries(archive)
            if sum(info.file_size for info in infos) > MAX_ARCHIVE_SIZE:
                raise ProjectParseError(
                    "Container exceeds total uncompressed size limit"
                )
            entries = {}
            total = 0
            for info in infos:
                if not safe_entry(info.filename):
                    raise ProjectParseError("Unsafe container entry path")
                data = read_entry(archive, info)
                total += len(data)
                if total > MAX_ARCHIVE_SIZE:
                    raise ProjectParseError(
                        "Container exceeds total uncompressed size limit"
                    )
                entries[info.filename] = data
        return build_project(entries)
    except (
        ValueError,
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        zlib.error,
        OverflowError,
    ) as error:
        raise ProjectParseError(str(error)) from error
