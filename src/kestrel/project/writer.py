"""Conservative copy-only project edits with post-write validation."""

import copy
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from kestrel.formats.inspect import parse_json
from kestrel.project.models import Clip, Identifier, Project, Raw
from kestrel.project.parser import objects, parse_project

TICKS_PER_SECOND = 10_000_000
type JsonPath = tuple[str | int, ...]
type Edit = tuple[str, JsonPath, Any]


class ProjectWriteError(ValueError):
    """The requested edit cannot be made conservatively."""


def object_path(value: Any, target: Raw, path: JsonPath = ()) -> JsonPath:
    if value is target:
        return path
    items = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, list)
        else []
    )
    for key, child in items:
        if isinstance(child, (dict, list)):
            try:
                return object_path(child, target, (*path, key))
            except LookupError:
                pass
    raise LookupError("Object is not in document")


def json_span(text: str, path: JsonPath, start: int = 0) -> tuple[int, int]:
    """Locate one existing JSON value without rewriting its surrounding text."""
    decoder = json.JSONDecoder()
    while text[start].isspace():
        start += 1
    if not path:
        return start, decoder.raw_decode(text, start)[1]
    kind = text[start]
    if kind not in "[{":
        raise ProjectWriteError("Edit path does not identify a container")
    index = start + 1
    number = 0
    matches = []
    while True:
        while text[index].isspace():
            index += 1
        if text[index] in "]}":
            break
        if kind == "{":
            key, index = decoder.raw_decode(text, index)
            while text[index].isspace():
                index += 1
            index += 1  # Colon; the complete document was already parsed.
            while text[index].isspace():
                index += 1
        else:
            key = number
        if key == path[0]:
            matches.append(json_span(text, path[1:], index))
        _, index = decoder.raw_decode(text, index)
        number += 1
        while text[index].isspace():
            index += 1
        if text[index] == ",":
            index += 1
        else:
            break
    if len(matches) != 1:
        raise ProjectWriteError("Edit path is absent or ambiguous")
    return matches[0]


def patched_entries(project: Project, edits: list[Edit]) -> dict[str, bytes]:
    result = dict(project.raw_entries)
    for name in {edit[0] for edit in edits}:
        original = result[name]
        text = original.decode("utf-8-sig")
        replacements = []
        for document, path, value in edits:
            if document == name:
                start, end = json_span(text, path)
                replacements.append(
                    (start, end, json.dumps(value, ensure_ascii=True, allow_nan=False))
                )
        replacements.sort()
        if any(
            a[1] > b[0] for a, b in zip(replacements, replacements[1:], strict=False)
        ):
            raise ProjectWriteError("Overlapping edits are not supported")
        for start, end, value in reversed(replacements):
            text = text[:start] + value + text[end:]
        result[name] = (
            b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
        ) + text.encode("utf-8")
    return result


def write_copy(
    source: Path,
    output: Path,
    project: Project,
    edits: list[Edit],
    validate: Callable[[Project], None],
) -> dict[str, Any]:
    if source.resolve() == output.resolve() or os.path.lexists(output):
        raise ProjectWriteError("Output must be a new path; overwriting is disabled")
    entries = patched_entries(project, edits)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".kestrel-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        with (
            zipfile.ZipFile(source, "r") as original,
            zipfile.ZipFile(temporary, "w") as generated,
        ):
            if original.namelist() != list(project.raw_entries):
                raise ProjectWriteError("Input container changed during editing")
            generated.comment = original.comment
            for info in original.infolist():
                if original.read(info) != project.raw_entries[info.filename]:
                    raise ProjectWriteError("Input container changed during editing")
                generated.writestr(copy.copy(info), entries[info.filename])
        verified = parse_project(temporary)
        if verified.raw_entries != entries:
            raise ProjectWriteError("Post-write entry validation failed")
        validate(verified)
        # Same-directory hard link publishes atomically and refuses an existing target.
        os.link(temporary, output)
        return {
            "output": str(output),
            "validated": True,
            "modified_entries": sorted(
                name for name in entries if entries[name] != project.raw_entries[name]
            ),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def gain_target(clip: Clip) -> tuple[Raw, list[Raw]] | None:
    existing: list[tuple[Raw, list[Raw]]] = []
    capable: list[tuple[Raw, list[Raw]]] = []
    for chain in objects(clip.raw.get("effectChainList", []), "effectChainList"):
        for effect in objects(chain.get("effectList", []), "effectList"):
            params = objects(effect.get("paramList", []), "paramList")
            matches = [p for p in params if p.get("name") == "VolumeGain"]
            if len(matches) > 1:
                raise ProjectWriteError("Duplicate VolumeGain parameters are ambiguous")
            if matches:
                existing.append((effect, params))
            identities = [effect.get("id"), effect.get("display")]
            if any(
                isinstance(value, str)
                and value.rsplit("/", 1)[-1].casefold() in {"clip_volume", "volume"}
                for value in identities
            ):
                capable.append((effect, params))
    candidates = existing or capable
    if len(candidates) > 1:
        raise ProjectWriteError("Multiple gain-capable effects are ambiguous")
    return candidates[0] if candidates else None


def set_audio_gain(source: str | Path, output: str | Path, db: float) -> dict[str, Any]:
    if not math.isfinite(db):
        raise ProjectWriteError("Gain must be finite")
    project = parse_project(source)
    timeline = project.active_timeline
    if timeline is None:
        raise ProjectWriteError("An unambiguous active timeline is required")
    for track in timeline.tracks:
        for clip in track.clips:
            if clip.type != 2 or clip.nested_timeline_id is not None:
                continue
            target = gain_target(clip)
            if target is None:
                continue
            effect, params = target
            matches = [p for p in params if p.get("name") == "VolumeGain"]
            doc = project.raw_documents[timeline.document]
            edit: Edit
            if matches:
                parameter = matches[0]
                fx = parameter.get("fxParam", {})
                if not isinstance(fx, dict):
                    raise ProjectWriteError(
                        "Malformed fxParam cannot be replaced safely"
                    )
                replacement = {**fx, "paramType": 2, "unValue": float(db)}
                path = object_path(doc, parameter)
                edit = (
                    (timeline.document, (*path, "fxParam"), replacement)
                    if "fxParam" in parameter
                    else (
                        timeline.document,
                        path,
                        {**parameter, "fxParam": replacement},
                    )
                )
            else:
                parameter = {
                    "name": "VolumeGain",
                    "fxParam": {"paramType": 2, "unValue": float(db)},
                }
                path = object_path(doc, effect)
                edit = (
                    (timeline.document, (*path, "paramList"), [*params, parameter])
                    if "paramList" in effect
                    else (timeline.document, path, {**effect, "paramList": [parameter]})
                )

            def validate(updated: Project, clip_id: Identifier = clip.id) -> None:
                current = updated.active_timeline
                found = (
                    [c for t in current.tracks for c in t.clips if c.id == clip_id]
                    if current
                    else []
                )
                if len(found) != 1 or found[0].audio_gain_db != db:
                    raise ProjectWriteError("Post-write audio gain validation failed")

            return write_copy(Path(source), Path(output), project, [edit], validate)
    raise ProjectWriteError("No active-timeline audio clip has a gain-capable effect")


def seconds_to_ticks(value: str | float) -> int:
    try:
        seconds = Decimal(str(value))
        ticks = seconds * TICKS_PER_SECOND
        if not ticks.is_finite() or ticks < 0 or ticks != ticks.to_integral_value():
            raise ProjectWriteError(
                "Trim must be nonnegative and representable in whole ticks"
            )
        return int(ticks)
    except InvalidOperation as error:
        raise ProjectWriteError("Invalid trim value") from error


def set_trim(
    source: str | Path,
    output: str | Path,
    left_seconds: str | float,
    right_seconds: str | float,
) -> dict[str, Any]:
    left, right = seconds_to_ticks(left_seconds), seconds_to_ticks(right_seconds)
    project = parse_project(source)
    timeline = project.active_timeline
    if timeline is None or len(project.timelines) != 1:
        raise ProjectWriteError("Trim requires one active timeline")
    clips = [c for t in timeline.tracks for c in t.clips]
    if not 1 <= len(clips) <= 2 or len({c.type for c in clips}) != len(clips):
        raise ProjectWriteError("Trim requires one source clip or one matching pair")
    first = clips[0]
    signature = (
        first.source_id,
        first.in_point,
        first.out_point,
        first.begin,
        first.end,
    )
    edits: list[Edit] = []
    expected: dict[Any, tuple[int, int, int]] = {}
    for clip in clips:
        if (
            clip.type not in {1, 2}
            or clip.transitions
            or clip.nested_timeline_id is not None
            or clip.resource is None
            or (clip.source_id, clip.in_point, clip.out_point, clip.begin, clip.end)
            != signature
        ):
            raise ProjectWriteError(
                "Trim requires matching source ranges without transitions or nesting"
            )
        if any(
            type(v) is not int
            for v in (clip.in_point, clip.out_point, clip.begin, clip.end)
        ):
            raise ProjectWriteError("Trim requires integer time fields")
        assert isinstance(clip.in_point, int) and isinstance(clip.out_point, int)
        assert isinstance(clip.begin, int) and isinstance(clip.end, int)
        if clip.out_point - clip.in_point != clip.end - clip.begin:
            raise ProjectWriteError("Only unit-speed clips are supported")
        new_in, new_out = clip.in_point + left, clip.out_point - right
        new_end = clip.end - left - right
        if new_in < 0 or new_in >= new_out:
            raise ProjectWriteError("Trim must leave a positive source range")
        speed = clip.raw.get("speed")
        if not isinstance(speed, dict) or speed.get("reverse", False) is not False:
            raise ProjectWriteError("Trim requires a forward speed object")
        if "offset" not in speed or "offsetEnd" not in speed:
            raise ProjectWriteError("Speed offsets are required")
        if (
            speed["offset"] != clip.in_point / TICKS_PER_SECOND
            or speed["offsetEnd"] != clip.out_point / TICKS_PER_SECOND
        ):
            raise ProjectWriteError("Speed offsets do not match the source range")
        if speed.get("speedParam"):
            if not isinstance(speed["speedParam"], str):
                raise ProjectWriteError("Unsupported speed parameter encoding")
            try:
                parameter = parse_json(speed["speedParam"].encode())
            except (ValueError, RecursionError) as error:
                raise ProjectWriteError("Malformed speed parameters") from error
            if (
                not isinstance(parameter, dict)
                or parameter.get("keyframeSets")
                or "_totalTime" in parameter
            ):
                raise ProjectWriteError(
                    "Speed keyframes are not supported by this experiment"
                )
        path = object_path(project.raw_documents[timeline.document], clip.raw)
        for key, value in {
            "inPoint": new_in,
            "outPoint": new_out,
            "tlEnd": new_end,
        }.items():
            edits.append((timeline.document, (*path, key), value))
        edits.extend(
            [
                (
                    timeline.document,
                    (*path, "speed", "offset"),
                    new_in / TICKS_PER_SECOND,
                ),
                (
                    timeline.document,
                    (*path, "speed", "offsetEnd"),
                    new_out / TICKS_PER_SECOND,
                ),
            ]
        )
        expected[clip.id] = new_in, new_out, new_end
    duration = next(iter(expected.values()))[2]
    for name, doc in project.raw_documents.items():
        if doc is project.metadata and "project_timeline_duration" in doc:
            edits.append((name, ("project_timeline_duration",), duration))
        if isinstance(doc, dict) and isinstance(doc.get("media_items"), dict):
            main_id = project.metadata.get("timeline_mediaId")
            item = doc["media_items"].get(main_id)
            if (
                isinstance(main_id, (str, int))
                and isinstance(item, dict)
                and "media_length" in item
            ):
                edits.append((name, ("media_items", main_id, "media_length"), duration))

    def validate(updated: Project) -> None:
        current = updated.active_timeline
        if current is None or current.duration != duration:
            raise ProjectWriteError("Post-write duration validation failed")
        for track in current.tracks:
            for clip in track.clips:
                a, b, end = expected[clip.id]
                if (
                    (clip.in_point, clip.out_point, clip.end) != (a, b, end)
                    or clip.raw["speed"]["offset"] != a / TICKS_PER_SECOND
                    or clip.raw["speed"]["offsetEnd"] != b / TICKS_PER_SECOND
                ):
                    raise ProjectWriteError("Post-write trim validation failed")
        if (
            "project_timeline_duration" in updated.metadata
            and updated.metadata["project_timeline_duration"] != duration
        ):
            raise ProjectWriteError("Post-write metadata validation failed")

    return write_copy(Path(source), Path(output), project, edits, validate)
