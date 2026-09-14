"""Instantiate empty containers by rewriting only a validated identity graph."""

import base64
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from kestrel.project.models import Project, Raw, Timeline
from kestrel.project.parser import object_value, objects, parse_project
from kestrel.project.writer import (
    Edit,
    JsonRenameKey,
    ProjectWriteError,
    object_path,
    write_copy,
)

INFO = "ProjectFolder/project_info.json"
CATALOG = "ProjectFolder/Medias/medias_info.json"
UUID_PATTERN = re.compile(
    r"\{[0-9a-fA-F-]{36}\}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)


def uuid_value(value: Any) -> UUID:
    if not isinstance(value, str) or not UUID_PATTERN.fullmatch(value):
        raise ProjectWriteError("Expected a textual UUID")
    try:
        return UUID(value)
    except ValueError as error:
        raise ProjectWriteError("Malformed UUID") from error


def uuid_style(value: UUID, original: str) -> str:
    body = original.strip("{}")
    result = str(value)
    if body == body.upper():
        result = result.upper()
    elif body != body.lower():
        result = "".join(
            c.upper() if old.isupper() else c
            for c, old in zip(result, body, strict=True)
        )
    return "{" + result + "}" if original.startswith("{") else result


def positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProjectWriteError(f"{label} must be a positive integer")
    assert isinstance(value, int)
    return value


def identity_payload(entries: list[Raw], key: int, size: int) -> tuple[Raw, bytes]:
    candidates = [entry for entry in entries if entry.get("key") == key]
    if len(candidates) != 1:
        raise ProjectWriteError(f"Missing or ambiguous identity userData key {key}")
    entry = candidates[0]
    if (
        type(entry["key"]) is not int
        or type(entry.get("size")) is not int
        or entry["size"] != size
        or not isinstance(entry.get("data"), str)
    ):
        raise ProjectWriteError(f"Unsupported identity payload for key {key}")
    try:
        payload = base64.b64decode(entry["data"], validate=True)
    except ValueError as error:
        raise ProjectWriteError(f"Malformed Base64 payload for key {key}") from error
    if len(payload) != size:
        raise ProjectWriteError(f"Incorrect identity payload length for key {key}")
    return entry, payload


@dataclass
class EmptyTemplate:
    project: Project
    timeline: Timeline
    info: Raw
    catalog: Raw
    item: Raw
    media_id: str
    timeline_uuid: str
    project_guid: str
    directory: str
    uuid_entries: list[Raw]
    bus_uuid: str
    instance_entry: Raw
    instance_uuid: str
    token_entry: Raw
    token: str


def discover_empty_template(project: Project) -> EmptyTemplate:
    """Verify all observed identity links; ambiguity prevents any write."""
    if (
        len(project.timelines) != 1
        or project.active_timeline is None
        or project.resources
        or any(t.clips for t in project.timelines[0].tracks)
    ):
        raise ProjectWriteError(
            "Template requires one active empty timeline and no source resources"
        )
    timeline = project.active_timeline
    info = object_value(project.raw_documents.get(INFO), "project metadata")
    catalog = object_value(project.raw_documents.get(CATALOG), "media catalog")
    media_id = info.get("timeline_mediaId")
    project_guid = info.get("project_guid")
    uuid_value(media_id)
    uuid_value(project_guid)
    assert isinstance(media_id, str) and isinstance(project_guid, str)
    items = object_value(catalog.get("media_items"), "media_items")
    if set(items) != {media_id}:
        raise ProjectWriteError(
            "Expected exactly one matching main timeline media item"
        )
    item = object_value(items[media_id], "main timeline media item")
    if (
        item.get("id") != media_id
        or type(item.get("media_type")) is not int
        or item["media_type"] != 1048576
    ):
        raise ProjectWriteError("Main timeline media item identity or type mismatch")
    structure = object_value(catalog.get("media_structure"), "media_structure")
    if structure.get("media_item") != media_id:
        raise ProjectWriteError("Media structure identity mismatch")
    pending: list[Any] = [v for k, v in structure.items() if k != "media_item"]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if value.get("media_item"):
                raise ProjectWriteError(
                    "Additional media structure references are ambiguous"
                )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    timeline_uuid = item.get("timeline_uuid")
    uuid_value(timeline_uuid)
    assert isinstance(timeline_uuid, str)
    if len({uuid_value(v) for v in (media_id, project_guid, timeline_uuid)}) != 3:
        raise ProjectWriteError("Project identity values must be distinct")
    directory = f"ProjectFolder/Medias/{media_id}/"
    if (
        timeline.document != directory + "timeline.wesproj"
        or directory + "extra.json" not in project.raw_entries
    ):
        raise ProjectWriteError("Main timeline media directory is inconsistent")
    if (
        timeline.duration != 0
        or info.get("project_timeline_duration") != 0
        or item.get("duration") != 0
    ):
        raise ProjectWriteError("Empty template duration must be zero")
    for value in (
        info.get("project_date_create"),
        info.get("project_date_modify"),
        item.get("create_time"),
    ):
        if type(value) is not int or not 0 <= value < 10_000_000_000:
            raise ProjectWriteError("Expected Unix timestamps in seconds")
    if not isinstance(info.get("project_source"), str) or not re.fullmatch(
        r"[0-9a-fA-F]{32}", info["project_source"]
    ):
        raise ProjectWriteError("Expected a 32-character hex project source identifier")
    for field in ("project_file_name", "proj_zip_save_path", "proj_cover_proj_path"):
        if not isinstance(info.get(field), str):
            raise ProjectWriteError(f"Expected text metadata field: {field}")
    resolution = info.get("project_timeline_resolution")
    rate = info.get("project_timeline_framerate")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not isinstance(rate, list)
        or len(rate) != 2
    ):
        raise ProjectWriteError("Expected paired resolution and frame-rate settings")
    for value in [*resolution, *rate]:
        positive_int(value, "Template setting")
    frame_rate = object_value(timeline.raw.get("frameRate"), "frameRate")
    if resolution != [
        timeline.raw.get("resolutionWidth"),
        timeline.raw.get("resolutionHeight"),
    ] or rate != [frame_rate.get("num"), frame_rate.get("den")]:
        raise ProjectWriteError("Duplicated timeline settings disagree")
    entries = objects(timeline.raw.get("userData"), "timeline.userData")
    matched = []
    # Two distinct, experimentally verified references to the same timeline UUID.
    for key, size in ((11000, 64), (30309, 38)):
        candidates = [entry for entry in entries if entry.get("key") == key]
        if len(candidates) != 1:
            raise ProjectWriteError(
                f"Missing or ambiguous timeline UUID userData key {key}"
            )
        entry = candidates[0]
        if (
            type(entry["key"]) is not int
            or type(entry.get("size")) is not int
            or entry["size"] != size
        ):
            raise ProjectWriteError("Unsupported timeline UUID payload size or key")
        if not isinstance(entry.get("data"), str) or len(timeline_uuid) != 38:
            raise ProjectWriteError("Unsupported timeline UUID payload format")
        try:
            payload = base64.b64decode(entry["data"], validate=True)
        except ValueError as error:
            raise ProjectWriteError("Malformed timeline UUID Base64 payload") from error
        if payload != timeline_uuid.encode("ascii") + bytes(size - 38):
            raise ProjectWriteError(
                "Timeline UUID payload does not match its catalog identity"
            )
        matched.append(entry)
    instance_entry, payload = identity_payload(entries, 3, 64)
    if payload[38:] != bytes(26) or payload[:1] != b"{" or payload[37:38] != b"}":
        raise ProjectWriteError("Unsupported key 3 UUID padding or format")
    try:
        instance_uuid = payload[:38].decode("ascii")
    except UnicodeDecodeError as error:
        raise ProjectWriteError("Non-ASCII key 3 UUID") from error
    uuid_value(instance_uuid)
    token_entry, payload = identity_payload(entries, 140, 32)
    if not re.fullmatch(rb"[0-9a-f]{32}", payload):
        raise ProjectWriteError("Unsupported key 140 token format")
    token = payload.decode("ascii")
    buses = objects(timeline.raw.get("audioBusInfos"), "audioBusInfos")
    if not buses:
        raise ProjectWriteError("Missing main audio bus")
    bus_ids = [uuid_value(bus.get("busUid")) for bus in buses]
    if len(set(bus_ids)) != len(bus_ids):
        raise ProjectWriteError("Ambiguous audio bus identities")
    bus_uuid = buses[0]["busUid"]
    for track in timeline.tracks:
        refs = track.raw.get("busUuids", [])
        if not isinstance(refs, list) or any(
            uuid_value(ref) not in bus_ids for ref in refs
        ):
            raise ProjectWriteError("Unresolved track bus reference")
    if (
        len(
            {
                uuid_value(v)
                for v in (
                    project_guid,
                    media_id,
                    timeline_uuid,
                    instance_uuid,
                    bus_uuid,
                )
            }
        )
        != 5
    ):
        raise ProjectWriteError("Instance identities must be distinct")
    return EmptyTemplate(
        project,
        timeline,
        info,
        catalog,
        item,
        media_id,
        timeline_uuid,
        project_guid,
        directory,
        matched,
        bus_uuid,
        instance_entry,
        instance_uuid,
        token_entry,
        token,
    )


def instantiate_template(
    template: str | Path,
    output: str | Path,
    *,
    name: str,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ProjectWriteError("Project name must not be empty")
    if (width is None) != (height is None):
        raise ProjectWriteError("Width and height must be supplied together")
    for label, value in (("width", width), ("height", height), ("fps", fps)):
        if value is not None:
            positive_int(value, label)
    project = parse_project(template)
    graph = discover_empty_template(project)
    old_ids = {
        uuid_value(v)
        for v in (
            graph.project_guid,
            graph.media_id,
            graph.timeline_uuid,
            graph.bus_uuid,
            graph.instance_uuid,
        )
    }
    reserved = set(old_ids)
    # Avoid all textual template identities, including stable track and bus UUIDs.
    pending: list[Any] = list(project.raw_documents.values())
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
            pending.extend(value.keys())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and UUID_PATTERN.fullmatch(value):
            reserved.add(uuid_value(value))
    generated: list[dict[str, str]] = []

    def fresh(field: str, original: str) -> str:
        for _ in range(10):
            candidate = uuid4()
            if candidate not in reserved:
                reserved.add(candidate)
                value = uuid_style(candidate, original)
                generated.append(
                    {"field": field, "original": original, "generated": value}
                )
                return value
        raise ProjectWriteError("Could not allocate a fresh unique identity")

    project_guid = fresh("project_guid", graph.project_guid)
    media_id = fresh("timeline_mediaId", graph.media_id)
    timeline_uuid = fresh("timeline_uuid", graph.timeline_uuid)
    bus_uuid = fresh("audioBusInfos[0].busUid", graph.bus_uuid)
    instance_uuid = fresh("userData[3]", graph.instance_uuid)
    # No derivation is proven: this is an independent instance-specific opaque token.
    project_source = secrets.token_hex(16)
    if project_source.casefold() in {
        graph.info["project_source"].casefold(),
        graph.token,
    } or any(project_source.casefold() == value.hex for value in reserved):
        raise ProjectWriteError("Opaque project identifier collision")
    generated.append(
        {
            "field": "project_source",
            "original": graph.info["project_source"],
            "generated": project_source,
        }
    )
    token = secrets.token_hex(16)
    if token in {
        graph.token,
        project_source,
        graph.info["project_source"].lower(),
    } or any(token == value.hex for value in reserved):
        raise ProjectWriteError("Opaque timeline identifier collision")
    generated.append(
        {"field": "userData[140]", "original": graph.token, "generated": token}
    )
    timestamp = int(time.time())
    save_path = str(Path(output).resolve())
    old_path = graph.info["proj_zip_save_path"]
    if "://" in old_path:
        raise ProjectWriteError("Unsupported save-path style")
    save_path = (
        save_path.replace("\\", "/")
        if "/" in old_path
        else save_path.replace("/", "\\")
    )

    def cover_identity(match: re.Match[str]) -> str:
        value = match.group()
        return (
            uuid_style(uuid_value(project_guid), value)
            if uuid_value(value) == uuid_value(graph.project_guid)
            else value
        )

    cover_path = UUID_PATTERN.sub(cover_identity, graph.info["proj_cover_proj_path"])
    updates: Raw = {
        "project_guid": project_guid,
        "timeline_mediaId": media_id,
        "project_source": project_source,
        "project_file_name": name,
        "project_date_create": timestamp,
        "project_date_modify": timestamp,
        "proj_zip_save_path": save_path,
        "proj_cover_proj_path": cover_path,
    }
    edits: list[Edit] = [(INFO, (key,), value) for key, value in updates.items()]
    item_path = ("media_items", graph.media_id)
    edits.extend(
        [
            (CATALOG, item_path, JsonRenameKey(media_id)),
            (CATALOG, (*item_path, "id"), media_id),
            (CATALOG, (*item_path, "timeline_uuid"), timeline_uuid),
            (CATALOG, (*item_path, "create_time"), timestamp),
            (CATALOG, ("media_structure", "media_item"), media_id),
        ]
    )
    for entry in graph.uuid_entries:
        payload = timeline_uuid.encode("ascii") + bytes(entry["size"] - 38)
        path = object_path(project.raw_documents[graph.timeline.document], entry)
        edits.append(
            (
                graph.timeline.document,
                (*path, "data"),
                base64.b64encode(payload).decode("ascii"),
            )
        )
    resolution = list(graph.info["project_timeline_resolution"])
    rate = list(graph.info["project_timeline_framerate"])
    timeline_path = object_path(
        project.raw_documents[graph.timeline.document], graph.timeline.raw
    )
    edits.append(
        (
            graph.timeline.document,
            (*timeline_path, "audioBusInfos", 0, "busUid"),
            bus_uuid,
        )
    )
    expected_tracks = []
    for track in graph.timeline.tracks:
        expected = dict(track.raw)
        if "busUuids" in track.raw:
            refs = [
                uuid_style(uuid_value(bus_uuid), ref)
                if uuid_value(ref) == uuid_value(graph.bus_uuid)
                else ref
                for ref in track.raw["busUuids"]
            ]
            expected["busUuids"] = refs
            path = object_path(
                project.raw_documents[graph.timeline.document], track.raw
            )
            for index, (old, new) in enumerate(
                zip(track.raw["busUuids"], refs, strict=True)
            ):
                if old != new:
                    edits.append(
                        (graph.timeline.document, (*path, "busUuids", index), new)
                    )
        expected_tracks.append(expected)
    for entry, payload in (
        (graph.instance_entry, instance_uuid.encode("ascii") + bytes(26)),
        (graph.token_entry, token.encode("ascii")),
    ):
        path = object_path(project.raw_documents[graph.timeline.document], entry)
        edits.append(
            (
                graph.timeline.document,
                (*path, "data"),
                base64.b64encode(payload).decode("ascii"),
            )
        )
    if width is not None and height is not None:
        resolution = [width, height]
        edits.extend(
            [
                (INFO, ("project_timeline_resolution",), resolution),
                (graph.timeline.document, (*timeline_path, "resolutionWidth"), width),
                (graph.timeline.document, (*timeline_path, "resolutionHeight"), height),
            ]
        )
    if fps is not None:
        rate[0] = fps
        edits.extend(
            [
                (INFO, ("project_timeline_framerate", 0), fps),
                (graph.timeline.document, (*timeline_path, "frameRate", "num"), fps),
            ]
        )
    new_directory = f"ProjectFolder/Medias/{media_id}/"
    renames = {
        entry: new_directory + entry[len(graph.directory) :]
        for entry in project.raw_entries
        if entry.startswith(graph.directory)
    }

    def validate(updated: Project) -> None:
        result = discover_empty_template(updated)
        actual_ids = (
            result.project_guid,
            result.media_id,
            result.timeline_uuid,
            result.bus_uuid,
            result.instance_uuid,
        )
        if actual_ids != (
            project_guid,
            media_id,
            timeline_uuid,
            bus_uuid,
            instance_uuid,
        ) or any(uuid_value(v) in old_ids for v in actual_ids):
            raise ProjectWriteError("Post-write identity freshness validation failed")
        if result.token != token or result.token == graph.token:
            raise ProjectWriteError("Post-write token freshness validation failed")
        if any(result.info.get(key) != value for key, value in updates.items()):
            raise ProjectWriteError("Post-write metadata validation failed")
        if (
            result.info["project_timeline_resolution"] != resolution
            or result.info["project_timeline_framerate"] != rate
        ):
            raise ProjectWriteError("Post-write settings validation failed")
        if result.item["create_time"] != timestamp:
            raise ProjectWriteError("Post-write creation timestamp validation failed")
        if [t.raw for t in result.timeline.tracks] != expected_tracks:
            raise ProjectWriteError(
                "Post-write empty track preservation validation failed"
            )
        if any(entry.startswith(graph.directory) for entry in updated.raw_entries):
            raise ProjectWriteError("Old timeline media directory remains")

    report = write_copy(
        Path(template), Path(output), project, edits, validate, renames=renames
    )
    report.update(
        template_timeline_media_id=graph.media_id,
        generated_timeline_media_id=media_id,
        template_timeline_uuid=graph.timeline_uuid,
        generated_timeline_uuid=timeline_uuid,
        template_project_guid=graph.project_guid,
        generated_project_guid=project_guid,
        project_name=name,
        resolution=resolution,
        fps=rate[0],
        renamed_entries=[
            {"original": old, "generated": new} for old, new in sorted(renames.items())
        ],
        generated_identities=generated,
    )
    return report
