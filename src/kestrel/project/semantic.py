"""Conservative semantic comparison of structured project containers."""

import base64
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from kestrel.formats.diff import json_changes
from kestrel.formats.inspect import parse_json
from kestrel.project.models import Clip, Project, RawObject, Timeline, Track
from kestrel.project.parser import parse_project

VOLATILE = frozenset(
    {
        "thisUId",
        "project_guid",
        "project_source",
        "project_date_modify",
        "proj_zip_save_path",
        "proj_cover_proj_path",
        "timeline_uuid",
    }
)
UUID_TEXT = re.compile(
    r"(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|\{[0-9a-fA-F-]{36}\})"
)


def uuid_key(value: Any) -> str | None:
    if not isinstance(value, str) or not UUID_TEXT.fullmatch(value):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def keyed_list(value: Any, field: str) -> bool:
    """Use unique integer user-data keys; other identity fields also allow text."""
    allowed_types = (int,) if field == "key" else (str, int)
    return (
        isinstance(value, list)
        and all(
            isinstance(item, dict) and type(item.get(field)) in allowed_types
            for item in value
        )
        and len({str(item[field]) for item in value}) == len(value)
    )


def register_ids(value: Any, path: str, references: dict[str, str]) -> None:
    """Associate generated UUIDs with their structural location, not a wildcard."""
    if isinstance(value, dict):
        for key, item in value.items():
            uid = uuid_key(item) if key in VOLATILE else None
            if uid is not None:
                references[uid] = path + "/" + key
            elif key not in VOLATILE:
                register_ids(item, path + "/" + key, references)
    elif isinstance(value, list):
        field = "id" if path.endswith("/effectList") else "key"
        keyed = keyed_list(value, field)
        for i, item in enumerate(value):
            register_ids(item, f"{path}/{item[field] if keyed else i}", references)


def normalize(value: Any, references: dict[str, str], context: str = "") -> Any:
    """Retain unknown payloads; normalize only identified generated references."""
    if isinstance(value, RawObject) and len(value.pairs) != len(value):
        return {"raw_pairs": [[k, normalize(v, references, k)] for k, v in value.pairs]}
    if isinstance(value, dict):
        result = {
            key: normalize(item, references, key)
            for key, item in value.items()
            if key not in VOLATILE
        }
        if context == "userDataEntry" and isinstance(value.get("data"), str):
            try:
                payload = base64.b64decode(value["data"], validate=True)
                text = payload.rstrip(b"\x00").decode("ascii")
            except (ValueError, UnicodeError):
                return result
            uid = uuid_key(text)
            if (
                uid is not None
                and uid in references
                and type(value.get("size")) is int
                and value["size"] == len(payload)
            ):
                result["data"] = {"generated_reference": references[uid]}
                result.pop("size", None)
        return result
    if isinstance(value, list):
        if context == "effectList" and keyed_list(value, "id"):
            return {str(item["id"]): normalize(item, references) for item in value}
        if context == "userData" and keyed_list(value, "key"):
            return {
                str(item["key"]): normalize(item, references, "userDataEntry")
                for item in value
            }
        return [
            normalize(
                item, references, "userDataEntry" if context == "userData" else ""
            )
            for item in value
        ]
    if context == "parameter" and isinstance(value, str):
        try:
            decoded = parse_json(value.encode())
        except (ValueError, RecursionError):
            return value
        if isinstance(decoded, (dict, list)):
            return normalize(decoded, references)
    return value


def pair_unique[T](
    left: list[T], right: list[T], keys: list[Callable[[T], Any]]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match unique anchors in successive passes; never guess between duplicates."""
    a, b = set(range(len(left))), set(range(len(right)))
    pairs = []
    for key in keys:
        ka = {i: key(left[i]) for i in a}
        kb = {i: key(right[i]) for i in b}
        ca, cb = Counter(ka.values()), Counter(kb.values())
        lookup = {
            value: i for i, value in kb.items() if value is not None and cb[value] == 1
        }
        for i in sorted(a):
            value = ka[i]
            if value is not None and ca[value] == 1 and value in lookup:
                j = lookup[value]
                pairs.append((i, j))
                a.remove(i)
                b.remove(j)
    return sorted(pairs), sorted(a), sorted(b)


def clip_keys() -> list[Callable[[Clip], Any]]:
    return [
        lambda c: (c.source_id, c.type, c.begin, c.end, c.in_point, c.out_point),
        lambda c: (c.source_id, c.type, c.begin),
        lambda c: (c.source_id, c.type, c.in_point, c.out_point),
        lambda c: (c.source_id, c.type),
    ]


def pair_clips(
    left: list[Clip], right: list[Clip]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pairs, removed, added = pair_unique(left, right, clip_keys())
    # Identical duplicate values are interchangeable; UUID alone never breaks a tie.
    for ai in list(removed):
        value = normalize(left[ai].raw, {})
        for bi in added:
            if value == normalize(right[bi].raw, {}):
                pairs.append((ai, bi))
                removed.remove(ai)
                added.remove(bi)
                break
    return sorted(pairs), removed, added


def track_keys(tracks: list[Track]) -> dict[tuple[str, int], Track]:
    counts: Counter[str] = Counter()
    result = {}
    for track in tracks:
        kind = json.dumps(track.type)
        result[kind, counts[kind]] = track
        counts[kind] += 1
    return result


def timeline_shape(timeline: Timeline) -> tuple[Any, ...]:
    return tuple(
        (
            track.type,
            tuple(
                sorted(json.dumps((clip.source_id, clip.type)) for clip in track.clips)
            ),
        )
        for track in timeline.tracks
    )


def brief(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(encoded) <= 500:
        return value
    return {
        "preview": encoded[:200],
        "serialized_length": len(encoded),
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def semantic_diff(left: Project, right: Project) -> dict[str, Any]:
    """Compare model structure; external assets and auxiliary documents are excluded."""
    changes: list[dict[str, Any]] = []
    a_refs: dict[str, str] = {}
    b_refs: dict[str, str] = {}
    timeline_pairs, removed, added = pair_unique(
        left.timelines,
        right.timelines,
        [
            lambda t: (t.document, t.id),
            timeline_shape,
        ],
    )
    # Active roots are a safe final anchor only when both remain unmatched.
    if left.active_timeline is not None and right.active_timeline is not None:
        ai = next(i for i, t in enumerate(left.timelines) if t is left.active_timeline)
        bi = next(
            i for i, t in enumerate(right.timelines) if t is right.active_timeline
        )
        if ai in removed and bi in added:
            timeline_pairs.append((ai, bi))
            removed.remove(ai)
            added.remove(bi)
    a_names = {
        (t.document, t.id): f"removed-timeline[{i}]"
        for i, t in enumerate(left.timelines)
    }
    b_names = {
        (t.document, t.id): f"added-timeline[{i}]"
        for i, t in enumerate(right.timelines)
    }
    metadata = []
    for project in (left, right):
        excluded = {"timelineInfos", "resources", "currentTimelineId"}
        metadata.append(
            {k: v for k, v in project.metadata.items() if k not in excluded}
        )
    comparisons: list[tuple[str, Any, Any]] = [("$/metadata", metadata[0], metadata[1])]
    for ai, bi in sorted(timeline_pairs):
        a, b = left.timelines[ai], right.timelines[bi]
        path = f"$/timelines/{ai}"
        a_names[a.document, a.id] = b_names[b.document, b.id] = path
    for ai, bi in sorted(timeline_pairs):
        a, b = left.timelines[ai], right.timelines[bi]
        path = a_names[a.document, a.id]
        comparisons.append(
            (
                path,
                {
                    k: v
                    for k, v in a.raw.items()
                    if k not in {"timelineId", "trackInfos"}
                },
                {
                    k: v
                    for k, v in b.raw.items()
                    if k not in {"timelineId", "trackInfos"}
                },
            )
        )
        at, bt = track_keys(a.tracks), track_keys(b.tracks)
        for key in sorted(at.keys() | bt.keys()):
            track_path = f"{path}/tracks/{key[0]}[{key[1]}]"
            if key not in at or key not in bt:
                track = at.get(key) or bt[key]
                changes.append(
                    {
                        "path": track_path,
                        "change": "removed" if key in at else "added",
                        "track_type": track.type,
                        "clip_count": len(track.clips),
                    }
                )
                continue
            ta, tb = at[key], bt[key]
            comparisons.append(
                (
                    track_path,
                    {k: v for k, v in ta.raw.items() if k not in {"uuid", "clipList"}},
                    {k: v for k, v in tb.raw.items() if k not in {"uuid", "clipList"}},
                )
            )
            for track, refs in ((ta, a_refs), (tb, b_refs)):
                uid = uuid_key(track.id)
                if uid:
                    refs[uid] = track_path
            pairs, ca, cb = pair_clips(ta.clips, tb.clips)
            for ia, ib in pairs:
                clip_path = f"{track_path}/clips/{ia}"
                va, vb = dict(ta.clips[ia].raw), dict(tb.clips[ib].raw)
                for raw, timeline, names in ((va, a, a_names), (vb, b, b_names)):
                    if "timelineId" in raw:
                        raw["timelineId"] = names.get(
                            (timeline.document, raw["timelineId"]),
                            {"unresolved": raw["timelineId"]},
                        )
                comparisons.append((clip_path, va, vb))
            for indices, clips, action in (
                (ca, ta.clips, "removed"),
                (cb, tb.clips, "added"),
            ):
                for i in indices:
                    clip = clips[i]
                    changes.append(
                        {
                            "path": f"{track_path}/clips/{i}",
                            "change": action,
                            "source_id": clip.source_id,
                            "clip_type": clip.type,
                            "begin": clip.begin,
                            "end": clip.end,
                            "in_point": clip.in_point,
                            "out_point": clip.out_point,
                        }
                    )
    for indices, timelines, action in (
        (removed, left.timelines, "removed"),
        (added, right.timelines, "added"),
    ):
        for i in indices:
            timeline = timelines[i]
            changes.append(
                {
                    "path": f"$/timelines/{i}",
                    "change": action,
                    "track_count": len(timeline.tracks),
                    "clip_count": sum(len(t.clips) for t in timeline.tracks),
                }
            )
    # Header/resource comparison is scoped to matched document pairs.
    doc_pairs = sorted(
        {
            (left.timelines[i].document, right.timelines[j].document)
            for i, j in timeline_pairs
        }
    )
    for i, (ad, bd) in enumerate(doc_pairs):
        va, vb = dict(left.raw_documents[ad]), dict(right.raw_documents[bd])
        for raw, doc, names in ((va, ad, a_names), (vb, bd, b_names)):
            raw.pop("timelineInfos", None)
            if "currentTimelineId" in raw:
                raw["currentTimelineId"] = names.get(
                    (doc, raw["currentTimelineId"]),
                    {"unresolved": raw["currentTimelineId"]},
                )
            if keyed_list(raw.get("resources"), "sourceUuid"):
                raw["resources"] = {str(r["sourceUuid"]): r for r in raw["resources"]}
        comparisons.append((f"$/documents/{i}", va, vb))
    for path, va, vb in comparisons:
        register_ids(va, path, a_refs)
        register_ids(vb, path, b_refs)
    for path, va, vb in comparisons:
        for change in json_changes(normalize(va, a_refs), normalize(vb, b_refs), path):
            changes.append(
                {
                    k: brief(v) if k in {"left", "right"} else v
                    for k, v in change.items()
                }
            )
    return {
        "mode": "semantic",
        "scope": "project structure and resource definitions",
        "equal": not changes,
        "change_count": len(changes),
        "changes": changes,
        "warnings": {"left": left.warnings, "right": right.warnings},
    }


def semantic_diff_files(left: str | Path, right: str | Path) -> dict[str, Any]:
    return semantic_diff(parse_project(left), parse_project(right))
