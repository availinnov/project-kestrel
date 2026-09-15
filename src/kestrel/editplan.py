"""Source-oriented editing decisions and exact frame-grid arithmetic."""

import json
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ClipEdit:
    catalog_id: str
    keep: bool
    trim_start_ticks: int
    trim_end_ticks: int


@dataclass(frozen=True)
class EditPlan:
    clips: tuple[ClipEdit, ...]


def parse_plan_json(text: str) -> EditPlan:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    data = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or set(data) != {"version", "clips"}:
        raise ValueError("Plan requires only version and clips")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Plan version must be 1")
    if not isinstance(data["clips"], list):
        raise ValueError("Plan clips must be an array")
    clips = []
    seen = set()
    for item in data["clips"]:
        if not isinstance(item, dict) or set(item) != {
            "catalog_id",
            "keep",
            "trim_start_ticks",
            "trim_end_ticks",
        }:
            raise ValueError("Clip requires catalog_id, keep and both trim fields only")
        identifier = item["catalog_id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("catalog_id must be a non-empty string")
        if identifier in seen:
            raise ValueError(f"Duplicate catalog_id: {identifier}")
        seen.add(identifier)
        if type(item["keep"]) is not bool:
            raise ValueError("keep must be boolean")
        for key in ("trim_start_ticks", "trim_end_ticks"):
            if type(item[key]) is not int or item[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            if not item["keep"] and item[key]:
                raise ValueError("Dropped sources must have zero trims")
        clips.append(ClipEdit(**item))
    return EditPlan(tuple(clips))


def snap_boundary(ticks: int, fps: Fraction) -> int:
    if type(ticks) is not int or ticks < 0 or fps <= 0:
        raise ValueError("Boundary and frame rate must be valid and non-negative")

    def half_up(value: Fraction) -> int:
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)

    frames = half_up(Fraction(ticks, 10_000_000) * fps)
    return half_up(Fraction(frames * 10_000_000) / fps)


def trimmed_range(full_out: int, edit: ClipEdit, fps: Fraction) -> tuple[int, int]:
    start = edit.trim_start_ticks
    end = full_out - edit.trim_end_ticks
    if start < 0 or end <= start or end > full_out:
        raise ValueError("Trims exceed or collapse the source range")
    start, end = snap_boundary(start, fps), snap_boundary(end, fps)
    if not 0 <= start < end <= full_out:
        raise ValueError("Frame-snapped trims collapse or exceed the source range")
    return start, end
