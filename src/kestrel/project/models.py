"""Normalized views with original fields and container bytes retained."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

type Identifier = str | int
type TimeValue = int | float
type Raw = dict[str, Any]

COLOR_TAG_PALETTE_ORDER = (1, 2, 3, 4, 5, 6, 7, 13, 8, 9, 10, 11, 12)


class RawObject(dict[str, Any]):
    """Dictionary view plus every original pair, including repeated keys."""

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        self.pairs = pairs


@dataclass
class Resource:
    id: Identifier
    document: str
    filename: str | None
    duration: TimeValue | None
    raw: Raw = field(repr=False)


@dataclass
class Transition:
    id: Identifier | None
    position: str
    type: Identifier | None
    begin: TimeValue | None
    end: TimeValue | None
    raw: Raw = field(repr=False)


@dataclass
class Clip:
    id: Identifier
    type: Identifier | None
    source_id: Identifier | None
    in_point: TimeValue | None
    out_point: TimeValue | None
    begin: TimeValue | None
    end: TimeValue | None
    nested_timeline_id: Identifier | None
    audio_volume: Raw
    transitions: list[Transition]
    raw: Raw = field(repr=False)
    resource: Resource | None = field(default=None, repr=False)
    nested_timeline: Timeline | None = field(default=None, repr=False)
    color_tag: int | None = None
    audio_gain_db: float | None = None


@dataclass
class Track:
    id: Identifier
    type: Identifier | None
    clips: list[Clip]
    raw: Raw = field(repr=False)


@dataclass
class Timeline:
    id: Identifier
    document: str
    tracks: list[Track]
    raw: Raw = field(repr=False)

    @property
    def duration(self) -> TimeValue | None:
        """End from time zero, including gaps; unknown if any clip end is absent."""
        clips = [clip for track in self.tracks for clip in track.clips]
        if any(clip.end is None for clip in clips):
            return None
        return max((clip.end for clip in clips if clip.end is not None), default=0)


@dataclass
class Project:
    metadata: Raw
    resources: list[Resource]
    timelines: list[Timeline]
    active_timeline: Timeline | None
    raw_documents: dict[str, Any] = field(repr=False)
    raw_entries: dict[str, bytes] = field(repr=False)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> TimeValue | None:
        return self.active_timeline.duration if self.active_timeline else None
