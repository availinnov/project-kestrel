"""Neutral summaries of normalized project models."""

from collections import Counter
from typing import Any

from kestrel.project.models import Project


def project_summary(project: Project) -> dict[str, Any]:
    """Report native time values and active duration without summing nested content."""
    tracks = [track for timeline in project.timelines for track in timeline.tracks]
    trims = []
    transitions = []
    nested = []
    for timeline in project.timelines:
        for track in timeline.tracks:
            for clip in track.clips:
                location = {
                    "document": timeline.document,
                    "timeline_id": timeline.id,
                    "track_id": track.id,
                    "clip_id": clip.id,
                }
                if clip.in_point is not None or clip.out_point is not None:
                    trims.append(
                        {
                            **location,
                            "in_point": clip.in_point,
                            "out_point": clip.out_point,
                            "begin": clip.begin,
                            "end": clip.end,
                            "type": clip.type,
                            "source_id": clip.source_id,
                        }
                    )
                if clip.transitions:
                    transitions.append(
                        {
                            **location,
                            "transitions": [
                                {
                                    "id": t.id,
                                    "position": t.position,
                                    "type": t.type,
                                    "begin": t.begin,
                                    "end": t.end,
                                }
                                for t in clip.transitions
                            ],
                        }
                    )
                if clip.nested_timeline_id is not None:
                    nested.append(
                        {
                            **location,
                            "target_timeline_id": clip.nested_timeline_id,
                            "resolved": clip.nested_timeline is not None,
                        }
                    )
    return {
        "timeline_count": len(project.timelines),
        "resource_count": len(project.resources),
        "track_count_by_type": dict(
            sorted(
                Counter(
                    "unknown" if t.type is None else str(t.type) for t in tracks
                ).items()
            )
        ),
        "clip_count": sum(len(t.clips) for t in tracks),
        "total_duration": project.duration,
        "time_unit": "native",
        "duration_basis": "active timeline end from time zero",
        "active_timeline_id": project.active_timeline.id
        if project.active_timeline
        else None,
        "clips_with_trim_values": trims,
        "clips_with_transitions": transitions,
        "nested_timeline_references": nested,
        "warnings": project.warnings,
    }
