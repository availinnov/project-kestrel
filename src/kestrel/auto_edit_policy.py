"""Explicit provisional Auto Edit v1 policy; independent of Filmora encoding."""

import copy
import math
from fractions import Fraction
from typing import Any, TypeGuard

from kestrel.editplan import ClipEdit, EditPlan, trimmed_range

MIN_DURATION_SECONDS = 3.0
BLACK_REJECT_RATIO = 0.80
TRIM_START_TICKS = 5_000_000
TRIM_END_TICKS = 7_000_000
MAX_CAPTURE_GAP_SECONDS = 180
SAME_SCENE_MIN_INLIERS = 10
LONG_DURATION_SECONDS = 20.0
RELATIVE_LOUD_THRESHOLD_DB = 15.0
GAIN_BREAKPOINT_DBFS = -41.869214
TAG_PRIORITY = (
    ("same_scene_review", 1),
    ("short_relative_loud_event", 7),
    ("long_clip_review", 2),
)
POLICY = dict(
    min_duration_seconds=MIN_DURATION_SECONDS,
    black_reject_ratio=BLACK_REJECT_RATIO,
    trim_start_ticks=TRIM_START_TICKS,
    trim_end_ticks=TRIM_END_TICKS,
    max_capture_gap_seconds=MAX_CAPTURE_GAP_SECONDS,
    same_scene_min_inliers=SAME_SCENE_MIN_INLIERS,
    long_duration_seconds=LONG_DURATION_SECONDS,
    relative_loud_threshold_db=RELATIVE_LOUD_THRESHOLD_DB,
    tag_priority=[tag for _, tag in TAG_PRIORITY],
    gain_model=dict(
        breakpoint_dbfs=GAIN_BREAKPOINT_DBFS,
        low_intercept=-7.315733,
        low_slope=-0.275133,
        high_intercept=-30.417396,
        high_slope=-0.826891,
    ),
    same_scene_note="Provisional review heuristic: calibration subset 10 positives, "
    "15 negatives; TP8 FN2 TN12 FP3, precision .727 recall .800 F1 .762; "
    "unlabeled controls remain. No deletion or best-take selection.",
)


def finite(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value)


def predict_gain(raw_rms_dbfs: float) -> float:
    if not finite(raw_rms_dbfs):
        raise ValueError("Raw RMS must be finite")
    if raw_rms_dbfs < GAIN_BREAKPOINT_DBFS:
        return -7.315733 - 0.275133 * raw_rms_dbfs
    return -30.417396 - 0.826891 * raw_rms_dbfs


def hard_quality_plan(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = copy.deepcopy(sources)
    for source in decisions:
        duration = source["original_duration"]
        video = source["analysis"]["signals"].get("video") or {}
        reasons = []
        if finite(duration) and duration < MIN_DURATION_SECONDS:
            reasons.append("duration_under_3s")
        ratio = video.get("black_frame_ratio")
        if finite(ratio) and ratio >= BLACK_REJECT_RATIO:
            reasons.append("black_screen")
        source.update(
            keep=not reasons,
            drop_reasons=reasons,
            reasons=list(reasons),
            warnings=[],
            planned_gain_db=None,
            applied_gain_db=None,
            color_tag=None,
            trim_start_ticks=0,
            trim_end_ticks=0,
        )
        if source["analysis"]["analysis_status"] != "ok":
            source["warnings"].append(
                source["analysis"].get("error", "Media analysis unavailable")
            )
    return decisions


def finalize_plan(
    decisions: list[dict[str, Any]], relations: list[dict[str, Any]], fps: Fraction
) -> tuple[EditPlan, list[dict[str, Any]]]:
    decisions = copy.deepcopy(decisions)
    by_id = {s["catalog_id"]: s for s in decisions}
    for source in decisions:
        if not source["keep"]:
            continue
        source.update(trim_start_ticks=TRIM_START_TICKS, trim_end_ticks=TRIM_END_TICKS)
        edit = ClipEdit(source["catalog_id"], True, TRIM_START_TICKS, TRIM_END_TICKS)
        try:
            trimmed_range(source["full_out_ticks"], edit, fps)
        except ValueError:
            source.update(trim_start_ticks=0, trim_end_ticks=0)
            source["reasons"].append("edge_trim_fallback_untrimmed")
        if source["original_duration"] >= LONG_DURATION_SECONDS:
            source["reasons"].append("long_clip_review")
        audio = source["analysis"]["signals"].get("audio") or {}
        short = audio.get("short_relative_loud_event_count")
        if finite(short) and short > 0:
            source["reasons"].append("short_relative_loud_event")
        raw_rms = audio.get("rms_dbfs")
        if audio.get("available") is True and finite(raw_rms):
            source["planned_gain_db"] = predict_gain(raw_rms)
        else:
            source["warnings"].append(
                "Audio analysis unavailable/silent; gain unchanged"
            )
    for relation in relations:
        ids = [relation[side]["catalog_id"] for side in ("left", "right")]
        gap, inliers = (
            relation.get("capture_time_gap_seconds"),
            relation.get("orb_best_inlier_count"),
        )
        tagged = (
            all(identifier in by_id and by_id[identifier]["keep"] for identifier in ids)
            and finite(gap)
            and 0 <= gap <= MAX_CAPTURE_GAP_SECONDS
            and finite(inliers)
            and inliers >= SAME_SCENE_MIN_INLIERS
        )
        relation["tagged"] = tagged
        if tagged:
            for identifier in ids:
                if "same_scene_review" not in by_id[identifier]["reasons"]:
                    by_id[identifier]["reasons"].append("same_scene_review")
    edits = []
    for source in decisions:
        source["color_tag"] = next(
            (tag for reason, tag in TAG_PRIORITY if reason in source["reasons"]), None
        )
        edits.append(
            ClipEdit(
                source["catalog_id"],
                source["keep"],
                source["trim_start_ticks"],
                source["trim_end_ticks"],
                source["planned_gain_db"],
                source["color_tag"],
                tuple(source["reasons"]),
            )
        )
    return EditPlan(tuple(edits)), decisions
