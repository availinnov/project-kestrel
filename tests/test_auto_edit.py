"""Generated control projects for Auto Edit policy and Filmora materialization."""

import copy
import json
import subprocess
import zipfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from kestrel.auto_edit import (
    project_auto_edit,
    publish_new_file,
    same_scene_measurements,
)
from kestrel.auto_edit_policy import (
    GAIN_BREAKPOINT_DBFS,
    finalize_plan,
    hard_quality_plan,
    predict_gain,
)
from kestrel.duplicate_analysis import describe_frame
from kestrel.duplicate_orb import OrbFrame
from kestrel.editplan import trimmed_range
from kestrel.project.parser import parse_project
from kestrel.project.sequence import build_constant_speed_param
from kestrel.project.writer import ProjectWriteError
from tests.test_dataset import dataset_fixture
from tests.test_duplicate_analysis import image
from tests.test_sequence import rewrite_fixture


def measurement(
    duration: float, black: float = 0, rms: float | None = -30, loud: int = 0
) -> dict[str, Any]:
    return dict(
        analysis_status="ok",
        signals=dict(
            duration_seconds=duration,
            video=dict(black_frame_ratio=black),
            audio=dict(
                available=rms is not None,
                rms_dbfs=rms,
                short_relative_loud_event_count=loud,
            ),
        ),
    )


def source(identifier: str, duration: float, **kwargs: Any) -> dict[str, Any]:
    return dict(
        catalog_id=identifier,
        filename=identifier + ".mp4",
        source_path=identifier,
        capture_time=100,
        original_duration=duration,
        full_out_ticks=int(duration * 10_000_000),
        analysis=measurement(duration, **kwargs),
    )


def relation(gap: int = 180, inliers: int = 10) -> dict[str, Any]:
    return dict(
        left=dict(catalog_id="a"),
        right=dict(catalog_id="b"),
        capture_time_gap_seconds=gap,
        orb_best_inlier_count=inliers,
    )


@pytest.mark.parametrize(
    ("duration", "black", "keep"),
    [(2.999, 0, False), (3, 0, True), (5, 0.8, False), (5, 0.799, True)],
)
def test_hard_rules(duration: float, black: float, keep: bool) -> None:
    decision = hard_quality_plan([source("a", duration, black=black)])[0]
    assert decision["keep"] is keep
    if not keep:
        plan, final = finalize_plan([decision], [], Fraction(60))
        assert plan.clips[0].gain_db is None
        assert final[0]["trim_start_ticks"] == 0


@pytest.mark.parametrize(
    ("gap", "inliers", "tagged"), [(180, 10, True), (181, 10, False), (10, 9, False)]
)
def test_review_only_priority_all_reasons(gap: int, inliers: int, tagged: bool) -> None:
    sources = [source("a", 25, loud=1), source("b", 3)]
    original = copy.deepcopy(sources)
    relations = [relation(gap, inliers)]
    plan, decisions = finalize_plan(hard_quality_plan(sources), relations, Fraction(60))
    assert all(edit.keep for edit in plan.clips)
    assert decisions[0]["color_tag"] == (1 if tagged else 4)
    assert {"short_relative_loud_event", "long_clip_review"} <= set(
        decisions[0]["reasons"]
    )
    assert ("same_scene_review" in decisions[0]["reasons"]) is tagged
    assert relations[0]["tagged"] is tagged
    assert sources == original
    assert (
        plan
        == finalize_plan(
            hard_quality_plan(sources), [relation(gap, inliers)], Fraction(60)
        )[0]
    )


def test_long_tag_and_trim_fallback() -> None:
    decisions = hard_quality_plan([source("a", 20), source("b", 3)])
    decisions[1]["full_out_ticks"] = (
        1_000_000  # inconsistent/very short frame span control
    )
    plan, final = finalize_plan(decisions, [], Fraction(60))
    assert final[0]["color_tag"] == 2
    assert plan.clips[0].trim_start_ticks == 5_000_000
    assert plan.clips[0].trim_end_ticks == 7_000_000
    for edit, s in zip(plan.clips, final, strict=True):
        start, end = trimmed_range(s["full_out_ticks"], edit, Fraction(60))
        assert end > start
    assert "edge_trim_fallback_untrimmed" in final[1]["reasons"]


@pytest.mark.parametrize("rms", [-60, GAIN_BREAKPOINT_DBFS, -20])
def test_fixed_piecewise_gain(rms: float) -> None:
    expected = (
        -7.315733 - 0.275133 * rms
        if rms < GAIN_BREAKPOINT_DBFS
        else -30.417396 - 0.826891 * rms
    )
    assert predict_gain(rms) == expected


def test_audio_failure_and_black_failure_keep() -> None:
    s = source("a", 5, rms=None)
    s["analysis"].update(analysis_status="missing_media", error="missing")
    s["analysis"]["signals"]["video"] = None
    plan, final = finalize_plan(hard_quality_plan([s]), [], Fraction(60))
    assert plan.clips[0].keep
    assert plan.clips[0].gain_db is None
    assert final[0]["warnings"]


def auto_fixture(tmp_path: Path, durations: tuple[float, ...] = (2.9, 25, 3)) -> Path:
    path = dataset_fixture(tmp_path)

    def change(docs: Any) -> None:
        resources = docs["ProjectFolder/Medias/main/timeline.wesproj"]["resources"]
        for i, duration in enumerate(durations):
            ticks = int(duration * 10_000_000)
            resources[i]["mediaLength"] = ticks
            docs["ProjectFolder/Medias/medias_info.json"]["media_items"][f"m{i}"][
                "media_length"
            ] = ticks
            docs[f"ProjectFolder/Medias/m{i}/media.json"]["sourceInfo"]["basicInfo"][
                "mediaLength"
            ] = ticks

    rewrite_fixture(path, change)
    return path


def install_signals(
    monkeypatch: Any, *, black: float = 0, no_audio: bool = False
) -> list[str]:
    calls = []

    def analyze(
        filename: str, ticks: int, audio: bool, executable: Any, config: Any
    ) -> Any:
        assert config.relative_loud_window_db == 15
        duration = ticks / 10_000_000
        return measurement(
            duration,
            black=black if filename.endswith("2.mp4") else 0,
            rms=None if no_audio else -30,
            loud=1 if filename.endswith("1.mp4") else 0,
        )

    def sample(filename: str, *args: Any) -> Any:
        calls.append(filename)
        assert not filename.endswith("0.mp4")
        return [replace(describe_frame(image(1)), orb=OrbFrame([], None))]

    monkeypatch.setattr("kestrel.auto_edit.analyze_source", analyze)
    monkeypatch.setattr("kestrel.auto_edit.sample_source", sample)
    monkeypatch.setattr(
        "kestrel.auto_edit.compare_orb", lambda *args: dict(orb_best_inlier_count=10)
    )
    return calls


def test_full_pipeline_archive_trims_gain_tags_contiguous(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_path = auto_fixture(tmp_path)
    before = source_path.read_bytes()
    calls = install_signals(monkeypatch)
    output = tmp_path / "auto.wfp"
    result = project_auto_edit(source_path, output)
    assert source_path.read_bytes() == before
    assert result["kept_count"] == 2
    assert result["dropped_count"] == 1
    assert len(calls) == 2  # exactly once per retained candidate source
    report = json.loads((tmp_path / "auto_report.json").read_text())
    kept = [s for s in report["sources"] if s["keep"]]
    assert all(s["color_tag"] == 1 for s in kept)
    assert {
        "long_clip_review",
        "short_relative_loud_event",
        "same_scene_review",
    } <= set(kept[0]["reasons"])
    assert all(s["applied_gain_db"] == predict_gain(-30) for s in kept)
    assert all(
        s["trim_start"] == 0.5 and s["trim_end"] == pytest.approx(0.7) for s in kept
    )
    original, updated = parse_project(source_path), parse_project(output)
    assert original.active_timeline and updated.active_timeline
    selected = report["materialization"]["selected_sources"]
    assert [s["catalog_id"] for s in selected] == ["m1", "m2"]
    end = 0
    for s in selected:
        assert s["tl_begin"] == end
        end = s["tl_end"]
        clips = [
            c
            for t in updated.active_timeline.tracks
            for c in t.clips
            if c.id in (s["video_clip_id"], s["audio_clip_id"])
        ]
        assert len(clips) == 2
        for clip in clips:
            assert clip.raw["speed"]["speedParam"] == build_constant_speed_param(
                s["full_out_point"]
            )
            assert clip.raw["speed"]["offset"] == clip.in_point / 10_000_000
            assert clip.raw["speed"]["offsetEnd"] == clip.out_point / 10_000_000
        assert next(c for c in clips if c.type == 1).color_tag == 1
    assert updated.duration == end
    for i in (1, 3):
        assert (
            original.active_timeline.tracks[i].raw
            == updated.active_timeline.tracks[i].raw
        )
    with zipfile.ZipFile(source_path) as a, zipfile.ZipFile(output) as b:
        for name in a.namelist():
            if name not in report["materialization"]["modified_entries"]:
                assert a.read(name) == b.read(name)


def test_black_reject_never_sampled_gain_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path = auto_fixture(tmp_path)
    calls = install_signals(monkeypatch, black=0.8, no_audio=True)
    project_auto_edit(path, tmp_path / "auto.wfp")
    report = json.loads((tmp_path / "auto_report.json").read_text())
    assert report["kept_count"] == 1
    assert calls == []  # remaining isolated source has no eligible comparison
    assert report["summary"]["dropped_black"] == 1
    assert report["summary"]["gain_adjusted"] == 0
    assert report["sources"][1]["applied_gain_db"] is None


def test_pair_decode_failure_does_not_tag(monkeypatch: Any) -> None:
    decisions = hard_quality_plan([source("a", 3), source("b", 3)])

    def fail(*args: Any) -> Any:
        raise subprocess.TimeoutExpired("ffmpeg", 60)

    monkeypatch.setattr("kestrel.auto_edit.sample_source", fail)
    relations = same_scene_measurements(decisions, "ffmpeg")
    assert len(relations) == 1 and "warning" in relations[0]
    plan, final = finalize_plan(decisions, relations, Fraction(60))
    assert all(s["keep"] and s["color_tag"] is None for s in final)
    assert all(edit.keep for edit in plan.clips)


def test_output_safety(tmp_path: Path) -> None:
    source_path = auto_fixture(tmp_path)
    with pytest.raises(ProjectWriteError):
        project_auto_edit(source_path, source_path)
    with pytest.raises(ProjectWriteError):
        project_auto_edit(source_path, tmp_path / "auto.wfp", report_path=source_path)


def test_all_rejected_empty_timeline(tmp_path: Path, monkeypatch: Any) -> None:
    source_path = auto_fixture(tmp_path, (1, 2, 2.9))
    install_signals(monkeypatch)
    result = project_auto_edit(source_path, tmp_path / "empty.wfp")
    assert result["kept_count"] == 0
    assert result["final_timeline_duration"] == 0
    project = parse_project(tmp_path / "empty.wfp")
    assert project.active_timeline
    assert all(not t.clips for t in project.active_timeline.tracks)


def test_time_rejected_pair_never_decodes(monkeypatch: Any) -> None:
    decisions = hard_quality_plan([source("a", 3), source("b", 3)])
    decisions[1]["capture_time"] = 281

    def forbidden(*args: Any) -> Any:
        raise AssertionError("Outside-window pair must never decode")

    monkeypatch.setattr("kestrel.auto_edit.sample_source", forbidden)
    assert same_scene_measurements(decisions, "ffmpeg") == []


def test_auto_edit_cli_missing_media_is_conservative(tmp_path: Path) -> None:
    source_path = auto_fixture(tmp_path)
    output, report = tmp_path / "cli.wfp", tmp_path / "cli.json"
    result = subprocess.run(
        [
            "kestrel",
            "project-auto-edit",
            str(source_path),
            str(output),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["kept_count"] == 2
    assert data["summary"]["gain_adjusted"] == 0
    assert data["summary"]["analysis_failures"] == 3
    assert set(data["timings_seconds"]) == {
        "inventory",
        "media_analysis",
        "hard_rule_planning",
        "duplicate_analysis",
        "final_planning",
        "wfp_materialization",
    }


def test_publish_cannot_overwrite_existing_output(tmp_path: Path) -> None:
    staged, output = tmp_path / "staged.bin", tmp_path / "existing.bin"
    staged.write_bytes(b"new")
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        publish_new_file(staged, output)
    assert output.read_bytes() == b"preserve"
