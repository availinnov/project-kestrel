"""Synthetic planning and shared materialization controls."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from kestrel.editplan import ClipEdit, parse_plan_json, snap_boundary, trimmed_range
from kestrel.project.parser import parse_project
from kestrel.project.sequence import apply_plan
from kestrel.project.writer import ProjectWriteError
from tests.test_sequence import fixture, rewrite_fixture


def entry(
    identifier: str, keep: bool = True, start: int = 0, end: int = 0
) -> dict[str, Any]:
    return dict(
        catalog_id=identifier, keep=keep, trim_start_ticks=start, trim_end_ticks=end
    )


def save_plan(tmp_path: Path, clips: list[dict[str, Any]]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(version=1, clips=clips)))
    return path


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 2, "clips": []},
        {"version": True, "clips": []},
        {"version": 1, "clips": {}},
        {"version": 1, "clips": [], "unknown": 1},
        {"version": 1, "clips": [entry("m0"), entry("m0")]},
        {"version": 1, "clips": [{}]},
        *[
            {"version": 1, "clips": [dict(entry("m0"), **{key: value})]}
            for key, value in [
                ("catalog_id", ""),
                ("keep", 1),
                ("trim_start_ticks", -1),
                ("trim_end_ticks", True),
                ("trim_end_ticks", 0.5),
                ("extra", 1),
            ]
        ],
        {"version": 1, "clips": [entry("m0", False, 1)]},
    ],
)
def test_invalid_plan(bad: Any) -> None:
    with pytest.raises(ValueError):
        parse_plan_json(json.dumps(bad))


def test_json_errors() -> None:
    for text in ("{", '{"version":1,"version":1,"clips":[]}'):
        with pytest.raises(ValueError):
            parse_plan_json(text)


@pytest.mark.parametrize(
    ("fps", "ticks", "expected"),
    [
        (Fraction(25), 0, 0),
        (Fraction(25), 199999, 0),
        (Fraction(25), 200000, 400000),
        (Fraction(25), 200001, 400000),
        (Fraction(25), 400000, 400000),
        (Fraction(30), 333333, 333333),
        (Fraction(60), 666667, 666667),
        (Fraction(30000, 1001), 667333, 667333),
    ],
)
def test_grid(fps: Fraction, ticks: int, expected: int) -> None:
    assert snap_boundary(ticks, fps) == expected


@pytest.mark.parametrize(
    ("start", "end"), [(10000000, 0), (0, 10000001), (6000000, 5000000), (9999999, 0)]
)
def test_collapsed_range(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        trimmed_range(10000000, ClipEdit("m0", True, start, end), Fraction(60))


@pytest.mark.parametrize("drops", [[], [0], [1], [2], [0, 2]])
@pytest.mark.parametrize(
    ("start", "end"), [(0, 0), (1000000, 0), (0, 1000000), (1000000, 1000000)]
)
def test_apply(tmp_path: Path, drops: list[int], start: int, end: int) -> None:
    source = fixture(tmp_path)
    plan = save_plan(
        tmp_path,
        [
            entry(
                f"m{i}",
                i not in drops,
                start if i not in drops else 0,
                end if i not in drops else 0,
            )
            for i in reversed(range(3))
        ],
    )
    output = tmp_path / "out.zip"
    report = apply_plan(source, plan, output)
    assert report["kept_source_count"] == 3 - len(drops)
    assert report["dropped_source_count"] == len(drops)
    assert len(report["generated_identity_fields"]) == 5 * (3 - len(drops))
    assert [s["catalog_id"] for s in report["selected_sources"]] == [
        f"m{i}" for i in range(3) if i not in drops
    ]
    before = parse_project(source)
    after = parse_project(output)
    assert before.active_timeline and after.active_timeline
    assert (
        before.raw_documents["catalog.json"]["media_items"]["m1"]
        == after.raw_documents["catalog.json"]["media_items"]["m1"]
    )
    previous = 0
    for selected in report["selected_sources"]:
        assert selected["tl_begin"] == previous
        assert (
            selected["tl_end"] - previous
            == selected["out_point"] - selected["in_point"]
        )
        previous = selected["tl_end"]
        pair = [
            c
            for t in after.active_timeline.tracks
            for c in t.clips
            if c.id in (selected["video_clip_id"], selected["audio_clip_id"])
        ]
        assert len(pair) == 2
        for clip in pair:
            assert (
                clip.in_point == selected["in_point"]
                and clip.out_point == selected["out_point"]
            )
            assert clip.raw["speed"]["offset"] == clip.in_point / 10000000
    assert after.duration == previous
    for i in (1, 3):
        assert (
            before.active_timeline.tracks[i].raw == after.active_timeline.tracks[i].raw
        )


@pytest.mark.parametrize("variant", ["unknown", "omitted", "ineligible", "empty"])
def test_resolution_errors(tmp_path: Path, variant: str) -> None:
    source = fixture(tmp_path)
    clips = [entry(f"m{i}") for i in range(3)]
    if variant == "unknown":
        clips.append(entry("missing"))
    if variant == "omitted":
        clips.pop()
    if variant == "ineligible":
        clips.append(entry("main"))
    if variant == "empty":
        clips = [entry(f"m{i}", False) for i in range(3)]
    with pytest.raises(ProjectWriteError):
        apply_plan(source, save_plan(tmp_path, clips), tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_dropped_catalog_only(tmp_path: Path) -> None:
    source = fixture(tmp_path)

    def change(docs: Any) -> None:
        resources = docs["main/timeline.wesproj"]["resources"]
        resource = resources.pop()
        docs["ProjectFolder/Medias/m2/media.json"] = {
            "file_name": "synthetic/2.mp4",
            "sourceInfo": {
                "basicInfo": {"mediaLength": 30000000, "streamType": 2},
                "vidStreamInfos": resource["vidStreamInfo"],
                "audStreamInfos": resource["audStreamInfo"],
            },
        }

    rewrite_fixture(source, change)
    before = parse_project(source)
    report = apply_plan(
        source,
        save_plan(tmp_path, [entry("m0"), entry("m1"), entry("m2", False)]),
        tmp_path / "out.zip",
    )
    after = parse_project(tmp_path / "out.zip")
    assert [r.raw for r in before.resources] == [r.raw for r in after.resources]
    assert len(report["generated_identity_fields"]) == 10


def test_cli_and_trim_adjustment(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    plan = save_plan(tmp_path, [entry("m0", True, 100000), entry("m1"), entry("m2")])
    result = subprocess.run(
        [
            "kestrel",
            "project-apply-plan",
            str(source),
            str(plan),
            str(tmp_path / "cli.zip"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "validated: True" in result.stdout
    report = apply_plan(source, plan, tmp_path / "out.zip")
    assert report["trim_adjustments"] == [
        {
            "catalog_id": "m0",
            "requested_in": 100000,
            "requested_out": 10000000,
            "actual_in": 166667,
            "actual_out": 10000000,
        }
    ]
    assert report["selected_sources"][0]["in_point"] == 166667
