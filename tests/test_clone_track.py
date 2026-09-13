"""Synthetic fixtures for conservative track and video-clip creation."""

import copy
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from kestrel.project import parse_project
from kestrel.project.models import Project
from kestrel.project.writer import ProjectWriteError, clone_video_track
from tests.test_writer import trim_source


def clone_source(tmp_path: Path, variant: str = "") -> Path:
    source = trim_source(tmp_path)
    project = parse_project(source)
    assert project.active_timeline is not None
    timeline = project.active_timeline
    track = timeline.tracks[0].raw
    clip = track["clipList"][0]
    track["busUuids"] = ["shared-bus"]
    track["unknown_required"] = {"settings": [1, 2], "opaque": "keep"}
    track["userData"] = [{"key": 21, "size": 4, "data": "AQAAAA=="}]
    clip["effectChainList"] = [
        {
            "name": "Basic",
            "effectList": [
                {
                    "id": "visual/effect/transform",
                    "thisUId": "effect-instance",
                    "paramList": [{"name": "Scale", "fxParam": {"unValue": 1.0}}],
                    "unknown": {"opaque": True},
                },
                {"id": "effect-without-instance-id", "unknown": [1, 2]},
            ],
        }
    ]
    clip["userData"] = [{"key": 13000, "size": 4, "data": "BAAAAA=="}]
    if variant == "duplicate":
        extra = copy.deepcopy(clip)
        extra["thisUId"] = "second-video"
        track["clipList"].append(extra)
    elif variant == "later":
        extra = copy.deepcopy(clip)
        extra.update(thisUId="later-video", tlBegin=200_000_000, tlEnd=300_000_000)
        track["clipList"].append(extra)
    elif variant == "transition":
        clip["postTransition"] = {"thisUId": "transition", "type": 5}
    elif variant == "unresolved":
        clip["sourceUuid"] = "missing"
    elif variant == "nested":
        clip["timelineId"] = 99
    elif variant == "title":
        clip["type"] = 4
    elif variant == "wrong_track":
        track["trackType"] = 2
    elif variant == "no_audio":
        timeline.raw["trackInfos"].pop()
    elif variant == "no_active":
        project.raw_documents[timeline.document]["currentTimelineId"] = 999
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.comment = b"synthetic archive comment"
        for name, data in project.raw_entries.items():
            if name == timeline.document:
                data = (
                    b"\xef\xbb\xbf\n"
                    + json.dumps(project.raw_documents[name], indent=2).encode()
                    + b"\n"
                )
            archive.writestr(name, data)
        archive.writestr("unknown/payload.bin", bytes(range(256)))
    return source


def test_clone_success_and_exact_preservation(tmp_path: Path) -> None:
    source = clone_source(tmp_path)
    original_bytes = source.read_bytes()
    before = parse_project(source)
    output = tmp_path / "cloned.zip"
    report = clone_video_track(source, output)
    after = parse_project(output)
    assert before.active_timeline is not None and after.active_timeline is not None
    old, new = before.active_timeline, after.active_timeline
    assert report["validated"]
    assert report["reused_existing_track"] is False
    assert report["target_track_id"] == new.tracks[-1].id
    assert (report["original_track_count"], report["generated_track_count"]) == (2, 3)
    assert len(new.tracks) == len(old.tracks) + 1
    assert [t.raw for t in new.tracks[:-1]] == [t.raw for t in old.tracks]
    clone = new.tracks[-1].clips[0]
    original = old.tracks[0].clips[0]
    assert new.tracks[-1].type == 1
    assert len(new.tracks[-1].clips) == 1
    assert clone.id != original.id
    assert new.tracks[-1].id not in {t.id for t in old.tracks}
    assert UUID(str(clone.id)).version == 4
    assert UUID(str(new.tracks[-1].id)).version == 4
    assert clone.resource is not None
    assert clone.resource is after.resources[0]
    assert clone.source_id == original.source_id
    assert clone.raw["enable"] is False
    assert clone.color_tag == 4
    assert clone.raw["speed"] == original.raw["speed"]
    assert (clone.in_point, clone.out_point, clone.begin, clone.end) == (
        original.in_point,
        original.out_point,
        original.begin,
        original.end,
    )
    assert sum(c.type == 2 for t in new.tracks for c in t.clips) == 1
    assert len(after.resources) == len(before.resources)
    assert [r.raw for r in after.resources] == [r.raw for r in before.resources]
    assert after.duration == before.duration
    assert after.metadata == before.metadata
    expected_track = copy.deepcopy(old.tracks[0].raw)
    expected_track["uuid"] = new.tracks[-1].id
    expected_track["clipList"] = [copy.deepcopy(original.raw)]
    expected_clip = expected_track["clipList"][0]
    expected_clip.update(thisUId=clone.id, enable=False)
    expected_clip["effectChainList"][0]["effectList"][0]["thisUId"] = clone.raw[
        "effectChainList"
    ][0]["effectList"][0]["thisUId"]
    assert new.tracks[-1].raw == expected_track
    generated = report["generated_identity_fields"]
    assert {item["field"] for item in generated} == {
        "track.uuid",
        "clip.thisUId",
        "clip.effectChainList[0].effectList[0].thisUId",
    }
    assert len({item["generated"] for item in generated}) == 3
    assert all(item["original"] != item["generated"] for item in generated)
    # One inserted JSON track; every pre-existing document byte survives unchanged.
    insertion = (
        "," + json.dumps(new.tracks[-1].raw, ensure_ascii=True, allow_nan=False)
    ).encode()
    assert after.raw_entries[new.document].count(insertion) == 1
    assert (
        after.raw_entries[new.document].replace(insertion, b"", 1)
        == before.raw_entries[old.document]
    )
    for name, data in before.raw_entries.items():
        if name != old.document:
            assert after.raw_entries[name] == data
    assert before.raw_entries.keys() == after.raw_entries.keys()
    assert source.read_bytes() == original_bytes
    assert report["source_uuid_equal"] and report["timeline_source_range_equal"]


@pytest.mark.parametrize(
    "variant",
    [
        "duplicate",
        "later",
        "transition",
        "unresolved",
        "nested",
        "title",
        "wrong_track",
        "no_audio",
        "no_active",
    ],
)
def test_clone_rejects_unsupported_input(tmp_path: Path, variant: str) -> None:
    source = clone_source(tmp_path, variant)
    original = source.read_bytes()
    output = tmp_path / "rejected.zip"
    with pytest.raises(ProjectWriteError):
        clone_video_track(source, output)
    assert not output.exists()
    assert source.read_bytes() == original


def test_clone_refuses_overwrite(tmp_path: Path) -> None:
    source = clone_source(tmp_path)
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"keep")
    for output in (source, existing):
        with pytest.raises(ProjectWriteError, match="overwriting"):
            clone_video_track(source, output)
    assert existing.read_bytes() == b"keep"


def test_clone_validation_failure_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = clone_source(tmp_path)
    output = tmp_path / "failed.zip"

    def invalid_clone(path: str | Path) -> Project:
        project = parse_project(path)
        if Path(path) != source:
            assert project.active_timeline is not None
            project.active_timeline.tracks[-1].clips[0].raw["enable"] = True
        return project

    monkeypatch.setattr("kestrel.project.writer.parse_project", invalid_clone)
    with pytest.raises(ProjectWriteError, match="validation"):
        clone_video_track(source, output)
    assert [p.name for p in tmp_path.iterdir()] == [source.name]


def test_clone_cli(tmp_path: Path) -> None:
    source = clone_source(tmp_path)
    output = tmp_path / "clone.zip"
    result = subprocess.run(
        ["kestrel", "project-clone-video-track", str(source), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "original_track_count: 2" in result.stdout
    assert "generated_track_count: 3" in result.stdout
    assert "validated: True" in result.stdout


def destination_track(
    identity: str, interval: tuple[int, int] | None = None, unsupported: str = ""
) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    if interval is not None:
        begin, end = interval
        clip: dict[str, Any] = {
            "thisUId": f"existing-{identity}",
            "type": 1,
            "sourceUuid": "source-a",
            "inPoint": 0,
            "outPoint": max(end - begin, 1),
            "tlBegin": begin,
            "tlEnd": end,
            "unknown": {"retain": [3, 4]},
        }
        if unsupported == "transition":
            clip["postTransition"] = {"type": 5}
        elif unsupported == "nested":
            clip["type"] = 7
            clip["timelineId"] = 999
        elif unsupported == "missing_end":
            del clip["tlEnd"]
        clips.append(clip)
    return {
        "uuid": identity,
        "trackType": 1,
        "clipList": clips,
        "unknown_track": {"settings": [9]},
        "busUuids": ["existing-bus"],
    }


def source_with_destinations(tmp_path: Path, tracks: list[dict[str, Any]]) -> Path:
    source = clone_source(tmp_path)
    project = parse_project(source)
    assert project.active_timeline is not None
    timeline = project.active_timeline
    timeline.raw["trackInfos"].extend(tracks)
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in project.raw_entries.items():
            if name == timeline.document:
                data = json.dumps(project.raw_documents[name], indent=2).encode()
            archive.writestr(name, data)
    return source


@pytest.mark.parametrize(
    "interval",
    [None, (-100_000_000, 0), (100_000_000, 200_000_000), (150_000_000, 250_000_000)],
)
def test_reuse_compatible_track(
    tmp_path: Path, interval: tuple[int, int] | None
) -> None:
    source = source_with_destinations(
        tmp_path, [destination_track("reuse-me", interval)]
    )
    before = parse_project(source)
    original = source.read_bytes()
    output = tmp_path / "reused.zip"
    report = clone_video_track(source, output)
    after = parse_project(output)
    assert before.active_timeline is not None and after.active_timeline is not None
    old, new = before.active_timeline, after.active_timeline
    assert report["reused_existing_track"] is True
    assert report["target_track_id"] == "reuse-me"
    assert report["original_track_count"] == report["generated_track_count"] == 3
    assert [t.raw for t in new.tracks[:2]] == [t.raw for t in old.tracks[:2]]
    assert new.tracks[2].raw == {
        **old.tracks[2].raw,
        "clipList": [*old.tracks[2].raw["clipList"], new.tracks[2].clips[-1].raw],
    }
    clone = new.tracks[2].clips[-1]
    assert clone.raw["enable"] is False
    assert clone.source_id == old.tracks[0].clips[0].source_id
    assert (clone.begin, clone.end, clone.in_point, clone.out_point) == (
        0,
        100_000_000,
        20_000_000,
        120_000_000,
    )
    assert all(
        item["field"] != "track.uuid" for item in report["generated_identity_fields"]
    )
    assert sum(c.id == clone.id for t in new.tracks for c in t.clips) == 1
    assert sum(c.type == 2 for t in new.tracks for c in t.clips) == 1
    insertion = (
        ("," if old.tracks[2].clips else "")
        + json.dumps(clone.raw, ensure_ascii=True, allow_nan=False)
    ).encode()
    assert (
        after.raw_entries[new.document].replace(insertion, b"", 1)
        == before.raw_entries[old.document]
    )
    for name, data in before.raw_entries.items():
        if name != old.document:
            assert after.raw_entries[name] == data
    assert source.read_bytes() == original


@pytest.mark.parametrize("alternative", [False, True])
@pytest.mark.parametrize(
    "interval", [(0, 50_000_000), (50_000_000, 150_000_000), (-1, 100_000_001)]
)
def test_overlap_uses_alternative_or_fallback(
    tmp_path: Path, alternative: bool, interval: tuple[int, int]
) -> None:
    tracks = [destination_track("occupied", interval)]
    if alternative:
        tracks.append(destination_track("available"))
    source = source_with_destinations(tmp_path, tracks)
    output = tmp_path / "result.zip"
    before = parse_project(source)
    report = clone_video_track(source, output)
    after = parse_project(output)
    assert report["reused_existing_track"] is alternative
    assert report["target_track_id"] != "occupied"
    if alternative:
        assert report["target_track_id"] == "available"
    assert report["generated_track_count"] == report["original_track_count"] + (
        not alternative
    )
    assert before.active_timeline is not None and after.active_timeline is not None
    assert before.active_timeline.tracks[2].raw == after.active_timeline.tracks[2].raw


def test_reuse_follows_timeline_order(tmp_path: Path) -> None:
    source = source_with_destinations(
        tmp_path, [destination_track("z-first"), destination_track("a-second")]
    )
    output = tmp_path / "result.zip"
    report = clone_video_track(source, output)
    assert report["target_track_id"] == "z-first"
    assert report["original_track_count"] == report["generated_track_count"] == 4
    before, after = parse_project(source), parse_project(output)
    assert before.active_timeline is not None and after.active_timeline is not None
    assert before.active_timeline.tracks[3].raw == after.active_timeline.tracks[3].raw


@pytest.mark.parametrize("unsupported", ["transition", "nested", "missing_end"])
def test_unsupported_destination_is_skipped(tmp_path: Path, unsupported: str) -> None:
    source = source_with_destinations(
        tmp_path,
        [
            destination_track("unsupported", (150_000_000, 200_000_000), unsupported),
            destination_track("available"),
        ],
    )
    report = clone_video_track(source, tmp_path / "result.zip")
    assert report["reused_existing_track"] is True
    assert report["target_track_id"] == "available"
