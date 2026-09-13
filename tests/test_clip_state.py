"""Synthetic pair state and stored numeric tag writer regressions."""

import base64
import copy
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kestrel.project import parse_project
from kestrel.project.writer import ProjectWriteError, set_clip_state
from tests.test_writer import trim_source


def state_source(
    tmp_path: Path, entries: list[dict[str, Any]] | None = None, variant: str = ""
) -> Path:
    path = trim_source(tmp_path)
    project = parse_project(path)
    assert project.active_timeline is not None
    timeline = project.active_timeline
    video = timeline.tracks[0].clips[0].raw
    audio = timeline.tracks[1].clips[0].raw
    if entries is not None:
        video["userData"] = entries
    audio["userData"] = [{"key": 13000, "size": 4, "data": "AgAAAA=="}]
    if variant == "disabled":
        video["enable"] = audio["enable"] = False
    elif variant == "duplicate":
        extra = copy.deepcopy(video)
        extra["thisUId"] = "duplicate"
        timeline.tracks[0].raw["clipList"].append(extra)
    elif variant == "mismatch":
        audio["inPoint"] += 1
    elif variant == "later":
        for track in timeline.tracks:
            extra = copy.deepcopy(track.clips[0].raw)
            extra.update(
                thisUId=f"later-{track.id}", tlBegin=200_000_000, tlEnd=300_000_000
            )
            track.raw["clipList"].insert(0, extra)
        timeline.tracks[0].raw["clipList"].append(
            {
                "thisUId": "title",
                "type": 4,
                "inPoint": 0,
                "outPoint": 1,
                "tlBegin": 0,
                "tlEnd": 1,
            }
        )
        timeline.tracks[0].raw["clipList"].append(
            {
                "thisUId": "nested",
                "type": 7,
                "timelineId": 99,
                "inPoint": 0,
                "outPoint": 1,
                "tlBegin": 0,
                "tlEnd": 1,
            }
        )
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in project.raw_entries.items():
            if name == timeline.document:
                data = json.dumps(project.raw_documents[name], indent=2).encode()
            archive.writestr(name, data)
    return path


@pytest.mark.parametrize("enable", [False, True])
def test_state_pair(tmp_path: Path, enable: bool) -> None:
    source = state_source(tmp_path, variant="disabled" if enable else "")
    original = source.read_bytes()
    output = tmp_path / "output.zip"
    set_clip_state(source, output, enable=enable)
    before, after = parse_project(source), parse_project(output)
    assert after.active_timeline is not None and before.active_timeline is not None
    for old, new in zip(
        before.active_timeline.tracks, after.active_timeline.tracks, strict=True
    ):
        expected = {**old.clips[0].raw, "enable": enable}
        assert new.clips[0].raw == expected
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "entries",
    [
        None,
        [],
        [{"key": 7, "data": "opaque"}],
        [
            {"key": 7, "data": "opaque"},
            {"key": 13000, "size": 4, "data": "AgAAAA==", "unknown": 9},
            {"key": 80, "data": "other"},
        ],
        [{"key": 13000, "unknown": 9}],
    ],
)
@pytest.mark.parametrize("combined", [False, True])
def test_numeric_tag_and_preservation(
    tmp_path: Path, entries: Any, combined: bool
) -> None:
    source = state_source(tmp_path, entries)
    original = source.read_bytes()
    output = tmp_path / "output.zip"
    set_clip_state(source, output, color_tag=4, enable=False if combined else None)
    before, after = parse_project(source), parse_project(output)
    assert before.active_timeline is not None and after.active_timeline is not None
    video, audio = [t.clips[0] for t in after.active_timeline.tracks]
    assert video.color_tag == 4
    assert audio.color_tag == 2
    expected_entries = copy.deepcopy(entries or [])
    expected = next((e for e in expected_entries if e["key"] == 13000), None)
    if expected is None:
        expected = {"key": 13000}
        expected_entries.append(expected)
    expected.update(size=4, data="BAAAAA==")
    assert video.raw["userData"] == expected_entries
    assert (
        int.from_bytes(base64.b64decode(expected["data"]), "little", signed=True) == 4
    )
    for old, new in zip(
        before.active_timeline.tracks, after.active_timeline.tracks, strict=True
    ):
        expected_raw = copy.deepcopy(old.clips[0].raw)
        if new.clips[0].type == 1:
            expected_raw["userData"] = expected_entries
        if combined:
            expected_raw["enable"] = False
        assert new.clips[0].raw == expected_raw
    for name, data in before.raw_entries.items():
        if name != before.active_timeline.document:
            assert after.raw_entries[name] == data
    assert source.read_bytes() == original


def test_first_pair_only_and_skip_non_source_clips(tmp_path: Path) -> None:
    source = state_source(tmp_path, variant="later")
    output = tmp_path / "output.zip"
    set_clip_state(source, output, enable=False, color_tag=4)
    a, b = parse_project(source), parse_project(output)
    assert a.active_timeline is not None and b.active_timeline is not None
    for ta, tb in zip(a.active_timeline.tracks, b.active_timeline.tracks, strict=True):
        for old, new in zip(ta.clips, tb.clips, strict=True):
            if old.id in {"clip-1", "clip-2"}:
                assert new.raw["enable"] is False
            else:
                assert old.raw == new.raw


@pytest.mark.parametrize("tag", [0, 14, True, 4.0])
def test_invalid_tag(tmp_path: Path, tag: Any) -> None:
    source = state_source(tmp_path)
    with pytest.raises(ProjectWriteError):
        set_clip_state(source, tmp_path / "output.zip", color_tag=tag)
    assert not (tmp_path / "output.zip").exists()


@pytest.mark.parametrize("variant", ["duplicate", "mismatch"])
def test_ambiguous_pair(tmp_path: Path, variant: str) -> None:
    source = state_source(tmp_path, variant=variant)
    with pytest.raises(ProjectWriteError, match="ambiguous"):
        set_clip_state(source, tmp_path / "output.zip", enable=False)
    assert not (tmp_path / "output.zip").exists()


def test_duplicate_tags_rejected(tmp_path: Path) -> None:
    source = state_source(tmp_path, [{"key": 13000}, {"key": 13000}])
    with pytest.raises(ProjectWriteError, match="Duplicate"):
        set_clip_state(source, tmp_path / "output.zip", color_tag=4)


def test_insertion_preserves_original_json_text(tmp_path: Path) -> None:
    source = state_source(tmp_path, [{"key": 7, "data": "opaque"}])
    output = tmp_path / "output.zip"
    set_clip_state(source, output, color_tag=4)
    before, after = parse_project(source), parse_project(output)
    assert before.active_timeline is not None
    name = before.active_timeline.document
    insertion = b',{"key": 13000, "size": 4, "data": "BAAAAA=="}'
    assert after.raw_entries[name].count(insertion) == 1
    assert (
        after.raw_entries[name].replace(insertion, b"", 1) == before.raw_entries[name]
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--disable"],
        ["--enable"],
        ["--color-tag", "4"],
        ["--disable", "--color-tag", "4"],
    ],
)
def test_state_cli(tmp_path: Path, options: list[str]) -> None:
    source = state_source(tmp_path)
    result = subprocess.run(
        [
            "kestrel",
            "project-set-clip-state",
            str(source),
            str(tmp_path / "output.zip"),
            *options,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated: True" in result.stdout


def test_no_options_and_overwrite_rejected(tmp_path: Path) -> None:
    source = state_source(tmp_path)
    original = source.read_bytes()
    with pytest.raises(ProjectWriteError, match="Specify"):
        set_clip_state(source, tmp_path / "unused.zip")
    with pytest.raises(ProjectWriteError, match="overwriting"):
        set_clip_state(source, source, enable=False)
    assert source.read_bytes() == original
