"""Generated container fixtures for the structured parser."""

import copy
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kestrel.project import ProjectParseError, parse_project, project_summary
from kestrel.project.models import RawObject


def document() -> dict[str, Any]:
    return {
        "currentTimelineId": 1,
        "unknown_project": {"keep": [1, 2]},
        "resources": [
            {
                "sourceUuid": "source-a",
                "filename": "file:/absent/source.bin",
                "mediaLength": 100,
                "unknown_resource": True,
            }
        ],
        "timelineInfos": [
            {
                "timelineId": 1,
                "unknown_timeline": "retain",
                "trackInfos": [
                    {
                        "uuid": "track-a",
                        "trackType": 2,
                        "unknown_track": 9,
                        "clipList": [
                            {
                                "thisUId": "clip-a",
                                "type": 2,
                                "sourceUuid": "source-a",
                                "inPoint": 10,
                                "outPoint": 30,
                                "tlBegin": 5,
                                "tlEnd": 25,
                                "unknown_clip": {"opaque": "data"},
                                "volumeKeyframe": {
                                    "parameter": json.dumps(
                                        {
                                            "keyframeSets": [
                                                {"_time": 0, "_value": 0.5}
                                            ],
                                            "unknown": 4,
                                        }
                                    )
                                },
                                "effectChainList": [
                                    {
                                        "effectList": [
                                            {"id": "audio/effect/volume", "custom": [3]}
                                        ]
                                    }
                                ],
                                "postTransition": {
                                    "thisUId": "transition-a",
                                    "type": 5,
                                    "tlBegin": 20,
                                    "tlEnd": 30,
                                    "unknown": 6,
                                },
                            },
                            {
                                "thisUId": "clip-nested",
                                "type": 7,
                                "timelineId": 2,
                                "inPoint": 0,
                                "outPoint": 10,
                                "tlBegin": 8,
                                "tlEnd": 18,
                            },
                        ],
                    },
                    {"uuid": "track-empty", "trackType": 1, "clipList": []},
                ],
            },
            {
                "timelineId": 2,
                "trackInfos": [
                    {
                        "uuid": "track-b",
                        "trackType": 1,
                        "clipList": [
                            {
                                "thisUId": "clip-b",
                                "type": 4,
                                "inPoint": 0,
                                "outPoint": 10,
                                "tlBegin": 0,
                                "tlEnd": 10,
                            }
                        ],
                    }
                ],
            },
        ],
    }


def container(
    tmp_path: Path,
    doc: dict[str, Any] | None = None,
    extra: dict[str, bytes] | None = None,
) -> Path:
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "Root/Items/main/timeline.data",
            json.dumps(document() if doc is None else doc),
        )
        archive.writestr(
            "Root/project_info.json",
            json.dumps(
                {
                    "project_guid": "synthetic-project",
                    "timeline_mediaId": "main",
                    "project_timeline_duration": 25,
                    "unknown_metadata": [9],
                }
            ),
        )
        archive.writestr("Root/extra.json", '{"opaque":{"a":1}}')
        archive.writestr("Root/unknown.bin", b"\x00\xffraw")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_model_and_raw_preservation(tmp_path: Path) -> None:
    doc = document()
    path = container(tmp_path, doc)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    project = parse_project(path)
    timeline = project.timelines[0]
    track = timeline.tracks[0]
    clip = track.clips[0]
    assert project.metadata["unknown_metadata"] == [9]
    assert project.raw_documents["Root/Items/main/timeline.data"] == doc
    assert project.raw_documents["Root/extra.json"] == {"opaque": {"a": 1}}
    assert project.raw_entries["Root/unknown.bin"] == b"\x00\xffraw"
    assert timeline.raw["unknown_timeline"] == "retain"
    assert track.raw["unknown_track"] == 9
    assert clip.raw["unknown_clip"] == {"opaque": "data"}
    assert clip.resource is project.resources[0]
    assert clip.resource.raw["unknown_resource"] is True
    assert clip.resource.filename == "file:/absent/source.bin"
    assert clip.resource.duration == 100
    assert clip.transitions[0].raw["unknown"] == 6
    assert clip.transitions[0].position == "postTransition"
    assert (
        clip.audio_volume["volumeKeyframe_decoded"]["keyframeSets"][0]["_value"] == 0.5
    )
    assert clip.audio_volume["effects"][0]["custom"] == [3]
    assert track.clips[1].nested_timeline is project.timelines[1]
    assert project.active_timeline is timeline
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == modified
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_summary(tmp_path: Path) -> None:
    result = project_summary(parse_project(container(tmp_path)))
    assert result["timeline_count"] == 2
    assert result["resource_count"] == 1
    assert result["track_count_by_type"] == {"1": 2, "2": 1}
    assert result["clip_count"] == 3
    assert result["total_duration"] == 25  # Overlap and nested content are not summed.
    assert result["time_unit"] == "native"
    assert result["clips_with_trim_values"][0]["in_point"] == 10
    assert len(result["clips_with_transitions"]) == 1
    assert result["nested_timeline_references"][0]["target_timeline_id"] == 2
    assert result["nested_timeline_references"][0]["resolved"]
    assert result["warnings"] == []
    json.dumps(result, allow_nan=False)


def test_unresolved_references_and_volume(tmp_path: Path) -> None:
    doc = document()
    clips = doc["timelineInfos"][0]["trackInfos"][0]["clipList"]
    clips[0]["sourceUuid"] = "missing"
    clips[0]["volumeKeyframe"]["parameter"] = "{broken"
    clips[1]["timelineId"] = 999
    project = parse_project(container(tmp_path, doc))
    assert len(project.warnings) == 3
    assert project.timelines[0].tracks[0].clips[0].resource is None
    assert not project_summary(project)["nested_timeline_references"][0]["resolved"]
    assert (
        project.timelines[0]
        .tracks[0]
        .clips[0]
        .audio_volume["volumeKeyframe"]["parameter"]
        == "{broken"
    )


def test_cyclic_reference_is_not_expanded(tmp_path: Path) -> None:
    doc = document()
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"][1]["timelineId"] = 1
    project = parse_project(container(tmp_path, doc))
    assert (
        project.timelines[0].tracks[0].clips[1].nested_timeline is project.timelines[0]
    )
    assert project_summary(project)["total_duration"] == 25


def test_document_scoped_ids(tmp_path: Path) -> None:
    second = document()
    second["resources"][0]["filename"] = "other.bin"
    project = parse_project(
        container(
            tmp_path,
            extra={"Root/Items/other/timeline.data": json.dumps(second).encode()},
        )
    )
    assert len(project.timelines) == 4
    assert len(project.resources) == 2
    assert project.timelines[2].tracks[0].clips[0].resource is project.resources[1]
    assert (
        project.timelines[2].tracks[0].clips[1].nested_timeline is project.timelines[3]
    )
    assert project.active_timeline is project.timelines[0]


@pytest.mark.parametrize("key", ["sourceUuid", "thisUId", "type"])
def test_invalid_identifier(tmp_path: Path, key: str) -> None:
    doc = document()
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0][key] = True
    with pytest.raises(ProjectParseError, match="identifier"):
        parse_project(container(tmp_path, doc))


@pytest.mark.parametrize("value", ["25", True, [], {}])
def test_invalid_time(tmp_path: Path, value: Any) -> None:
    doc = document()
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["tlEnd"] = value
    with pytest.raises(ProjectParseError, match="numeric time"):
        parse_project(container(tmp_path, doc))


def test_missing_time_is_not_zero(tmp_path: Path) -> None:
    doc = document()
    del doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["tlEnd"]
    project = parse_project(container(tmp_path, doc))
    assert project.duration is None
    assert "incomplete timeline range" in project.warnings[0]


@pytest.mark.parametrize("section", ["resources", "timelineInfos", "tracks", "clips"])
def test_duplicate_ids(tmp_path: Path, section: str) -> None:
    doc = document()
    if section == "tracks":
        items = doc["timelineInfos"][0]["trackInfos"]
    elif section == "clips":
        items = doc["timelineInfos"][0]["trackInfos"][0]["clipList"]
    else:
        items = doc[section]
    items.append(copy.deepcopy(items[0]))
    with pytest.raises(ProjectParseError, match="Duplicate"):
        parse_project(container(tmp_path, doc))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "..\\escape"])
def test_unsafe_entry(tmp_path: Path, name: str) -> None:
    with pytest.raises(ProjectParseError, match="Unsafe"):
        parse_project(container(tmp_path, extra={name: b"content"}))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.zip"]


def test_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kestrel.project.parser.MAX_ARCHIVE_SIZE", 10)
    with pytest.raises(ProjectParseError, match="size limit"):
        parse_project(container(tmp_path))


def test_empty_and_unknown_containers(tmp_path: Path) -> None:
    with pytest.raises(ProjectParseError, match="No supported"):
        parse_project(container(tmp_path, {}))
    doc = {
        "currentTimelineId": 1,
        "timelineInfos": [{"timelineId": 1, "trackInfos": []}],
    }
    project = parse_project(container(tmp_path, doc))
    assert project.duration == 0
    assert project_summary(project)["clip_count"] == 0


def test_missing_active_timeline(tmp_path: Path) -> None:
    doc = document()
    doc["currentTimelineId"] = 999
    project = parse_project(container(tmp_path, doc))
    assert project.duration is None
    assert project.warnings


@pytest.mark.parametrize("machine", [True, False])
def test_cli_summary(tmp_path: Path, machine: bool) -> None:
    path = container(tmp_path)
    result = subprocess.run(
        ["kestrel", "project-summary", str(path), *(["--json"] if machine else [])],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0
    assert result.stderr == ""
    if machine:
        assert json.loads(result.stdout)["clip_count"] == 3
    else:
        assert "timeline_count: 2" in result.stdout


@pytest.mark.parametrize("data", [b"not zip", b"PK\x03\x04broken"])
def test_cli_malformed(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "bad.zip"
    path.write_bytes(data)
    result = subprocess.run(
        ["kestrel", "project-summary", str(path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "error" in json.loads(result.stdout)
    assert "Traceback" not in result.stderr


def test_malformed_document(tmp_path: Path) -> None:
    with pytest.raises(ProjectParseError, match="Invalid structured"):
        parse_project(container(tmp_path, extra={"bad.json": b"{broken"}))


def test_repeated_auxiliary_keys_are_preserved(tmp_path: Path) -> None:
    data = b'{"item":1,"item":2,"nested":{"key":3,"key":4}}'
    project = parse_project(container(tmp_path, extra={"catalog.json": data}))
    raw = project.raw_documents["catalog.json"]
    assert isinstance(raw, RawObject)
    assert raw.pairs[:2] == [("item", 1), ("item", 2)]
    assert raw["nested"].pairs == [("key", 3), ("key", 4)]
    assert project.raw_entries["catalog.json"] == data
    assert len(project.warnings) == 1


def test_repeated_structural_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectParseError, match="Ambiguous repeated"):
        parse_project(
            container(
                tmp_path,
                extra={"ambiguous.json": b'{"timelineInfos":[],"timelineInfos":[]}'},
            )
        )
