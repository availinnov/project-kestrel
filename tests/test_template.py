"""Entirely synthetic empty templates and identity-graph regression tests."""

import base64
import copy
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from kestrel.project import ProjectParseError, parse_project
from kestrel.project.parser import build_project
from kestrel.project.semantic import semantic_diff, semantic_diff_files
from kestrel.project.template import (
    CATALOG,
    INFO,
    discover_empty_template,
    instantiate_template,
)
from kestrel.project.writer import ProjectWriteError

PROJECT_ID = "{11111111-1111-4111-8111-111111111111}"
MEDIA_ID = "{22222222-2222-4222-8222-222222222222}"
TIMELINE_ID = "{33333333-3333-4333-8333-333333333333}"
INSTANCE_ID = "{55555555-5555-4555-8555-555555555555}"
DIRECTORY = f"ProjectFolder/Medias/{MEDIA_ID}/"
DOCUMENT = DIRECTORY + "timeline.wesproj"


def template_documents() -> dict[str, Any]:
    return {
        INFO: {
            "project_guid": PROJECT_ID,
            "timeline_mediaId": MEDIA_ID,
            "project_file_name": "synthetic",
            "project_source": "a" * 32,
            "project_date_create": 1_600_000_000,
            "project_date_modify": 1_600_000_001,
            "project_timeline_duration": 0,
            "project_timeline_resolution": [1920, 1080],
            "project_timeline_framerate": [60, 1],
            "project_timeline_ratio": [16, 9],
            "project_color_space": 0,
            "project_sample_rate": 44100,
            "proj_zip_save_path": "C:/synthetic/template.zip",
            "proj_cover_proj_path": (
                f"C:/synthetic/backup/{PROJECT_ID}/Cover\\cover.bin"
            ),
            "unknown_metadata": {"stable_asset_id": "retain"},
        },
        CATALOG: {
            "media_structure": {"media_item": MEDIA_ID, "Folder": {"visible": "true"}},
            "media_items": {
                MEDIA_ID: {
                    "id": MEDIA_ID,
                    "timeline_uuid": TIMELINE_ID,
                    "media_type": 1048576,
                    "name": "Keep timeline label",
                    "duration": 0,
                    "create_time": 1_600_000_000,
                    "mark_info_list": [{"mark_in": -1, "mark_out": -1}],
                    "enable_modify_mediaId": 0,
                    "unknown": [1, 2, 3],
                }
            },
            "unknown_catalog": {"retain": 7},
        },
        DOCUMENT: {
            "resources": None,
            "currentTimelineId": 7,
            "serialNumber": 8,
            "projectName": "Stable internal label",
            "timelineInfos": [
                {
                    "timelineId": 7,
                    "resolutionWidth": 1920,
                    "resolutionHeight": 1080,
                    "frameRate": {"num": 60, "den": 1},
                    "aspectRatioX": 16,
                    "aspectRatioY": 9,
                    "audioBusInfos": [
                        {"busUid": "44444444-4444-4444-8444-444444444444"}
                    ],
                    "trackInfos": [
                        {
                            "uuid": "track-a",
                            "trackType": 2,
                            "clipList": [],
                            "busUuids": ["44444444-4444-4444-8444-444444444444"],
                        },
                        {
                            "uuid": "track-v",
                            "trackType": 1,
                            "clipList": [],
                            "unknown": "keep",
                        },
                    ],
                    "userData": [
                        {
                            "key": 50,
                            "size": 6,
                            "data": base64.b64encode(b"label\0").decode(),
                        },
                        {
                            "key": 11000,
                            "size": 64,
                            "data": base64.b64encode(
                                TIMELINE_ID.encode() + bytes(26)
                            ).decode(),
                            "unknown": 8,
                        },
                        {
                            "key": 30309,
                            "size": 38,
                            "data": base64.b64encode(TIMELINE_ID.encode()).decode(),
                        },
                        # Unknown payloads must not be swept up by global replacement.
                        {
                            "key": 999,
                            "size": 38,
                            "data": base64.b64encode(TIMELINE_ID.encode()).decode(),
                        },
                        {
                            "key": 3,
                            "size": 64,
                            "data": base64.b64encode(
                                INSTANCE_ID.encode() + bytes(26)
                            ).decode(),
                            "unknown": "preserve",
                        },
                        {
                            "key": 140,
                            "size": 32,
                            "data": base64.b64encode(b"b" * 32).decode(),
                        },
                    ],
                    "unknown_setting": {"retain": True},
                }
            ],
        },
        DIRECTORY + "extra.json": {"opaque": [1, 2]},
    }


def write_template(
    tmp_path: Path, documents: dict[str, Any] | None = None, thumbnail: bool = True
) -> Path:
    source = tmp_path / "template.zip"
    data = template_documents() if documents is None else documents
    with zipfile.ZipFile(source, "w") as archive:
        archive.comment = b"synthetic archive comment"
        entries = {
            name: ("\ufeff\n" + json.dumps(value, indent=2) + "\n").encode()
            for name, value in data.items()
        }
        entries[DIRECTORY] = b""
        entries[DIRECTORY + "unknown/bytes.bin"] = bytes(range(256))
        entries["unrelated/data.bin"] = b"\x00\xffopaque"
        entries["unrelated/repeated.json"] = b'{"key":1,"key":2}'
        if thumbnail:
            entries[DIRECTORY + "thumbnail.png"] = b"synthetic thumbnail bytes"
        for name, payload in entries.items():
            info = zipfile.ZipInfo(name, (2020, 1, 2, 3, 4, 6))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.comment = b"entry-comment"
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
    return source


@pytest.mark.parametrize("resources", [None, []])
def test_empty_parser_and_semantic(resources: Any) -> None:
    docs = template_documents()
    docs[DOCUMENT]["resources"] = resources
    project = build_project({name: json.dumps(v).encode() for name, v in docs.items()})
    assert project.resources == []
    assert project.active_timeline is not None
    assert len(project.active_timeline.tracks) == 2
    assert project.duration == 0
    assert project.active_timeline.raw["frameRate"]["num"] == 60
    assert semantic_diff(project, project)["equal"]
    del docs[DOCUMENT]["resources"]
    missing = build_project({name: json.dumps(v).encode() for name, v in docs.items()})
    assert missing.resources == []
    assert semantic_diff(project, missing)["equal"]


@pytest.mark.parametrize("resources", [{}, {"unexpected": 1}, [42], "", False])
def test_malformed_resources_still_rejected(resources: Any) -> None:
    docs = template_documents()
    docs[DOCUMENT]["resources"] = resources
    with pytest.raises(ProjectParseError):
        build_project({name: json.dumps(v).encode() for name, v in docs.items()})


def test_null_resources_with_clips_rejected() -> None:
    docs = template_documents()
    docs[DOCUMENT]["timelineInfos"][0]["trackInfos"][0]["clipList"] = [
        {"thisUId": "clip"}
    ]
    with pytest.raises(ProjectParseError, match="Null resources"):
        build_project({name: json.dumps(v).encode() for name, v in docs.items()})


@pytest.mark.parametrize(
    ("width", "height", "fps"),
    [(None, None, None), (3840, 2160, 60), (None, None, 25), (3840, 2160, 25)],
)
@pytest.mark.parametrize("thumbnail", [False, True])
def test_instantiate_and_preserve(
    tmp_path: Path,
    width: int | None,
    height: int | None,
    fps: int | None,
    thumbnail: bool,
) -> None:
    source = write_template(tmp_path, thumbnail=thumbnail)
    before_bytes = source.read_bytes()
    before = parse_project(source)
    output = tmp_path / "fresh.zip"
    start = int(time.time())
    report = instantiate_template(
        source, output, name="fresh", width=width, height=height, fps=fps
    )
    after = parse_project(output)
    old, new = discover_empty_template(before), discover_empty_template(after)
    assert report["validated"]
    old_ids = {UUID(PROJECT_ID), UUID(MEDIA_ID), UUID(TIMELINE_ID)}
    new_ids = {UUID(new.project_guid), UUID(new.media_id), UUID(new.timeline_uuid)}
    assert not old_ids & new_ids
    assert len(new_ids) == 3
    assert all(uid.version == 4 for uid in new_ids)
    assert new.info["project_file_name"] == "fresh"
    assert new.info["project_source"] != old.info["project_source"]
    assert len(new.info["project_source"]) == 32
    int(new.info["project_source"], 16)
    assert start <= new.info["project_date_create"] <= int(time.time())
    assert (
        new.info["project_date_modify"]
        == new.item["create_time"]
        == new.info["project_date_create"]
    )
    assert new.info["proj_zip_save_path"] == output.resolve().as_posix()
    assert PROJECT_ID not in new.info["proj_cover_proj_path"]
    assert new.project_guid in new.info["proj_cover_proj_path"]
    assert new.info["proj_cover_proj_path"].endswith("/Cover\\cover.bin")
    assert new.info["project_timeline_resolution"] == [width or 1920, height or 1080]
    assert new.info["project_timeline_framerate"] == [fps or 60, 1]
    assert new.timeline.raw["frameRate"] == {"num": fps or 60, "den": 1}
    assert set(new.catalog["media_items"]) == {new.media_id}
    assert (
        new.item["id"] == new.catalog["media_structure"]["media_item"] == new.media_id
    )
    for entry, size in zip(new.uuid_entries, (64, 38), strict=True):
        assert base64.b64decode(entry["data"]) == new.timeline_uuid.encode() + bytes(
            size - 38
        )
        assert entry["size"] == size
    assert new.timeline.raw["userData"][0] == old.timeline.raw["userData"][0]
    assert new.timeline.raw["userData"][3] == old.timeline.raw["userData"][3]
    expected_tracks = copy.deepcopy([t.raw for t in old.timeline.tracks])
    expected_tracks[0]["busUuids"] = [new.bus_uuid]
    assert [t.raw for t in new.timeline.tracks] == expected_tracks
    assert after.resources == [] and after.duration == 0
    assert new.item == {
        **old.item,
        "id": new.media_id,
        "timeline_uuid": new.timeline_uuid,
        "create_time": new.item["create_time"],
    }
    assert new.info["project_timeline_ratio"] == old.info["project_timeline_ratio"]
    assert new.timeline.raw["audioBusInfos"] == [{"busUid": new.bus_uuid}]
    assert new.bus_uuid != old.bus_uuid
    assert UUID(new.bus_uuid).version == 4
    assert new.instance_uuid != old.instance_uuid
    assert base64.b64decode(
        new.instance_entry["data"], validate=True
    ) == new.instance_uuid.encode() + bytes(26)
    assert new.instance_entry == {
        **old.instance_entry,
        "data": new.instance_entry["data"],
    }
    assert new.token != old.token
    assert (
        base64.b64decode(new.token_entry["data"], validate=True) == new.token.encode()
    )
    assert len(new.token) == 32
    int(new.token, 16)
    assert after.raw_documents[new.timeline.document]["serialNumber"] == 8
    assert new.timeline.raw["timelineId"] == 7
    renamed = {
        entry["original"]: entry["generated"] for entry in report["renamed_entries"]
    }
    assert set(renamed) == {n for n in before.raw_entries if n.startswith(DIRECTORY)}
    assert (new.directory + "thumbnail.png" in after.raw_entries) is thumbnail
    modified = set(report["modified_entries"])
    with zipfile.ZipFile(source) as a, zipfile.ZipFile(output) as b:
        assert a.comment == b.comment
        for ai, bi in zip(a.infolist(), b.infolist(), strict=True):
            assert bi.filename == renamed.get(ai.filename, ai.filename)
            assert (ai.date_time, ai.comment, ai.compress_type, ai.external_attr) == (
                bi.date_time,
                bi.comment,
                bi.compress_type,
                bi.external_attr,
            )
            if bi.filename not in modified:
                assert a.read(ai) == b.read(bi)
    assert set(after.raw_entries) == {renamed.get(n, n) for n in before.raw_entries}
    assert not any(n.startswith(DIRECTORY) for n in after.raw_entries)
    assert source.read_bytes() == before_bytes
    assert set(p.name for p in tmp_path.iterdir()) == {"template.zip", "fresh.zip"}


@pytest.mark.parametrize(
    "variant",
    [
        "wrong_item_id",
        "wrong_structure",
        "missing_media",
        "multiple_media",
        "payload_mismatch",
        "payload_padding",
        "payload_size",
        "missing_payload",
        "duplicate_payload",
        "extra_timeline",
        "duration",
        "wrong_settings",
        "bad_project_source",
        "missing_extra",
    ],
)
def test_invalid_graph_rejected(tmp_path: Path, variant: str) -> None:
    docs = template_documents()
    info, catalog = docs[INFO], docs[CATALOG]
    timeline = docs[DOCUMENT]["timelineInfos"][0]
    if variant == "wrong_item_id":
        catalog["media_items"][MEDIA_ID]["id"] = PROJECT_ID
    elif variant == "wrong_structure":
        catalog["media_structure"]["media_item"] = PROJECT_ID
    elif variant == "missing_media":
        info["timeline_mediaId"] = PROJECT_ID
    elif variant == "multiple_media":
        catalog["media_items"][PROJECT_ID] = copy.deepcopy(
            catalog["media_items"][MEDIA_ID]
        )
    elif variant == "payload_mismatch":
        timeline["userData"][1]["data"] = base64.b64encode(
            PROJECT_ID.encode() + bytes(26)
        ).decode()
    elif variant == "payload_padding":
        timeline["userData"][1]["data"] = base64.b64encode(
            TIMELINE_ID.encode() + b"x" * 26
        ).decode()
    elif variant == "payload_size":
        timeline["userData"][1]["size"] = 63
    elif variant == "missing_payload":
        timeline["userData"].pop(2)
    elif variant == "duplicate_payload":
        timeline["userData"].append(copy.deepcopy(timeline["userData"][1]))
    elif variant == "extra_timeline":
        other = copy.deepcopy(timeline)
        other["timelineId"] = 99
        docs[DOCUMENT]["timelineInfos"].append(other)
    elif variant == "duration":
        catalog["media_items"][MEDIA_ID]["duration"] = 1
    elif variant == "wrong_settings":
        timeline["resolutionWidth"] = 3840
    elif variant == "bad_project_source":
        info["project_source"] = "unknown"
    elif variant == "missing_extra":
        del docs[DIRECTORY + "extra.json"]
    source = write_template(tmp_path, docs)
    with pytest.raises((ProjectWriteError, ProjectParseError)):
        instantiate_template(source, tmp_path / "rejected.zip", name="rejected")
    assert not (tmp_path / "rejected.zip").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"width": 3840},
        {"height": 2160},
        {"width": 0, "height": 2160},
        {"width": 1920, "height": -1},
        {"fps": 0},
        {"fps": -1},
        {"fps": 25.5},
        {"fps": True},
    ],
)
def test_invalid_settings(tmp_path: Path, options: dict[str, Any]) -> None:
    source = write_template(tmp_path)
    with pytest.raises(ProjectWriteError):
        instantiate_template(source, tmp_path / "rejected.zip", name="bad", **options)
    assert not (tmp_path / "rejected.zip").exists()


def test_overwrite_and_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_template(tmp_path)
    original = source.read_bytes()
    existing = tmp_path / "existing.zip"
    existing.write_bytes(b"preserve")
    for output in (source, existing):
        with pytest.raises(ProjectWriteError, match="overwriting"):
            instantiate_template(source, output, name="new")
    assert source.read_bytes() == original and existing.read_bytes() == b"preserve"
    discovery = discover_empty_template
    calls = 0

    def fail_validation(project: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ProjectWriteError("Injected post-write validation failure")
        return discovery(project)

    monkeypatch.setattr(
        "kestrel.project.template.discover_empty_template", fail_validation
    )
    with pytest.raises(ProjectWriteError, match="validation"):
        instantiate_template(source, tmp_path / "failed.zip", name="new")
    assert {p.name for p in tmp_path.iterdir()} == {"template.zip", "existing.zip"}


def test_template_cli_and_semantic_diff(tmp_path: Path) -> None:
    source = write_template(tmp_path)
    output = tmp_path / "generated.zip"
    result = subprocess.run(
        [
            "kestrel",
            "project-instantiate-template",
            str(source),
            str(output),
            "--name",
            "generated",
            "--width",
            "3840",
            "--height",
            "2160",
            "--fps",
            "25",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated: True" in result.stdout
    assert "generated_timeline_uuid:" in result.stdout
    assert not semantic_diff_files(source, output)["equal"]
    diff = subprocess.run(
        ["kestrel", "diff", str(source), str(output), "--semantic", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(diff.stdout)["change_count"] > 0


def test_denominator_and_backslash_path_style(tmp_path: Path) -> None:
    docs = template_documents()
    docs[INFO]["proj_zip_save_path"] = "C:\\synthetic\\template.zip"
    docs[INFO]["project_timeline_framerate"][1] = 1001
    docs[DOCUMENT]["timelineInfos"][0]["frameRate"]["den"] = 1001
    source = write_template(tmp_path, docs)
    output = tmp_path / "generated.zip"
    instantiate_template(source, output, name="new", fps=25)
    graph = discover_empty_template(parse_project(output))
    assert graph.info["project_timeline_framerate"] == [25, 1001]
    assert graph.timeline.raw["frameRate"] == {"num": 25, "den": 1001}
    assert graph.info["proj_zip_save_path"] == str(output.resolve()).replace("/", "\\")


def test_repeated_instantiation_generates_independent_graphs(tmp_path: Path) -> None:
    source = write_template(tmp_path)
    output_a, output_b = tmp_path / "a.zip", tmp_path / "b.zip"
    instantiate_template(source, output_a, name="a")
    instantiate_template(output_a, output_b, name="b")
    first = discover_empty_template(parse_project(output_a))
    second = discover_empty_template(parse_project(output_b))
    assert {first.project_guid, first.media_id, first.timeline_uuid}.isdisjoint(
        {second.project_guid, second.media_id, second.timeline_uuid}
    )
    assert first.info["project_source"] != second.info["project_source"]
    assert first.project_guid not in second.info["proj_cover_proj_path"]
    assert first.bus_uuid != second.bus_uuid
    assert first.instance_uuid != second.instance_uuid
    assert first.token != second.token


def test_bus_links_and_unrelated_identities_preserved(tmp_path: Path) -> None:
    docs = template_documents()
    timeline = docs[DOCUMENT]["timelineInfos"][0]
    other_bus = "66666666-6666-4666-8666-666666666666"
    timeline["audioBusInfos"].append({"busUid": other_bus, "unknown": 123})
    timeline["trackInfos"][0]["busUuids"].append(other_bus)
    timeline["trackInfos"].append(
        {
            "uuid": "track-b",
            "trackType": 2,
            "clipList": [],
            "busUuids": [timeline["audioBusInfos"][0]["busUid"]],
        }
    )
    source = write_template(tmp_path, docs)
    output = tmp_path / "generated.zip"
    instantiate_template(source, output, name="new")
    result = discover_empty_template(parse_project(output))
    assert result.timeline.tracks[0].raw["busUuids"] == [result.bus_uuid, other_bus]
    assert result.timeline.tracks[2].raw["busUuids"] == [result.bus_uuid]
    assert result.timeline.raw["audioBusInfos"][1] == timeline["audioBusInfos"][1]
    assert result.timeline.raw["userData"][:1] == timeline["userData"][:1]
    assert result.timeline.raw["userData"][3] == timeline["userData"][3]


@pytest.mark.parametrize("key", [3, 140])
@pytest.mark.parametrize(
    "variant",
    ["missing", "duplicate", "size", "base64", "length", "content", "padding"],
)
def test_malformed_instance_payload_rejected(
    tmp_path: Path, key: int, variant: str
) -> None:
    docs = template_documents()
    entries = docs[DOCUMENT]["timelineInfos"][0]["userData"]
    entry = next(item for item in entries if item["key"] == key)
    if variant == "missing":
        entries.remove(entry)
    elif variant == "duplicate":
        entries.append(copy.deepcopy(entry))
    elif variant == "size":
        entry["size"] -= 1
    elif variant == "base64":
        entry["data"] = "!invalid!"
    else:
        payload = base64.b64decode(entry["data"])
        if variant == "length":
            payload = payload[:-1]
        elif variant == "content":
            payload = b"z" + payload[1:]
        else:
            payload = payload[:-1] + b"\x00" if key == 140 else payload[:-1] + b"x"
        entry["data"] = base64.b64encode(payload).decode()
    source = write_template(tmp_path, docs)
    output = tmp_path / "rejected.zip"
    with pytest.raises(ProjectWriteError):
        instantiate_template(source, output, name="new")
    assert not output.exists()


@pytest.mark.parametrize(
    "variant", ["missing", "duplicate", "unknown_reference", "malformed_reference"]
)
def test_invalid_bus_graph_rejected(tmp_path: Path, variant: str) -> None:
    docs = template_documents()
    timeline = docs[DOCUMENT]["timelineInfos"][0]
    if variant == "missing":
        timeline["audioBusInfos"] = []
    elif variant == "duplicate":
        timeline["audioBusInfos"] *= 2
    else:
        timeline["trackInfos"][0]["busUuids"] = [
            INSTANCE_ID if variant == "unknown_reference" else "invalid"
        ]
    source = write_template(tmp_path, docs)
    output = tmp_path / "rejected.zip"
    with pytest.raises(ProjectWriteError):
        instantiate_template(source, output, name="new")
    assert not output.exists()
