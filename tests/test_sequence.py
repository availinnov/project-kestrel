"""Synthetic range rounding and sequence preservation controls."""

import base64
import json
import subprocess
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from kestrel.project.models import Resource
from kestrel.project.parser import parse_project
from kestrel.project.sequence import (
    SPEED_MD5,
    build_constant_speed_param,
    build_sequence,
    initial_source_range,
)
from kestrel.project.writer import ProjectWriteError


@pytest.mark.parametrize(
    ("duration", "num", "den", "end"),
    [
        (10_000_000, 25, 1, 10_000_000),
        (219350000, 60, 1, 219333333),
        (120750000, 60, 1, 120833333),
        (6660000, 60, 1, 6666667),
        (333333, 30, 1, 333333),
        (667333, 30000, 1001, 667333),
    ],
)
def test_range(duration: int, num: int, den: int, end: int) -> None:
    assert initial_source_range(
        Resource("r", "d", None, duration, {}), {"num": num, "den": den}
    ) == (0, end)


@pytest.mark.parametrize("duration", [None, 0, -1, True, 0.5])
def test_invalid_duration(duration: object) -> None:
    resource = Resource("r", "d", None, None, {})
    resource.raw["test"] = duration
    if isinstance(duration, (int, float)):
        resource.duration = duration
    with pytest.raises(ProjectWriteError):
        initial_source_range(resource, {"num": 60, "den": 1})


@pytest.mark.parametrize(
    "fps",
    [None, {}, {"num": 0, "den": 1}, {"num": 60, "den": 0}, {"num": 60.0, "den": 1}],
)
def test_invalid_fps(fps: object) -> None:
    with pytest.raises(ProjectWriteError):
        initial_source_range(Resource("r", "d", None, 10, {}), fps)


def fixture(tmp_path: Path, overlap: bool = False) -> Path:
    linkage = "{11111111-1111-4111-8111-111111111111}"
    resources = [
        dict(
            sourceUuid=f"s{i}",
            filename=f"file:/synthetic/{i}.mp4",
            mediaLength=(i + 1) * 10_000_000,
            streamType=2,
            vidStreamInfo=[{"vidStreamId": 0}],
            audStreamInfo=[{"audStreamId": 1}],
        )
        for i in range(3)
    ]
    tracks = []
    for kind in (1, 2):
        tracks.append(
            dict(
                uuid=f"t{kind}",
                trackType=kind,
                clipList=[
                    dict(
                        thisUId=f"c{kind}",
                        type=kind,
                        sourceUuid="s0",
                        filename=resources[0]["filename"],
                        inPoint=0,
                        outPoint=10_000_000,
                        tlBegin=0,
                        tlEnd=10_000_000,
                        speed=dict(offset=0.0, offsetEnd=1.0, reverse=False),
                        effectChainList=[
                            dict(
                                effectList=[
                                    dict(
                                        id="video/effect/transform"
                                        if kind == 1
                                        else "audio/effect/volume",
                                        thisUId=f"e{kind}",
                                    )
                                ]
                            )
                        ],
                        opaque="preserve",
                        userData=[
                            dict(
                                key=3,
                                size=64 if kind == 1 else 38,
                                data=base64.b64encode(
                                    linkage.encode() + (bytes(26) if kind == 1 else b"")
                                ).decode(),
                            ),
                            dict(key=10, size=2, data=base64.b64encode(b"m0").decode()),
                        ],
                    )
                ],
            )
        )
        if not overlap:
            tracks.append(dict(uuid=f"empty{kind}", trackType=kind, clipList=[]))
    docs = {
        "info.json": dict(
            project_guid="keep",
            timeline_mediaId="main",
            project_date_modify=123,
            project_timeline_duration=10_000_000,
        ),
        "catalog.json": dict(
            media_items={
                "main": {"duration": 10_000_000},
                **{
                    f"m{i}": dict(
                        id=f"m{i}",
                        media_type=8,
                        download_url=f"synthetic/{i}.mp4",
                        name=str(i),
                        media_length=(i + 1) * 10_000_000,
                    )
                    for i in range(3)
                },
            }
        ),
        "main/timeline.wesproj": dict(
            currentTimelineId=1,
            resources=resources,
            timelineInfos=[
                dict(timelineId=1, frameRate={"num": 60, "den": 1}, trackInfos=tracks)
            ],
        ),
    }
    path = tmp_path / "input.zip"
    docs["main/extra.json"] = dict(
        mediaClipsMapInfo={linkage: {"mediaId": "m0", "subClips": {}}},
        allMarkersInfo={"beatDetectInfo": {linkage: {"level": 0}}},
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, doc in docs.items():
            archive.writestr(name, json.dumps(doc))
        archive.writestr("opaque.bin", b"\0\xffunchanged")
    return path


@pytest.mark.parametrize("count", [1, 3])
def test_sequence(tmp_path: Path, count: int) -> None:
    source = fixture(tmp_path)
    before = parse_project(source)
    output = tmp_path / "out.zip"
    report = build_sequence(source, output, count)
    after = parse_project(output)
    assert report["validated"] and report["generated_source_count"] == count
    assert [r.raw for r in before.resources] == [r.raw for r in after.resources]
    assert before.raw_entries["opaque.bin"] == after.raw_entries["opaque.bin"]
    assert (
        before.metadata["project_date_modify"] == after.metadata["project_date_modify"]
    )
    assert [s["source_uuid"] for s in report["selected_sources"]] == [
        f"s{i}" for i in range(count)
    ]
    assert before.active_timeline and after.active_timeline
    new_ids = [x["generated"] for x in report["generated_identity_fields"]]
    assert len(new_ids) == len(set(new_ids)) == count * 5
    previous = 0
    for selected in report["selected_sources"]:
        assert selected["tl_begin"] == previous
        previous = selected["tl_end"]
        pair = [
            c
            for t in after.active_timeline.tracks
            for c in t.clips
            if c.id in (selected["video_clip_id"], selected["audio_clip_id"])
        ]
        assert len(pair) == 2
        for c in pair:
            assert isinstance(c.out_point, int)
            assert c.raw["speed"] == dict(
                offset=0.0,
                offsetEnd=c.out_point / 10_000_000,
                reverse=False,
                speedParam=build_constant_speed_param(c.out_point),
            )
            assert (
                c.in_point == 0
                and c.begin == selected["tl_begin"]
                and c.end == previous
            )
    assert after.duration == previous


@pytest.mark.parametrize("count", [0, -1, 4])
def test_invalid_count(tmp_path: Path, count: int) -> None:
    with pytest.raises(ProjectWriteError):
        build_sequence(fixture(tmp_path), tmp_path / "out.zip", count)
    assert not (tmp_path / "out.zip").exists()


def test_overlap(tmp_path: Path) -> None:
    with pytest.raises(ProjectWriteError, match="overlap"):
        build_sequence(fixture(tmp_path, True), tmp_path / "out.zip", 1)


def test_cli(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "kestrel",
            "project-build-sequence",
            str(fixture(tmp_path)),
            str(tmp_path / "out.zip"),
            "--count",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated: True" in result.stdout


@pytest.mark.parametrize("ticks", [6666667, 219333333, 10_000_000, 10**17 + 1])
def test_dynamic_speed(ticks: int) -> None:
    value = json.loads(build_constant_speed_param(ticks), parse_float=Decimal)
    assert value["Version"] == 3 and value["ParameterType"] == 0
    assert value["MD5"] == SPEED_MD5
    assert value["_totalTime"] == Decimal(ticks) / Decimal(10_000_000)
    assert value["keyframeSets"] == [
        {"_time": 0, "Interpolation": 6, "_value": 1},
        {"_time": value["_totalTime"], "Interpolation": 6, "_value": 1},
    ]


def rewrite_fixture(path: Path, change: Any) -> None:
    with zipfile.ZipFile(path) as archive:
        data = {n: archive.read(n) for n in archive.namelist()}
    docs = {n: json.loads(v) for n, v in data.items() if n != "opaque.bin"}
    change(docs)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("opaque.bin", data["opaque.bin"])
        for n, doc in docs.items():
            archive.writestr(n, json.dumps(doc))


def test_catalog_only_resource_creation(tmp_path: Path) -> None:
    source = fixture(tmp_path)

    def change(docs: Any) -> None:
        timeline = docs["main/timeline.wesproj"]
        for index in (1, 2):
            resource = timeline["resources"][index]
            docs[f"ProjectFolder/Medias/m{index}/media.json"] = {
                "file_name": f"synthetic/{index}.mp4",
                "sourceInfo": {
                    "basicInfo": {
                        "mediaLength": resource["mediaLength"],
                        "streamType": 2,
                    },
                    "vidStreamInfos": resource["vidStreamInfo"],
                    "audStreamInfos": resource["audStreamInfo"],
                },
            }
        timeline["resources"] = timeline["resources"][:1]

    rewrite_fixture(source, change)
    before = parse_project(source)
    output = tmp_path / "new.zip"
    report = build_sequence(source, output, 3)
    after = parse_project(output)
    assert (
        len(after.resources) == 3 and after.resources[0].raw == before.resources[0].raw
    )
    assert len({r.id for r in after.resources}) == 3
    assert [s["catalog_id"] for s in report["selected_sources"]] == ["m0", "m1", "m2"]
    for selected in report["selected_sources"]:
        assert (
            after.raw_documents["main/extra.json"]["mediaClipsMapInfo"][
                selected["key3"]
            ]["mediaId"]
            == selected["catalog_id"]
        )
    for key in ("m0", "m1", "m2"):
        assert (
            before.raw_documents["catalog.json"]["media_items"][key]
            == after.raw_documents["catalog.json"]["media_items"][key]
        )


@pytest.mark.parametrize(
    "variant", ["no_template", "bad_link", "unresolved", "no_audio"]
)
def test_reject_unsupported(tmp_path: Path, variant: str) -> None:
    source = fixture(tmp_path)

    def change(docs: Any) -> None:
        timeline = docs["main/timeline.wesproj"]
        if variant == "no_template":
            timeline["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["speed"][
                "reverse"
            ] = True
        elif variant == "bad_link":
            docs["main/extra.json"]["mediaClipsMapInfo"] = {}
        elif variant == "unresolved":
            timeline["resources"] = []
        else:
            for resource in timeline["resources"]:
                resource["audStreamInfo"] = []

    rewrite_fixture(source, change)
    with pytest.raises(ProjectWriteError):
        build_sequence(source, tmp_path / "rejected.zip", 1)
    assert not (tmp_path / "rejected.zip").exists()
