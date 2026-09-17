"""Selective archive reads and retained-state dataset controls."""

import copy
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kestrel.dataset import extract_dataset
from kestrel.project.parser import ProjectParseError
from tests.test_sequence import fixture, rewrite_fixture


def dataset_fixture(tmp_path: Path) -> Path:
    path = fixture(tmp_path)

    def change(d: Any) -> None:
        d["ProjectFolder/project_info.json"] = d.pop("info.json")
        d["ProjectFolder/Medias/medias_info.json"] = d.pop("catalog.json")
        d["ProjectFolder/Medias/medias_info.json"]["media_items"]["main"]["id"] = "main"
        d["ProjectFolder/Medias/main/timeline.wesproj"] = d.pop("main/timeline.wesproj")
        d["ProjectFolder/Medias/main/extra.json"] = d.pop("main/extra.json")
        for i in range(3):
            d[f"ProjectFolder/Medias/m{i}/media.json"] = {
                "sourceInfo": {
                    "basicInfo": {
                        "createDate": 100 + i,
                        "mediaLength": (i + 1) * 10000000,
                    },
                    "vidStreamInfos": [
                        {
                            "fourCC": 123,
                            "width": 1920,
                            "height": 1080,
                            "frameRate": {"num": 60, "den": 1},
                        }
                    ],
                    "audStreamInfos": [{}],
                }
            }

    rewrite_fixture(path, change)
    return path


def load(tmp_path: Path, source: Path) -> tuple[Any, Any]:
    output = tmp_path / "dataset.json"
    report = extract_dataset(source, output)
    return json.loads(output.read_text()), report


def test_basic_and_binary_skip(tmp_path: Path, monkeypatch: Any) -> None:
    source = dataset_fixture(tmp_path)
    with zipfile.ZipFile(source, "a") as z:
        for i in range(100):
            z.writestr(f"cache/{i}.png", b"fake" * 1000)
    before = source.read_bytes()
    original = zipfile.ZipFile.open

    def guarded(self: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
        n = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert not n.endswith((".png", ".bin"))
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    data, report = load(tmp_path, source)
    assert source.read_bytes() == before
    assert report["ignored_binary_entries"] == 101
    assert report["used_sources"] == 1 and report["dropped_sources"] == 2
    assert data["timeline_fragments"][0]["audio_pair_status"] == "matched"
    assert data["timeline_fragments"][0]["source_in"] == 0
    assert data["timeline_fragments"][0]["source_out"] == 10000000


@pytest.mark.parametrize(
    "variant",
    [
        "split",
        "repeat",
        "missing",
        "ambiguous",
        "effects",
        "transition",
        "unknown_transition",
        "legacy",
        "nonvideo",
    ],
)
def test_features(tmp_path: Path, variant: str) -> None:
    source = dataset_fixture(tmp_path)

    def change(d: Any) -> None:
        t = d["ProjectFolder/Medias/main/timeline.wesproj"]
        tracks = t["timelineInfos"][0]["trackInfos"]
        v = tracks[0]["clipList"][0]
        a = tracks[2]["clipList"][0]
        if variant in ("split", "repeat"):
            second = copy.deepcopy(v)
            second["thisUId"] = "second"
            second.update(tlBegin=20000000, tlEnd=30000000)
            if variant == "split":
                v["outPoint"] = 5000000
                v["tlEnd"] = 5000000
                second.update(inPoint=5000000, tlEnd=25000000)
            tracks[0]["clipList"].insert(0, second)
        if variant == "missing":
            tracks[2]["clipList"] = []
        if variant == "ambiguous":
            second = copy.deepcopy(a)
            second["thisUId"] = "extra-audio"
            tracks[2]["clipList"].append(second)
        if variant == "effects":
            v["effectChainList"] = [
                {
                    "effectList": [
                        {
                            "id": "simple",
                            "paramList": [
                                {
                                    "name": "Brightness",
                                    "fxParam": {"paramType": 2, "unValue": 0.5},
                                }
                            ],
                        },
                        {
                            "id": "unknown",
                            "paramList": [
                                {"name": "opaque", "fxParam": {"unValue": "blob"}}
                            ],
                        },
                    ]
                }
            ]
            a["effectChainList"] = [
                {
                    "effectList": [
                        {
                            "id": "gain",
                            "paramList": [
                                {
                                    "name": "VolumeGain",
                                    "fxParam": {"paramType": 2, "unValue": -6},
                                }
                            ],
                        }
                    ]
                }
            ]
        if variant in ("transition", "unknown_transition"):
            v["postTransition"] = {
                "thisUId": "tr",
                "id": "2981D185-D52E-44f4-ABD5-3CE83890E32E"
                if variant == "transition"
                else "unknown",
                "tlBegin": 9000000,
                "tlEnd": 10000000,
            }
        if variant == "legacy":
            t["resources"] = None
            for track in tracks:
                for c in track["clipList"]:
                    c.pop("sourceUuid", None)
        if variant == "nonvideo":
            d["ProjectFolder/Medias/m2/media.json"]["sourceInfo"]["vidStreamInfos"] = []

    rewrite_fixture(source, change)
    data, report = load(tmp_path, source)
    fragments = data["timeline_fragments"]
    if variant in ("split", "repeat"):
        assert len(fragments) == 2
        assert [f["sequence_index"] for f in fragments] == [0, 1]
        assert fragments[0]["clip_id"] == "c1"
        assert data["sources"][0]["retained_source_ticks"] == (
            10000000 if variant == "split" else 20000000
        )
        assert data["sources"][0]["overlapping_retained_ranges"] == (
            variant == "repeat"
        )
    if variant in ("missing", "ambiguous"):
        assert fragments[0]["audio_pair_status"] == variant
    if variant == "effects":
        assert fragments[0]["image_adjustments"] == {"brightness": 0.5}
        assert fragments[0]["audio_gain_db"] == -6
        assert data["unparsed_effects_summary"] == {"unknown": 1}
    if variant in ("transition", "unknown_transition"):
        assert len(data["transitions"]) == 1
        assert data["transitions"][0]["duration_ticks"] == 1000000
        assert data["transitions"][0]["type"] == (
            "unknown"
            if variant == "unknown_transition"
            else "2981D185-D52E-44f4-ABD5-3CE83890E32E"
        )
    if variant == "legacy":
        assert fragments[0]["audio_pair_status"] == "matched"
    if variant == "nonvideo":
        assert report["imported_video_sources"] == 2


@pytest.mark.parametrize("variant", ["active", "catalog", "geometry", "duplicate"])
def test_reject(tmp_path: Path, variant: str) -> None:
    source = dataset_fixture(tmp_path)

    def change(d: Any) -> None:
        t = d["ProjectFolder/Medias/main/timeline.wesproj"]
        v = t["timelineInfos"][0]["trackInfos"][0]["clipList"][0]
        if variant == "active":
            t["currentTimelineId"] = 999
        if variant == "catalog":
            v["userData"][1].update(data="eHg=", size=2)
        if variant == "geometry":
            v["outPoint"] = 0

    rewrite_fixture(source, change)
    if variant == "duplicate":
        with zipfile.ZipFile(source) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        name = "ProjectFolder/Medias/medias_info.json"
        entries[name] = entries[name].replace(
            b'"media_items": {', b'"media_items": {"m0":{},', 1
        )
        with zipfile.ZipFile(source, "w") as z:
            for n, b in entries.items():
                z.writestr(n, b)
    with pytest.raises(ProjectParseError):
        extract_dataset(source, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_cli_and_input_protection(tmp_path: Path) -> None:
    source = dataset_fixture(tmp_path)
    result = subprocess.run(
        ["kestrel", "dataset-extract", str(source), str(tmp_path / "cli.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "validated: True" in result.stdout
    before = source.read_bytes()
    with pytest.raises(ProjectParseError):
        extract_dataset(source, source)
    assert source.read_bytes() == before
