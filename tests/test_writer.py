"""Synthetic copy-only writer experiments and preservation checks."""

import copy
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from kestrel.project import parse_project
from kestrel.project.parser import parse_clip
from kestrel.project.writer import (
    ProjectWriteError,
    gain_target,
    set_audio_gain,
    set_trim,
)
from tests.test_audio_gain import gain_clip
from tests.test_project import container, document


def gain_source(tmp_path: Path, *, missing: bool = False) -> Path:
    doc = document()
    clip = gain_clip(0)
    if missing:
        effect = clip["effectChainList"][0]["effectList"][1]
        effect["paramList"].pop()
        effect.update(id="audio/effect/volume", display="volume")
    second = copy.deepcopy(clip)
    second["thisUId"] = "second-audio"
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"] = [clip, second]
    return container(
        tmp_path,
        doc,
        {
            "catalog.json": b'{"item":1, "item":2}',
            "unknown/bytes.bin": bytes(range(256)),
        },
    )


def trim_source(tmp_path: Path, change: str = "") -> Path:
    doc = document()
    doc["timelineInfos"] = doc["timelineInfos"][:1]
    clips: list[dict[str, Any]] = []
    for kind in (1, 2):
        clips.append(
            {
                "thisUId": f"clip-{kind}",
                "type": kind,
                "sourceUuid": "source-a",
                "inPoint": 20_000_000,
                "outPoint": 120_000_000,
                "tlBegin": 0,
                "tlEnd": 100_000_000,
                "speed": {
                    "offset": 2.0,
                    "offsetEnd": 12.0,
                    "reverse": False,
                    "opaque": "keep",
                },
                "unknown": [1, 2, 3],
            }
        )
    doc["resources"][0]["mediaLength"] = 200_000_000
    doc["timelineInfos"][0]["trackInfos"] = [
        {"uuid": f"track-{c['type']}", "trackType": c["type"], "clipList": [c]}
        for c in clips
    ]
    if change == "transition":
        clips[0]["postTransition"] = {"type": 5}
    elif change == "speed":
        clips[0]["speed"]["reverse"] = True
    elif change == "mismatch":
        clips[1]["inPoint"] = 10_000_000
    elif change == "multiple":
        extra = copy.deepcopy(clips[0])
        extra["thisUId"] = "another"
        doc["timelineInfos"][0]["trackInfos"][0]["clipList"].append(extra)
    elif change == "keyframes":
        clips[0]["speed"]["speedParam"] = '{"keyframeSets":[{"_value":2}]}'
    return container(
        tmp_path,
        doc,
        {
            "catalog.json": (
                b'{"opaque":1,"opaque":2,"media_items":{"main":{"media_length":100000000,'
                b'"unknown":42},"source-a":{"media_length":200000000}}}'
            )
        },
    )


@pytest.mark.parametrize("db", [0.0, -6.0, 3.5])
@pytest.mark.parametrize("missing", [False, True])
def test_gain_copy(tmp_path: Path, db: float, missing: bool) -> None:
    source = gain_source(tmp_path, missing=missing)
    output = tmp_path / "modified.zip"
    original_bytes = source.read_bytes()
    before = parse_project(source)
    result = set_audio_gain(source, output, db)
    after = parse_project(output)
    assert result["validated"]
    assert source.read_bytes() == original_bytes
    assert after.active_timeline is not None and before.active_timeline is not None
    clips = after.active_timeline.tracks[0].clips
    assert clips[0].audio_gain_db == db
    assert clips[1].raw == before.active_timeline.tracks[0].clips[1].raw
    doc = after.active_timeline.document
    assert before.raw_entries.keys() == after.raw_entries.keys()
    for name, data in before.raw_entries.items():
        if name != doc:
            assert after.raw_entries[name] == data
    old_raw = copy.deepcopy(before.raw_documents[doc])
    new_raw = copy.deepcopy(after.raw_documents[doc])
    old_raw["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["effectChainList"] = []
    new_raw["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["effectChainList"] = []
    assert old_raw == new_raw
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "modified.zip",
        "synthetic.zip",
    ]


def test_trim_pair_and_catalog(tmp_path: Path) -> None:
    source = trim_source(tmp_path)
    before = parse_project(source)
    data = source.read_bytes()
    output = tmp_path / "trimmed.zip"
    set_trim(source, output, "1.25", "0.5")
    after = parse_project(output)
    assert source.read_bytes() == data
    assert after.active_timeline is not None
    for track in after.active_timeline.tracks:
        clip = track.clips[0]
        assert (clip.in_point, clip.out_point, clip.begin, clip.end) == (
            32_500_000,
            115_000_000,
            0,
            82_500_000,
        )
        assert clip.raw["speed"] == {
            "offset": 3.25,
            "offsetEnd": 11.5,
            "reverse": False,
            "opaque": "keep",
        }
        assert clip.raw["unknown"] == [1, 2, 3]
    assert after.metadata["project_timeline_duration"] == 82_500_000
    assert after.resources[0].raw == before.resources[0].raw
    assert after.raw_entries["catalog.json"] == before.raw_entries[
        "catalog.json"
    ].replace(b'"media_length":100000000', b'"media_length":82500000')
    modified = {
        after.active_timeline.document,
        "Root/project_info.json",
        "catalog.json",
    }
    for name, entry in before.raw_entries.items():
        if name not in modified:
            assert after.raw_entries[name] == entry


@pytest.mark.parametrize(
    "change", ["transition", "speed", "mismatch", "multiple", "keyframes"]
)
def test_trim_unsupported_is_rejected(tmp_path: Path, change: str) -> None:
    source = trim_source(tmp_path, change)
    output = tmp_path / "output.zip"
    with pytest.raises(ProjectWriteError):
        set_trim(source, output, 1, 1)
    assert not output.exists()


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("-1", "0"),
        ("nan", "0"),
        ("inf", "0"),
        ("0.00000001", "0"),
        ("8", "2"),
        ("bad", "0"),
    ],
)
def test_invalid_trim(tmp_path: Path, left: str, right: str) -> None:
    source = trim_source(tmp_path)
    with pytest.raises(ProjectWriteError):
        set_trim(source, tmp_path / "output.zip", left, right)
    assert not (tmp_path / "output.zip").exists()


@pytest.mark.parametrize("operation", ["gain", "trim"])
def test_no_overwrite(tmp_path: Path, operation: str) -> None:
    source = gain_source(tmp_path) if operation == "gain" else trim_source(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "existing.zip"
    output.write_bytes(b"keep")
    for target in (source, output):
        with pytest.raises(ProjectWriteError, match="overwriting"):
            if operation == "gain":
                set_audio_gain(source, target, -6)
            else:
                set_trim(source, target, 1, 1)
    assert source.read_bytes() == original
    assert output.read_bytes() == b"keep"


def test_post_write_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = gain_source(tmp_path)
    output = tmp_path / "output.zip"

    def parse_with_failed_gain(path: str | Path) -> Any:
        project = parse_project(path)
        if Path(path) != source:
            assert project.active_timeline is not None
            project.active_timeline.tracks[0].clips[0].audio_gain_db = 99
        return project

    monkeypatch.setattr("kestrel.project.writer.parse_project", parse_with_failed_gain)
    with pytest.raises(ProjectWriteError, match="validation"):
        set_audio_gain(source, output, -6)
    assert [p.name for p in tmp_path.iterdir()] == [source.name]


def test_zip_metadata_and_json_surroundings(tmp_path: Path) -> None:
    source = gain_source(tmp_path)
    before = parse_project(source)
    assert before.active_timeline is not None
    doc = before.active_timeline.document
    # Rebuild the generated fixture with distinctive metadata and whitespace.
    with zipfile.ZipFile(source, "w") as archive:
        archive.comment = b"archive-comment"
        for name, data in before.raw_entries.items():
            info = zipfile.ZipInfo(name, (2020, 1, 2, 3, 4, 6))
            info.comment = b"entry-comment"
            info.external_attr = 0o600 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            if name == doc:
                data = b"\xef\xbb\xbf \n" + data + b"\n\t"
            archive.writestr(info, data)
    output = tmp_path / "out.zip"
    set_audio_gain(source, output, -6)
    with zipfile.ZipFile(source) as a, zipfile.ZipFile(output) as b:
        assert a.comment == b.comment
        for ia, ib in zip(a.infolist(), b.infolist(), strict=True):
            assert (
                ia.filename,
                ia.date_time,
                ia.comment,
                ia.external_attr,
                ia.compress_type,
            ) == (
                ib.filename,
                ib.date_time,
                ib.comment,
                ib.external_attr,
                ib.compress_type,
            )
        assert b.read(doc) == a.read(doc).replace(
            b'"unValue": 0', b'"unValue": -6.0', 1
        )


@pytest.mark.parametrize("operation", ["gain", "trim"])
def test_writer_cli(tmp_path: Path, operation: str) -> None:
    source = gain_source(tmp_path) if operation == "gain" else trim_source(tmp_path)
    output = tmp_path / "output.zip"
    command = "project-set-audio-gain" if operation == "gain" else "project-set-trim"
    options = (
        ["--db", "-6"]
        if operation == "gain"
        else ["--left-seconds", "1", "--right-seconds", "2"]
    )
    result = subprocess.run(
        ["kestrel", command, str(source), str(output), *options],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validated: True" in result.stdout
    assert output.exists()


def controlled_audio_clip() -> dict[str, Any]:
    """Reproduce observed effect identities with entirely synthetic content."""
    raw = gain_clip(0)
    raw["effectChainList"] = [
        {"name": "Basic", "effectList": []},
        {
            "name": "BasicFollow",
            "effectList": [
                {
                    "id": f"audio/effect/{name}",
                    "display": name,
                    "thisUId": f"synthetic-{name}",
                    "unknown": {"retain": True},
                }
                for name in (
                    "clip_volume",
                    "change_channel",
                    "volume",
                    "ducking",
                    "fade",
                    "equalizer",
                )
            ],
        },
        {"name": "Effect", "effectList": []},
    ]
    return raw


@pytest.mark.parametrize("reverse", [False, True])
def test_controlled_gain_target_and_copy(tmp_path: Path, reverse: bool) -> None:
    raw = controlled_audio_clip()
    if reverse:
        raw["effectChainList"][1]["effectList"].reverse()
        raw["effectChainList"].reverse()
    target = gain_target(parse_clip(raw, []))
    assert target is not None
    assert target[0]["id"] == "audio/effect/volume"
    assert target[0]["display"] == "volume"
    doc = document()
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"] = [raw]
    source = container(tmp_path, doc)
    original = source.read_bytes()
    output = tmp_path / "modified.zip"
    assert set_audio_gain(source, output, -6)["validated"]
    project = parse_project(output)
    assert project.active_timeline is not None
    clip = project.active_timeline.tracks[0].clips[0]
    assert clip.audio_gain_db == -6.0
    expected = copy.deepcopy(raw)
    for chain in expected["effectChainList"]:
        for effect in chain["effectList"]:
            if effect["id"] == "audio/effect/volume":
                effect["paramList"] = [
                    {"name": "VolumeGain", "fxParam": {"paramType": 2, "unValue": -6.0}}
                ]
    assert clip.raw == expected
    assert source.read_bytes() == original


def test_existing_gain_wins_over_known_fallback() -> None:
    raw = controlled_audio_clip()
    effects = raw["effectChainList"][1]["effectList"]
    effects[0]["paramList"] = [
        {"name": "VolumeGain", "fxParam": {"paramType": 2, "unValue": 3.0}}
    ]
    # Even duplicated fallback identities cannot supersede the existing parameter.
    effects.append(copy.deepcopy(effects[2]))
    target = gain_target(parse_clip(raw, []))
    assert target is not None and target[0] is effects[0]


@pytest.mark.parametrize(
    "identity",
    [
        {"id": "audio/effect/clip_volume", "display": "clip_volume"},
        {"id": "other/effect/volume", "display": "volume"},
        {"display": "volume"},
    ],
)
def test_gain_fallback_requires_exact_id(identity: dict[str, str]) -> None:
    raw = controlled_audio_clip()
    raw["effectChainList"] = [{"effectList": [identity]}]
    assert gain_target(parse_clip(raw, [])) is None


@pytest.mark.parametrize("existing", [False, True])
def test_gain_ambiguous_exact_targets_rejected(existing: bool) -> None:
    raw = controlled_audio_clip()
    effects = raw["effectChainList"][1]["effectList"]
    if existing:
        effects[2]["paramList"] = [
            {"name": "VolumeGain", "fxParam": {"paramType": 2, "unValue": -6.0}}
        ]
    effects.append(copy.deepcopy(effects[2]))
    with pytest.raises(ProjectWriteError, match="ambiguous"):
        gain_target(parse_clip(raw, []))


def ordinary_speed_parameter() -> dict[str, Any]:
    """Synthetic values using the observed two-boundary constant-speed schema."""
    return {
        "Version": 3,
        "ParameterType": 0,
        "keyframeSets": [
            {"_time": 0.0, "Interpolation": 6, "_value": 1.0},
            {"_time": 20.0, "Interpolation": 6, "_value": 1.0},
        ],
        "MD5": "synthetic-opaque-checksum",
        "_totalTime": 20.0,
    }


def speed_source(tmp_path: Path, parameter: Any) -> Path:
    source = trim_source(tmp_path)
    project = parse_project(source)
    assert project.active_timeline is not None
    timeline = project.active_timeline
    for track in timeline.tracks:
        track.clips[0].raw["speed"]["speedParam"] = json.dumps(parameter, indent=2)
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in project.raw_entries.items():
            if name == timeline.document:
                data = json.dumps(project.raw_documents[name]).encode()
            archive.writestr(name, data)
    return source


@pytest.mark.parametrize(("left", "right"), [(1, 0), (0, 1), (1, 1)])
def test_ordinary_speed_trim_preserves_parameter(
    tmp_path: Path, left: int, right: int
) -> None:
    source = speed_source(tmp_path, ordinary_speed_parameter())
    before = parse_project(source)
    original = source.read_bytes()
    output = tmp_path / "trimmed.zip"
    set_trim(source, output, left, right)
    after = parse_project(output)
    assert before.active_timeline is not None and after.active_timeline is not None
    for old_track, new_track in zip(
        before.active_timeline.tracks, after.active_timeline.tracks, strict=True
    ):
        old, new = old_track.clips[0], new_track.clips[0]
        assert new.raw["speed"]["speedParam"] == old.raw["speed"]["speedParam"]
        assert new.raw["speed"]["offset"] == 2 + left
        assert new.raw["speed"]["offsetEnd"] == 12 - right
        assert new.in_point == (2 + left) * 10_000_000
        assert new.out_point == (12 - right) * 10_000_000
        assert new.end == (10 - left - right) * 10_000_000
    assert source.read_bytes() == original
    # The unchanged source-span parameter must also support a subsequent trim.
    set_trim(output, tmp_path / "trimmed-again.zip", 0, 1)


@pytest.mark.parametrize(
    "variant",
    [
        "ramp",
        "constant_nonunit",
        "interior_keyframe",
        "interpolation",
        "start_time",
        "end_time",
        "short_total",
        "extra_field",
        "extra_frame_field",
        "version",
        "parameter_type",
        "missing_total",
        "empty_frames",
        "boolean_value",
        "not_object",
    ],
)
def test_unknown_or_varying_speed_rejected(tmp_path: Path, variant: str) -> None:
    parameter = ordinary_speed_parameter()
    frames = parameter["keyframeSets"]
    if variant == "ramp":
        frames[1]["_value"] = 2.0
    elif variant == "constant_nonunit":
        frames[0]["_value"] = frames[1]["_value"] = 0.5
    elif variant == "interior_keyframe":
        frames.insert(1, {"_time": 5.0, "Interpolation": 6, "_value": 1.0})
    elif variant == "interpolation":
        frames[1]["Interpolation"] = 1
    elif variant == "start_time":
        frames[0]["_time"] = 1.0
    elif variant == "end_time":
        frames[1]["_time"] = 19.0
    elif variant == "short_total":
        parameter["_totalTime"] = frames[1]["_time"] = 10.0
    elif variant == "extra_field":
        parameter["unknown"] = 1
    elif variant == "extra_frame_field":
        frames[0]["tangent"] = 1
    elif variant == "version":
        parameter["Version"] = 4
    elif variant == "parameter_type":
        parameter["ParameterType"] = 1
    elif variant == "missing_total":
        del parameter["_totalTime"]
    elif variant == "empty_frames":
        parameter["keyframeSets"] = []
    elif variant == "boolean_value":
        frames[0]["_value"] = True
    elif variant == "not_object":
        parameter = {}
    source = speed_source(tmp_path, parameter)
    original = source.read_bytes()
    output = tmp_path / "rejected.zip"
    with pytest.raises(ProjectWriteError):
        set_trim(source, output, 1, 0)
    assert not output.exists()
    assert source.read_bytes() == original
