"""Synthetic fixtures for conservative semantic project comparison."""

import base64
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kestrel.formats.diff import diff_files
from kestrel.project import parse_project
from kestrel.project.parser import build_project
from kestrel.project.semantic import semantic_diff, semantic_diff_files
from tests.test_project import container, document


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return semantic_diff(
        build_project({"timeline.data": json.dumps(a).encode()}),
        build_project({"timeline.data": json.dumps(b).encode()}),
    )


def first_clip(doc: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0]
    return result


def paths(result: dict[str, Any]) -> list[str]:
    return [change["path"] for change in result["changes"]]


def test_volatile_ids_and_metadata() -> None:
    a, b = document(), document()
    for index, key in enumerate(
        (
            "thisUId",
            "project_guid",
            "project_source",
            "project_date_modify",
            "proj_zip_save_path",
            "proj_cover_proj_path",
            "timeline_uuid",
        )
    ):
        a[key], b[key] = f"old-{index}", f"new-{index}"
    first_clip(b)["thisUId"] = "regenerated"
    first_clip(b)["postTransition"]["thisUId"] = "transition-new"
    b["timelineInfos"][0]["trackInfos"][0]["uuid"] = "track-new"
    assert compare(a, b)["equal"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inPoint", 12),
        ("outPoint", 35),
        ("tlBegin", 7),
        ("tlEnd", 28),
        ("enable", False),
        ("scriptBuf", "new script"),
        ("text", "new text"),
    ],
)
def test_clip_property_changes(field: str, value: Any) -> None:
    a, b = document(), document()
    first_clip(b)["thisUId"] = "regenerated"
    first_clip(b)[field] = value
    result = compare(a, b)
    assert result["change_count"] == 1
    assert paths(result)[0].endswith("/" + field)
    assert result["changes"][0]["right"] == value


def test_reorder_clips_and_change_multiple_ranges() -> None:
    a, b = document(), document()
    clip = first_clip(b)
    clip.update(inPoint=20, outPoint=40, tlBegin=50, tlEnd=70, thisUId="new")
    b["timelineInfos"][0]["trackInfos"][0]["clipList"].reverse()
    result = compare(a, b)
    assert result["change_count"] == 4
    assert all(change["change"] == "changed" for change in result["changes"])


def test_repeated_sources_match_by_position() -> None:
    a = document()
    extra = copy.deepcopy(first_clip(a))
    extra.update(thisUId="second", tlBegin=50, tlEnd=70)
    a["timelineInfos"][0]["trackInfos"][0]["clipList"].append(extra)
    b = copy.deepcopy(a)
    b["timelineInfos"][0]["trackInfos"][0]["clipList"][2].update(
        thisUId="new", inPoint=15
    )
    b["timelineInfos"][0]["trackInfos"][0]["clipList"].reverse()
    result = compare(a, b)
    assert result["change_count"] == 1
    assert paths(result)[0].endswith("/clips/2/inPoint")


def test_ambiguous_clips_report_add_remove() -> None:
    a = document()
    extra = copy.deepcopy(first_clip(a))
    extra["thisUId"] = "second"
    a["timelineInfos"][0]["trackInfos"][0]["clipList"].append(extra)
    b = copy.deepcopy(a)
    first_clip(b)["inPoint"] = 12
    result = compare(a, b)
    assert not result["equal"]
    assert {c["change"] for c in result["changes"]} == {"added", "removed"}


def test_identical_duplicate_clips_do_not_create_changes() -> None:
    a = document()
    extra = copy.deepcopy(first_clip(a))
    extra["thisUId"] = "duplicate"
    a["timelineInfos"][0]["trackInfos"][0]["clipList"].append(extra)
    b = copy.deepcopy(a)
    first_clip(b)["thisUId"] = "generated-again"
    assert compare(a, b)["equal"]


def test_effect_order_and_parameters() -> None:
    a = document()
    effects = first_clip(a)["effectChainList"][0]["effectList"]
    effects.extend([{"id": "effect-b", "enable": True}, {"id": "effect-c", "gain": 1}])
    b = copy.deepcopy(a)
    first_clip(b)["effectChainList"][0]["effectList"].reverse()
    assert compare(a, b)["equal"]
    first_clip(b)["effectChainList"][0]["effectList"][0]["gain"] = 2
    result = compare(a, b)
    assert result["change_count"] == 1
    assert paths(result)[0].endswith("/effectList/effect-c/gain")


def test_duplicate_effect_ids_keep_order() -> None:
    a = document()
    first_clip(a)["effectChainList"] = [
        {"effectList": [{"id": "same", "gain": 1}, {"id": "same", "gain": 2}]}
    ]
    b = copy.deepcopy(a)
    first_clip(b)["effectChainList"][0]["effectList"].reverse()
    assert not compare(a, b)["equal"]


def reference(uid: str) -> dict[str, Any]:
    payload = ("{" + uid + "}").encode() + bytes(26)
    return {"key": 3, "size": len(payload), "data": base64.b64encode(payload).decode()}


def test_known_uuid_userdata_reference_and_retargeting() -> None:
    a, b = document(), document()
    uid_a = "11111111-1111-4111-8111-111111111111"
    uid_b = "22222222-2222-4222-8222-222222222222"
    first_clip(a).update(thisUId=uid_a, userData=[reference(uid_a)])
    first_clip(b).update(thisUId=uid_b, userData=[reference(uid_b)])
    assert compare(a, b)["equal"]
    transition_id = "33333333-3333-4333-8333-333333333333"
    first_clip(b)["postTransition"]["thisUId"] = transition_id
    first_clip(b)["userData"] = [reference(transition_id)]
    assert not compare(a, b)["equal"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("AQAAAA==", "DQAAAA=="),  # Numeric tags remain meaningful.
        ("b25l", "dHdv"),  # Text remains meaningful.
        ("%%%", "!!!"),  # Invalid Base64 remains visible.
    ],
)
def test_non_uuid_userdata_is_retained(left: str, right: str) -> None:
    a, b = document(), document()
    first_clip(a)["userData"] = [{"key": 13000, "size": 4, "data": left}]
    first_clip(b)["userData"] = [{"key": 13000, "size": 4, "data": right}]
    assert not compare(a, b)["equal"]


def test_unknown_uuid_and_mixed_payload_are_retained() -> None:
    a, b = document(), document()
    first_clip(a)["userData"] = [reference("11111111-1111-4111-8111-111111111111")]
    first_clip(b)["userData"] = [reference("22222222-2222-4222-8222-222222222222")]
    assert not compare(a, b)["equal"]
    for doc in (a, b):
        clip = first_clip(doc)
        clip["thisUId"] = "11111111-1111-4111-8111-111111111111"
        clip["userData"][0]["data"] = base64.b64encode(
            (clip["thisUId"] + " meaningful").encode()
        ).decode()
    first_clip(b)["userData"][0]["data"] += "AA=="
    assert not compare(a, b)["equal"]


def test_audio_parameter_and_transition_changes() -> None:
    a, b = document(), document()
    parameter = json.loads(first_clip(b)["volumeKeyframe"]["parameter"])
    first_clip(b)["volumeKeyframe"]["parameter"] = json.dumps(parameter, indent=2)
    assert compare(a, b)["equal"]
    parameter["keyframeSets"][0]["_value"] = 0.25
    first_clip(b)["volumeKeyframe"]["parameter"] = json.dumps(parameter)
    first_clip(b)["postTransition"]["tlEnd"] = 35
    result = compare(a, b)
    assert result["change_count"] == 2
    assert any(path.endswith("/_value") for path in paths(result))
    assert any(path.endswith("/postTransition/tlEnd") for path in paths(result))


def test_nested_timeline_renumbering_and_removal() -> None:
    a, b = document(), document()
    b["timelineInfos"][1]["timelineId"] = 99
    b["timelineInfos"][0]["trackInfos"][0]["clipList"][1]["timelineId"] = 99
    assert compare(a, b)["equal"]
    b["timelineInfos"].pop()
    result = compare(a, b)
    assert any(
        c["change"] == "removed" and c["path"] == "$/timelines/1"
        for c in result["changes"]
    )
    assert any(path.endswith("/timelineId") for path in paths(result))
    assert any(
        c["change"] == "added" and c["path"] == "$/timelines/1"
        for c in compare(b, a)["changes"]
    )


def test_unknown_properties_and_large_scripts_are_reported() -> None:
    a, b = document(), document()
    first_clip(b)["unknown_clip"]["opaque"] = "changed"
    first_clip(b)["scriptBuf"] = "x" * 2000
    result = compare(a, b)
    assert result["change_count"] == 2
    script = next(c for c in result["changes"] if c["path"].endswith("/scriptBuf"))
    assert "sha256" in script["right"]
    assert len(json.dumps(result)) < 1500


def test_cli_semantic_and_raw_unchanged(tmp_path: Path) -> None:
    a, b = document(), document()
    first_clip(b)["thisUId"] = "new-id"
    left_dir, right_dir = tmp_path / "left", tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left, right = container(left_dir, a), container(right_dir, b)
    before = left.read_bytes(), right.read_bytes()
    assert semantic_diff_files(left, right)["equal"]
    assert not diff_files(left, right)["equal"]
    for machine in (False, True):
        result = subprocess.run(
            [
                "kestrel",
                "diff",
                str(left),
                str(right),
                "--semantic",
                *(["--json"] if machine else []),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stderr == ""
        if machine:
            assert json.loads(result.stdout)["equal"]
        else:
            assert "change_count: 0" in result.stdout
    result = subprocess.run(
        ["kestrel", "diff", str(left), str(right), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == diff_files(left, right)
    assert before == (left.read_bytes(), right.read_bytes())
    project = parse_project(left)
    raw = copy.deepcopy(project.raw_documents)
    assert semantic_diff(project, project)["equal"]
    assert project.raw_documents == raw


def test_semantic_cli_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    path.write_bytes(b"bad")
    result = subprocess.run(
        ["kestrel", "diff", str(path), str(path), "--semantic", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "error" in json.loads(result.stdout)
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("owner", ["clip", "effect"])
def test_userdata_insertion_matches_by_key(position: int, owner: str) -> None:
    a = document()
    target = first_clip(a)
    if owner == "effect":
        target = target["effectChainList"][0]["effectList"][0]
    target["userData"] = [
        {"key": 7, "size": 1, "data": "AQ=="},
        {"key": 80, "size": 1, "data": "Ag=="},
    ]
    b = copy.deepcopy(a)
    target = first_clip(b)
    if owner == "effect":
        target = target["effectChainList"][0]["effectList"][0]
    entry = {"key": 13000, "size": 4, "data": "AgAAAA=="}
    target["userData"].insert(position, entry)
    result = compare(a, b)
    assert result["change_count"] == 1
    change = result["changes"][0]
    assert change["path"].endswith("/userData/13000")
    assert change["change"] == "added"
    assert change["right"] == entry
    reverse = compare(b, a)
    assert reverse["change_count"] == 1
    assert reverse["changes"][0]["change"] == "removed"
    assert reverse["changes"][0]["path"].endswith("/userData/13000")


def test_userdata_reorder_and_update_matches_by_key() -> None:
    a = document()
    first_clip(a)["userData"] = [
        {"key": 7, "size": 1, "data": "AQ=="},
        {"key": 13000, "size": 4, "data": "AgAAAA=="},
    ]
    b = copy.deepcopy(a)
    first_clip(b)["userData"].reverse()
    assert compare(a, b)["equal"]
    first_clip(b)["userData"][0]["data"] = "AwAAAA=="
    result = compare(a, b)
    assert result["change_count"] == 1
    assert paths(result)[0].endswith("/userData/13000/data")


@pytest.mark.parametrize("keys", [[7, 7], [True, 7], ["7", 80]])
def test_ambiguous_userdata_is_not_collapsed(keys: list[Any]) -> None:
    a = document()
    first_clip(a)["userData"] = [
        {"key": keys[0], "data": "AQ=="},
        {"key": keys[1], "data": "Ag=="},
    ]
    b = copy.deepcopy(a)
    first_clip(b)["userData"].reverse()
    assert not compare(a, b)["equal"]
