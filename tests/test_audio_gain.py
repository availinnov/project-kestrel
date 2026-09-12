"""Generated fixtures for decoded audio gain parameters."""

import copy
from pathlib import Path
from typing import Any

import pytest

from kestrel.project import parse_project
from kestrel.project.parser import parse_clip
from tests.test_project import container, document


def gain_clip(value: Any) -> dict[str, Any]:
    return {
        "thisUId": "audio-a",
        "type": 2,
        "inPoint": 0,
        "outPoint": 10,
        "tlBegin": 0,
        "tlEnd": 10,
        "effectChainList": [
            {
                "effectList": [
                    {
                        "id": "unrelated",
                        "paramList": [{"name": "Other", "fxParam": {"unValue": 99}}],
                    },
                    {
                        "id": "audio/effect/clip_volume",
                        "display": "clip_volume",
                        "paramList": [
                            {
                                "name": "Other",
                                "fxParam": {"paramType": 2, "unValue": 123},
                            },
                            {
                                "name": "VolumeGain",
                                "fxParam": {
                                    "paramType": 2,
                                    "unValue": value,
                                    "unknown": "retain",
                                },
                            },
                        ],
                    },
                ]
            }
        ],
    }


@pytest.mark.parametrize("value", [0, 0.0, -6.0, -12.5, 3.0, 6])
@pytest.mark.parametrize("reordered", [False, True])
def test_gain_values(value: float, reordered: bool) -> None:
    raw = gain_clip(value)
    if reordered:
        effects = raw["effectChainList"][0]["effectList"]
        effects[1]["paramList"].reverse()
        effects.reverse()
        raw["effectChainList"].insert(0, {"effectList": []})
    before = copy.deepcopy(raw)
    warnings: list[str] = []
    clip = parse_clip(raw, warnings)
    assert clip.audio_gain_db == float(value)
    assert type(clip.audio_gain_db) is float
    assert clip.raw == before
    assert warnings == []


@pytest.mark.parametrize("identity", [{"display": "Volume"}, {}, {"id": "custom-gain"}])
def test_gain_identity_fallback(identity: dict[str, str]) -> None:
    raw = gain_clip(-6)
    effect = raw["effectChainList"][0]["effectList"][1]
    effect.pop("id")
    effect.pop("display")
    effect.update(identity)
    assert parse_clip(raw, []).audio_gain_db == -6.0


@pytest.mark.parametrize(
    "value", [None, True, "-6.0", [], float("inf"), float("nan"), 10**400]
)
def test_invalid_gain_preserves_raw(value: Any) -> None:
    raw = gain_clip(value)
    warnings: list[str] = []
    clip = parse_clip(raw, warnings)
    assert clip.audio_gain_db is None
    assert clip.raw is raw
    assert len(warnings) == 1


@pytest.mark.parametrize("param_type", [1, 2.0, True, None])
def test_wrong_parameter_type(param_type: Any) -> None:
    raw = gain_clip(-6)
    raw["effectChainList"][0]["effectList"][1]["paramList"][1]["fxParam"][
        "paramType"
    ] = param_type
    warnings: list[str] = []
    assert parse_clip(raw, warnings).audio_gain_db is None
    assert len(warnings) == 1


def test_missing_gain() -> None:
    raw = gain_clip(-6)
    raw["effectChainList"][0]["effectList"].pop()
    warnings: list[str] = []
    assert parse_clip(raw, warnings).audio_gain_db is None
    assert warnings == []


def test_ambiguous_gain_is_not_selected_by_position() -> None:
    raw = gain_clip(-6)
    effects = raw["effectChainList"][0]["effectList"]
    other = copy.deepcopy(effects[1])
    other["id"] = "audio/effect/volume"
    effects.append(other)
    warnings: list[str] = []
    assert parse_clip(raw, warnings).audio_gain_db is None
    assert "ambiguous" in warnings[0]
    assert "audio/effect/volume" in warnings[0]


def test_gain_in_container(tmp_path: Path) -> None:
    doc = document()
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0] = gain_clip(-6)
    project = parse_project(container(tmp_path, doc))
    assert project.timelines[0].tracks[0].clips[0].audio_gain_db == -6.0
