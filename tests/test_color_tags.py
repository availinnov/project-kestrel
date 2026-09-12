"""Synthetic numeric clip tag fixtures."""

import base64
import copy
from pathlib import Path
from typing import Any

import pytest

from kestrel.project import parse_project
from kestrel.project.models import COLOR_TAG_PALETTE_ORDER
from kestrel.project.parser import parse_clip
from tests.test_project import container, document


def tagged_clip(entries: Any) -> dict[str, Any]:
    return {
        "thisUId": "tagged",
        "inPoint": 0,
        "outPoint": 10,
        "tlBegin": 0,
        "tlEnd": 10,
        "userData": entries,
    }


@pytest.mark.parametrize("value", range(1, 14))
def test_known_tag_values(value: int) -> None:
    entry = {
        "key": 13000,
        "size": 4,
        "data": base64.b64encode(value.to_bytes(4, "little")).decode(),
        "unknown": [9],
    }
    raw = tagged_clip([{"key": 7, "data": "opaque"}, entry])
    before = copy.deepcopy(raw)
    warnings: list[str] = []
    clip = parse_clip(raw, warnings)
    assert clip.color_tag == value
    assert clip.raw == before
    assert clip.raw["userData"][1] is entry
    assert warnings == []


@pytest.mark.parametrize("value", [0, 14, 0x01020304, -1, -(2**31), 2**31 - 1])
def test_unknown_values_remain_numeric(value: int) -> None:
    raw = tagged_clip(
        [
            {
                "key": 13000,
                "size": 4,
                "data": base64.b64encode(
                    value.to_bytes(4, "little", signed=True)
                ).decode(),
            }
        ]
    )
    warnings: list[str] = []
    assert parse_clip(raw, warnings).color_tag == value
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "entry",
    [
        {"key": 13000, "size": 3, "data": "AQAAAA=="},
        {"key": 13000, "size": 4.0, "data": "AQAAAA=="},
        {"key": 13000, "size": 4, "data": "%%%"},
        {"key": 13000, "size": 4, "data": "AQ=="},
        {"key": 13000, "size": 4, "data": "AQAAAAA="},
        {"key": 13000, "size": 4, "data": "AQAAAA"},
        {"key": 13000, "size": 4, "data": "\u2603"},
        {"key": 13000, "size": 4, "data": None},
        {"key": 13000},
    ],
)
def test_invalid_tag_is_retained(entry: dict[str, Any]) -> None:
    raw = tagged_clip([entry])
    before = copy.deepcopy(raw)
    warnings: list[str] = []
    assert parse_clip(raw, warnings).color_tag is None
    assert raw == before
    assert len(warnings) == 1


def test_absent_tag() -> None:
    raw = tagged_clip([{"key": 7, "data": "opaque"}])
    warnings: list[str] = []
    assert parse_clip(raw, warnings).color_tag is None
    del raw["userData"]
    assert parse_clip(raw, warnings).color_tag is None
    assert warnings == []


def test_duplicate_tag_is_ambiguous() -> None:
    raw = tagged_clip(
        [
            {"key": 13000, "size": 4, "data": "AQAAAA=="},
            {"key": 13000, "size": 4, "data": "AgAAAA=="},
        ]
    )
    warnings: list[str] = []
    assert parse_clip(raw, warnings).color_tag is None
    assert len(raw["userData"]) == 2
    assert "ambiguous" in warnings[0]


def test_container_tag(tmp_path: Path) -> None:
    doc = document()
    entry = {"key": 13000, "size": 4, "data": "DQAAAA=="}
    doc["timelineInfos"][0]["trackInfos"][0]["clipList"][0]["userData"] = [entry]
    project = parse_project(container(tmp_path, doc))
    clip = project.timelines[0].tracks[0].clips[0]
    assert clip.color_tag == 13
    assert clip.raw["userData"] == [entry]


def test_palette_order() -> None:
    assert COLOR_TAG_PALETTE_ORDER == (1, 2, 3, 4, 5, 6, 7, 13, 8, 9, 10, 11, 12)
