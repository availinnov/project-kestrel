"""Declarative rule validation and deterministic action evaluation."""

import copy
import json
from typing import Any

import pytest

from kestrel.rules import evaluate_rules, parse_rules


def rule(op: str = "<", value: Any = 2) -> dict[str, Any]:
    return dict(
        rule_id="r",
        enabled=True,
        mode="auto",
        priority=10,
        when=dict(signal="video.value", op=op, value=value),
        then={"drop": True},
    )


def parse(items: list[Any]) -> list[dict[str, Any]]:
    return parse_rules(json.dumps(dict(version=1, rules=items)))


@pytest.mark.parametrize(
    ("op", "actual", "expected"),
    [
        ("<", 1, True),
        ("<=", 2, True),
        (">", 3, True),
        (">=", 2, True),
        ("==", 2, True),
        ("!=", 3, True),
        ("<", 2, False),
    ],
)
def test_operators(op: str, actual: int, expected: bool) -> None:
    assert (
        bool(evaluate_rules(parse([rule(op)]), {"video": {"value": actual}}))
        == expected
    )


def test_groups_priority_disabled_and_dedup() -> None:
    a = rule()
    b = copy.deepcopy(a)
    b.update(rule_id="b", priority=20, mode="review", then={"warning": "check"})
    b["when"] = {"any": [a["when"], {"signal": "missing", "op": "!=", "value": 1}]}
    a["when"] = {"all": [a["when"], {"signal": "available", "op": "==", "value": True}]}
    disabled = copy.deepcopy(a)
    disabled.update(rule_id="disabled", enabled=False)
    parsed = parse([a, b, disabled])
    actions = evaluate_rules(
        parsed + parsed, {"video": {"value": 1}, "available": True}
    )
    assert [a["rule_id"] for a in actions] == ["b", "r"]
    assert evaluate_rules(parse([rule("!=")]), {}) == []
    assert evaluate_rules(parse([rule()]), {"video": {"value": True}}) == []


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "bad"},
        {"priority": True},
        {"enabled": 1},
        {"extra": 1},
        {"then": {"drop": False}},
        {"then": {"color_tag": 14}},
        {"then": {"normalize_audio": {"target_lufs": "x"}}},
        {"when": {"all": []}},
        {"when": {"signal": "x", "op": "eval", "value": 1}},
        {"when": {"signal": "x", "op": "<", "value": "x"}},
    ],
)
def test_invalid(change: Any) -> None:
    item = rule()
    item.update(change)
    with pytest.raises(ValueError):
        parse([item])


def test_duplicates_and_version() -> None:
    for text in (
        "{",
        '{"version":true,"rules":[]}',
        '{"version":1,"version":1,"rules":[]}',
    ):
        with pytest.raises(ValueError):
            parse_rules(text)
    with pytest.raises(ValueError):
        parse([rule(), rule()])
