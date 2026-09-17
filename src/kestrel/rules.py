"""Strict declarative rules over neutral measurements; never applies actions."""

import json
import math
import operator
from typing import Any

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def lookup(signals: dict[str, Any], path: str) -> Any:
    value: Any = signals
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def validate_condition(value: Any, depth: int = 0) -> None:
    if depth > 16 or not isinstance(value, dict):
        raise ValueError("Invalid or excessively nested condition")
    if set(value) in ({"all"}, {"any"}):
        children = next(iter(value.values()))
        if not isinstance(children, list) or not children:
            raise ValueError("all/any requires a non-empty array")
        for child in children:
            validate_condition(child, depth + 1)
    elif set(value) == {"signal", "op", "value"}:
        if not isinstance(value["signal"], str) or not all(
            p.isidentifier() for p in value["signal"].split(".")
        ):
            raise ValueError("Invalid signal path")
        if not isinstance(value["op"], str) or value["op"] not in OPS:
            raise ValueError("Unknown condition operator")
        if not finite_number(value["value"]) and type(value["value"]) not in (
            str,
            bool,
        ):
            raise ValueError("Condition value must be a finite scalar")
        if value["op"] not in ("==", "!=") and not finite_number(value["value"]):
            raise ValueError("Ordered comparison requires a numeric value")
    else:
        raise ValueError("Unknown condition fields")


def parse_rules(text: str) -> list[dict[str, Any]]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate rule JSON field: {key}")
            result[key] = value
        return result

    data = json.loads(text, object_pairs_hook=pairs)
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "rules"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["rules"], list)
    ):
        raise ValueError("Rules require version 1 and a rules array")
    seen = set()
    for rule in data["rules"]:
        if not isinstance(rule, dict) or set(rule) != {
            "rule_id",
            "enabled",
            "mode",
            "priority",
            "when",
            "then",
        }:
            raise ValueError("Invalid rule fields")
        identifier = rule["rule_id"]
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in seen
        ):
            raise ValueError("Rule IDs must be unique non-empty strings")
        seen.add(identifier)
        if (
            type(rule["enabled"]) is not bool
            or rule["mode"] not in ("auto", "review")
            or type(rule["priority"]) is not int
        ):
            raise ValueError("Invalid enabled, mode or priority")
        validate_condition(rule["when"])
        actions = rule["then"]
        if (
            not isinstance(actions, dict)
            or not actions
            or set(actions) - {"drop", "normalize_audio", "color_tag", "warning"}
        ):
            raise ValueError("Invalid proposed actions")
        for kind, value in actions.items():
            if kind == "drop" and value is not True:
                raise ValueError("drop must be true")
            if kind == "color_tag" and (type(value) is not int or not 1 <= value <= 13):
                raise ValueError("color_tag must be an integer from 1 through 13")
            if kind == "warning" and (not isinstance(value, str) or not value.strip()):
                raise ValueError("warning must be a non-empty code")
            if kind == "normalize_audio" and (
                not isinstance(value, dict)
                or set(value) != {"target_lufs"}
                or not finite_number(value["target_lufs"])
            ):
                raise ValueError("normalize_audio requires finite target_lufs")
    return sorted(data["rules"], key=lambda r: (-r["priority"], r["rule_id"]))


def matches(condition: dict[str, Any], signals: dict[str, Any]) -> bool:
    for group in ("all", "any"):
        if group in condition:
            values = (matches(c, signals) for c in condition[group])
            return all(values) if group == "all" else any(values)
    actual = lookup(signals, condition["signal"])
    expected = condition["value"]
    # Unavailable measurements never trigger even a != rule.
    if actual is None:
        return False
    if finite_number(actual) and finite_number(expected):
        return bool(OPS[condition["op"]](actual, expected))
    if type(actual) is type(expected) and condition["op"] in ("==", "!="):
        return bool(OPS[condition["op"]](actual, expected))
    return False


def evaluate_rules(
    rules: list[dict[str, Any]], signals: dict[str, Any]
) -> list[dict[str, Any]]:
    actions = []
    seen = set()
    for rule in sorted(rules, key=lambda r: (-r["priority"], r["rule_id"])):
        if not rule["enabled"] or not matches(rule["when"], signals):
            continue
        for kind, value in rule["then"].items():
            action = dict(type=kind, rule_id=rule["rule_id"], mode=rule["mode"])
            if kind == "color_tag":
                action["value"] = value
            elif kind == "warning":
                action["code"] = value
            elif kind == "normalize_audio":
                action.update(value)
            key = json.dumps(action, sort_keys=True)
            if key not in seen:
                actions.append(action)
                seen.add(key)
    return actions
