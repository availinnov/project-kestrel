"""Offline rule evaluation over saved media-analysis measurements."""

import json
import time
from pathlib import Path
from typing import Any

from kestrel.project.parser import ProjectParseError
from kestrel.rules import action_statistics, evaluate_source_rules, parse_rules


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    def reject_duplicate_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate {description} JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicate_fields,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def validate_analysis(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the reusable portion of Media Analysis v1."""
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Analysis requires version 1")
    for field in ("project", "detector_config", "measurement_notes", "summary"):
        if not isinstance(data.get(field), dict):
            raise ValueError(f"Analysis requires a {field} object")
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Analysis requires a sources array")
    required = {
        "catalog_id",
        "filename",
        "source_duration_ticks",
        "signals",
        "analysis_status",
        "actions",
    }
    catalog_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or not required.issubset(source):
            raise ValueError(f"Invalid analysis source at index {index}")
        catalog_id = source["catalog_id"]
        if (
            not isinstance(catalog_id, str)
            or not catalog_id
            or catalog_id in catalog_ids
            or not isinstance(source["filename"], str)
            or not isinstance(source["analysis_status"], str)
        ):
            raise ValueError(f"Invalid source identity or status at index {index}")
        catalog_ids.add(catalog_id)
        if not isinstance(source["signals"], dict):
            raise ValueError(f"Invalid source signals at index {index}")
        if not isinstance(source["actions"], list):
            raise ValueError(f"Invalid source actions at index {index}")
    return sources


def rules_evaluate(
    analysis_path: str | Path,
    rules_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Re-evaluate rules from saved signals without reading source media."""
    started = time.perf_counter()
    analysis_path = Path(analysis_path)
    rules_path = Path(rules_path)
    output_path = Path(output_path)
    resolved_output = output_path.resolve()
    if resolved_output in {
        analysis_path.resolve(),
        rules_path.resolve(),
    }:
        raise ProjectParseError(
            "Rules output must be a new file distinct from analysis and rules inputs"
        )
    if output_path.exists():
        raise ProjectParseError("Rules output already exists; refusing to overwrite")
    try:
        data = _load_json_object(analysis_path, "analysis")
        sources = validate_analysis(data)
        rules = parse_rules(rules_path.read_text(encoding="utf-8-sig"))
        evaluate_source_rules(sources, rules)
        statistics = action_statistics(sources)

        summary = data.get("summary")
        if isinstance(summary, dict):
            rule_counts = statistics["rule_match_counts"]
            summary_updates = {
                "rule_match_counts": rule_counts,
                "proposed_quiet_normalization_count": rule_counts.get(
                    "normalize_quiet_audio", 0
                ),
                "proposed_loud_normalization_count": rule_counts.get(
                    "normalize_loud_audio", 0
                ),
                "proposed_short_peak_review_count": rule_counts.get(
                    "review_short_loud_event", 0
                ),
                "proposed_short_loud_event_review_count": rule_counts.get(
                    "review_short_loud_event", 0
                ),
            }
            for field, value in summary_updates.items():
                if field in summary:
                    summary[field] = value

        data["rule_evaluation"] = {
            "rules_source": str(rules_path),
            "rule_count": len(rules),
            "matched_rule_count": sum(statistics["rule_match_counts"].values()),
        }
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)

        return {
            "output": str(output_path),
            "source_count": len(sources),
            "evaluated_source_count": len(sources),
            "rule_count": len(rules),
            "rule_match_counts": statistics["rule_match_counts"],
            "action_type_counts": statistics["action_type_counts"],
            "runtime_seconds": time.perf_counter() - started,
            "output_bytes": output_path.stat().st_size,
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        raise ProjectParseError(str(error)) from error
