"""JSON-only same-scene rule exploration; no production policy or media decoding."""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from kestrel.project.parser import ProjectParseError

KNOWN_POSITIVES = [
    ["VID_20250308_105013.mp4", "VID_20250308_105025.mp4"],
    ["VID_20250313_165213.mp4", "VID_20250313_165220.mp4"],
    ["VID_20250315_114914.mp4", "VID_20250315_114922.mp4"],
    ["VID_20250315_114914.mp4", "VID_20250315_114936.mp4"],
    ["VID_20250315_114922.mp4", "VID_20250315_114936.mp4"],
    ["VID_20250316_125425.mp4", "VID_20250316_125435.mp4"],
]
TIME_LIMIT = 180
FEATURE_DEFINITIONS = {
    "normalized_counts": "For each directional frame match divide count by "
    "max(1,min(keypoints_query,keypoints_target)); clip to [0,1], then take maximum. "
    "These are saturated count ratios, not unique-correspondence fractions: raw "
    "descriptor reuse cannot be corrected without individual correspondences.",
    "support_counts": "Count distinct physical frame pairs with >=4/8/12 inliers "
    "in either direction; do not double-count forward/reverse comparisons.",
    "orb_source_frame_coverage": "At >=4 inliers, participating sampled frames / "
    "sampled_frame_count per source; scalar coverage is the smaller fraction.",
    "raw_counts": "Existing maximum good-match and inlier counts retained unchanged.",
    "missing_data": "Missing frame-level information gives null derived features "
    "with an explicit unavailable reason; rules using missing features abstain.",
}


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    def name(value: str) -> str:
        return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()

    a, b = name(left), name(right)
    if a == b:
        raise ValueError("A pair must contain distinct sources")
    return (a, b) if a < b else (b, a)


def pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return canonical_pair(pair["left"]["filename"], pair["right"]["filename"])


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def time_eligible(gap: Any) -> bool:
    return finite(gap) and 0 <= gap <= TIME_LIMIT


def merge_labels(
    known: list[list[str]], reviewed: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    manual: dict[tuple[str, str], bool] = {}

    def add(key: tuple[str, str], value: bool, origin: str) -> None:
        if key in labels and labels[key]["same_scene"] != value:
            raise ValueError(f"Conflicting labels for {key}")
        entry = labels.setdefault(key, dict(same_scene=value, origins=[]))
        if origin not in entry["origins"]:
            entry["origins"].append(origin)

    for left, right in known:
        add(canonical_pair(left, right), True, "known_positive")
    for pair in reviewed:
        value = pair.get("manual_same_scene")
        if value is None:
            continue
        if type(value) is not bool:
            raise ValueError("manual_same_scene must be true, false, or null")
        key = pair_key(pair)
        add(key, value, "manual")
        manual[key] = value
    return labels, dict(
        total_known_positives=len({canonical_pair(*p) for p in known}),
        total_manual_positives=sum(manual.values()),
        total_manual_negatives=sum(not value for value in manual.values()),
        total_unique_labeled_pairs=len(labels),
    )


def derive_features(
    pair: dict[str, Any], left_samples: int | None, right_samples: int | None
) -> tuple[dict[str, Any], list[str]]:
    signals = pair["signals"]
    features = {k: v for k, v in signals.items() if k != "orb_frame_matches"}
    normalized = (
        "orb_best_inliers_per_min_keypoints",
        "orb_best_good_matches_per_min_keypoints",
    )
    support = [f"orb_frame_pair_count_with_inliers_ge_{n}" for n in (4, 8, 12)]
    coverage = [
        "orb_source_frame_coverage",
        "orb_source_frame_coverage_left",
        "orb_source_frame_coverage_right",
    ]
    for field in (*normalized, *support, *coverage):
        features[field] = None
    matches = signals.get("orb_frame_matches")
    if not isinstance(matches, list) or not matches:
        return features, [
            "No frame-level ORB matches; normalized counts/support/coverage unavailable"
        ]
    required = (
        "orb_keypoints_left",
        "orb_keypoints_right",
        "orb_inlier_count",
        "orb_good_match_count",
    )
    if any(any(not finite(m.get(k)) or m[k] < 0 for k in required) for m in matches):
        return features, ["Incomplete frame-level counts/keypoints"]
    for field, count in zip(
        normalized, ("orb_inlier_count", "orb_good_match_count"), strict=True
    ):
        features[field] = max(
            min(
                1.0,
                m[count]
                / max(1, min(m["orb_keypoints_left"], m["orb_keypoints_right"])),
            )
            for m in matches
        )
    physical: dict[tuple[int, int], float] = {}
    for match in matches:
        i, j = match.get("query_frame_index"), match.get("target_frame_index")
        direction = match.get("direction")
        if (
            type(i) is not int
            or type(j) is not int
            or i < 0
            or j < 0
            or direction not in ("left_to_right", "right_to_left")
        ):
            return features, [
                "Frame identities/directions missing; support/coverage unavailable"
            ]
        if direction == "right_to_left":
            i, j = j, i
        key = (i, j)
        physical[key] = max(physical.get(key, 0), match["orb_inlier_count"])
    for field, n in zip(support, (4, 8, 12), strict=True):
        features[field] = sum(count >= n for count in physical.values())
    if not left_samples or not right_samples:
        return features, ["Sampled frame counts unavailable; coverage unavailable"]
    if any(i >= left_samples or j >= right_samples for i, j in physical):
        raise ValueError("Frame index exceeds source sampled-frame count")
    supported = [key for key, count in physical.items() if count >= 4]
    left = len({i for i, _ in supported}) / left_samples
    right = len({j for _, j in supported}) / right_samples
    features.update(
        orb_source_frame_coverage_left=left,
        orb_source_frame_coverage_right=right,
        orb_source_frame_coverage=min(left, right),
    )
    return features, []


def candidate_rules() -> list[dict[str, Any]]:
    """Small fixed search; all candidates also require the inclusive time gate."""
    rules: list[dict[str, Any]] = []

    def add(name: str, family: str, clauses: list[list[tuple[str, float]]]) -> None:
        rules.append(
            dict(
                rule_id=name,
                family=family,
                any_of=[
                    [
                        {"feature": feature, "minimum": minimum}
                        for feature, minimum in clause
                    ]
                    for clause in clauses
                ],
            )
        )

    counts = (4, 6, 8, 10, 12, 15, 20, 30, 50)
    for n in counts:
        add(f"inliers_{n}", "inlier_count", [[("orb_best_inlier_count", n)]])
        add(
            f"good_matches_{n}",
            "good_match_count",
            [[("orb_best_good_match_count", n)]],
        )
        for similarity in (0.3, 0.5, 0.7, 0.8):
            add(
                f"inliers_{n}_image_{similarity}",
                "image_and_orb",
                [
                    [
                        ("orb_best_inlier_count", n),
                        ("image_similarity_median", similarity),
                    ]
                ],
            )
    for n in (4, 6, 8):
        for similarity in (0.5, 0.7, 0.8):
            add(
                f"override_50_or_inliers_{n}_image_{similarity}",
                "strong_orb_override",
                [
                    [("orb_best_inlier_count", 50)],
                    [
                        ("orb_best_inlier_count", n),
                        ("image_similarity_median", similarity),
                    ],
                ],
            )
    for inliers in (4, 8, 12):
        for frame_pairs in (2, 5, 10):
            add(
                f"support_{inliers}_pairs_{frame_pairs}",
                "multi_frame_support",
                [[(f"orb_frame_pair_count_with_inliers_ge_{inliers}", frame_pairs)]],
            )
    for n in (4, 6, 8):
        for fraction in (0.4, 0.6, 0.8):
            add(
                f"inliers_{n}_coverage_{fraction}",
                "frame_coverage",
                [
                    [
                        ("orb_best_inlier_count", n),
                        ("orb_source_frame_coverage", fraction),
                    ]
                ],
            )
    for ratio in (0.01, 0.02, 0.05, 0.1):
        add(
            f"inliers_4_normalized_{ratio}",
            "normalized_inliers",
            [
                [
                    ("orb_best_inlier_count", 4),
                    ("orb_best_inliers_per_min_keypoints", ratio),
                ]
            ],
        )
    return rules


def predict(rule: dict[str, Any], pair: dict[str, Any]) -> bool | None:
    if not time_eligible(pair.get("capture_time_gap_seconds")):
        return False
    unknown = False
    for clause in rule["any_of"]:
        values = [(pair["features"].get(c["feature"]), c["minimum"]) for c in clause]
        if any(finite(value) and value < minimum for value, minimum in values):
            continue
        if all(finite(value) for value, _ in values):
            return True
        unknown = True
    return None if unknown else False


def evaluate_rule(rule: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fn = tn = fp = 0
    false_positives, false_negatives, abstentions = [], [], []
    for pair in pairs:
        prediction = predict(rule, pair)
        if prediction is None:
            abstentions.append(pair)
        elif pair["same_scene"]:
            if prediction:
                tp += 1
            else:
                fn += 1
                false_negatives.append(pair)
        elif prediction:
            fp += 1
            false_positives.append(pair)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(
        rule,
        true_positives=tp,
        false_negatives=fn,
        true_negatives=tn,
        false_positives=fp,
        precision=precision,
        recall=recall,
        f1=f1,
        mismatched_positive_pairs=false_negatives,
        mismatched_negative_pairs=false_positives,
        abstained_pairs=abstentions,
    )


def missing_pair_coverage(
    sources: list[dict[str, Any]],
    existing: set[tuple[str, str]],
    old_neighbor_distance: int | None = None,
) -> dict[str, Any]:
    valid = [
        s for s in sources if finite(s.get("capture_time")) and s["capture_time"] > 0
    ]
    valid.sort(key=lambda s: (s["capture_time"], s["filename"].casefold()))
    source_indices = {s["filename"].casefold(): i for i, s in enumerate(sources)}
    eligible, missing = [], []
    for i, left in enumerate(valid):
        for j in range(i + 1, len(valid)):
            right = valid[j]
            gap = right["capture_time"] - left["capture_time"]
            if gap > TIME_LIMIT:
                break
            entry = dict(
                left=left["filename"],
                right=right["filename"],
                capture_time_gap_seconds=gap,
                source_distance=abs(
                    source_indices[left["filename"].casefold()]
                    - source_indices[right["filename"].casefold()]
                ),
            )
            eligible.append(entry)
            if canonical_pair(left["filename"], right["filename"]) not in existing:
                missing.append(entry)
    return dict(
        potential_short_gap_pair_count=len(eligible),
        missing_short_gap_pair_count=len(missing),
        missing_due_to_old_neighbor_limit=sum(
            p["source_distance"] > old_neighbor_distance for p in missing
        )
        if old_neighbor_distance is not None
        else None,
        missing_pairs=missing,
        sources_without_capture_time=len(sources) - len(valid),
        note="Production candidates should use all <=180 s pairs, without a "
        "source-distance gate. Missing pairs have no visual measurements; "
        "their labels and rule behavior cannot be inferred.",
    )


def calibrate_data(
    analysis: dict[str, Any],
    review: dict[str, Any],
    known: list[list[str]] | None = None,
) -> dict[str, Any]:
    known = KNOWN_POSITIVES if known is None else known
    labels, counts = merge_labels(known, review["pairs"])
    sources = {s["filename"].casefold(): s for s in analysis["sources"]}
    if len(sources) != len(analysis["sources"]):
        raise ValueError("Ambiguous source filenames")
    measured: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in analysis["pairs"]:
        key = pair_key(pair)
        if key in measured:
            raise ValueError(f"Duplicate analysis pair {key}")
        measured[key] = pair
    dataset, excluded, unavailable = [], [], []
    for key, label in sorted(labels.items()):
        pair = measured.get(key)
        if pair is None:
            unavailable.append(
                dict(pair=list(key), **label, reason="Pair absent from analysis")
            )
            continue
        entry = dict(
            left=pair["left"]["filename"],
            right=pair["right"]["filename"],
            capture_time_gap_seconds=pair.get("capture_time_gap_seconds"),
            **label,
        )
        if not time_eligible(entry["capture_time_gap_seconds"]):
            excluded.append(
                dict(entry, reason="Outside <=180 s gate or gap unavailable")
            )
            continue
        if pair.get("analysis_status") != "ok":
            unavailable.append(dict(entry, reason="Visual measurements unavailable"))
            continue
        left = sources.get(entry["left"].casefold(), {})
        right = sources.get(entry["right"].casefold(), {})
        features, missing = derive_features(
            pair, left.get("sampled_frame_count"), right.get("sampled_frame_count")
        )
        dataset.append(dict(entry, features=features, unavailable_features=missing))
    candidates = [evaluate_rule(rule, dataset) for rule in candidate_rules()]
    high_recall = [
        c
        for c in candidates
        if c["recall"] is not None
        and c["recall"] >= 0.85
        and c["precision"] is not None
        and c["precision"] >= 0.5
        and not c["abstained_pairs"]
    ]
    high_recall.sort(key=lambda c: (-c["recall"], -c["precision"], c["rule_id"]))
    positive_count = sum(p["same_scene"] for p in dataset)
    baseline_precision = positive_count / len(dataset) if dataset else None
    full_recall = [
        c for c in candidates if c["recall"] == 1 and not c["abstained_pairs"]
    ]
    tradeoffs: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["abstained_pairs"]:
            continue
        confusion_key = tuple(
            candidate[k]
            for k in (
                "true_positives",
                "false_negatives",
                "true_negatives",
                "false_positives",
            )
        )
        group = tradeoffs.setdefault(
            confusion_key,
            dict(
                **{
                    k: candidate[k]
                    for k in (
                        "true_positives",
                        "false_negatives",
                        "true_negatives",
                        "false_positives",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
                equivalent_rule_ids=[],
            ),
        )
        group["equivalent_rule_ids"].append(candidate["rule_id"])
    behavior = [
        dict(
            pair, candidate_results={c["rule_id"]: predict(c, pair) for c in candidates}
        )
        for pair in dataset
    ]
    return dict(
        schema_version=1,
        production_rule=None,
        production_action=None,
        product_note="Future action is review color tag #1 only. This report does "
        "not tag clips, change rules, or select a production policy.",
        interpretation="Exploratory results on a small, top-ranked review sample. "
        "Precision/recall are conditional on measured, time-eligible labeled pairs; "
        "they do not estimate corpus performance. Abstentions are reported separately.",
        label_dataset_summary=counts,
        time_filter_summary=dict(
            inclusive_max_gap_seconds=TIME_LIMIT,
            total_analysis_pairs=len(measured),
            time_eligible_analysis_pairs=sum(
                time_eligible(p.get("capture_time_gap_seconds"))
                for p in measured.values()
            ),
            remaining_labeled_pairs=len(dataset),
            remaining_known_positives=sum(
                "known_positive" in p["origins"] for p in dataset
            ),
            remaining_manual_positives=sum(
                "manual" in p["origins"] and p["same_scene"] for p in dataset
            ),
            remaining_manual_negatives=sum(not p["same_scene"] for p in dataset),
            excluded_labeled_pairs=excluded,
            known_positives_excluded=[
                p for p in excluded if "known_positive" in p["origins"]
            ],
            unavailable_labeled_pairs=unavailable,
        ),
        derived_feature_definitions=FEATURE_DEFINITIONS,
        labeled_dataset=dataset,
        candidate_rules=candidates,
        high_recall_shortlist=[c["rule_id"] for c in high_recall[:10]],
        distinct_observed_tradeoffs=list(tradeoffs.values()),
        calibration_diagnostics=dict(
            positive_label_fraction=baseline_precision,
            full_recall_rule_count=len(full_recall),
            full_recall_rules_rejecting_any_manual_negative=[
                c["rule_id"] for c in full_recall if c["true_negatives"] > 0
            ],
            coverage_values_on_labeled_pairs=sorted(
                {
                    p["features"]["orb_source_frame_coverage"]
                    for p in dataset
                    if p["features"]["orb_source_frame_coverage"] is not None
                }
            ),
            note="Compare full-recall precision with the positive-label fraction: "
            "matching every eligible labeled pair provides no visual discrimination. "
            "No production rule is chosen. Missing-pair measurements and broader "
            "manual labels are needed before deciding whether new features are needed.",
        ),
        known_positive_behavior=[
            p for p in behavior if "known_positive" in p["origins"]
        ],
        manual_positive_behavior=[
            p for p in behavior if "manual" in p["origins"] and p["same_scene"]
        ],
        manual_hard_negative_behavior=[p for p in behavior if not p["same_scene"]],
        missing_pair_coverage=missing_pair_coverage(
            analysis["sources"],
            set(measured),
            analysis.get("config", {}).get("max_neighbor_distance"),
        ),
    )


def duplicate_calibrate(
    analysis_path: str | Path, review_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    analysis_path, review_path, output = map(
        Path, (analysis_path, review_path, output_path)
    )
    if output.exists() or output.resolve() in {
        analysis_path.resolve(),
        review_path.resolve(),
    }:
        raise ProjectParseError("Output must be a new file distinct from JSON inputs")
    try:
        analysis_bytes, review_bytes = (
            analysis_path.read_bytes(),
            review_path.read_bytes(),
        )
        result = calibrate_data(json.loads(analysis_bytes), json.loads(review_bytes))
        result["inputs"] = dict(
            analysis=str(analysis_path),
            review=str(review_path),
            analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
            review_sha256=hashlib.sha256(review_bytes).hexdigest(),
        )
        result["runtime_seconds"] = time.perf_counter() - started
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return dict(
            output=str(output),
            runtime_seconds=time.perf_counter() - started,
            labeled_pairs=result["time_filter_summary"]["remaining_labeled_pairs"],
            candidate_rule_count=len(result["candidate_rules"]),
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ProjectParseError(str(error)) from error
