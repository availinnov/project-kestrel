"""Offline control evaluation, kept separate from descriptor measurements."""

import argparse
import json
from pathlib import Path
from typing import Any

from kestrel.duplicate_calibration import canonical_pair, merge_labels, pair_key


def evaluate_controls(
    analysis: dict[str, Any], positive_pairs: list[list[str]]
) -> dict[str, Any]:
    """Evaluate known positives separately from controls without manual labels."""
    positives = {canonical_pair(*pair) for pair in positive_pairs}
    measured = []
    observed = set()
    for pair in analysis["pairs"]:
        key = pair_key(pair)
        observed.add(key)
        if pair["analysis_status"] == "ok":
            measured.append(
                dict(
                    pair,
                    signals={
                        k: v
                        for k, v in pair["signals"].items()
                        if k != "orb_frame_matches"
                    },
                    evaluation_label=("positive" if key in positives else "unlabeled"),
                )
            )
    thresholds = []
    threshold_specs: list[tuple[str, list[float], bool]] = [
        ("image_similarity_median", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], True),
        ("hash_best_match_median", [4, 8, 12, 16, 20, 24], False),
    ]
    if measured and "orb_best_inlier_ratio" in measured[0]["signals"]:
        threshold_specs += [
            ("orb_best_inlier_ratio", [0.25, 0.5, 0.75, 0.9], True),
            ("orb_best_good_match_ratio", [0.01, 0.025, 0.05, 0.1, 0.2], True),
        ]
    for signal, candidates, higher in threshold_specs:
        for threshold in candidates:
            matched = [
                p
                for p in measured
                if (
                    p["signals"][signal] >= threshold
                    if higher
                    else p["signals"][signal] <= threshold
                )
            ]
            tp = sum(p["evaluation_label"] == "positive" for p in matched)
            unlabeled_matched = len(matched) - tp
            total_positive = sum(p["evaluation_label"] == "positive" for p in measured)
            thresholds.append(
                dict(
                    signal=signal,
                    threshold=threshold,
                    positive_controls_matched=tp,
                    known_positives_missed=total_positive - tp,
                    unlabeled_pairs_matched=unlabeled_matched,
                    matched_unlabeled_pairs=[
                        p for p in matched if p["evaluation_label"] != "positive"
                    ],
                    missed_positive_pairs=[
                        p
                        for p in measured
                        if p["evaluation_label"] == "positive" and p not in matched
                    ],
                )
            )
    rankings = {}
    ranking_specs = [
        ("image_similarity_median", True),
        ("hash_best_match_median", False),
    ]
    if measured and "orb_best_inlier_ratio" in measured[0]["signals"]:
        ranking_specs += [
            ("orb_best_inlier_ratio", True),
            ("orb_best_good_match_ratio", True),
        ]
    for signal, reverse in ranking_specs:
        ordered = sorted(
            measured,
            key=lambda p: (
                p["signals"][signal],
                p["signals"].get("orb_best_inlier_count", 0)
                if signal.startswith("orb_")
                else 0,
            ),
            reverse=reverse,
        )
        rankings[signal] = dict(
            top_15=ordered[:15],
            top_20=ordered[:20],
            least_similar_5=ordered[-5:],
            nearest_controls=[
                p for p in ordered if p["evaluation_label"] != "positive"
            ][:15],
        )
    return dict(
        **analysis["summary"],
        evaluation_note="Other neighbors are unlabeled controls, not confirmed "
        "negatives. No false-positive rate or production threshold is calculated.",
        positive_controls=[p for p in measured if p["evaluation_label"] == "positive"],
        unlabeled_controls=[
            p for p in measured if p["evaluation_label"] == "unlabeled"
        ],
        ungenerated_positive_pairs=[list(p) for p in sorted(positives - observed)],
        unmeasured_positive_pairs=[
            p
            for p in analysis["pairs"]
            if p["analysis_status"] != "ok" and pair_key(p) in positives
        ],
        rankings=rankings,
        threshold_exploration=thresholds,
    )


def write_review_queue(
    report: dict[str, Any],
    output: Path,
    *,
    previous_analysis: dict[str, Any] | None = None,
    reviewed_pairs: list[dict[str, Any]] | None = None,
) -> None:
    """Exclusively create a queue; preserve every existing file including labels."""
    candidates = report["unlabeled_controls"]
    labels, _ = merge_labels([], reviewed_pairs or [])
    previous_keys = (
        {pair_key(p) for p in previous_analysis["pairs"]}
        if previous_analysis is not None
        else set()
    )
    candidates = [
        p
        for p in candidates
        if pair_key(p) not in labels and pair_key(p) not in previous_keys
    ]
    ordered = sorted(
        candidates,
        key=lambda p: (
            p["signals"]["orb_best_inlier_count"],
            p["signals"]["orb_best_good_match_ratio"],
            p["signals"]["orb_best_inlier_ratio"],
        ),
        reverse=True,
    )
    queue = dict(
        ranking="orb_best_inlier_count, then good-match ratio, then inlier ratio; "
        "counts prioritize geometric support over ratios from very few matches",
        pairs=[
            dict(p, manual_same_scene=None)
            for p in (ordered if previous_analysis is not None else ordered[:20])
        ],
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(queue, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument(
        "labels", type=Path, help="JSON list of positive filename pairs"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument(
        "--previous-analysis",
        type=Path,
        help="Queue only newly measured pairs absent from this analysis",
    )
    parser.add_argument(
        "--review-labels",
        type=Path,
        action="append",
        help="Preserve labels from an existing queue; repeatable",
    )
    args = parser.parse_args()
    input_paths = {args.analysis.resolve(), args.labels.resolve()}
    input_paths.update(p.resolve() for p in args.review_labels or [])
    if args.previous_analysis is not None:
        input_paths.add(args.previous_analysis.resolve())
    if args.output.exists() or args.output.resolve() in input_paths:
        parser.error("Output must be a new file distinct from inputs")
    positives = json.loads(args.labels.read_text(encoding="utf-8"))
    reviewed = [
        p
        for path in args.review_labels or []
        for p in json.loads(path.read_text(encoding="utf-8"))["pairs"]
    ]
    merged, counts = merge_labels(positives, reviewed)
    report = evaluate_controls(
        json.loads(args.analysis.read_text(encoding="utf-8")),
        positives,
    )
    report["preserved_review_label_summary"] = counts
    report["preserved_review_labels"] = [
        dict(pair=list(key), **value) for key, value in sorted(merged.items())
    ]
    if args.previous_analysis is not None:
        previous = json.loads(args.previous_analysis.read_text(encoding="utf-8"))
        previous_keys = {pair_key(p) for p in previous["pairs"]}
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        newly_measured = [
            p for p in analysis["pairs"] if pair_key(p) not in previous_keys
        ]
        report["measurement_coverage"] = dict(
            previously_measured_candidate_pair_count=len(analysis["pairs"])
            - len(newly_measured),
            newly_measured_pair_count=len(newly_measured),
            newly_measured_pairs=[
                dict(
                    p,
                    signals={
                        k: v
                        for k, v in p["signals"].items()
                        if k != "orb_frame_matches"
                    },
                )
                for p in newly_measured
            ],
        )
    if args.review_queue is not None:
        if args.review_queue.exists() or args.review_queue.resolve() in input_paths | {
            args.output.resolve()
        }:
            parser.error("Review queue must be a new file distinct from inputs/output")
        previous = (
            json.loads(args.previous_analysis.read_text(encoding="utf-8"))
            if args.previous_analysis is not None
            else None
        )
        write_review_queue(
            report,
            args.review_queue,
            previous_analysis=previous,
            reviewed_pairs=reviewed,
        )
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
