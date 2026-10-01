"""Offline control evaluation, kept separate from descriptor measurements."""

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_controls(
    analysis: dict[str, Any], positive_pairs: list[list[str]]
) -> dict[str, Any]:
    """Evaluate known positives separately from controls without manual labels."""
    positives = {tuple(sorted(pair)) for pair in positive_pairs}
    measured = []
    observed = set()
    for pair in analysis["pairs"]:
        key = tuple(sorted((pair["left"]["filename"], pair["right"]["filename"])))
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
            if p["analysis_status"] != "ok"
            and tuple(sorted((p["left"]["filename"], p["right"]["filename"])))
            in positives
        ],
        rankings=rankings,
        threshold_exploration=thresholds,
    )


def write_review_queue(report: dict[str, Any], output: Path) -> None:
    """Exclusively create a queue; preserve every existing file including labels."""
    candidates = report["unlabeled_controls"]
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
        pairs=[dict(p, manual_same_scene=None) for p in ordered[:20]],
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
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {
        args.analysis.resolve(),
        args.labels.resolve(),
    }:
        parser.error("Output must be a new file distinct from inputs")
    report = evaluate_controls(
        json.loads(args.analysis.read_text(encoding="utf-8")),
        json.loads(args.labels.read_text(encoding="utf-8")),
    )
    if args.review_queue is not None:
        if args.review_queue.exists() or args.review_queue.resolve() in {
            args.analysis.resolve(),
            args.labels.resolve(),
            args.output.resolve(),
        }:
            parser.error("Review queue must be a new file distinct from inputs/output")
        write_review_queue(report, args.review_queue)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
