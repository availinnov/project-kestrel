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
                    evaluation_label=(
                        "positive" if key in positives else "unlabeled"
                    ),
                )
            )
    thresholds = []
    for signal, candidates, higher in (
        ("image_similarity_median", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95], True),
        ("hash_best_match_median", [4, 8, 12, 16, 20, 24], False),
    ):
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
    for signal, reverse in (
        ("image_similarity_median", True),
        ("hash_best_match_median", False),
    ):
        ordered = sorted(measured, key=lambda p: p["signals"][signal], reverse=reverse)
        rankings[signal] = dict(
            top_15=ordered[:15],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument(
        "labels", type=Path, help="JSON list of positive filename pairs"
    )
    parser.add_argument("output", type=Path)
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
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
