"""Deterministic cached ORB features and neutral geometric matching signals."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from kestrel.audio_calibration import percentile

ORB_PARAMETERS = dict(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)
ORB_DEFINITIONS = {
    "opencv_version": cv2.__version__,
    "orb_parameters": ORB_PARAMETERS,
    "orb_working_size": "320x180 grayscale, aspect-preserving with black padding",
    "orb_matching": "Directional BFMatcher NORM_HAMMING, k=2; strict ratio 0.75",
    "orb_good_match_ratio": "good count / max(1, min(left count, right count)); "
    "directional matches can reuse target descriptors, so ratio can exceed 1",
    "orb_homography": "At least 4 good matches; RANSAC 5.0 px, maxIters=2000, "
    "confidence=0.995; RNG seed=0 before every estimate; one OpenCV thread",
    "orb_inlier_ratio": "inlier count / good count; count=0 and ratio=0 if "
    "insufficient matches, missing descriptors, or homography estimation fails",
    "orb_source_summary": "All frame combinations in both directions. Independent "
    "maximum for each best metric; median of per-query-frame best ratios, "
    "pooled across both directions. Maxima need not describe the same frame pair.",
}


@dataclass(frozen=True)
class OrbFrame:
    points: Any
    descriptors: Any

    @property
    def count(self) -> int:
        return len(self.points)


def describe_orb(image: Any) -> OrbFrame:
    cv2.setNumThreads(1)
    orb = cv2.ORB_create(**ORB_PARAMETERS)  # type: ignore[attr-defined]
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return OrbFrame(
        np.asarray([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2),
        descriptors,
    )


def match_orb(left: OrbFrame, right: OrbFrame) -> dict[str, Any]:
    good = []
    if (
        left.descriptors is not None
        and right.descriptors is not None
        and right.count >= 2
    ):
        candidates = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            left.descriptors, right.descriptors, k=2
        )
        good = [
            m
            for neighbors in candidates
            if len(neighbors) == 2
            for m, n in [neighbors]
            if m.distance < 0.75 * n.distance
        ]
    inliers = 0
    if len(good) >= 4:
        cv2.setRNGSeed(0)
        try:
            homography, mask = cv2.findHomography(
                np.asarray([left.points[m.queryIdx] for m in good], dtype=np.float32),
                np.asarray([right.points[m.trainIdx] for m in good], dtype=np.float32),
                cv2.RANSAC,
                5.0,
                maxIters=2000,
                confidence=0.995,
            )
            if (
                homography is not None
                and mask is not None
                and np.isfinite(homography).all()
            ):
                inliers = int(mask.sum())
        except cv2.error:
            pass
    return dict(
        orb_keypoints_left=left.count,
        orb_keypoints_right=right.count,
        orb_good_match_count=len(good),
        orb_good_match_ratio=len(good) / max(1, min(left.count, right.count)),
        orb_inlier_count=inliers,
        orb_inlier_ratio=inliers / len(good) if good else 0.0,
    )


def compare_orb(left: list[OrbFrame], right: list[OrbFrame]) -> dict[str, Any]:
    matrices = [
        [[match_orb(a, b) for b in targets] for a in queries]
        for queries, targets in ((left, right), (right, left))
    ]
    rows = [row for matrix in matrices for row in matrix]
    all_matches = [match for row in rows for match in row]
    summary = {
        "orb_best_" + metric: max(m["orb_" + metric] for m in all_matches)
        for metric in (
            "good_match_count",
            "good_match_ratio",
            "inlier_count",
            "inlier_ratio",
        )
    }
    for name, metric in (
        ("orb_match_ratio_median", "orb_good_match_ratio"),
        ("orb_inlier_ratio_median", "orb_inlier_ratio"),
    ):
        summary[name] = percentile(
            sorted(max(m[metric] for m in row) for row in rows), 0.5
        )
    summary["orb_frame_matches"] = [
        dict(query_frame_index=i, target_frame_index=j, direction=direction, **match)
        for direction, matrix in zip(
            ("left_to_right", "right_to_left"), matrices, strict=True
        )
        for i, row in enumerate(matrix)
        for j, match in enumerate(row)
    ]
    return summary
