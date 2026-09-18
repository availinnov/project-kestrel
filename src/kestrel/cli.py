"""Command-line interface for kestrel."""

import argparse
import json
from importlib.metadata import version
from typing import Any

from kestrel.audio_calibration import audio_calibrate
from kestrel.audio_gain_evaluation import audio_gain_evaluate
from kestrel.dataset import extract_dataset
from kestrel.formats.diff import diff_files
from kestrel.formats.inspect import inspect_file
from kestrel.media_analysis import media_analyze
from kestrel.media_detectors import DetectorConfig
from kestrel.project import ProjectParseError, parse_project, project_summary
from kestrel.project.semantic import semantic_diff_files
from kestrel.project.sequence import apply_plan, build_sequence
from kestrel.project.template import instantiate_template
from kestrel.project.writer import (
    ProjectWriteError,
    clone_video_track,
    set_audio_gain,
    set_clip_state,
    set_trim,
)
from kestrel.rules_evaluation import rules_evaluate


def render(result: dict[str, Any]) -> str:
    """Render metadata and structural changes in a readable form."""
    lines = []
    for key, value in result.items():
        if key == "unified_diff":
            lines.append(str(value))
        elif isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=True, indent=2)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(prog="kestrel")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('project-kestrel')}"
    )
    commands = parser.add_subparsers(dest="command")
    inspect_parser = commands.add_parser(
        "inspect", help="Inspect file metadata and structure"
    )
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--json", action="store_true", help="Output JSON")
    diff_parser = commands.add_parser(
        "diff", help="Compare file contents and structure"
    )
    diff_parser.add_argument("left")
    diff_parser.add_argument("right")
    diff_parser.add_argument("--json", action="store_true", help="Output JSON")
    diff_parser.add_argument(
        "--semantic", action="store_true", help="Compare meaningful project structure"
    )
    summary_parser = commands.add_parser(
        "project-summary", help="Summarize a structured project container"
    )
    summary_parser.add_argument("path")
    summary_parser.add_argument("--json", action="store_true", help="Output JSON")
    gain_parser = commands.add_parser(
        "project-set-audio-gain", help="Write a copy with one audio gain change"
    )
    gain_parser.add_argument("input")
    gain_parser.add_argument("output")
    gain_parser.add_argument("--db", type=float, required=True)
    trim_parser = commands.add_parser(
        "project-set-trim", help="Write a trimmed single-source project copy"
    )
    trim_parser.add_argument("input")
    trim_parser.add_argument("output")
    trim_parser.add_argument(
        "--left-seconds",
        required=True,
        help="Amount removed from the current source start",
    )
    trim_parser.add_argument(
        "--right-seconds",
        required=True,
        help="Amount removed from the current source end",
    )
    state_parser = commands.add_parser(
        "project-set-clip-state",
        help="Write a copy with source pair state and numeric tag changes",
    )
    state_parser.add_argument("input")
    state_parser.add_argument("output")
    state_group = state_parser.add_mutually_exclusive_group()
    state_group.add_argument(
        "--enable", dest="enable", action="store_const", const=True
    )
    state_group.add_argument(
        "--disable", dest="enable", action="store_const", const=False
    )
    state_parser.set_defaults(enable=None)
    state_parser.add_argument(
        "--color-tag",
        type=int,
        choices=range(1, 14),
        help="Stored numeric tag (1 through 13)",
    )
    clone_parser = commands.add_parser(
        "project-clone-video-track",
        help="Write a disabled source clone on a compatible or new track",
    )
    clone_parser.add_argument("input")
    clone_parser.add_argument("output")
    media_parser = commands.add_parser(
        "media-analyze", help="Measure source signals and propose rule actions"
    )
    media_parser.add_argument("input")
    media_parser.add_argument("output")
    media_parser.add_argument("--rules")
    media_parser.add_argument("--ffmpeg", default="ffmpeg")
    media_parser.add_argument("--max-frames", type=int, default=24)
    media_parser.add_argument("--sample-rate", type=float, default=1.0)
    media_parser.add_argument("--black-luma-threshold", type=float, default=0.05)
    media_parser.add_argument("--high-peak-dbfs", type=float, default=-3.0)
    media_parser.add_argument("--relative-loud-window-db", type=float, default=12.0)
    media_parser.add_argument("--short-loud-event-max-seconds", type=float, default=1.0)
    media_parser.add_argument(
        "--filename",
        action="append",
        help="Analyze only this source basename; may be repeated",
    )
    media_parser.add_argument("--report", help="Write per-source measurement report")
    rules_parser = commands.add_parser(
        "rules-evaluate", help="Re-evaluate rules over saved analysis signals"
    )
    rules_parser.add_argument("analysis")
    rules_parser.add_argument("rules")
    rules_parser.add_argument("output")
    calibration_parser = commands.add_parser(
        "audio-calibrate", help="Calibrate source RMS against manual fragment gains"
    )
    calibration_parser.add_argument("dataset")
    calibration_parser.add_argument("analysis")
    calibration_parser.add_argument("output")
    calibration_parser.add_argument(
        "--exclude-filename",
        action="append",
        help="Exclude this source basename from calibration; may be repeated",
    )
    gain_evaluate_parser = commands.add_parser(
        "audio-gain-evaluate", help="Cross-validate deterministic audio gain models"
    )
    gain_evaluate_parser.add_argument("calibration")
    gain_evaluate_parser.add_argument("output")
    dataset_parser = commands.add_parser(
        "dataset-extract", help="Extract source usage and retained fragments"
    )
    dataset_parser.add_argument("input")
    dataset_parser.add_argument("output")
    plan_parser = commands.add_parser(
        "project-apply-plan", help="Apply source keep/drop and trim decisions"
    )
    plan_parser.add_argument("input")
    plan_parser.add_argument("plan")
    plan_parser.add_argument("output")
    sequence_parser = commands.add_parser(
        "project-build-sequence",
        help="Build an experimental sequence from existing sources",
    )
    sequence_parser.add_argument("input")
    sequence_parser.add_argument("output")
    sequence_parser.add_argument("--count", type=int, required=True)
    template_parser = commands.add_parser(
        "project-instantiate-template",
        help="Create a fresh empty project from a template",
    )
    template_parser.add_argument("template")
    template_parser.add_argument("output")
    template_parser.add_argument("--name", required=True)
    template_parser.add_argument("--width", type=int)
    template_parser.add_argument("--height", type=int)
    template_parser.add_argument("--fps", type=int)
    parser.set_defaults(json=False)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "inspect":
            result = inspect_file(args.path)
        elif args.command == "project-summary":
            result = project_summary(parse_project(args.path))
        elif args.command == "project-set-audio-gain":
            result = set_audio_gain(args.input, args.output, args.db)
        elif args.command == "project-set-trim":
            result = set_trim(
                args.input, args.output, args.left_seconds, args.right_seconds
            )
        elif args.command == "project-set-clip-state":
            result = set_clip_state(
                args.input, args.output, enable=args.enable, color_tag=args.color_tag
            )
        elif args.command == "project-clone-video-track":
            result = clone_video_track(args.input, args.output)
        elif args.command == "media-analyze":
            try:
                config = DetectorConfig(
                    max_frames=args.max_frames,
                    sample_rate=args.sample_rate,
                    black_luma_threshold=args.black_luma_threshold,
                    high_peak_dbfs=args.high_peak_dbfs,
                    relative_loud_window_db=args.relative_loud_window_db,
                    short_loud_event_max_seconds=args.short_loud_event_max_seconds,
                )
            except ValueError as error:
                raise ProjectParseError(str(error)) from error
            result = media_analyze(
                args.input,
                args.output,
                args.rules,
                ffmpeg=args.ffmpeg,
                config=config,
                filenames=args.filename,
                report_path=args.report,
            )
        elif args.command == "rules-evaluate":
            result = rules_evaluate(args.analysis, args.rules, args.output)
        elif args.command == "audio-calibrate":
            result = audio_calibrate(
                args.dataset,
                args.analysis,
                args.output,
                exclude_filenames=args.exclude_filename,
            )
        elif args.command == "audio-gain-evaluate":
            result = audio_gain_evaluate(args.calibration, args.output)
        elif args.command == "dataset-extract":
            result = extract_dataset(args.input, args.output)
        elif args.command == "project-apply-plan":
            result = apply_plan(args.input, args.plan, args.output)
        elif args.command == "project-build-sequence":
            result = build_sequence(args.input, args.output, args.count)
        elif args.command == "project-instantiate-template":
            result = instantiate_template(
                args.template,
                args.output,
                name=args.name,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
        else:
            result = (
                semantic_diff_files(args.left, args.right)
                if args.semantic
                else diff_files(args.left, args.right)
            )
    except (OSError, ProjectParseError, ProjectWriteError) as error:
        result = {"error": str(error)}
    print(
        json.dumps(result, ensure_ascii=True, indent=2) if args.json else render(result)
    )
    if "error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
