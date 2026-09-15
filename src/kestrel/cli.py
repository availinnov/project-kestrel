"""Command-line interface for kestrel."""

import argparse
import json
from importlib.metadata import version
from typing import Any

from kestrel.formats.diff import diff_files
from kestrel.formats.inspect import inspect_file
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
