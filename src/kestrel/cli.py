"""Command-line interface for kestrel."""

import argparse
import json
from importlib.metadata import version
from typing import Any

from kestrel.formats.diff import diff_files
from kestrel.formats.inspect import inspect_file
from kestrel.project import ProjectParseError, parse_project, project_summary
from kestrel.project.semantic import semantic_diff_files


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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "inspect":
            result = inspect_file(args.path)
        elif args.command == "project-summary":
            result = project_summary(parse_project(args.path))
        else:
            result = (
                semantic_diff_files(args.left, args.right)
                if args.semantic
                else diff_files(args.left, args.right)
            )
    except (OSError, ProjectParseError) as error:
        result = {"error": str(error)}
    print(
        json.dumps(result, ensure_ascii=True, indent=2) if args.json else render(result)
    )
    if "error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
