"""Command-line interface for kestrel."""

import argparse
from importlib.metadata import version


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(prog="kestrel")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('project-kestrel')}",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
