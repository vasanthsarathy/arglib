"""ArgLib command-line interface."""

from __future__ import annotations

import argparse
import json

from arglib import __version__
from arglib.io import load
from arglib.viz import to_dot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arglib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dot_parser = subparsers.add_parser("dot", help="Export a graph JSON file to DOT.")
    dot_parser.add_argument("path", help="Path to a graph JSON file.")

    diag_parser = subparsers.add_parser(
        "diagnostics",
        help="Print diagnostics for a graph JSON file.",
    )
    diag_parser.add_argument("path", help="Path to a graph JSON file.")

    subparsers.add_parser("version", help="Print the ArgLib version.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    graph = load(args.path)
    if args.command == "dot":
        print(to_dot(graph))
        return 0

    if args.command == "diagnostics":
        print(json.dumps(graph.diagnostics(), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
