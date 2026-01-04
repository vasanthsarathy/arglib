"""ArgLib command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arglib import __version__
from arglib.io import load, validate_graph_payload
from arglib.viz import to_dot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arglib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dot_parser = subparsers.add_parser("dot", help="Export a graph JSON file to DOT.")
    dot_parser.add_argument("path", help="Path to a graph JSON file.")
    dot_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the graph JSON before exporting.",
    )

    diag_parser = subparsers.add_parser(
        "diagnostics",
        help="Print diagnostics for a graph JSON file.",
    )
    diag_parser.add_argument("path", help="Path to a graph JSON file.")
    diag_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the graph JSON before computing diagnostics.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a graph JSON file against the schema."
    )
    validate_parser.add_argument("path", help="Path to a graph JSON file.")

    subparsers.add_parser("version", help="Print the ArgLib version.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "dot":
        graph = load(args.path, validate=args.validate)
        print(to_dot(graph))
        return 0

    if args.command == "diagnostics":
        graph = load(args.path, validate=args.validate)
        print(json.dumps(graph.diagnostics(), indent=2, sort_keys=True))
        return 0

    if args.command == "validate":
        payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
        validate_graph_payload(payload)
        print("OK")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
