"""Command line: ``python -m tools.coverage run [--backend B] [--direction input|output]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import features, report, runner
from .model import DIRECTIONS
from .upstream import UpstreamError

LANGUAGES = sorted(features.PARSERS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.coverage", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="probe every feature and write coverage.json")
    run.add_argument(
        "--backend", action="append", choices=LANGUAGES, help="repeatable; default: all"
    )
    run.add_argument("--direction", choices=DIRECTIONS, help="default: both")
    run.add_argument("--output", type=Path, default=Path("coverage.json"))
    run.add_argument(
        "--jobs", type=int, default=0, help="worker processes; default: up to 8"
    )
    run.add_argument(
        "--offline", action="store_true", help="use cached upstream sources only"
    )
    listing = commands.add_parser(
        "features", help="print the feature list of one language"
    )
    listing.add_argument("backend", choices=LANGUAGES)
    listing.add_argument("--offline", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> int:
    languages = args.backend or LANGUAGES
    directions = [args.direction] if args.direction else list(DIRECTIONS)
    jobs = []
    for language in languages:
        feature_list = features.load(language, offline=args.offline)
        for direction in directions:
            if direction == "input" and language not in runner.INPUT_BACKENDS:
                continue
            jobs += [(feature, direction, language) for feature in feature_list]
    results = runner.run(jobs, args.jobs or None)
    data = report.build(results, languages, directions)
    report.write(data, args.output)
    print(report.summary(data))
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        for feature in features.load(args.backend, offline=args.offline):
            print(f"{feature.group:<44} {feature.name}")
        return 0
    except UpstreamError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
