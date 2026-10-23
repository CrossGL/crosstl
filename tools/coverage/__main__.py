"""Command line: ``python -m tools.coverage run [--backend B] [--direction input|output]``.

``merge`` combines the reports of runners that measured different backends, such as the Linux and macOS CI jobs, and
rebuilds the CrossGL IR row from all of them. ``publish`` writes what CI publishes to the ``coverage`` branch,
``issues`` reconciles the tracking and bug issues with a report, and ``compare`` shows how a pull request's report
differs from the published one. ``tools`` lists the reference tool of every language and whether it is installed, so
you can tell before a local run which rows it will report as untested.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import compare, execution, features, issues, publish, report, runner, validators
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
    run.add_argument(
        "--feature",
        action="append",
        help="measure only the feature with this upstream name; repeatable",
    )
    run.add_argument("--output", type=Path, default=Path("coverage.json"))
    run.add_argument(
        "--jobs", type=int, default=0, help="worker processes; default: up to 8"
    )
    run.add_argument(
        "--offline", action="store_true", help="use cached upstream sources only"
    )
    run.add_argument(
        "--execute",
        action="store_true",
        help="also run supported compute probes on the conformance harness's runtimes and compare them with the "
        "native probe",
    )
    run.add_argument(
        "--strict",
        action="store_true",
        help="fail when a requested backend's validator is missing, instead of reporting its probes as untested, "
        "and with --execute when no runtime runs a language's executable probes",
    )
    merging = commands.add_parser(
        "merge",
        help="combine reports from several runners and rebuild the CrossGL IR row",
    )
    merging.add_argument("reports", nargs="+", type=Path)
    merging.add_argument("--output", type=Path, default=Path("coverage.json"))
    publishing = commands.add_parser(
        "publish",
        help="write coverage.json, coverage.svg, COVERAGE.md, history.csv and trend.svg for the coverage branch",
    )
    publishing.add_argument("report", type=Path)
    publishing.add_argument("--output-dir", type=Path, required=True)
    publishing.add_argument(
        "--history", type=Path, help="the previously published history.csv, if any"
    )
    publishing.add_argument("--repository", default="CrossGL/crosstl")
    reconciling = commands.add_parser(
        "issues", help="reconcile the coverage tracking and bug issues with a report"
    )
    reconciling.add_argument("report", type=Path)
    reconciling.add_argument("--repository", default="CrossGL/crosstl")
    reconciling.add_argument(
        "--dry-run", action="store_true", help="print the plan and change nothing"
    )
    reconciling.add_argument(
        "--new-bugs",
        type=int,
        default=issues.NEW_BUGS,
        help=f"most bug issues to open in one run; default: {issues.NEW_BUGS}",
    )
    comparing = commands.add_parser(
        "compare",
        help="print, as Markdown, how a report differs from the published one",
    )
    comparing.add_argument("published", type=Path)
    comparing.add_argument("report", type=Path)
    commands.add_parser(
        "tools", help="list each language's reference tool and the version found"
    )
    listing = commands.add_parser(
        "features", help="print the feature list of one language"
    )
    listing.add_argument("backend", choices=LANGUAGES)
    listing.add_argument("--offline", action="store_true")
    return parser


def _lowering_gaps(language: str) -> list[str]:
    """The tools a language's SPIR-V lowering needs that are not installed."""
    tools = (validators.SPIRV_COMPILERS[language][0], "spirv-dis")
    return [tool for tool in tools if not validators.command(tool)]


def _run(args: argparse.Namespace) -> int:
    languages = args.backend or LANGUAGES
    directions = [args.direction] if args.direction else list(DIRECTIONS)
    missing = [
        f"{language} ({validators.TOOLS[language]})"
        for language in languages
        if not validators.available(language)
    ] + [
        f"{language} lowering to SPIR-V ({', '.join(_lowering_gaps(language))})"
        for language in languages
        if language in validators.SPIRV_COMPILERS and not validators.can_lower(language)
    ]
    if (
        "spirv" in languages
        and "input" in directions
        and not validators.can_lower("glsl")
    ):
        missing.append(f"spirv input ({', '.join(_lowering_gaps('glsl'))})")
    if args.strict and missing:
        print("error: missing tools for " + "; ".join(missing), file=sys.stderr)
        return 2
    jobs = []
    for language in languages:
        feature_list = features.load(language, offline=args.offline)
        if args.feature:
            feature_list = [f for f in feature_list if f.name in args.feature]
        for direction in directions:
            backends = (
                runner.INPUT_BACKENDS
                if direction == "input"
                else runner.OUTPUT_BACKENDS
            )
            if language not in backends:
                continue
            jobs += [(feature, direction, language) for feature in feature_list]
    results = runner.run(jobs, args.jobs or None)
    if args.execute:
        results, used = execution.run(results)
        executed = sum(1 for r in results if r.runtime)
        print(f"executed {executed} probes on {', '.join(used) or 'no runtime'}")
        idle = execution.idle(results, used)
        if args.strict and idle:
            print(f"error: no runtime executes {', '.join(idle)}", file=sys.stderr)
            return 2
    data = report.build(results, languages, directions)
    report.write(data, args.output)
    print(report.summary(data))
    if args.feature:
        for result in results:
            detail = f": {result.detail}" if result.detail else ""
            print(
                f"{result.feature.language} {result.direction} {result.feature.name} {result.outcome}{detail}"
            )
    print(f"wrote {args.output}")
    return 0


def _merge(args: argparse.Namespace) -> int:
    try:
        data = report.merge([report.load(path) for path in args.reports])
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report.write(data, args.output)
    print(report.summary(data))
    print(f"wrote {args.output}")
    return 0


def _publish(args: argparse.Namespace) -> int:
    previous = ""
    if args.history and args.history.is_file():
        previous = args.history.read_text(encoding="utf-8")
    data = report.load(args.report)
    for path in publish.write_site(data, args.output_dir, previous, args.repository):
        print(f"wrote {path}")
    return 0


def _issues(args: argparse.Namespace) -> int:
    data = report.load(args.report)
    try:
        actions = issues.reconcile(
            data,
            issues.GitHub(args.repository),
            new_bugs=args.new_bugs,
            dry_run=args.dry_run,
        )
    except issues.GitHubError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{len(actions)} issue changes" + (" planned" if args.dry_run else ""))
    return 0


def _how_to_install(tool: str) -> str:
    if tool == "metal":
        return "needs macOS with Xcode's Metal toolchain: xcodebuild -downloadComponent MetalToolchain"
    variable = "CROSSTL_COVERAGE_" + re.sub(r"\W", "_", tool).upper()
    if tool == "rust-gpu":
        return f"set {variable} to a built rust-gpu checkout, see tools/coverage/rustgpu.py"
    return f"put {tool} on PATH or set {variable}"


def _tools() -> int:
    rows = [("language", "directions", "tool", "found")]
    for language in LANGUAGES:
        tool = validators.TOOLS[language]
        directions = ", ".join(
            direction
            for direction, backends in (
                ("output", runner.OUTPUT_BACKENDS),
                ("input", runner.INPUT_BACKENDS),
            )
            if language in backends
        )
        if validators.available(language):
            found = validators.version(tool) or "yes, no version reported"
        else:
            found = "no: " + _how_to_install(tool)
        rows.append((language, directions, tool, found))
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    for row in rows:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
            + "  "
            + row[3]
        )
    lowering = ", ".join(
        f"{language} {'yes' if validators.can_lower(language) else 'no'}"
        for language in sorted(validators.SPIRV_COMPILERS)
    )
    print(
        f"\nLowering to SPIR-V for the CrossGL IR row (also needs spirv-dis): {lowering}"
    )
    print("CI uses the versions pinned in .github/workflows/coverage.yml.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "merge":
            return _merge(args)
        if args.command == "publish":
            return _publish(args)
        if args.command == "issues":
            return _issues(args)
        if args.command == "tools":
            return _tools()
        if args.command == "compare":
            published, current = (
                report.load(path) for path in (args.published, args.report)
            )
            print(compare.compare(published, current), end="")
            return 0
        for feature in features.load(args.backend, offline=args.offline):
            print(f"{feature.group:<44} {feature.name}")
        return 0
    except UpstreamError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
