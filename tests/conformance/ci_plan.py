"""Plan the legs of the conformance and corpus workflows.

Both workflows run on every pull request, so that their single ``result`` check can be required, and this
planning step decides how much of them to run. Neither runs in the merge queue, where its result is not a required
check, so a merge-group run could not hold a pull request back:

* A pull request whose changes touch none of the workflow's paths runs no legs, and its result check passes at
  once. When the changed paths cannot be determined, every path counts as changed.
* Pull requests run only the Linux legs; macOS and Windows runners are scarce and shared, so their legs run only
  while the pull request is labelled ``ci:full``.
* A corpus pull request leg also skips the slices whose source language is in ``LEAN_SKIPPED_LANGUAGES``: the
  Metal-source slices take about 97 % of the corpus CPU time (455 of 468 CPU minutes in the nightly run
  44900545251), which held a Linux runner for over two hours on every pull request push. The nightly schedule,
  manual dispatch and ``ci:full`` run every slice. The planned slices travel in each leg's ``slices`` key, which is
  empty for every slice.
* Pushes to main (conformance only), the nightly schedule and manual dispatch run every leg.

The legs themselves are JSON lists in the workflow (``LINUX_LEGS`` and ``HOSTED_LEGS``), read from the
environment. Usage::

    python3 -m tests.conformance.ci_plan --workflow corpus --event pull_request \\
        --base <sha> --head <sha> --full false

It writes ``run`` and ``legs`` to ``$GITHUB_OUTPUT`` and a one-line explanation to standard output. Only the
standard library is used, so the step needs no installation.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Shared by both workflows: the translator, the harness modules they import, their own workflow and actions, and
# the packaging that decides what is installed.
_SHARED = (
    "crosstl/**",
    ".github/actions/install-linux-dxc/**",
    ".github/actions/install-shader-validators/**",
    "pyproject.toml",
)

PATHS = {
    "conformance": (
        _SHARED
        + (
            "tests/conformance/**",
            ".github/workflows/conformance.yml",
        )
    ),
    "corpus": (
        _SHARED
        + (
            "tests/conformance/*.py",
            "tests/conformance/corpus/**",
            ".github/workflows/corpus.yml",
            "tools/ensure_metal_toolchain.py",
        )
    ),
}

# Events whose changes are filtered by path; every other event runs the workflow's legs unconditionally.
FILTERED_EVENTS = ("pull_request",)

# Events that run only the Linux legs unless the pull request carries the ci:full label.
LEAN_EVENTS = {
    "conformance": ("pull_request",),
    "corpus": ("pull_request",),
}

# Corpus source languages whose slices a lean run skips.
LEAN_SKIPPED_LANGUAGES = {
    "conformance": (),
    "corpus": ("metal",),
}

CORPUS_MANIFEST = os.path.join(os.path.dirname(__file__), "corpus", "manifest.json")


def lean_slices(workflow, manifest=CORPUS_MANIFEST):
    """The comma-separated slices a lean run of ``workflow`` measures; None when it measures every slice."""
    skipped = LEAN_SKIPPED_LANGUAGES[workflow]
    if not skipped:
        return None
    with open(manifest, encoding="utf-8") as handle:
        slices = json.load(handle)["slices"]
    return ",".join(
        name for name, entry in slices.items() if entry["language"] not in skipped
    )


def _pattern(glob):
    """Compile a GitHub Actions ``paths`` glob: ``**`` crosses directories, ``*`` and ``?`` do not."""
    expression = ""
    index = 0
    while index < len(glob):
        if glob.startswith("**", index):
            expression += ".*"
            index += 2
        elif glob[index] == "*":
            expression += "[^/]*"
            index += 1
        elif glob[index] == "?":
            expression += "[^/]"
            index += 1
        else:
            expression += re.escape(glob[index])
            index += 1
    return re.compile(expression + r"\Z")


def relevant(workflow, paths):
    """Whether any of ``paths`` is one the workflow covers."""
    patterns = [_pattern(glob) for glob in PATHS[workflow]]
    return any(pattern.match(path) for path in paths for pattern in patterns)


def changed_paths(base, head, git="git"):
    """Paths changed between the merge base of ``base`` and ``head``, and ``head``; None when git cannot tell."""
    if not base or not head or set(base) == {"0"}:
        return None
    try:
        completed = subprocess.run(
            [git, "diff", "--name-only", "--no-renames", f"{base}...{head}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line for line in completed.stdout.splitlines() if line]


def plan(workflow, event, full, linux_legs, hosted_legs, paths=None):
    """Return ``(legs, reason)`` for one workflow run.

    ``paths`` is the list of changed paths, or None when it is unknown; it matters only for filtered events.
    """
    if event in FILTERED_EVENTS:
        if paths is None:
            reason = "changed paths unknown, so every path counts as changed"
        elif not relevant(workflow, paths):
            return [], f"no change under the {workflow} paths ({len(paths)} changed)"
        else:
            reason = f"{workflow} paths changed"
    else:
        reason = f"{event} runs every leg"
    if event in LEAN_EVENTS[workflow] and not full:
        slices = lean_slices(workflow)
        if slices is None:
            return list(linux_legs), f"{reason}; Linux legs only"
        skipped = ", ".join(LEAN_SKIPPED_LANGUAGES[workflow])
        return (
            [{**leg, "slices": slices} for leg in linux_legs],
            f"{reason}; Linux legs only, without the {skipped}-source slices",
        )
    return list(linux_legs) + list(hosted_legs), reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow", choices=sorted(PATHS), required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    parser.add_argument(
        "--full", choices=("true", "false"), default="false", help="ci:full label"
    )
    arguments = parser.parse_args(argv)

    linux_legs = json.loads(os.environ["LINUX_LEGS"])
    hosted_legs = json.loads(os.environ["HOSTED_LEGS"])
    paths = None
    if arguments.event in FILTERED_EVENTS:
        paths = changed_paths(arguments.base, arguments.head)
    legs, reason = plan(
        arguments.workflow,
        arguments.event,
        arguments.full == "true",
        linux_legs,
        hosted_legs,
        paths,
    )
    output = os.environ.get("GITHUB_OUTPUT")
    lines = [
        f"run={'true' if legs else 'false'}",
        f"legs={json.dumps(legs, separators=(',', ':'))}",
    ]
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))
    names = ", ".join(leg["name"] for leg in legs) or "none"
    print(f"{arguments.workflow}: {reason}. Legs: {names}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
