"""Read GitHub Actions workflows as text, for tests that check how a job or step is written.

YAML parsing loses comments and the order of steps across jobs, so these helpers slice the text instead. Sections
are found by indentation: a job is a two-space key under ``jobs:``, a step starts at ``- name:``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
RUNNER_OSES = {"ubuntu-latest", "windows-latest", "macOS-latest"}

_STEP = re.compile(r"^\s*-\s+name:\s*", re.MULTILINE)


def workflow_texts() -> dict[str, str]:
    """Every workflow in ``.github/workflows``, keyed by file name."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def nested_yaml_section(text: str, name: str, indent: int) -> list[str]:
    """The lines under the first ``name:`` key at ``indent`` spaces, without the key itself."""
    lines = text.splitlines()
    header = " " * indent + name + ":"
    for index, line in enumerate(lines):
        if line.rstrip() != header:
            continue
        section = []
        for body in lines[index + 1 :]:
            if body.strip() and _indent(body) <= indent:
                break
            section.append(body)
        return section
    return []


def _list_values(section: list[str], key: str) -> list[str]:
    for index, line in enumerate(section):
        if line.strip() != f"{key}:":
            continue
        values = []
        for item in section[index + 1 :]:
            if item.strip() and _indent(item) <= _indent(line):
                break
            match = re.match(r"^\s*-\s+(.+?)\s*$", item)
            if match:
                values.append(match.group(1).strip("\"'"))
        return values
    return []


def workflow_event_path_filters(workflow: str, event: str) -> list[str]:
    """An event's ``paths`` filter: ``["**"]`` when the event has none, ``[]`` when it is not a trigger."""
    section = nested_yaml_section(workflow, event, 2)
    paths = _list_values(section, "paths")
    if paths:
        return paths
    triggers = nested_yaml_section(workflow, "on", 0)
    declared = any(line.split("#", 1)[0].rstrip() == f"  {event}:" for line in triggers)
    if declared and not any(
        line.strip().startswith("paths-ignore:") for line in section
    ):
        return ["**"]
    return []


def _step_match(text: str, name: str) -> re.Match[str] | None:
    pattern = rf"^\s*-\s+name:\s*{re.escape(name)}\s*$"
    return re.search(pattern, text, flags=re.MULTILINE)


def workflow_step_section(text: str, name: str) -> str:
    """The step named ``name``, up to the next step, or ``""``."""
    match = _step_match(text, name)
    if not match:
        return ""
    following = _STEP.search(text, match.end())
    return text[match.start() : following.start() if following else len(text)]


def workflow_step_after(text: str, later: str, earlier: str) -> bool:
    """Whether both steps exist and ``later`` comes after ``earlier``."""
    first, second = _step_match(text, earlier), _step_match(text, later)
    return bool(first and second and second.start() > first.start())


def workflow_job_names(workflow: str) -> list[str]:
    return [
        match.group(1)
        for line in nested_yaml_section(workflow, "jobs", 0)
        if (match := re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line))
    ]


def workflow_job_text(workflow: str, job: str) -> str:
    """The body of one job, without its key, or ``""``."""
    return "\n".join(nested_yaml_section(workflow, job, 2))


def workflow_job_step_section(workflow: str, job: str, step: str) -> str:
    return workflow_step_section(workflow_job_text(workflow, job), step)


def workflow_job_step_after(workflow: str, job: str, later: str, earlier: str) -> bool:
    return workflow_step_after(workflow_job_text(workflow, job), later, earlier)


def workflow_job_timeout_minutes(workflow: str, job: str) -> int | None:
    match = re.search(
        r"^    timeout-minutes:\s*([0-9]+)\s*$",
        workflow_job_text(workflow, job),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def _values(raw: str) -> set[str]:
    items = (item.strip().strip("\"'") for item in raw.strip().strip("[]").split(","))
    return {item for item in items if item}


def matrix_values(text: str, key: str) -> set[str]:
    """The values of a matrix key written as a flow list, on one line or several.

    A key filled at run time from ``fromJSON(needs.<job>.outputs.<name>)`` must declare its full list in an
    environment variable ``FULL_<KEY>S``, which is returned instead.
    """
    inline = re.search(
        rf"^\s*{re.escape(key)}:\s*(\[[^\n]+\])\s*$", text, flags=re.MULTILINE
    )
    if inline:
        return _values(inline.group(1))
    block = re.search(
        rf"^\s*{re.escape(key)}:\s*\n\s*\[\s*(.*?)\s*\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if block:
        return _values(block.group(1))
    dynamic = re.search(
        rf"^\s*{re.escape(key)}:\s*"
        r"\$\{\{\s*fromJSON\(needs\.[\w-]+\.outputs\.[\w-]+\)\s*\}\}\s*$",
        text,
        flags=re.MULTILINE,
    )
    if dynamic:
        name = "FULL_" + key.upper().replace("-", "_") + "S"
        declared = re.search(
            rf"^\s*{name}:\s*'(\[[^\n]+\])'\s*$", text, flags=re.MULTILINE
        )
        assert declared, f"run-time matrix key {key} must declare {name}"
        return _values(declared.group(1))
    raise AssertionError(f"matrix key not found: {key}")
