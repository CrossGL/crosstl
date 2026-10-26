"""Assertions for workflow coverage and required execution prerequisites."""

import re
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

from tests import workflow_text

ROOT = Path(__file__).resolve().parents[1]

# The longest current-tree MLX kernel families run in their own jobs, one matrix
# leg per runner and family group. GitHub expands at most 256 legs from one
# matrix, so the groups are split over several jobs. The first job defines the
# steps and the others alias them.
CURRENT_PARITY_JOB = "mlx-current-native-parity"
CURRENT_PARITY_JOBS = (
    CURRENT_PARITY_JOB,
    "mlx-current-quantized-parity",
    "mlx-current-gemm-parity",
)
CURRENT_PARITY_UPLOAD = "Upload current MLX native parity evidence"
MAX_MATRIX_LEGS = 256


def current_parity_matrix(workflow):
    """Return the groups and include entries of every current-tree parity job."""
    jobs = yaml.safe_load(workflow)["jobs"]
    groups, include = [], []
    for name in CURRENT_PARITY_JOBS:
        matrix = jobs[name]["strategy"]["matrix"]
        groups += matrix["group"]
        include += matrix.get("include", [])
    return {"group": groups, "include": include}


def bounded_step_deadline(step, runner_os):
    """Count literal command deadlines, excluding explicit platform mismatches."""
    condition = step.get("if", "")
    # Count disjunctions conservatively rather than interpreting arbitrary expressions.
    if "||" not in condition:
        for operator, platform in re.findall(
            r"runner\.os (==|!=) '([^']+)'", condition
        ):
            if (runner_os == platform) != (operator == "=="):
                return 0
    return sum(
        int(value)
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    )


def assert_native_step_prerequisites(
    step, *, job, pinned=False, group=None, family=None
):
    """Require exact native prerequisites without accepting optional execution."""
    assert job in (
        "portable-host",
        "metal-host",
        "mlx-metal-porting",
        CURRENT_PARITY_JOB,
    )
    # A parity step runs in exactly one family group of its matrix, or in every
    # shard group of one family.
    assert ((group is not None) + (family is not None)) == (job == CURRENT_PARITY_JOB)
    definition = yaml.safe_load(step)
    if isinstance(definition, list):
        assert len(definition) == 1
        definition = definition[0]
    assert isinstance(definition, dict)
    assert "continue-on-error" not in definition
    if job == CURRENT_PARITY_JOB:
        expected = (
            f"matrix.group == '{group}'"
            if family is None
            else f"matrix.family == '{family}'"
        )
        assert definition.get("if") == expected
        return
    if job != "portable-host":
        assert "if" not in definition
        return
    expected = (
        "${{ !cancelled() && steps.pinned-upstream.outcome == 'success' }}"
        if pinned
        else "${{ !cancelled() && (steps.portable-host-contracts.outcome == 'success' || "
        "(runner.os == 'macOS' && steps.platform-host-abi.outcome == 'success') || "
        "steps.host-warp-identity.outcome == 'success') }}"
    )
    assert " ".join(definition.get("if", "").split()) == expected


def _matches_path(parts, patterns):
    if not patterns:
        return not parts
    if patterns[0] == "**":
        return _matches_path(parts, patterns[1:]) or bool(
            parts and _matches_path(parts[1:], patterns)
        )
    return bool(
        parts
        and fnmatchcase(parts[0], patterns[0])
        and _matches_path(parts[1:], patterns[1:])
    )


def assert_paths_covered(filters, *required, root=ROOT):
    assert filters and not any(pattern.startswith("!") for pattern in filters)
    for path in required:
        # Before Python 3.13, a trailing ** yields directories only.
        pattern = path + "/*" if path.endswith("/**") else path
        candidates = (
            [
                item.relative_to(root).as_posix()
                for item in root.glob(pattern)
                if item.is_file()
            ]
            if any(character in path for character in "*?[")
            else [path]
        )
        assert candidates, f"No files match required workflow path: {path}"
        for candidate in candidates:
            assert any(
                _matches_path(candidate.split("/"), pattern.split("/"))
                for pattern in filters
            ), candidate


def assert_workflow_triggers(workflow, *required):
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event), *required
        )


def current_parity_step(workflow, name, *, group):
    """Return a current-tree parity step after checking its group and upload."""
    job = workflow_text.workflow_job_text(workflow, CURRENT_PARITY_JOB)
    step = workflow_text.workflow_step_section(job, name)
    assert step, name
    assert_native_step_prerequisites(step, job=CURRENT_PARITY_JOB, group=group)
    assert group in current_parity_matrix(workflow)["group"]
    return _checked_parity_step(workflow, job, name, step)


def current_parity_shard_step(workflow, name, *, family, count):
    """Return the step that runs every shard group of a family, after checking the matrix.

    The groups ``<family>-0`` to ``<family>-<count - 1>`` carry their family and shard
    index through the matrix ``include`` list. The one step selects on the family and
    reads the index from ``matrix.shard``.
    """
    job = workflow_text.workflow_job_text(workflow, CURRENT_PARITY_JOB)
    step = workflow_text.workflow_step_section(job, name)
    assert step, name
    assert_native_step_prerequisites(step, job=CURRENT_PARITY_JOB, family=family)
    matrix = current_parity_matrix(workflow)
    shards = {
        entry["group"]: entry["shard"]
        for entry in matrix.get("include", [])
        if entry.get("family") == family
    }
    assert shards == {f"{family}-{index}": index for index in range(count)}, family
    assert set(shards) <= set(matrix["group"]), family
    assert "_SHARD_INDEX: ${{ matrix.shard }}" in step, name
    return _checked_parity_step(workflow, job, name, step)


def _checked_parity_step(workflow, job, name, step):
    # A family that also ran in the porting job would spend its budget twice.
    assert not workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", name
    )
    upload = workflow_text.workflow_step_section(job, CURRENT_PARITY_UPLOAD)
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert job.index(name) < job.index(CURRENT_PARITY_UPLOAD)
    return step
