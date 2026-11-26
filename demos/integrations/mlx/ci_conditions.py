"""Spell the condition under which a heavy MLX workflow job runs.

The ``ci-plan`` job writes the heavy jobs a change can affect to its ``jobs``
output as a JSON list, and each heavy job runs only when that list names it.

This module reads no files. Tests that run inside heavy jobs import the
condition from here rather than from ``ci_legs``, whose planner reads every
MLX file, so that importing it does not make those jobs depend on the whole
MLX tree.
"""

JOBS_OUTPUT = "needs.ci-plan.outputs.jobs"


def job_condition(job):
    """Return the ``if`` of a heavy job: the plan's ``jobs`` output names it."""
    return f"contains(fromJSON({JOBS_OUTPUT}), '{job}')"
