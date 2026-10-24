"""Accept lane-divergent returns that no later collective can observe.

Software subgroups lower collectives to workgroup barriers, so an invocation
that returns early must not miss one. A return after the entry point's last
collective is safe even when it diverges; a return before a later collective,
including one in a later loop iteration or in a helper's caller, is not.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_TRAILING_RETURNS"
WORKGROUP = (64, 1, 1)
GROUPS = 2
LIMIT = 7
GUARD = [0x6D2B79F5] * 4
# Distinct per-invocation values; each subgroup sum differs.
VALUES = [(index * 2654435761 + 12345) % 1000003 for index in range(GROUPS * 64)]

ACCEPTED = {
    # The divergent guard follows the only collective.
    "after-collective": (
        """
    uint total = simd_sum(values[index]);
    if (lane >= LIMIT) { return; }
    results[4u + index] = total + lane;
    """
    ),
    # A nested block, a uniform loop and a helper call follow the guard,
    # none of which reaches a collective.
    # A helper's return after its own collective resumes the caller.
    "helper-tail": (
        """
    results[4u + index] = guarded(values[index], lane);
    """
    ),
    "nested-tail": (
        """
    uint total = simd_sum(values[index]);
    {
        uint scaled = total;
        if (lane >= LIMIT) { return; }
        for (uint round = 0u; round < 2u; round++) { scaled += lane; }
        results[4u + index] = offset(scaled, lane) - lane;
    }
    """
    ),
}
REJECTED = {
    "collective-after": (
        """
    uint total = simd_sum(values[index]);
    if (lane >= LIMIT) { return; }
    results[4u + index] = simd_sum(total + lane);
    """
    ),
    "later-iteration": (
        """
    uint total = 0u;
    for (uint round = 0u; round < 2u; round++) {
        total += simd_sum(values[index] + round);
        if (lane >= LIMIT) { return; }
    }
    results[4u + index] = total;
    """
    ),
}


def _source(body):
    return f"""#include <metal_stdlib>
using namespace metal;
constant uint LIMIT = {LIMIT}u;
uint offset(uint value, uint lane) {{ return value + lane; }}
uint guarded(uint value, uint lane) {{
    uint total = simd_sum(value);
    if (lane >= LIMIT) {{ return total; }}
    return total + lane;
}}
kernel void trailing(const device uint* values [[buffer(0)]],
                     device uint* results [[buffer(1)]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint index [[thread_position_in_grid]]) {{
{body}
}}
"""


def _expected(case):
    words = []
    for index in range(len(VALUES)):
        lane = index % 32
        group = index - lane
        total = sum(VALUES[group : group + 32]) & 0xFFFFFFFF
        if case == "helper-tail":
            words.append((total + (lane if lane < LIMIT else 0)) & 0xFFFFFFFF)
        elif lane >= LIMIT:
            words.append(0xDEADBEEF)
        elif case == "after-collective":
            words.append((total + lane) & 0xFFFFFFFF)
        else:
            words.append((total + 2 * lane) & 0xFFFFFFFF)
    return GUARD + words + GUARD


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "uint", WORKGROUP, source=_source(ACCEPTED[case])
    )
    initial = GUARD + [0xDEADBEEF] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {
                "dtype": "uint32",
                "shape": [len(VALUES)],
                "values": VALUES,
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(initial)],
                "values": initial,
            },
        },
    )
    expected = _expected(case)
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": expected,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", sorted(ACCEPTED))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_trailing_divergent_return_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    assert "return" in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("case", sorted(REJECTED))
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_divergent_return_before_a_collective_is_rejected(tmp_path, target, case):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "products.metal").write_text(_source(REJECTED[case]))
    options = {"software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("products.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=WORKGROUP,
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {f"project.translate.{target}-software-subgroup-invalid"}, codes


@pytest.mark.parametrize("case", sorted(ACCEPTED))
def test_trailing_divergent_return_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native trailing returns")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="trailing",
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_trailing_returns_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_trailing_divergent_returns.py" in step
    assert "if:" not in step and "continue-on-error" not in step
