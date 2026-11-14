"""Loop bounds read through accessor helpers that forward resource identities.

A kernel that keeps a storage pointer in an aggregate reads it through
generated accessor helpers. Each helper also receives every resource
identity, but its value depends only on the reference and index it reads.
A software-subgroup loop whose bound is such a read stays uniform when the
body writes an unrelated resource. A bound whose own operands the body
writes is still rejected.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FORWARDED_LOOP_BOUNDS"
WORKGROUP = (64, 1, 1)
GROUPS = 2
STEPS = 3
SCALE = 5
GUARD = [0x6D2B79F5] * 4
VALUES = [(index * 2654435761 + 12345) % 1000003 for index in range(GROUPS * 64)]

ACCEPTED = """
  for (int step = 0; step < params->steps; step++) {
    accumulate(total, values[index] + uint(reader.scaled(step)));
    sink.store(index, total);
  }
"""
# The bound reads ``limit`` through the helper, and the body writes it.
REJECTED = """
  int limit = params->steps;
  for (int step = 0; step < reader.scaled(limit); step++) {
    accumulate(total, values[index]);
    limit = limit - 1;
  }
"""


def _source(loop):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Params {{ int steps; int scale; }};
struct Sink {{
  device uint* out;
  void store(uint i, uint v) {{ out[i] = v; }}
}};
struct Reader {{
  const constant Params* params;
  int scaled(int v) const {{ return v * params->scale; }}
}};
void accumulate(thread uint& total, uint v) {{ total += simd_sum(v); }}
kernel void bounded(const device uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    const constant Params* params [[buffer(2)]],
                    uint index [[thread_position_in_grid]]) {{
  uint total = 0u;
  Sink sink{{results}};
  Reader reader{{params}};
{loop}
  results[index] = total;
}}
"""


def _index_assertions():
    """MLX-independent bounds: the arrays hold fewer than 2^31 elements."""
    return tuple(
        {
            "source": "products.metal",
            "expression": "reference.offset + index",
            "function": function,
            "minimum": 0,
            "maximum": maximum,
        }
        for function, maximum in (
            ("crosstl_resource_load_uint", 2**31 - 1),
            ("crosstl_resource_store_uint", 2**31 - 1),
            ("crosstl_resource_load_Params", 0),
        )
    )


def _expected():
    words = []
    for index in range(len(VALUES)):
        group = index - index % 32
        lanes = sum(VALUES[group : group + 32])
        total = sum(lanes + 32 * step * SCALE for step in range(STEPS))
        words.append(total & 0xFFFFFFFF)
    return words + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        WORKGROUP,
        source=_source(ACCEPTED),
        index_range_assertions=_index_assertions() if target == "opengl" else (),
    )
    params = next(
        binding
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]) == "params"
    )
    layout = params["scalarLayout"]
    assert layout["elementType"] == "int32", layout
    assert [member["name"] for member in layout["structMembers"]] == [
        "steps",
        "scale",
    ]
    initial = [0xDEADBEEF] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "uint32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "uint32", "shape": [len(initial)], "values": initial},
            "params": {"dtype": "int32", "shape": [2], "values": [STEPS, SCALE]},
        },
    )
    expected = _expected()
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "uint32", "shape": [len(expected)], "values": expected}},
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


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_forwarded_resource_loop_bound_compiles(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    if target == "opengl":
        assert "crosstl_resource_load_Params" in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_loop_bound_operand_written_by_the_body_is_rejected(tmp_path, target):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "products.metal").write_text(_source(REJECTED))
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
            index_range_assertions=(_index_assertions() if target == "opengl" else ()),
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    codes = {item["code"] for item in report["diagnostics"]}
    assert codes == {f"project.translate.{target}-software-subgroup-invalid"}, codes


def test_forwarded_resource_loop_bound_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native loop bounds")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="bounded",
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_forwarded_loop_bounds_are_required_on_all_native_platforms():
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
    assert "test_forwarded_resource_loop_bounds.py" in step
    assert "if:" not in step and "continue-on-error" not in step
