"""Execution-only subgroup barriers require checked workgroup participation."""

import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_EXECUTION_RUNTIME"
BARRIER = "subgroupExecutionBarrier();"


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("form", ["standalone", "helper", "loop"])
def test_execution_barrier_compiles_without_a_reduction(tmp_path, target, form):
    helpers = ""
    body = BARRIER
    if form == "helper":
        helpers = f"void inner() {{{BARRIER}}} void outer() {{inner();}}"
        body = "outer();"
    elif form == "loop":
        body = f"for (int i=0; i<3; ++i) {{{BARRIER}}}"
    source = _canonical(body + "results[invocation] = invocation;", helpers)
    generated = _codegen(target).generate_stage(parse(source), "compute")
    assert "subgroupExecutionBarrier" not in generated
    assert (
        "GroupMemoryBarrierWithGroupSync();" in generated
        if target == "directx"
        else "barrier();" in generated
    )
    assert "groupshared" not in generated and "shared uint" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body,helpers",
    [
        (f"if(invocation<32u) {{{BARRIER}}}", ""),
        (f"if(invocation==0u) {{return;}} {BARRIER}", ""),
        (f"for(uint i=0u;i<invocation;++i) {{{BARRIER}}}", ""),
        (f"while(invocation>0u) {{{BARRIER}}}", ""),
        (
            "inner(invocation);",
            f"void inner(uint lane) {{if(lane==0u) {{return;}} {BARRIER}}}",
        ),
        ("if(invocation<32u) {inner();}", f"void inner() {{{BARRIER}}}"),
        (
            "outer(invocation);",
            f"void inner(uint lane) {{if(lane<32u) {{{BARRIER}}}}} void outer(uint lane) {{inner(lane);}}",
        ),
        (
            "results[invocation] = invocation==0u ? inner() : 0u;",
            f"uint inner() {{{BARRIER} return 1u;}}",
        ),
        (
            "bool selected = invocation==0u && inner();",
            f"bool inner() {{{BARRIER} return true;}}",
        ),
        (f"for(int i=0;i<3;++i) {{if(invocation==0u) {{break;}} {BARRIER}}}", ""),
        (f"for(int i=0;i<3;++i) {{if(invocation==0u) {{continue;}} {BARRIER}}}", ""),
        ("inner();", f"void inner() {{{BARRIER} inner();}}"),
    ],
)
def test_execution_barrier_rejects_unproven_participation(target, body, helpers):
    with pytest.raises(ValueError) as error:
        _codegen(target).generate_stage(parse(_canonical(body, helpers)), "compute")
    assert getattr(error.value, "reason", None) in {
        "potentially-divergent-control-flow",
        "early-return-unproven",
        "helper-call-not-uniform",
        "helper-call-recursive",
    }


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("software", [False, True])
def test_execution_barrier_rejects_arguments(target, software):
    with pytest.raises(ValueError) as error:
        _codegen(target, software).generate_stage(
            parse(_canonical("subgroupExecutionBarrier(1u);")), "compute"
        )
    assert getattr(error.value, "reason", None) == "invalid-argument-count"


def test_directx_native_execution_barrier_requires_an_explicit_contract():
    with pytest.raises(ValueError) as error:
        _codegen("directx", False).generate_stage(parse(_canonical(BARRIER)), "compute")
    assert getattr(error.value, "reason", None) == "execution-barrier-contract-unproven"


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_execution_barrier_rejects_non_compute_software_stages(target):
    source = f"shader Test {{ fragment {{ void main() {{{BARRIER}}} }} }}"
    with pytest.raises(ValueError) as error:
        _codegen(target).generate_stage(parse(source), "fragment")
    assert getattr(error.value, "reason", None) == "entry-point-contract-invalid"


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("software", [False, True])
def test_execution_barrier_name_keeps_source_function_ownership(
    tmp_path, target, software
):
    helpers = "uint subgroupExecutionBarrier(uint value) {return value + 7u;}"
    argument = "WaveActiveSum(invocation)" if software else "invocation"
    source = _canonical(
        f"results[invocation] = subgroupExecutionBarrier({argument});", helpers
    )
    generated = _codegen(target, software).generate_stage(parse(source), "compute")
    assert "uint subgroupExecutionBarrier(uint value)" in generated
    assert "subgroupExecutionBarrier(" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_execution_barrier_contract_is_reset_between_generations(target):
    generator = _codegen(target)
    source = _canonical(BARRIER)
    first = generator.generate_stage(parse(source), "compute")
    with pytest.raises(ValueError):
        generator.generate_stage(
            parse(_canonical(f"if(invocation==0u) {{{BARRIER}}}")), "compute"
        )
    assert generator.generate_stage(parse(source), "compute") == first
    other = _canonical("results[invocation] = WaveActiveSum(invocation);")
    assert generator.generate_stage(parse(other), "compute") == _codegen(
        target
    ).generate_stage(parse(other), "compute")


@pytest.mark.parametrize(
    "target,name",
    [("directx", "GroupMemoryBarrierWithGroupSync"), ("opengl", "barrier")],
)
def test_execution_barrier_target_intrinsic_is_not_captured(tmp_path, target, name):
    source = _canonical(
        f"results[invocation] = {name}(); {BARRIER}", f"uint {name}() {{return 7u;}}"
    )
    generated = _codegen(target).generate_stage(parse(source), "compute")
    assert f"uint {name}(" not in generated
    assert len(re.findall(rf"(?<![A-Za-z0-9_]){name}\(\);", generated)) == 1
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(
    "target,name",
    [("directx", "GroupMemoryBarrierWithGroupSync"), ("opengl", "barrier")],
)
@pytest.mark.parametrize("scope", ["global", "local"])
@pytest.mark.parametrize("qualifier", ["const ", ""])
def test_execution_barrier_target_intrinsic_is_not_shadowed_by_values(
    tmp_path, target, name, scope, qualifier
):
    # Nonconstant OpenGL globals require a reflected uniform location.
    location = (
        " @location(1)"
        if target == "opengl" and scope == "global" and not qualifier
        else ""
    )
    declaration = f"{qualifier}uint {name}{location} = 7u;"
    body = f"results[invocation] = {name}; {BARRIER}"
    source = _canonical(
        body if scope == "global" else declaration + body,
        declaration if scope == "global" else "",
    )
    if target == "directx" and scope == "global":
        with pytest.raises(ValueError) as error:
            _codegen(target).generate_stage(parse(source), "compute")
        assert getattr(error.value, "reason", None) == "target-intrinsic-shadowed"
        return
    generated = _codegen(target).generate_stage(parse(source), "compute")
    assert f"uint {name} " not in generated
    assert len(re.findall(rf"(?<![A-Za-z0-9_]){name}\(\);", generated)) == 1
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("stage", ["compute", "fragment"])
def test_atomic_fence_helpers_keep_the_unique_emitted_stage(tmp_path, stage):
    helpers = "void inner() {atomicThreadFence(mem_threadgroup, memory_order_acq_rel, thread_scope_threadgroup);} void outer() {inner();}"
    source = f"shader Test {{{helpers} {stage} {{void main() {{outer();}}}}}}"
    generator = _codegen("opengl", False)
    if stage != "compute":
        with pytest.raises(ValueError) as error:
            generator.generate(parse(source))
        assert getattr(error.value, "reason", None) == "unsupported-shader-stage"
    else:
        generated = generator.generate(parse(source))
        assert "memoryBarrierShared();" in generated
        _compile(generated, "opengl", tmp_path)


def test_atomic_fence_helper_stage_is_not_guessed_for_mixed_modules():
    source = """shader Test {
        void inner() {atomicThreadFence(mem_threadgroup, memory_order_acq_rel, thread_scope_threadgroup);}
        compute {void main() {inner();}}
        fragment {void main() {inner();}}
    }"""
    with pytest.raises(ValueError) as error:
        _codegen("opengl", False).generate(parse(source))
    assert getattr(error.value, "reason", None) == "unsupported-shader-stage"


@pytest.mark.parametrize(
    "expression", ["subgroupBarrier()", "subgroupExecutionBarrier", "gl_SubgroupID"]
)
def test_execution_barrier_does_not_enable_raw_hardware_builtins(expression):
    source = _canonical(
        f"{expression}; results[invocation] = WaveActiveSum(invocation);"
    )
    with pytest.raises(ValueError) as error:
        _codegen("opengl").generate_stage(parse(source), "compute")
    assert getattr(error.value, "reason", None) == "subgroup-builtin-unsupported"


def _source(nested, iterations):
    fence = "atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_acq_rel, thread_scope_threadgroup);"
    synchronize = fence + "\n    simdgroup_barrier(mem_flags::mem_none);"
    helpers = (
        f"void inner() {{{synchronize}}} void outer() {{inner();}}" if nested else ""
    )
    call = "outer();" if nested else synchronize
    return f"""#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void exchange(device uint* results [[buffer(0)]],
                     uint lane [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {{
    if (group.x >= 2u) {{ return; }}
    threadgroup uint scratch[128];
    uint value = group.x * 1000u + lane;
    for (int iteration=0; iteration<{iterations}; ++iteration) {{
        scratch[lane] = value;
        {call}
        value = scratch[(lane / 32u) * 32u + (lane + 1u) % 32u] + uint(iteration + 1);
        {call}
    }}
    results[5u + group.x * 128u + lane] = value;
}}
"""


def _request(root, target, nested, iterations):
    source, descriptor, package = _package(
        root, target, "uint", (32, 2, 2), source=_source(nested, iterations)
    )
    guard = 0xDEADBEEF
    expected = [guard] * 396
    for group in range(2):
        for lane in range(128):
            expected[5 + group * 128 + lane] = (
                group * 1000
                + lane // 32 * 32
                + (lane + iterations) % 32
                + iterations * (iterations + 1) // 2
            )
    inputs = {"results": {"dtype": "uint32", "shape": [396], "values": [guard] * 396}}
    outputs = _bound_values(
        descriptor, {"results": {"dtype": "uint32", "shape": [396], "values": expected}}
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupSize": [32, 2, 2], "workgroupCount": [3, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("nested", [False, True])
def test_fenced_execution_barriers_package_and_compile(tmp_path, target, nested):
    _, request, _ = _request(tmp_path, target, nested, 3)
    generated = request.artifact_path.read_text()
    if target == "metal":
        assert "simdgroup_barrier(mem_flags::mem_none)" in generated
        assert "atomic_thread_fence(" in generated
    else:
        assert "subgroupExecutionBarrier" not in generated
        assert (
            "GroupMemoryBarrier();" in generated
            if target == "directx"
            else "memoryBarrierShared();" in generated
        )
    # Acquire/release fence orders are available starting with Metal 4.1.
    _compile(generated, target, tmp_path, metal_compile_flags=("-std=metal4.1",))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("iterations", [1, 3])
def test_fenced_execution_barriers_execute_with_scratch_reuse(
    tmp_path, nested, iterations
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required execution barrier controls")
    target = {"linux": "opengl", "darwin": "metal", "win32": "directx"}[sys.platform]
    source, request, outputs = _request(tmp_path, target, nested, iterations)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="exchange",
        metal_language_version="4.1",
    )


def test_execution_barrier_runtime_is_required_in_ci():
    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = workflow.split("- name: Validate collective helper arguments", 1)[1].split(
        "- name:", 1
    )[0]
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_subgroup_execution_barriers.py" in step
