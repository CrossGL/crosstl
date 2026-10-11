"""Workgroup pointer bounds follow exact subgroup geometry and lazy guards."""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLWorkgroupPointerError,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_codegen.test_GLSL_codegen import (
    assert_glsl_compute_validates_if_available,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import (
    _check_outputs,
    _package,
)

REQUIRE_ENV = "CROSTL_REQUIRE_WORKGROUP_RANGE_READBACKS"


def _canonical(body, *, extent=128, view="values", extra="", declarations=""):
    return f"""shader Bounds {{
        RWStructuredBuffer<float> output @register(u0);
        {declarations}
        float read_shared(threadgroup float* sharedValues,
                          uint3 lid, uint group, uint groups) {{
            {body}
        }}
        compute {{
            @numthreads(128, 1, 1)
            void main(uint3 lid @gl_LocalInvocationID,
                      uint group @gl_SubgroupID,
                      uint groups @gl_NumSubgroups) {{
                threadgroup float values[{extent}];
                if (lid.x < {extent}u) {{ values[lid.x] = float(lid.x); }}
                memoryBarrierShared();
                barrier();
                float result = read_shared({view}, lid, group, groups);
                {extra}
                output[lid.x] = WaveActiveSum(result);
            }}
        }}
    }}"""


@pytest.mark.parametrize("component", ["lid.x", "lid.r", "lid[0]"])
@pytest.mark.parametrize("form", ["true", "false", "if", "endpoint"])
def test_guarded_component_reads_compile(tmp_path, component, form):
    read = f"sharedValues[{component} * 4u + 3u]"
    body = {
        "true": f"return {component} < groups ? {read} : 0.0;",
        "false": f"return {component} >= groups ? 0.0 : {read};",
        "if": f"if ({component} < groups) {{ return {read}; }} return 0.0;",
        "endpoint": f"return {component} != 127u ? sharedValues[{component}] : 0.0;",
    }[form]
    extent = 127 if form == "endpoint" else 16
    generated = GLSLCodeGen(software_subgroup_width=32).generate_stage(
        parse(_canonical(body, extent=extent)), "compute"
    )
    assert_glsl_compute_validates_if_available(generated, tmp_path, "component_guard")


@pytest.mark.parametrize("offset", [0, 5])
def test_subgroup_index_preserves_view_offset_and_all_call_contexts(tmp_path, offset):
    generated = GLSLCodeGen(software_subgroup_width=32).generate_stage(
        parse(
            _canonical(
                "return sharedValues[group * 4u + 3u];",
                extent=16 + offset,
                view=f"values + {offset}u",
                extra="result += read_shared(values, lid, group, groups);",
            )
        ),
        "compute",
    )
    assert_glsl_compute_validates_if_available(generated, tmp_path, "subgroup_offset")


@pytest.mark.parametrize(
    "body,kwargs",
    [
        ("return sharedValues[group * 4u + 3u];", {"extent": 15}),
        ("return sharedValues[group * 4u + 3u];", {"view": "values + 125u"}),
        ("return sharedValues[lid.x * 4u + 3u];", {}),
        (
            "return lid.x <= groups ? sharedValues[lid.x * 4u + 3u] : 0.0;",
            {"extent": 16},
        ),
        (
            "if (lid.x < groups) { lid.x = 127u; return sharedValues[lid.x * 4u + 3u]; } return 0.0;",
            {},
        ),
        (
            "groups = 128u; return lid.x < groups ? sharedValues[lid.x * 4u + 3u] : 0.0;",
            {},
        ),
        (
            "return sharedValues[group * 4u + 3u];",
            {"extra": "result += read_shared(values, lid, 100u, groups);"},
        ),
        (
            "return lid.x < groups ? sharedValues[lid.x * 4u + 3u] : sharedValues[128];",
            {},
        ),
        (
            "return lid.x++ < groups ? sharedValues[lid.x * 4u + 3u] : 0.0;",
            {"extent": 16},
        ),
        (
            "if (lid.x < groups) { change(lid); return sharedValues[lid.x * 4u + 3u]; } return 0.0;",
            {"declarations": "void change(inout uint3 value) { value.x = 127u; }"},
        ),
    ],
)
def test_unsafe_workgroup_views_remain_diagnostic(body, kwargs):
    with pytest.raises(OpenGLWorkgroupPointerError) as error:
        GLSLCodeGen(software_subgroup_width=32).generate_stage(
            parse(_canonical(body, **kwargs)), "compute"
        )
    assert error.value.reason in {"view-out-of-bounds", "unprovable-view-access"}


def test_generator_reuse_does_not_retain_safe_call_bounds(tmp_path):
    generator = GLSLCodeGen(software_subgroup_width=32)
    body = "return sharedValues[group * 4u + 3u];"
    generated = generator.generate_stage(parse(_canonical(body, extent=16)), "compute")
    assert_glsl_compute_validates_if_available(generated, tmp_path, "reuse_safe")
    with pytest.raises(OpenGLWorkgroupPointerError):
        generator.generate_stage(parse(_canonical(body, extent=15)), "compute")


def _native_source(size, offset):
    return f"""#include <metal_stdlib>
using namespace metal;
uint read_group(threadgroup uint* values, uint group) {{
    return values[group * 4u + 3u];
}}
uint read_guarded(threadgroup uint* values, uint3 lid, uint groups) {{
    return lid.x < groups ? values[lid.x * 4u + 3u] : 77u;
}}
uint read_wrapped(threadgroup uint* values, uint3 lid, uint groups) {{
    return read_guarded(values, lid, groups);
}}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint3 lid [[thread_position_in_threadgroup]],
                     uint3 workgroup [[threadgroup_position_in_grid]],
                     uint group [[simdgroup_index_in_threadgroup]],
                     uint groups [[simdgroups_per_threadgroup]]) {{
    threadgroup uint values[{size + offset + 1}];
    uint index = workgroup.x * {size}u + lid.x;
    values[lid.x + {offset}u] = inputWords[index];
    if (lid.x == 0u) {{ values[{size + offset}] = 29u; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first = read_group(values + {offset}u, group);
    uint second = read_wrapped(values + {offset}u, lid, groups);
    uint third = read_wrapped(values + {offset + 1}u, lid, groups);
    outputWords[index * 4u] = first;
    outputWords[index * 4u + 1u] = second;
    outputWords[index * 4u + 2u] = third;
    outputWords[index * 4u + 3u] = simd_sum(first + second + third);
}}
"""


@pytest.mark.parametrize("size", [32, 64, 128])
@pytest.mark.parametrize("offset", [0, 5])
def test_workgroup_helper_bounds_native_readbacks(tmp_path, size, offset):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required workgroup range readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path, target, "uint", (size, 1, 1), source=_native_source(size, offset)
    )
    words = [index * 7 + 19 for index in range(512)]
    wanted = []
    for start in range(0, len(words), size):
        group_words = words[start : start + size]
        rows = []
        for lane in range(size):
            first = group_words[(lane // 32) * 4 + 3]
            second = group_words[lane * 4 + 3] if lane < size // 32 else 77
            third = group_words[lane * 4 + 4] if lane < size // 32 else 77
            rows.append([first, second, third])
        for start_lane in range(0, size, 32):
            reduced = sum(sum(row) for row in rows[start_lane : start_lane + 32])
            for row in rows[start_lane : start_lane + 32]:
                wanted.extend([*row, reduced])
    guards = [0xBAD00000 + index for index in range(17)]

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xDEADBEEF] * len(wanted) + guards),
    }
    expected = _bound_values(
        descriptor,
        {"inputWords": payload(words), "outputWords": payload(wanted + guards)},
    )
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {key: value for key, value in data.items() if key != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [len(words) // size, 1, 1], "workgroupSize": [size, 1, 1]},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    executor = _executor(target)
    records = {}
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        records["generated"] = {"outputs": result.outputs, "details": result.details}
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            original_source, original_module = _compile(source, target, original)
            state, native = _native_request(request)
            native = replace(
                native,
                artifact_path=original_source,
                module_path=original_module,
                entry_point="products",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
            records["originalMetal"] = {
                "outputs": actual,
                "moduleSha256": (
                    hashlib.sha256(original_module.read_bytes()).hexdigest()
                ),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "size": size,
                    "offset": offset,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "sourceSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "validationModuleSha256": (
                        hashlib.sha256(module.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
        for record in records.values():
            _check_outputs(record["outputs"], expected, names, "uint")
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_workgroup_range_gate_is_required_on_every_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "test_software_subgroup_workgroup_ranges.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "-n auto" in step and "--timeout-seconds 360" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_software_subgroup_workgroup_ranges.py",
        )
