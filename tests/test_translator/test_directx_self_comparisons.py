"""NaN-sensitive private self-comparisons survive optimized HLSL compilation."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_SELF_COMPARISONS"


def _source(width, operator):
    kind = "float" + (str(width) if width > 1 else "")
    boolean = "bool" + (str(width) if width > 1 else "")
    values = ", ".join(f"asfloat(inputs[id.x * {width}u + {i}u])" for i in range(width))
    writes = "\n".join(
        f"outputs[id.x * {width}u + {i}u] = uint(result{'.' + 'xyzw'[i] if width > 1 else ''});"
        for i in range(width)
    )
    return f"""shader Classification {{
        StructuredBuffer<uint> inputs @register(t0);
        RWStructuredBuffer<uint> outputs @register(u0);
        {boolean} classify({kind} value) {{ return value {operator} value; }}
        compute {{
            @numthreads(1, 1, 1)
            void main(uint3 id @gl_GlobalInvocationID) {{
                {kind} value = {kind}({values});
                {boolean} result = classify(value);
                {writes}
            }}
        }}
    }}"""


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("operator", ["==", "!="])
def test_self_comparisons_retain_dynamic_nan_classification(tmp_path, width, operator):
    generated = HLSLCodeGen().generate_stage(parse(_source(width, operator)), "compute")
    comparison = ">" if operator == "!=" else "<="
    assert f"((asuint(value) & 0x7fffffffu) {comparison} 0x7f800000u)" in generated
    artifact, _module = _compile(generated, "directx", tmp_path)
    if not shutil.which("dxc"):
        pytest.skip("DXC is required to inspect optimized self-comparisons")
    assembly = tmp_path / "optimized.ll"
    result = subprocess.run(
        [
            "dxc",
            "-T",
            "cs_6_6",
            "-E",
            "CSMain",
            "-O3",
            "-WX",
            str(artifact),
            "-Fc",
            str(assembly),
            "-Fo",
            str(tmp_path / "optimized.dxil"),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ir = assembly.read_text()
    assert len(re.findall(r"icmp u(?:gt|lt) i32", ir)) == width
    assert "fcmp fast" not in ir


@pytest.mark.parametrize("kind", ["int", "uint", "bool", "double"])
def test_non_binary32_comparisons_remain_unchanged(kind):
    source = (
        f"shader Other {{ bool classify({kind} value) {{ return value != value; }} }}"
    )
    generated = HLSLCodeGen().generate(parse(source))
    assert "(value != value)" in generated and "0x7fffffffu" not in generated


@pytest.mark.parametrize("expression", ["values[0]", "next_value()", "value++"])
def test_self_comparison_does_not_collapse_repeated_evaluations(expression):
    source = f"""shader Reads {{
        RWStructuredBuffer<float> values @register(u0);
        float next_value() {{ return values[0]; }}
        bool classify(float value) {{ return {expression} != {expression}; }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    assert generated.count(expression) >= 2
    assert "0x7fffffffu" not in generated


def test_global_storage_is_not_collapsed_to_one_read():
    source = "shader Shared { groupshared float value; bool classify() { return value != value; } }"
    generated = HLSLCodeGen().generate(parse(source))
    assert "(value != value)" in generated and "0x7fffffffu" not in generated


def _native_source(width):
    kind = "float" + (str(width) if width > 1 else "")
    boolean = "bool" + (str(width) if width > 1 else "")
    values = ", ".join(
        f"as_type<float>(inputWords[tid * {width}u + {i}u])" for i in range(width)
    )
    writes = "\n".join(
        f"outputWords[(tid * {width}u + {i}u) * 2u] = uint(equal{'.' + 'xyzw'[i] if width > 1 else ''}); "
        f"outputWords[(tid * {width}u + {i}u) * 2u + 1u] = uint(unequal{'.' + 'xyzw'[i] if width > 1 else ''});"
        for i in range(width)
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{boolean} same({kind} value) {{ return value == value; }}
{boolean} different({kind} value) {{ return value != value; }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {{
    {kind} value = {kind}({values});
    {boolean} equal = same(value);
    {boolean} unequal = different(value);
    {writes}
}}
"""


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_self_comparisons_execute_with_raw_float_payloads(tmp_path, width):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native floating classification")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path,
        target,
        "uint",
        (1, 1, 1),
        source=_native_source(width),
        software_subgroups=False,
    )
    words = [
        0x00000000,
        0x7FC12345,
        0x80000000,
        0xFFC12345,
        0x00000001,
        0xFF800000,
        0x7F800001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F800000,
        0xFF800001,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ] * 3
    wanted = []
    for word in words:
        nan = (word & 0x7FFFFFFF) > 0x7F800000
        wanted.extend((int(not nan), int(nan)))
    guards = [0xBAD00000 + i for i in range(17)]

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
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {key: value for key, value in data.items() if key != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [len(words) // width, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(
        request.artifact_path.read_text(),
        target,
        compiled,
        metal_compile_flags=("-fno-fast-math",),
    )
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
            original_source, original_module = _compile(
                source, target, original, metal_compile_flags=("-fno-fast-math",)
            )
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
                    "width": width,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "metalCompileFlags": (
                        ["-fno-fast-math"] if target == "metal" else []
                    ),
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
            assert record["outputs"] == expected
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_float_self_comparison_gate_is_required_on_every_native_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate subgroup-guarded returns"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_directx_self_comparisons.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_directx_self_comparisons.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
