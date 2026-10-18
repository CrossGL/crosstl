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
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_SELF_COMPARISONS"


def _source(width, operator, precision="float"):
    kind = precision + (str(width) if width > 1 else "")
    boolean = "bool" + (str(width) if width > 1 else "")
    values = ", ".join(
        (
            f"inputs[id.x * {width}u + {i}u]"
            if precision == "half"
            else f"asfloat(inputs[id.x * {width}u + {i}u])"
        )
        for i in range(width)
    )
    writes = "\n".join(
        f"outputs[id.x * {width}u + {i}u] = uint(result{'.' + 'xyzw'[i] if width > 1 else ''});"
        for i in range(width)
    )
    return f"""shader Classification {{
        StructuredBuffer<{"half" if precision == "half" else "uint"}> inputs @register(t0);
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
@pytest.mark.parametrize("precision", ["float", "half"])
def test_self_comparisons_retain_dynamic_nan_classification(
    tmp_path, width, operator, precision
):
    generated = HLSLCodeGen().generate_stage(
        parse(_source(width, operator, precision)), "compute"
    )
    comparison = ">" if operator == "!=" else "<="
    bits, mask, infinity = (
        ("asuint16", "0x7fffu", "0x7c00u")
        if precision == "half"
        else ("asuint", "0x7fffffffu", "0x7f800000u")
    )
    assert f"(({bits}(value) & {mask}) {comparison} {infinity})" in generated
    artifact, _module = _compile(
        generated, "directx", tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )
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
            "-enable-16bit-types",
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
    assert len(re.findall(r"icmp u(?:gt|lt) i(?:16|32)", ir)) == width
    assert "fcmp fast" not in ir


@pytest.mark.parametrize("kind", ["int", "uint", "bool", "double"])
def test_other_operand_types_remain_unchanged(kind):
    source = (
        f"shader Other {{ bool classify({kind} value) {{ return value != value; }} }}"
    )
    generated = HLSLCodeGen().generate(parse(source))
    assert "(value != value)" in generated and "0x7fffffffu" not in generated


@pytest.mark.parametrize("dtype", ["bfloat16", "Narrow"])
@pytest.mark.parametrize("operator", ["==", "!="])
def test_bfloat_self_comparisons_classify_logical_payload(dtype, operator):
    source = f"""shader Classification {{
        typedef bfloat16 Narrow;
        bool classify({dtype} value) {{ return value {operator} value; }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    comparison = ">" if operator == "!=" else "<="
    assert f"((uint(value) & 0x7fffu) {comparison} 0x7f80u)" in generated
    assert "__crossgl_bfloat16_to_float" not in generated


@pytest.mark.parametrize("expression", ["values[0]", "next_value()"])
def test_bfloat_self_comparison_preserves_repeated_evaluations(expression):
    source = f"""shader Reads {{
        RWStructuredBuffer<bfloat16> values @register(u0);
        bfloat16 next_value() {{ return values[0]; }}
        bool classify(bfloat16 value) {{ return {expression} != {expression}; }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    assert "0x7f80u" not in generated
    assert generated.count("__crossgl_bfloat16_to_float(uint(") >= 2


def test_bfloat_postfix_self_comparison_remains_diagnostic():
    source = """shader Mutation {
        bool classify(bfloat16 value) { return value++ != value++; }
    }"""
    with pytest.raises(ValueError, match="cannot preserve postfix"):
        HLSLCodeGen().generate(parse(source))


def test_bfloat_global_storage_is_not_collapsed_to_one_read():
    generated = HLSLCodeGen().generate(parse("""shader Shared {
            groupshared bfloat16 value;
            bool classify() { return value != value; }
        }"""))
    assert "0x7f80u" not in generated
    assert generated.count("__crossgl_bfloat16_to_float(uint(value))") == 2


@pytest.mark.parametrize("expression", ["values[0]", "next_value()", "value++"])
@pytest.mark.parametrize("precision", ["float", "half"])
def test_self_comparison_does_not_collapse_repeated_evaluations(expression, precision):
    source = f"""shader Reads {{
        RWStructuredBuffer<{precision}> values @register(u0);
        {precision} next_value() {{ return values[0]; }}
        bool classify({precision} value) {{ return {expression} != {expression}; }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    rendered = (
        "asfloat16(values[uint(0)])"
        if precision == "half" and expression == "values[0]"
        else expression
    )
    assert generated.count(rendered) >= 2
    assert "0x7fffffffu" not in generated
    assert "0x7fffu" not in generated


@pytest.mark.parametrize("precision", ["float", "half"])
def test_global_storage_is_not_collapsed_to_one_read(precision):
    source = f"shader Shared {{ groupshared {precision} value; bool classify() {{ return value != value; }} }}"
    generated = HLSLCodeGen().generate(parse(source))
    assert "(value != value)" in generated and "0x7fffffffu" not in generated
    assert "0x7fffu" not in generated


def _native_source(width, aggregate=False, precision="float"):
    kind = precision + (str(width) if width > 1 else "")
    boolean = "bool" + (str(width) if width > 1 and not aggregate else "")
    values = ", ".join(
        (
            f"as_type<{precision}>(ushort(inputWords[tid * {width}u + {i}u]))"
            if precision in {"half", "bfloat"}
            else f"as_type<float>(inputWords[tid * {width}u + {i}u])"
        )
        for i in range(width)
    )
    writes = "\n".join(
        f"outputWords[(tid * {width}u + {i}u) * 2u] = uint(equal{'.' + 'xyzw'[i] if width > 1 and not aggregate else ''}); "
        f"outputWords[(tid * {width}u + {i}u) * 2u + 1u] = uint(unequal{'.' + 'xyzw'[i] if width > 1 and not aggregate else ''});"
        for i in range(width)
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{boolean} same({kind} value) {{ return {"all(value == value)" if aggregate else "value == value"}; }}
{boolean} different({kind} value) {{ return {"any(value != value)" if aggregate else "value != value"}; }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {{
    {kind} value = {kind}({values});
    {boolean} equal = same(value);
    {boolean} unequal = different(value);
    {writes}
}}
"""


@pytest.mark.parametrize(
    "width,aggregate,precision",
    [
        (width, aggregate, precision)
        for width, aggregate in (
            (1, False),
            (2, False),
            (3, False),
            (4, False),
            (2, True),
            (3, True),
            (4, True),
        )
        for precision in ("float", "half")
    ]
    + [(1, False, "bfloat")],
)
def test_self_comparisons_execute_with_raw_float_payloads(
    tmp_path, width, aggregate, precision
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native floating classification")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    group_size = 2 if precision in {"half", "bfloat"} else 1
    source, descriptor, package = _package(
        tmp_path,
        target,
        "uint",
        (group_size, 1, 1),
        source=_native_source(width, aggregate, precision),
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
    mask, infinity = 0x7FFFFFFF, 0x7F800000
    if precision in {"half", "bfloat"}:
        words = list(range(65536))
        words.extend(range(-len(words) % (width * group_size)))
        mask, infinity = 0x7FFF, 0x7C00 if precision == "half" else 0x7F80
    wanted = []
    for word in words:
        nan = (word & mask) > infinity
        wanted.extend((int(not nan), int(nan)))
    if aggregate:
        wanted = []
        for start in range(0, len(words), width):
            nan = any((word & mask) > infinity for word in words[start : start + width])
            wanted.extend([int(not nan), int(nan)] * width)
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
        {
            "workgroupCount": [len(words) // (width * group_size), 1, 1],
            "workgroupSize": [group_size, 1, 1],
        },
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(
        request.artifact_path.read_text(),
        target,
        compiled,
        metal_compile_flags=("-fno-fast-math",),
        directx_compile_flags=("-enable-16bit-types",),
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
                    "precision": precision,
                    "aggregate": aggregate,
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
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate subgroup-guarded returns"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_directx_self_comparisons.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_directx_self_comparisons.py",
        )
