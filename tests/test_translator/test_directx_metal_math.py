"""DirectX lowering of canonical Metal math and eager selection."""

import json
import math
import os
import shutil
import subprocess
import sys

import pytest

from crosstl import translate
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLSelectOverloadError
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_warnings_clean_if_available,
)

REQUIRE_RUNTIME_ENV = "CROSTL_REQUIRE_DIRECTX_METAL_MATH"
SOURCE = """
#include <metal_stdlib>
using namespace metal;

float record(thread float& counter, float value) {
    counter += 1.0f;
    return value;
}

kernel void math_probe(
    device const float* values [[buffer(0)]],
    device float* results [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    float x = values[2 * i];
    float y = values[2 * i + 1];
    results[8 * i] = metal::fabs(x);
    results[8 * i + 1] = metal::fmin(x, y);
    results[8 * i + 2] = metal::fmax(x, y);
    results[8 * i + 3] = metal::select(x, y, i % 2 == 0);
    float2 selected = metal::select(float2(x, y), float2(y, x), bool2(true, false));
    results[8 * i + 4] = selected.x;
    results[8 * i + 5] = selected.y;
    float left_count = 0.0f;
    float right_count = 0.0f;
    results[8 * i + 6] = metal::select(record(left_count, x), record(right_count, y), false);
    results[8 * i + 7] = left_count + right_count;
}
"""


def _translate(tmp_path):
    source = tmp_path / "math.metal"
    source.write_text(SOURCE, encoding="utf-8")
    return translate(str(source), backend="directx", format_output=False)


def test_metal_math_translates_and_compiles_for_directx(tmp_path):
    generated = _translate(tmp_path)
    assert "abs(x)" in generated
    assert "min(x, y)" in generated
    assert "max(x, y)" in generated
    assert "select(((i % 2) == 0), y, x)" in generated
    assert "select(bool2(true, false), float2(y, x), float2(x, y))" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


def test_directx_nested_metal_math_preserves_result_types(tmp_path):
    source = """
    shader NestedMath {
        float stableHypot(float x, float y) {
            float a = fmax(fabs(x), fabs(y));
            float b = fmin(fabs(x), fabs(y));
            float r = (b / a) * (b / a);
            float h1 = sqrt(2.0) * a;
            float h2 = a + (a * r) / 2.0;
            float h3 = a * sqrt(1.0 + r);
            bool first = a == b;
            bool second = sqrt(1.0 + r) == 1.0 && r > 0.0;
            return select(select(h3, h2, second), h1, first);
        }
        RWStructuredBuffer<float> results @ binding(0);
        compute {
            layout(local_size_x = 1) in;
            void main(uvec3 index @ gl_GlobalInvocationID) {
                results[index.x] = stableHypot(float(index.x), 2.0);
                ivec2 result = select(ivec2(1), ivec2(2), select(bvec2(false), bvec2(true), bvec2(true, false)));
                results[index.x + 1u] = float(result.x);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate(parse(source))
    assert "max(abs(x), abs(y))" in generated
    assert "return select(first, h1, select(second, h2, h3));" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


@pytest.mark.parametrize("vector", [False, True])
def test_hlsl_select_roundtrip_retains_hlsl_argument_order(tmp_path, vector):
    result_type = "float2" if vector else "float"
    condition = "bool2(true, false)" if vector else "false"
    source = tmp_path / "selection.hlsl"
    source.write_text(
        f"""RWStructuredBuffer<{result_type}> results : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {{
            results[id.x] = select({condition}, {result_type}(20.0), {result_type}(10.0));
        }}""",
        encoding="utf-8",
    )
    intermediate = translate(str(source), backend="crossgl", format_output=False)
    assert "select(" in intermediate
    generated = translate(str(source), backend="directx", format_output=False)
    assert f"select({condition}," in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


def test_hlsl_select_roundtrip_preserves_source_helper(tmp_path):
    source = tmp_path / "helper.hlsl"
    source.write_text(
        """float select(float x, float y, float z, float w) { return x - y + z + w; }
        RWStructuredBuffer<float> results : register(u0);
        [numthreads(1, 1, 1)]
        void main() { results[0] = select(10.0, 20.0, 30.0, 40.0); }
        """,
        encoding="utf-8",
    )
    intermediate = translate(str(source), backend="crossgl", format_output=False)
    assert "select(10.0, 20.0, 30.0, 40.0)" in intermediate
    generated = translate(str(source), backend="directx", format_output=False)
    assert "select_(10.0, 20.0, 30.0, 40.0)" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


def test_hlsl_select_rejects_ambiguous_source_overload(tmp_path):
    source = tmp_path / "ambiguous.hlsl"
    source.write_text(
        """float select(float x, float y, float z) { return x - y + z; }
        RWStructuredBuffer<float> results : register(u0);
        [numthreads(1, 1, 1)]
        void main() { results[0] = select(10.0, 20.0, 30.0); }
        """,
        encoding="utf-8",
    )
    with pytest.raises(HLSLSelectOverloadError) as error:
        translate(str(source), backend="directx", format_output=False)
    assert error.value.reason == "select-overload-ownership-unresolved"


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("fabs(vec2(1.0))", "vec2"),
        ("fmin(vec2(1.0), vec2(2.0))", "vec2"),
        ("fmax(vec2(1.0), vec2(2.0))", "vec2"),
        ("select(ivec2(1), ivec2(2), bvec2(true))", "ivec2"),
        ("select(false, true, true)", "bool"),
    ],
)
def test_directx_metal_math_result_type(expression, expected):
    ast = parse(f"shader ResultType {{ {expected} f() {{ return {expression}; }} }}")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    assert HLSLCodeGen().expression_result_type(call) == expected


@pytest.mark.parametrize(
    "name,arity", [("fabs", 1), ("fmin", 2), ("fmax", 2), ("select", 3)]
)
def test_directx_metal_math_preserves_source_helpers(tmp_path, name, arity):
    parameters = ", ".join(f"float a{i}" for i in range(arity))
    arguments = ", ".join(f"{i}.0" for i in range(arity))
    source = f"""
        shader SourceHelper {{
            float {name}({parameters}) {{ return a0 + 10.0; }}
            RWStructuredBuffer<float> results @ binding(0);
            compute {{
                layout(local_size_x = 1) in;
                void main() {{ results[0] = {name}({arguments}); }}
            }}
        }}
    """
    generated = HLSLCodeGen().generate(parse(source))
    target_name = "select_" if name == "select" else name
    assert f"float {target_name}({parameters})" in generated
    assert f"results[0] = {target_name}({arguments});" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


def test_directx_source_select_alias_avoids_existing_names(tmp_path):
    ast = parse("""shader SelectNames {
        float select(float x, float y, bool choose) { return choose ? y : x; }
        float select_(float value) { return value + 1.0; }
        RWStructuredBuffer<float> results @ binding(0);
        compute {
            layout(local_size_x = 1) in;
            void main() {
                float select__ = 3.0;
                results[0] = select(select_(1.0), select__, false);
            }
        }
    }""")
    generated = HLSLCodeGen().generate(ast)
    assert "float select___(float x, float y, bool choose)" in generated
    assert "results[0] = select___(select_(1.0), select__, false);" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


@pytest.mark.parametrize(
    "name,target,arity", [("fabs", "abs", 1), ("fmin", "min", 2), ("fmax", "max", 2)]
)
@pytest.mark.parametrize("binding", ["function", "local", "global"])
def test_directx_metal_math_rejects_shadowed_target(name, target, arity, binding):
    parameters = ", ".join(f"float a{i}" for i in range(arity))
    arguments = ", ".join("1.0" for _ in range(arity))
    declaration = f"float {target}({parameters}) {{ return 99.0; }}"
    global_declaration = (
        declaration
        if binding == "function"
        else (f"float {target} = 99.0;" if binding == "global" else "")
    )
    local_declaration = f"float {target} = 99.0;" if binding == "local" else ""
    ast = parse(f"""shader ShadowedTarget {{
        {global_declaration}
        float probe() {{ {local_declaration} return {name}({arguments}); }}
    }}""")
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(ast)
    assert error.value.reason == "metal-math-target-shadowed"


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("2.0", "select-condition-not-boolean"),
        ("ivec2(0, -1)", "select-condition-not-boolean"),
        ("unknownCondition()", "select-condition-unresolved"),
    ],
)
def test_directx_select_rejects_unproven_condition(condition, reason):
    ast = parse(f"""shader InvalidSelect {{
        float probe() {{ return select(1.0, 2.0, {condition}); }}
    }}""")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    call.source_location = {"line": 2, "column": 32}
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(ast)
    assert error.value.reason == reason
    assert error.value.source_location == call.source_location


@pytest.mark.parametrize(
    "expression", ["fabs()", "fmin(1.0)", "fmax(1.0)", "select(1.0, 2.0)"]
)
def test_directx_metal_math_rejects_invalid_arity(expression):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(
            parse(f"shader Arity {{ float f() {{ return {expression}; }} }}")
        )
    assert error.value.reason == "metal-math-invalid-arity"


def test_directx_metal_math_executes(tmp_path):
    if os.environ.get(REQUIRE_RUNTIME_ENV) != "1":
        pytest.skip(f"set {REQUIRE_RUNTIME_ENV}=1 to require native Direct3D execution")
    assert sys.platform == "win32", "Native Direct3D checks require Windows"
    compiler = shutil.which("dxc")
    assert compiler is not None, "DXC is required"
    generated = _translate(tmp_path)
    artifact = tmp_path / "math.hlsl"
    module = tmp_path / "math.dxil"
    artifact.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [
            compiler,
            "-T",
            "cs_6_6",
            "-E",
            "CSMain",
            "-WX",
            str(artifact),
            "-Fo",
            str(module),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    pairs = [
        (3.0, -4.0),
        (-2.0, 5.0),
        (math.nan, 7.0),
        (7.0, math.nan),
        (-math.inf, math.inf),
        (math.inf, -math.inf),
        (math.nan, math.nan),
    ]
    expected = []
    for index, (x, y) in enumerate(pairs):
        minimum = y if math.isnan(x) else (x if math.isnan(y) else min(x, y))
        maximum = y if math.isnan(x) else (x if math.isnan(y) else max(x, y))
        expected.extend(
            [abs(x), minimum, maximum, y if index % 2 == 0 else x, y, y, x, 2.0]
        )
    buffers = {}
    for name, binding, values, count in (
        ("values", 0, [value for pair in pairs for value in pair], len(pairs) * 2),
        ("results", 1, None, len(expected)),
    ):
        output = values is None
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                type_name=("RW" if output else "") + "StructuredBuffer<float>",
                set=0,
                binding=binding,
                access="read_write" if output else "read",
            ),
            source="expectedOutput" if output else "input",
            dtype="float32",
            shape=(count,),
            value=values,
        )
    request = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(1, 1, 1),
            workgroup_count=(len(pairs), 1, 1),
        ),
    )
    runtime = DirectXComputeRuntime()
    availability = runtime.is_available(None, None)
    assert availability.available, availability.reason
    outputs = runtime.dispatch(None, None, request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    actual = outputs["results"]["values"]
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        assert math.isnan(observed) if math.isnan(reference) else observed == reference
