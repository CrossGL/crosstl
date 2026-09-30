"""OpenGL lowering and numerical execution of canonical Metal math."""

import json
import math
import os
import shutil
import subprocess
import sys

import pytest

from crosstl import translate
from crosstl.project.native_runtime_drivers import OpenGLComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeFixture,
    RuntimeResourceBinding,
)
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen, OpenGLMetalMathError

REQUIRE_RUNTIME_ENV = "CROSTL_REQUIRE_OPENGL_METAL_MATH"
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
    results[16 * i] = metal::fabs(x);
    results[16 * i + 1] = metal::fmin(x, y);
    results[16 * i + 2] = metal::fmax(x, y);
    results[16 * i + 3] = metal::select(x, y, i % 2 == 0);
    float2 selected = metal::select(float2(x, y), float2(y, x), bool2(true, false));
    results[16 * i + 4] = selected.x;
    results[16 * i + 5] = selected.y;
    float left_count = 0.0f;
    float right_count = 0.0f;
    results[16 * i + 6] = metal::select(record(left_count, x), record(right_count, y), false);
    float2 magnitude = metal::fabs(float2(x, y));
    float2 lower = metal::fmin(float2(x, y), float2(y, x));
    float2 upper = metal::fmax(float2(x, y), float2(y, x));
    results[16 * i + 8] = magnitude.x;
    results[16 * i + 9] = magnitude.y;
    results[16 * i + 10] = lower.x;
    results[16 * i + 11] = lower.y;
    results[16 * i + 12] = upper.x;
    results[16 * i + 13] = upper.y;
    results[16 * i + 14] = metal::fmin(record(left_count, x), record(right_count, y));
    results[16 * i + 15] = metal::fmax(record(left_count, x), record(right_count, y));
    results[16 * i + 7] = left_count + right_count;
}
"""


def _translate(tmp_path):
    source = tmp_path / "math.metal"
    source.write_text(SOURCE, encoding="utf-8")
    return translate(str(source), backend="opengl", format_output=False)


def _compile(generated, tmp_path):
    artifact = tmp_path / "math.comp"
    module = tmp_path / "math.spv"
    artifact.write_text(generated, encoding="utf-8")
    compiler = shutil.which("glslangValidator")
    validator = shutil.which("spirv-val")
    if compiler is None or validator is None:
        assert (
            os.environ.get(REQUIRE_RUNTIME_ENV) != "1"
        ), "GLSL and SPIR-V tools required"
        return artifact, module
    for command in (
        [
            compiler,
            "-G",
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(artifact),
            "-o",
            str(module),
        ],
        [validator, "--target-env", "opengl4.5", str(module)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "WARNING" not in result.stdout + result.stderr
    return artifact, module


def _compute(body, helpers=""):
    return parse(f"""shader Math {{
        {helpers}
        RWStructuredBuffer<float> results @ binding(0);
        compute {{
            layout(local_size_x = 1) in;
            void main() {{ {body} }}
        }}
    }}""")


def test_metal_math_translates_and_compiles_for_opengl(tmp_path):
    generated = _translate(tmp_path)
    assert "abs(x)" in generated
    assert "crossgl_fmin_float(x, y)" in generated
    assert "crossgl_fmax_float(x, y)" in generated
    assert "crossgl_select_float_bool(x, y, ((i % 2) == 0))" in generated
    assert "isnan(left) ? right" in generated
    assert "condition.x ? right.x : left.x" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "value_type",
    [
        "float",
        "vec2",
        "vec3",
        "vec4",
        "double",
        "dvec2",
        "int",
        "ivec3",
        "uint",
        "uvec4",
        "bool",
        "bvec2",
        "int64_t",
        "u64vec2",
    ],
)
@pytest.mark.parametrize("scalar_condition", [False, True])
def test_opengl_selection_preserves_scalar_and_vector_types(
    tmp_path, value_type, scalar_condition
):
    width = int(value_type[-1]) if value_type[-1].isdigit() else 1
    mask = "bool" if scalar_condition or width == 1 else f"bvec{width}"
    expression = f"select({value_type}(0), {value_type}(1), {mask}(true))"
    generated = GLSLCodeGen().generate(_compute(f"{value_type} chosen = {expression};"))
    assert f"{value_type} crossgl_select_{value_type}_{mask}" in generated
    _compile(generated, tmp_path)


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
def test_opengl_metal_math_result_type(expression, expected):
    ast = parse(f"shader ResultType {{ {expected} f() {{ return {expression}; }} }}")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    assert GLSLCodeGen().expression_result_type(call) == expected


def test_opengl_nested_metal_math_compiles(tmp_path):
    generated = GLSLCodeGen().generate(_compute("""
        float x = fmax(fabs(-2.0), fabs(1.0));
        float y = fmin(fabs(-2.0), fabs(1.0));
        results[0] = select(select(x, y, false), x, true);
        ivec2 chosen = select(ivec2(1), ivec2(2), select(bvec2(false), bvec2(true), bvec2(true, false)));
        results[1] = float(chosen.x);
    """))
    _compile(generated, tmp_path)


@pytest.mark.parametrize("value_type", ["float", "vec3", "double", "dvec4"])
def test_opengl_floating_math_preserves_width(tmp_path, value_type):
    generated = GLSLCodeGen().generate(
        _compute(
            f"{value_type} result = fmin(fabs({value_type}(-2.0)), fmax({value_type}(1.0), {value_type}(3.0)));"
        )
    )
    assert f"{value_type} crossgl_fmin_{value_type}" in generated
    assert f"{value_type} crossgl_fmax_{value_type}" in generated
    _compile(generated, tmp_path)


def test_opengl_metal_math_helpers_reset_between_modules():
    generator = GLSLCodeGen()
    first = generator.generate(_compute("results[0] = fmin(1.0, 2.0);"))
    second = generator.generate(_compute("results[0] = 1.0;"))
    assert "crossgl_fmin_float" in first
    assert "crossgl_fmin_float" not in second


@pytest.mark.parametrize("version", ["#version 330 core", "#version 300 es"])
def test_opengl_metal_math_rejects_unproven_profiles(version):
    generator = GLSLCodeGen()
    generator.current_glsl_version_line = version
    ast = parse("shader Math { float f() { return fmin(1.0, 2.0); } }")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    with pytest.raises(OpenGLMetalMathError) as error:
        generator.generate_expression(call)
    assert error.value.reason == "unsupported-profile"
    assert error.value.target_profile == version


@pytest.mark.parametrize(
    "name,arity", [("fabs", 1), ("fmin", 2), ("fmax", 2), ("select", 3)]
)
def test_opengl_metal_math_preserves_source_helpers(tmp_path, name, arity):
    parameters = ", ".join(f"float a{i}" for i in range(arity))
    arguments = ", ".join(f"{i}.0" for i in range(arity))
    generated = GLSLCodeGen().generate(
        _compute(
            f"results[0] = {name}({arguments});",
            f"float {name}({parameters}) {{ return a0 + 10.0; }}",
        )
    )
    assert f"results[0] = {name}({arguments});" in generated
    _compile(generated, tmp_path)


def test_opengl_metal_math_helper_names_avoid_source_names(tmp_path):
    generated = GLSLCodeGen().generate(
        _compute(
            """
        float crossgl_select_float_bool = 3.0;
        results[0] = select(crossgl_select_float_bool, 1.0, false);
        results[1] = fmin(crossgl_fmin_float(2.0), 4.0);
    """,
            "float crossgl_fmin_float(float x) { return x + 1.0; }",
        )
    )
    assert "crossgl_select_float_bool_2(" in generated
    assert "crossgl_fmin_float_2(" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "operation,builtin,expression",
    [
        ("fabs", "abs", "fabs(1.0)"),
        ("fmin", "isnan", "fmin(1.0, 2.0)"),
        ("fmax", "isnan", "fmax(1.0, 2.0)"),
    ],
)
def test_opengl_metal_math_rejects_shadowed_builtins(operation, builtin, expression):
    with pytest.raises(OpenGLMetalMathError) as error:
        GLSLCodeGen().generate(
            _compute(
                f"results[0] = {expression};",
                f"float {builtin}(float x) {{ return x + 1.0; }}",
            )
        )
    assert error.value.operation == operation
    assert error.value.reason == "target-builtin-shadowed"


@pytest.mark.parametrize(
    "expression,reason",
    [
        ("fabs()", "invalid-arity"),
        ("fmin(1.0)", "invalid-arity"),
        ("fmax(1.0, 2.0, 3.0)", "invalid-arity"),
        ("select(1.0, 2.0)", "invalid-arity"),
        ("fabs(1)", "non-floating-operand"),
        ("fmin(vec2(1.0), vec3(1.0))", "operand-type-mismatch"),
        ("select(1.0, 2.0, 1)", "non-boolean-condition"),
        ("select(1.0, 2.0, 0.5)", "non-boolean-condition"),
        ("select(1.0, 2.0, unknown)", "unresolved-operand-type"),
        ("select(vec2(1.0), vec2(2.0), bvec3(true))", "condition-shape-mismatch"),
    ],
)
def test_opengl_metal_math_diagnostics(expression, reason):
    ast = _compute(f"results[0] = {expression};")
    call = next(node for node in ast.walk() if isinstance(node, FunctionCallNode))
    call.source_location = {"line": 6, "column": 28}
    with pytest.raises(OpenGLMetalMathError) as error:
        GLSLCodeGen().generate(ast)
    assert error.value.reason == reason
    assert (
        error.value.project_diagnostic_code
        == "project.translate.opengl-metal-math-unrepresentable"
    )
    assert error.value.source_location == call.source_location


def test_opengl_metal_math_executes(tmp_path):
    if os.environ.get(REQUIRE_RUNTIME_ENV) != "1":
        pytest.skip(f"set {REQUIRE_RUNTIME_ENV}=1 to require native OpenGL execution")
    assert sys.platform.startswith("linux"), "Native EGL checks require Linux"
    generated = _translate(tmp_path)
    artifact, module = _compile(generated, tmp_path)
    pairs = [
        (3.0, -4.0),
        (-2.0, 5.0),
        (math.nan, 7.0),
        (7.0, math.nan),
        (-math.inf, math.inf),
        (math.inf, -math.inf),
        (math.nan, math.nan),
        (-0.0, 0.0),
        (0.0, -0.0),
    ]
    expected = []
    for index, (x, y) in enumerate(pairs):
        minimum = y if math.isnan(x) else (x if math.isnan(y) else min(x, y))
        maximum = y if math.isnan(x) else (x if math.isnan(y) else max(x, y))
        expected.extend(
            [
                abs(x),
                minimum,
                maximum,
                y if index % 2 == 0 else x,
                y,
                y,
                x,
                6.0,
                abs(x),
                abs(y),
                minimum,
                minimum,
                maximum,
                maximum,
                minimum,
                maximum,
            ]
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
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=generated,
        buffers=buffers,
        constants={},
        entry_point="main",
        dispatch=RuntimeDispatchGeometry(
            entry_point="main",
            workgroup_size=(1, 1, 1),
            workgroup_count=(len(pairs), 1, 1),
        ),
    )
    runtime = OpenGLComputeRuntime(context_backends=("egl",))
    availability = runtime.is_available(
        None,
        RuntimeExecutionRequest(
            fixture=RuntimeFixture(
                id="opengl-metal-math",
                selector=RuntimeArtifactSelector(target="opengl"),
                entry_point="main",
            ),
            artifact=request.artifact,
            artifact_path=artifact,
            project_root=tmp_path,
        ),
    )
    assert availability.available, availability.reason
    (tmp_path / "runtime.json").write_text(
        json.dumps(availability.details, indent=2), encoding="utf-8"
    )
    outputs = runtime.dispatch(None, None, request)
    (tmp_path / "outputs.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    actual = outputs["results"]["values"]
    assert len(actual) == len(expected)
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        assert math.isnan(observed) if math.isnan(reference) else observed == reference
        if reference == 0.0 and index % 16 in {0, 3, 4, 5, 6, 8, 9}:
            assert math.copysign(1.0, observed) == math.copysign(1.0, reference)
