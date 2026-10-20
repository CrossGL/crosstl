"""Signed increments used as values keep wrapping semantics through step helpers (ADR 0001, section 1)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.semantics import RuleInexpressibleError
from crosstl.translator.semantics.step_values import (
    STEP_HELPER_TARGETS,
    step_helper_source,
)

ROOT = Path(__file__).resolve().parents[2]
METAL_REQUIRE_ENV = "CROSTL_REQUIRE_METAL_STEP_VALUES"

GLSL_KERNEL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp {{ int a[]; }};
layout(std430, binding = 1) buffer Outp {{ int o[]; }};
void main() {{
    uint i = gl_GlobalInvocationID.x;
    int x = a[i];
{body}
}}
"""

# Every form of a signed step used as a value, at each value's wrapping edge when the input is INT_MAX or INT_MIN.
HLSL_STEPS = """
RWStructuredBuffer<int> values : register(u0);
RWStructuredBuffer<int> results : register(u1);

int Pass(int value)
{
    return value;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int a = values[id.x];
    int b = a;
    int c = a;
    int d = a;
    int k = a;
    int trips = 0;
    while (k++ != values[id.x] + 3 && trips < 8)
    {
        trips += 1;
    }
    uint base = id.x * 10;
    results[base + 0] = a++;
    results[base + 1] = a;
    results[base + 2] = ++b;
    results[base + 3] = b;
    results[base + 4] = Pass(c--);
    results[base + 5] = c;
    results[base + 6] = --d;
    results[base + 7] = d;
    results[base + 8] = k;
    results[base + 9] = trips;
}
"""
STEP_INPUTS = [0x7FFFFFFF, 0x80000000, 0, 0xFFFFFFFF, 5]
GUARD = [0xDEADBEEF] * 4


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def glsl(tmp_path, body, backend):
    return translate(tmp_path, GLSL_KERNEL.format(body=body), backend, ".comp")


def step_reference(value):
    """The int32 results of ``HLSL_STEPS`` for one input, with two's-complement wrapping."""

    def wrap(number):
        return number & 0xFFFFFFFF

    return [
        value,
        wrap(value + 1),
        wrap(value + 1),
        wrap(value + 1),
        value,
        wrap(value - 1),
        wrap(value - 1),
        wrap(value - 1),
        wrap(value + 4),
        3,
    ]


def test_step_helper_sources_use_the_wrapping_polyfill():
    assert step_helper_source("__cgl_wrap_add", True, 1) == (
        "int cgl_rule_wrap_postinc(inout int x0) "
        "{ int x1 = x0; x0 = asint(asuint(x0) + 1u); return x1; }"
    )
    assert step_helper_source("__cgl_wrap_sub", False, 3) == (
        "ivec3 cgl_rule_wrap_predec3(inout ivec3 x0) "
        "{ x0 = asint(asuint(x0) - 1u); return x0; }"
    )


def test_increment_used_as_a_value_calls_a_step_helper_on_metal(tmp_path):
    body = (
        "    int y = x++;\n    int z = --x;\n    ivec2 v = ivec2(x, 1);\n    ivec2 w = v++;\n"
        "    int n = 0;\n    for (int k = 0; k++ < 3;) { n += 1; }\n"
        "    o[i] = y + z + w.x + v.y + n;"
    )
    metal = glsl(tmp_path, body, "metal")
    assert "int cgl_rule_wrap_postinc(thread int& x0) {" in metal
    assert "int2 cgl_rule_wrap_postinc2(thread int2& x0) {" in metal
    assert "int cgl_rule_wrap_predec(thread int& x0) {" in metal
    assert "x0 = as_type<int>(as_type<uint>(x0) + 1u);" in metal
    assert "int y = cgl_rule_wrap_postinc(x);" in metal
    assert "int z = cgl_rule_wrap_predec(x);" in metal
    assert "int2 w = cgl_rule_wrap_postinc2(v);" in metal
    assert "for (int k = 0; cgl_rule_wrap_postinc(k) < 3; )" in metal
    assert "x++" not in metal and "--x" not in metal
    # Each helper is emitted once, ahead of its callers.
    assert metal.count("int cgl_rule_wrap_postinc(thread int& x0)") == 1
    assert metal.index("cgl_rule_wrap_postinc(thread") < metal.index("kernel void")


def test_statement_increments_keep_the_in_place_rewrite(tmp_path):
    metal = glsl(tmp_path, "    x++;\n    --x;\n    o[i] = x;", "metal")
    assert "x = as_type<int>(as_type<uint>(x) + 1u);" in metal
    assert "x = as_type<int>(as_type<uint>(x) - 1u);" in metal
    assert "cgl_rule_wrap_" not in metal


@pytest.mark.parametrize(
    ("backend", "parameter"),
    [("directx", "inout int x0"), ("slang", "inout int x0")],
)
def test_step_helpers_keep_inout_parameters(tmp_path, backend, parameter):
    assert backend in STEP_HELPER_TARGETS
    output = translate(tmp_path, HLSL_STEPS, backend, ".hlsl")
    assert f"int cgl_rule_wrap_postinc({parameter})" in output
    assert "cgl_rule_wrap_postinc(a)" in output
    assert "Pass(cgl_rule_wrap_postdec(c))" in output
    assert "a++" not in output


@pytest.mark.parametrize("backend", ["cuda", "hip", "mojo", "vulkan"])
def test_increment_used_as_a_value_is_refused_without_inout_write_back(
    tmp_path, backend
):
    assert backend not in STEP_HELPER_TARGETS
    with pytest.raises(RuleInexpressibleError) as refusal:
        glsl(tmp_path, "    int y = x++;\n    o[i] = y;", backend)
    assert "passes inout parameters by value" in str(refusal.value)
    assert (
        refusal.value.project_diagnostic_code == "project.translate.rule-inexpressible"
    )


def test_step_of_an_lvalue_with_side_effects_is_refused(tmp_path):
    source = GLSL_KERNEL.format(body="    o[i] = o[next(x)]++;").replace(
        "void main()", "int next(int v) { return v; }\nvoid main()"
    )
    with pytest.raises(RuleInexpressibleError) as refusal:
        translate(tmp_path, source, "metal", ".comp")
    assert "lvalue with side effects" in str(refusal.value)


def test_program_function_with_a_reserved_step_helper_name_is_refused(tmp_path):
    source = GLSL_KERNEL.format(body="    o[i] = x++;").replace(
        "void main()",
        "int cgl_rule_wrap_postinc(int v) { return v; }\nvoid main()",
    )
    with pytest.raises(RuleInexpressibleError) as refusal:
        translate(tmp_path, source, "metal", ".comp")
    assert "reserved for the rule helper" in str(refusal.value)


def test_glsl_keeps_native_increments_used_as_values(tmp_path):
    output = glsl(tmp_path, "    int y = x++;\n    o[i] = y;", "opengl")
    assert "x++" in output
    assert "cgl_rule_wrap_" not in output


def _run(command, cwd):
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True, timeout=180
    ).stdout


def _compile_metal(source, tmp_path):
    shader = tmp_path / "steps.metal"
    shader.write_text(source)
    air, library = tmp_path / "steps.air", tmp_path / "steps.metallib"
    _run(
        ["xcrun", "--sdk", "macosx", "metal", "-Werror", "-c", str(shader)]
        + ["-o", str(air)],
        tmp_path,
    )
    _run(
        ["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(library)], tmp_path
    )
    return library


def test_metal_step_helpers_compile(tmp_path):
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler requires macOS with xcrun")
    metal = translate(tmp_path, HLSL_STEPS, "metal", ".hlsl")
    try:
        library = _compile_metal(metal, tmp_path)
    except subprocess.CalledProcessError as error:
        if "metal" in (error.stderr or "") and "not found" in (error.stderr or ""):
            pytest.skip("the Metal toolchain is not installed")
        raise
    assert library.is_file()


def test_metal_step_values_wrap_at_the_int32_edges(tmp_path):
    if os.environ.get(METAL_REQUIRE_ENV) != "1":
        pytest.skip(f"set {METAL_REQUIRE_ENV}=1 to require native Metal execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    metal = translate(tmp_path, HLSL_STEPS, "metal", ".hlsl")
    library = _compile_metal(metal, tmp_path)
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    _run(["swiftc", str(fixture), "-o", str(runner)], tmp_path)
    expected = [word for value in STEP_INPUTS for word in step_reference(value)]
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [STEP_INPUTS],
                "outputCount": len(expected) + len(GUARD),
                "threadCount": len(STEP_INPUTS),
            }
        )
    )
    readback = json.loads(
        _run([str(runner), str(library), "kernel_main", str(request)], tmp_path)
    )
    (tmp_path / "readback.json").write_text(json.dumps(readback, indent=2))
    values = readback["values"]
    assert values[: len(expected)] == expected
    # The output buffer is prefilled with the guard word; the kernel writes nothing past its results.
    assert values[len(expected) :] == GUARD
