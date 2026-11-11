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
from crosstl.translator.semantics.hub_types import is_pure_callee
from crosstl.translator.semantics.step_values import (
    STEP_HELPER_TARGETS,
    step_helper_name,
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


@pytest.mark.parametrize("backend", ["cuda", "hip", "mojo"])
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


# Steps used as values on a buffer element, a struct member, a vector component, a shared array element and a local
# in a loop condition: each argument lives in a different SPIR-V storage class.
GLSL_STEP_LVALUES = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Inp { int seed[]; };
layout(std430, binding = 1) buffer Outp { int o[]; };
struct S { int m; ivec2 v; };
shared int counts[2];
void main() {
    o[0] = seed[0];
    int r0 = o[0]++;
    S s;
    s.m = -2147483647 - 1;
    s.v = ivec2(5, 2147483647);
    int r1 = s.m--;
    int r2 = ++s.v.y;
    counts[1] = 7;
    int r3 = counts[1]++;
    int k = 0;
    int t = 0;
    while (k++ < 3) { t += 10; }
    o[1] = r0; o[2] = o[0]; o[3] = r1; o[4] = s.m; o[5] = r2; o[6] = s.v.y;
    o[7] = r3; o[8] = counts[1]; o[9] = k; o[10] = t;
}
"""
STEP_LVALUE_RESULTS = [
    -(2**31),
    2**31 - 1,
    -(2**31),
    -(2**31),
    2**31 - 1,
    -(2**31),
    -(2**31),
    7,
    8,
    4,
    30,
]


def _assemble_spirv(tmp_path, assembly, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not installed")
    source, binary = tmp_path / f"{name}.spvasm", tmp_path / f"{name}.spv"
    source.write_text(assembly)
    _run(
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        tmp_path,
    )
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)], tmp_path)
    return binary


@pytest.mark.parametrize(
    ("source", "suffix"), [(HLSL_STEPS, ".hlsl"), (GLSL_STEP_LVALUES, ".comp")]
)
def test_spirv_step_helpers_validate(tmp_path, source, suffix):
    assert "vulkan" in STEP_HELPER_TARGETS
    assembly = translate(tmp_path, source, "vulkan", suffix)
    _assemble_spirv(tmp_path, assembly, "steps")


def _run_spirv_through_metal(tmp_path, source, suffix, inputs, output_count, threads):
    """Run the SPIR-V for ``source`` on the Metal device after spirv-cross converts it to MSL."""
    if os.environ.get(METAL_REQUIRE_ENV) != "1":
        pytest.skip(f"set {METAL_REQUIRE_ENV}=1 to require native Metal execution")
    assert sys.platform == "darwin", "Native Metal execution requires macOS"
    assert shutil.which("spirv-cross"), "spirv-cross is required to run SPIR-V on Metal"
    binary = _assemble_spirv(
        tmp_path, translate(tmp_path, source, "vulkan", suffix), "steps"
    )
    msl = _run(
        ["spirv-cross", str(binary), "--msl", "--msl-version", "20100"], tmp_path
    )
    library = _compile_metal(msl, tmp_path)
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    _run(["swiftc", str(fixture), "-o", str(runner)], tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"inputs": inputs, "outputCount": output_count, "threadCount": threads}
        )
    )
    readback = _run([str(runner), str(library), "main0", str(request)], tmp_path)
    return json.loads(readback)["values"]


def test_spirv_step_values_wrap_at_the_int32_edges(tmp_path):
    expected = [word for value in STEP_INPUTS for word in step_reference(value)]
    values = _run_spirv_through_metal(
        tmp_path,
        HLSL_STEPS,
        ".hlsl",
        [STEP_INPUTS],
        len(expected) + len(GUARD),
        len(STEP_INPUTS),
    )
    assert values[: len(expected)] == expected
    assert values[len(expected) :] == GUARD


def test_spirv_step_values_write_back_to_every_storage_class(tmp_path):
    values = _run_spirv_through_metal(
        tmp_path,
        GLSL_STEP_LVALUES,
        ".comp",
        [[0x7FFFFFFF]],
        len(STEP_LVALUE_RESULTS),
        1,
    )
    assert values == [word & 0xFFFFFFFF for word in STEP_LVALUE_RESULTS]


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


@pytest.mark.parametrize("intrinsic", ["__cgl_wrap_add", "__cgl_wrap_sub"])
@pytest.mark.parametrize("postfix", [True, False])
@pytest.mark.parametrize("lanes", [1, 3])
def test_step_helpers_are_not_pure(intrinsic, postfix, lanes):
    # A step helper writes its operand, so an expression that calls it must not be evaluated twice.
    assert not is_pure_callee(step_helper_name(intrinsic, postfix, lanes))
    assert is_pure_callee(intrinsic)


ARRAY_COPY_STEP = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer Data { float o[8]; } data;
void main() {
    float b[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0));
    int i = 0;
    float part[3];
    part = b[i++];
    data.o[0] = part[0] + part[2];
    data.o[1] = float(i);
}
"""


def test_metal_binds_a_stepped_array_row_once(tmp_path):
    # Metal copies C arrays element by element. The stepped row is bound to a thread reference first, so 'i' steps
    # once; the copy loop used to repeat 'b[cgl_rule_wrap_postinc(i)][k]' for every element.
    path = tmp_path / "copy.comp"
    path.write_text(ARRAY_COPY_STEP)

    code = "".join(crosstl.translate(str(path), backend="metal").split())

    assert "threadauto&__crossgl_array_source_0=b[cgl_rule_wrap_postinc(i)];" in code
    assert (
        "part[__crossgl_array_index_1]=__crossgl_array_source_0[__crossgl_array_index_1];"
        in code
    )
    assert code.count("cgl_rule_wrap_postinc(i)") == 1


def test_metal_refuses_a_stepped_copy_from_threadgroup_storage(tmp_path):
    # Only thread storage is bound by reference; a shared array would need a threadgroup reference.
    path = tmp_path / "copy.comp"
    path.write_text(
        ARRAY_COPY_STEP.replace(
            "    float b[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0));\n",
            "    b[0][0] = 1.0;\n",
        ).replace("void main() {", "shared float b[2][3];\nvoid main() {")
    )

    with pytest.raises(Exception, match="operands have effects"):
        crosstl.translate(str(path), backend="metal")
