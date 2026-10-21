"""8- and 16-bit integers keep their width on targets without narrow registers (defect S6)."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics import (
    NarrowIntegerUnsupportedError,
    legalize_narrow_integers,
)
from crosstl.translator.semantics.narrow import (
    NARROW_TARGETS,
    narrow_from_name,
    normalized,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

TARGETS = sorted(NARROW_TARGETS)

CUDA_KERNEL = """__global__ void kernel(const int *inp, int *outp) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
{body}
}}
"""
GLSL_KERNEL = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp {{ int a[]; }};
layout(std430, binding = 1) buffer Outp {{ int o[]; }};
{helpers}
void main() {{
    uint i = gl_GlobalInvocationID.x;
{body}
}}
"""
METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void k(device const int* a [[buffer(0)]], device int* o [[buffer(1)]],
              uint i [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {{
{body}
}}
"""


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    # Metal kernels state the threadgroup size their host dispatches them with.
    options = {"threadgroup_size": 64} if suffix == ".metal" else None
    output = crosstl.translate(
        str(path), backend=backend, format_output=False, source_options=options
    )
    # WGSL from Metal and CUDA kernels has unrelated interface defects (scalar invocation-id builtins and buffer
    # parameters), so only WGSL from GLSL is checked natively.
    if backend == "vulkan" or suffix == ".comp":
        validate(tmp_path, output, backend)
    return output


def validate(tmp_path, output, backend):
    tool = {"vulkan": ("spirv-as", "spirv-val"), "wgsl": ("naga",)}[backend]
    paths = [shutil.which(name) for name in tool]
    if None in paths:
        return
    if backend == "wgsl":
        source = tmp_path / "out.wgsl"
        source.write_text(output)
        result = subprocess.run(
            [paths[0], str(source)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr + output
        return
    source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    source.write_text(output)
    for command in (
        [paths[0], str(source), "-o", str(binary)],
        [paths[1], "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr + output


def parse_cgl(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def wgsl_line(output, needle):
    return next(line.strip() for line in output.splitlines() if needle in line)


# Types and normalization ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "kind", "bits", "lanes"),
    [
        ("uchar", "u", 8, 1),
        ("unsigned char", "u", 8, 1),
        ("char", "i", 8, 1),
        ("int16_t", "i", 16, 1),
        ("ushort", "u", 16, 1),
        ("u8vec3", "u", 8, 3),
        ("vec2<i16>", "i", 16, 2),
        ("uchar4", "u", 8, 4),
        ("short2", "i", 16, 2),
    ],
)
def test_narrow_spellings(name, kind, bits, lanes):
    narrow = narrow_from_name(name)
    assert (narrow.kind, narrow.bits, narrow.lanes) == (kind, bits, lanes)


@pytest.mark.parametrize("name", ["int", "uint", "min16int", "min16uint", "uint4"])
def test_wide_and_minimum_precision_types_are_not_narrow(name):
    assert narrow_from_name(name) is None


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("uchar", 300, 44),
        ("char", 200, -56),
        ("short", 32768, -32768),
        ("u16", -1, 65535),
    ],
)
def test_literal_normalization_folds_modulo_the_width(name, value, expected):
    result = normalized(
        A.LiteralNode(value, A.PrimitiveType("int")), narrow_from_name(name)
    )
    if expected < 0:
        assert isinstance(result, A.UnaryOpNode) and result.operand.value == -expected
    else:
        assert result.value == expected


def test_signed_normalization_is_exempt_from_rule_carrying():
    result = normalized(A.IdentifierNode("x"), narrow_from_name("char"))
    assert result.op == "-" and result.annotations["cgl_rule_exempt"]
    assert result.left.op == "^" and result.left.left.op == "&"


def test_other_targets_are_not_legalized():
    ast = parse_cgl("shader s { uint8 f(uint8 x) { return x + uint8(1); } }")
    for target in ("metal", "directx", "opengl", "cuda"):
        assert legalize_narrow_integers(ast, "cgl", target) is False
    assert (
        repr(ast.functions[0].return_type) == "NamedType(name=uint8, generic_args=[])"
    )


# Stores and conversions -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", TARGETS)
def test_promoted_uchar_arithmetic_wraps_on_the_store(tmp_path, target):
    body = """    unsigned char v = (unsigned char)inp[i];
    v = v + 1;
    outp[i] = (int)v;"""
    output = translate(tmp_path, CUDA_KERNEL.format(body=body), target, ".cu")
    if target == "wgsl":
        assert wgsl_line(output, "var v") == "var v: i32 = (i32(inp[i]) & 255);"
        assert wgsl_line(output, "v = ") == "v = (i32((v + 1)) & 255);"
        assert "u8" not in output
    else:
        assert "OpBitwiseAnd" in output and "OpTypeInt 8" not in output


@pytest.mark.parametrize("target", TARGETS)
def test_promoted_arithmetic_is_not_narrowed_when_read(tmp_path, target):
    body = """    unsigned char v = (unsigned char)inp[i];
    outp[i] = (int)(v * v + 1);"""
    output = translate(tmp_path, CUDA_KERNEL.format(body=body), target, ".cu")
    if target == "wgsl":
        assert wgsl_line(output, "outp[i]") == "outp[i] = i32(((v * v) + 1));"


@pytest.mark.parametrize("target", TARGETS)
def test_signed_char_conversion_sign_extends(tmp_path, target):
    body = """    signed char c = (signed char)inp[i];
    short s = (short)(c * 300);
    outp[i] = (int)c + (int)s;"""
    output = translate(tmp_path, CUDA_KERNEL.format(body=body), target, ".cu")
    if target == "wgsl":
        assert (
            wgsl_line(output, "var c")
            == "var c: i32 = (((i32(inp[i]) & 255) ^ 128) - 128);"
        )
        assert "& 65535) ^ 32768) - 32768)" in wgsl_line(output, "var s")


@pytest.mark.parametrize("target", TARGETS)
def test_explicit_narrow_glsl_arithmetic_wraps_at_its_width(tmp_path, target):
    body = """    uint16_t v = uint16_t(a[i]);
    o[i] = int(v * v) + int(-v);"""
    output = translate(
        tmp_path, GLSL_KERNEL.format(helpers="", body=body), target, ".comp"
    )
    if target == "wgsl":
        line = wgsl_line(output, "o[i]")
        assert line.count("& 65535") == 2, line


@pytest.mark.parametrize("target", TARGETS)
def test_narrow_increment_statement_wraps(tmp_path, target):
    body = """    uchar v = uchar(a[i]);
    v++;
    --v;
    v += uchar(250);
    o[i] = int(v);"""
    output = translate(
        tmp_path, METAL_KERNEL.format(helpers="", body=body), target, ".metal"
    )
    if target == "wgsl":
        assert "v = (i32((v + 1)) & 255);" in output
        assert "v = (i32((v - 1)) & 255);" in output
        assert "v = (i32((v + 250)) & 255);" in output


@pytest.mark.parametrize("target", TARGETS)
def test_narrow_increment_used_as_a_value_is_refused(tmp_path, target):
    body = """    uchar v = uchar(a[i]);
    o[i] = int(v++);"""
    with pytest.raises(NarrowIntegerUnsupportedError) as raised:
        translate(
            tmp_path, METAL_KERNEL.format(helpers="", body=body), target, ".metal"
        )
    assert raised.value.code == f"project.translate.{target}-narrow-integer-unsupported"
    assert raised.value.reason_code == "step-value"


@pytest.mark.parametrize("target", TARGETS)
def test_metal_vector_arithmetic_is_not_promoted(tmp_path, target):
    body = """    uchar4 v = uchar4(a[i]);
    v = v + uchar4(250);
    o[i] = int(v.x) + int((v * uchar4(2)).y);"""
    output = translate(
        tmp_path, METAL_KERNEL.format(helpers="", body=body), target, ".metal"
    )
    if target == "wgsl":
        assert "vec4<i32>(255)" in wgsl_line(output, "var v")
        assert "vec4<i32>(255)" in wgsl_line(output, "o[i]")


@pytest.mark.parametrize("target", TARGETS)
def test_parameters_and_returns_are_normalized(tmp_path, target):
    helpers = "uchar bump(uchar x) { return x + 1; }"
    body = """    uchar v = bump(uchar(a[i]));
    o[i] = int(bump(v)) + int(bump(300));"""
    output = translate(
        tmp_path, METAL_KERNEL.format(helpers=helpers, body=body), target, ".metal"
    )
    if target == "wgsl":
        assert "fn bump(x: i32) -> i32" in output
        assert "return (i32((x + 1)) & 255);" in output
        assert "bump(44)" in output


@pytest.mark.parametrize("target", TARGETS)
def test_workgroup_arrays_hold_normalized_words(tmp_path, target):
    body = """    threadgroup ushort tmp[64];
    tmp[l] = ushort(a[i] * 3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = int(tmp[l]);"""
    output = translate(
        tmp_path, METAL_KERNEL.format(helpers="", body=body), target, ".metal"
    )
    if target == "wgsl":
        assert "array<i32, 64>" in output
        assert "& 65535" in wgsl_line(output, "tmp[")


@pytest.mark.parametrize("target", TARGETS)
def test_thread_references_address_widened_registers(tmp_path, target):
    helpers = "void bump(thread uchar& x) { x = x + 1; }"
    body = """    uchar v = uchar(a[i]);
    bump(v);
    o[i] = int(v);"""
    if target == "wgsl":
        pytest.skip("the WGSL emitter has no reference parameters")
    output = translate(
        tmp_path, METAL_KERNEL.format(helpers=helpers, body=body), target, ".metal"
    )
    assert "OpTypeInt 8" not in output
