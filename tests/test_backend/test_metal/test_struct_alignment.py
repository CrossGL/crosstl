"""A Metal struct's ``alignas`` imports as ``@metal_alignas(N)`` and keeps its MSL layout on every target."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalImportRefused
from crosstl.translator.codegen.generic_struct_utils import (
    GenericStructAttributeError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
)

KERNEL = """#include <metal_stdlib>
using namespace metal;
struct {alignas} Rec { float a; float b; };
kernel void k(device Rec* recs [[buffer(0)]], device float* out [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    out[id] = recs[id].a + recs[id].b + float(sizeof(Rec));
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def kernel(alignas):
    return KERNEL.replace("{alignas}", alignas)


def run_tools(tmp_path, commands):
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize(
    "alignas",
    [
        "alignas(16)",
        "alignas(sizeof(float4))",
        "alignas(float4)",
        "alignas(8) alignas(16)",
        "alignas(0) alignas(16)",
    ],
)
def test_struct_alignment_imports_as_one_folded_attribute(tmp_path, alignas):
    cgl = translate(tmp_path, kernel(alignas), "cgl")
    assert "@metal_alignas(16)\n    struct Rec {" in cgl
    assert "alignas(16) struct" not in cgl
    # The alignment rounds the 8-byte member list up to a 16-byte object.
    assert "float(16)" in cgl
    assert "@buffer_layout(metal) RWStructuredBuffer<Rec> recs" in cgl


def test_alignas_zero_leaves_the_natural_alignment(tmp_path):
    cgl = translate(tmp_path, kernel("alignas(0)"), "cgl")
    assert "metal_alignas" not in cgl
    assert "float(8)" in cgl
    assert "@buffer_layout" not in cgl


def test_an_alignment_the_members_already_have_keeps_the_default_layout(tmp_path):
    cgl = translate(tmp_path, kernel("alignas(4)"), "cgl")
    assert "@metal_alignas(4)\n    struct Rec {" in cgl
    assert "float(8)" in cgl
    assert "@buffer_layout" not in cgl


def test_metal_redeclares_the_struct_alignment(tmp_path):
    metal = translate(tmp_path, kernel("alignas(16)"), "metal")
    assert "struct alignas(16) Rec {" in metal
    assert "device Rec* recs" in metal


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_other_targets_read_elements_at_the_msl_stride(tmp_path, backend):
    generated = translate(tmp_path, kernel("alignas(16)"), backend)
    # Four 32-bit words per element: the 16-byte MSL stride, not the 8-byte member list.
    assert "(uint(id) * 4u)" in generated
    assert "((uint(id) * 4u) + 1u)" in generated
    if backend == "opengl" and shutil.which("glslangValidator"):
        (tmp_path / "out.comp").write_text(generated)
        run_tools(tmp_path, [["glslangValidator", "out.comp"]])


def test_spirv_reads_elements_at_the_msl_stride(tmp_path):
    assembly = translate(tmp_path, kernel("alignas(16)"), "vulkan")
    assert "OpTypeRuntimeArray" in assembly
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(assembly)
        run_tools(
            tmp_path,
            [
                [
                    "spirv-as",
                    "--target-env",
                    "vulkan1.1",
                    "out.spvasm",
                    "-o",
                    "out.spv",
                ],
                ["spirv-val", "--target-env", "vulkan1.1", "out.spv"],
            ],
        )


def test_an_alignment_that_is_not_a_power_of_two_is_refused(tmp_path):
    with pytest.raises(MetalImportRefused, match="not a constant power of two"):
        translate(
            tmp_path,
            kernel("alignas(3)").replace(" + float(sizeof(Rec))", ""),
            "cgl",
        )


GENERIC = """#include <metal_stdlib>
using namespace metal;
template <typename T> struct alignas(4 * sizeof(T)) V3 { T x; T y; T z; };
kernel void k(device V3<float>* values [[buffer(0)]], device float* out [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    out[id] = values[id].x + values[id].z;
}
"""


def test_a_dependent_alignment_stays_on_the_generic_struct(tmp_path):
    cgl = translate(tmp_path, GENERIC, "cgl")
    assert "@metal_alignas(4 * metal_sizeof(T))\n    generic<T> struct V3 {" in cgl


def test_a_specialization_does_not_drop_a_dependent_alignment(tmp_path):
    with pytest.raises(GenericStructAttributeError, match="V3<float>"):
        translate(tmp_path, GENERIC, "metal")


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_crossgl_keeps_attributes_before_a_generic_struct():
    ast = parse(
        "shader main { @metal_alignas(16) generic<T> struct V { T x; } "
        "compute { void main() {} } }"
    )
    (struct,) = ast.structs
    assert [attribute.name for attribute in struct.attributes] == ["metal_alignas"]
    assert [parameter.name for parameter in struct.generic_params] == ["T"]


CROSSGL = """shader main {
    @metal_alignas(%s)
    struct Rec { float a; float b; }
    compute {
        void main(@buffer_layout(metal) RWStructuredBuffer<Rec> recs @buffer(0),
                  RWStructuredBuffer<float> out_ @buffer(1),
                  uint id @gl_GlobalInvocationID) {
            out_[id] = recs[id].b;
        }
    }
}
"""


def translate_crossgl(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["opengl", "directx", "vulkan"])
def test_msl_buffer_layout_applies_a_crossgl_struct_alignment(tmp_path, backend):
    generated = translate_crossgl(tmp_path, CROSSGL % "32", backend)
    if backend != "vulkan":
        assert "((uint(id) * 8u) + 1u)" in generated


@pytest.mark.parametrize("alignment", ["3", "0", "N"])
def test_msl_buffer_layout_refuses_a_malformed_struct_alignment(tmp_path, alignment):
    with pytest.raises(StructuredLayoutUnsupportedError, match="metal_alignas"):
        translate_crossgl(tmp_path, CROSSGL % alignment, "opengl")
