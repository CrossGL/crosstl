"""Array extents that name integer constants are folded to their values for every target."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.constant_extents import fold_constant_extents
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

# A header shared with the host declares the constants after a structure sized by them.
METAL = """#include <metal_stdlib>
using namespace metal;
struct Item { float4x4 transform; };
struct Frame { Item items[Total]; float weights[Count * 2]; };
static constant constexpr size_t Width = 4;
static constant constexpr size_t Height = 3;
static constant constexpr size_t Total = Width * Height;
enum Sizes { Count = 2 };
vertex float4 v(constant Frame& frame [[buffer(0)]], uint vid [[vertex_id]]) {
    return frame.items[vid].transform[0] * frame.weights[vid & 3u];
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "shader.metal"
    path.write_text(METAL)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "metal"])
def test_structure_extents_named_by_later_constants(tmp_path, backend):
    output = translate(tmp_path, backend)
    if backend == "vulkan":
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            (tmp_path / "out.spvasm").write_text(output)
            _run(
                ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "o.spv"],
                tmp_path,
            )
            _run(["spirv-val", "--target-env", "vulkan1.1", "o.spv"], tmp_path)
        return
    assert "items[12]" in output and "weights[4]" in output
    assert "items[Total]" not in output
    if backend == "opengl" and shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(output)
        _run(["glslangValidator", "out.vert"], tmp_path)


def _program(source):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(source)


def _extents(program):
    return [
        (
            member.member_type.size.value
            if isinstance(member.member_type.size, A.LiteralNode)
            else member.member_type.size.name
        )
        for struct in program.structs
        for member in struct.members
        if isinstance(member.member_type, A.ArrayType)
    ]


def test_shadowable_extents_are_kept():
    program = _program(
        "shader s {\n"
        "    const int N = 4;\n"
        "    const int M = 2;\n"
        "    struct S { float shadowed[N]; float own[M]; };\n"
        "    compute {\n"
        "        void main() {\n"
        "            const int N = 8;\n"
        "            S s;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    # N is also a local of main, so an extent that names it could mean either.
    assert fold_constant_extents(program, "directx") == 1
    assert _extents(program) == ["N", 2]


def test_crossgl_output_keeps_the_names():
    program = _program(
        "shader s { const int M = 2; struct S { float a[M]; }; compute { void main() { S s; } } }"
    )
    assert fold_constant_extents(program, "cgl") == 0
    assert _extents(program) == ["M"]


SPIRV_CONSTANTS = """#include <metal_stdlib>
using namespace metal;
constant uint Width = 4;
constant uint Total = Width * 3;
constant float Scale = 0.5;
constant float Half = Scale * Scale;
constant float table[3] = {1.0, 2.0, 3.0};
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    o[i] = float(Total) * Half + table[i % 3];
}
"""


def test_immutable_globals_are_spirv_constants(tmp_path):
    # An initializer that names an earlier constant used to load it at module scope, which spirv-val rejects.
    path = tmp_path / "kernel.metal"
    path.write_text(SPIRV_CONSTANTS)
    output = crosstl.translate(str(path), backend="vulkan", format_output=False)

    def named(name):
        (identifier,) = re.findall(rf'OpName (%\w+) "{name}"', output)
        return re.escape(identifier)

    assert re.search(rf"{named('Total')} = OpConstant %\w+ 12\n", output)
    assert re.search(rf"{named('Half')} = OpConstant %\w+ 0\.25\n", output)
    # An array stays a variable, which a dynamic index points into.
    assert re.search(rf"{named('table')} = OpVariable ", output)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(output)
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "o.spv"],
            tmp_path,
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", "o.spv"], tmp_path)
