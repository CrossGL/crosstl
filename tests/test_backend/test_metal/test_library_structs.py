"""Metal standard library structures and implicit texture-write conversions on import."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.library_structs import (
    metal_library_struct_source,
    referenced_library_struct_names,
)
from crosstl.backend.Metal.MetalCrossGLCodeGen import parse_metal_structs
from crosstl.translator.codegen.metal_library_structs import (
    METAL_LIBRARY_STRUCTS,
    is_metal_library_struct,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TESSELLATION_FACTOR_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void fill(device MTLQuadTessellationFactorsHalf* quads [[buffer(0)]],
                 device MTLTriangleTessellationFactorsHalf* triangles [[buffer(1)]],
                 uint id [[thread_position_in_grid]]) {
    float factor = 2.0;
    quads[id].edgeTessellationFactor[0] = factor;
    quads[id].insideTessellationFactor[1] = factor * 0.5f;
    triangles[id].edgeTessellationFactor[2] = factor;
    triangles[id].insideTessellationFactor = factor;
}
"""


def _translate(tmp_path, source, backend, stem="kernel"):
    path = tmp_path / f"{stem}.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _assert_metal_compiles(tmp_path, metal, stem):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    source = tmp_path / f"{stem}.out.metal"
    source.write_text(metal, encoding="utf-8")
    result = _run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / f"{stem}.air"),
        ]
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def _assert_glsl_compute_compiles(tmp_path, glsl, stem):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    source = tmp_path / f"{stem}.comp"
    source.write_text(glsl, encoding="utf-8")
    result = _run([glslang, "--target-env", "opengl", str(source)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_library_struct_parses_and_matches_its_crossgl_declaration():
    for name, members in METAL_LIBRARY_STRUCTS.items():
        (struct,) = parse_metal_structs(metal_library_struct_source(name))
        assert struct.name == name
        assert [member.name for member in struct.members] == [
            member_name for _type, member_name, _length in members
        ]


def test_referenced_library_structs_are_declared_once_in_crossgl(tmp_path):
    crossgl = _translate(tmp_path, TESSELLATION_FACTOR_KERNEL, "cgl")

    assert crossgl.count("struct MTLQuadTessellationFactorsHalf {") == 1
    assert crossgl.count("struct MTLTriangleTessellationFactorsHalf {") == 1
    assert "float16[4] edgeTessellationFactor;" in crossgl
    assert "float16[2] insideTessellationFactor;" in crossgl
    assert "float16 insideTessellationFactor;" in crossgl

    ast = Parser(Lexer(crossgl).get_tokens()).parse()
    structs = {struct.name: struct for struct in ast.structs}
    assert is_metal_library_struct(structs["MTLQuadTessellationFactorsHalf"])
    assert is_metal_library_struct(structs["MTLTriangleTessellationFactorsHalf"])


def test_user_declared_struct_with_a_library_name_is_not_redeclared():
    source = """
    struct MTLDrawPrimitivesIndirectArguments { uint vertexCount; uint instanceCount; };
    kernel void k(device MTLDrawPrimitivesIndirectArguments* args [[buffer(0)]]) {}
    """
    from crosstl.backend.Metal import MetalLexer, MetalParser

    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    assert referenced_library_struct_names(ast) == []


def test_crossgl_struct_differing_from_the_library_layout_is_not_the_library_struct():
    crossgl = """
    shader main {
        struct MTLDispatchThreadgroupsIndirectArguments {
            uint[2] threadgroupsPerGrid;
        }
    }
    """
    ast = Parser(Lexer(crossgl).get_tokens()).parse()
    assert not is_metal_library_struct(ast.structs[0])


def test_library_structs_round_trip_to_metal_without_redeclaration(tmp_path):
    metal = _translate(tmp_path, TESSELLATION_FACTOR_KERNEL, "metal")

    assert "struct MTLQuadTessellationFactorsHalf" not in metal
    assert "struct MTLTriangleTessellationFactorsHalf" not in metal
    assert "device MTLQuadTessellationFactorsHalf" in metal
    _assert_metal_compiles(tmp_path, metal, "tessellation_factors")


@pytest.mark.parametrize("backend", ["opengl", "directx", "vulkan"])
def test_library_structs_translate_to_other_targets(tmp_path, backend):
    generated = _translate(tmp_path, TESSELLATION_FACTOR_KERNEL, backend)

    assert "MTLQuadTessellationFactorsHalf" in generated
    if backend == "opengl":
        _assert_glsl_compute_compiles(tmp_path, generated, "tessellation_factors")


TEXTURE_WRITE_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void clear(texture2d<float, access::write> colour [[texture(0)]],
                  texture2d<half, access::write> half_colour [[texture(1)]],
                  texture2d<uint, access::write> ids [[texture(2)]],
                  uint2 tid [[thread_position_in_grid]]) {
    colour.write(0, tid);
    colour.write(float4(1.0), tid);
    half_colour.write(0.5h, tid);
    ids.write(3u, tid);
}
"""


def test_scalar_texture_write_values_convert_to_the_texel_type(tmp_path):
    crossgl = _translate(tmp_path, TEXTURE_WRITE_KERNEL, "cgl")

    assert "imageStore(colour, tid, vec4(0));" in crossgl
    assert "imageStore(colour, tid, vec4(1.0));" in crossgl
    assert "imageStore(half_colour, tid, f16vec4(float16(0.5)));" in crossgl
    assert "imageStore(ids, tid, uvec4(3u));" in crossgl


def test_scalar_texture_write_values_compile_on_metal_and_opengl(tmp_path):
    metal = _translate(tmp_path, TEXTURE_WRITE_KERNEL, "metal")
    assert "colour.write(float4(0), uint2(tid));" in metal
    _assert_metal_compiles(tmp_path, metal, "texture_writes")

    glsl = _translate(tmp_path, TEXTURE_WRITE_KERNEL, "opengl")
    assert "imageStore(colour, ivec2(tid), vec4(0));" in glsl
    _assert_glsl_compute_compiles(tmp_path, glsl, "texture_writes")
