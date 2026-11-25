"""Metal input of a mesh primitive's ``[[primitive_culled]]`` member.

The Metal importer wrote the member as ``@primitive_culled``, a name no other CrossGL producer uses, so GLSL output
declared a user ``perprimitiveEXT out bool`` varying that glslang rejects ("'out' : cannot be bool"). It now writes
``@gl_CullPrimitiveEXT``, the CrossGL spelling the HLSL and SPIR-V importers use, which every generator writes as its
own cull flag: ``gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT``, ``BuiltIn CullPrimitiveEXT``, ``SV_CullPrimitive``
and ``[[primitive_culled]]``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

MESH = """#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float4 position [[position]];
};

struct Primitive {
    bool culled [[primitive_culled]];
};

[[mesh]] void probe(mesh<Vertex, Primitive, 3, 1, topology::triangle> output) {
    output.set_primitive_count(1);
    output.set_vertex(0, Vertex{float4(0.0, 0.0, 0.0, 1.0)});
    output.set_vertex(1, Vertex{float4(1.0, 0.0, 0.0, 1.0)});
    output.set_vertex(2, Vertex{float4(0.0, 1.0, 0.0, 1.0)});
    output.set_index(0, 0);
    output.set_index(1, 1);
    output.set_index(2, 2);
    output.set_primitive(0, Primitive{true});
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "probe.metal"
    path.write_text(MESH)
    return crosstl.translate(str(path), backend=backend)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True)


def _metal_compiler():
    return (
        shutil.which("xcrun") is not None
        and _run(["xcrun", "-sdk", "macosx", "metal", "--version"]).returncode == 0
    )


def test_the_cull_member_imports_as_the_crossgl_builtin(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert re.search(r"bool culled @ ?gl_CullPrimitiveEXT;", crossgl), crossgl
    assert "@primitive_culled" not in crossgl


def test_glsl_output_writes_the_builtin_cull_flag(tmp_path):
    glsl = _translate(tmp_path, "opengl")
    assert "gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = true;" in glsl, glsl
    assert not re.search(r"\bout\s+bool\b", glsl), glsl
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "out.mesh"
    path.write_text(glsl)
    result = _run(
        [validator, "-V", "--target-env", "vulkan1.2", "-S", "mesh", str(path)]
        + ["-o", str(tmp_path / "out.spv")]
    )
    assert result.returncode == 0, result.stdout + glsl


def test_spirv_output_decorates_a_per_primitive_cull_builtin(tmp_path):
    spirv = _translate(tmp_path, "vulkan")
    variable = re.search(r"OpDecorate (%\w+) BuiltIn CullPrimitiveEXT", spirv)
    assert variable is not None, spirv
    assert f"OpDecorate {variable.group(1)} PerPrimitiveEXT" in spirv
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("spirv-as and spirv-val are not installed")
    text, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    text.write_text(spirv)
    assembled = _run(
        [assembler, "--target-env", "vulkan1.2", str(text), "-o", str(binary)]
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = _run([validator, "--target-env", "vulkan1.2", str(binary)])
    assert validated.returncode == 0, validated.stderr + spirv


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_family_output_writes_sv_cull_primitive(tmp_path, backend):
    code = _translate(tmp_path, backend)
    assert re.search(r"bool culled\s*:\s*SV_CullPrimitive\s*;", code), code


def test_metal_round_trip_keeps_the_primitive_culled_member(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert re.search(r"bool culled\s*\[\[primitive_culled\]\];", metal), metal
    if not _metal_compiler():
        pytest.skip("the Metal compiler is not installed")
    path = tmp_path / "out.metal"
    path.write_text(metal)
    result = _run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(path)]
        + ["-o", str(tmp_path / "out.air")]
    )
    assert result.returncode == 0, result.stderr + metal
