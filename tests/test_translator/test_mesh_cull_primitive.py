"""A mesh primitive's cull flag on every target.

CrossGL spells the flag ``gl_CullPrimitiveEXT``, the SPIR-V importer's name for ``BuiltIn CullPrimitiveEXT``; the HLSL
and Slang importers now read ``SV_CullPrimitive`` as it, and the emitters accept either spelling. GLSL output wrote a
``SV_CullPrimitive`` member as a user ``perprimitiveEXT out bool`` (glslang: "'out' : cannot be bool"), SPIR-V and
Metal output refused it, and DirectX and Slang output refused ``gl_CullPrimitiveEXT``. The GLSL-style mesh stage
refused ``gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT`` for HLSL and Metal. They now write
``gl_MeshPrimitivesEXT[i].gl_CullPrimitiveEXT`` (GL_EXT_mesh_shader), ``BuiltIn CullPrimitiveEXT`` with
``PerPrimitiveEXT`` (SPV_EXT_mesh_shader), ``SV_CullPrimitive`` and a ``[[primitive_culled]]`` member of the Metal
primitive structure.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

ENTRY_FORM = """shader Probe {{
    struct Vertex {{
        vec4 position @ gl_Position;
    }}
    struct Primitive {{
        bool culled @ {semantic};
    }}
    mesh {{
        void main(@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], @ primitives out Primitive prims[1])
            @numthreads(1, 1, 1)
            @outputtopology(triangle) {{
            SetMeshOutputCounts(3, 1);
            verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);
            verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);
            tris[0] = uvec3(0, 1, 2);
            prims[0].culled = true;
        }}
    }}
}}
"""

HLSL = """struct V { float4 pos : SV_Position; };
struct P { bool culled : SV_CullPrimitive; };
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices V verts[3], out indices uint3 tris[1], out primitives P prims[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0].pos = float4(0, 0, 0, 1);
    verts[1].pos = float4(1, 0, 0, 1);
    verts[2].pos = float4(0, 1, 0, 1);
    tris[0] = uint3(0, 1, 2);
    prims[0].culled = true;
}
"""

GLSL = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[2].gl_Position = vec4(0.0, 1.0, 0.0, 1.0);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
    gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = true;
}
"""

SOURCES = {
    "crossgl-sv": (ENTRY_FORM.format(semantic="SV_CullPrimitive"), ".cgl"),
    "crossgl-gl": (ENTRY_FORM.format(semantic="gl_CullPrimitiveEXT"), ".cgl"),
    "hlsl": (HLSL, ".hlsl"),
}


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True)


def test_hlsl_import_spells_the_cull_flag_as_the_crossgl_builtin(tmp_path):
    crossgl = _translate(tmp_path, HLSL, ".hlsl", "cgl")
    assert re.search(r"bool culled @ ?gl_CullPrimitiveEXT\b", crossgl), crossgl


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_glsl_writes_the_builtin_member(tmp_path, source):
    glsl = _translate(tmp_path, *SOURCES[source], "opengl")
    assert "gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = true;" in glsl
    assert not re.search(r"\bout\s+bool\b", glsl), glsl
    validator = shutil.which("glslangValidator")
    if validator is not None:
        path = tmp_path / "out.mesh"
        path.write_text(glsl)
        result = _run(
            [validator, "-V", "--target-env", "vulkan1.2", "-S", "mesh", str(path)]
            + ["-o", str(tmp_path / "out.spv")]
        )
        assert result.returncode == 0, result.stdout + glsl


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_spirv_decorates_a_per_primitive_cull_builtin(tmp_path, source):
    spirv = _translate(tmp_path, *SOURCES[source], "vulkan")
    variable = re.search(r"OpDecorate (%\w+) BuiltIn CullPrimitiveEXT", spirv)
    assert variable is not None, spirv
    assert f"OpDecorate {variable.group(1)} PerPrimitiveEXT" in spirv
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler and validator:
        text, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        text.write_text(spirv)
        assembled = _run(
            [assembler, "--target-env", "vulkan1.2", str(text), "-o", str(binary)]
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = _run([validator, "--target-env", "vulkan1.2", str(binary)])
        assert validated.returncode == 0, validated.stderr + spirv


@pytest.mark.parametrize("source", sorted(SOURCES) + ["glsl"])
def test_metal_declares_a_primitive_culled_member(tmp_path, source):
    text, suffix = SOURCES.get(source, (GLSL, ".mesh"))
    metal = _translate(tmp_path, text, suffix, "metal")
    assert re.search(r"bool \w+\s*\[\[primitive_culled\]\];", metal), metal
    if (
        shutil.which("xcrun")
        and _run(["xcrun", "-sdk", "macosx", "metal", "--version"]).returncode == 0
    ):
        path = tmp_path / "out.metal"
        path.write_text(metal)
        result = _run(
            ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(path)]
            + ["-o", str(tmp_path / "out.air")]
        )
        assert result.returncode == 0, result.stderr + metal


@pytest.mark.parametrize("source", sorted(SOURCES) + ["glsl"])
@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_family_writes_sv_cull_primitive(tmp_path, source, backend):
    text, suffix = SOURCES.get(source, (GLSL, ".mesh"))
    code = _translate(tmp_path, text, suffix, backend)
    assert re.search(r"bool \w+\s*:\s*SV_CullPrimitive\s*;", code), code
