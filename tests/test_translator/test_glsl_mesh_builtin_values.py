"""GLSL mesh stages write ``uint`` values to the ``int`` per-primitive built-ins through ``int(...)``.

GL_EXT_mesh_shader declares ``gl_PrimitiveID``, ``gl_Layer`` and ``gl_ViewportIndex`` of ``gl_MeshPrimitivesEXT``
as ``int``; HLSL and MSL declare the members that carry them ``uint``. GLSL has no implicit ``uint`` to ``int``
conversion, so glslang rejected ``gl_MeshPrimitivesEXT[0].gl_Layer = uint(1);`` with "cannot convert from
'const uint' to 'perprimitiveNV out int Layer'".
"""

import shutil
import subprocess

import pytest

import crosstl

METAL = """#include <metal_stdlib>
using namespace metal;

struct Vertex {{
    float4 position [[position]];
}};

struct Primitive {{
    uint value [[{attribute}]];
}};

[[mesh]] void probe(mesh<Vertex, Primitive, 3, 1, topology::triangle> output) {{
    output.set_primitive_count(1);
    output.set_vertex(0, Vertex{{float4(0.0, 0.0, 0.0, 1.0)}});
    output.set_vertex(1, Vertex{{float4(1.0, 0.0, 0.0, 1.0)}});
    output.set_vertex(2, Vertex{{float4(0.0, 1.0, 0.0, 1.0)}});
    output.set_index(0, 0);
    output.set_index(1, 1);
    output.set_index(2, 2);
    output.set_primitive(0, Primitive{{1}});
}}
"""

HLSL = """struct Vertex {{
    float4 position : SV_Position;
}};

struct Primitive {{
    uint value : {semantic};
}};

[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main(uint index : SV_GroupIndex, out vertices Vertex verts[3],
          out indices uint3 tris[1], out primitives Primitive prims[1]) {{
    SetMeshOutputCounts(3, 1);
    verts[0].position = float4(0.0, 0.0, 0.0, 1.0);
    verts[1].position = float4(1.0, 0.0, 0.0, 1.0);
    verts[2].position = float4(0.0, 1.0, 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
    {write}
}}
"""

CROSSGL_INT_MEMBER = """shader Probe {
    struct Vertex {
        vec4 position @gl_Position;
    }
    struct Primitive {
        int layer @gl_Layer;
    }
    mesh {
        @numthreads(1, 1, 1)
        void main(out Vertex verts[3] @vertices, out Primitive prims[1] @primitives,
                  out uvec3 tris[1] @indices, uint index @gl_LocalInvocationIndex) @outputtopology(triangle) {
            SetMeshOutputCounts(3u, 1u);
            verts[0] = Vertex(vec4(0.0, 0.0, 0.0, 1.0));
            verts[1] = Vertex(vec4(1.0, 0.0, 0.0, 1.0));
            verts[2] = Vertex(vec4(0.0, 1.0, 0.0, 1.0));
            tris[0] = uvec3(0u, 1u, 2u);
            prims[0].layer = int(index);
        }
    }
}
"""

BUILTINS = {
    "gl_PrimitiveID": ("primitive_id", "SV_PrimitiveID"),
    "gl_Layer": ("render_target_array_index", "SV_RenderTargetArrayIndex"),
    "gl_ViewportIndex": ("viewport_array_index", "SV_ViewportArrayIndex"),
}


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.mesh"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "--target-env", "vulkan1.3", "-S", "mesh", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("builtin", sorted(BUILTINS))
def test_metal_primitive_constructor_writes_int(tmp_path, builtin):
    attribute, _ = BUILTINS[builtin]
    code = _translate(tmp_path, "probe.metal", METAL.format(attribute=attribute))

    assert f"gl_MeshPrimitivesEXT[0].{builtin} = int(uint(1));" in code
    _glslang(tmp_path, code)


@pytest.mark.parametrize("builtin", sorted(BUILTINS))
def test_hlsl_member_write_of_a_uint_value_writes_int(tmp_path, builtin):
    _, semantic = BUILTINS[builtin]
    source = HLSL.format(semantic=semantic, write="prims[0].value = index + 1u;")
    code = _translate(tmp_path, "probe.hlsl", source)

    assert f"gl_MeshPrimitivesEXT[0].{builtin} = int(" in code
    _glslang(tmp_path, code)


def test_hlsl_whole_primitive_write_of_a_struct_value_writes_int(tmp_path):
    write = (
        "Primitive primitive;\n    primitive.value = index;\n    prims[0] = primitive;"
    )
    source = HLSL.format(semantic="SV_RenderTargetArrayIndex", write=write)
    code = _translate(tmp_path, "probe.hlsl", source)

    assert "gl_MeshPrimitivesEXT[0].gl_Layer = int(primitive.value);" in code
    _glslang(tmp_path, code)


def test_integer_literal_and_int_member_are_written_unchanged(tmp_path):
    source = HLSL.format(
        semantic="SV_RenderTargetArrayIndex", write="prims[0].value = 1;"
    )
    code = _translate(tmp_path, "probe.hlsl", source)
    assert "gl_MeshPrimitivesEXT[0].gl_Layer = 1;" in code

    code = _translate(tmp_path, "probe.cgl", CROSSGL_INT_MEMBER)
    assert "gl_MeshPrimitivesEXT[0].gl_Layer = int(gl_LocalInvocationIndex);" in code
    assert "int(int(" not in code
    _glslang(tmp_path, code)
