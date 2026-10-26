"""Mesh shaders whose SetMeshOutputCounts calls the DirectX emitter refuses are refused with a typed code.

The emitter refuses a mesh stage that does not call SetMeshOutputCounts exactly once, or calls it from a loop or
from thread-varying control flow (``if (gl_LocalInvocationIndex == 0) SetMeshOutputsEXT(...)``). Those refusals
were bare ``ValueError``s, which the project pipeline reports as translator errors rather than typed refusals.
"""

import pytest

import crosstl
from crosstl.translator.codegen.directx_codegen import (
    DirectXMeshOutputCountsError,
)

MESH = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 4) in;
layout(triangles, max_vertices = 4, max_primitives = 2) out;
void main() {
    if (gl_LocalInvocationIndex == 0u) {
        SetMeshOutputsEXT(4, 2);
    }
    gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    if (gl_LocalInvocationIndex < 2u) {
        gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = uvec3(0u, 1u, 2u);
    }
}
"""


@pytest.mark.parametrize(
    "call, words",
    [
        (
            "if (gl_LocalInvocationIndex == 0u) {\n        SetMeshOutputsEXT(4, 2);\n    }",
            "thread-varying",
        ),
        (
            "for (uint k = 0u; k < 1u; ++k) {\n        SetMeshOutputsEXT(4, 2);\n    }",
            "loop",
        ),
        ("SetMeshOutputsEXT(4, 2);\n    SetMeshOutputsEXT(4, 2);", "exactly once"),
    ],
)
def test_refused_output_counts_are_a_typed_refusal(tmp_path, call, words):
    source = MESH.replace(
        "if (gl_LocalInvocationIndex == 0u) {\n        SetMeshOutputsEXT(4, 2);\n    }",
        call,
    )
    path = tmp_path / "shader.mesh"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(DirectXMeshOutputCountsError, match=words) as raised:
        crosstl.translate(str(path), backend="directx", format_output=False)
    assert raised.value.project_diagnostic_code == (
        "project.translate.directx-mesh-output-counts-unsupported"
    )
