"""GLSL mesh-stage built-in outputs in generated SPIR-V (GL_EXT_mesh_shader section 7.1.6).

A GLSL mesh shader writes ``gl_MeshVerticesEXT[i].gl_Position``, ``gl_MeshPrimitivesEXT[i].gl_PrimitiveID`` and
``gl_PrimitiveTriangleIndicesEXT[i]`` without declaring them. The member stores were refused as not naming
storage, and a SPIR-V import's declared ``gl_PrimitiveTriangleIndicesEXT`` array became a user output at a
location instead of the ``PrimitiveTriangleIndicesEXT`` built-in. ``perprimitiveEXT`` outputs keep their
``PerPrimitiveEXT`` decoration through GLSL and SPIR-V import, and ``gl_MeshPrimitivesEXT[i].gl_PrimitiveShadingRateEXT``
is the ``PrimitiveShadingRateKHR`` built-in.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

MESH = """#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_fragment_shading_rate : require
layout(local_size_x = 3) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
layout(location = 0) out vec4 outColor[];
layout(location = 1) perprimitiveEXT out vec4 primColor[];
void main() {
    uint i = gl_LocalInvocationIndex;
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[i].gl_Position = vec4(float(i), 1.0, 0.0, 1.0);
    gl_MeshVerticesEXT[i].gl_PointSize = 2.0;
    outColor[i] = vec4(1.0);
    if (i == 0) {
        gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
        gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 7;
        gl_MeshPrimitivesEXT[0].gl_Layer = 1;
        gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = false;
        gl_MeshPrimitivesEXT[0].gl_PrimitiveShadingRateEXT = gl_ShadingRateFlag2VerticalPixelsEXT;
        primColor[0] = vec4(0.5);
    }
}
"""

LINE_INDICES_IN_TRIANGLE_MESH = """#version 450
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    SetMeshOutputsEXT(3, 1);
    gl_PrimitiveLineIndicesEXT[0] = uvec2(0, 1);
}
"""


def _tools(*names):
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not on PATH")


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _validate(tmp_path, assembly):
    _tools("spirv-as", "spirv-val")
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    _run(["spirv-as", "--target-env", "vulkan1.3", str(source), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.3", str(binary)])


def _translate(path):
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _decorated(assembly, decoration):
    return set(re.findall(rf"OpDecorate %(\w+) {decoration}\b", assembly))


def _check_mesh_outputs(assembly):
    for builtin in (
        "PointSize",
        "PrimitiveTriangleIndicesEXT",
        "PrimitiveId",
        "Layer",
        "CullPrimitiveEXT",
        "PrimitiveShadingRateKHR",
    ):
        assert f"BuiltIn {builtin}" in assembly, builtin
    assert re.search(r"OpMemberDecorate %\w+ 0 BuiltIn Position", assembly)
    located = dict(re.findall(r"OpDecorate %(\w+) Location (\d+)", assembly))
    assert sorted(located.values()) == ["0", "1"], located
    primitive_color = next(name for name, value in located.items() if value == "1")
    assert primitive_color in _decorated(assembly, "PerPrimitiveEXT")
    indices = _decorated(assembly, "BuiltIn PrimitiveTriangleIndicesEXT")
    assert not indices & set(located)
    assert "OpCapability FragmentShadingRateKHR" in assembly


def test_glsl_mesh_builtin_outputs_lower_to_spirv_builtins(tmp_path):
    source = tmp_path / "shader.mesh"
    source.write_text(MESH, encoding="utf-8")
    assembly = _translate(source)
    _check_mesh_outputs(assembly)
    _validate(tmp_path, assembly)


def test_spirv_mesh_import_keeps_builtin_indices_and_per_primitive_outputs(tmp_path):
    _tools("glslangValidator", "spirv-dis")
    source = tmp_path / "shader.mesh"
    source.write_text(MESH, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    disassembly = tmp_path / "shader.mesh.spvasm"
    _run(
        [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.3",
            str(source),
            "-o",
            str(binary),
        ]
    )
    _run(["spirv-dis", str(binary), "-o", str(disassembly)])

    crossgl = crosstl.translate(str(disassembly), backend="cgl", format_output=False)
    assert re.search(r"primColor\[1\][^;]*@perprimitiveEXT", crossgl), crossgl

    assembly = _translate(disassembly)
    _check_mesh_outputs(assembly)
    _validate(tmp_path, assembly)


def test_index_array_of_another_topology_is_refused(tmp_path):
    source = tmp_path / "shader.mesh"
    source.write_text(LINE_INDICES_IN_TRIANGLE_MESH, encoding="utf-8")
    # A triangle mesh stage has no line index array to write.
    with pytest.raises(SpirvLoweringRefused, match="gl_PrimitiveLineIndicesEXT"):
        _translate(source)
