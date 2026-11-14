"""GLSL GL_EXT_mesh_shader built-in output arrays lowered to SPIR-V mesh outputs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

FIXTURES = Path(__file__).parent / "spirv_mesh_builtins"


def generate(source_code):
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source_code).tokens).parse())


def assert_validates(spv_code, tmp_path):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    assemble = subprocess.run(
        [spirv_as, "--target-env", "vulkan1.3", str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    validate = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.3", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def mesh_shader(body, topology="triangles", max_primitives=1):
    return f"""
    shader Mesh {{
        mesh {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout({topology}, max_vertices = 3, max_primitives = {max_primitives}) out;
            void main() {{
                SetMeshOutputCounts(3, {max_primitives});
                {body}
            }}
        }}
    }}
    """


def builtin_variable(spv_code, builtin):
    match = re.search(rf"OpDecorate (%\w+) BuiltIn {builtin}\b", spv_code)
    return match.group(1) if match else None


def test_mesh_vertex_positions_and_triangle_indices(tmp_path):
    spv_code = generate(mesh_shader("""
                gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, -1.0, 0.0, 1.0);
                gl_MeshVerticesEXT[1].gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
                gl_MeshVerticesEXT[2].gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
                gl_MeshVerticesEXT[2].gl_PointSize = 2.0;
                gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex] = uvec3(0, 1, 2);
    """))

    assert "OpEntryPoint MeshEXT" in spv_code
    assert re.search(r"OpMemberDecorate %\w+ 0 BuiltIn Position", spv_code)
    assert builtin_variable(spv_code, "PointSize")
    assert builtin_variable(spv_code, "PrimitiveTriangleIndicesEXT")
    assert spv_code.count("OpStore") >= 5
    assert_validates(spv_code, tmp_path)


def test_mesh_per_primitive_builtins_are_per_primitive_outputs(tmp_path):
    spv_code = generate(mesh_shader("""
                gl_MeshVerticesEXT[0].gl_Position = vec4(0.0);
                gl_MeshVerticesEXT[1].gl_Position = vec4(0.0);
                gl_MeshVerticesEXT[2].gl_Position = vec4(0.0);
                gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
                gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 7;
                gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = false;
    """))

    primitive_id = builtin_variable(spv_code, "PrimitiveId")
    cull = builtin_variable(spv_code, "CullPrimitiveEXT")
    assert primitive_id and cull
    assert f"OpDecorate {primitive_id} PerPrimitiveEXT" in spv_code
    assert f"OpDecorate {cull} PerPrimitiveEXT" in spv_code
    assert_validates(spv_code, tmp_path)


def test_mesh_line_indices_follow_the_declared_topology(tmp_path):
    spv_code = generate(
        mesh_shader(
            """
                gl_MeshVerticesEXT[0].gl_Position = vec4(0.0);
                gl_MeshVerticesEXT[1].gl_Position = vec4(1.0);
                gl_MeshVerticesEXT[2].gl_Position = vec4(1.0);
                gl_PrimitiveLineIndicesEXT[0] = uvec2(0, 1);
        """,
            topology="lines",
        )
    )

    assert builtin_variable(spv_code, "PrimitiveLineIndicesEXT")
    assert "PrimitiveTriangleIndicesEXT" not in spv_code
    assert_validates(spv_code, tmp_path)


def test_mesh_builtin_members_without_a_lowering_are_refused():
    with pytest.raises(SpirvLoweringRefused, match="gl_ClipDistance"):
        generate(mesh_shader("""
                gl_MeshVerticesEXT[0].gl_ClipDistance = 1.0;
        """))


def test_mesh_builtin_literal_index_beyond_max_vertices_is_refused():
    with pytest.raises(SpirvLoweringRefused, match="exceeds"):
        generate(mesh_shader("""
                gl_MeshVerticesEXT[3].gl_Position = vec4(0.0);
        """))


def test_spirv_mesh_import_round_trips_to_spirv(tmp_path):
    # glslang -V --target-env vulkan1.3 output for a GLSL triangle mesh shader that
    # writes gl_MeshVerticesEXT, gl_MeshPrimitivesEXT and gl_PrimitiveTriangleIndicesEXT.
    spv_code = crosstl.translate(
        str(FIXTURES / "triangle.mesh.spvasm"), backend="vulkan"
    )

    assert re.search(r"OpMemberDecorate %\w+ 0 BuiltIn Position", spv_code)
    assert builtin_variable(spv_code, "PrimitiveId")
    assert spv_code.count("BuiltIn PrimitiveTriangleIndicesEXT") == 1
    # The imported index-array declaration is the built-in, not a Location output.
    assert not re.search(r"OpDecorate %\w+ Location", spv_code)
    assert_validates(spv_code, tmp_path)
