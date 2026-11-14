"""GLSL interface block instances of geometry, tessellation and mesh stages reach SPIR-V as interface variables,
and stage layout counts written as constant expressions size the stage's arrays."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics import (
    InterfaceBlockInstanceUnsupportedError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.constant_extents import fold_stage_layout_counts
from crosstl.translator.semantics.interface_block_instances import (
    declare_interface_block_instances,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from tests.spirv_friendly_names import translate_spirv

HAS_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))
HAS_GLSLANG = bool(shutil.which("glslangValidator"))

MESH = """#version 460
#extension GL_EXT_mesh_shader : require
const uint GroupX = 2;
const uint GroupY = 2;
layout(local_size_x = GroupX, local_size_y = GroupY) in;
layout(triangles, max_vertices = 4 * GroupX * GroupY, max_primitives = 2 * GroupX * GroupY) out;
layout(location = 0) out Block { vec2 uv; flat uint id; } vout[];
layout(location = 2) perprimitiveEXT out PrimitiveBlock { vec4 colour; } pout[];
layout(location = 3) out vec3 tint[];
void main() {
    uint base = gl_LocalInvocationIndex * 4u;
    SetMeshOutputsEXT(4u * GroupX * GroupY, 2u * GroupX * GroupY);
    for (uint i = 0u; i < 4u; i++) {
        gl_MeshVerticesEXT[base + i].gl_Position = vec4(float(i), 0.0, 0.0, 1.0);
        vout[base + i].uv = vec2(float(i));
        vout[base + i].id = i;
        tint[base + i] = vec3(1.0);
    }
    uint prim = gl_LocalInvocationIndex * 2u;
    pout[prim].colour = vec4(1.0);
    pout[prim + 1u].colour = vec4(0.5);
    gl_PrimitiveTriangleIndicesEXT[prim] = uvec3(base, base + 1u, base + 2u);
    gl_PrimitiveTriangleIndicesEXT[prim + 1u] = uvec3(base + 1u, base + 3u, base + 2u);
}
"""

TESC = """#version 450
layout(vertices = 3) out;
layout(location = 0) in vec3 inPosition[];
layout(location = 0) out ControlBlock { vec3 position; } tcs_out[];
void main() {
    tcs_out[gl_InvocationID].position = inPosition[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = vec4(inPosition[gl_InvocationID], 1.0);
    gl_TessLevelOuter[0] = 1.0;
    gl_TessLevelOuter[1] = 1.0;
    gl_TessLevelOuter[2] = 1.0;
    gl_TessLevelInner[0] = 1.0;
}
"""

TESE = """#version 450
layout(triangles, equal_spacing, ccw) in;
layout(location = 0) in ControlBlock { vec3 position; } tes_in[];
layout(location = 0) out EvaluationBlock { vec3 position; flat int kind; } tes_out;
void main() {
    vec3 p = gl_TessCoord.x * tes_in[0].position + gl_TessCoord.y * tes_in[1].position
        + gl_TessCoord.z * tes_in[2].position;
    tes_out.position = p;
    tes_out.kind = 1;
    gl_Position = vec4(p, 1.0);
}
"""

GEOM = """#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in VertexBlock { vec3 colour; } gin[];
layout(location = 0) out GeometryBlock { vec3 colour; } gout;
void main() {
    for (int i = 0; i < 3; i++) {
        gout.colour = gin[i].colour;
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_spirv(tmp_path, spirv):
    if not HAS_SPIRV_TOOLS:
        return
    (tmp_path / "out.spvasm").write_text(spirv)
    _run(
        ["spirv-as", "--target-env", "vulkan1.3", "out.spvasm", "-o", "out.spv"],
        tmp_path,
    )
    _run(["spirv-val", "--target-env", "vulkan1.3", "out.spv"], tmp_path)


def _program(tmp_path, name, source):
    crossgl = crosstl.translate(
        _write(tmp_path, name, source), backend="cgl", format_output=False
    )
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(crossgl)


def _decorations(spirv, name):
    return sorted(
        line.split(f"%{name} ", 1)[1].strip()
        for line in spirv.splitlines()
        if line.strip().startswith(f"OpDecorate %{name} ")
    )


@pytest.mark.parametrize(
    "name, source, instances",
    [
        (
            "shader.mesh",
            MESH,
            {"vout": "%_arr_Block_", "pout": "%_arr_PrimitiveBlock_"},
        ),
        ("shader.tesc", TESC, {"tcs_out": "%_arr_ControlBlock_"}),
        (
            "shader.tese",
            TESE,
            {"tes_in": "%_arr_ControlBlock_", "tes_out": "%EvaluationBlock"},
        ),
        ("shader.geom", GEOM, {"gin": "%_arr_VertexBlock_", "gout": "%GeometryBlock"}),
    ],
    ids=["mesh", "tessellation-control", "tessellation-evaluation", "geometry"],
)
def test_interface_block_instances_become_spirv_variables(
    tmp_path, name, source, instances
):
    spirv = translate_spirv(_write(tmp_path, name, source))
    for instance, type_prefix in instances.items():
        assert "Location 0" in _decorations(spirv, instance) or (
            "Location 2" in _decorations(spirv, instance)
        )
        declaration = next(
            line
            for line in spirv.splitlines()
            if line.strip().startswith(f"%{instance} = OpVariable")
        )
        pointer = declaration.split()[3]
        pointee = next(
            line
            for line in spirv.splitlines()
            if line.strip().startswith(f"{pointer} = OpTypePointer")
        ).split()[-1]
        assert pointee.startswith(type_prefix), (instance, pointee)
    _validate_spirv(tmp_path, spirv)


def test_mesh_block_counts_follow_the_stage_layout(tmp_path):
    spirv = translate_spirv(_write(tmp_path, "shader.mesh", MESH))
    # max_vertices = 4 * 2 * 2 and max_primitives = 2 * 2 * 2.
    assert "%_arr_Block_int_16 = OpTypeArray %Block %int_16" in spirv
    assert "%_arr_PrimitiveBlock_int_8 = OpTypeArray %PrimitiveBlock %int_8" in spirv
    assert _decorations(spirv, "pout") == ["Location 2", "PerPrimitiveEXT"]
    assert "OutputVertices 16" in spirv and "OutputPrimitivesEXT 8" in spirv


def test_mesh_layout_constant_expressions_reach_glsl(tmp_path):
    glsl = crosstl.translate(
        _write(tmp_path, "shader.mesh", MESH), backend="opengl", format_output=False
    )
    assert "max_vertices = 16, max_primitives = 8" in glsl
    assert "out vec3 tint[16];" in glsl
    if HAS_GLSLANG:
        (tmp_path / "out.mesh").write_text(glsl)
        _run(["glslangValidator", "--target-env", "vulkan1.3", "out.mesh"], tmp_path)


@pytest.mark.parametrize(
    "name, source, blocks",
    [
        ("shader.tesc", TESC, ["out ControlBlock {"]),
        ("shader.geom", GEOM, ["in VertexBlock {", "out GeometryBlock {"]),
    ],
    ids=["tessellation-control", "geometry"],
)
def test_glsl_keeps_the_interface_blocks(tmp_path, name, source, blocks):
    glsl = crosstl.translate(
        _write(tmp_path, name, source), backend="opengl", format_output=False
    )
    for block in blocks:
        assert block in glsl
        assert f"{block[:-2]} " not in glsl.replace(block, "")


def test_stage_layout_counts_fold_only_known_constants(tmp_path):
    program = _program(tmp_path, "shader.mesh", MESH)
    assert fold_stage_layout_counts(program, "cgl") == 0
    assert fold_stage_layout_counts(program, "vulkan") == 2
    stage = next(iter(program.stages.values()))
    values = {
        entry.name: entry.arguments[0]
        for qualifier in stage.layout_qualifiers
        for entry in qualifier.entries
        if entry.arguments
    }
    assert values["max_vertices"].value == 16
    assert values["max_primitives"].value == 8
    # A single constant name is left to the emitters, which already resolve it.
    assert isinstance(values["local_size_x"], A.IdentifierNode)


def test_stage_layout_count_naming_a_specialization_constant_is_kept(tmp_path):
    source = MESH.replace(
        "const uint GroupX = 2;", "layout(constant_id = 0) const uint GroupX = 2;"
    )
    program = _program(tmp_path, "shader.mesh", source)
    fold_stage_layout_counts(program, "vulkan")
    stage = next(iter(program.stages.values()))
    entry = next(
        entry
        for qualifier in stage.layout_qualifiers
        for entry in qualifier.entries
        if entry.name == "max_vertices"
    )
    assert not isinstance(entry.arguments[0], A.LiteralNode)


def test_declares_only_used_and_undeclared_instances(tmp_path):
    program = _program(tmp_path, "shader.geom", GEOM)
    assert declare_interface_block_instances(program, "opengl") == 0
    assert declare_interface_block_instances(program, "vulkan") == 2
    names = [v.name for v in program.global_variables]
    assert names.count("gin") == 1 and names.count("gout") == 1
    # A second run finds them declared.
    assert declare_interface_block_instances(program, "vulkan") == 0
    gout = next(v for v in program.global_variables if v.name == "gout")
    assert gout.qualifiers == ["out"]
    assert [a.name for a in gout.attributes] == ["location"]


def test_member_locations_are_refused_for_spirv(tmp_path):
    source = GEOM.replace(
        "out GeometryBlock { vec3 colour; } gout;",
        "out GeometryBlock { layout(location = 4) vec3 colour; } gout;",
    )
    program = _program(tmp_path, "shader.geom", source)
    with pytest.raises(InterfaceBlockInstanceUnsupportedError, match="location"):
        apply_rule_carrying(program, "glsl", "vulkan")
