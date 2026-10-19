"""Structs holding samplers and textures reach SPIR-V as separate resources.

A Vulkan module cannot keep an opaque value inside a struct, so the SPIR-V emitter refused every access such as
``texture(material.albedoMap, uv)``. :mod:`crosstl.translator.semantics.opaque_structs` now flattens each opaque
member of a uniform struct into its own uniform (one array dimension per struct array crossed), expands struct
parameters into data plus resource parameters, and turns local copies into recorded indices. These tests check the
rewrite on the AST, that the modules validate, and that the forms without a faithful rewrite are refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    OpaqueStructUnsupportedError,
    legalize_opaque_struct_members,
)

ROOT = Path(__file__).resolve().parents[2]

MATERIALS = """
shader Materials {
    struct Material { vec4 tint; sampler2D albedo; }
    struct Probe { samplerCube faces[2]; }
    struct Scene { Material materials[4]; Probe probes[3]; float exposure; }
    struct FSIn { vec2 uv; }
    struct FSOut { vec4 color; }
    vec4 shade(Material m, vec2 uv) { return texture(m.albedo, uv) * m.tint; }
    vec4 shadeTwice(Material m, vec2 uv) { return shade(m, uv) * 0.5; }
    fragment {
        uniform Scene scene;
        uniform int first;
        FSOut main(FSIn input) {
            FSOut o;
            int which = first;
            Material m = scene.materials[which];
            which = 3;
            vec4 c = texture(m.albedo, input.uv) * m.tint;
            m = scene.materials[which];
            c += shadeTwice(m, input.uv) + shade(scene.materials[2], input.uv);
            c += texture(scene.probes[which].faces[first], vec3(input.uv, 1.0));
            o.color = c * scene.exposure;
            return o;
        }
    }
}
"""


def parse(source):
    return Parser(Lexer(source).tokens).parse()


def translate(tmp_path, source, backend="vulkan"):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def validate(tmp_path, spirv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(spirv)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def stage(program):
    return next(iter(program.stages.values()))


def type_text(node):
    if isinstance(node, A.ArrayType):
        return f"{type_text(node.element_type)}[{node.size.value}]"
    return node.name


def test_uniform_struct_members_become_descriptor_arrays():
    program = parse(MATERIALS)
    legalize_opaque_struct_members(program, "vulkan")
    declared = {v.name: v for v in stage(program).local_variables}
    assert set(declared) == {
        "scene",
        "scene_materials_albedo",
        "scene_probes_faces",
        "first",
    }
    # One dimension per struct array crossed; Vulkan descriptor arrays are one-dimensional, so `probes[3]` and
    # `faces[2]` become six faces indexed row-major.
    assert type_text(declared["scene_materials_albedo"].var_type) == "sampler2D[4]"
    assert type_text(declared["scene_probes_faces"].var_type) == "samplerCube[6]"
    assert declared["scene_probes_faces"].qualifiers == ["uniform"]
    material = next(s for s in program.structs if s.name == "Material")
    assert [m.name for m in material.members] == ["tint"]
    probe = next(s for s in program.structs if s.name == "Probe")
    assert probe.members == []
    scene = next(s for s in program.structs if s.name == "Scene")
    assert [m.name for m in scene.members] == ["materials", "exposure"]


def test_parameters_expand_and_calls_pass_the_selected_resources():
    program = parse(MATERIALS)
    legalize_opaque_struct_members(program, "vulkan")
    shade = next(f for f in program.functions if f.name == "shade")
    assert [p.name for p in shade.parameters] == ["m", "m_albedo", "uv"]
    assert type_text(shade.parameters[1].param_type) == "sampler2D"
    entry = stage(program).entry_point
    statements = entry.body.statements
    # The copy records its index once, `m.albedo` reads the recorded index, and the reassignment updates it.
    assert statements[2].name == "m_index0"
    assert statements[2].initial_value.arguments[0].name == "which"
    assert statements[3].name == "m"
    assert statements[3].initial_value.index_expr.name == "m_index0"
    sampled = statements[5].initial_value.left.arguments[0]
    assert (sampled.array_expr.name, sampled.index_expr.name) == (
        "scene_materials_albedo",
        "m_index0",
    )
    update = statements[6].expression
    assert (update.target.name, update.value.arguments[0].name) == ("m_index0", "which")
    call = statements[8].expression.value.right
    assert call_name(call) == "shade"
    passed = call.arguments[1]
    assert passed.array_expr.name == "scene_materials_albedo"
    assert passed.index_expr.value == 2
    face = statements[9].expression.value.arguments[0]
    assert face.array_expr.name == "scene_probes_faces"
    # int(which) * 2 + int(first)
    assert face.index_expr.operator == "+"
    assert face.index_expr.left.right.value == 2


def call_name(node):
    return node.function.name


def test_materials_module_validates(tmp_path):
    spirv = translate(tmp_path, MATERIALS)
    validate(tmp_path, spirv)
    names = dict(re.findall(r'^OpName %(\w+) "(\w+)"$', spirv, re.M))
    albedo = next(i for i, n in names.items() if n == "scene_materials_albedo")
    assert re.search(rf"^%{albedo} = OpVariable %\w+ UniformConstant$", spirv, re.M)
    assert re.search(rf"= OpAccessChain %\w+ %{albedo} %\w+$", spirv, re.M)


def test_struct_of_only_resources_is_dropped(tmp_path):
    source = """
    shader OnlyTextures {
        struct Maps { sampler2D a; sampler2D b; }
        struct FSOut { vec4 color; }
        fragment {
            uniform Maps maps;
            FSOut main(vec2 uv) {
                FSOut o;
                Maps local = maps;
                o.color = texture(maps.a, uv) + texture(local.b, uv);
                return o;
            }
        }
    }
    """
    program = parse(source)
    legalize_opaque_struct_members(program, "vulkan")
    assert [v.name for v in stage(program).local_variables] == ["maps_a", "maps_b"]
    validate(tmp_path, translate(tmp_path, source))


def test_other_targets_keep_the_struct():
    program = parse(MATERIALS)
    legalize_opaque_struct_members(program, "metal")
    material = next(s for s in program.structs if s.name == "Material")
    assert [m.name for m in material.members] == ["tint", "albedo"]


@pytest.mark.parametrize(
    "helper, body, reason",
    [
        ("", "Material m; o.color = m.tint;", "is not a copy"),
        (
            "",
            "Material m = scene.materials[0]; m = other.materials[1]; o.color = m.tint;",
            "reassigned from a different resource",
        ),
        (
            "Material make(Material m) { return m; }",
            "o.color = make(scene.materials[0]).tint;",
            "returns struct Material",
        ),
        ("", "Material m = Material(vec4(1.0)); o.color = m.tint;", "is not a copy"),
        ("", "o.color = Material(vec4(1.0)).tint;", "is constructed"),
        (
            "vec4 write(inout Material m) { return m.tint; }",
            "o.color = write(scene.materials[0]);",
            "is written",
        ),
        (
            "vec4 all(Material ms[2]) { return ms[0].tint; }",
            "o.color = all(scene.materials);",
            "holds an array of resources",
        ),
    ],
)
def test_forms_without_a_faithful_rewrite_are_refused(tmp_path, helper, body, reason):
    source = f"""
    shader Refused {{
        struct Material {{ vec4 tint; sampler2D albedo; }}
        struct Scene {{ Material materials[2]; }}
        struct FSOut {{ vec4 color; }}
        {helper}
        fragment {{
            uniform Scene scene;
            uniform Scene other;
            FSOut main(vec2 uv) {{ FSOut o; {body} return o; }}
        }}
    }}
    """
    with pytest.raises(OpaqueStructUnsupportedError, match=reason):
        translate(tmp_path, source)


def test_non_uniform_struct_of_resources_is_refused(tmp_path):
    source = """
    shader Private {
        struct Material { vec4 tint; sampler2D albedo; }
        struct FSOut { vec4 color; }
        fragment {
            Material m;
            FSOut main(vec2 uv) { FSOut o; o.color = texture(m.albedo, uv); return o; }
        }
    }
    """
    with pytest.raises(OpaqueStructUnsupportedError, match="is not a uniform"):
        translate(tmp_path, source)


@pytest.mark.parametrize(
    "example",
    [
        "examples/graphics/ComplexShader.cgl",
        "examples/cross_platform/UniversalPBRShader.cgl",
    ],
)
def test_examples_with_material_structs_validate(tmp_path, example):
    spirv = crosstl.translate(
        str(ROOT / example), backend="vulkan", format_output=False
    )
    validate(tmp_path, spirv)
