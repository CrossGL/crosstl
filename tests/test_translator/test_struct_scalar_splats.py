"""Structures built from one scalar, and values stored into GLSL's flattened stage outputs.

HLSL's ``(VSOutput)0`` converts the scalar to every component of the structure. Targets that read a structure
constructor's arguments as its members get the structure written member by member; GLSL, which declares the
members of a vertex or fragment output structure as separate outputs, stores those values into the outputs.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import OpenGLStructConstructionError
from crosstl.translator.semantics.hub_types import call_name
from crosstl.translator.semantics.struct_scalar_splats import (
    expand_struct_scalar_splats,
    struct_member_types,
)

ZERO_CAST_VERTEX_HLSL = """
struct VSInput {
    float3 Pos : POSITION0;
    float2 UV : TEXCOORD0;
};
struct VSOutput {
    float4 Pos : SV_POSITION;
    float2 UV : TEXCOORD0;
    float3 Normal : NORMAL0;
    int Id : TEXCOORD1;
};
VSOutput main(VSInput input) {
    VSOutput output = (VSOutput)0;
    output.UV = input.UV;
    output.Pos = float4(input.Pos, 1.0);
    return output;
}
"""

ZERO_CAST_FRAGMENT_HLSL = """
struct VSOutput {
    float4 Pos : SV_POSITION;
    float2 UV : TEXCOORD0;
};
struct FSOutput {
    float4 Color : SV_TARGET0;
    float4 Normal : SV_TARGET1;
};
FSOutput main(VSOutput input) {
    FSOutput output = (FSOutput)0;
    output.Color = float4(input.UV, 0.0, 1.0);
    return output;
}
"""

SPLAT_CGL = """
shader Splats {
    struct Inner { mat4 m; bool flag; uint count; }
    struct Outer { vec3 n[2]; Inner inner; float scale; }
    struct Single { vec4 v; }
    struct Flat { vec2 a; int b; }
    float next(float x) { return x + 1.0; }
    fragment {
        vec4 main(vec2 uv @ TexCoord0) @ gl_FragColor {
            float k = uv.x;
            Outer zero = Outer(0);
            Outer filled = Outer(k);
            Flat flat = Flat(k);
            Flat called = Flat(next(k));
            Single single = Single(0);
            return vec4(zero.scale + filled.scale + flat.a.x + called.a.x) + single.v;
        }
    }
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_glsl(code, stage):
    if shutil.which("glslangValidator") is None:
        return
    _run(["glslangValidator", "--stdin", "-S", stage], input=code)


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def _validate_metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def _main_body(glsl):
    return glsl[glsl.index("void main()") :]


def _local(program, name):
    entry = next(iter(program.stages.values())).entry_point
    for statement in entry.body.statements:
        if isinstance(statement, A.VariableNode) and statement.name == name:
            return statement.initial_value
    raise AssertionError(name)


def _spelled(node):
    if isinstance(node, A.FunctionCallNode):
        arguments = ", ".join(_spelled(argument) for argument in node.arguments)
        return f"{call_name(node)}({arguments})"
    if isinstance(node, A.LiteralNode):
        return str(node.value)
    if isinstance(node, A.IdentifierNode):
        return node.name
    return type(node).__name__


def test_splat_writes_every_component_of_nested_arrays_and_structures():
    program = parse(SPLAT_CGL)
    assert expand_struct_scalar_splats(program, "directx", "opengl") == 2

    assert _spelled(_local(program, "zero")) == (
        "Outer(vec3[2](vec3(0), vec3(0)), Inner(mat4(0), bool(0), uint(0)), float(0))"
    )
    # A matrix constructor from a non-zero scalar fills only the diagonal, so that splat is not rewritten.
    assert _spelled(_local(program, "filled")) == "Outer(k)"
    assert _spelled(_local(program, "flat")) == "Flat(vec2(k), int(k))"
    # The scalar is evaluated once per component, so a call that may have effects is not duplicated.
    assert _spelled(_local(program, "called")) == "Flat(next(k))"
    # A one-member structure of a vector is an ordinary constructor already.
    assert _spelled(_local(program, "single")) == "Single(0)"


def test_splat_is_written_for_every_target_and_left_to_other_sources():
    # The HLSL and Slang generators take a structure constructor's arguments as its members too.
    for target in ("directx", "dx12", "slang", "opengl"):
        program = parse(SPLAT_CGL)
        assert expand_struct_scalar_splats(program, "directx", target) == 2
        assert _spelled(_local(program, "flat")) == "Flat(vec2(k), int(k))"
    # Metal's S{x} imports as the same call and initializes the first member only.
    for source in ("metal", "opengl", "cgl"):
        program = parse(SPLAT_CGL)
        assert expand_struct_scalar_splats(program, source, "opengl") == 0


@pytest.mark.parametrize("spelling", ["mat4", "mat3x4", "mat4x2", "dmat2x3"])
def test_matrix_member_spelling_parses_back_to_the_same_type(spelling):
    program = parse(f"shader M {{ struct S {{ {spelling} m; int i; }} }}")
    member_type = struct_member_types(program)["S"][0][1]
    expanded = parse(
        "shader M { struct S { "
        + f"{spelling} m; int i; }} "
        + "fragment { vec4 main() @ gl_FragColor { S s = S(0); return vec4(0.0); } } }"
    )
    assert expand_struct_scalar_splats(expanded, "directx", "opengl") == 1
    constructor = _local(expanded, "s").arguments[0]
    reparsed = parse("shader M { struct T { " + f"{call_name(constructor)} m; }} }}")
    reparsed_type = struct_member_types(reparsed)["T"][0][1]
    assert repr(reparsed_type) == repr(member_type)


def test_zero_cast_vertex_output_initializes_every_glsl_output(tmp_path):
    glsl = _translate(tmp_path, ZERO_CAST_VERTEX_HLSL, ".hlsl", "opengl")
    body = _main_body(glsl)

    # The HLSL importer writes (VSOutput)0 as the zero value of every member.
    assert "VSOutput" not in body
    assert re.search(r"gl_Position = vec4\(0\.0\);", body)
    assert re.search(r"\bNormal = vec3\(0\.0\);", body)
    assert re.search(r"\bId = 0;", body)
    # The zero stores come before the source's own member writes.
    assert body.index("gl_Position = vec4(0.0);") < body.index("gl_Position = vec4(Pos")
    _validate_glsl(glsl, "vert")


def test_zero_cast_fragment_output_initializes_every_glsl_output(tmp_path):
    glsl = _translate(tmp_path, ZERO_CAST_FRAGMENT_HLSL, ".hlsl", "opengl")
    body = _main_body(glsl)

    assert body.count("= vec4(0.0);") == 2
    _validate_glsl(glsl, "frag")


def test_zero_cast_vertex_output_reaches_spirv_and_metal(tmp_path):
    spirv = _translate(tmp_path, ZERO_CAST_VERTEX_HLSL, ".hlsl", "vulkan")
    assert "OpEntryPoint Vertex" in spirv
    _validate_spirv(tmp_path, spirv)

    metal = _translate(tmp_path, ZERO_CAST_VERTEX_HLSL, ".hlsl", "metal")
    assert "VSOutput{float4(0.0), float2(0.0), float3(0.0), 0}" in metal
    _validate_metal(tmp_path, metal)


def _validate_hlsl(tmp_path, code, profile, entry):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(code)
    _run([compiler, "-T", profile, "-E", entry, str(source)])


def test_zero_cast_vertex_output_stays_a_splat_in_hlsl(tmp_path):
    hlsl = _translate(tmp_path, ZERO_CAST_VERTEX_HLSL, ".hlsl", "directx")
    assert "VSOutput VSMain" in hlsl or "VSOutput main" in hlsl
    # HLSL has no one-argument vector constructor: ``float3(0)`` does not compile.
    assert "output.Normal = float3(0.0, 0.0, 0.0);" in hlsl
    assert not re.search(r"float[234]\(0\)", hlsl)
    _validate_hlsl(tmp_path, hlsl, "vs_6_0", "VSMain")


def test_non_zero_cast_sets_every_hlsl_member(tmp_path):
    source = ZERO_CAST_VERTEX_HLSL.replace("(VSOutput)0", "(VSOutput)1")
    hlsl = _translate(tmp_path, source, ".hlsl", "directx")

    assert "output.Pos = float4(1, 1, 1, 1);" in hlsl
    assert "output.Normal = float3(1, 1, 1);" in hlsl
    assert "output.Id = int(1);" in hlsl
    _validate_hlsl(tmp_path, hlsl, "vs_6_0", "VSMain")


def _stage_output_cgl(body):
    return (
        "shader Outputs {\n"
        "    struct VSOutput { vec4 Pos @ gl_Position; vec2 UV @ TexCoord0; }\n"
        "    vertex {\n"
        "        VSOutput main(vec3 p @ Position) {\n"
        f"            {body}\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _cgl_to_glsl(tmp_path, source):
    return _translate(tmp_path, source, ".cgl", "opengl")


def test_flattened_output_declaration_and_assignment_store_each_member(tmp_path):
    declared = _main_body(
        _cgl_to_glsl(
            tmp_path,
            _stage_output_cgl(
                "VSOutput o = VSOutput(vec4(p, 1.0), p.xy); o.UV = p.yz; return o;"
            ),
        )
    )
    assert re.search(r"gl_Position = vec4\(p, 1\.0\);\s+\w+ = p\.xy;", declared)

    assigned = _cgl_to_glsl(
        tmp_path,
        _stage_output_cgl("VSOutput o; o = VSOutput(vec4(p, 1.0), p.xy); return o;"),
    )
    assert re.search(
        r"gl_Position = vec4\(p, 1\.0\);\s+\w+ = p\.xy;", _main_body(assigned)
    )
    assert " o = " not in assigned
    _validate_glsl(assigned, "vert")


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "VSOutput o = VSOutput(vec4(p, 1.0), p.xy);"
            " o = VSOutput(vec4(o.UV, 0.0, 1.0), o.Pos.xy); return o;",
            "flattened-stage-output-self-reference",
        ),
        (
            "VSOutput o = make(p); return o;",
            "flattened-stage-output-value",
        ),
    ],
)
def test_flattened_output_values_that_cannot_be_split_are_refused(
    tmp_path, body, reason
):
    source = _stage_output_cgl(body).replace(
        "    vertex {",
        "    VSOutput make(vec3 q) { return VSOutput(vec4(q, 1.0), q.xy); }\n    vertex {",
    )
    with pytest.raises(OpenGLStructConstructionError) as refused:
        _cgl_to_glsl(tmp_path, source)
    assert refused.value.reason == reason
