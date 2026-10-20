"""Stage-level interface variables become entry-point structures for MSL and HLSL.

The SPIR-V importer declares each vertex or fragment input and output as a stage-level variable of a ``void
main()``. MSL and HLSL entry points take their inputs and return their outputs as structures; the Metal emitter
used to declare the variables as uninitialized locals, and HLSL left fragment ones undeclared.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.semantics.stage_interface_structs import (
    StageInterfaceUnsupportedError,
    gather_stage_interface_variables,
)

VERTEX_CGL = """
shader main {
    vertex {
        int gl_VertexID @input @gl_VertexID;
        vec3 outColor @output @location(0);
        vec3 inPos @input @location(0);
        int outId @output @location(1) @flat;
        int gl_InstanceID @input @gl_InstanceID;
        vec4 gl_Position @output @gl_Position;
        float gl_PointSize @output @gl_PointSize;
        void main() {
            outColor = shade(inPos);
            outId = gl_InstanceID;
            place(inPos);
            return;
        }
    }
    vec3 shade(vec3 p) {
        return p * float(gl_VertexID);
    }
    void place(vec3 p) {
        gl_Position = vec4(p, 1.0);
    }
}
"""

FRAGMENT_CGL = """
shader main {
    fragment {
        vec4 outFragColor @output @location(0);
        vec3 inColor @input @location(0);
        vec4 gl_FragCoord @input @gl_FragCoord;
        int inId @input @location(1) @flat;
        vec4 outSecond @output @location(1);
        void main() {
            outFragColor = vec4(inColor * gl_FragCoord.x, float(inId));
            if (inId > 3) {
                return;
            }
            outSecond = vec4(1.0);
        }
    }
}
"""

MRT_HLSL = """
struct PSOut { float4 a : SV_Target0; float4 b : SV_Target1; };
struct PSIn { float4 pos : SV_Position; float3 c : TEXCOORD0; };
PSOut main(PSIn i) { PSOut o; o.a = float4(i.c, 1.0); o.b = i.pos; return o; }
"""


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


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


def _validate_hlsl(tmp_path, code, profile, entry):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(code)
    _run([compiler, "-T", profile, "-E", entry, str(source)])


def _struct(program, name):
    return next(struct for struct in program.structs if struct.name == name)


def _attributes(member):
    return [attribute.name for attribute in member.attributes]


def test_vertex_interface_becomes_entry_structures_and_helper_parameters():
    program = parse(VERTEX_CGL)

    assert gather_stage_interface_variables(program, "metal") == 1

    stage = next(iter(program.stages.values()))
    assert stage.local_variables == []
    inputs = _struct(program, "VertexInput")
    outputs = _struct(program, "VertexOutput")
    assert [member.name for member in inputs.members] == ["inPos"]
    # Unused gl_PerVertex members are dropped; used built-in outputs stay members, named as the HLSL and MSL
    # vertex contracts name them.
    assert [member.name for member in outputs.members] == [
        "outColor",
        "outId",
        "position",
    ]
    assert _attributes(outputs.members[1]) == ["location", "flat"]
    assert _attributes(outputs.members[2]) == ["gl_Position"]
    entry = stage.entry_point
    assert [(p.name, p.param_type.name, _attributes(p)) for p in entry.parameters] == [
        ("input", "VertexInput", []),
        ("cgl_VertexID_bits", "uint", ["gl_VertexID"]),
        ("cgl_InstanceID_bits", "uint", ["gl_InstanceID"]),
    ]
    conversions = entry.body.statements[:2]
    assert [(s.name, s.initial_value.arguments[0].name) for s in conversions] == [
        ("cgl_VertexID", "cgl_VertexID_bits"),
        ("cgl_InstanceID", "cgl_InstanceID_bits"),
    ]
    assert entry.return_type.name == "VertexOutput"
    returns = [s for s in entry.body.statements if isinstance(s, A.ReturnNode)]
    assert [r.value.name for r in returns] == ["output"]
    helpers = {function.name: function for function in program.functions}
    assert [p.name for p in helpers["shade"].parameters] == ["p", "cgl_VertexID"]
    assert [(p.name, p.qualifiers) for p in helpers["place"].parameters] == [
        ("p", []),
        ("output", ["inout"]),
    ]


@pytest.mark.parametrize("target", ["opengl", "vulkan", "webgl"])
def test_targets_with_stage_variables_keep_them(target):
    program = parse(VERTEX_CGL)

    assert gather_stage_interface_variables(program, target) == 0
    assert len(next(iter(program.stages.values())).local_variables) == 7


def test_metal_vertex_stage_reads_inputs_and_returns_outputs(tmp_path):
    code = _translate(tmp_path, VERTEX_CGL, "metal")

    assert re.search(
        r"vertex VertexOutput vertex_main\(VertexInput input \[\[stage_in\]\],\s*"
        r"uint cgl_VertexID_bits \[\[vertex_id\]\],\s*"
        r"uint cgl_InstanceID_bits \[\[instance_id\]\]\)",
        code,
    )
    assert "int cgl_VertexID = int(cgl_VertexID_bits);" in code
    assert "__attribute__((unused)) float3 inPos" not in code
    assert re.search(r"output\.outColor = shade\(input\.inPos, cgl_VertexID\);", code)
    assert re.search(r"place\(input\.inPos, output\);", code)
    assert "return output;" in code
    _validate_metal(tmp_path, code)


def test_hlsl_vertex_stage_reads_inputs_and_returns_outputs(tmp_path):
    code = _translate(tmp_path, VERTEX_CGL, "directx")

    assert re.search(
        r"VertexOutput VSMain\(VertexInput input, uint cgl_VertexID_bits : SV_VertexID, "
        r"uint cgl_InstanceID_bits : SV_InstanceID\)",
        code,
    )
    _validate_hlsl(tmp_path, code, "vs_6_0", "VSMain")


def test_fragment_outputs_become_colour_attachments(tmp_path):
    metal = _translate(tmp_path, FRAGMENT_CGL, "metal")
    hlsl = _translate(tmp_path, FRAGMENT_CGL, "directx")

    assert re.search(r"float4 outFragColor \[\[color\(0\)\]\];", metal)
    assert re.search(r"float4 outSecond \[\[color\(1\)\]\];", metal)
    assert re.search(r"float4 \w+ \[\[position\]\]", metal)
    assert metal.count("return output;") == 2
    assert re.search(r"float4 outFragColor\s*: SV_TARGET0;", hlsl, re.I)
    assert re.search(r"float4 outSecond\s*: SV_TARGET1;", hlsl, re.I)
    _validate_metal(tmp_path, metal)
    _validate_hlsl(tmp_path, hlsl, "ps_6_0", "PSMain")


def test_metal_hlsl_render_targets_are_colour_attachments(tmp_path):
    code = _translate(tmp_path, MRT_HLSL, "metal", suffix=".hlsl")

    assert "float4 a [[color(0)]];" in code
    assert "float4 b [[color(1)]];" in code
    _validate_metal(tmp_path, code)


def test_entry_points_with_their_own_interface_are_refused():
    source = VERTEX_CGL.replace("void main() {", "void main(int extra @ TexCoord5) {")

    with pytest.raises(StageInterfaceUnsupportedError) as raised:
        gather_stage_interface_variables(parse(source), "metal")
    assert raised.value.reason_code == "mixed-entry-interface"


def test_helpers_declaring_an_interface_name_are_refused():
    source = VERTEX_CGL.replace(
        "return p * float(gl_VertexID);",
        "float inPos = 1.0;\n        return p * inPos;",
    )

    with pytest.raises(StageInterfaceUnsupportedError) as raised:
        gather_stage_interface_variables(parse(source), "directx")
    assert raised.value.reason_code == "shadowed-interface-variable"


SPIRV_PERVERTEX_CGL = """
shader main {
    @glsl_interface_block(out)
    @glsl_interface_block_name(gl_PerVertex)
    struct gl_PerVertex {
        float4 gl_Position;
        float gl_PointSize @mediump;
        float gl_ClipDistance[8];
        float gl_CullDistance[8];
    }
    vertex {
        float4 _ua_position @input @location(0);
        float4 ua_color @input @location(1);
        float4 gl_Position @output @gl_Position;
        float gl_PointSize @output @gl_PointSize @mediump;
        void main() {
            gl_Position = _ua_position + ua_color;
            return;
        }
    }
}
"""


@pytest.mark.parametrize("target", ["metal", "directx"])
def test_spirv_pervertex_block_is_dropped_and_member_names_are_plain(tmp_path, target):
    program = parse(SPIRV_PERVERTEX_CGL)

    assert gather_stage_interface_variables(program, target) == 1

    assert [struct.name for struct in program.structs] == [
        "VertexInput",
        "VertexOutput",
    ]
    # The leading underscore goes unless it would make two members collide.
    assert [m.name for m in _struct(program, "VertexInput").members] == [
        "ua_position",
        "ua_color",
    ]
    assert [m.name for m in _struct(program, "VertexOutput").members] == ["position"]


def test_hlsl_spirv_pervertex_block_emits_no_precision_semantic(tmp_path):
    code = _translate(tmp_path, SPIRV_PERVERTEX_CGL, "directx")

    assert "gl_PerVertex" not in code
    assert "mediump" not in code
    assert "float4 position: SV_POSITION;" in code
    assert "output.position = (input.ua_position + input.ua_color);" in code
    _validate_hlsl(tmp_path, code, "vs_6_0", "VSMain")
