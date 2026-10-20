"""Array values: constructors outside declarations, whole-array copies, and arrays in stage interfaces.

GLSL builds an array value with a constructor anywhere an expression may stand and assigns arrays as values. HLSL,
MSL and SPIR-V build arrays from elements only in aggregate initializers, and MSL's C arrays cannot be assigned.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.semantics.array_values import (
    ArrayConstructorUnsupportedError,
    array_constructor_shape,
    hoist_array_constructors,
)
from crosstl.translator.semantics.rewrite import child_slots, get_child

ASSIGNED_CGL = """
shader Arrays {
    struct Pair { vec3 N[2]; int id; }
    struct FragmentInput { vec2 uv @ TexCoord0; }
    struct FragmentOutput { vec4 color @ gl_FragColor; }
    vec3 pick(vec3 v[2], int i) { return v[i]; }
    fragment {
    FragmentOutput main(FragmentInput input) {
        FragmentOutput output;
        vec3 a[2] = vec3[2](vec3(input.uv, 0.0), vec3(1.0));
        Pair p = Pair(vec3[2](vec3(input.uv, 1.0), vec3(2.0)), 3);
        a = vec3[2](a[1], a[0]);
        vec3 c = pick(vec3[2](vec3(2.0), vec3(input.uv, 3.0)), 1);
        output.color = vec4(a[0] + a[1] + p.N[1] + c, float(p.id));
        return output;
    }
    }
}
"""

STAGE_ARRAY_CGL = """
shader StageArrays {
    struct VSOutput {
        vec4 Pos @ gl_Position;
        vec3 N[2] @ Normal;
        int id @ TexCoord3;
    }
    vertex {
    VSOutput main(vec3 p @ Position) {
        VSOutput output = VSOutput(vec4(p, 1.0), vec3[2](vec3(0.0), vec3(p)), int(0));
        return output;
    }
    }
}
"""

COPIES_GLSL = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
struct S { vec2 v[2]; float w; };
void main() {
    vec2 a[2] = vec2[2](uv, uv * 2.0);
    vec2 b[2] = a;
    const vec2 k[2] = a;
    S s = S(a, 1.0);
    S t = S(vec2[2](uv, uv), 2.0);
    s.v = t.v;
    float m[2][3];
    float n[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0));
    m = n;
    color = vec4(b[1] + s.v[0] + k[0], m[1][2], 1.0);
}
"""


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    # SPIR-V keeps its symbolic ids only when formatted.
    return crosstl.translate(
        str(path), backend=backend, format_output=backend == "vulkan"
    )


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


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def _validate_hlsl(tmp_path, code, profile, entry):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(code)
    _run([compiler, "-T", profile, "-E", entry, str(source)])


def _entry_statements(program):
    return next(iter(program.stages.values())).entry_point.body.statements


def _constructors(program):
    found, stack, seen = [], [program], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if array_constructor_shape(node) is not None:
            found.append(node)
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    return found


@pytest.mark.parametrize("target", ["metal", "directx", "vulkan"])
def test_array_constructors_become_initializer_lists_or_locals(target):
    program = parse(ASSIGNED_CGL)

    rewritten = hoist_array_constructors(program, target)

    # The declaration's own constructor is its initializer list; the other three are rewritten.
    assert rewritten == 3
    assert _constructors(program) == []
    statements = _entry_statements(program)
    names = [getattr(statement, "name", None) for statement in statements]
    assert names.count("cgl_array_value_0") == 1
    assert names.count("cgl_array_value_1") == 1
    hoisted = statements[names.index("cgl_array_value_0")]
    assert isinstance(hoisted.initial_value, A.ArrayLiteralNode)
    assert names.index("cgl_array_value_0") < names.index("c")
    pair = statements[names.index("p")].initial_value
    assert isinstance(pair.arguments[0], A.ArrayLiteralNode)


def test_glsl_keeps_array_constructors_as_values():
    program = parse(ASSIGNED_CGL)

    assert hoist_array_constructors(program, "opengl") == 0
    assert len(_constructors(program)) == 4


@pytest.mark.parametrize(
    ("statement", "reason"),
    [
        (
            "for (int i = 0; i < vec3[2](a[0], a[1])[0].x; i++) { a[0] += vec3(1.0); }",
            "array-constructor-in-loop-header",
        ),
        (
            "float r = bump(k) + vec3[2](a[0], a[1])[0].x;",
            "array-constructor-after-effects",
        ),
        (
            "float r = k > 0 ? vec3[2](vec3(bump(k)), a[1])[0].x : 0.0;",
            "array-constructor-conditional-effects",
        ),
    ],
)
def test_array_constructors_that_cannot_move_are_refused(statement, reason):
    source = f"""
shader Refused {{
    float bump(inout int k) {{ k = k + 1; return float(k); }}
    struct FragmentInput {{ vec2 uv @ TexCoord0; }}
    struct FragmentOutput {{ vec4 color @ gl_FragColor; }}
    fragment {{
    FragmentOutput main(FragmentInput input) {{
        FragmentOutput output;
        int k = 0;
        vec3 a[2] = vec3[2](vec3(input.uv, 0.0), vec3(1.0));
        {statement}
        output.color = vec4(a[0], 1.0);
        return output;
    }}
    }}
}}
"""
    with pytest.raises(ArrayConstructorUnsupportedError) as raised:
        hoist_array_constructors(parse(source), "metal")
    assert raised.value.reason_code == reason


def test_metal_copies_assigned_arrays_element_by_element(tmp_path):
    code = _translate(tmp_path, ASSIGNED_CGL, ".cgl", "metal")

    body = code[code.index("fragment_main") :]
    assert re.search(r"\ba = cgl_array_value_\d+;", body) is None
    assert re.search(
        r"for \(int (__crossgl_array_index_\d+) = 0; \1 < 2; \+\+\1\) \{\s*"
        r"a\[\1\] = cgl_array_value_\d+\[\1\];",
        body,
    )
    assert "Pair{{float3(input.uv, 1.0), float3(2.0)}, 3}" in body
    _validate_metal(tmp_path, code)


def test_metal_declarations_and_aggregates_take_array_values(tmp_path):
    code = _translate(tmp_path, COPIES_GLSL, ".frag", "metal")

    body = code[code.index("fragment_main") :]
    assert "float2 b[2] = {a[0], a[1]};" in body
    assert "float2 k[2] = {a[0], a[1]};" in body
    assert "S{{a[0], a[1]}, 1.0}" in body
    assert re.search(r"s\.v\[(__crossgl_array_index_\d+)\] = t\.v\[\1\]", body)
    assert re.search(
        r"m\[(__crossgl_array_index_\d+)\]\[(__crossgl_array_index_\d+)\] =\s*"
        r"n\[\1\]\[\2\]",
        body,
    )
    assert re.search(r"\] = (a|n);", body) is None
    _validate_metal(tmp_path, code)


def test_metal_refuses_array_copies_with_effectful_operands(tmp_path):
    source = """#version 450
layout(location = 0) out vec4 color;
int advance(inout int i) {
    i = i + 1;
    return i;
}
void main() {
    float rows[3][2];
    float row[2] = float[2](1.0, 2.0);
    int i = 0;
    rows[advance(i)] = row;
    color = vec4(rows[0][0]);
}
"""
    with pytest.raises(MetalLoweringRefused, match="array assignment"):
        _translate(tmp_path, source, ".frag", "metal")


def test_metal_stage_output_arrays_take_their_elements_in_order(tmp_path):
    code = _translate(tmp_path, STAGE_ARRAY_CGL, ".cgl", "metal")

    assert "float3 N_0 " in code and "float3 N_1 " in code
    assert "VSOutput{float4(p, 1.0), float3(0.0), float3(p), int(0)}" in code
    _validate_metal(tmp_path, code)


def test_metal_stage_output_arrays_split_array_values(tmp_path):
    source = STAGE_ARRAY_CGL.replace(
        "VSOutput output = VSOutput(vec4(p, 1.0), vec3[2](vec3(0.0), vec3(p)), int(0));",
        "vec3 n[2] = vec3[2](vec3(0.0), vec3(p));\n"
        "        VSOutput output = VSOutput(vec4(p, 1.0), n, int(0));",
    )

    code = _translate(tmp_path, source, ".cgl", "metal")

    assert "VSOutput{float4(p, 1.0), n[0], n[1], int(0)}" in code
    _validate_metal(tmp_path, code)


def test_spirv_stage_structure_members_after_an_array_take_later_locations(tmp_path):
    code = _translate(tmp_path, STAGE_ARRAY_CGL, ".cgl", "vulkan")

    # Without SPIRV-Tools the module keeps numeric ids; name them through OpName.
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', code))
    prefix = "CrossGL_vertex_output_main_"
    locations = {
        names.get(target, target)[len(prefix) :]: value
        for target, value in re.findall(r"OpDecorate %(\w+) Location (\d+)", code)
        if names.get(target, target).startswith(prefix)
    }
    assert locations == {"N": "1", "id": "3"}
    _validate_spirv(tmp_path, code)


def test_hlsl_and_spirv_build_hoisted_array_values(tmp_path):
    hlsl = _translate(tmp_path, ASSIGNED_CGL, ".cgl", "directx")
    spirv = _translate(tmp_path, ASSIGNED_CGL, ".cgl", "vulkan")

    assert "float3[2](" not in hlsl and "vec3[2](" not in hlsl
    assert re.search(r"float3 cgl_array_value_0\[2\] = \{", hlsl)
    _validate_hlsl(tmp_path, hlsl, "ps_6_0", "PSMain")
    _validate_spirv(tmp_path, spirv)


PARAMETERS_GLSL = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
float written(vec2 v[2]) { v[0] = vec2(9.0); return v[0].x + v[1].y; }
float read(vec2 r[2]) { return r[0].x + r[1].y; }
void fill(out vec2 o[2]) { o[0] = uv; o[1] = uv; }
float forwarded(vec2 w[2]) { fill(w); return w[1].x; }
void main() {
    vec2 a[2] = vec2[2](uv, uv * 2.0);
    vec2 b[2];
    fill(b);
    color = vec4(a[0], written(a) + read(a) + forwarded(a), b[1].x);
}
"""


def test_metal_callees_get_their_own_copy_of_written_array_parameters(tmp_path):
    code = _translate(tmp_path, PARAMETERS_GLSL, ".frag", "metal")

    assert re.search(
        r"float written\(float2 (cgl_array_argument_v_\d+)\[2\]\) \{\s*"
        r"float2 v\[2\] = \{\1\[0\], \1\[1\]\};\s*v\[0\] = float2\(9\.0\);",
        code,
    )
    assert re.search(
        r"float forwarded\(float2 (cgl_array_argument_w_\d+)\[2\][^)]*\) \{\s*"
        r"float2 w\[2\] = \{\1\[0\], \1\[1\]\};",
        code,
    )
    assert re.search(r"float read\(float2 r\[2\]\)", code)
    assert re.search(r"void fill\(float2 o\[2\]", code)
    _validate_metal(tmp_path, code)


@pytest.mark.parametrize("target", ["directx", "vulkan", "opengl"])
def test_by_value_array_parameters_keep_their_names_on_value_targets(tmp_path, target):
    code = _translate(tmp_path, PARAMETERS_GLSL, ".frag", target)

    assert "cgl_array_argument" not in code
