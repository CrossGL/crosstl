"""Program identifiers that HLSL reads as keywords are renamed in HLSL output.

GLSL and Metal allow names such as ``line``, ``point``, ``register``,
``cbuffer``, ``vector`` or ``string`` that DXC parses as primitive types,
storage classes, buffer keywords or built-in types. Every declaration and
reference is renamed, and the result compiles with DXC when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import crosstl
from crosstl.translator import parse
from crosstl.translator.ast import MemberAccessNode
from crosstl.translator.codegen import hlsl_keyword_names
from crosstl.translator.codegen.hlsl_keyword_names import (
    HLSL_KEYWORD_NAMES,
    HLSL_TYPE_KEYWORD_NAMES,
    legalize_hlsl_keyword_names,
)

GLSL_COMPUTE = """#version 450
layout(local_size_x = 8) in;
struct point { float linear; vec2 vector; };
layout(std140, binding = 0) uniform register { float cbuffer; vec4 technique; } tbuffer;
layout(std430, binding = 1) buffer groupshared { float packoffset[]; } triangle;
layout(std430, binding = 2) buffer Buffer { point string[]; };
float line(float row_major, point column_major) {
    return row_major + column_major.linear * column_major.vector.y;
}
void main() {
    uint nointerpolation = gl_GlobalInvocationID.x;
    point lineadj;
    lineadj.linear = tbuffer.cbuffer + tbuffer.technique.x;
    lineadj.vector = vec2(2.0, 3.0);
    mat2 matrix = mat2(1.0);
    float triangleadj = line(triangle.packoffset[nointerpolation], lineadj);
    float export = triangleadj + matrix[1][1];
    float friend = export;
    float typename = friend;
    string[nointerpolation] = lineadj;
    triangle.packoffset[nointerpolation] = typename;
}
"""

MSL_COMPUTE = """#include <metal_stdlib>
using namespace metal;
struct Params { float vector; uint matrix; };
float point(float in, float out) { return in + out; }
kernel void k(device float* buf [[buffer(0)]],
              constant Params& p [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
    float in = buf[tid];
    float out = in * 2.0;
    float inout = out + p.vector;
    float texture = point(in, inout);
    int string = int(p.matrix);
    float shared = texture + float(string);
    buf[tid] = shared;
}
"""

MSL_GRAPHICS = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float2 vector; float sample; };
struct triangle { float4 string; };
vertex VOut point(constant triangle& cbuffer [[buffer(0)]]) {
    VOut o;
    o.position = cbuffer.string;
    o.vector = o.position.xy;
    o.sample = 1.0;
    return o;
}
fragment float4 line(VOut in [[stage_in]],
                     texture2d<float> shared [[texture(0)]],
                     sampler linear [[sampler(0)]]) {
    float4 texel = shared.sample(linear, in.vector);
    return texel * in.sample;
}
"""

GLSL_PROGRAM_SCOPE_TYPE_NAMES = """#version 450
layout(local_size_x = 1) in;
uniform float vector;
layout(std430, binding = 0) buffer Values { float string[]; };
float matrix(float half_value) { return half_value * vector; }
void main() {
    float uint4 = matrix(2.0);
    string[0] = uint4;
}
"""

GLSL_COLLISION = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Out { float values[]; };
void main() {
    float point = 1.0;
    float point_ = 2.0;
    values[0] = point + point_;
}
"""

GLSL_TYPE_ONLY_NAMES = """#version 450
layout(local_size_x = 1) in;
struct payload { float vertices; };
layout(std430, binding = 0) buffer Values { float values[]; };
void main() {
    payload sample_value = payload(2.0);
    sample_value.vertices += 1.0;
    float indices = sample_value.vertices;
    values[0] = indices;
}
"""

MSL_TWO_KERNELS = """#include <metal_stdlib>
using namespace metal;

kernel void first_kernel(device float* point [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
    float line = 1.0;
    point[id] = line;
}

kernel void second_kernel(device float* triangle [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    float register = 2.0;
    triangle[id] = register;
}
"""

CGL_METHOD_CALL = """shader Names {
    struct Shape {
        float line;
    };
    compute {
        void main() {
            Shape s;
            s.line = 1.0;
            s.line(2.0);
        }
    }
}
"""

CGL_PROGRAM = """shader Names {
    struct line {
        float vector;
    };
    compute {
        void main() {
            line triangle;
            triangle.vector = 1.0;
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _compile(tmp_path, hlsl, profile, entry):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / f"{entry}.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", profile, "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _identifiers(code):
    return set(re.findall(r"\b[A-Za-z_]\w*\b", code))


def test_glsl_keyword_names_are_renamed_in_every_declaration_position(tmp_path):
    hlsl = _translate(tmp_path, GLSL_COMPUTE, ".comp")
    assert re.search(r"struct point_\s*\{", hlsl)
    assert re.search(r"\bfloat linear_;", hlsl)
    # DXC accepts a built-in type spelling as a struct member or local.
    assert re.search(r"\bfloat2 vector;", hlsl)
    assert re.search(r"cbuffer register_\b", hlsl)
    assert re.search(r"\bfloat cbuffer_;", hlsl)
    assert re.search(r"\bfloat4 technique_;", hlsl)
    assert re.search(r"RWStructuredBuffer<float> triangle_\b", hlsl)
    assert re.search(r"\bRWByteAddressBuffer \w+ : register\(u2\)", hlsl)
    assert re.search(r"float line_\(float row_major_, point_ column_major_\)", hlsl)
    for local in ("nointerpolation", "lineadj", "triangleadj", "export"):
        assert re.search(rf"\b{local}_\b", hlsl), local
    assert re.search(r"\bfloat2x2 matrix = ", hlsl)
    assert "friend_" in hlsl and "typename_" in hlsl
    # Only HLSL's own uses of these keywords remain.
    leftover = _identifiers(hlsl) & {
        "point",
        "line",
        "triangle",
        "linear",
        "string",
        "technique",
        "packoffset",
        "lineadj",
        "triangleadj",
        "nointerpolation",
        "row_major",
        "column_major",
        "export",
        "friend",
        "typename",
    }
    assert leftover == set()
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


def test_metal_keyword_names_are_renamed(tmp_path):
    hlsl = _translate(tmp_path, MSL_COMPUTE, ".metal")
    assert re.search(r"\bfloat vector;", hlsl)
    assert re.search(r"\buint matrix;", hlsl)
    assert re.search(r"float point_\(float in_, float out_\)", hlsl)
    assert re.search(r"\bfloat inout_ = ", hlsl)
    assert re.search(r"\bint string = ", hlsl)
    assert re.search(r"\bfloat shared_ = ", hlsl)
    assert "p.vector" in hlsl and "p.matrix" in hlsl
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


def test_methods_and_stage_members_keep_working_after_renames(tmp_path):
    hlsl = _translate(tmp_path, MSL_GRAPHICS, ".metal")
    # DXC accepts ``sample`` as a member, and the texture method is no member.
    assert re.search(r"\bfloat sample\s*:\s*TEXCOORD", hlsl)
    assert re.search(r"shared_\.Sample\(linear_, in_\.vector\)", hlsl)
    assert "in_.sample)" in hlsl
    assert re.search(r"ConstantBuffer<triangle_> cbuffer_\b", hlsl)
    assert "cbuffer_.string" in hlsl
    _compile(tmp_path, hlsl, "vs_6_0", "VSMain")
    _compile(tmp_path, hlsl, "ps_6_0", "PSMain")


def test_type_spellings_are_renamed_only_at_program_scope(tmp_path):
    hlsl = _translate(tmp_path, GLSL_PROGRAM_SCOPE_TYPE_NAMES, ".comp")
    assert re.search(r"\bfloat vector_;", hlsl)
    assert re.search(r"\bfloat matrix_\(float half_value\)", hlsl)
    assert re.search(r"\(half_value \* vector_\)", hlsl)
    # A local keeps a spelling DXC accepts there.
    assert re.search(r"\bfloat uint4 = matrix_\(2\.0\);", hlsl)
    assert not re.search(r"\b(vector|matrix)\b", hlsl)
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


def test_renames_avoid_existing_program_names(tmp_path):
    hlsl = _translate(tmp_path, GLSL_COLLISION, ".comp")
    assert re.search(r"\bfloat point__ = 1\.0;", hlsl)
    assert re.search(r"\bfloat point_ = 2\.0;", hlsl)
    assert re.search(r"\(point__ \+ point_\)", hlsl)
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


def test_type_only_keywords_rename_types_and_keep_value_names(tmp_path):
    hlsl = _translate(tmp_path, GLSL_TYPE_ONLY_NAMES, ".comp")
    assert re.search(r"struct payload_\s*\{\s*float vertices;", hlsl)
    assert re.search(r"\bpayload_ sample_value\b", hlsl)
    assert re.search(r"\bfloat indices = ", hlsl) and "indices_" not in hlsl
    assert "payload " not in hlsl.replace("payload_", "")
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


CGL_TYPE_ALIASES = """shader AliasProgram {
    typedef float float16_t;
    typedef uint vector;

    float16_t twice(float16_t value) {
        return value * 2.0;
    }

    compute {
        buffer float values[];

        @workgroup_size(4, 1, 1)
        void main() {
            uint index = gl_GlobalInvocationID.x;
            vector offset = index;
            float16_t local[2];
            local[0] = values[offset];
            local[1] = twice(local[0]);
            values[index] = local[1];
        }
    }
}
"""


def test_type_aliases_are_renamed_with_every_use(tmp_path):
    # HLSL reads float16_t as binary16 and vector as float4, so an alias named
    # either must be renamed together with the types that name it.
    ast = parse(CGL_TYPE_ALIASES)
    renamed = legalize_hlsl_keyword_names(ast)
    aliases = [node for node in renamed.global_variables if node.is_type_alias]
    assert sorted(alias.name for alias in aliases) == ["float16_t_", "vector_"]
    entry = next(iter(renamed.stages.values())).entry_point
    offset, local = entry.body.statements[1:3]
    assert (offset.var_type.name, local.var_type.element_type.name) == (
        "vector_",
        "float16_t_",
    )
    function = renamed.functions[0]
    assert function.return_type.name == "float16_t_"
    assert function.parameters[0].param_type.name == "float16_t_"

    hlsl = _translate(tmp_path, CGL_TYPE_ALIASES, ".cgl")
    assert "float16_t" not in hlsl and not re.search(r"\bvector\b", hlsl)
    assert re.search(r"\buint offset = index;", hlsl)
    assert re.search(r"\bfloat local\[2\];", hlsl)
    assert re.search(r"\bfloat twice\(float value\)", hlsl)
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


def test_renaming_copies_the_ast_and_keeps_other_programs_untouched():
    ast = parse(CGL_PROGRAM)
    renamed = legalize_hlsl_keyword_names(ast)
    assert renamed is not ast
    assert [struct.name for struct in ast.structs] == ["line"]
    assert [struct.name for struct in renamed.structs] == ["line_"]
    assert [member.name for member in renamed.structs[0].members] == ["vector"]
    stage = next(iter(renamed.stages.values()))
    local = stage.entry_point.body.statements[0]
    assert (local.name, local.var_type.name) == ("triangle_", "line_")

    plain = parse(
        CGL_PROGRAM.replace("line", "Line")
        .replace("vector", "value")
        .replace("triangle", "shape")
    )
    assert legalize_hlsl_keyword_names(plain) is plain


def test_keyword_set_covers_each_declaration_class():
    for name in (
        "in",
        "out",
        "inout",
        "line",
        "point",
        "triangle",
        "register",
        "cbuffer",
        "tbuffer",
        "technique",
        "string",
        "vector",
        "matrix",
        "sampler",
        "half",
        "dword",
        "float4",
        "uint3x4",
        "min16float2",
        "float16_t",
        "Buffer",
        "Texture2D",
        "RayQuery",
        "typename",
    ):
        assert name in HLSL_KEYWORD_NAMES, name
    # Accepted spellings stay as written.
    for name in ("texture", "Texture", "main", "pass", "compile", "NULL"):
        assert name not in HLSL_TYPE_KEYWORD_NAMES, name
    # DXC rejects these only as struct or cbuffer names.
    for name in ("payload", "indices", "vertices", "sample", "SamplerState"):
        assert name not in HLSL_KEYWORD_NAMES, name
        assert name in HLSL_TYPE_KEYWORD_NAMES, name
    assert HLSL_KEYWORD_NAMES < HLSL_TYPE_KEYWORD_NAMES


def test_every_entry_point_of_a_stage_is_renamed(tmp_path):
    hlsl = _translate(tmp_path, MSL_TWO_KERNELS, ".metal")
    for name in ("point", "line", "triangle", "register"):
        assert f"{name}_" in hlsl, name
        assert not re.search(rf"\b{name}\b(?!\()", hlsl), name
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain_2")


def test_member_renames_skip_method_callees_in_the_copy():
    renamed = legalize_hlsl_keyword_names(parse(CGL_METHOD_CALL))
    members = [
        node.member
        for node in hlsl_keyword_names._iter_nodes(renamed)
        if isinstance(node, MemberAccessNode)
    ]
    # The field access is renamed; the method callee keeps its spelling.
    assert members == ["line_", "line"]


def _count_walks(monkeypatch):
    walks = []
    walk = hlsl_keyword_names._iter_nodes

    def counting_walk(root, **options):
        walks.append(options.get("include_opaque", False))
        return walk(root, **options)

    monkeypatch.setattr(hlsl_keyword_names, "_iter_nodes", counting_walk)
    return walks


def test_programs_without_keyword_names_are_walked_once(monkeypatch):
    plain = parse(
        CGL_PROGRAM.replace("line", "Line")
        .replace("vector", "value")
        .replace("triangle", "shape")
    )
    walks = _count_walks(monkeypatch)
    assert legalize_hlsl_keyword_names(plain) is plain
    assert walks == [False]


def test_renaming_walks_each_tree_a_bounded_number_of_times(monkeypatch):
    ast = parse(CGL_PROGRAM)
    walks = _count_walks(monkeypatch)
    assert legalize_hlsl_keyword_names(ast) is not ast
    # Declarations, then the one function body (``vector`` is reserved only
    # at program scope), the names in use, and the rename of the copy.
    assert walks == [False, False, True, False]
