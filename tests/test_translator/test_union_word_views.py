"""Union members word storage carries element by element (:mod:`crosstl.translator.semantics.union_word_views`).

HLSL, GLSL and SPIR-V keep a ``@union_layout`` union as 32-bit words. A structure member, a narrow vector and a
vector or scalar that covers part of the storage become lane views over the whole storage, and every access names
the lane it reads or writes. Native execution of the same shapes from Metal source is in ``test_union_storage.py``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.union_word_views import (
    UnionWordViewError,
    split_union_word_views,
)

PROGRAM = """shader P {{
    struct Pair {{
        float16 d;
        float16 m;
    }}
    struct Mixed {{
        uint16 a;
        float16 b;
    }}
    @union_layout(4, 4, little_endian, metal)
    struct U {{
        @union_member_layout(0, 4, 4) uint w;
        @union_member_layout(0, 4, 2) Pair p;
        @union_member_layout(0, 4, 4) f16vec2 dm;
        @union_member_layout(0, 4, 2) Mixed mixed;
        @union_member_layout(0, 2, 2) uint16 low;
    }}
    @union_layout(16, 16, little_endian, metal)
    struct R {{
        @union_member_layout(0, 16, 16) vec4 data;
        @union_member_layout(0, 16, 16) vec3 center;
    }}
    compute {{
        layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        buffer uint* inp;
        buffer float* outp;
        void main() {{
            U u;
            R r;
            u.w = inp[0];
            {body}
        }}
    }}
}}
"""

BODY = """
            f16vec2 v = u.dm;
            Pair q = u.p;
            outp[0] = float(u.p.d) + float(u.p.m) * 10.0 + float(v.y) * 100.0 + float(q.d);
            u.dm = u.dm.yx;
            outp[1] = float(u.dm.x);
            u.p.m = u.dm[0];
            outp[2] = float(u.w & 65535u);
            r.data = vec4(1.0, 2.0, 3.0, 4.0);
            r.center = vec3(5.0, 6.0, 7.0);
            outp[3] = r.data.w + r.center.z + r.center[1];
"""


def parse(body=BODY):
    return Parser(Lexer(PROGRAM.format(body=body)).get_tokens()).parse()


def members(program, name):
    struct = next(node for node in program.structs if node.name == name)
    return {member.name: member for member in struct.members}


def spelling(type_node):
    if isinstance(type_node, A.ArrayType):
        return f"{spelling(type_node.element_type)}[{type_node.size.value}]"
    if isinstance(type_node, A.VectorType):
        return f"{spelling(type_node.element_type)}{type_node.size}"
    return type_node.name


def layout(member):
    attribute = next(a for a in member.attributes if a.name == "union_member_layout")
    return [argument.value for argument in attribute.arguments]


def write(tmp_path, body=BODY):
    path = tmp_path / "union.cgl"
    path.write_text(PROGRAM.format(body=body), encoding="utf-8")
    return path


def translate(tmp_path, backend, body=BODY):
    return crosstl.translate(
        str(write(tmp_path, body)), backend=backend, format_output=False
    )


def require_tool(name):
    tool = shutil.which(name)
    if tool is None:
        pytest.skip(f"{name} is not installed")
    return tool


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_members_become_lane_views_over_the_whole_storage():
    program = parse()
    assert split_union_word_views(program, "directx") > 0
    union = members(program, "U")
    assert {name: spelling(m.member_type) for name, m in union.items()} == {
        "w": "uint",
        "p": "float16[2]",
        "dm": "float16[2]",
        "mixed_uint16": "uint16[2]",
        "mixed_float16": "float16[2]",
        "low": "uint16[2]",
    }
    assert layout(union["p"]) == [0, 4, 2]
    assert layout(union["mixed_uint16"]) == [0, 4, 2]
    record = members(program, "R")
    assert spelling(record["data"].member_type) == "float4"
    assert spelling(record["center"].member_type) == "float[4]"
    assert layout(record["center"]) == [0, 16, 4]


def test_glsl_reads_and_writes_name_the_lanes(tmp_path):
    generated = translate(tmp_path, "opengl")
    get = "crossgl_union_get_float16_lane_w1"
    put = "crossgl_union_set_float16_lane_w1"
    assert "struct U {\n    uint CrossGLUnionStorage;\n};" in generated
    assert "struct R {\n    uvec4 CrossGLUnionStorage;\n};" in generated
    # Whole reads construct the value from its lanes; d and dm.x are lane 0, m and dm.y lane 1.
    assert (
        f"vec2 v = vec2({get}(u.CrossGLUnionStorage, 0), {get}(u.CrossGLUnionStorage, 1));"
        in generated
    )
    assert (
        f"Pair q = Pair({get}(u.CrossGLUnionStorage, 0), {get}(u.CrossGLUnionStorage, 1));"
        in generated
    )
    # A whole write that reads the member goes through a temporary, one lane write per element.
    assert (
        f"vec2 crossgl_union_value1 = vec2({get}(u.CrossGLUnionStorage, 1), "
        f"{get}(u.CrossGLUnionStorage, 0));" in generated
    )
    assert f"{put}(u.CrossGLUnionStorage, 0, crossgl_union_value1.x);" in generated
    assert f"{put}(u.CrossGLUnionStorage, 1, crossgl_union_value1.y);" in generated
    assert (
        f"{put}(u.CrossGLUnionStorage, 1, {get}(u.CrossGLUnionStorage, 0));"
        in generated
    )
    # vec3 writes its three words and leaves the fourth, its padding, as it was.
    assert (
        "r.CrossGLUnionStorage[2] = floatBitsToUint(crossgl_union_value2.z);"
        in generated
    )
    assert "r.CrossGLUnionStorage[3]" not in generated


def test_glsl_output_compiles_with_glslang(tmp_path):
    validator = require_tool("glslangValidator")
    shader = tmp_path / "union.comp"
    shader.write_text(translate(tmp_path, "opengl"), encoding="utf-8")
    run_tool([validator, str(shader)])


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = require_tool("dxc")
    shader = tmp_path / "union.hlsl"
    shader.write_text(translate(tmp_path, "directx"), encoding="utf-8")
    run_tool([dxc, "-T", "cs_6_2", "-enable-16bit-types", "-E", "CSMain", str(shader)])


def test_spirv_carries_a_padded_vector(tmp_path):
    spirv_as = require_tool("spirv-as")
    spirv_val = require_tool("spirv-val")
    body = """
            r.data = vec4(1.0, 2.0, 3.0, 4.0);
            r.center = vec3(5.0, 6.0, 7.0);
            outp[0] = r.data.w + r.center.z + r.center[1];
"""
    source = tmp_path / "padded.cgl"
    source.write_text(
        PROGRAM.format(body=body)
        .replace("@union_member_layout(0, 4, 2) Pair p;", "")
        .replace("@union_member_layout(0, 4, 4) f16vec2 dm;", "")
        .replace("@union_member_layout(0, 4, 2) Mixed mixed;", "")
        .replace("@union_member_layout(0, 2, 2) uint16 low;", ""),
        encoding="utf-8",
    )
    asm = tmp_path / "padded.spvasm"
    asm.write_text(crosstl.translate(str(source), backend="vulkan"), encoding="utf-8")
    binary = tmp_path / "padded.spv"
    run_tool([spirv_as, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    run_tool([spirv_val, "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.parametrize("target", ["metal", "cuda", "hip", "cgl", "wgsl"])
def test_targets_with_unions_or_their_own_lowering_keep_the_members(target):
    program = parse()
    assert split_union_word_views(program, target) == 0
    assert spelling(members(program, "U")["dm"].member_type) == "f16vec2"


def test_supported_word_views_are_left_alone():
    source = """shader P {
        @union_layout(8, 8, little_endian, metal)
        struct Bits {
            @union_member_layout(0, 8, 8) uvec2 words;
            @union_member_layout(0, 8, 4) u8vec4[2] bytes;
            @union_member_layout(0, 8, 2) float16[4] halves;
        }
        compute { void main() { Bits b; b.words = uvec2(1u, 2u); } }
    }"""
    program = Parser(Lexer(source).get_tokens()).parse()
    assert split_union_word_views(program, "opengl") == 0


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ("uint* p = &u.dm.x;", "no address"),
        ("u.dm.x++;", "no update"),
        ("u.dm += u.dm;", "whole-member write"),
        ("u.dm.xy = u.dm;", "swizzle '.xy'"),
        ("outp[0] = float(u.p.z);", "not a field"),
        ("outp[0] = (u.dm = u.dm).x;", "whole-member write"),
    ],
    ids=["address", "increment", "compound", "swizzle-write", "field", "nested-write"],
)
def test_accesses_without_an_exact_lane_form_are_refused(tmp_path, body, detail):
    with pytest.raises(UnionWordViewError) as error:
        translate(tmp_path, "opengl", body)
    assert detail in str(error.value)
    assert error.value.code.endswith("union-word-view-unsupported")


def test_an_initializer_of_a_split_first_member_is_refused(tmp_path):
    source = """shader P {
        @union_layout(4, 4, little_endian, metal)
        struct H {
            @union_member_layout(0, 4, 4) f16vec2 dm;
            @union_member_layout(0, 4, 4) uint w;
        }
        compute {
            buffer uint* outp;
            void main() { H h = H(f16vec2(1.0, 2.0)); outp[0] = h.w; }
        }
    }"""
    path = tmp_path / "init.cgl"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(UnionWordViewError, match="first member"):
        crosstl.translate(str(path), backend="directx")


def test_a_member_name_another_structure_shares_needs_a_known_object_type(tmp_path):
    source = """shader P {
        struct Plain { f16vec2 dm; }
        @union_layout(4, 4, little_endian, metal)
        struct H {
            @union_member_layout(0, 4, 4) uint w;
            @union_member_layout(0, 4, 4) f16vec2 dm;
        }
        Plain make() { Plain p; p.dm = f16vec2(1.0, 2.0); return p; }
        compute {
            buffer float* outp;
            void main() {
                H h;
                h.w = 0u;
                Plain plain = make();
                outp[0] = float(plain.dm.x) + float(h.dm.y);
                outp[1] = float(make().dm.x);
            }
        }
    }"""
    path = tmp_path / "shared.cgl"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(UnionWordViewError, match="cannot be determined"):
        crosstl.translate(str(path), backend="opengl")
    # With every object type known, only the union's member is split.
    path.write_text(
        source.replace("outp[1] = float(make().dm.x);", ""), encoding="utf-8"
    )
    generated = crosstl.translate(str(path), backend="opengl", format_output=False)
    assert "plain.dm.x" in generated
    assert "crossgl_union_get_float16_lane_w1(h.CrossGLUnionStorage, 1)" in generated
