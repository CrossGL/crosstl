"""A block-level ``@ row_major`` orders the matrices of its constant buffer or block instance."""

import re

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import BlockMatrixOrderError
from crosstl.translator.semantics.block_matrix_orders import (
    distribute_block_matrix_orders,
)

TARGETS = ("opengl", "directx", "metal", "vulkan", "slang")

STORAGE_BLOCK = """shader P {
    struct B {
        mat2x3 m%(m)s;
        vec4 c;
        mat3x2 k[2]%(k)s;
    };
    struct O {
        float o[8];
    };
    B b @row_major @glsl_buffer_block(std430) @binding(0);
    O outp @glsl_buffer_block(std430) @binding(1);
    compute {
    layout(local_size_x = 1) in;
    void main() {
        vec3 r = b.m * vec2(1.0, 10.0);
        outp.o[0] = r.x;
        outp.o[3] = b.m[1][0];
        outp.o[5] = b.c.y;
        outp.o[6] = b.k[1][2][1];
        outp.o[7] = (b.k[0] * vec3(1.0, 10.0, 100.0)).y;
    }
    }
}
"""

CONSTANT_BUFFER = """shader P {
    cbuffer U @binding(0) @row_major @buffer_layout(std140) {
        mat2x3 m%(m)s;
        vec4 c;
        mat3x2 k%(k)s;
    };
    struct O {
        float o[8];
    };
    O outp @glsl_buffer_block(std430) @binding(1);
    compute {
    layout(local_size_x = 1) in;
    void main() {
        vec3 r = m * vec2(1.0, 10.0);
        outp.o[0] = r.x;
        outp.o[3] = m[1][0];
        outp.o[5] = c.y;
        outp.o[7] = (k * vec3(1.0, 10.0, 100.0)).y;
    }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _members(struct):
    return {
        member.name: {attribute.name for attribute in member.attributes}
        for member in struct.members
    }


def _block_only(template):
    return template % {"m": "", "k": ""}


def _member_marked(template):
    return template % {"m": " @row_major", "k": " @row_major"}


@pytest.mark.parametrize("backend", TARGETS)
def test_storage_block_order_matches_member_orders(tmp_path, backend):
    block = _translate(tmp_path, _block_only(STORAGE_BLOCK), backend)
    members = _translate(tmp_path, _member_marked(STORAGE_BLOCK), backend)
    assert block == members


@pytest.mark.parametrize("backend", TARGETS)
def test_constant_buffer_order_matches_member_orders(tmp_path, backend):
    block = _translate(tmp_path, _block_only(CONSTANT_BUFFER), backend)
    members = _translate(tmp_path, _member_marked(CONSTANT_BUFFER), backend)
    assert block == members


def test_constant_buffer_members_are_declared_row_major(tmp_path):
    hlsl = _translate(tmp_path, _block_only(CONSTANT_BUFFER), "directx")
    glsl = _translate(tmp_path, _block_only(CONSTANT_BUFFER), "opengl")
    spirv = _translate(tmp_path, _block_only(CONSTANT_BUFFER), "vulkan")
    assert "row_major float3x2 m;" in hlsl
    assert "layout(row_major) mat2x3 m;" in glsl
    assert re.search(r"OpMemberDecorate %\w+ 0 RowMajor", spirv)


def test_only_unordered_members_holding_matrices_are_marked():
    program = _parse("""shader P {
        struct Inner {
            mat2 n;
        };
        struct Plain {
            vec4 v;
        };
        struct B {
            mat3 a;
            mat3 b @column_major;
            Inner i;
            Plain p;
            mat2 list[3];
            vec4 c;
        };
        B b @row_major @glsl_buffer_block(std430) @binding(0);
        compute {
        layout(local_size_x = 1) in;
        void main() {
        }
        }
    }""")
    assert distribute_block_matrix_orders(program, "opengl") == 3
    members = _members(next(s for s in program.structs if s.name == "B"))
    assert members == {
        "a": {"row_major"},
        "b": {"column_major"},
        "i": {"row_major"},
        "p": set(),
        "list": {"row_major"},
        "c": set(),
    }
    inner = next(s for s in program.structs if s.name == "Inner")
    assert _members(inner) == {"n": set()}


def test_crossgl_output_keeps_the_block_order(tmp_path):
    code = _translate(tmp_path, _block_only(CONSTANT_BUFFER), "cgl")
    assert "@row_major" in code
    assert "mat2x3 m;" in code


def test_struct_declared_with_two_orders_is_refused(tmp_path):
    source = _block_only(STORAGE_BLOCK).replace(
        "    O outp",
        "    B plain @glsl_buffer_block(std430) @binding(2);\n    O outp",
    )
    with pytest.raises(BlockMatrixOrderError, match="'b'.*'plain'"):
        _translate(tmp_path, source, "opengl")


def test_struct_with_ordered_members_is_not_refused(tmp_path):
    source = _member_marked(STORAGE_BLOCK).replace(
        "    O outp",
        "    B plain @glsl_buffer_block(std430) @binding(2);\n    O outp",
    )
    assert "void main()" in _translate(tmp_path, source, "opengl")
