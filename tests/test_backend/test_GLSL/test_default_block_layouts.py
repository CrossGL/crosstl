"""Default block layouts and block matrix orders of GLSL uniform and shader storage blocks (GLSL 4.60 4.4.5)."""

import re

import pytest

from crosstl._crosstl import translate
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def generate_crossgl(code: str, shader_type: str = "fragment") -> str:
    tokens = GLSLLexer(code).tokenize()
    ast = GLSLParser(tokens, shader_type).parse()
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def member_line(crossgl: str, name: str) -> str:
    lines = [
        line.strip()
        for line in crossgl.splitlines()
        if re.search(rf"\b{name}\b[^;]*;$", line.strip()) and "=" not in line
    ]
    assert lines, f"no declaration of {name} in:\n{crossgl}"
    return lines[0]


def test_default_uniform_layout_reaches_later_blocks_and_leaves_no_statement():
    crossgl = generate_crossgl("""
        #version 330 core
        layout(std140, row_major) uniform;
        uniform Blk { mat4 m; vec4 v; } b;
        out vec4 color;
        void main() { color = b.m * b.v; }
        """)
    assert "uniform;" not in crossgl
    assert "@buffer_layout(std140)" in crossgl
    assert member_line(crossgl, "m") == "mat4 m @row_major;"
    assert member_line(crossgl, "v") == "vec4 v;"


def test_block_matrix_order_reaches_members_that_name_none():
    crossgl = generate_crossgl("""
        #version 450
        layout(std140, row_major, binding = 1) uniform U {
            mat4 a;
            layout(column_major) mat4 b;
            mat3x2 c[2];
            vec4 d;
        } u;
        out vec4 color;
        void main() { color = u.a * u.b * u.d + vec4(u.c[0][0], 0.0, 0.0); }
        """)
    assert member_line(crossgl, "a") == "mat4 a @row_major;"
    assert member_line(crossgl, "b") == "mat4 b @column_major;"
    assert member_line(crossgl, "c") == "mat3x2 c[2] @row_major;"
    assert member_line(crossgl, "d") == "vec4 d;"


def test_struct_members_inherit_the_block_matrix_order():
    crossgl = generate_crossgl("""
        #version 450
        struct Light { mat4 transform; vec4 color; };
        layout(row_major) uniform;
        layout(std140, binding = 0) uniform U { Light light; float scale; } u;
        out vec4 color;
        void main() { color = u.light.transform * u.light.color * u.scale; }
        """)
    assert member_line(crossgl, "light") == "Light light @row_major;"
    assert member_line(crossgl, "scale") == "float scale;"


def test_later_defaults_and_block_qualifiers_override_earlier_ones():
    crossgl = generate_crossgl("""
        #version 450
        layout(std140, row_major) uniform;
        layout(column_major) uniform;
        layout(binding = 0) uniform A { mat4 ma; } a;
        layout(row_major) uniform;
        layout(column_major, binding = 1) uniform B { mat4 mb; } b;
        layout(binding = 2) uniform C { mat4 mc; } c;
        out vec4 color;
        void main() { color = (a.ma * b.mb * c.mc)[0]; }
        """)
    assert member_line(crossgl, "ma") == "mat4 ma;"
    assert member_line(crossgl, "mb") == "mat4 mb;"
    assert member_line(crossgl, "mc") == "mat4 mc @row_major;"


def test_default_buffer_layout_sets_the_packing_of_later_storage_blocks():
    crossgl = generate_crossgl(
        """
        #version 450
        layout(local_size_x = 1) in;
        layout(binding = 0) buffer Before { float values[4]; } before;
        layout(std140) buffer;
        layout(binding = 1) buffer After { float values[4]; } after;
        void main() { after.values[0] = before.values[1]; }
        """,
        "compute",
    )
    assert "buffer;" not in crossgl
    assert re.search(r"Before before @glsl_buffer_block\(std430\)", crossgl)
    assert re.search(r"After after @glsl_buffer_block\(std140\)", crossgl)


def test_default_uniform_layout_does_not_reach_loose_uniforms():
    crossgl = generate_crossgl("""
        #version 330 core
        layout(row_major) uniform;
        uniform mat4 transform;
        out vec4 color;
        void main() { color = transform[0]; }
        """)
    assert "@row_major" not in crossgl
    assert "uniform;" not in crossgl


@pytest.mark.parametrize(
    "declaration",
    ["layout(binding = 0) uniform;", "layout(std140, offset = 4) buffer;"],
)
def test_default_block_layouts_refuse_qualifiers_no_default_may_hold(declaration):
    with pytest.raises(SyntaxError, match="cannot be a default"):
        generate_crossgl(f"""
            #version 450
            {declaration}
            out vec4 color;
            void main() {{ color = vec4(1.0); }}
            """)


def test_default_row_major_reaches_every_target(tmp_path):
    source = tmp_path / "rows.frag"
    source.write_text("""
        #version 450
        layout(std140, row_major) uniform;
        layout(binding = 0) uniform Blk { mat4 m; vec4 v; } b;
        layout(location = 0) out vec4 color;
        void main() { color = b.m * b.v; }
        """)
    crossgl = tmp_path / "rows.cgl"
    crossgl.write_text(translate(str(source), backend="cgl"))

    assert "row_major float4x4 m;" in translate(str(crossgl), backend="directx")
    assert "layout(row_major) mat4 m;" in translate(str(crossgl), backend="opengl")
    assert "transpose(" in translate(str(crossgl), backend="metal")
    # Without SPIRV-Tools the ids stay numeric, so match the block id loosely.
    spirv = translate(str(crossgl), backend="vulkan")
    assert re.search(r"OpMemberDecorate %\w+ 0 RowMajor", spirv)


def test_block_matrix_order_reaches_storage_block_members():
    crossgl = generate_crossgl(
        """
        #version 450
        layout(local_size_x = 1) in;
        struct Light { mat2 n; vec4 c; };
        layout(std430, row_major, binding = 0) buffer Src {
            mat3 m;
            layout(column_major) mat4 kept;
            Light light;
            vec4 v;
        } src;
        layout(std430, binding = 1) buffer Dst { vec4 o; } dst;
        void main() { dst.o = vec4(src.m[0], src.kept[1][2]) + src.v + src.light.c; }
        """,
        "compute",
    )
    assert member_line(crossgl, "m") == "mat3 m @row_major;"
    assert member_line(crossgl, "kept") == "mat4 kept @column_major;"
    assert member_line(crossgl, "light") == "Light light @row_major;"
    assert member_line(crossgl, "v") == "vec4 v;"


def test_default_buffer_matrix_order_reaches_storage_block_members():
    crossgl = generate_crossgl(
        """
        #version 450
        layout(local_size_x = 1) in;
        layout(row_major) buffer;
        layout(std430, binding = 0) buffer Src { mat3 m; } src;
        layout(std430, binding = 1) buffer Dst { vec4 o; } dst;
        void main() { dst.o = vec4(src.m[0], 1.0); }
        """,
        "compute",
    )
    assert "buffer;" not in crossgl
    assert member_line(crossgl, "m") == "mat3 m @row_major;"


@pytest.mark.parametrize(
    "backend, stored_row_major",
    [
        # The storage block's member is declared as its transpose, so a read of a column transposes it back.
        ("opengl", r"transpose\(src\.m\)\[0\]"),
        ("vulkan", r"OpTranspose"),
        ("metal", r"transpose\("),
        # Byte-address buffers: the transposed member is loaded column by column, then transposed back.
        ("directx", r"transpose\(transpose\("),
    ],
)
def test_storage_block_matrix_order_reaches_every_target(
    tmp_path, backend, stored_row_major
):
    source = tmp_path / "rows.comp"
    source.write_text("""
        #version 450
        layout(local_size_x = 1) in;
        layout(std430, row_major, binding = 0) buffer Src { mat3 m; } src;
        layout(std430, binding = 1) buffer Dst { vec4 o[2]; } dst;
        void main() { dst.o[0] = vec4(src.m[0], src.m[1][2]); dst.o[1] = vec4(src.m[2], 0.0); }
        """)
    assert re.search(stored_row_major, translate(str(source), backend=backend))


@pytest.mark.parametrize("block", ["uniform", "buffer"])
def test_block_matrix_order_skips_structs_without_matrices(block):
    crossgl = generate_crossgl(
        f"""
        #version 450
        layout(local_size_x = 1) in;
        struct Plain {{ vec4 c; }};
        struct Inner {{ mat2 n; }};
        struct Outer {{ Inner inner; float f; }};
        layout(row_major, binding = 0) {block} Src {{
            Plain plain;
            Outer outer;
        }} src;
        layout(std430, binding = 1) buffer Dst {{ vec4 o; }} dst;
        void main() {{ dst.o = src.plain.c + vec4(src.outer.inner.n[0], src.outer.f, 0.0); }}
        """,
        "compute",
    )
    # A matrix order changes no scalar or vector, so a struct of them keeps none.
    assert member_line(crossgl, "plain") == "Plain plain;"
    # A struct reaches the matrices of the structs it nests.
    assert member_line(crossgl, "outer") == "Outer outer @row_major;"
