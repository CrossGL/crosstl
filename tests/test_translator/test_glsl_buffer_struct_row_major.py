"""GLSL output keeps ``layout(row_major)`` on buffer-block members of structure type.

GLSL 4.60 section 4.4.5: ``row_major`` on a block member of structure type applies to the matrices of that
structure, at any depth. Uniform blocks kept the qualifier on such members but buffer blocks dropped it, so their
nested matrices were read column by column.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

BUFFER = """
shader main {
    struct Inner {
        mat2x3 n;
    };
    struct L {
        Inner inner;
        vec4 c;
        mat3x2 k;
    };
    struct I {
        L ls[2] @row_major;
        mat3x2 m @row_major;
        L plain;
        L cm @column_major;
    };
    struct O {
        vec4 o[3];
    };
    I src @glsl_buffer_block(std430) @binding(0) @readonly;
    O dst @glsl_buffer_block(std430) @binding(1);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            dst.o[0] = vec4(src.ls[1].inner.n[0], src.ls[0].inner.n[1].z);
            dst.o[1] = vec4(src.ls[1].k[2], src.m[1]);
            dst.o[2] = vec4(src.plain.k[0], src.cm.k[1]);
        }
    }
}
"""

GLSL_SOURCE = """#version 450
layout(local_size_x = 1) in;
struct Inner { mat2x3 n; };
struct L { Inner inner; vec4 c; mat3x2 k; };
layout(std430, row_major, binding = 0) readonly buffer I { L ls[2]; float f; } src;
layout(std430, binding = 1) buffer O { vec4 o[2]; } dst;
void main() {
    dst.o[0] = vec4(src.ls[1].inner.n[0], src.ls[0].inner.n[1].z);
    dst.o[1] = vec4(src.ls[1].k[2], src.ls[0].k[1].y, src.f);
}
"""


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _block(code, name):
    match = re.search(rf"buffer {name} \{{(.*?)\}}", _flat(code))
    assert match, code
    return match.group(1)


def _translate(tmp_path, source, suffix, backend="opengl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_struct_members_keep_row_major(tmp_path):
    block = _block(_translate(tmp_path, BUFFER, ".cgl"), "I")
    assert "layout(row_major) L ls[2];" in block, block


def test_matrix_members_are_transposed_once(tmp_path):
    # row_major_members declares the matrix member as its column-major transpose and drops its row_major.
    block = _block(_translate(tmp_path, BUFFER, ".cgl"), "I")
    assert " mat2x3 m;" in block, block
    assert "layout(row_major) mat" not in block, block


def test_other_members_get_no_order(tmp_path):
    block = _block(_translate(tmp_path, BUFFER, ".cgl"), "I")
    assert re.search(r"(?<!\)) L plain;", block), block
    assert re.search(r"(?<!\)) L cm;", block), block


def test_block_row_major_reaches_nested_struct_members_in_glsl_round_trip(tmp_path):
    block = _block(_translate(tmp_path, GLSL_SOURCE, ".comp"), "I")
    assert "layout(row_major) L ls[2];" in block, block
    assert re.search(r"(?<!\)) float f;", block), block


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator unavailable"
)
@pytest.mark.parametrize(
    "source,suffix",
    [(BUFFER, ".cgl"), (GLSL_SOURCE, ".comp")],
    ids=["crossgl", "glsl"],
)
def test_nested_matrices_are_row_major_in_spirv(tmp_path, source, suffix):
    output = tmp_path / "out.comp"
    output.write_text(_translate(tmp_path, source, suffix), encoding="utf-8")
    result = subprocess.run(
        [
            "glslangValidator",
            "-V",
            "-H",
            "-S",
            "comp",
            str(output),
            "-o",
            str(tmp_path / "out.spv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Inner.n and L.k, reached through ls, are decorated RowMajor (glslang gives each layout its own struct type).
    decorations = re.findall(
        r"MemberDecorate \d+\((Inner|L)\) (\d+) (RowMajor|ColMajor)", result.stdout
    )
    assert ("Inner", "0", "RowMajor") in decorations, result.stdout
    assert ("L", "2", "RowMajor") in decorations, result.stdout
