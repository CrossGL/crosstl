"""Row-major arrays of non-square matrices in SPIR-V uniform blocks have their own array types.

A row-major ``mat3x2`` in a std140 block is two 16-byte rows, a column-major one three 16-byte columns (GLSL 4.60
section 7.6.2.2, rules 5 and 7), so arrays of them have element strides of 32 and 48 bytes. SPIR-V decorates the
array type with ``ArrayStride``, so the generator declares a second array type for the row-major member, as glslang
does, instead of refusing the member.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import crosstl
from tests.spirv_friendly_names import friendly_spirv

OUTPUT = """
    struct O {
        float o[8];
    };
    O outp @glsl_buffer_block(std430) @binding(1);
"""

MIXED_ORDERS = (
    """shader main {
    cbuffer U @binding(0) @buffer_layout(std140) {
        mat3x2 k[2] @row_major;
        vec4 c;
        mat3x2 j[2];
    };
"""
    + OUTPUT
    + """
    compute {
    layout(local_size_x = 1) in;
    void main() {
        outp.o[0] = k[1][2][1];
        outp.o[1] = (k[0] * vec3(1.0, 10.0, 100.0)).y;
        mat3x2 copy[2] = k;
        outp.o[2] = copy[1][0][0];
        outp.o[3] = c.y;
        outp.o[4] = j[1][2][1];
        outp.o[5] = (j[0] * vec3(1.0, 10.0, 100.0)).x;
    }
    }
}
"""
)

NESTED_ARRAYS = (
    """shader main {
    cbuffer U @binding(0) @buffer_layout(std140) {
        mat2x3 q[2][2] @row_major;
        mat2x3 p[2];
    };
"""
    + OUTPUT
    + """
    compute {
    layout(local_size_x = 1) in;
    void main() {
        outp.o[0] = q[1][0][1][2];
        mat2x3 inner[2] = q[1];
        outp.o[1] = inner[1][0][1];
        outp.o[2] = p[1][1][2];
    }
    }
}
"""
)

SQUARE = (
    """shader main {
    cbuffer U @binding(0) @buffer_layout(std140) {
        mat3 a[2] @row_major;
        mat3 b[2];
    };
"""
    + OUTPUT
    + """
    compute {
    layout(local_size_x = 1) in;
    void main() {
        outp.o[0] = a[1][2][1] + b[0][1][2];
    }
    }
}
"""
)

TWO_BLOCKS = (
    """shader main {
    cbuffer A @binding(0) @buffer_layout(std140) {
        mat2x4 x[3] @row_major;
    };
    cbuffer B @binding(2) @buffer_layout(std140) {
        vec4 pad;
        mat2x4 y[3] @row_major;
        mat2x4 z[3];
    };
"""
    + OUTPUT
    + """
    compute {
    layout(local_size_x = 1) in;
    void main() {
        outp.o[0] = x[2][1][3] + y[1][0][2] + z[0][1][1];
    }
    }
}
"""
)


def _assembly(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )


def _validate(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _members(assembly, struct):
    """Each member of ``struct``: its type id, the ArrayStride of every array level, and its order decoration."""
    types = re.search(rf"%{struct} = OpTypeStruct ((?:%\w+ ?)+)", assembly)
    arrays = dict(re.findall(r"(%\w+) = OpTypeArray (%\w+) %\w+", assembly))
    strides = {
        target: int(stride)
        for target, stride in re.findall(
            r"OpDecorate (%\w+) ArrayStride (\d+)", assembly
        )
    }
    members = []
    for index, type_id in enumerate(types.group(1).split()):
        levels = []
        level = type_id
        while level in arrays:
            levels.append(strides.get(level))
            level = arrays[level]
        order = re.findall(
            rf"OpMemberDecorate %{struct} {index} (RowMajor|ColMajor)", assembly
        )
        members.append((type_id, levels, order))
    return members


def test_row_major_array_has_its_own_array_type(tmp_path):
    assembly = _assembly(tmp_path, MIXED_ORDERS)
    (k_type, k_strides, k_order), _c, (j_type, j_strides, j_order) = _members(
        assembly, "U"
    )
    assert k_type != j_type
    assert (k_strides, k_order) == ([32], ["RowMajor"])
    assert (j_strides, j_order) == ([48], ["ColMajor"])
    assert re.search(r"OpMemberDecorate %U 2 Offset 80\b", assembly)
    _validate(tmp_path, assembly)


def test_every_array_level_takes_the_row_major_stride(tmp_path):
    assembly = _assembly(tmp_path, NESTED_ARRAYS)
    (q_type, q_strides, q_order), (p_type, p_strides, p_order) = _members(assembly, "U")
    assert (q_strides, q_order) == ([96, 48], ["RowMajor"])
    assert (p_strides, p_order) == ([32], ["ColMajor"])
    assert re.search(r"OpMemberDecorate %U 1 Offset 192\b", assembly)
    _validate(tmp_path, assembly)


def test_square_matrix_arrays_share_one_array_type(tmp_path):
    assembly = _assembly(tmp_path, SQUARE)
    (a_type, a_strides, a_order), (b_type, b_strides, b_order) = _members(assembly, "U")
    assert a_type == b_type
    assert a_strides == b_strides == [48]
    assert (a_order, b_order) == (["RowMajor"], ["ColMajor"])
    _validate(tmp_path, assembly)


def test_blocks_share_the_row_major_array_type(tmp_path):
    assembly = _assembly(tmp_path, TWO_BLOCKS)
    ((x_type, x_strides, x_order),) = _members(assembly, "A")
    _pad, (y_type, y_strides, y_order), (z_type, z_strides, z_order) = _members(
        assembly, "B"
    )
    assert x_type == y_type != z_type
    assert x_strides == y_strides == [64]
    assert z_strides == [32]
    assert x_order == y_order == ["RowMajor"]
    assert z_order == ["ColMajor"]
    _validate(tmp_path, assembly)
