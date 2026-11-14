"""Rows of fixed multidimensional private arrays passed as private pointers.

A Metal helper that takes ``thread float*`` may be called with ``x[r]`` for a local ``float x[R][C]``: the row
decays to a pointer to its first element. DirectX and OpenGL lower private pointer parameters to ``inout`` arrays with
a base offset, and both refused such calls with ``missing-view-backing`` because only whole local arrays could back a
view. The row is now passed in place as an ``inout`` array of its own extent, and views into the row add their
offset. A side-effecting row index, a view past the end of the row and two views of one array in one call (whose
``inout`` copies would lose each other's writes) fail closed.
"""

import re

import pytest

from crosstl._crosstl import translate
from crosstl.translator.ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
)
from crosstl.translator.codegen.directx_codegen import (
    DirectXPrivatePointerParameterError,
)
from crosstl.translator.codegen.GLSL_codegen import OpenGLPrivatePointerParameterError
from crosstl.translator.codegen.private_array_rows import (
    array_row_shape,
    array_row_view,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

PRELUDE = """#include <metal_stdlib>
using namespace metal;

// The shape of a generated O(m^2) butterfly: read the row, then copy back.
void mix_pair(thread float* x) {
  float tmp[2];
  tmp[0] = x[0] + x[1];
  tmp[1] = x[0] - x[1];
  for (int i = 0; i < 2; i++) { x[i] = tmp[i]; }
}

void bump(thread float* x) { x[0] += 1.0f; }

void bump_two(thread float* x) { x[0] += 1.0f; x[1] += 1.0f; }

void pair(thread float* a, thread float* b) { a[0] = 1.0f; b[0] = 2.0f; }
"""

ROWS_KERNEL = """
kernel void rows(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
  float x[3][2];
  for (short r = 0; r < 3; r++) {
    for (short c = 0; c < 2; c++) { x[r][c] = in[gid * 6 + r * 2 + c]; }
  }
  for (short r = 0; r < 3; r++) { mix_pair(x[r]); }
  bump(&x[1][1]);
  bump(x[2] + 1);
  for (short r = 0; r < 3; r++) {
    for (short c = 0; c < 2; c++) { out[gid * 6 + r * 2 + c] = x[r][c]; }
  }
}
"""

CUBE_KERNEL = """
kernel void cube(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  float x[2][3][2];
  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 3; b++) { x[a][b][0] = float(a); x[a][b][1] = float(b); }
  }
  mix_pair(x[gid % 2][2]);
  out[gid] = x[gid % 2][2][0] + x[gid % 2][2][1];
}
"""

TARGETS = (
    ("directx", DirectXPrivatePointerParameterError),
    ("opengl", OpenGLPrivatePointerParameterError),
)


def _translate(tmp_path, body, backend):
    source = tmp_path / "rows.metal"
    source.write_text(PRELUDE + body, encoding="utf-8")
    return translate(
        str(source), backend=backend, source_backend="metal", format_output=False
    )


def _literal(value):
    return LiteralNode(value, PrimitiveType("int"))


def test_row_shape_lists_every_extent():
    cube = ArrayType(
        ArrayType(ArrayType(PrimitiveType("float"), _literal(4)), _literal(3)),
        _literal(2),
    )
    extents, element = array_row_shape(cube, lambda size: size.value)
    assert extents == (2, 3, 4) and element.name == "float"
    row = ArrayAccessNode(
        ArrayAccessNode(IdentifierNode("x"), IdentifierNode("a")), IdentifierNode("b")
    )
    root, extent, _element, indices = array_row_view(row, {"x": (extents, element)}.get)
    assert (root, extent, [index.name for index in indices]) == ("x", 4, ["a", "b"])
    # A plane of the cube decays to a pointer to rows, not to scalars.
    assert array_row_view(row.array, {"x": (extents, element)}.get) is None
    # A literal subscript outside its dimension selects no row.
    for outer, inner in ((2, 0), (0, 3), (-1, 0)):
        beyond = ArrayAccessNode(
            ArrayAccessNode(IdentifierNode("x"), _literal(outer)), _literal(inner)
        )
        assert array_row_view(beyond, {"x": (extents, element)}.get) is None
    inside = ArrayAccessNode(
        ArrayAccessNode(IdentifierNode("x"), _literal(1)), _literal(2)
    )
    assert array_row_view(inside, {"x": (extents, element)}.get)[1] == 4
    # One-dimensional and non-literal arrays have no rows.
    flat = ArrayType(PrimitiveType("float"), _literal(4))
    assert array_row_shape(flat, lambda size: size.value) is None
    assert array_row_shape(cube, lambda size: None) is None


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_rows_pass_in_place_as_inout_arrays(tmp_path, target):
    generated = _translate(tmp_path, ROWS_KERNEL, target)
    compact = " ".join(generated.split())
    assert "mix_pair(x[r], 0);" in compact
    # Views into a row add their element offset to the row's base.
    assert re.search(r"bump\(x\[1\], (?:int\()?1\)?\);", compact)
    assert re.search(r"bump\(x\[2\], (?:int\()?1\)?\);", compact)
    assert "inout float x[2], int x_base" in compact
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_innermost_rows_of_higher_rank_arrays_pass_in_place(tmp_path, target):
    generated = _translate(tmp_path, CUBE_KERNEL, target)
    assert "[2], 0);" in " ".join(generated.split())
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(("target", "error_type"), TARGETS)
@pytest.mark.parametrize(
    "body, reason",
    (
        (
            "float x[2][2]; bump_two(&x[gid % 2][1]); out[0] = x[0][0];",
            "view-out-of-bounds",
        ),
        (
            "float x[2][2]; pair(x[gid % 2], x[0]); out[0] = x[0][0];",
            "aliased-view-backing",
        ),
        (
            "float x[2][2]; pair(x[0], &x[0][1]); out[0] = x[0][0];",
            "aliased-view-backing",
        ),
        (
            "float x[4][2]; uint r = gid; mix_pair(x[r++ % 4]); out[0] = x[0][0];",
            "side-effecting-view-offset",
        ),
    ),
)
def test_unsafe_row_views_fail_closed(tmp_path, target, error_type, body, reason):
    kernel = (
        "kernel void run(device float* out [[buffer(0)]], "
        f"uint gid [[thread_position_in_grid]]) {{ {body} }}\n"
    )
    with pytest.raises(error_type) as error:
        _translate(tmp_path, kernel, target)
    assert error.value.reason == reason
