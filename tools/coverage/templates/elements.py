"""Matrix-buffer, struct-buffer-packing, struct-array-buffer, nested-struct-buffer, integer-buffer and
unsigned-buffer probes: output buffers whose elements are matrices, structs with a vector, an array or a struct
member, or integers, in every language.

Each probe writes the elements in place and reads them back, so a translation has to keep the element's layout and
its member offsets for the host to read what the native probe writes. The matrix probe fills each element by columns,
assigns its second column and changes one component of the first. HLSL fills and indexes a matrix by rows, so its
probe writes the transposed constructor and walks a column one row at a time; a ``float2x2`` in a structured
buffer is stored by columns, as GLSL's ``mat2`` is.

``Packed`` puts a float after a three-component vector, at offset 12 in std430, a structured buffer, WGSL and C.
Metal aligns ``float3`` to 16 bytes, so its probe declares a ``packed_float3``; OpenCL C sizes ``float3`` as 16, so its
struct is a layout of its own. WebGL 2 has no storage buffers, and CUDA, HIP, OpenCL C and Mojo have no matrices.
"""

from __future__ import annotations

from .base import ABSENT, BINS, NESTED_STRUCTS, Template
from .statements import c_text

NAMES = (
    "matrix-buffer",
    "struct-buffer-packing",
    "struct-array-buffer",
    "nested-struct-buffer",
    "integer-buffer",
    "unsigned-buffer",
)

_MATRIX = (
    "outp[i] = $mat2(fin[i], 1.0, 2.0, fin[i] * 0.5);\noutp[i][1] = outp[i][0] * 2.0 + outp[i][1];\n"
    "outp[i][0][1] += 0.25;"
)
_PACKED = (
    "outp[i].a = $vec3(fin[i], 1.0, fin[i] * 2.0);\noutp[i].b = outp[i].a.x + outp[i].a.z;\n"
    "outp[i].a.y += outp[i].b * 0.5;"
)
_BINS = (
    "int k = iin[i] & 1;\noutp[i].n = iin[i] + 1;\noutp[i].v[0] = fin[i];\noutp[i].v[1] = fin[i] * 2.0;\n"
    "outp[i].v[2] = float(outp[i].n);\noutp[i].v[k] += outp[i].v[2 - k];"
)
_FRAME = (
    "outp[i].p.a = fin[i];\noutp[i].p.b = fin[i] * 3.0;\noutp[i].w = outp[i].p.a + outp[i].p.b;\n"
    "Pair q = outp[i].p;\noutp[i].p.a = q.b - 1.0;"
)
_INTEGER = "outp[i] = iin[i] * 3 - 7;\noutp[i] = outp[i] * 5 + iin[i];"
_UNSIGNED = "outp[i] = uint(iin[i]) * 3u;\noutp[i] -= 5u;\noutp[i] = outp[i] >> 1u;"

_PACKED_STRUCT = "struct Packed {\n    $vec3 a;\n    float b;\n};"


def _family(packed: str = _PACKED_STRUCT, matrix: str = _MATRIX) -> dict[str, Template]:
    """The probes of a language that writes them as GLSL does, with ``packed`` the declaration of ``Packed``."""
    return {
        "matrix-buffer": Template(matrix, output="$mat2"),
        "struct-buffer-packing": Template(_PACKED, prelude=packed, output="Packed"),
        "struct-array-buffer": Template(_BINS, prelude=BINS, output="Bins"),
        "nested-struct-buffer": Template(
            _FRAME, prelude=NESTED_STRUCTS, output="Frame"
        ),
        "integer-buffer": Template(_INTEGER, output="$int"),
        "unsigned-buffer": Template(_UNSIGNED, output="$uint"),
    }


GLSL = _family()
CROSSGL = _family()
METAL = _family("struct Packed {\n    packed_float3 a;\n    float b;\n};")

# CrossGL's column k is the k-th element of each HLSL row, so [r][1] walks the second column.
FAMILY = _family(
    matrix=(
        "outp[i] = float2x2(fin[i], 2.0, 1.0, fin[i] * 0.5);\n"
        "outp[i][0][1] = outp[i][0][0] * 2.0 + outp[i][0][1];\n"
        "outp[i][1][1] = outp[i][1][0] * 2.0 + outp[i][1][1];\noutp[i][1][0] += 0.25;"
    )
)

WEBGL = dict.fromkeys(NAMES, ABSENT)

_WGSL_PACKED = "struct Packed {\n    a: vec3<f32>,\n    b: f32,\n}"
_WGSL_BINS = "struct Bins {\n    v: array<f32, 3>,\n    n: i32,\n}"
_WGSL_FRAME = "struct Pair {\n    a: f32,\n    b: f32,\n}\n\nstruct Frame {\n    p: Pair,\n    w: f32,\n}"

WGSL = {
    "matrix-buffer": Template(_MATRIX, output="$mat2"),
    "struct-buffer-packing": Template(_PACKED, _WGSL_PACKED, output="Packed"),
    "struct-array-buffer": Template(
        "let k = iin[i] & 1;\noutp[i].n = iin[i] + 1;\noutp[i].v[0] = fin[i];\noutp[i].v[1] = fin[i] * 2.0;\n"
        "outp[i].v[2] = f32(outp[i].n);\noutp[i].v[k] += outp[i].v[2 - k];",
        _WGSL_BINS,
        output="Bins",
    ),
    "nested-struct-buffer": Template(
        _FRAME.replace("Pair q", "let q"), _WGSL_FRAME, output="Frame"
    ),
    "integer-buffer": Template(_INTEGER, output="$int"),
    "unsigned-buffer": Template(_UNSIGNED.replace("uint(", "u32("), output="$uint"),
}


def _c(
    construct: str, unsigned: str, structs: tuple[str, str, str]
) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share, in C with casts: ``construct`` opens a three-component vector and
    ``structs`` declares ``Packed``, ``Bins`` and ``Frame``."""

    def text(source: str) -> str:
        return c_text(source.replace("$vec3(", construct), unsigned)

    packed, bins, frame = structs
    return {
        "matrix-buffer": ABSENT,
        "struct-buffer-packing": Template(text(_PACKED), packed, output="Packed"),
        "struct-array-buffer": Template(text(_BINS), bins, output="Bins"),
        "nested-struct-buffer": Template(text(_FRAME), frame, output="Frame"),
        "integer-buffer": Template(_INTEGER, output="$int"),
        "unsigned-buffer": Template(text(_UNSIGNED), output="$uint"),
    }


CUDA = _c(
    "make_float3(",
    "unsigned int",
    ("struct Packed {\n    float3 a;\n    float b;\n};", BINS, NESTED_STRUCTS),
)

OPENCL = _c(
    "(float3)(",
    "uint",
    (
        "typedef struct {\n    float3 a;\n    float b;\n} Packed;",
        "typedef struct {\n    float v[3];\n    int n;\n} Bins;",
        "typedef struct {\n    float a;\n    float b;\n} Pair;\n\n"
        "typedef struct {\n    Pair p;\n    float w;\n} Frame;",
    ),
)

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_PACKED = f"{_RUST_DERIVE}pub struct Packed {{\n    a: Vec3,\n    b: f32,\n}}"
_RUST_BINS = f"{_RUST_DERIVE}pub struct Bins {{\n    v: [f32; 3],\n    n: i32,\n}}"
_RUST_FRAME = (
    f"{_RUST_DERIVE}pub struct Pair {{\n    a: f32,\n    b: f32,\n}}\n\n"
    f"{_RUST_DERIVE}pub struct Frame {{\n    p: Pair,\n    w: f32,\n}}"
)

RUST = {
    "matrix-buffer": Template(
        "outp[i] = $mat2::from_cols($vec2::new(fin[i], 1.0), $vec2::new(2.0, fin[i] * 0.5));\n"
        "outp[i].y_axis = outp[i].x_axis * 2.0 + outp[i].y_axis;\noutp[i].x_axis.y += 0.25;",
        output="$mat2",
    ),
    "struct-buffer-packing": Template(
        _PACKED.replace("$vec3(", "$vec3::new("), _RUST_PACKED, output="Packed"
    ),
    "struct-array-buffer": Template(
        "let k = (iin[i] & 1) as usize;\noutp[i].n = iin[i] + 1;\noutp[i].v[0] = fin[i];\n"
        "outp[i].v[1] = fin[i] * 2.0;\noutp[i].v[2] = outp[i].n as f32;\noutp[i].v[k] += outp[i].v[2 - k];",
        _RUST_BINS,
        output="Bins",
    ),
    "nested-struct-buffer": Template(
        _FRAME.replace("Pair q", "let q"), _RUST_FRAME, output="Frame"
    ),
    "integer-buffer": Template(_INTEGER, output="$int"),
    "unsigned-buffer": Template(
        "outp[i] = iin[i] as u32 * 3;\noutp[i] = outp[i].wrapping_sub(5);\noutp[i] >>= 1;",
        output="$uint",
    ),
}

_MOJO_PACKED = (
    "@fieldwise_init\nstruct Packed(TrivialRegisterPassable):\n"
    "    var a: SIMD[DType.float32, 3]\n    var b: Float32"
)
_MOJO_FRAME = (
    "@fieldwise_init\nstruct Pair(TrivialRegisterPassable):\n    var a: Float32\n    var b: Float32\n\n\n"
    "@fieldwise_init\nstruct Frame(TrivialRegisterPassable):\n    var p: Pair\n    var w: Float32"
)
# An Array member is not register-passable, so Bins is only copyable.
_MOJO_BINS = "@fieldwise_init\nstruct Bins(Copyable, Movable):\n    var v: Array[Float32, 3]\n    var n: Int32"

MOJO = {
    "matrix-buffer": ABSENT,
    "struct-buffer-packing": Template(
        "outp[i].a = SIMD[DType.float32, 3](fin[i], 1.0, fin[i] * 2.0)\n"
        "outp[i].b = outp[i].a[0] + outp[i].a[2]\noutp[i].a[1] += outp[i].b * 0.5",
        _MOJO_PACKED,
        output="Packed",
    ),
    "nested-struct-buffer": Template(
        _FRAME.replace("Pair q", "var q").replace(";", ""),
        _MOJO_FRAME,
        output="Frame",
    ),
    "struct-array-buffer": Template(
        "var k = Int(iin[i] & 1)\noutp[i].n = iin[i] + 1\noutp[i].v[0] = fin[i]\noutp[i].v[1] = fin[i] * 2.0\n"
        "outp[i].v[2] = Float32(outp[i].n)\noutp[i].v[k] += outp[i].v[2 - k]",
        _MOJO_BINS,
        output="Bins",
    ),
    "integer-buffer": Template(_INTEGER.replace(";", ""), output="$int"),
    "unsigned-buffer": Template(
        "outp[i] = UInt32(iin[i]) * 3\noutp[i] -= 5\noutp[i] = outp[i] >> 1",
        output="$uint",
    ),
}
