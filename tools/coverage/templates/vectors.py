"""Vector-shift, vector-bitwise, vector-truncation and matrix-resize probes: integer vector operators with a vector or
a scalar on the right, and constructors that drop or add vector components and matrix rows and columns.

The operator probes shift and mask integer vectors by vectors and by scalars, which GLSL, HLSL, Metal, OpenCL C, Rust
and Mojo allow. WGSL shifts only by an unsigned vector of the same width and takes no scalar operand on a bitwise
operator, so its probes widen each scalar. No probe shifts a negative value left, which C++ leaves undefined.

GLSL builds a shorter vector from a longer one with a constructor, ``vec2(v)``, and a smaller or larger matrix from
another one, ``mat2(m)`` taking the upper-left corner and ``mat4(m)`` filling the rest from the identity. HLSL
truncates with a cast, ``(float2)v`` and ``(float2x2)m``, and writes a larger matrix out by rows; Slang casts only
matrices. Metal, WGSL and OpenCL C have neither, so their probes take swizzles and build matrices from columns,
and glam truncates with ``truncate`` and resizes with ``from_mat3`` and its relatives. CUDA and HIP vector types
have no operators or conversions, CUDA, HIP, OpenCL C and Mojo have no matrices, and Mojo has no three-component
vectors.
"""

from __future__ import annotations

import re

from .base import ABSENT, Template
from .statements import c_text

NAMES = ("vector-shift", "vector-bitwise", "vector-truncation", "matrix-resize")

# Each sum stays below 2^24, so every term reaches the output exactly.
_SHIFT_SUM = (
    "{f}(l{x} + l{y}) + {f}(r{x} - r{y} - 32) * 256.0 + {f}(v{x}) * 2048.0 + {f}(v{y} >> 29{u}) * 262144.0"
    " + {f}(w{x} & 255{u}) * 32768.0 + {f}(w{y} + 1{u}) * 4194304.0"
)
_BITWISE_SUM = (
    "{f}(c{x}) + {f}(c{y} + 16) * 16.0 + {f}(d{x} + d{y} * 16) * 256.0 + {f}(e{x} - 256{u}) * 32768.0"
    " + {f}(e{y} >> 12{u}) * 131072.0 + {f}(e{y} & 255{u}) * 2097152.0"
)
_TRUNCATION_SUM = "a.x + a.y * 4.0 + b.z * 16.0 + {f}(j.x + j.y * 16) * 64.0 + (c.x + c.y * 32.0) * 8192.0"
# Indexed m[column][row], as GLSL, Metal and WGSL index a matrix.
_RESIZE_SUM = (
    "a[0][0] + a[0][1] * 2.0 + a[1][0] * 4.0 + a[1][1] * 8.0 + b[1][1] * 16.0 + b[2][2] * 512.0 + b[0][2] * 1024.0"
    " + b[2][0] * 2048.0 + c[2][2] * 4096.0 + n[1][2] * 8192.0 + c[3][3] * 65536.0 + c[3][0] * 131072.0"
    " + c[0][3] * 262144.0"
)
_C = {"x": ".x", "y": ".y", "f": "float", "u": "u"}

_SHIFT = (
    "$ivec2 a = $ivec2(iin[i], -64);\n$ivec2 s = $ivec2(iin[i] & 3, 2);\n$ivec2 l = $ivec2(iin[i], 5) << s;\n"
    "$ivec2 r = a >> 1;\n$uvec2 u = $uvec2(uint(iin[i]) + 64u, 2147483648u);\n$uvec2 v = u >> $uvec2(s);\n"
    f"$uvec2 w = u << 3u;\noutp[i] = {_SHIFT_SUM.format(**_C)};"
)
_BITWISE = (
    "$ivec2 a = $ivec2(iin[i], -6);\n$ivec2 b = $ivec2(5, iin[i] * 3);\n$ivec2 c = (a & b) | (a ^ 9);\n"
    "$ivec2 d = ~a & 15;\n$uvec2 u = $uvec2(uint(iin[i]) | 256u, 61680u);\n"
    f"$uvec2 e = (u ^ $uvec2(255u, 255u)) & (u | 3u);\noutp[i] = {_BITWISE_SUM.format(**_C)};"
)
_ELEMENT = re.compile(r"\b(\w)\[(\d)\]\[(\d)\]")


def _transposed(text: str) -> str:
    """``m[c][r]`` as ``m[r][c]``, for HLSL, which indexes a matrix by rows."""
    return _ELEMENT.sub(r"\1[\3][\2]", text)


_M3 = "fin[i], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0"

# GLSL and CrossGL; WebGL 2 has the same constructors.
GLSL = {
    "vector-shift": Template(_SHIFT),
    "vector-bitwise": Template(_BITWISE),
    "vector-truncation": Template(
        "$vec4 v = $vec4(fin[i], 2.0, 3.0, 4.0);\n$vec2 a = $vec2(v);\n$vec3 b = $vec3(v);\n"
        "$ivec3 k = $ivec3(iin[i], 5, 6);\n$ivec2 j = $ivec2(k);\n$vec2 c = $vec2($ivec3(iin[i] * 2, 7, 8));\n"
        f"outp[i] = {_TRUNCATION_SUM.format(f='float')};"
    ),
    "matrix-resize": Template(
        f"$mat3 m = $mat3({_M3});\n$mat2 a = $mat2(m);\n$mat3 b = $mat3(a * 2.0);\n$mat4 c = $mat4(m);\n"
        f"$mat2x3 n = $mat2x3(m);\noutp[i] = {_RESIZE_SUM};"
    ),
}

# HLSL fills a matrix by rows and indexes it m[row][column], so its scalars and subscripts are transposed.
HLSL = {
    "vector-shift": Template(_SHIFT),
    "vector-bitwise": Template(_BITWISE),
    "vector-truncation": Template(
        "float4 v = float4(fin[i], 2.0, 3.0, 4.0);\nfloat2 a = (float2)v;\nfloat3 b = (float3)v;\n"
        "int3 k = int3(iin[i], 5, 6);\nint2 j = (int2)k;\nfloat2 c = (float2)int3(iin[i] * 2, 7, 8);\n"
        f"outp[i] = {_TRUNCATION_SUM.format(f='float')};"
    ),
    "matrix-resize": Template(
        "float3x3 m = float3x3(fin[i], 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0);\nfloat2x2 a = (float2x2)m;\n"
        "float2x2 h = a * 2.0;\nfloat3x3 b = float3x3(h[0][0], h[0][1], 0.0, h[1][0], h[1][1], 0.0, 0.0, 0.0, 1.0);\n"
        "float4x4 c = float4x4(float4(m[0], 0.0), float4(m[1], 0.0), float4(m[2], 0.0), float4(0.0, 0.0, 0.0, 1.0));\n"
        f"float3x2 n = (float3x2)m;\noutp[i] = {_transposed(_RESIZE_SUM)};"
    ),
}

_METAL_RESIZE = (
    f"float3x3 m = float3x3({_M3});\nfloat2x2 a = float2x2(m[0].xy, m[1].xy);\nfloat2x2 h = a * 2.0;\n"
    "float3x3 b = float3x3(float3(h[0], 0.0), float3(h[1], 0.0), float3(0.0, 0.0, 1.0));\n"
    "float4x4 c = float4x4(float4(m[0], 0.0), float4(m[1], 0.0), float4(m[2], 0.0), float4(0.0, 0.0, 0.0, 1.0));\n"
    f"float2x3 n = float2x3(m[0], m[1]);\noutp[i] = {_RESIZE_SUM};"
)
METAL = {
    "vector-shift": Template(_SHIFT),
    "vector-bitwise": Template(_BITWISE),
    "vector-truncation": Template(
        "float4 v = float4(fin[i], 2.0, 3.0, 4.0);\nfloat2 a = v.xy;\nfloat3 b = v.xyz;\nint3 k = int3(iin[i], 5, 6);\n"
        "int2 j = k.xy;\nfloat2 c = float2(int3(iin[i] * 2, 7, 8).xy);\n"
        f"outp[i] = {_TRUNCATION_SUM.format(f='float')};"
    ),
    "matrix-resize": Template(_METAL_RESIZE),
}
# Slang truncates no vector by cast, constructor or assignment, but casts matrices as HLSL does.
SLANG = {"vector-truncation": METAL["vector-truncation"]}

_W = {"x": ".x", "y": ".y", "f": "f32", "u": "u"}
WGSL = {
    "vector-shift": Template(
        "let a = vec2<i32>(iin[i], -64);\nlet s = vec2<i32>(iin[i] & 3, 2);\n"
        "let l = vec2<i32>(iin[i], 5) << vec2<u32>(s);\nlet r = a >> vec2<u32>(1u);\n"
        "let u = vec2<u32>(u32(iin[i]) + 64u, 2147483648u);\nlet v = u >> vec2<u32>(s);\n"
        f"let w = u << vec2<u32>(3u);\noutp[i] = {_SHIFT_SUM.format(**_W)};"
    ),
    "vector-bitwise": Template(
        "let a = vec2<i32>(iin[i], -6);\nlet b = vec2<i32>(5, iin[i] * 3);\n"
        "let c = (a & b) | (a ^ vec2<i32>(9));\nlet d = ~a & vec2<i32>(15);\n"
        "let u = vec2<u32>(u32(iin[i]) | 256u, 61680u);\n"
        f"let e = (u ^ vec2<u32>(255u)) & (u | vec2<u32>(3u));\noutp[i] = {_BITWISE_SUM.format(**_W)};"
    ),
    "vector-truncation": Template(
        "let v = vec4<f32>(fin[i], 2.0, 3.0, 4.0);\nlet a = v.xy;\nlet b = v.xyz;\nlet k = vec3<i32>(iin[i], 5, 6);\n"
        "let j = k.xy;\nlet c = vec2<f32>(vec3<i32>(iin[i] * 2, 7, 8).xy);\n"
        f"outp[i] = {_TRUNCATION_SUM.format(f='f32')};"
    ),
    "matrix-resize": Template(
        _METAL_RESIZE.replace("float2x2 ", "let ")
        .replace("float3x3 ", "let ")
        .replace("float4x4 ", "let ")
        .replace("float2x3 ", "let ")
        .replace("float2x2(", "mat2x2<f32>(")
        .replace("float3x3(", "mat3x3<f32>(")
        .replace("float4x4(", "mat4x4<f32>(")
        .replace("float2x3(", "mat2x3<f32>(")
        .replace("float3(", "vec3<f32>(")
        .replace("float4(", "vec4<f32>(")
    ),
}

OPENCL = {
    "vector-shift": Template(
        c_text(
            _SHIFT.replace("$uvec2(s)", "convert_uint2(s)")
            .replace("$ivec2(", "(int2)(")
            .replace("$uvec2(", "(uint2)(")
            .replace("$ivec2", "int2")
            .replace("$uvec2", "uint2"),
            "uint",
        )
    ),
    "vector-bitwise": Template(
        c_text(
            _BITWISE.replace("$ivec2(", "(int2)(")
            .replace("$uvec2(", "(uint2)(")
            .replace("$ivec2", "int2")
            .replace("$uvec2", "uint2"),
            "uint",
        )
    ),
    "vector-truncation": Template(
        c_text(
            "float4 v = (float4)(fin[i], 2.0, 3.0, 4.0);\nfloat2 a = v.xy;\nfloat3 b = v.xyz;\n"
            "int3 k = (int3)(iin[i], 5, 6);\nint2 j = k.xy;\n"
            "float2 c = convert_float2(((int3)(iin[i] * 2, 7, 8)).xy);\n"
            f"outp[i] = {_TRUNCATION_SUM.format(f='float')};",
            "uint",
        )
    ),
    "matrix-resize": ABSENT,
}

CUDA = dict.fromkeys(NAMES, ABSENT)

_AXES = ("x_axis", "y_axis", "z_axis", "w_axis")


def _axes(text: str) -> str:
    """``m[c][r]`` as glam names it, ``m.y_axis.z``; ``n`` is an array of columns, so ``n[1].z``."""

    def name(match: re.Match) -> str:
        matrix, column, row = match[1], int(match[2]), "xyzw"[int(match[3])]
        return (
            f"n[{column}].{row}" if matrix == "n" else f"{matrix}.{_AXES[column]}.{row}"
        )

    return _ELEMENT.sub(name, text)


RUST = {
    "vector-shift": Template(
        "let a = IVec2::new(iin[i], -64);\nlet s = IVec2::new(iin[i] & 3, 2);\nlet l = IVec2::new(iin[i], 5) << s;\n"
        "let r = a >> 1i32;\nlet u = UVec2::new(iin[i] as u32 + 64, 0x8000_0000);\nlet v = u >> s.as_uvec2();\n"
        "let w = u << 3u32;\noutp[i] = (l.x + l.y) as f32 + (r.x - r.y - 32) as f32 * 256.0 + v.x as f32 * 2048.0"
        " + (v.y >> 29) as f32 * 262144.0 + (w.x & 255) as f32 * 32768.0 + (w.y + 1) as f32 * 4194304.0;"
    ),
    "vector-bitwise": Template(
        "let a = IVec2::new(iin[i], -6);\nlet b = IVec2::new(5, iin[i] * 3);\nlet c = (a & b) | (a ^ 9);\n"
        "let d = !a & 15;\nlet u = UVec2::new(iin[i] as u32 | 256, 0xF0F0);\n"
        "let e = (u ^ UVec2::splat(255)) & (u | 3);\noutp[i] = c.x as f32 + (c.y + 16) as f32 * 16.0"
        " + (d.x + d.y * 16) as f32 * 256.0 + (e.x - 256) as f32 * 32768.0 + (e.y >> 12) as f32 * 131072.0"
        " + (e.y & 255) as f32 * 2097152.0;"
    ),
    "vector-truncation": Template(
        "let v = Vec4::new(fin[i], 2.0, 3.0, 4.0);\nlet a = v.truncate().truncate();\nlet b = v.truncate();\n"
        "let k = IVec3::new(iin[i], 5, 6);\nlet j = k.truncate();\n"
        "let c = IVec3::new(iin[i] * 2, 7, 8).truncate().as_vec2();\n"
        "outp[i] = a.x + a.y * 4.0 + b.z * 16.0 + (j.x + j.y * 16) as f32 * 64.0 + (c.x + c.y * 32.0) * 8192.0;"
    ),
    # glam has no non-square matrices, so n is the pair of columns mat2x3(m) would hold.
    "matrix-resize": Template(
        "let m = Mat3::from_cols(Vec3::new(fin[i], 1.0, 2.0), Vec3::new(3.0, 4.0, 5.0), Vec3::new(6.0, 7.0, 8.0));\n"
        "let a = Mat2::from_mat3(m);\nlet b = Mat3::from_mat2(a * 2.0);\nlet c = Mat4::from_mat3(m);\n"
        f"let n = [m.x_axis, m.y_axis];\noutp[i] = {_axes(_RESIZE_SUM)};"
    ),
}

_M = {"x": "[0]", "y": "[1]", "f": "Float32", "u": ""}
MOJO = {
    "vector-shift": Template(
        "var a = $ivec2(iin[i], -64)\nvar s = $ivec2(iin[i] & 3, 2)\nvar l = $ivec2(iin[i], 5) << s\nvar r = a >> 1\n"
        "var u = $uvec2(UInt32(iin[i]) + 64, 2147483648)\nvar v = u >> s.cast[DType.uint32]()\nvar w = u << 3\n"
        f"outp[i] = {_SHIFT_SUM.format(**_M)}"
    ),
    "vector-bitwise": Template(
        "var a = $ivec2(iin[i], -6)\nvar b = $ivec2(5, iin[i] * 3)\nvar c = (a & b) | (a ^ 9)\nvar d = ~a & 15\n"
        "var u = $uvec2(UInt32(iin[i]) | 256, 61680)\nvar e = (u ^ $uvec2(255, 255)) & (u | 3)\n"
        f"outp[i] = {_BITWISE_SUM.format(**_M)}"
    ),
    "vector-truncation": ABSENT,
    "matrix-resize": ABSENT,
}
