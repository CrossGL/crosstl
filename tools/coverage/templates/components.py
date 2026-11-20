"""Vector-halves and numeric-components probes: the names OpenCL C has for vector components besides x, y, z and w.

OpenCL C names a vector's components by index, ``.s0`` to ``.sF`` or ``.S0`` to ``.SF``, and several at once as
``.s01``; and it names the halves of a vector ``.lo``, ``.hi``, ``.even`` and ``.odd``, where a half of a two-component
vector is a scalar (OpenCL C 3.0, "Vector Components"). Both forms read and assign. Each probe uses them on a
``float4``, so a translation can write them as the swizzles ``.xy``, ``.zw``, ``.xz`` and ``.yw`` that every target
has, and CrossGL's probes are those swizzles.

No other language has these names. They stand for the shuffles that the swizzle probes measure, so SPIR-V, whose
probes come from GLSL and HLSL, does not list them either (:func:`..constructs.features`).
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

NAMES = ("vector-halves", "numeric-components")

# The output scales each component it sums by its own power of 16.
_HALVES_SUM = (
    "h.y + e.x * 16.0 + s * 256.0 + v.x * 4096.0 + v.y * 65536.0 + v.z * 1048576.0"
)
_COMPONENTS_SUM = "a.y + b.x * 16.0 + v.{} * 256.0 + v.{} * 4096.0 + v.{} * 65536.0"

OPENCL = {
    "vector-halves": Template(
        "float4 v = (float4)(fin[i], 1.0f, 2.0f, 3.0f);\nfloat2 h = v.hi;\nfloat2 e = v.even;\n"
        "float s = v.odd.hi;\nv.lo = v.odd;\nv.even = (float2)((float)iin[i], 5.0f);\n"
        f"outp[i] = {c_text(_HALVES_SUM, 'uint')};"
    ),
    "numeric-components": Template(
        "float4 v = (float4)(fin[i], 1.0f, 2.0f, 3.0f);\nfloat2 a = v.s01;\nfloat2 b = v.s32;\n"
        "v.s13 = (float2)((float)iin[i], 5.0f);\nv.S2 = v.s0 + 4.0f;\n"
        f"outp[i] = {c_text(_COMPONENTS_SUM.format('s1', 'S2', 's3'), 'uint')};"
    ),
}

CROSSGL = {
    "vector-halves": Template(
        "$vec4 v = $vec4(fin[i], 1.0, 2.0, 3.0);\n$vec2 h = v.zw;\n$vec2 e = v.xz;\n"
        "float s = v.yw.y;\nv.xy = v.yw;\nv.xz = $vec2(float(iin[i]), 5.0);\n"
        f"outp[i] = {_HALVES_SUM};"
    ),
    "numeric-components": Template(
        "$vec4 v = $vec4(fin[i], 1.0, 2.0, 3.0);\n$vec2 a = v.xy;\n$vec2 b = v.wz;\n"
        "v.yw = $vec2(float(iin[i]), 5.0);\nv.z = v.x + 4.0;\n"
        f"outp[i] = {_COMPONENTS_SUM.format('y', 'z', 'w')};"
    ),
}

ELSEWHERE = dict.fromkeys(NAMES, ABSENT)
