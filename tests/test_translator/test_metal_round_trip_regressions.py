"""Metal sources whose round trip through CrossGL stopped compiling (corpus regressions on ``main``).

Each case is a minimal form of a third-party kernel that translated to usable Metal before and to Metal the
compiler rejects after:

- a call ``max_value<uint>()`` beside the explicit specialization ``max_value<uint32_t>`` named an undeclared
  function, because the template machinery compared the spellings of one type;
- ``min(ids[i], (uint8_t)n)`` mixed a ``uchar`` buffer element with the ``uint`` register of an 8-bit scalar;
- a user ``float dot(float, float)`` took over every two-argument ``dot``, including ``dot`` of two ``float4``;
- ``half3 * f`` with a ``float`` ``f`` became ``float3(h) * f``, which cannot be stored back into a ``half3``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.scalar_type_identity import canonical_scalar_type_spelling

TEMPLATE_SPECIALIZATIONS = """#include <metal_stdlib>
using namespace metal;
template <typename T>
inline T max_value();
template <>
inline uint32_t max_value<uint32_t>() { return 0xFFFFFFFFu; }
template <>
inline uint8_t max_value<uint8_t>() { return 0xFF; }
kernel void k(device uint* o [[buffer(0)]], device uchar* p [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
  o[id] = (o[id] == max_value<uint>()) ? 1u : 0u;
  p[id] = max_value<uchar>();
}
"""

NARROW_BUILTIN_ARGUMENTS = """#include <metal_stdlib>
using namespace metal;
kernel void k(const device uint8_t* ids [[buffer(0)]], device uint8_t* o [[buffer(1)]],
              constant size_t& n [[buffer(2)]], const device short* s [[buffer(3)]],
              device short* so [[buffer(4)]], uint id [[thread_position_in_grid]]) {
  const uint8_t i = min(ids[id], (uint8_t)(n - 1));
  o[id] = i;
  short a = max(s[id], (short)(n));
  so[id] = clamp(s[id], a, (short)7);
}
"""

SCALAR_DOT_OVERLOAD = """#include <metal_stdlib>
using namespace metal;
static inline float dot(float x, float y) { return x * y; }
kernel void k(device const float4* a [[buffer(0)]], device const float* b [[buffer(1)]],
              device float* o [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  o[id] = dot(a[id], a[id]) + dot(b[id], b[id]);
}
"""

VECTOR_SCALAR_OPERANDS = """#include <metal_stdlib>
using namespace metal;
kernel void k(device int4* o [[buffer(0)]], device half4* h [[buffer(1)]],
              constant uint& d [[buffer(2)]], constant float& f [[buffer(3)]],
              uint id [[thread_position_in_grid]]) {
  int3 v = o[id].xyz;
  o[id].xyz = v / d;
  half3 c = h[id].xyz;
  c *= f;
  constexpr float bands = 2.0;
  c = saturate(fmod(c * bands, 1.0));
  h[id].xyz = c + f;
  o[id].w = int(all(v < d));
}
"""

CASES = {
    "template-specializations": TEMPLATE_SPECIALIZATIONS,
    "narrow-builtin-arguments": NARROW_BUILTIN_ARGUMENTS,
    "scalar-dot-overload": SCALAR_DOT_OVERLOAD,
    "vector-scalar-operands": VECTOR_SCALAR_OPERANDS,
}


def translate(tmp_path, source, backend="metal"):
    path = tmp_path / "kernel.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def flat(code):
    return " ".join(code.split())


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("uint32_t", "uint"),
        ("unsigned int", "uint"),
        ("metal::uint32_t", "uint"),
        ("device uint16_t*", "device ushort*"),
        ("vec<uint8_t, 2>", "vec<uchar, 2>"),
        ("size_t", "ulong"),
        ("unsigned  long", "ulong"),
        ("int64_t", "long"),
        ("int8_t", "signed char"),
        # ``char`` and ``signed char`` are distinct C++ types; user names are left alone.
        ("char", "char"),
        ("my_uint32_t", "my_uint32_t"),
        ("uint32_t_like", "uint32_t_like"),
    ],
)
def test_scalar_alias_spellings(spelling, canonical):
    assert canonical_scalar_type_spelling(spelling) == canonical


def test_alias_spelled_template_call_selects_the_explicit_specialization(tmp_path):
    code = flat(translate(tmp_path, TEMPLATE_SPECIALIZATIONS))
    # The declared spelling names the specialization; an alias spelled call reaches it.
    assert "inline uint max_value_u3cuint32_t_u3e()" in code
    assert "inline uint max_value_u3cuint8_t_u3e()" in code
    assert "o[id] == max_value_u3cuint32_t_u3e()" in code
    assert "p[id] = max_value_u3cuint8_t_u3e();" in code
    assert "max_value_u3cuint_u3e" not in code and "max_value_u3cuchar_u3e" not in code


def test_narrow_builtin_arguments_take_their_register_type(tmp_path):
    code = flat(translate(tmp_path, NARROW_BUILTIN_ARGUMENTS))
    assert "min(uint(ids[id]), uint(" in code
    assert "max(int(s[id]), int(" in code
    assert "clamp(int(s[id]), int(a), int(" in code


def test_vector_dot_reaches_the_library_beside_a_scalar_overload(tmp_path):
    code = flat(translate(tmp_path, SCALAR_DOT_OVERLOAD))
    assert "float dot_fn(float x, float y)" in code
    assert "o[id] = dot(a[id], a[id]) + dot_fn(b[id], b[id]);" in code


def test_metal_vector_scalar_operands_take_the_component_type(tmp_path):
    code = flat(translate(tmp_path, VECTOR_SCALAR_OPERANDS))
    assert "o[id].xyz = v / int(d);" in code
    assert "c *= half(f);" in code
    assert "c = saturate(fmod(c * half(bands), 1.0));" in code
    assert "h[id].xyz = c + half(f);" in code
    assert "all(v < int(d))" in code


def test_glsl_keeps_the_metal_vector_scalar_rule(tmp_path):
    # GLSL would divide unsigned (uvec3) and compute the half product in float without the conversions.
    # (GLSL refuses fmod of half vectors, so the banding line is left out.)
    source = VECTOR_SCALAR_OPERANDS.replace(
        "  c = saturate(fmod(c * bands, 1.0));\n", ""
    )
    code = flat(translate(tmp_path, source, "opengl"))
    assert "(v / int(d))" in code
    assert "uvec3" not in code


@pytest.mark.parametrize("case", sorted(CASES))
def test_metal_compiles(tmp_path, case):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler is not installed")
    source = tmp_path / "out.metal"
    source.write_text(translate(tmp_path, CASES[case]))
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
