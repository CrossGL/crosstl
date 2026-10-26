"""Generic functions deduce their type parameter from ``vecN<T>`` parameters.

``generic<T> T pick(vec2<T> v)`` stores its parameter as ``VectorType(T, 2)``, which generators spell ``T2``.
Deduction compared ``T2`` with the argument's ``ivec2``/``int2`` and bound nothing: GLSL and SPIR-V refused the call,
while HLSL and Metal emitted an undeclared ``pick(...)`` call. Specializations must also give ``vec2<T>`` locals,
returns and constructors their concrete vector type.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.generic_vector_deduction import (
    bind_generic_vector_parameter,
    concrete_generic_vector_name,
    vector_element_and_size,
)

TARGETS = ("opengl", "directx", "metal", "vulkan", "slang", "cuda")

PICK = textwrap.dedent("""
    shader main {
        generic<T> T pick(vec2<T> v) {
            return v.x + v.y;
        }
        compute {
            RWStructuredBuffer<int> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                ivec2 a = ivec2(1, 2);
                vec2 b = vec2(1.0, 2.0);
                out_[gid.x] = CALL;
            }
        }
    }
    """)

SWAP = textwrap.dedent("""
    shader main {
        generic<T> vec2<T> swap2(vec2<T> v) {
            vec2<T> r = vec2<T>(v.y, v.x);
            T s = r.x;
            r.y = r.y + s;
            return r;
        }
        compute {
            RWStructuredBuffer<int> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                ivec2 a = ivec2(1, 2);
                vec2 b = vec2(1.0, 2.0);
                ivec2 c = swap2(a);
                vec2 d = swap2(b);
                out_[gid.x] = c.x + int(d.y);
            }
        }
    }
    """)

ELEMENTS = textwrap.dedent("""
    shader main {
        generic<T> T pick(vec2<T> v) {
            T s = v.x + v.y;
            return s;
        }
        compute {
            RWStructuredBuffer<int> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                u16vec2 a = u16vec2(1, 2);
                vec2<i64> b = vec2<i64>(3, 4);
                uvec3 c = uvec3(1u, 2u, 3u);
                out_[gid.x] = int(pick(a)) + int(pick(b)) + int(pick(c.xy));
            }
        }
    }
    """)

TWICE_MEMBER = textwrap.dedent("""
    shader main {
        generic<T> T twice(T x) {
            T y = x + x;
            return y;
        }
        compute {
            RWStructuredBuffer<int> out_ @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                uvec3 c = uvec3(1u, 2u, 3u);
                out_[gid.x] = int(twice(c.x));
            }
        }
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _assert_no_generic_leak(code):
    for leak in ("T2", "vec2<T>", "pick(", "swap2("):
        assert leak not in code, leak


@pytest.mark.parametrize("backend", TARGETS)
def test_vector_parameter_deduces_element_type(tmp_path, backend):
    code = _translate(tmp_path, PICK.replace("CALL", "pick(a) + int(pick(b))"), backend)
    if backend == "vulkan":
        assert "OpFunctionCall" in code
        return
    assert "pick_int(" in code
    assert "pick_float(" in code
    _assert_no_generic_leak(code)


@pytest.mark.parametrize("backend", TARGETS)
def test_specialization_substitutes_vector_locals_returns_and_constructors(
    tmp_path, backend
):
    code = _translate(tmp_path, SWAP, backend)
    if backend == "vulkan":
        assert "OpFunctionCall" in code
        return
    _assert_no_generic_leak(code)
    int_vector, float_vector = {
        "opengl": ("ivec2", "vec2"),
    }.get(backend, ("int2", "float2"))
    assert f"{int_vector} swap2_int({int_vector} v)" in code
    assert f"{float_vector} swap2_float({float_vector} v)" in code
    constructor = "make_int2" if backend == "cuda" else int_vector
    assert f"{int_vector} r = {constructor}(v.y, v.x);" in code
    assert "int s = r.x;" in code
    assert "float s = r.x;" in code


@pytest.mark.parametrize(
    "backend, expected",
    [
        (
            "opengl",
            # Desktop GLSL 4.50 has no 16-bit integers; u16 is a masked uvec2.
            ("pick_u16(uvec2 v)", "pick_i64(i64vec2 v)", "pick_uint(uvec2 v)"),
        ),
        (
            "directx",
            ("pick_u16(uint16_t2 v)", "pick_i64(int64_t2 v)", "pick_uint(uint2 v)"),
        ),
        ("metal", ("pick_u16(", "pick_i64(long2 v)", "pick_uint(uint2 v)")),
    ],
)
def test_element_spellings_deduce_canonical_crossgl_scalars(
    tmp_path, backend, expected
):
    code = _translate(tmp_path, ELEMENTS, backend)
    for fragment in expected:
        assert fragment in code, fragment
    _assert_no_generic_leak(code)


def test_glsl_generic_call_takes_member_argument_as_concrete_parameter(tmp_path):
    code = _translate(tmp_path, TWICE_MEMBER, "opengl")
    assert "uint twice_uint(uint x)" in code
    assert "int(twice_uint(c.x))" in code


@pytest.mark.parametrize("backend", ("opengl", "directx", "metal", "vulkan", "cuda"))
def test_undeducible_vector_call_fails_closed(tmp_path, backend):
    with pytest.raises(Exception, match="pick"):
        _translate(tmp_path, PICK.replace("CALL", "pick(3)"), backend)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("ivec2", ("int", 2)),
        ("vec3", ("float", 3)),
        ("u16vec4", ("u16", 4)),
        ("i64vec2", ("i64", 2)),
        ("vec2<i64>", ("i64", 2)),
        ("int2", ("int", 2)),
        ("uint16_t3", ("u16", 3)),
        ("ushort2", ("u16", 2)),
        ("long4", ("i64", 4)),
        ("int64_t2", ("i64", 2)),
        ("bfloat2", ("bfloat16", 2)),
        ("half3", ("half", 3)),
        ("min16float2", ("min16float", 2)),
        ("v2int", ("int", 2)),
        ("v3u64", ("u64", 3)),
        ("float", None),
        ("mat2", None),
        ("T2", None),
        ("vec5", None),
    ],
)
def test_vector_element_and_size(spelling, expected):
    assert vector_element_and_size(spelling) == expected


def test_bind_generic_vector_parameter_requires_matching_size():
    bound = []
    assert bind_generic_vector_parameter(
        "vec2<T>", "ivec2", lambda e, a: bound.append((e, a))
    )
    assert bound == [("T", "int")]
    assert not bind_generic_vector_parameter("vec2<T>", "ivec3", bound.append)
    assert not bind_generic_vector_parameter("T", "ivec2", bound.append)
    assert bound == [("T", "int")]


def test_concrete_generic_vector_name():
    assert concrete_generic_vector_name("vec3<T>", {"T": "u16"}) == "u16vec3"
    assert concrete_generic_vector_name("vec2<T>", {"T": "int"}) == "ivec2"
    assert (
        concrete_generic_vector_name("vec2<T>", {"T": "min16float"})
        == "vec2<min16float>"
    )
    assert concrete_generic_vector_name("vec2<U>", {"T": "int"}) is None
    assert concrete_generic_vector_name("ivec2", {"T": "int"}) is None
