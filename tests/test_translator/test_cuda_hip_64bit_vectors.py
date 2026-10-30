"""CUDA and HIP output declare vectors of 64-bit integers as ``longlong2``..``ulonglong4``.

CrossGL spells them ``i64vec2``/``u64vec3`` or ``vec2<i64>`` (the spelling the CUDA and HIP importers give
``longlong2``). Both generators emitted those spellings, or ``i642``, unchanged, which names no CUDA or HIP type,
and the vector helpers narrowed a 64-bit component or scalar operand to ``int``.
"""

import pytest

import crosstl
from crosstl.translator.codegen.c_wide_integer_vectors import (
    wide_integer_vector_constructor,
    wide_integer_vector_type,
)

BACKENDS = ("cuda", "hip")

KERNEL = """shader P {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer u64vec2* outp;
        buffer i64vec4* signed_outp;
        {functions}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def translate(tmp_path, backend, body, functions=""):
    path = tmp_path / "kernel.cgl"
    path.write_text(KERNEL.format(body=body, functions=functions))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("spelling", "native"),
    [
        ("i64vec2", "longlong2"),
        ("u64vec3", "ulonglong3"),
        ("vec4<i64>", "longlong4"),
        ("vec2 < uint64_t >", "ulonglong2"),
        ("vec3<int64>", "longlong3"),
        ("ulonglong4", "ulonglong4"),
    ],
)
def test_spellings_map_to_the_native_vector(spelling, native):
    assert wide_integer_vector_type(spelling) == native
    assert wide_integer_vector_constructor(spelling) == f"make_{native}"


@pytest.mark.parametrize(
    "spelling", ["i64vec5", "vec2<i32>", "uvec2", "vec2<Foo>", "i64"]
)
def test_other_spellings_are_not_wide_vectors(spelling):
    assert wide_integer_vector_type(spelling) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_declarations_constructors_and_buffers_use_the_native_vectors(
    tmp_path, backend
):
    code = translate(
        tmp_path,
        backend,
        """u64vec2 a = u64vec2(1u, 2u);
            vec3<i64> b = vec3<i64>(3, 4, 5);
            i64vec4 w = i64vec4(b, int64_t(a.x));
            outp[i] = a;
            signed_outp[i] = w;""",
    )
    assert "ulonglong2* outp" in code
    assert "longlong4* signed_outp" in code
    assert "ulonglong2 a = make_ulonglong2(1u, 2u);" in code
    assert "longlong3 b = make_longlong3(3, 4, 5);" in code
    assert "longlong4 w = make_longlong4(b.x, b.y, b.z, int64_t(a.x));" in code
    for stray in ("u64vec", "i64vec", "vec3<i64>", "i643", "u642"):
        assert stray not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_function_signatures_use_the_native_vectors(tmp_path, backend):
    code = translate(
        tmp_path,
        backend,
        "outp[i] = twice(u64vec2(i, i));",
        functions="u64vec2 twice(u64vec2 v) { return v + v; }",
    )
    assert "ulonglong2 twice(ulonglong2 v)" in code
    assert "make_ulonglong2((lhs.x + rhs.x), (lhs.y + rhs.y))" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_operands_keep_all_64_bits(tmp_path, backend):
    code = translate(
        tmp_path,
        backend,
        """u64vec2 a = u64vec2(i, i);
            uint64_t big = a.x << 40u;
            outp[i] = a * big;""",
    )
    # The helper's scalar parameter must not narrow ``big`` to unsigned int.
    assert "ulonglong2 lhs, unsigned long long rhs" in code
    assert "unsigned int rhs" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_bitwise_operators_lower_componentwise(tmp_path, backend):
    code = translate(
        tmp_path,
        backend,
        """u64vec2 a = u64vec2(i, i);
            uint64_t mask = uint64_t(255u);
            outp[i] = (a ^ mask) | (a << 3);""",
    )
    assert "make_ulonglong2((a.x ^ mask), (a.y ^ mask))" in code
    assert "make_ulonglong2((a.x << 3), (a.y << 3))" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_signed_helpers_use_64_bit_forms(tmp_path, backend):
    code = translate(
        tmp_path,
        backend,
        """i64vec2 s = i64vec2(int64_t(i) - int64_t(5), int64_t(-7));
            i64vec2 t = abs(s) + sign(s) - s;
            signed_outp[i] = i64vec4(t, -t);""",
    )
    assert "make_longlong2(llabs(value.x), llabs(value.y))" in code
    assert "cgl_longlong2_neg" in code
    assert "cgl_longlong2_sign" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_swizzles_and_components_keep_the_64_bit_type(tmp_path, backend):
    code = translate(
        tmp_path,
        backend,
        """i64vec4 w = i64vec4(int64_t(i));
            i64vec2 lo = w.xy;
            uint64_t first = u64vec2(w.zw).x;
            outp[i] = u64vec2(lo) + u64vec2(first);""",
    )
    assert "longlong2 lo = make_longlong2(w.x, w.y);" in code
    assert "i64vec" not in code and "u64vec" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_imported_long_long_vectors_round_trip(tmp_path, backend):
    source = tmp_path / f"kernel.{'cu' if backend == 'cuda' else 'hip'}"
    source.write_text("""__global__ void k(longlong2* out, const ulonglong3* in) {
    int i = threadIdx.x;
    ulonglong3 v = in[i];
    out[i] = make_longlong2((long long)v.x, (long long)v.y);
}
""")
    code = crosstl.translate(str(source), backend=backend, format_output=False)
    assert "longlong2* out" in code
    assert "ulonglong3 v = " in code
    assert "make_longlong2(" in code
    for stray in ("i642", "u643", "vec2<i64>", "vec3<u64>"):
        assert stray not in code
