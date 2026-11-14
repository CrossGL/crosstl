"""Generic CrossGL vectors (``vec2<u16>``) translate exactly like their named spellings (``u16vec2``).

The CUDA and HIP importers write ``ushort2``, ``longlong2`` and ``char2`` as ``vec2<u16>``, ``vec2<i64>`` and
``vec2<i8>``. GLSL emitted those as ``u162``/``i642``/``i82`` with a verbatim ``vec2<u16>(...)`` constructor, HLSL and
Metal did the same for 64-bit elements, and HLSL dropped the wrap-around of 8-bit vectors.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.crossgl_vector_names import (
    crossgl_named_vector_spelling,
    crossgl_named_vector_type,
)

TARGETS = ("opengl", "directx", "metal")

# element, named vector, constructor arguments
ELEMENTS = (
    ("i8", "i8vec2", "1, 2"),
    ("u8", "u8vec2", "1u, 2u"),
    ("i16", "i16vec2", "1, 2"),
    ("u16", "u16vec2", "1u, 2u"),
    ("i64", "i64vec2", "1, 2"),
    ("u64", "u64vec2", "1u, 2u"),
    ("f16", "f16vec2", "1.0, 2.0"),
)

SHADER = textwrap.dedent("""
    shader main {
        compute {
            layout(std430, binding = 0) buffer Out { int data[]; } out_buf;
            layout(local_size_x = 1) in;
            void main() {
                VEC a = VEC(ARGS);
                VEC b = a + a;
                out_buf.data[0] = int(b.x);
            }
        }
    }
    """)

CUDA_KERNEL = textwrap.dedent("""
    __global__ void k(int* out) {
        ushort2 a = make_ushort2(1, 2);
        longlong2 b = make_longlong2(3, 4);
        char2 c = make_char2(5, 6);
        out[0] = (int)(a.x + a.y) + (int)(b.x + b.y) + (int)(c.x + c.y);
    }
    """)


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _shader(vector, args):
    return SHADER.replace("VEC", vector).replace("ARGS", args)


def _assert_no_generic_leak(code, element):
    assert "vec2<" not in code
    assert f"{element}2 " not in code
    assert f"{element}2(" not in code


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("element,named,args", ELEMENTS)
def test_generic_vector_matches_named_vector(tmp_path, backend, element, named, args):
    generic = _translate(tmp_path, _shader(f"vec2<{element}>", args), backend)
    by_name = _translate(tmp_path, _shader(named, args), backend)

    assert generic == by_name
    _assert_no_generic_leak(generic, element)


def test_generic_int8_vector_keeps_hlsl_wrap_around(tmp_path):
    code = _translate(tmp_path, _shader("vec2<i8>", "1, 2"), "directx")

    assert "<< 24u) >> 24" in code


@pytest.mark.parametrize("backend", TARGETS)
def test_cuda_narrow_and_wide_vectors_name_target_types(tmp_path, backend):
    code = _translate(tmp_path, CUDA_KERNEL, backend, suffix=".cu")

    for element in ("u16", "i64", "i8"):
        _assert_no_generic_leak(code, element)


def test_cuda_vectors_lower_to_target_vectors(tmp_path):
    glsl = _translate(tmp_path, CUDA_KERNEL, "opengl", suffix=".cu")
    hlsl = _translate(tmp_path, CUDA_KERNEL, "directx", suffix=".cu")
    msl = _translate(tmp_path, CUDA_KERNEL, "metal", suffix=".cu")

    assert "i64vec2 b = i64vec2(3, 4);" in glsl
    assert "bitfieldExtract(ivec2(5, 6), 0, 8)" in glsl
    assert "int64_t2 b = int64_t2(3, 4);" in hlsl
    assert "<< 24u) >> 24" in hlsl
    assert "long2 b = long2(3, 4);" in msl


@pytest.mark.parametrize(
    "spelling,named",
    [
        ("vec2<i64>", "i64vec2"),
        ("vec3< u16 >", "u16vec3"),
        ("vec4<float>", "vec4"),
        ("vec2<int>", "ivec2"),
        ("vec3<u32>", "uvec3"),
        ("vec4<bool>", "bvec4"),
        ("vec2<f64>", "dvec2"),
        ("vec2<half>", "f16vec2"),
        ("vec4<bfloat16>", "bfloat16vec4"),
    ],
)
def test_named_spelling_of_generic_vectors(spelling, named):
    assert crossgl_named_vector_spelling(spelling) == named


@pytest.mark.parametrize(
    "spelling", ["vec2<Pair>", "vec5<i32>", "vec2<T>", "u16vec2", "array<i32, 2>"]
)
def test_spellings_without_a_named_vector_are_unchanged(spelling):
    assert crossgl_named_vector_spelling(spelling) == spelling


def test_named_vector_type_rejects_unknown_elements_and_sizes():
    assert crossgl_named_vector_type("u8", 4) == "u8vec4"
    assert crossgl_named_vector_type("Pair", 2) is None
    assert crossgl_named_vector_type("i32", 5) is None
    assert crossgl_named_vector_type(None, 2) is None
