"""SFINAE constraints spelled with ``::metal::`` and Boolean equality resolve.

PyTorch's MPS kernels overload ``cast_to`` through constraints such as
``::metal::enable_if_t<!::metal::is_same_v<U, T> && (is_complex_v<T> ==
is_complex_v<U>), bool>``. The constraint evaluator recognized ``enable_if_t``
and the standard traits only without the leading global ``::``. It also had no
``==`` / ``!=`` between Boolean operands. Every candidate was therefore
"unrecognized", and the call was refused.

The imported helper's parameter is named ``from``, which is a CrossGL
keyword, so the cast ``(uint8)from`` in its body must also parse.
"""

import textwrap

import pytest

import crosstl

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct alignas(1) float8_e4m3fn {
      uchar x;
    };

    template <typename T>
    constexpr constant bool is_float8_v = ::metal::is_same_v<T, float8_e4m3fn>;

    template <typename T>
    constexpr constant bool is_complex_v =
        ::metal::is_same_v<T, float2> || ::metal::is_same_v<T, half2>;

    template <typename T, typename U,
              ::metal::enable_if_t<::metal::is_same_v<U, T>, bool> = true>
    inline T cast_to(const U from) {
      return from;
    }

    template <
        typename T,
        typename U,
        ::metal::enable_if_t<
            !::metal::is_same_v<U, T> && (is_complex_v<T> == is_complex_v<U>) &&
                !is_float8_v<T> && !is_float8_v<U>,
            bool> = true>
    inline T cast_to(const U from) {
      return static_cast<T>(from);
    }

    template <typename T, typename U,
              ::metal::enable_if_t<is_complex_v<T> && !is_complex_v<U>, bool> = true>
    inline T cast_to(const U from) {
      return T(float(from), 0.0);
    }

    kernel void k(device const float *src [[buffer(0)]],
                  device uchar *dst [[buffer(1)]],
                  device float *same [[buffer(2)]],
                  uint i [[thread_position_in_grid]])
    {
      dst[i] = cast_to<uchar>(src[i]);
      same[i] = cast_to<float>(src[i]);
    }
    """)


def _translate(tmp_path, backend):
    source = tmp_path / "cast_to.metal"
    source.write_text(KERNEL, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_the_enabled_overload_is_selected(tmp_path):
    code = _translate(tmp_path, "cgl")
    # uchar <- float takes the static_cast overload; float <- float the
    # identity overload.
    assert "cast_to_uchar_float(float from)" in code
    assert "return (uint8)from;" in code
    assert "cast_to_float_float(float from)" in code


@pytest.mark.parametrize(
    ("backend", "marker"),
    [
        ("directx", "cast_to_uchar_float"),
        ("opengl", "cast_to_uchar_float"),
        ("metal", "cast_to_uchar_float"),
        # SPIR-V keeps the helper as a function but names it only in OpName.
        ("vulkan", "OpEntryPoint GLCompute"),
    ],
)
def test_the_selected_overload_reaches_every_target(tmp_path, backend, marker):
    assert marker in _translate(tmp_path, backend)
