"""CUDA and HIP casts to a lone ``unsigned`` or ``signed`` import as casts to ``u32`` and ``i32``.

A sign keyword by itself is a simple-type-specifier that names ``int`` (C++ [dcl.type.simple]), so ``(unsigned)x``,
``(const unsigned)x``, the functional cast ``unsigned(x)`` and ``(unsigned *)p`` are casts. CUDA input read
``(unsigned)x`` and ``unsigned(x)`` as expressions and stopped with a syntax error; HIP input stopped at
``unsigned(x)`` and ``(unsigned *)p``.
"""

from __future__ import annotations

import os
import sys

import pytest

import crosstl

TARGETS = ("cuda", "hip", "metal", "directx", "opengl", "vulkan")
REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_SIGN_ONLY_CASTS"

SOURCE = """
__global__ void probe(int* data) {
    int x = data[0];
    data[1] = (int)((unsigned)x >> 28);
    data[2] = (int)(unsigned(x) % 7u);
    data[3] = (signed)((unsigned)x / 2u);
    data[4] = signed(3u) - 5;
    data[5] = (int)((const unsigned)x > 10u);
    data[6] = (int)(unsigned)(signed)(unsigned)x;
}
"""


def _source(language, source=SOURCE):
    if language == "hip":
        return "#include <hip/hip_runtime.h>\n" + source
    return source


def _import(tmp_path, language, source=SOURCE):
    path = tmp_path / ("probe.cu" if language == "cuda" else "probe.hip")
    path.write_text(_source(language, source))
    return crosstl.translate(str(path), backend="cgl", format_output=False)


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_sign_only_casts_import_as_integer_casts(tmp_path, language):
    crossgl = _import(tmp_path, language)
    assert "data[1] = i32((u32(x) >> 28));" in crossgl
    assert "data[2] = i32((u32(x) % 7u));" in crossgl
    assert "data[3] = i32((u32(x) / 2u));" in crossgl
    assert "data[4] = (i32(3u) - 5);" in crossgl
    assert "data[5] = i32((u32(x) > 10u));" in crossgl
    assert "data[6] = i32(u32(i32(u32(x))));" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_a_sign_only_pointer_cast_imports_as_a_pointer_cast(tmp_path, language):
    source = (
        "__device__ unsigned top(int* data) {\n"
        "    unsigned* words = (unsigned *)data;\n"
        "    return words[0] >> 31;\n"
        "}\n"
    )
    crossgl = _import(tmp_path, language, source)
    assert "var words: ptr<u32> = ptr<u32>(data);" in crossgl


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_sign_only_casts_translate_to_every_target(tmp_path, language):
    path = tmp_path / "probe.cgl"
    path.write_text(_import(tmp_path, language))
    for target in TARGETS:
        assert crosstl.translate(str(path), backend=target, format_output=False)


def _expected(x):
    word = x & 0xFFFFFFFF

    def signed(value):
        value &= 0xFFFFFFFF
        return value - (1 << 32) if value & 0x80000000 else value

    return [x, word >> 28, word % 7, signed(word // 2), -2, int(word > 10), x]


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_metal_translation_computes_what_the_casts_compute(tmp_path, language):
    if sys.platform != "darwin" and os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip("the Metal dispatch needs macOS")
    from tests.metal_compute_dispatch import dispatch_word_buffer

    path = tmp_path / "probe.cgl"
    path.write_text(_import(tmp_path, language))
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    for x in (-5, 7):
        words = dispatch_word_buffer(metal, tmp_path, [x] + [0] * 6, REQUIRE_ENV)
        assert words == _expected(x)
