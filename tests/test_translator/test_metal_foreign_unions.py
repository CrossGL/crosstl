"""CUDA and HIP unions translate to Metal unions when Metal lays them out the same.

The CUDA and HIP importers write a union as a CrossGL struct with the ``@union_layout(size, alignment, little_endian,
cuda)`` contract (``hip`` for HIP). The Metal generator accepted only the ``metal`` ABI, so a kernel that writes one
member of a union and reads another, the usual way CUDA reinterprets bits, was refused. MSL unions follow C++, as
nvcc's and hipcc's do, so the union is accepted when each member's Metal size and alignment are the recorded ones,
and refused when they differ (CUDA's ``float3`` is 12 bytes aligned to 4, Metal's 16 aligned to 16).
"""

from __future__ import annotations

import struct

import pytest

import crosstl
from crosstl.translator.codegen.metal_feature_errors import (
    UnsupportedMetalFeatureError,
)
from tests.metal_compute_dispatch import compile_metal_library, dispatch_word_buffer

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_FOREIGN_UNION_EXECUTION"

KERNEL = """
union Bits {
    float f;
    unsigned int u;
};
extern "C" __global__ void probe(float* data) {
    Bits b;
    b.f = data[0];
    b.u = (b.u >> 16) | 0x3F800000u;
    data[0] = b.f;
}
"""


def _metal(tmp_path, source, suffix):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _crossgl_metal(source):
    return crosstl.translator.codegen.metal_codegen.MetalCodeGen().generate(
        crosstl.translator.parse(source)
    )


@pytest.mark.parametrize(
    "suffix,source",
    [(".cu", KERNEL), (".hip", "#include <hip/hip_runtime.h>\n" + KERNEL)],
    ids=["cuda", "hip"],
)
def test_cuda_and_hip_unions_reinterpret_bits_on_the_metal_gpu(
    tmp_path, suffix, source
):
    generated = _metal(tmp_path, source, suffix)

    assert "union Bits {" in generated
    assert "struct Bits" not in generated
    word = 0x40490FDB
    (signed,) = struct.unpack("<i", struct.pack("<I", word))
    result = dispatch_word_buffer(generated, tmp_path, [signed], REQUIRE_ENV)
    assert struct.unpack("<I", struct.pack("<i", result[0]))[0] == (
        (word >> 16) | 0x3F800000
    )


def test_foreign_union_with_metal_layout_members_compiles(tmp_path):
    generated = _crossgl_metal("""
    shader Words {
        @union_layout(16, 8, little_endian, cuda)
        struct Words {
            @union_member_layout(0, 16, 4) uint w[4];
            @union_member_layout(0, 8, 8) vec2 v;
            @union_member_layout(0, 1, 1) bool b;
            @union_member_layout(0, 2, 2) half h;
        }
        uint first_word(vec2 value) {
            Words words;
            words.v = value;
            return words.w[0];
        }
    }
    """)

    assert "union Words {" in generated
    assert "uint w[4];" in generated
    assert "float2 v;" in generated
    compile_metal_library(generated, tmp_path, REQUIRE_ENV)


@pytest.mark.parametrize(
    "declaration,reason",
    [
        (
            """
            @union_layout(12, 4, little_endian, cuda)
            struct Vec {
                @union_member_layout(0, 12, 4) vec3 v;
                @union_member_layout(0, 4, 4) uint w;
            }
            """,
            "member-layout-mismatch",
        ),
        (
            """
            @union_layout(16, 16, little_endian, hip)
            struct Padded {
                @union_member_layout(0, 4, 4) uint w;
            }
            """,
            "union-layout-mismatch",
        ),
    ],
    ids=["cuda-float3", "over-aligned"],
)
def test_foreign_union_whose_metal_layout_differs_is_refused(declaration, reason):
    with pytest.raises(UnsupportedMetalFeatureError) as error:
        _crossgl_metal(f"shader Foreign {{ {declaration} }}")

    assert error.value.feature == "union-layout-contract"
    assert error.value.reason == reason
