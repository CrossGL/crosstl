"""DirectX software subgroups keep proving loops over binary16 workgroup bits.

``lower_half_private_storage`` stores binary16 workgroup elements as
``as_type<uint16>(value)`` and reads them as ``as_type<float16>(element)``. A
bit reinterpretation writes nothing, so the uniform names its operand mentions,
such as a constant stride in a subscript, stay uniform and the loop around a
software subgroup barrier stays provably uniform.
"""

import textwrap

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_native_16_bit_compute_validates_if_available,
)

SOURCE = "shaders/strided_half_scan.metal"
ENTRY = "strided_half_scan"
KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    kernel void strided_half_scan(
        const device half* in [[buffer(0)]],
        device half* out [[buffer(1)]],
        const constant uint& axis_size [[buffer(2)]],
        const constant uint& stride [[buffer(3)]],
        uint lid [[thread_index_in_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
      constexpr int simd_size = 32;
      threadgroup half tile[64];
      half carry = 0;
      uint limit = stride - 1u;
      for (uint j = 0; j < axis_size; j++) {
        if (simd_lane < limit) {
          tile[lid] = in[(j * 2u + simd_group) * stride + simd_lane];
        } else {
          tile[lid] = half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        half value = tile[simd_lane * 2u + simd_group];
        value = simd_prefix_inclusive_sum(value) + carry;
        carry = simd_shuffle(value, ushort(simd_size - 1));
        out[(j * 2u + simd_group) * stride + simd_lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    """).strip()


def _translate(tmp_path, kernel=KERNEL):
    source = tmp_path / SOURCE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(kernel + "\n", encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        targets=("directx",),
        output_dir="out",
        entry_points={SOURCE: ENTRY},
        workgroup_size=(64, 1, 1),
        source_options={
            "metal": {
                "target_options": {
                    "directx": {
                        "relative_wave_shuffle_out_of_range": "self",
                        "software_subgroup_width": 32,
                    }
                }
            }
        },
    )
    payload = translate_project(config, format_output=False).to_json()
    errors = [item for item in payload["diagnostics"] if item["severity"] == "error"]
    generated = sorted((tmp_path / "out").rglob("*.hlsl"))
    return errors, generated[0].read_text(encoding="utf-8") if generated else None


def test_loop_over_lowered_workgroup_bits_keeps_its_uniform_proof(tmp_path):
    errors, generated = _translate(tmp_path)
    assert errors == []
    assert "groupshared uint16_t" in generated
    assert "asuint16(" in generated and "asfloat16(" in generated
    assert_directx_native_16_bit_compute_validates_if_available(generated, tmp_path)


def test_a_loop_whose_bound_varies_by_lane_is_still_refused(tmp_path):
    errors, _ = _translate(
        tmp_path, KERNEL.replace("j < axis_size", "j < axis_size + lid")
    )
    assert [item["code"] for item in errors][:1] == [
        "project.translate.directx-software-subgroup-invalid"
    ]
