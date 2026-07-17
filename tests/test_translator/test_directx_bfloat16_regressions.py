import textwrap

import pytest

import crosstl.translator
from crosstl.project import translate_project
from crosstl.translator.codegen.directx_codegen import (
    DirectXBFloat16UnsupportedError,
    HLSLCodeGen,
)


def test_directx_bfloat16_builtin_decodes_and_preserves_return_contract():
    shader = """
    shader ExactBFloatBuiltins {
        bfloat16_t truncate_value(bfloat16_t value) {
            return trunc(value);
        }

        bfloat16_t remainder_value(bfloat16_t left, bfloat16_t right) {
            return fmod(left, right);
        }

        bool classify_value(bfloat16_t value) {
            return isnan(value);
        }

        float truncate_as_float(bfloat16_t value) {
            return trunc(value);
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    decoded = "__crossgl_bfloat16_to_float(uint(value))"
    rounded_trunc = f"__crossgl_bfloat16_from_float(float(trunc({decoded})))"
    assert f"return {rounded_trunc};" in generated
    assert (
        "return __crossgl_bfloat16_from_float(float(fmod("
        "__crossgl_bfloat16_to_float(uint(left)), "
        "__crossgl_bfloat16_to_float(uint(right)))));"
    ) in generated
    assert f"return isnan({decoded});" in generated
    assert f"return __crossgl_bfloat16_to_float(uint({rounded_trunc}));" in generated


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("exp", "exp"),
        ("log10", "log10"),
        ("rint", "round"),
    ],
)
def test_directx_bfloat16_transcendental_builtin_rounds_back_to_bfloat(
    source_name,
    target_name,
):
    shader = f"""
    shader ExactBFloatTranscendental {{
        bfloat16_t transform(bfloat16_t value) {{
            return {source_name}(value);
        }}
    }}
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    decoded = "__crossgl_bfloat16_to_float(uint(value))"
    assert (
        "return __crossgl_bfloat16_from_float(float(" f"{target_name}({decoded})));"
    ) in generated


def test_directx_bfloat16_unknown_builtin_still_fails_closed():
    shader = """
    shader UnknownBFloatBuiltin {
        bfloat16_t transform(bfloat16_t value) {
            return sin(value);
        }
    }
    """

    with pytest.raises(DirectXBFloat16UnsupportedError) as exc_info:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.operation == "sin"
    assert exc_info.value.reason == "unsupported-bfloat16-builtin"


@pytest.mark.parametrize(
    "call",
    ["fmod(value, other)", "fmod(other, value)"],
)
def test_directx_bfloat16_mixed_builtin_arguments_fail_closed(call):
    shader = f"""
    shader MixedBFloatBuiltin {{
        bfloat16_t remainder(bfloat16_t value, float other) {{
            return {call};
        }}
    }}
    """

    with pytest.raises(DirectXBFloat16UnsupportedError) as exc_info:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.operation == "fmod"
    assert exc_info.value.reason == "unsupported-bfloat16-builtin"


def test_directx_half_and_bfloat_entry_resources_do_not_share_source_type(tmp_path):
    (tmp_path / "copy.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;
            typedef bfloat bfloat16_t;

            template <typename T>
            void copy_impl(const device T* in, device T* out) {
              out[0] = in[0];
            }

            template <typename T>
            [[kernel]] void copy_values(
                const device T* in [[buffer(0)]],
                device T* out [[buffer(1)]]) {
              copy_impl<T>(in, out);
            }

            instantiate_kernel("copy_half", copy_values, half)
            instantiate_kernel("copy_bfloat", copy_values, bfloat16_t)
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(
        tmp_path,
        targets=["directx"],
        output_dir="out",
        format_output=False,
    )
    payload = report.to_json()

    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    artifact = payload["artifacts"][0]
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert "StructuredBuffer<float16_t> in_ : register(t0);" in generated
    assert "RWStructuredBuffer<float16_t> out_ : register(u1);" in generated
    assert "StructuredBuffer<uint16_t> copy_bfloat_in : register(t1);" in generated
    assert "RWStructuredBuffer<uint16_t> copy_bfloat_out : register(u2);" in generated
    assert "void copy_impl_half(StructuredBuffer<float16_t> in_" in generated
    assert "void copy_impl_bfloat16_t(StructuredBuffer<uint16_t> in_" in generated
    assert "copy_impl_half(in_, int64_t(0), out_, int64_t(0));" in generated
    assert (
        "copy_impl_bfloat16_t(copy_bfloat_in, int64_t(0), "
        "copy_bfloat_out, int64_t(0));"
    ) in generated


def test_mlx_gemv_materialized_helper_preserves_bfloat_storage_boundary(tmp_path):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/gemv.h.
    (tmp_path / "gemv.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;
            typedef bfloat bfloat16_t;

            template <typename T>
            struct GEMVKernel {
              template <typename U = T>
              static void load_unsafe(const device T* src, thread U dst[1]) {
                dst[0] = static_cast<U>(src[0]);
              }

              static void run(
                  const device T* input [[buffer(0)]],
                  device float* output [[buffer(1)]]) {
                thread float value[1];
                load_unsafe<float>(input, value);
                output[0] = value[0];
              }
            };

            template <typename T>
            [[kernel]] void gemv(
                const device T* input [[buffer(0)]],
                device float* output [[buffer(1)]]) {
              using gemv_kernel = GEMVKernel<T>;
              gemv_kernel::run(input, output);
            }

            instantiate_kernel("gemv_bfloat", gemv, bfloat16_t)
            """).strip(),
        encoding="utf-8",
    )

    payload = translate_project(
        tmp_path,
        targets=["directx"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    assert artifact["requiredCapabilities"] == ["directx.native-16bit-types"]
    assert artifact["bfloat16Lowering"] == {
        "status": "exact",
        "approximationUsed": False,
        "registerRepresentation": "uint-low-16-bits",
        "storageRepresentation": "native-uint16",
        "roundingMode": "round-to-nearest-ties-to-even",
    }

    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert "StructuredBuffer<uint16_t> input : register(t0);" in generated
    assert "RWStructuredBuffer<float> output : register(u1);" in generated
    assert (
        "void GEMVKernel_bfloat16_t__load_unsafe__float("
        "StructuredBuffer<uint16_t> src, int64_t src_offset, "
        "inout float dst[1])" in generated
    )
    assert (
        "dst[0] = __crossgl_bfloat16_to_float("
        "uint(src[uint(src_offset)]));" in generated
    )
    assert "StructuredBuffer<half>" not in generated
    assert "__crossgl_bfloat16_to_float(half(" not in generated


def test_materialized_struct_helper_narrows_bfloat_resource_writes(tmp_path):
    (tmp_path / "writer.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;
            typedef bfloat bfloat16_t;

            template <typename T>
            struct Writer {
              static void run(device T* output) {
                output[0] = static_cast<T>(1.0f);
              }
            };

            template <typename T>
            [[kernel]] void write_value(device T* output [[buffer(0)]]) {
              Writer<T>::run(output);
            }

            instantiate_kernel("write_bfloat", write_value, bfloat16_t)
            """).strip(),
        encoding="utf-8",
    )

    payload = translate_project(
        tmp_path,
        targets=["directx"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert payload["diagnostics"] == []
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "RWStructuredBuffer<uint16_t> output" in generated
    assert (
        "output[uint(output_offset)] = "
        "uint16_t(__crossgl_bfloat16_from_float(float(1.0)));" in generated
    )


def test_directx_bfloat_storage_native_uint16_mode_narrows_buffer_writes():
    shader = """
    shader ExactBFloatStorage {
        StructuredBuffer<bfloat16_t> input;
        RWStructuredBuffer<bfloat16_t> output;

        uint16_t preserve_bits(uint16_t bits) {
            uint16_t copy = bits;
            return copy;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                bfloat16_t value = input[0];
                output[0] = value;
            }
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert "StructuredBuffer<uint16_t> input : register(t0);" in generated
    assert "RWStructuredBuffer<uint16_t> output : register(u0);" in generated
    assert "uint16_t preserve_bits(uint16_t bits)" in generated
    assert "uint16_t copy = bits;" in generated
    assert "output[0] = uint16_t(value);" in generated
    assert "min16uint" not in generated
    assert "output[0] = value;" not in generated


def test_directx_bfloat_storage_preserves_explicit_minimum_precision_types():
    shader = """
    shader ExactBFloatStorageMinimumPrecision {
        StructuredBuffer<bfloat16_t> input;
        RWStructuredBuffer<bfloat16_t> output;

        min16uint preserve_unsigned(min16uint value) {
            min16uint copy = value;
            return copy;
        }

        min16int preserve_signed(min16int value) {
            min16int copy = value;
            return copy;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                output[0] = input[0];
            }
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert "StructuredBuffer<uint16_t> input : register(t0);" in generated
    assert "RWStructuredBuffer<uint16_t> output : register(u0);" in generated
    assert "min16uint preserve_unsigned(min16uint value)" in generated
    assert "min16uint copy = value;" in generated
    assert "min16int preserve_signed(min16int value)" in generated
    assert "min16int copy = value;" in generated
    assert "uint16_t preserve_unsigned" not in generated
    assert "int16_t preserve_signed" not in generated


def test_directx_bfloat_storage_narrows_writes_through_resource_parameters():
    shader = """
    shader ExactBFloatResourceParameter {
        StructuredBuffer<bfloat16_t> input;
        RWStructuredBuffer<bfloat16_t> output;

        void store_value(
            RWStructuredBuffer<bfloat16_t> destination,
            bfloat16_t value
        ) {
            destination[0] = value;
        }

        void store_physical_value(
            RWStructuredBuffer<uint16_t> destination,
            bfloat16_t value
        ) {
            destination[0] = value;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                store_value(output, input[0]);
                store_physical_value(output, input[0]);
            }
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert (
        "void store_value(RWStructuredBuffer<uint16_t> destination, uint value)"
        in generated
    )
    assert "destination[0] = uint16_t(value);" in generated
    assert "destination[0] = value;" not in generated
    assert (
        "void store_physical_value("
        "RWStructuredBuffer<uint16_t> destination, uint value)" in generated
    )


def test_directx_bfloat_storage_narrows_canonical_buffer_store_values():
    shader = """
    shader ExactBFloatCanonicalStore {
        RWStructuredBuffer<bfloat16_t> output;

        void store_value(bfloat16_t value) {
            buffer_store(output, 0u, value);
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                store_value(bfloat16_t(1.0));
            }
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert "RWStructuredBuffer<uint16_t> output : register(u0);" in generated
    assert " = uint16_t(value);" in generated
    assert " = value;" not in generated
