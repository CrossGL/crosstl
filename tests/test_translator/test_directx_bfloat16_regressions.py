import textwrap

import pytest

import crosstl.translator
from crosstl._crosstl import translate
from crosstl.project import translate_project
from crosstl.translator.codegen.directx_codegen import (
    DirectXBFloat16UnsupportedError,
    HLSLCodeGen,
)


@pytest.mark.parametrize("dtype", ("bfloat16", "Narrow"))
@pytest.mark.parametrize("context", ("constructor", "return", "argument"))
@pytest.mark.parametrize("reverse", (False, True))
def test_directx_mixed_bfloat_selection_decodes_before_selection(
    dtype, context, reverse
):
    selection = "take ? fallback : value" if reverse else "take ? value : fallback"
    expression = {
        "constructor": f"float({dtype}({selection}))",
        "return": selection,
        "argument": f"consume({selection})",
    }[context]
    shader = f"""shader Selection {{
        typedef bfloat16 Narrow;
        float consume(float value) {{ return value; }}
        float choose({dtype} value, float fallback, bool take) {{
            return {expression};
        }}
    }}"""
    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))
    decoded = "__crossgl_bfloat16_to_float(uint(value))"
    wanted = (
        f"(take ? fallback : {decoded})"
        if reverse
        else f"(take ? {decoded} : fallback)"
    )
    assert wanted in generated
    assert selection not in generated
    if context == "constructor":
        assert f"__crossgl_bfloat16_from_float(float({wanted}))" in generated


@pytest.mark.parametrize("dtype", ("bfloat16", "Narrow"))
def test_directx_bfloat_selection_keeps_payload_until_outer_conversion(dtype):
    shader = f"""shader Selection {{
        typedef bfloat16 Narrow;
        float choose({dtype} left, {dtype} right, bool take) {{
            return take ? left : right;
        }}
    }}"""
    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))
    assert (
        "return __crossgl_bfloat16_to_float(uint((take ? left : right)));" in generated
    )
    assert "__crossgl_bfloat16_to_float(uint(left))" not in generated
    assert "__crossgl_bfloat16_to_float(uint(right))" not in generated


def test_directx_nested_bfloat_selection_keeps_single_evaluation():
    shader = """shader Selection {
        bfloat16 produce(inout uint calls) { calls++; return bfloat16(2.0); }
        float choose(bool outer, bool inner, inout uint calls) {
            return bfloat16(outer ? (inner ? produce(calls) : 123.0) : 17.0);
        }
    }"""
    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))
    assert generated.count("produce(calls)") == 1
    assert (
        "inner ? __crossgl_bfloat16_to_float(uint(produce(calls))) : 123.0" in generated
    )


@pytest.mark.parametrize(
    "declarations",
    (
        "using Narrow = bfloat;",
        "typedef bfloat Narrow;",
    ),
)
@pytest.mark.parametrize("helper", (False, True))
def test_directx_bfloat_aliases_preserve_logical_conversion(
    tmp_path, declarations, helper
):
    generated = []
    for dtype, aliases in (("bfloat", ""), ("Narrow", declarations)):
        source = tmp_path / f"{dtype}.metal"
        helper_source = (
            f"{dtype} identity({dtype} value) {{ return value; }}" if helper else ""
        )
        expression = "identity(value)" if helper else "value"
        source.write_text(
            f"""#include <metal_stdlib>
using namespace metal;
{aliases}
{helper_source}
kernel void convert(const device uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {{
    {dtype} value = {dtype}(as_type<bfloat>(ushort(values[i])));
    results[i] = as_type<uint>(float({expression}));
}}
""",
            encoding="utf-8",
        )
        generated.append(translate(str(source), backend="directx", format_output=False))
    assert (
        "__crossgl_bfloat16_from_uint16(uint16_t(uint16_t(values.Load(i))))"
        in generated[0]
    )
    assert f"asuint(__crossgl_bfloat16_to_float(uint({expression})))" in generated[0]
    assert generated[1] == generated[0]


def test_directx_canonical_bfloat_alias_chain_retains_helper_conversions():
    generated = []
    for dtype, aliases in (
        ("bfloat16", ""),
        ("Narrow", "typedef bfloat16 Base; typedef Base Narrow;"),
    ):
        shader = f"""
shader Conversions {{
    {aliases}
    {dtype} narrow(float value) {{ return {dtype}(value); }}
    {dtype} identity({dtype} value) {{ return value; }}
    float widen({dtype} value) {{ return float(identity(value)); }}
    float arithmetic({dtype} left, {dtype} right) {{ return float(left + right); }}
}}
"""
        generated.append(HLSLCodeGen().generate(crosstl.translator.parse(shader)))
    assert "return __crossgl_bfloat16_from_float(float(value));" in generated[0]
    assert "return __crossgl_bfloat16_to_float(uint(identity(value)));" in generated[0]
    assert generated[1] == generated[0]


@pytest.mark.parametrize(
    "dtype,aliases,expected",
    (
        ("Narrow", {"Narrow": "bfloat16"}, True),
        ("const Narrow", {"Narrow": "volatile bfloat16"}, True),
        ("Narrow", {"Narrow": "Base", "Base": "bfloat16"}, True),
        ("Narrow", {"Narrow": "uint"}, False),
        ("Narrow", {"Narrow": "Narrow"}, False),
        ("Narrow", {"Narrow": "Base", "Base": "Narrow"}, False),
        ("Narrow", {"Narrow": "bfloat16[2]"}, False),
        ("Narrow", {"Narrow": "bfloat16*"}, False),
    ),
)
def test_directx_bfloat_alias_classification_is_logical_and_cycle_safe(
    dtype, aliases, expected
):
    generator = HLSLCodeGen()
    generator.hlsl_type_aliases = aliases
    assert generator.is_hlsl_bfloat16_type(dtype) is expected


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
        ("acos", "acos"),
        ("asin", "asin"),
        ("atan", "atan"),
        ("cos", "cos"),
        ("cosh", "cosh"),
        ("exp", "exp"),
        ("log", "log"),
        ("log10", "log10"),
        ("log2", "log2"),
        ("rint", "round"),
        ("rsqrt", "rsqrt"),
        ("sin", "sin"),
        ("sinh", "sinh"),
        ("sqrt", "sqrt"),
        ("tan", "tan"),
        ("tanh", "tanh"),
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


@pytest.mark.parametrize("source_name", ["acosh", "asinh", "atanh"])
def test_directx_bfloat16_inverse_hyperbolic_uses_float_helper_and_rounds_back(
    source_name,
):
    shader = f"""
    shader ExactBFloatInverseHyperbolic {{
        bfloat16_t transform(bfloat16_t value) {{
            return {source_name}(value);
        }}
    }}
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    decoded = "__crossgl_bfloat16_to_float(uint(value))"
    helper_name = f"__crossgl_{source_name}_float"
    assert f"float {helper_name}(float value)" in generated
    assert (
        "return __crossgl_bfloat16_from_float(float(" f"{helper_name}({decoded})));"
    ) in generated


def test_directx_bfloat16_unknown_builtin_still_fails_closed():
    shader = """
    shader UnknownBFloatBuiltin {
        bfloat16_t transform(bfloat16_t value) {
            return cospi(value);
        }
    }
    """

    with pytest.raises(DirectXBFloat16UnsupportedError) as exc_info:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.operation == "cospi"
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
    assert "StructuredBuffer<uint16_t> in_ : register(t0);" in generated
    assert "RWStructuredBuffer<uint16_t> out_ : register(u1);" in generated
    assert "StructuredBuffer<uint16_t> copy_bfloat_in : register(t1);" in generated
    assert "RWStructuredBuffer<uint16_t> copy_bfloat_out : register(u2);" in generated
    assert "void copy_impl_half(StructuredBuffer<uint16_t> in_" in generated
    assert "void copy_impl_bfloat16_t(StructuredBuffer<uint16_t> in_" in generated
    assert "copy_impl_half(in_, int64_t(0), out_, int64_t(0));" in generated
    assert (
        "copy_impl_bfloat16_t(copy_bfloat_in, int64_t(0), "
        "copy_bfloat_out, int64_t(0));"
    ) in generated
    from crosstl.translator.resource_storage import (
        BINARY16_STORAGE,
        parse_resource_storage_header,
    )

    assert parse_resource_storage_header(generated) == {
        "in_": BINARY16_STORAGE,
        "out_": BINARY16_STORAGE,
    }


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


def test_directx_bfloat_helpers_preserve_minimum_precision_without_native_types():
    shader = """
    shader MinimumPrecisionBFloatConversions {
        bfloat16_t from_bits(min16uint bits) {
            return as_type<bfloat16_t>(bits);
        }

        min16uint to_bits(bfloat16_t value) {
            return as_type<min16uint>(value);
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert "uint from_bits(min16uint bits)" in generated
    assert "min16uint to_bits(uint value)" in generated
    assert "uint __crossgl_bfloat16_from_uint16(min16uint value)" in generated
    assert "min16uint __crossgl_bfloat16_to_uint16(uint value)" in generated
    assert "__crossgl_bfloat16_from_uint16(uint16_t value)" not in generated
    assert "uint16_t __crossgl_bfloat16_to_uint16(uint value)" not in generated


def test_directx_bfloat_helpers_follow_artifact_native_16_bit_mode():
    shader = """
    shader MixedPrecisionBFloatConversions {
        float16_t preserve_native_half(float16_t value) {
            return value;
        }

        bfloat16_t from_bits(min16uint bits) {
            return as_type<bfloat16_t>(bits);
        }

        min16uint to_bits(bfloat16_t value) {
            return as_type<min16uint>(value);
        }
    }
    """

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(shader)
    )

    assert "float16_t preserve_native_half(float16_t value)" in generated
    assert "uint from_bits(min16uint bits)" in generated
    assert "min16uint to_bits(uint value)" in generated
    assert "uint __crossgl_bfloat16_from_uint16(uint16_t value)" in generated
    assert "uint16_t __crossgl_bfloat16_to_uint16(uint value)" in generated
    assert "__crossgl_bfloat16_from_uint16(min16uint value)" not in generated
    assert "min16uint __crossgl_bfloat16_to_uint16(uint value)" not in generated


def test_directx_bfloat_helper_native_mode_resets_between_artifacts():
    native_shader = """
    shader NativeBFloatConversions {
        bfloat16_t from_bits(uint16_t bits) {
            return as_type<bfloat16_t>(bits);
        }
    }
    """
    minimum_precision_shader = """
    shader MinimumPrecisionBFloatConversions {
        bfloat16_t from_bits(min16uint bits) {
            return as_type<bfloat16_t>(bits);
        }
    }
    """
    generator = HLSLCodeGen(target_profile="dx12")

    native_generated = generator.generate(crosstl.translator.parse(native_shader))
    minimum_precision_generated = generator.generate(
        crosstl.translator.parse(minimum_precision_shader)
    )

    assert "uint __crossgl_bfloat16_from_uint16(uint16_t value)" in native_generated
    assert (
        "uint __crossgl_bfloat16_from_uint16(min16uint value)"
        in minimum_precision_generated
    )
    assert (
        "__crossgl_bfloat16_from_uint16(uint16_t value)"
        not in minimum_precision_generated
    )


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


def test_directx_bfloat_fixed_array_returns_keep_payload_bits(tmp_path):
    # Reduced from MLX 9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8,
    # mlx/backend/metal/kernels/binary_ops.h (DivMod). A uint2 return used
    # to coerce each bfloat16 payload numerically (uint(bf16_to_float(x))),
    # so WARP stored 3 instead of the bfloat16 bits of 3.0.
    (tmp_path / "divmod.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;
            typedef bfloat bfloat16_t;

            struct DivMod {
              template <typename T>
              metal::array<T, 2> operator()(T x, T y) {
                return {x / y, x};
              }
            };

            template <typename T>
            struct Widen {
              metal::array<T, 3> operator()(float x) {
                return {static_cast<T>(x), static_cast<T>(2.0f)};
              }
            };

            template <typename T>
            [[kernel]] void divmod(
                device const T* a [[buffer(0)]],
                device const T* b [[buffer(1)]],
                device T* c [[buffer(2)]],
                device T* d [[buffer(3)]],
                uint index [[thread_position_in_grid]]) {
              auto out = DivMod{}(a[index], b[index]);
              auto wide = Widen<T>{}(1.5f);
              c[index] = out[0];
              d[index] = out[1] + wide[2];
            }

            instantiate_kernel("divmod_bfloat", divmod, bfloat16_t)
            instantiate_kernel("divmod_float", divmod, float)
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
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")

    # bfloat16 arrays use the struct lowering whose fields keep the logical
    # bfloat16 type, so payloads are copied bit-for-bit.
    assert "struct CrossGLFixedArray_uint_2 {" in generated
    assert (
        "CrossGLFixedArray_uint_2 DivMod__operator_call__bfloat16_t("
        "inout DivMod self, uint x, uint y) {" in generated
    )
    assert (
        ".value1 = x;" in generated
    ), "a bfloat16 lane must be copied, not numerically converted"
    assert ".value0 = __crossgl_bfloat16_from_float(float((" in generated
    assert "c[index] = uint16_t(out_.value0);" in generated
    assert "uint2 DivMod__operator_call__bfloat16_t" not in generated
    assert "uint(__crossgl_bfloat16_to_float(" not in generated
    # Float lanes that initialize a bfloat16 array are rounded exactly once,
    # and value-initialized lanes are bfloat16 zero payloads.
    assert "struct CrossGLFixedArray_uint_3 {" in generated
    assert ".value0 = __crossgl_bfloat16_from_float(float(x));" in generated
    assert ".value1 = __crossgl_bfloat16_from_float(float(2.0));" in generated
    assert ".value2 = uint(0);" in generated
    # Arithmetic on a returned lane decodes the payload first.
    assert (
        "__crossgl_bfloat16_to_float(uint(out_.value1)) + "
        "__crossgl_bfloat16_to_float(uint(wide.value2))" in generated
    )

    # Non-bfloat16 scalar arrays keep the compact vector lowering.
    assert (
        "float2 DivMod__operator_call__float(inout DivMod self, float x, float y)"
        in generated
    )
