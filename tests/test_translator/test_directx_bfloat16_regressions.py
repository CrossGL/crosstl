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
    assert "StructuredBuffer<half> in_ : register(t0);" in generated
    assert "RWStructuredBuffer<half> out_ : register(u1);" in generated
    assert "StructuredBuffer<uint16_t> copy_bfloat_in : register(t1);" in generated
    assert "RWStructuredBuffer<uint16_t> copy_bfloat_out : register(u2);" in generated
    assert "void copy_impl_half(StructuredBuffer<half> in_" in generated
    assert "void copy_impl_bfloat16_t(StructuredBuffer<uint16_t> in_" in generated
    assert "copy_impl_half(in_, int64_t(0), out_, int64_t(0));" in generated
    assert (
        "copy_impl_bfloat16_t(copy_bfloat_in, int64_t(0), "
        "copy_bfloat_out, int64_t(0));"
    ) in generated
