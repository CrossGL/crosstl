"""Quantize HLSL UNORM storage texture stores as Direct3D converts them."""

import re
import shutil
import struct
import subprocess

import pytest

from crosstl import translate, translate_with_metadata
from crosstl.backend.DirectX.unorm_store_rounding import (
    HLSLUnormStoreRoundingError,
    scale_buffer_suffix,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

OPTIONS = {"unorm_store_rounding": "single_precision"}
HLSL = """
cbuffer Constants : register(b0) { float level; }
cbuffer Tints : register(b3) { float4 Tint; }
cbuffer Other : register(b9, space1) { float4 Unused; }
RWTexture2D<unorm float4> color : register(u0);
RWTexture3D<unorm float2> pair : register(u1, space2);
RWTexture1D<unorm float> single : register(u2);
RWTexture2D<float4> plain : register(u3);
Texture2D<unorm float4> source : register(t0);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    color[id.xy] = source[id.xy] * level + Tint;
    color[id.xy].z = 0.3;
    color[id.xy] += 0.25;
    pair[id] = float2(level, 0.6);
    single[id.x] = level;
    plain[id.xy] = color[id.xy];
}
"""


def _translate(tmp_path, source, backend, options=OPTIONS, name="unorm.hlsl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return translate_with_metadata(
        str(path), backend=backend, format_output=False, source_options=options
    )


def _stores(code, image):
    return re.findall(rf"imageStore\({image}, [^;]*;", code)


def test_unorm_stores_use_the_scales_of_their_register(tmp_path):
    code = _translate(tmp_path, HLSL, "cgl").code

    color = _stores(code, "color")
    assert len(color) == 3
    assert all("crosstl_unorm_store_scale_u0))" in store for store in color)
    assert "crosstl_unorm_store_scale_u1_space2" in _stores(code, "pair")[0]
    assert "crosstl_unorm_store_scale_u2" in _stores(code, "single")[0]
    # Float textures keep the target conversion.
    assert "crosstl_hlsl_unorm_store" not in _stores(code, "plain")[0]
    # The scale buffers follow the highest b register of space 0, in the
    # order of the first store to each texture.
    for register, suffix in zip((4, 5, 6), ("u0", "u1_space2", "u2")):
        assert (
            f"@ register(b{register})\n    @ buffer_layout(hlsl_cbuffer)\n"
            f"    cbuffer crosstl_unorm_store_{suffix} {{"
        ) in code
        assert f"vec4 crosstl_unorm_store_scale_{suffix};" in code
    assert code.count("float crosstl_hlsl_unorm_store_channel(") == 1


def test_option_defaults_to_the_target_conversion(tmp_path):
    for options in (None, {"unorm_store_rounding": "target"}):
        code = _translate(tmp_path, HLSL, "cgl", options).code
        assert "crosstl_unorm_store" not in code
        assert "crosstl_hlsl_unorm_store" not in code


def test_invalid_option_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unorm_store_rounding must be one of"):
        _translate(tmp_path, HLSL, "cgl", {"unorm_store_rounding": "exact"})


@pytest.mark.parametrize("target", ["metal", "opengl"])
def test_targets_report_scale_buffers_by_name(tmp_path, target):
    result = _translate(tmp_path, HLSL, target)
    records = {record["name"]: record for record in result.resource_bindings}
    scale_records = [name for name in records if name.startswith("crosstl_unorm")]
    assert sorted(scale_records) == [
        "crosstl_unorm_store_u0",
        "crosstl_unorm_store_u1_space2",
        "crosstl_unorm_store_u2",
    ]
    slots = [
        (record["namespace"], record["binding"])
        for record in result.resource_bindings
        if record["kind"] == "constant-buffer"
    ]
    assert len(slots) == len(set(slots)) == 6
    for register, name in zip(("b4", "b5", "b6"), sorted(scale_records)):
        assert records[name]["kind"] == "constant-buffer"
        assert records[name]["register"] == register
    if target == "metal":
        # Halfway cases round to even; Metal's round() rounds them away.
        assert "rint(scaled) / scale" in result.code
        _compile(result.code, "metal", tmp_path)
    elif shutil.which("glslangValidator"):
        shader = tmp_path / "unorm.comp"
        shader.write_text(result.code, encoding="utf-8")
        completed = subprocess.run(
            ["glslangValidator", "-S", "comp", str(shader)],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "source, message",
    [
        (
            "RWTexture2D<unorm float4> t;\n"
            "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID)"
            " { t[id.xy] = 0.5; }",
            "explicit u register",
        ),
        (
            "RWTexture2D<unorm float4> t[2] : register(u0);\n"
            "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID)"
            " { t[id.z][id.xy] = 0.5; }",
            "global UNORM storage texture",
        ),
        (
            "RWTexture2D<unorm float4> t : register(u0);\n"
            "void write(RWTexture2D<unorm float4> t, uint2 p) { t[p] = 0.5; }\n"
            "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID)"
            " { write(t, id.xy); }",
            "parameter or local",
        ),
        (
            "RWTexture2D<unorm float4> t : register(u0);\n"
            "float crosstl_unorm_store_scale_u0;\n"
            "[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID)"
            " { t[id.xy] = 0.5; }",
            "already declares",
        ),
    ],
)
def test_stores_without_a_bindable_scale_fail_closed(tmp_path, source, message):
    with pytest.raises(HLSLUnormStoreRoundingError, match=message):
        _translate(tmp_path, source, "cgl")


@pytest.mark.parametrize(
    "register, suffix",
    [
        ("u0", "u0"),
        ("u7, space0", "u7"),
        ("u3, space1", "u3_space1"),
        (" U12 ", "u12"),
        ("t0", None),
        (None, None),
    ],
)
def test_scale_buffer_names(register, suffix):
    assert scale_buffer_suffix(register) == suffix


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _direct3d_unorm(value, bits):
    """Reference: saturate, scale in single precision, round half to even."""
    if value != value:
        return 0
    scale = float(2**bits - 1)
    return round(_f32(min(max(value, 0.0), 1.0) * scale))


def _quantized(value, bits):
    """The helper, evaluated in single precision as the targets do."""
    scale = float(2**bits - 1)
    if value != value:
        return 0.0
    scaled = _f32(_f32(min(max(value, 0.0), 1.0)) * scale)
    return _f32(round(scaled) / scale)


@pytest.mark.parametrize("bits", [8, 16])
def test_quantized_values_convert_to_the_direct3d_level(bits):
    """Rounding the exact product of the quantized value gives Direct3D's level.

    Apple GPUs round the exact product of the stored float; the quantized
    level k / (2^n - 1) is within a few units in the last place of k / scale,
    so the exact product rounds to k.
    """
    scale = 2**bits - 1
    values = [_f32(0.3), _f32(0.6), 0.5, 1.0, 0.0, -0.25, 1.5, float("nan")]
    values += [_f32((k + 0.5) / scale) for k in range(0, scale, max(1, scale // 509))]
    for value in values:
        level = _direct3d_unorm(value, bits)
        stored = _quantized(value, bits)
        assert round(stored * scale) == level, (value, stored)
    # The case that motivates the option: 0.3f * 255 is 76.5 in single
    # precision, which rounds to 76, while the exact product rounds to 77.
    assert _direct3d_unorm(_f32(0.3), 8) == 76
    assert round(_f32(0.3) * 255) == 77
    assert round(_quantized(_f32(0.3), 8) * 255) == 76


def test_hlsl_round_is_round_half_to_even_on_metal(tmp_path):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader main { compute { @ numthreads(1, 1, 1)\n"
        "void roundValues(uvec3 id @ gl_GlobalInvocationID) @ stage_entry {\n"
        "float a = roundEven(2.5); float b = round(2.5); } } }\n",
        encoding="utf-8",
    )
    code = translate(str(path), backend="metal", format_output=False)
    assert "rint(2.5)" in code and "round(2.5)" in code
    _compile(code, "metal", tmp_path)
    hlsl = translate(str(path), backend="directx", format_output=False)
    assert "roundEven" not in hlsl and hlsl.count("round(2.5)") == 2
