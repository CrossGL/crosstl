"""Reference-typed values are accepted as Metal texture call arguments.

MSL passes small constant buffers as references, as in ``constant float& bias [[buffer(0)]]``. The CrossGL import
keeps the reference type, and the Metal generator checked texture call arguments against the declared type, so
``t.sample(s, uv, level(bias))`` was rejected because ``float&`` is not a scalar. A reference reads as the value it
refers to; the checks now compare that value type.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.ast import PrimitiveType, ReferenceType
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

REFERENCE_ARGUMENTS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct V { float4 p [[position]]; float2 uv; };

    fragment float4 f(V in [[stage_in]],
                      texture2d<float> t [[texture(0)]],
                      depth2d<float> d [[texture(1)]],
                      sampler s [[sampler(0)]],
                      sampler cs [[sampler(1)]],
                      constant float & mipBias [[buffer(0)]],
                      constant float & ref [[buffer(1)]],
                      constant float2 & offsetUV [[buffer(2)]],
                      constant int2 & texel [[buffer(3)]]) {
        float4 a = t.sample(s, in.uv, level(mipBias));
        float4 b = t.sample(s, in.uv + offsetUV, bias(mipBias));
        float4 c = t.read(uint2(texel), 0);
        float e = d.sample_compare(cs, offsetUV, ref);
        return a + b + c + e;
    }
    """)

MISTYPED_REFERENCE_LOD = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct V { float4 p [[position]]; float2 uv; };

    fragment float4 f(V in [[stage_in]],
                      texture2d<float> t [[texture(0)]],
                      sampler s [[sampler(0)]],
                      constant float2 & lod [[buffer(0)]]) {
        return t.sample(s, in.uv, level(lod));
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "references.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, *command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("float&", "float"),
        ("float2 & ", "float2"),
        ("float", "float"),
        (None, None),
    ],
)
def test_referenced_value_type_of_type_names(value, expected):
    assert MetalCodeGen.metal_referenced_value_type(value) == expected


def test_referenced_value_type_of_reference_nodes():
    referent = PrimitiveType("float")

    assert MetalCodeGen.metal_referenced_value_type(ReferenceType(referent)) is referent
    assert MetalCodeGen.metal_referenced_value_type(referent) is referent


def test_reference_arguments_translate_to_metal(tmp_path):
    metal = _translate(tmp_path, REFERENCE_ARGUMENTS, "metal")

    assert "t.sample(s, in_.uv, level(mipBias))" in metal
    assert "bias(mipBias)" in metal
    assert "d.sample_compare(cs, offsetUV, ref)" in metal
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not installed")
    (tmp_path / "a.metal").write_text(metal, encoding="utf-8")
    _run(tmp_path, "xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air")


@pytest.mark.parametrize("backend", ["opengl", "directx", "vulkan"])
def test_reference_arguments_translate_to_other_targets(tmp_path, backend):
    assert _translate(tmp_path, REFERENCE_ARGUMENTS, backend)


def test_mistyped_reference_lod_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="scalar numeric lod argument: lod has type"):
        _translate(tmp_path, MISTYPED_REFERENCE_LOD, "metal")
