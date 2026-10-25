"""A 16-bit floating depth reference is a valid texture-compare argument on every target.

MSL ``sample_compare`` takes a ``float`` reference, and a ``half`` argument converts to it implicitly (MSL 3.2
section 6.12.3), so a varying declared ``half`` is a legal compare value. The importer types it as CrossGL
``float16``, which HLSL spells ``float16_t`` under native 16-bit types; both are scalar floats, so the compare must
translate rather than be rejected as a non-floating argument.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.validation import is_floating_scalar_type, is_numeric_scalar_type

SOURCE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; half shadow_depth; float2 uv; };
fragment float4 frag_main(VOut in [[stage_in]], depth2d<float> shadowMap [[texture(0)]],
                          sampler s [[sampler(0)]]) {
    float lit = shadowMap.sample_compare(s, in.uv, in.shadow_depth);
    return float4(lit);
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "shader.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("name", ["float16", "float16_t", "min16float", "half"])
def test_16_bit_float_names_are_floating_scalars(name):
    assert is_floating_scalar_type(name)
    assert is_numeric_scalar_type(name)


def test_hlsl_compares_against_a_half_reference(tmp_path):
    hlsl = translate(tmp_path, "directx")
    assert "shadowMap.SampleCmp(s, in_.uv, in_.shadow_depth)" in hlsl
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "shader.hlsl"
    path.write_text(hlsl)
    result = subprocess.run(
        [dxc, "-T", "ps_6_2", "-enable-16bit-types", "-E", "frag_main", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + hlsl


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal"])
def test_other_targets_compare_against_a_half_reference(tmp_path, backend):
    code = translate(tmp_path, backend)
    assert "shadow_depth" in code
