"""Slang output declares a texture and the function parameters it is passed to as the same kind of texture.

A texture that is only passed to a function which samples its parameter with a separate sampler was declared as a
combined ``Sampler2D<float4>`` while the parameter was a separate ``Texture2D<float4>``, and slangc rejected the call.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """shader S {
    sampler2D tex;
    sampler2D other;
    sampler smp;
    float lookup(sampler2D t, sampler s, float x) {
        return textureLod(t, s, vec2(x, 0.5), 0.0).x;
    }
    float relay(sampler2D r, sampler s, float x) {
        return lookup(r, s, x);
    }
    float combined(sampler2D c, float x) {
        return textureLod(c, vec2(x, 0.5), 0.0).x;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = lookup(tex, smp, 0.25) + relay(tex, smp, 0.5) + combined(other, 0.75);
        }
    }
}
"""


def generate(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def test_arguments_take_the_kind_of_their_parameters(tmp_path):
    code = generate(tmp_path)
    assert "Texture2D<float4> tex" in code
    assert "float lookup(Texture2D<float4> t, SamplerState s, float x)" in code
    # relay samples nothing itself; it passes r on to lookup, which samples it separately.
    assert "float relay(Texture2D<float4> r, SamplerState s, float x)" in code
    # A texture sampled on its own stays combined.
    assert "Sampler2D<float4> other" in code
    assert "float combined(Sampler2D<float4> c, float x)" in code
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
