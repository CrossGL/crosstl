"""Writes to components of a vector in a byte-addressed GLSL buffer block store each component in Metal."""

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """#version 450
layout (local_size_x = 64) in;
struct Particle
{
    vec2 pos;
    vec2 vel;
    vec4 gradientPos;
};
layout (std140, binding = 0) buffer Particles { Particle particles[]; };
layout (binding = 1) uniform UBO { float deltaT; int component; } ubo;
void main()
{
    uint index = gl_GlobalInvocationID.x;
    particles[index].gradientPos.x = particles[index].gradientPos.x + 0.02 * ubo.deltaT;
    particles[index].gradientPos.yz += vec2(1.0, 2.0);
    particles[index].vel.y *= 0.5;
    particles[index].pos.yx = particles[index].pos;
    particles[index].gradientPos[ubo.component] = 3.0;
}
"""


def _translate(tmp_path):
    path = tmp_path / "shader.comp"
    path.write_text(COMPUTE)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_component_writes_are_scalar_stores(tmp_path):
    code = _translate(tmp_path)
    # No load is written as a store target.
    assert not re.search(r"\)\)\)\.[xyzw]+\s*[+*]?=", code)
    assert not re.search(r"float[234]\([^;]*\)\[[^\]]*\]\s*=", code)
    assert "__crossgl_component_store_0" in code


def test_component_writes_compile(tmp_path):
    code = _translate(tmp_path)
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
