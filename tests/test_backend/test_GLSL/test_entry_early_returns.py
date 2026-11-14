"""``return;`` in a GLSL ``void main()`` returns the stage outputs once the entry returns them.

The GLSL importer turns ``void main()`` with stage outputs into an entry that returns them, but kept an early
``return;`` as is, so Metal rejected the entry ("non-void function 'fragment_main' should return a value",
sascha-vulkan ``deferred.frag``) and the other targets were given a value-returning function without a value.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

FRAGMENT = """#version 450
layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform UBO { int mode; } ubo;
layout (location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(inUV, 0.0, 1.0);
    if (ubo.mode > 0) {
        outColor.a = 0.5;
        return;
    }
    outColor.rgb *= 2.0;
}
"""

VERTEX = """#version 450
layout (location = 0) in vec4 inPos;
layout (location = 0) out vec3 outColor;
void main()
{
    outColor = vec3(1.0);
    gl_Position = inPos;
    if (inPos.w < 0.0) {
        return;
    }
    outColor = inPos.xyz;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_fragment_early_return_returns_the_output(tmp_path):
    crossgl = _translate(tmp_path, FRAGMENT, "frag", "cgl")
    assert not re.search(r"return\s*;", crossgl)
    assert len(re.findall(r"return outColor;", crossgl)) == 2


def test_vertex_early_return_returns_the_output_struct(tmp_path):
    crossgl = _translate(tmp_path, VERTEX, "vert", "cgl")
    assert not re.search(r"return\s*;", crossgl)
    assert len(re.findall(r"return output;", crossgl)) == 2


@pytest.mark.parametrize(("source", "suffix"), [(FRAGMENT, "frag"), (VERTEX, "vert")])
def test_metal_entries_compile(tmp_path, source, suffix):
    code = _translate(tmp_path, source, suffix, "metal")
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


@pytest.mark.parametrize(("source", "suffix"), [(FRAGMENT, "frag"), (VERTEX, "vert")])
def test_glsl_round_trip_validates(tmp_path, source, suffix):
    code = _translate(tmp_path, source, suffix, "opengl")
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    artifact = tmp_path / f"out.{suffix}"
    artifact.write_text(code)
    result = subprocess.run(
        [glslang, str(artifact)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
