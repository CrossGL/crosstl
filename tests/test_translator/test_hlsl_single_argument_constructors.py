"""A one-argument HLSL vector constructor of an argument of unknown width is a cast."""

import shutil
import subprocess

import crosstl

SOURCE = """#version 450
layout(location = 0) in vec3 inN;
layout(location = 1) in vec3 inV;
layout(location = 0) out vec4 outColor;
void main() {
    vec3 R = reflect(-inV, inN);
    vec3 specular = vec3(pow(max(dot(R, inV), 0.0), 8.0));
    vec3 same = vec3(inN);
    outColor = vec4(specular + same, 1.0);
}
"""


def test_scalar_splats_of_unknown_width_are_casts(tmp_path):
    path = tmp_path / "shader.frag"
    path.write_text(SOURCE)
    hlsl = crosstl.translate(str(path), backend="directx")
    # float3(scalar) is not HLSL ("too few elements in vector initialization"); the cast splats it.
    assert (
        "float3 specular = ((float3)(pow(max(dot(R, input.inV), 0.0), 8.0)));" in hlsl
    )
    assert "float3 same = float3(input.inN);" in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
