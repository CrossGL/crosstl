"""GLSL fragment shaders with several outputs return them as one structure in HLSL and MSL."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import StageInterfaceUnsupportedError

GBUFFER = """#version 450
layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
float linearDepth(float d) { return d * 2.0; }
void writeNormal(vec3 n) { outNormal = vec4(normalize(n), 1.0); }
void main() {
    outPosition = vec4(inUV, 0.0, 1.0);
    outPosition.a = linearDepth(gl_FragCoord.z);
    writeNormal(inNormal);
    outAlbedo = vec4(outNormal.xyz * 0.5, 1.0);
}
"""


def _translate(tmp_path, source, backend, name="shader.frag"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_returns_every_fragment_output_in_one_structure(tmp_path):
    hlsl = _translate(tmp_path, GBUFFER, "directx")

    struct = re.search(r"struct FragmentOutput\s*\{([^}]*)\}", hlsl)
    assert struct, hlsl
    for index, name in enumerate(("outPosition", "outNormal", "outAlbedo")):
        assert re.search(
            rf"float4 {name}\s*:\s*SV_TARGET{index};", struct.group(1)
        ), hlsl
    assert re.search(r"FragmentOutput PSMain\(FragmentInput input", hlsl), hlsl
    # A helper that writes an output takes the structure inout; reads go through it too.
    assert re.search(
        r"void writeNormal\(float3 n, inout FragmentOutput \w+\)", hlsl
    ), hlsl
    assert "output.outAlbedo = float4((output.outNormal.xyz * 0.5), 1.0);" in hlsl
    assert re.search(r"return output;", hlsl)
    assert not re.search(r"^float4 outPosition;", hlsl, re.M)
    if shutil.which("dxc"):
        source = tmp_path / "out.hlsl"
        source.write_text(hlsl)
        _run(["dxc", "-T", "ps_6_0", "-E", "PSMain", str(source)])


def test_metal_returns_every_fragment_output_in_one_structure(tmp_path):
    metal = _translate(tmp_path, GBUFFER, "metal")

    for index, name in enumerate(("outPosition", "outNormal", "outAlbedo")):
        assert re.search(rf"float4 {name} \[\[color\({index}\)\]\];", metal), metal
    assert re.search(
        r"fragment FragmentOutput fragment_main\(FragmentInput input \[\[stage_in\]\]",
        metal,
    )
    if shutil.which("xcrun") and (
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
        ).returncode
        == 0
    ):
        source = tmp_path / "out.metal"
        source.write_text(metal)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_targets_with_module_outputs_keep_them(tmp_path, backend):
    code = _translate(tmp_path, GBUFFER, backend)

    assert "FragmentOutput" not in code


def test_module_outputs_of_several_graphics_stages_are_refused(tmp_path):
    source = """
    shader main {
        out vec4 colour @location(0);
        vertex {
            vec4 main(vec4 position @location(0)) @ gl_Position { return position; }
        }
        fragment {
            void main() { colour = vec4(1.0); }
        }
    }
    """
    with pytest.raises(
        StageInterfaceUnsupportedError, match="no single vertex or fragment stage"
    ):
        _translate(tmp_path, source, "directx", "shader.cgl")
