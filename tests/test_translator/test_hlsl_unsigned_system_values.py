"""HLSL entry points receive the vertex, instance, primitive and sample indices as uint, as DXIL requires."""

import re
import shutil
import subprocess

import pytest

import crosstl

METAL_VERTEX = """#include <metal_stdlib>
using namespace metal;
vertex float4 vmain(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    return float4(float(vid), float(iid), 0.0, 1.0);
}
"""

CROSSGL_FRAGMENT = """
shader main {
    fragment {
        vec4 main(int prim @ gl_PrimitiveID, int sample @ gl_SampleID) @ gl_FragColor {
            return vec4(float(prim), float(sample), 0.0, 1.0);
        }
    }
}
"""


def _translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _compile(tmp_path, hlsl, profile, entry):
    if shutil.which("dxc") is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(hlsl)
    result = subprocess.run(
        ["dxc", "-T", profile, "-E", entry, str(source)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source, name, semantics, profile, entry, signed",
    [
        # MSL's [[vertex_id]] and [[instance_id]] are uint, and the import keeps them uint.
        (
            METAL_VERTEX,
            "shader.metal",
            ("SV_VertexID", "SV_InstanceID"),
            "vs_6_0",
            "VSMain",
            False,
        ),
        (
            CROSSGL_FRAGMENT,
            "shader.cgl",
            ("SV_PrimitiveID", "SV_SampleIndex"),
            "ps_6_0",
            "PSMain",
            True,
        ),
    ],
)
def test_unsigned_system_values_arrive_as_uint(
    tmp_path, source, name, semantics, profile, entry, signed
):
    hlsl = _translate(tmp_path, source, name)

    for semantic in semantics:
        parameter = re.search(rf"(\w+) (\w+)\s*:\s*{semantic}\b", hlsl)
        assert parameter, hlsl
        assert parameter.group(1) == "uint", hlsl
        # An int parameter keeps reading the int it read before; a uint one reads the system value directly.
        converted = re.search(rf"int \w+ = int\({parameter.group(2)}\);", hlsl)
        assert bool(converted) == signed, hlsl
    _compile(tmp_path, hlsl, profile, entry)
