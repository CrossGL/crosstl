"""CrossGL texture residency queries lower to HLSL's ``out uint Status`` overloads and ``CheckAccessFullyMapped``.

Each query gets a ``uint`` Status local declared at the top of its function. The read gains the neutral optional
operands Status follows (a zero offset, cube textures excepted, and a ``0.0f`` clamp) and the code is
``CheckAccessFullyMapped(status) ? 0 : 1``, sequenced after the read by the comma operator.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.texture_residency import (
    TextureResidencyUnsupportedError,
)

SOURCE = """
shader Residency {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        uniform sampler2D colorMap;
        uniform sampler2DShadow shadowMap;
        uniform sampler2DArray layers;
        uniform samplerCube cube;
        uniform sampler3D volume;
        uniform sampler2DMS multisampled;
        uniform samplerBuffer texels;
        layout(rgba8) uniform image2D image;

        bool residentSample(vec2 uv, out vec4 texel) {
            return textureResident(textureResidency(texture(colorMap, uv), texel));
        }

        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            vec4 a; vec4 b; vec4 c; vec4 d; vec4 e; float s; vec4 l; vec4 f; vec4 g; vec4 h; vec4 i;
            vec4 j; vec4 k; vec4 m; float n; vec4 o; vec4 p; vec4 q; vec4 r;
            int code = textureResidency(texture(colorMap, input.uv), a);
            code |= textureResidency(texture(colorMap, input.uv, 0.5), b);
            code |= textureResidency(texelFetch(colorMap, ivec2(input.uv), 0), c);
            code |= textureResidency(textureGather(colorMap, input.uv, 1), d);
            code |= textureResidency(textureGradOffset(colorMap, input.uv, vec2(0.1), vec2(0.2), ivec2(1, 0)), e);
            code |= textureResidency(textureCompare(shadowMap, input.uv, 0.5), s);
            code |= textureResidency(textureLod(layers, vec3(input.uv, 1.0), 0.0), l);
            code |= textureResidency(texture(cube, vec3(input.uv, 1.0)), f);
            code |= textureResidency(textureOffset(volume, vec3(input.uv, 1.0), ivec3(1)), g);
            code |= textureResidency(textureClamp(colorMap, input.uv, 1.0), h);
            code |= textureResidency(textureGatherCompare(shadowMap, input.uv, 0.5), i);
            code |= textureResidency(textureLodOffset(colorMap, input.uv, 1.0, ivec2(1, 1)), j);
            code |= textureResidency(texelFetchOffset(colorMap, ivec2(input.uv), 0, ivec2(1, 1)), k);
            code |= textureResidency(textureCompareLod(shadowMap, input.uv, 0.5, 0.0), n);
            code |= textureResidency(texelFetch(multisampled, ivec2(input.uv), 1), o);
            code |= textureResidency(texelFetch(texels, int(input.uv.x)), p);
            code |= textureResidency(imageLoad(image, ivec2(input.uv)), q);
            code |= textureResidency(textureGather(cube, vec3(input.uv, 1.0)), r);
            for (int it = 0; it < 2 && textureResident(textureResidency(texture(colorMap, input.uv + float(it)), m)); it++) {
                b += m;
            }
            bool helper = residentSample(input.uv, m);
            vec4 sum = a + b + c + d + e + vec4(s) + l + f + g + h + i + j + k + m + vec4(n) + o + p + q + r;
            output.color = textureResident(code) && helper ? sum : vec4(0.0);
            return output;
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _compact(code):
    return " ".join(code.split())


def _compile(tmp_path, hlsl, entry="PSMain"):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "shader.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    # The comma operator sits where DXC does not take it for a constructor list.
    assert "warning" not in result.stderr, result.stderr[-2000:]


def test_reads_take_their_status_overloads(tmp_path):
    code = _translate(tmp_path, SOURCE)
    flat = _compact(code)
    for texel, read in (
        ("a", "colorMap.Sample(colorMapSampler, input.uv, int2(0, 0), 0.0f)"),
        ("b", "colorMap.SampleBias(colorMapSampler, input.uv, 0.5, int2(0, 0), 0.0f)"),
        ("c", "colorMap.Load(int3(int2(input.uv), 0), int2(0, 0))"),
        ("d", "colorMap.GatherGreen(colorMapSampler, input.uv, int2(0, 0))"),
        (
            "e",
            "colorMap.SampleGrad(colorMapSampler, input.uv, float2(0.1, 0.1), float2(0.2, 0.2), "
            "int2(1, 0), 0.0f)",
        ),
        ("s", "shadowMap.SampleCmp(shadowMapSampler, input.uv, 0.5, int2(0, 0), 0.0f)"),
        (
            "l",
            "layers.SampleLevel(layersSampler, float3(input.uv, 1.0), 0.0, int2(0, 0))",
        ),
        # Cube textures take no offset.
        ("f", "cube.Sample(cubeSampler, float3(input.uv, 1.0), 0.0f)"),
        (
            "g",
            "volume.Sample(volumeSampler, float3(input.uv, 1.0), int3(1, 1, 1), 0.0f)",
        ),
        ("h", "colorMap.Sample(colorMapSampler, input.uv, int2(0, 0), 1.0)"),
        ("i", "shadowMap.GatherCmp(shadowMapSampler, input.uv, 0.5, int2(0, 0))"),
        ("j", "colorMap.SampleLevel(colorMapSampler, input.uv, 1.0, int2(1, 1))"),
        (
            "n",
            "shadowMap.SampleCmpLevelZero(shadowMapSampler, input.uv, 0.5, int2(0, 0))",
        ),
        ("o", "multisampled.Load(int2(input.uv), 1, int2(0, 0))"),
        # Buffers and storage images take only the location before Status.
        ("p", "texels.Load(int(input.uv.x))"),
        ("q", "image.Load(int2(input.uv))"),
        ("r", "cube.Gather(cubeSampler, float3(input.uv, 1.0))"),
    ):
        assert f"({texel} = {read[:-1]}, crosstl_residency_status_" in flat, texel
    assert "int code = ((a = colorMap.Sample(" in flat
    assert "CheckAccessFullyMapped(crosstl_residency_status_" in flat
    assert "((code == 0) && helper)" in flat
    assert "textureResidency" not in code and "textureResident" not in code
    _compile(tmp_path, code)


def test_status_locals_are_declared_at_the_top_of_each_function(tmp_path):
    flat = _compact(_translate(tmp_path, SOURCE))
    helper = flat[flat.index("bool residentSample(") :]
    assert (
        helper.split("{", 1)[1].lstrip().startswith("uint crosstl_residency_status_0;")
    )
    entry = flat[flat.index(" PSMain(") :]
    body = entry.split("{", 1)[1]
    declarations = [f"uint crosstl_residency_status_{n};" for n in range(1, 20)]
    assert body.lstrip().startswith(" ".join(declarations))
    # A query in a loop condition reads the local declared before the loop.
    assert (
        "it < 2" in entry
        and "crosstl_residency_status_19), CheckAccessFullyMapped(" in entry
    )


def test_status_names_avoid_program_identifiers(tmp_path):
    source = SOURCE.replace(
        "FragmentOutput output;",
        "FragmentOutput output; uint crosstl_residency_status_0 = 1u;",
    )
    flat = _compact(_translate(tmp_path, source))
    helper = flat[flat.index("bool residentSample(") :]
    assert (
        helper.split("{", 1)[1].lstrip().startswith("uint crosstl_residency_status_1;")
    )
    assert "uint crosstl_residency_status_0 = 1u;" in flat


def test_resident_test_takes_one_code(tmp_path):
    source = SOURCE.replace("textureResident(code)", "textureResident(code, 1)")
    with pytest.raises(TextureResidencyUnsupportedError, match="one residency code"):
        _translate(tmp_path, source)


def test_hlsl_aliases_lower_residency(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE, encoding="utf-8")
    code = crosstl.translate(str(path), backend="dx12", format_output=False)
    assert "CheckAccessFullyMapped(crosstl_residency_status_" in code
