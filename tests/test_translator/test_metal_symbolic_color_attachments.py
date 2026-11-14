"""Metal colour attachments indexed by enumerators keep their render target on every target.

Metal samples index G-buffer attachments with enumerators of a header shared with the host
(``[[color(RenderTargetAlbedo)]]``). The importer only read integer literals and imported every
other index as ``@gl_FragColor``, so all attachments moved to render target 0 without a diagnostic.
A symbolic output index now becomes ``@location(name)``, which the binding-constant pass folds.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.color_attachments import MetalColorAttachmentIndexError

GBUFFER = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    typedef enum RenderTargetIndex {
        RenderTargetLighting = 0,
        RenderTargetAlbedo = 1,
        RenderTargetDepth = 3
    } RenderTargetIndex;
    struct GBufferData {
        half4 lighting [[color(RenderTargetLighting)]];
        half4 albedo [[color(RenderTargetAlbedo)]];
        float4 depth [[color(RenderTargetDepth)]];
    };
    fragment GBufferData gbuffer(float4 pos [[position]]) {
        GBufferData g;
        g.lighting = half4(1.0h);
        g.albedo = half4(0.5h);
        g.depth = float4(pos.z);
        return g;
    }
    """)

FRAMEBUFFER_FETCH = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    enum AttachmentIndex { AttachmentLighting, AttachmentAlbedo = 0x2, AttachmentNormal };
    fragment float4 light(float4 albedo [[color(AttachmentAlbedo)]],
                          float4 normal [[color(AttachmentNormal)]]) {
        return albedo * normal;
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _spirv_location(code, member):
    ids = re.findall(rf'OpName (%\S+) "CrossGL_fragment_output_gbuffer_{member}"', code)
    for identifier in ids:
        match = re.search(rf"OpDecorate {re.escape(identifier)} Location (\d+)", code)
        if match:
            return int(match.group(1))
    return None


def test_symbolic_output_index_imports_as_a_location(tmp_path):
    code = _translate(tmp_path, GBUFFER, "cgl")
    assert re.search(r"lighting @location\(RenderTargetLighting\);", code), code
    assert re.search(r"albedo @location\(RenderTargetAlbedo\);", code), code
    assert re.search(r"depth @location\(RenderTargetDepth\);", code), code
    assert "@gl_FragColor" not in code


@pytest.mark.parametrize(
    ("backend", "patterns"),
    [
        (
            "metal",
            [
                r"lighting \[\[color\(0\)\]\]",
                r"albedo \[\[color\(1\)\]\]",
                r"depth \[\[color\(3\)\]\]",
            ],
        ),
        (
            "opengl",
            [
                r"layout\(location = 0\) out vec4 lighting;",
                r"layout\(location = 1\) out vec4 albedo;",
                r"layout\(location = 3\) out vec4 depth;",
            ],
        ),
        (
            "directx",
            [
                r"lighting\s*:\s*SV_TARGET;",
                r"albedo\s*:\s*SV_TARGET1;",
                r"depth\s*:\s*SV_TARGET3;",
            ],
        ),
    ],
)
def test_symbolic_output_index_keeps_its_render_target(tmp_path, backend, patterns):
    code = _translate(tmp_path, GBUFFER, backend)
    for pattern in patterns:
        assert re.search(pattern, code), (pattern, code)


def test_symbolic_output_index_keeps_its_spirv_location(tmp_path):
    code = _translate(tmp_path, GBUFFER, "vulkan")
    assert [
        _spirv_location(code, member) for member in ("lighting", "albedo", "depth")
    ] == [0, 1, 3], code


def test_framebuffer_fetch_resolves_a_declared_enumerator(tmp_path):
    code = _translate(tmp_path, FRAMEBUFFER_FETCH, "cgl")
    assert re.search(r"vec4 albedo @gl_FragColor2\b", code), code
    assert re.search(r"vec4 normal @gl_FragColor3\b", code), code
    metal = _translate(tmp_path, FRAMEBUFFER_FETCH, "metal")
    assert re.search(r"albedo \[\[color\(2\)\]\]", metal), metal
    assert re.search(r"normal \[\[color\(3\)\]\]", metal), metal


@pytest.mark.parametrize(
    "declaration",
    [
        "",
        "enum AttachmentIndex { AttachmentAlbedo = 1 + 1 };",
        "enum A { AttachmentAlbedo = 1 }; enum B { AttachmentAlbedo = 2 };",
        "enum class AttachmentIndex { AttachmentAlbedo = 2 };",
    ],
    ids=["undeclared", "expression", "ambiguous", "scoped"],
)
def test_framebuffer_fetch_with_an_unresolved_index_is_refused(tmp_path, declaration):
    source = textwrap.dedent(f"""
        #include <metal_stdlib>
        using namespace metal;
        {declaration}
        fragment float4 light(float4 albedo [[color(AttachmentAlbedo)]]) {{
            return albedo;
        }}
        """)
    with pytest.raises(MetalColorAttachmentIndexError, match="AttachmentAlbedo"):
        _translate(tmp_path, source, "cgl")


def test_directx_fragment_output_locations_take_their_render_target(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(
        textwrap.dedent("""
            shader main {
                struct Targets {
                    vec4 first;
                    vec4 second @location(2);
                    vec4 third @location(0);
                }
                fragment {
                    Targets main(vec4 pos @gl_FragCoord) {
                        Targets t;
                        t.first = pos;
                        t.second = pos;
                        t.third = pos;
                        return t;
                    }
                }
            }
            """),
        encoding="utf-8",
    )
    code = crosstl.translate(str(path), backend="directx", format_output=False)
    assert re.search(r"second\s*:\s*SV_TARGET2;", code), code
    assert re.search(r"third\s*:\s*SV_TARGET;", code), code
    # The member without a location takes the first render target left free.
    assert re.search(r"first\s*:\s*SV_TARGET1;", code), code
    assert ": location" not in code


def test_directx_half_precision_render_targets_are_accepted(tmp_path):
    code = _translate(tmp_path, GBUFFER, "directx")
    assert re.search(r"float16_t4 lighting\s*:\s*SV_TARGET;", code), code


def test_gbuffer_compiles_with_the_native_compilers(tmp_path):
    if shutil.which("xcrun"):
        metal = _translate(tmp_path, GBUFFER, "metal")
        (tmp_path / "a.metal").write_text(metal, encoding="utf-8")
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr[-2000:]
    if shutil.which("glslangValidator"):
        glsl = _translate(tmp_path, GBUFFER, "opengl")
        (tmp_path / "a.frag").write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            ["glslangValidator", "a.frag"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout[-2000:]
    if not (shutil.which("xcrun") or shutil.which("glslangValidator")):
        pytest.skip("neither the Metal toolchain nor glslang is installed")
