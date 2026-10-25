"""Framebuffer-fetch structures whose ``[[color(n)]]`` members name the attachment with an enumerator.

A Metal fragment function may take a structure that is not its ``[[stage_in]]`` input; ``[[color(n)]]`` on its
members then reads attachment ``n`` back (MSL section 5.2.3.4). Apple's deferred-lighting samples spell ``n`` with
an enumerator of a header shared with the host. The importer resolved enumerators only on parameters, so these
members became ``@location(name)``: ordinary varyings. Every target then lost the framebuffer fetch, and SPIR-V
put the first attachment and the first varying both at input location 0 and failed with an untyped ``ValueError``.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.color_attachments import (
    MetalColorAttachmentIndexError,
    framebuffer_fetch_struct_names,
)
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_location_conflicts import (
    InterfaceLocationConflictError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.colour_attachment_inputs import (
    ColourAttachmentInputUnsupportedError,
)

SINGLE_PASS = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    typedef enum RenderTargetIndex {
        RenderTargetLighting = 0,
        RenderTargetAlbedo = 1,
        RenderTargetDepth = 3
    } RenderTargetIndex;
    struct QuadInOut {
        float4 position [[position]];
        float3 eye_position;
    };
    struct GBufferData {
        half4 lighting [[color(RenderTargetLighting)]];
        half4 albedo [[color(RenderTargetAlbedo)]];
        float depth [[color(RenderTargetDepth)]];
    };
    struct AccumLightBuffer {
        half4 lighting [[color(RenderTargetLighting)]];
    };
    fragment AccumLightBuffer shade(QuadInOut in [[stage_in]], GBufferData gbuffer) {
        AccumLightBuffer output;
        output.lighting = gbuffer.lighting + gbuffer.albedo * half(gbuffer.depth)
            + half4(half3(in.eye_position), 0.0h);
        return output;
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_fetched_members_resolve_their_enumerators(tmp_path):
    code = _translate(tmp_path, SINGLE_PASS, "cgl")
    assert re.search(r"lighting @gl_FragColor;", code), code
    assert re.search(r"albedo @gl_FragColor1;", code), code
    assert re.search(r"depth @gl_FragColor3;", code), code
    # The output structure keeps its symbolic render target.
    assert re.search(r"lighting @location\(RenderTargetLighting\);", code), code


@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx"])
def test_targets_without_framebuffer_fetch_refuse_the_fetch(tmp_path, backend):
    with pytest.raises(ColourAttachmentInputUnsupportedError, match="gbuffer"):
        _translate(tmp_path, SINGLE_PASS, backend)


def test_metal_keeps_the_fetched_attachments(tmp_path):
    code = _translate(tmp_path, SINGLE_PASS, "metal")
    struct = re.search(r"struct GBufferData \{(.*?)\};", code, re.S)
    assert struct, code
    members = struct.group(1)
    assert re.search(r"lighting \[\[color\(0\)\]\]", members), members
    assert re.search(r"albedo \[\[color\(1\)\]\]", members), members
    assert re.search(r"depth \[\[color\(3\)\]\]", members), members
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            xcrun,
            "metal",
            "-std=metal3.0",
            "-c",
            str(shader),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def test_fetched_member_with_an_unresolved_index_is_refused(tmp_path):
    source = SINGLE_PASS.replace(
        "[[color(RenderTargetAlbedo)]]", "[[color(RenderTargetAlbedo + 1)]]"
    )
    with pytest.raises(MetalColorAttachmentIndexError, match="RenderTargetAlbedo"):
        _translate(tmp_path, source, "cgl")


class _Function:
    def __init__(self, qualifier, *params):
        self.qualifier = qualifier
        self.params = list(params)


class _Param:
    def __init__(self, vtype, *attributes):
        self.vtype = vtype
        self.attributes = [
            type("Attribute", (), {"name": name})() for name in attributes
        ]


def test_only_non_stage_in_fragment_parameters_are_fetch_structures():
    functions = [
        _Function("fragment", _Param("Varyings", "stage_in"), _Param("const GBuffer")),
        _Function("vertex", _Param("Other")),
        _Function("fragment", _Param("float4", "color")),
        _Function(None, _Param("Helper")),
    ]
    names = framebuffer_fetch_struct_names(
        functions, ["Varyings", "GBuffer", "Other", "Helper"]
    )
    assert names == {"GBuffer"}


def test_spirv_location_conflicts_are_typed():
    source = """
    shader Inputs {
        vertex {
            vec4 main(vec4 a @location(0), vec4 b @location(0)) @gl_Position {
                return a + b;
            }
        }
    }
    """
    with pytest.raises(InterfaceLocationConflictError) as caught:
        VulkanSPIRVCodeGen().generate(Parser(Lexer(source).tokens).parse())
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.project_diagnostic_code == (
        "project.translate.vulkan-interface-location-conflict"
    )
    assert "Duplicate SPIR-V input location 0" in str(error)
