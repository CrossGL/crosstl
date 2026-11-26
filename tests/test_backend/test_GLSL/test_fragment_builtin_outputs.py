"""GLSL fragment shaders that write depth, the sample mask or the stencil reference import with stage-level outputs.

Those built-ins stay stage-level writes in CrossGL, so the colour outputs must stay ``out`` declarations as well:
returning the colour from ``main`` instead would mix a returned entry interface with stage-level writes, which
the structured-interface targets (Metal, HLSL, Slang) refuse.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

BUILTIN_WRITES = [
    pytest.param(
        "#extension GL_ARB_shader_stencil_export : require",
        "gl_FragStencilRefARB = int(uv.x * 255.0);",
        {"metal": "[[stencil]]", "directx": "SV_StencilRef"},
        id="stencil-ref-arb",
    ),
    pytest.param(
        "",
        "gl_FragDepth = uv.x;",
        {"metal": "[[depth(any)]]", "directx": "SV_DEPTH"},
        id="depth",
    ),
    pytest.param(
        "",
        "gl_SampleMask[0] = 1;",
        {"metal": "[[sample_mask]]", "directx": "SV_Coverage"},
        id="sample-mask",
    ),
]


def import_glsl(source):
    register_default_sources()
    spec = SOURCE_REGISTRY.get("glsl")
    ast = spec.parse(textwrap.dedent(source), file_path="/tmp/builtin-output.frag")
    return spec.reverse_codegen_factory().generate(ast)


def fragment_source(extension, builtin_write, colour):
    colour_declaration = "layout(location = 0) out vec4 color;" if colour else ""
    colour_write = "color = vec4(uv, 0.0, 1.0);" if colour else ""
    return f"""\
#version 460
{extension}
layout(location = 0) in vec2 uv;
{colour_declaration}
void main() {{
    {colour_write}
    {builtin_write}
}}
"""


def translate_glsl(tmp_path, source, backend):
    path = tmp_path / "builtin-output.frag"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(("extension", "builtin_write", "spellings"), BUILTIN_WRITES)
def test_colour_output_stays_declared_beside_builtin_write(
    extension, builtin_write, spellings
):
    crossgl = import_glsl(fragment_source(extension, builtin_write, colour=True))

    assert "out vec4 color @location(0);" in crossgl
    assert "void main(FragmentInput input)" in crossgl
    assert "vec4 main(" not in crossgl
    assert "return color;" not in crossgl


@pytest.mark.parametrize(("extension", "builtin_write", "spellings"), BUILTIN_WRITES)
def test_builtin_write_alone_invents_no_colour_output(
    extension, builtin_write, spellings
):
    crossgl = import_glsl(fragment_source(extension, builtin_write, colour=False))

    assert "void main(FragmentInput input)" in crossgl
    assert "gl_FragColor" not in crossgl
    assert "return" not in crossgl


@pytest.mark.parametrize("colour", [True, False], ids=["with-colour", "alone"])
@pytest.mark.parametrize(("extension", "builtin_write", "spellings"), BUILTIN_WRITES)
@pytest.mark.parametrize("backend", ["metal", "directx", "slang"])
def test_structured_targets_translate_builtin_writes(
    tmp_path, backend, extension, builtin_write, spellings, colour
):
    output = translate_glsl(
        tmp_path, fragment_source(extension, builtin_write, colour), backend
    )

    if backend in spellings:
        assert spellings[backend] in output
    if backend == "metal":
        assert ("[[color(0)]]" in output) == colour
    if backend == "directx":
        assert ("SV_TARGET0" in output) == colour


def test_stencil_ref_alone_round_trips_to_glsl_without_colour_output(tmp_path):
    source = fragment_source(
        "#extension GL_ARB_shader_stencil_export : require",
        "gl_FragStencilRefARB = int(uv.x * 255.0);",
        colour=False,
    )

    glsl = translate_glsl(tmp_path, source, "opengl")

    assert "gl_FragStencilRefARB = int((uv.x * 255.0));" in glsl
    assert " out vec4" not in glsl
