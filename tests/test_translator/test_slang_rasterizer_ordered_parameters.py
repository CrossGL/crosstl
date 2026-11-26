"""Slang parameters that receive a rasterizer-ordered resource take the rasterizer-ordered type.

slangc converts neither a ``RasterizerOrderedTexture2D`` to a ``RWTexture2D`` parameter nor the reverse (E30019), so
a helper whose parameter receives a ``@rasterizer_ordered`` resource, directly or through another helper, declares
the ``RasterizerOrdered*`` type, and an explicit ``@rasterizer_ordered`` parameter does too instead of writing the
attribute as a semantic. A parameter that receives both an ordered and a plain resource is refused.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_rasterizer_order import (
    rasterizer_ordered_parameters,
)
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

THROUGH_HELPERS = """shader main {
    image2D accum @rasterizer_ordered @rgba8 @register(u1);
    image2D plain @rgba8 @register(u2);
    void inner(image2D target, uvec2 p, vec4 c) {
        imageStore(target, p, imageLoad(target, p) * 0.5 + c);
    }
    void outer(image2D target, uvec2 p, vec4 c) {
        inner(target, p, c);
    }
    void tint(image2D target @rasterizer_ordered, uvec2 p) {
        imageStore(target, p, imageLoad(target, p) * 2.0);
    }
    void other(image2D target, uvec2 p) {
        imageStore(target, p, vec4(1.0));
    }
    fragment {
        vec4 main(vec4 coord @ gl_FragCoord) @ gl_FragColor {
            uvec2 p = uvec2(coord.xy);
            outer(accum, p, vec4(1.0));
            tint(accum, p);
            other(plain, p);
            return vec4(1.0);
        }
    }
}
"""

MIXED = """shader main {
    image2D accum @rasterizer_ordered @rgba8 @register(u1);
    image2D plain @rgba8 @register(u2);
    void blend(image2D target, uvec2 p) {
        imageStore(target, p, imageLoad(target, p) * 0.5);
    }
    fragment {
        vec4 main(vec4 coord @ gl_FragCoord) @ gl_FragColor {
            uvec2 p = uvec2(coord.xy);
            blend(accum, p);
            blend(plain, p);
            return vec4(1.0);
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang", format_output=False)


def _ast(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_the_parameters_that_receive_an_ordered_resource_are_found():
    parameters = rasterizer_ordered_parameters(_ast(THROUGH_HELPERS))

    assert sorted(parameters.ordered) == [
        ("inner", "target"),
        ("outer", "target"),
        ("tint", "target"),
    ]
    assert not parameters.mixed


def test_helper_parameters_declare_the_rasterizer_ordered_type(tmp_path):
    code = _translate(tmp_path, THROUGH_HELPERS)

    assert "RasterizerOrderedTexture2D<unorm float4> accum" in code
    assert "void inner(RasterizerOrderedTexture2D<unorm float4> target" in code
    assert "void outer(RasterizerOrderedTexture2D<unorm float4> target" in code
    assert "void tint(RasterizerOrderedTexture2D<unorm float4> target, uint2 p)" in code
    assert "void other(RWTexture2D<unorm float4> target" in code
    assert "rasterizer_ordered" not in code


def test_slangc_compiles_the_ordered_helpers(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    code = _translate(tmp_path, THROUGH_HELPERS)
    (tmp_path / "out.slang").write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            slangc,
            "out.slang",
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "fragment",
            "-o",
            "out.spv",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def test_a_parameter_receiving_ordered_and_plain_resources_is_refused(tmp_path):
    parameters = rasterizer_ordered_parameters(_ast(MIXED))
    assert parameters.mixed == {("blend", "target")}

    with pytest.raises(
        SlangLoweringRefused, match="receives both a rasterizer-ordered and a plain"
    ):
        _translate(tmp_path, MIXED)
