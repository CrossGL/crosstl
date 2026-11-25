"""The ``int`` coverage-mask input of GLSL and SPIR-V reaches Metal's ``uint [[sample_mask]]``.

GLSL declares ``int gl_SampleMaskIn[]`` and the SPIR-V importer writes ``int gl_SampleMaskIn[1] @input
@gl_SampleMaskIn``; Metal reads the mask as one ``uint`` on ``[[sample_mask]]``. The Metal generator refused both
``int`` forms. The entry point now takes the ``uint`` under a fresh name and a local of the declared type holds it.
"""

from __future__ import annotations

import re

import pytest

import crosstl
from crosstl.translator.refusals import MetalRefusal
from tests.metal_compute_dispatch import compile_metal_library

REQUIRE_ENV = "CROSSTL_REQUIRE_METAL_SAMPLE_MASK_INPUT_COMPILE"

SHADER = """
shader main {{
    fragment {{
        float4 color @output @location(0);
        float2 uv @input @location(0);
        {declaration} @input @gl_SampleMaskIn @flat;
        void main() {{
            color = float4(uv[0], uv[1], float(({read} & {one})), 1.0);
            return;
        }}
    }}
}}
"""


def _metal(tmp_path, declaration, read, one="1"):
    path = tmp_path / "mask.cgl"
    path.write_text(
        SHADER.format(declaration=declaration, read=read, one=one), encoding="utf-8"
    )
    return crosstl.translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize(
    "declaration,read,one,local",
    [
        (
            "int gl_SampleMaskIn[1]",
            "gl_SampleMaskIn[0]",
            "1",
            "int cgl_SampleMaskIn[1] = {int(_crossglSampleMaskIn)};",
        ),
        (
            "uint gl_SampleMaskIn[1]",
            "gl_SampleMaskIn[0]",
            "1u",
            "uint cgl_SampleMaskIn[1] = {_crossglSampleMaskIn};",
        ),
        (
            "int gl_SampleMaskIn",
            "gl_SampleMaskIn",
            "1",
            "int cgl_SampleMaskIn = int(_crossglSampleMaskIn);",
        ),
    ],
    ids=["int-array", "uint-array", "int"],
)
def test_coverage_mask_input_is_read_from_the_uint_attribute(
    tmp_path, declaration, read, one, local
):
    generated = _metal(tmp_path, declaration, read, one)

    assert re.search(r"uint _crossglSampleMaskIn\s*\[\[sample_mask\]\]", generated)
    assert local in generated
    compile_metal_library(generated, tmp_path, REQUIRE_ENV)


def test_a_mask_of_more_than_one_word_is_still_refused(tmp_path):
    with pytest.raises(MetalRefusal, match="sample_mask"):
        _metal(tmp_path, "int gl_SampleMaskIn[2]", "gl_SampleMaskIn[0]")
