"""Slang refuses bfloat16 instead of printing it as an undefined type name."""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_numeric_types import (
    slang_unrepresentable_type_reason,
)
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

SOURCE = """
shader BFloat {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<bfloat16> output @ binding(0)) {
            bfloat16 x = output[0];
            bfloat16vec2 v = bfloat16vec2(x, x);
            output[1] = v.x + v.y;
        }
    }
}
"""


def test_bfloat16_programs_are_refused():
    with pytest.raises(SlangLoweringRefused) as excinfo:
        SlangCodeGen().generate(crosstl.translator.parse(SOURCE))

    message = str(excinfo.value)
    assert "Slang has no bfloat16 type for 'bfloat16'" in message
    assert "Slang has no bfloat16 type for 'bfloat16vec2'" in message


def test_half_programs_still_translate():
    generated = SlangCodeGen().generate(
        crosstl.translator.parse(
            SOURCE.replace("bfloat16vec2", "f16vec2").replace("bfloat16", "half")
        )
    )

    assert "RWStructuredBuffer<half> output" in generated
    assert "bfloat" not in generated


@pytest.mark.parametrize(
    ("spelling", "refused"),
    [
        ("bfloat16", True),
        ("bfloat16_t", True),
        ("bfloat", True),
        ("bf16", True),
        ("bfloat16vec4", True),
        ("bfloat3", True),
        ("bfloat2x2", True),
        ("half", False),
        ("float16", False),
        ("float", False),
        ("bfloat16_buffer", False),
    ],
)
def test_unrepresentable_type_spellings(spelling, refused):
    assert (slang_unrepresentable_type_reason(spelling) is not None) is refused
