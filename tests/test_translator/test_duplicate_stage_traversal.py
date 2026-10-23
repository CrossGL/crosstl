"""Rule passes visit every stage block of a kind, not only the first.

A CrossGL module may declare several blocks of one stage kind, as a Metal library with several vertex or fragment
functions imports to. :class:`~crosstl.translator.ast.StageMap` keeps each block under the same key, and
``stages[key]`` names only the first. The structural traversal of the rule passes
(:mod:`crosstl.translator.semantics.rewrite`) yielded that key once per block and then read ``stages[key]``, so it
visited the first block as many times as there were blocks and never the others. Every pass that walks the whole
program then skipped the later blocks: a fixed sampler state in the second fragment block reached HLSL verbatim and
SPIR-V as a descriptor declared inside a function, instead of being refused.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.rewrite import (
    StageSlot,
    child_slots,
    get_child,
    set_child,
)
from crosstl.translator.semantics.sampler_states import (
    FixedSamplerStateUnsupportedError,
)

SOURCE = """shader DuplicateStages {
    struct VOut {
        vec4 position @gl_Position;
        vec2 uv;
    }

    fragment {
        @ stage_entry
        vec4 plain(VOut in_, texture2D colour @texture(0)) {
            return vec4(in_.uv, 0.0, 1.0);
        }
    }

    fragment {
        @ stage_entry
        vec4 sampled(VOut in_, texture2D colour @texture(0)) {
            const sampler linear = sampler(filter::linear);
            return texture(colour, linear, in_.uv);
        }
    }
}
"""


def stage(name):
    return A.StageNode(A.ShaderStage.VERTEX, A.FunctionNode(name, "void", [], []))


def program_with_two_vertex_stages():
    first, second = stage("first"), stage("second")
    stages = A.StageMap()
    stages.append(A.ShaderStage.VERTEX, first)
    stages.append(A.ShaderStage.VERTEX, second)
    return (
        A.ShaderNode("Two", A.ExecutionModel.GRAPHICS_PIPELINE, stages=stages),
        first,
        second,
    )


def translate(tmp_path, backend):
    path = tmp_path / "duplicate_stages.cgl"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_traversal_yields_each_stage_block_of_a_kind():
    program, first, second = program_with_two_vertex_stages()
    slots = [(f, k) for f, k in child_slots(program) if f == "stages"]
    assert slots == [
        ("stages", StageSlot(A.ShaderStage.VERTEX, 0)),
        ("stages", StageSlot(A.ShaderStage.VERTEX, 1)),
    ]
    assert [get_child(program, f, k) for f, k in slots] == [first, second]


def test_replacing_a_later_block_keeps_the_first_as_the_keyed_stage():
    program, first, second = program_with_two_vertex_stages()
    replacement = stage("replacement")
    set_child(program, "stages", StageSlot(A.ShaderStage.VERTEX, 1), replacement)
    assert program.stages[A.ShaderStage.VERTEX] is first
    assert program.stages.all_for_stage(A.ShaderStage.VERTEX) == [first, replacement]

    set_child(program, "stages", StageSlot(A.ShaderStage.VERTEX, 0), second)
    assert program.stages[A.ShaderStage.VERTEX] is second
    assert program.stages.all_for_stage(A.ShaderStage.VERTEX) == [second, replacement]


@pytest.mark.parametrize("backend", ["opengl", "directx", "vulkan"])
def test_a_fixed_sampler_in_a_later_block_is_refused(tmp_path, backend):
    with pytest.raises(FixedSamplerStateUnsupportedError):
        translate(tmp_path, backend)


def test_metal_keeps_the_fixed_sampler_of_a_later_block(tmp_path):
    code = translate(tmp_path, "metal")
    assert "fragment float4 plain(" in code
    assert "fragment float4 sampled(" in code
    assert "sampler linear = sampler(filter::linear);" in code
