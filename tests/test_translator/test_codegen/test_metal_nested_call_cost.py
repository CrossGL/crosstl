"""The Metal generator renders each argument of a nested user call a bounded number of times.

Every call with two or more arguments was first tried as a texture operation, and that attempt rendered the call's
arguments before the generator rendered them again as an ordinary call. Each level of nesting multiplied the work,
so a chain of nested calls cost time exponential in its depth. A call that names no texture operation and whose
first argument is no texture or image resource now skips the attempt (``texture_call_names``).
"""

import pytest

import crosstl
from crosstl.translator.codegen import metal_codegen
from crosstl.translator.codegen.texture_call_names import (
    names_sampled_texture_operation,
)

DEPTH = 6


def nested_source(depth, position=0):
    expression = "x"
    for level in range(depth):
        operands = [expression, f"y{level % 2}"]
        if position:
            operands.reverse()
        expression = f"mix2({operands[0]}, {operands[1]})"
    return f"""
shader Nested {{
    float mix2(float a, float b) {{ return a * 0.5 + b; }}
    compute {{
        void main(RWStructuredBuffer<float> data @buffer(0), uint i @gl_GlobalInvocationID) {{
            float x = data[i];
            float y0 = 1.0;
            float y1 = 2.0;
            data[i] = {expression};
        }}
    }}
}}
"""


def count_identifier_renders(monkeypatch, tmp_path, depth, position=0):
    calls = {"x": 0}
    original = metal_codegen.MetalCodeGen.generate_expression

    def counting(self, expr, *args, **kwargs):
        if getattr(expr, "name", None) == "x":
            calls["x"] += 1
        return original(self, expr, *args, **kwargs)

    monkeypatch.setattr(metal_codegen.MetalCodeGen, "generate_expression", counting)
    path = tmp_path / f"nested_{depth}_{position}.cgl"
    path.write_text(nested_source(depth, position))
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    return metal, calls["x"]


@pytest.mark.parametrize("position", [0, 1])
def test_nested_calls_render_the_innermost_argument_a_bounded_number_of_times(
    monkeypatch, tmp_path, position
):
    metal, renders = count_identifier_renders(monkeypatch, tmp_path, DEPTH, position)
    assert metal.count("mix2(") == DEPTH + 1
    # Linear in the depth; the texture attempt made it at least 2**DEPTH, up to 8**DEPTH.
    assert renders <= 8 * DEPTH


@pytest.mark.parametrize("position", [0, 1])
def test_render_count_grows_linearly_with_depth(monkeypatch, tmp_path, position):
    _, shallow = count_identifier_renders(monkeypatch, tmp_path, DEPTH // 2, position)
    _, deep = count_identifier_renders(monkeypatch, tmp_path, DEPTH, position)
    assert deep <= 3 * shallow


def test_texture_operation_names_are_recognized():
    for name in ("texture", "textureLod", "textureGrad", "texelFetch", "textureGather"):
        assert names_sampled_texture_operation(name), name
    for name in ("mix2", "dot", "max"):
        assert not names_sampled_texture_operation(name), name
