"""A wrapping compound assignment whose value can write its target reads the target after the value.

Targets without wrapping signed arithmetic rewrite ``t += v`` as ``t = asint(asuint(t) + asuint(v))``, which reads
``t`` as the left operand. glslang, Mesa and DXC read the target of a compound assignment after its value, so when
``v`` can write ``t`` the value is evaluated into a local first.
"""

import re

import pytest

import crosstl

SOURCE = """#version 450
layout(local_size_x = 4) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
shared int tile[4];
int touch() { tile[0] = 5; return 1; }
int bump(inout int t) { t = t * 3 + 1; return t - 2; }
void main() {
    int acc = o[1];
    int y = o[2];
    ivec2 v = ivec2(o[3], o[4]);
    acc += y * 2;
    acc += abs(y);
    acc -= bump(y);
    tile[0] = o[0];
    tile[0] += touch();
    v.y += bump(v.y);
    y += bump(y);
    o[0] = acc + tile[0] + y + v.y;
}
"""


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    path = tmp_path_factory.mktemp("compound") / "shader.comp"
    path.write_text(SOURCE)
    return {
        backend: crosstl.translate(str(path), backend=backend, format_output=False)
        for backend in ("directx", "metal", "vulkan", "opengl")
    }


def _lines(code):
    return [line.strip() for line in code.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "value, target, read",
    [
        ("touch()", "tile[0]", "uint(tile[0])"),
        ("bump(v.y)", "v.y", "uint(v.y)"),
        ("bump(y)", "y", "uint(y)"),
    ],
)
def test_hlsl_value_that_writes_the_target_is_evaluated_first(
    outputs, value, target, read
):
    lines = _lines(outputs["directx"])
    declaration = next(
        index
        for index, line in enumerate(lines)
        if re.fullmatch(rf"int cgl_compound_value_\d+ = {re.escape(value)};", line)
    )
    name = lines[declaration].split()[1]
    assert lines[declaration + 1].startswith(f"{target} = int(({read} ")
    assert lines[declaration + 1].endswith(f"uint({name})));")


def test_values_that_cannot_write_the_target_are_not_hoisted(outputs):
    lines = _lines(outputs["directx"])
    assert "acc = int((uint(acc) + uint(int((uint(y) * 2u)))));" in lines
    assert "acc = int((uint(acc) + asuint(abs(y))));" in lines
    # ``bump`` writes ``y``, not the local ``acc``.
    assert "acc = int((uint(acc) - uint(bump(y))));" in lines
    assert sum("cgl_compound_value_" in line for line in lines) == 6


def test_metal_copies_the_component_back_before_the_target_is_read(outputs):
    lines = _lines(outputs["metal"])
    copy_back = lines.index("v.y = __crossgl_ref1;")
    assert re.fullmatch(
        r"int cgl_compound_value_\d+ = bump\(__crossgl_ref1\);", lines[copy_back - 1]
    )
    assert lines[copy_back + 1].startswith("v.y = as_type<int>(as_type<uint>(v.y) + ")


def test_spirv_evaluates_the_value_into_a_local(outputs):
    # Runtime order is covered by the conformance case g_reference_copy_order.
    names = re.findall(r'OpName %\d+ "(cgl_compound_value_\d+)"', outputs["vulkan"])
    assert len(names) == 3


def test_glsl_keeps_the_native_compound_assignment(outputs):
    lines = _lines(outputs["opengl"])
    assert "v.y += bump(v.y);" in lines
    assert not any("cgl_compound_value_" in line for line in lines)
