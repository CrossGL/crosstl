"""GLSL import keeps an assignment that is an operand as one parenthesized group.

The importer wrote ``(b = b + 1.0) * 2.0`` as ``(b = (b + 1.0) * 2.0)``, so the surrounding operator bound to the
right-hand side of the assignment and every target computed something else.
"""

import re

import pytest

import crosstl

COMPUTE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data { float values[]; } data;
void main() {
    float b = data.values[0];
    float c = (b = b + 1.0) * 2.0;
    float d = (b += 1.0) + 1.0;
    float e = -(b -= 0.5);
    float f = c > 3.0 ? (d = 2.0) : (e = 1.0);
    data.values[0] = (b *= 2.0) - c + d + e + f;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.comp"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_binary_operands_keep_the_assignment_grouped(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "cgl"))
    assert "float c = ((b = (b + 1.0)) * 2.0);" in code
    assert "float d = ((b += 1.0) + 1.0);" in code
    assert "((((b *= 2.0) - c) + d) + e) + f)" in code


def test_unary_and_conditional_operands_keep_the_assignment_grouped(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "cgl"))
    assert "float e = (-(b -= 0.5));" in code
    assert "float f = ((c > 3.0) ? (d = 2.0) : (e = 1.0));" in code


def test_statement_assignments_stay_unparenthesized(tmp_path):
    source = COMPUTE.replace(
        "    data.values[0] = (b *= 2.0) - c + d + e + f;\n",
        "    for (int i = 0; i < 2; i += 1) { b = b + 1.0; }\n    data.values[0] = b;\n",
    )
    code = _flat(_translate(tmp_path, source, "cgl"))
    assert "b = (b + 1.0);" in code
    assert "(b = (b + 1.0));" not in code
    assert re.search(r"for \(int i = 0; \(?i < 2\)?; i \+= 1\)", code)


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
def test_targets_keep_the_assignment_grouped(tmp_path, backend):
    code = _flat(_translate(tmp_path, COMPUTE, backend))
    grouped = re.search(r"float c = \(*b = \(?b \+ 1\.0\)?\) \* 2\.0\)?;", code)
    assert grouped, code
