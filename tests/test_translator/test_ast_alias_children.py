"""A child kept under two field names is one child.

``ArrayAccessNode`` stores its operands as ``array``/``array_expr`` and ``index``/``index_expr``, and
``AssignmentNode`` as ``target``/``left`` and ``value``/``right``. ``child_nodes`` used to yield such a child once
per name, so the GLSL control-flow interval analysis applied the ``k++`` in ``vals[k++] = 1.0`` twice, took ``k`` to
be 2 afterwards and refused the in-range ``vals[k]`` (issue #3233).
"""

import shutil
import subprocess

import crosstl
from crosstl.translator.ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    IdentifierNode,
    UnaryOpNode,
)

INCREMENT_INDEX_COMPUTE = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            int k = 0;
            float vals[2];
            vals[k++] = 1.0;
            vals[k] = 2.0;
            outp[0] = vals[0] + vals[1];
        }
    }
}
"""

INCREMENT_INDEX_FRAGMENT = """
shader P {
    fragment {
        vec4 main() @ gl_FragColor {
            int k = 0;
            float vals[2];
            vals[k++] = 1.0;
            vals[k] = 2.0;
            return vec4(vals[0], vals[1], 0.0, 1.0);
        }
    }
}
"""

INCREMENT_INDEX_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Out { float outp[]; };
void main() {
    int k = 0;
    float vals[2];
    vals[k++] = 1.0;
    vals[k] = 2.0;
    outp[0] = vals[0] + vals[1];
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    shader = tmp_path / f"out.{stage}"
    shader.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_aliased_fields_yield_one_child():
    index = UnaryOpNode("++", IdentifierNode("k"))
    index.is_postfix = True
    access = ArrayAccessNode(IdentifierNode("vals"), index)
    children = list(access.child_nodes())
    assert children.count(index) == 1
    assert [child is index for child in children].count(True) == 1
    assert len(children) == 2

    assignment = AssignmentNode(access, IdentifierNode("one"))
    assignment_children = list(assignment.child_nodes())
    assert len(assignment_children) == 2
    assert assignment_children[0] is access


def test_distinct_operands_in_different_fields_are_still_children():
    left = IdentifierNode("a")
    right = IdentifierNode("a")
    children = list(BinaryOpNode(left, "+", right).child_nodes())
    assert any(child is left for child in children)
    assert any(child is right for child in children)


def test_opengl_keeps_the_increment_in_an_index_once(tmp_path):
    code = _translate(tmp_path, INCREMENT_INDEX_COMPUTE, "inc.cgl", "opengl")
    compact = " ".join(code.split())
    assert "vals[(k++)] = 1.0;" in compact or "vals[k++] = 1.0;" in compact
    assert "vals[k] = 2.0;" in compact
    _glslang(tmp_path, code, "comp")


def test_webgl_keeps_the_increment_in_an_index_once(tmp_path):
    code = _translate(tmp_path, INCREMENT_INDEX_FRAGMENT, "inc.cgl", "webgl")
    assert "vals[k] = 2.0;" in " ".join(code.split())
    _glslang(tmp_path, code, "frag")


def test_glsl_round_trip_keeps_the_increment_in_an_index_once(tmp_path):
    crossgl = _translate(tmp_path, INCREMENT_INDEX_GLSL, "inc.comp", "cgl")
    code = _translate(tmp_path, crossgl, "round.cgl", "opengl")
    assert "vals[k] = 2.0;" in " ".join(code.split())
    _glslang(tmp_path, code, "comp")
