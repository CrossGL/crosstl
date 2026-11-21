"""Mojo spellings of CrossGL pointer dereference and pointer member access, and the refusal of unhandled nodes.

Mojo has no prefix dereference operator (a prefix ``*`` unpacks arguments) and no ``->``. A pointer is dereferenced
with an empty subscript, so ``*p`` is written ``p[]`` and ``p->m`` is written ``p[].m``. An expression node the Mojo
generator has no rule for is refused with a diagnostic that names the node instead of being printed as its repr,
which is valid Mojo call syntax and so passes the generated-text leak check. A statement that holds no expression,
such as a bare ``return`` or an empty ``for`` clause, is not a node and is still written.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.ast import ExpressionNode
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

POINTERS = """
shader Pointers {
    struct S {
        float a;
        float b;
    };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* x;
        buffer S* s;
        buffer float* o;

        void main() {
            o[0] = *(x + 1);
            o[1] = *x;
            o[2] = (s + 1)->a;
            o[3] = s->b;
            *o = 1.0;
            s->a = 2.0;
            (s + 1)->b = *x + s->a;
        }
    }
}
"""

EXPECTED_LINES = (
    "o[0] = (x + 1)[]",
    "o[1] = x[]",
    "o[2] = (s + 1)[].a",
    "o[3] = s[].b",
    "o[] = 1.0",
    "s[].a = 2.0",
    "(s + 1)[].b = (x[] + s[].a)",
)


def _body_lines(code):
    return [line.strip() for line in code.splitlines()]


def _generate(source):
    return MojoCodeGen().generate(parse(source))


def test_pointer_dereference_and_member_access_use_the_dereference_subscript():
    lines = _body_lines(_generate(POINTERS))

    for expected in EXPECTED_LINES:
        assert expected in lines


def test_pointer_expressions_print_no_starred_expression_or_node_repr():
    code = _generate(POINTERS)

    assert "PointerAccessNode" not in code
    assert re.search(r"\w\)?->\w", code) is None
    assert re.search(r"\(\*", code) is None


def _generated_initializer(text, type_name="float"):
    """The generator after writing ``<type_name> v = <text>;`` in a compute stage, and the parsed ``<text>``."""
    source = (
        "shader E {\n"
        "    struct S { float a; float b; };\n"
        "    compute {\n"
        "        buffer float* x;\n"
        "        buffer S* s;\n"
        "        buffer vec4* q;\n"
        f"        void main() {{ {type_name} v = {text}; }}\n"
        "    }\n"
        "}\n"
    )
    ast = parse(source)
    codegen = MojoCodeGen()
    code = codegen.generate(ast)
    statement = next(iter(ast.stages.values())).entry_point.body.statements[0]
    return codegen, code, statement.initial_value


@pytest.mark.parametrize(
    ("text", "result_type"),
    [("*x", "float"), ("*(s + 1)", "S"), ("s->a", "float"), ("(s + 1)->b", "float")],
)
def test_dereferenced_pointer_has_the_pointee_type(text, result_type):
    codegen, _code, value = _generated_initializer(text, result_type)

    assert codegen.expression_result_type(value) == result_type


def test_pointer_member_access_on_a_vector_pointer_is_a_component_access():
    _codegen, code, _value = _generated_initializer("q->y")

    assert "var v: Float32 = q[][1]" in _body_lines(code)


def test_dereference_of_a_value_that_is_not_a_pointer_is_refused():
    with pytest.raises(MojoRefusal, match="operand of type float is not a pointer"):
        _generated_initializer("*x[0]")


class _UnhandledExpression(ExpressionNode):
    """An expression node with no Mojo rule, whose repr is valid Mojo call syntax."""

    def __repr__(self):
        return "UnhandledExpression(value=1)"


def test_expression_node_without_a_mojo_rule_is_refused_by_name():
    with pytest.raises(MojoRefusal, match="_UnhandledExpression") as refusal:
        MojoCodeGen().generate_expression(_UnhandledExpression())

    assert "UnhandledExpression(value=1)" not in str(refusal.value)


ABSENT_EXPRESSIONS = """
shader Absent {
    void touch(int n) {
        if (n > 2) {
            return;
        }
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* o;

        void main() {
            int i = 0;
            for (; i < 4; i++) {
                o[i] = 1.0;
            }
            for (int j = 0;; j++) {
                if (j > 3) {
                    break;
                }
            }
            for (;;) {
                break;
            }
            touch(i);
            return;
        }
    }
}
"""


def test_statements_with_an_absent_expression_print_no_none():
    """A bare ``return`` and the empty clauses of a ``for`` hold no expression node, so nothing is refused."""
    code = _generate(ABSENT_EXPRESSIONS)
    lines = _body_lines(code)

    assert "None" not in lines
    assert lines.count("return") == 2
    assert lines.count("while True:") == 2
    assert "while (i < 4):" in lines
    assert "while None:" not in lines
    assert "return None" not in lines


MOJO_KERNEL = """\
@fieldwise_init
struct S(Copyable):
    var a: Float32
    var b: Float32

def kernel(x: UnsafePointer[Float32, MutAnyOrigin], s: UnsafePointer[S, MutAnyOrigin], \
o: UnsafePointer[Float32, MutAnyOrigin]):
{body}
"""


def test_generated_pointer_lines_build_with_mojo(tmp_path):
    """The generated statements build in a Mojo kernel whose parameters are the shader's buffers.

    The rest of the generated program declares its buffers at module scope, which Mojo does not accept yet, so the
    statements are checked in a kernel that takes the same buffers as parameters.
    """
    mojo = find_mojo_compiler()
    lines = _body_lines(_generate(POINTERS))
    body = [line for line in lines if line in EXPECTED_LINES]
    assert len(body) == len(EXPECTED_LINES)

    source = tmp_path / "pointers.mojo"
    source.write_text(
        MOJO_KERNEL.format(body="\n".join(f"    {line}" for line in body))
    )
    result = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / "pointers.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
