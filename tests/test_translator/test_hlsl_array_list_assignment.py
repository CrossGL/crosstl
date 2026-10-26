"""HLSL assignment of a brace list to a fixed array.

HLSL takes brace lists only in declarations, so ``a = {...}`` becomes element stores. A list whose elements read
the array, or may have effects a callee could observe, has to be evaluated before the first store: it initializes
a temporary that is then assigned whole. Indexing a nested array removes one extent, so ``g[1]`` of
``int g[2][2]`` is an ``int[2]`` that can initialize a row.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.ast import ArrayLiteralNode, AssignmentNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_array_list_assignment import (
    hlsl_list_needs_temporary,
    hlsl_root_name,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.rewrite import child_slots, get_child


def _shader(body, helpers=""):
    return f"""
shader Lists {{
    struct Pair {{
        int v;
        int w;
    }}
    {helpers}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* inp;
        buffer int* outp;
        void main() {{
            int x = inp[0];
{body}
        }}
    }}
}}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _hlsl(source):
    return HLSLCodeGen().generate(_parse(source))


def _compile(tmp_path, code):
    dxc = shutil.which("dxc")
    if not dxc:
        return
    (tmp_path / "out.hlsl").write_text(code)
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def _assignments(node):
    """Assignments of a brace list, found through the structural children only (no back-references)."""
    found = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, AssignmentNode) and isinstance(
            current.value, ArrayLiteralNode
        ):
            found.append(current)
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )
    return found


def test_a_list_that_reads_its_array_initializes_a_temporary_first(tmp_path):
    code = _hlsl(_shader("""
            int a[3] = {x, x + 1, x + 2};
            a = {a[2], a[0], a[1]};
            outp[0] = a[0] + a[1] * 10 + a[2] * 100;
"""))
    match = re.search(
        r"int (__crossgl_array_list_\d+)\[3\] = \{a\[2\], a\[0\], a\[1\]\};\s*a = \1;",
        code,
    )
    assert match, code
    assert "a[0] = a[2]" not in code
    _compile(tmp_path, code)


def test_struct_and_nested_arrays_are_assigned_whole(tmp_path):
    code = _hlsl(_shader("""
            Pair q[2] = {{x, 1}, {x + 5, 2}};
            q = {{q[1].v, q[0].w}, q[0]};
            int g[2][2] = {{x, 1}, {2, 3}};
            g = {g[1], g[0]};
            outp[0] = q[0].v + q[1].w + g[0][1];
"""))
    assert re.search(
        r"Pair (__crossgl_array_list_\d+)\[2\] = \{\{q\[1\]\.v, q\[0\]\.w\}, q\[0\]\};"
        r"\s*q = \1;",
        code,
    ), code
    assert re.search(
        r"int (__crossgl_array_list_\d+)\[2\]\[2\] = \{g\[1\], g\[0\]\};\s*g = \1;",
        code,
    ), code
    assert "q[0].x = " not in code and "q[1] = q[0]" not in code
    _compile(tmp_path, code)


def test_a_list_that_calls_a_function_is_evaluated_before_shared_stores(tmp_path):
    """A callee can see a groupshared array, so its stores wait until the whole list is evaluated."""
    code = _hlsl(
        _shader(
            """
            sa = {bump(x), 1};
            outp[0] = sa[0] + sa[1];
""",
            helpers="shared int sa[2];\n    int bump(int v) { return v + 1; }",
        )
    )
    assert re.search(
        r"int (__crossgl_array_list_\d+)\[2\] = \{bump\(x\), 1\};\s*sa = \1;", code
    ), code
    _compile(tmp_path, code)


def test_a_call_cannot_observe_a_local_array(tmp_path):
    """A local is visible to a callee only through an argument that names it, so plain stores are kept."""
    code = _hlsl(
        _shader(
            """
            int a[2] = {x, x};
            a = {bump(x), 1};
            outp[0] = a[0] + a[1];
""",
            helpers="int bump(int v) { return v + 1; }",
        )
    )
    assert "__crossgl_array_list_" not in code
    assert re.search(r"a\[0\] = bump\(x\);", code), code
    _compile(tmp_path, code)


def test_a_list_independent_of_its_array_keeps_the_element_stores(tmp_path):
    code = _hlsl(_shader("""
            int a[2] = {x, x};
            int b[2] = {x * 2, x * 3};
            a = {x * 4, b[0] + b[1]};
            outp[0] = a[0] + a[1];
"""))
    assert "__crossgl_array_list_" not in code
    assert re.search(r"a\[0\] = \(?x \* 4\)?;", code), code
    assert re.search(r"a\[1\] = \(?b\[0\] \+ b\[1\]\)?;", code), code
    _compile(tmp_path, code)


def test_a_row_of_a_nested_array_initializes_a_row(tmp_path):
    code = _hlsl(_shader("""
            int g[2][2] = {{x, 1}, {2, 3}};
            int h[2][2] = {g[1], g[0]};
            int r[2] = g[1];
            outp[0] = h[0][0] + r[1] + g[1][0];
"""))
    assert "int h[2][2] = {g[1], g[0]};" in code
    _compile(tmp_path, code)


@pytest.mark.parametrize(
    "statement, needs",
    [
        ("a = {a[1], 0};", True),
        ("a = {b[1], b[0]};", False),
        ("a = {bump(1), 0};", True),
        ("a = {b[0]++, 0};", True),
        ("a = {};", False),
    ],
)
def test_temporary_decision(statement, needs):
    source = _shader(
        f"""
            int a[2] = {{x, x}};
            int b[2] = {{x, x}};
            {statement}
            outp[0] = a[0];
""",
        helpers="int bump(int v) { return v + 1; }",
    )
    (assignment,) = _assignments(_parse(source))
    assert hlsl_root_name(assignment.target) == "a"
    assert hlsl_list_needs_temporary(None, assignment.target, assignment.value) is needs


class _Scope:
    """The parts of the HLSL generator's scope the decision reads."""

    def __init__(self, locals_, globals_=()):
        self.local_variable_types = {name: "int[2]" for name in locals_}
        self.global_variable_types = {name: "int[2]" for name in globals_}


@pytest.mark.parametrize(
    "statement, scope, needs",
    [
        ("a = {bump(1), 0};", _Scope(["a"]), False),
        ("a = {b[0]++, 0};", _Scope(["a", "b"]), False),
        ("a = {bump(a[0]), 0};", _Scope(["a"]), True),
        ("a = {bump(1), 0};", _Scope(["a"], ["a"]), True),
        ("a = {bump(1), 0};", _Scope([]), True),
    ],
)
def test_effects_matter_only_for_a_destination_a_callee_can_see(
    statement, scope, needs
):
    source = _shader(
        f"""
            int a[2] = {{x, x}};
            int b[2] = {{x, x}};
            {statement}
            outp[0] = a[0];
""",
        helpers="int bump(int v) { return v + 1; }",
    )
    (assignment,) = _assignments(_parse(source))
    assert (
        hlsl_list_needs_temporary(scope, assignment.target, assignment.value) is needs
    )


def test_rendered_destinations_have_no_root():
    assert hlsl_root_name("a") is None
    assert hlsl_root_name("s.v") is None
