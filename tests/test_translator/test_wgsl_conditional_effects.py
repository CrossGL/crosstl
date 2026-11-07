"""WGSL conditionals evaluate only the branch their condition picks.

WGSL output wrote ``c ? a : b`` as ``select(b, a, c)``, and ``select`` evaluates both value arguments (WGSL section
17.3). ``int x = c ? bump() : 7;`` therefore called ``bump()`` even when ``c`` was false. Conditionals whose branches
have side effects are now lowered to ``if``/``else`` where they are a whole statement value, and refused elsewhere.
WGSL expression statements other than calls and updates are written as phony assignments (WGSL section 9.2.1).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_conditional_effects import is_effectful_builtin

SHADER = """
shader Effects {
    compute {
        buffer int data[];
        int counter;
        int bump() { counter += 1; return counter; }
        int pure_add(int v) { int t = v; t += 1; return t; }
        int touch(inout int v) { v += 1; return v; }
        int indirect() { return bump(); }
        int pick(bool c) { return c ? bump() : 7; }
        [numthreads(1, 1, 1)]
        void main() {
            bool c = data[0] > 0;
            int x = c ? bump() : 7;
            int y = 0;
            y += c ? 0 : indirect();
            c ? bump() : pure_add(1);
            int z = c ? pure_add(1) : 2;
            int w = c ? (c ? bump() : 1) : 2;
            int v = c ? touch(y) : 3;
            data[1] = x + y + z + w + v + pick(c);
        }
    }
}
"""


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="wgsl")


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _naga(code, tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    result = subprocess.run(
        [naga, "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr + code


@pytest.fixture
def effects(tmp_path):
    return _translate(tmp_path, SHADER)


def test_effectful_branches_are_never_selected(effects):
    for call in ("bump()", "indirect()", "touch("):
        assert not re.search(r"select\([^;]*" + re.escape(call), effects), effects


def test_initializer_becomes_if_else(effects):
    flat = _flat(effects)
    assert "var x: i32; if (c) { x = bump(); } else { x = 7; }" in flat, effects


def test_compound_assignment_becomes_if_else(effects):
    flat = _flat(effects)
    assert "if (c) { y += 0; } else { y += indirect(); }" in flat, effects


def test_expression_statement_becomes_if_else(effects):
    flat = _flat(effects)
    assert "if (c) { bump(); } else { pure_add(1); }" in flat, effects


def test_return_becomes_if_else(effects):
    flat = _flat(effects)
    assert "if (c) { return bump(); } else { return 7; }" in flat, effects


def test_nested_conditional_is_lowered_again(effects):
    flat = _flat(effects)
    assert (
        "var w: i32; if (c) { if (c) { w = bump(); } else { w = 1; } } else { w = 2; }"
        in flat
    ), effects


def test_inout_parameter_is_a_side_effect(effects):
    flat = _flat(effects)
    assert "var v: i32; if (c) { v = touch(&y); } else { v = 3; }" in flat, effects


def test_pure_branches_keep_select(effects):
    assert "var z: i32 = select(2, pure_add(1), c);" in effects, effects


def test_output_validates_with_naga(effects, tmp_path):
    _naga(effects, tmp_path)


@pytest.mark.parametrize(
    "statement",
    [
        "int r = 1 + (c ? bump() : 0);",
        "data[c ? bump() : 0] = 1;",
        "if (c ? bump() > 0 : false) { data[2] = 1; }",
    ],
)
def test_conditional_inside_a_larger_expression_is_refused(tmp_path, statement):
    source = f"""
shader Refused {{
    compute {{
        buffer int data[];
        int counter;
        int bump() {{ counter += 1; return counter; }}
        [numthreads(1, 1, 1)]
        void main() {{
            bool c = data[0] > 0;
            {statement}
        }}
    }}
}}
"""
    with pytest.raises(ValueError, match="select evaluates both branches"):
        _translate(tmp_path, source)


def test_buffer_store_builtin_is_a_side_effect(tmp_path):
    source = """
shader Atomics {
    compute {
        buffer int data[];
        [numthreads(1, 1, 1)]
        void main() {
            bool c = data[0] > 0;
            int old = c ? atomicAdd(data[1], 1) : 0;
            data[2] = old;
        }
    }
}
"""
    code = _translate(tmp_path, source)
    assert "select(" not in code, code
    assert "if (c)" in code, code


@pytest.mark.parametrize(
    "name, effectful",
    [
        ("atomicAdd", True),
        ("InterlockedAdd", True),
        ("imageStore", True),
        ("groupMemoryBarrier", True),
        ("EmitVertex", True),
        ("TraceRay", True),
        ("texture", False),
        ("max", False),
        ("imageLoad", False),
    ],
)
def test_builtin_effect_classification(name, effectful):
    assert is_effectful_builtin(name) is effectful


def test_initializer_reading_its_own_name_is_refused(tmp_path):
    source = """
shader Shadow {
    compute {
        buffer int data[];
        int counter;
        int bump() { counter += 1; return counter; }
        [numthreads(1, 1, 1)]
        void main() {
            int x = data[0];
            bool c = x > 0;
            {
                int x = c ? bump() : x;
                data[1] = x;
            }
        }
    }
}
"""
    with pytest.raises(ValueError, match="reads a variable named x"):
        _translate(tmp_path, source)


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("data[0];", "_ = data[0];"),
        ("(a + 1);", "_ = (a + 1);"),
        ("a;", "_ = a;"),
    ],
)
def test_value_statements_are_phony_assignments(tmp_path, statement, expected):
    source = f"""
shader Values {{
    compute {{
        buffer int data[];
        [numthreads(1, 1, 1)]
        void main() {{
            int a = data[1];
            {statement}
            data[2] = a;
        }}
    }}
}}
"""
    code = _translate(tmp_path, source)
    assert expected in code, code
    _naga(code, tmp_path)
