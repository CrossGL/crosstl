"""DirectX hoists atomics out of control-flow conditions without changing when they run.

``Interlocked*`` returns the original value through an ``out`` argument, so the atomic has to run as a statement
before its condition is tested. DirectX used to refuse atomics in ``if``, ``else if``, ``while``, ``for``,
``do``/``while`` and ``switch`` conditions. An ``else if`` atomic must still run only after the earlier conditions
failed, and a loop condition must run once per iteration, including iterations started by ``continue``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

PROGRAM = """shader main {
    struct B { uint head; int tail; uint data[]; };
    B b @glsl_buffer_block(std430) @binding(0);
    RWStructuredBuffer<uint> tb @binding(1);
    shared uint g;
    shared int h[4];
    compute {
    layout(local_size_x = 64) in;
    void main() {
        BODY
    }
    }
}
"""

TEMP = r"(__crossgl_atomic_expr_\d+)"


def _hlsl(tmp_path, body):
    path = tmp_path / "shader.cgl"
    path.write_text(PROGRAM.replace("BODY", body), encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _entry(code):
    body = code[code.index("void CSMain(") :]
    return re.sub(r"\s+", " ", body)


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "shader.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


CASES = {
    "if-group-shared": (
        "if (atomicAdd(g, 1u) == 0u) { b.head = 1u; }",
        rf"uint {TEMP}; InterlockedAdd\(g, 1u, \1\); if \(\1 == 0u\) \{{ b\.Store\(0, 1u\); \}}",
    ),
    "if-group-shared-element": (
        "if (atomicAdd(h[1], 1) == 0) { b.head = 1u; }",
        rf"int {TEMP}; InterlockedAdd\(h\[1\], 1, \1\); if \(\1 == 0\) \{{",
    ),
    "if-typed-buffer": (
        "if (atomicAdd(tb[0], 1u) == 0u) { b.head = 1u; } else { b.head = 2u; }",
        rf"uint {TEMP}; InterlockedAdd\(tb\[0\], 1u, \1\); if \(\1 == 0u\) \{{ b\.Store\(0, 1u\); \}} else \{{ "
        r"b\.Store\(0, 2u\); \}",
    ),
    "while": (
        "while (atomicAdd(g, 1u) < 4u) { if (b.tail == 0) { continue; } b.tail = 2; }",
        rf"while \(true\) \{{ uint {TEMP}; InterlockedAdd\(g, 1u, \1\); if \(!\(\1 < 4u\)\) \{{ break; \}} "
        r"if \(asint\(b\.Load\(4\)\) == 0\) \{ continue; \}",
    ),
    "spin-lock": (
        "while (atomicCompSwap(g, 0u, 1u) != 0u) { } atomicExchange(g, 0u);",
        rf"while \(true\) \{{ uint {TEMP}; InterlockedCompareExchange\(g, 0u, 1u, \1\); "
        r"if \(!\(\1 != 0u\)\) \{ break; \} \}",
    ),
    "for": (
        "for (int i = 0; atomicAdd(g, 1u) < 4u; i++) { b.tail = i; }",
        rf"for \(int i = 0; ; i\+\+\) \{{ uint {TEMP}; InterlockedAdd\(g, 1u, \1\); "
        r"if \(!\(\1 < 4u\)\) \{ break; \} b\.Store\(4, asuint\(i\)\); \}",
    ),
    "do-while": (
        "do { if (b.tail == 0) { continue; } b.tail = 2; } while (atomicAdd(g, 1u) < 4u);",
        r"bool (__crossgl_do_while_first_\d+) = true; while \(true\) \{ if \(!\1\) \{ "
        rf"uint {TEMP}; InterlockedAdd\(g, 1u, \2\); if \(!\(\2 < 4u\)\) \{{ break; \}} \}} \1 = false; "
        r"if \(asint\(b\.Load\(4\)\) == 0\) \{ continue; \}",
    ),
    "switch": (
        "switch (atomicAdd(g, 1u)) { case 0u: b.head = 1u; break; default: break; }",
        rf"uint {TEMP}; InterlockedAdd\(g, 1u, \1\); switch \(\1\) \{{ case 0u:",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_directx_hoists_atomics_out_of_conditions(tmp_path, case):
    body, expected = CASES[case]
    code = _hlsl(tmp_path, body)
    assert "atomicAdd" not in code and "atomicCompSwap" not in code
    assert re.search(expected, _entry(code)), _entry(code)
    _dxc(tmp_path, code)


def test_directx_else_if_atomics_run_only_after_earlier_conditions_fail(tmp_path):
    code = _hlsl(
        tmp_path,
        """if (b.head == 0u) { b.data[0] = 1u; }
        else if (atomicAdd(g, 1u) == 0u) { b.data[0] = 2u; }
        else if (b.head == 3u) { b.data[0] = 3u; }
        else if (atomicAdd(h[2], 2) > 1) { b.data[0] = 4u; }
        else { b.data[0] = 5u; }""",
    )
    entry = _entry(code)
    first, second = re.findall(TEMP, entry)[::3]
    expected = (
        "if (b.Load(0) == 0u) { b.Store(8, 1u); } else { "
        f"uint {first}; InterlockedAdd(g, 1u, {first}); "
        f"if ({first} == 0u) {{ b.Store(8, 2u); }} else {{ if (b.Load(0) == 3u) {{ b.Store(8, 3u); }} else {{ "
        f"int {second}; InterlockedAdd(h[2], 2, {second}); "
        f"if ({second} > 1) {{ b.Store(8, 4u); }} else {{ b.Store(8, 5u); }} }} }} }}"
    )
    assert expected in entry, entry
    _dxc(tmp_path, code)


class _Condition:
    def __init__(self, name):
        self.name = name


class _ChainGenerator:
    """The generator surface the ``if`` lowering uses; conditions named ``atomic*`` need a prelude."""

    def hlsl_expression_contains_typed_buffer_atomic(self, condition):
        return condition.name.startswith("atomic")

    def render_hlsl_typed_buffer_atomic_value_expression(
        self, condition, expected_type, indent
    ):
        name = condition.name
        return f"{'    ' * indent}{expected_type} t_{name} = {name};\n", f"(t_{name})"

    def hlsl_strip_wrapping_parentheses(self, text):
        return text[1:-1] if text.startswith("(") else text

    def hlsl_condition_expression(self, condition):
        return condition.name

    def generate_scoped_statement_body(self, body, indent):
        return f"{'    ' * indent}{body};\n"


def test_flat_else_if_chains_nest_each_atomic_condition_after_the_earlier_ones():
    from types import SimpleNamespace

    from crosstl.translator.codegen.hlsl_atomic_conditions import (
        generate_hlsl_atomic_condition_if,
    )

    node = SimpleNamespace(
        else_if_conditions=[
            _Condition("atomicB"),
            _Condition("c"),
            _Condition("atomicD"),
        ],
        else_if_bodies=["sb", "sc", "sd"],
    )
    code = generate_hlsl_atomic_condition_if(
        _ChainGenerator(), node, _Condition("a"), "sa", "se", 1
    )
    assert code == (
        "    if (a) {\n        sa;\n    } else {\n"
        "        bool t_atomicB = atomicB;\n"
        "        if (t_atomicB) {\n            sb;\n        } else if (c) {\n            sc;\n        } else {\n"
        "            bool t_atomicD = atomicD;\n"
        "            if (t_atomicD) {\n                sd;\n            } else {\n                se;\n            }\n"
        "        }\n"
        "    }\n"
    )
    assert (
        generate_hlsl_atomic_condition_if(
            _ChainGenerator(), SimpleNamespace(), _Condition("a"), "sa", None, 0
        )
        is None
    )


def test_directx_hoists_atomics_in_nested_loop_conditions(tmp_path):
    code = _hlsl(
        tmp_path,
        """for (uint i = 0u; atomicAdd(tb[1], 1u) < 8u; i++) {
            do { b.data[i] = i; } while (atomicMax(h[0], int(i)) < 3);
        }""",
    )
    entry = _entry(code)
    assert re.search(
        rf"for \(uint i = 0u; ; i\+\+\) \{{ uint {TEMP}; InterlockedAdd\(tb\[1\], 1u, \1\);",
        entry,
    ), entry
    assert re.search(
        rf"int {TEMP}; InterlockedMax\(h\[0\], int\(i\), \1\); if \(!\(\1 < 3\)\)",
        entry,
    ), entry
    _dxc(tmp_path, code)


def test_directx_atomic_free_conditions_keep_their_native_form(tmp_path):
    code = _hlsl(
        tmp_path,
        """uint n = atomicAdd(g, 1u);
        while (n < 4u) { n++; }
        do { n--; } while (n > 2u);
        if (n == 0u) { b.head = n; } else if (n == 1u) { b.head = 2u; }
        switch (n) { case 0u: b.head = 1u; break; default: break; }""",
    )
    entry = _entry(code)
    for native in (
        "while ((n < 4u)) {",
        "} while ((n > 2u));",
        "if (n == 0u) { b.Store(0, n); } else { if (n == 1u) {",
        "switch (n) {",
    ):
        assert native in entry, entry
    assert "while (true)" not in entry
    _dxc(tmp_path, code)
