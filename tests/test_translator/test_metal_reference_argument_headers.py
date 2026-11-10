"""Reference arguments copied in and out of Metal ``thread`` temporaries in statement headers.

A call in an ``if``, ``while``, ``for`` or ``switch`` header whose ``inout`` argument is threadgroup or device
storage is given a statement of its own, so the copy-in runs before each evaluation and the copy-out right after
it (OpenGL Shading Language 4.60, sections 6.1.1, 6.2 and 6.3).
"""

import shutil
import subprocess

import pytest

import crosstl

HEADERS = """shader S {
    compute {
        layout(std430, binding = 0) buffer Seeds { int seed[]; } seeds;
        layout(std430, binding = 1) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        bool below(inout int a, int limit) { a = a + 1; return a < limit; }
        bool flip(inout int a) { a = 0 - a; return a > 0; }
        int take(inout int a) { a = a + 10; return a; }
        void main() {
            int base = seeds.seed[0];
            tile[0] = base;
            tile[1] = 0;
            tile[2] = 0;
            tile[3] = 100;
            int n = 0;
            while (below(tile[3], 103)) { n = n + 1; }
            int m = 0;
            if (flip(tile[0])) { m = 1; } else if (below(tile[1], 1)) { m = 2; } else if (flip(tile[2])) { m = 3; } else { m = 4; }
            int k = 0;
            for (int i = take(tile[2]); below(tile[1], 5); i = i + 1) {
                k = k + i;
                if (k > 100) { break; }
                continue;
            }
            switch (take(tile[3])) { case 113: n = n + 10; break; default: n = 0; }
            outBuf.values[0] = n;
            outBuf.values[1] = tile[3];
            outBuf.values[2] = m;
            outBuf.values[3] = tile[0];
            outBuf.values[4] = tile[1];
            outBuf.values[5] = k;
            outBuf.values[6] = tile[2];
        }
    }
}
"""

DEVICE_WHILE = """RWStructuredBuffer<int> outBuf : register(u0);
bool below(inout int a, int limit) { a = a + 1; return a < limit; }
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outBuf[1] = 0;
    int n = 0;
    while (below(outBuf[1], 3)) { n = n + 1; }
    outBuf[0] = n;
}
"""


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _lines(code):
    return [line.strip() for line in code.splitlines() if line.strip()]


def _follows(lines, *expected):
    """Whether ``expected`` appear as consecutive lines."""
    count = len(expected)
    return any(
        tuple(lines[index : index + count]) == expected
        for index in range(len(lines) - count + 1)
    )


def test_while_condition_becomes_a_guard_evaluated_each_iteration(tmp_path):
    lines = _lines(_translate(tmp_path, HEADERS))
    assert _follows(
        lines,
        "for (; ; ) {",
        "int __crossgl_ref2 = tile[3];",
        "bool __crossgl_cond1 = below(__crossgl_ref2, 103);",
        "tile[3] = __crossgl_ref2;",
        "if (!__crossgl_cond1) {",
        "break;",
        "}",
        "n = n + 1;",
    )
    assert not any(line.startswith("while (below(") for line in lines)


def test_else_if_condition_is_evaluated_only_after_earlier_conditions_fail(tmp_path):
    lines = _lines(_translate(tmp_path, HEADERS))
    assert _follows(
        lines,
        "int __crossgl_ref4 = tile[0];",
        "bool __crossgl_cond3 = flip(__crossgl_ref4);",
        "tile[0] = __crossgl_ref4;",
        "if (__crossgl_cond3) {",
        "m = 1;",
        "} else {",
        "int __crossgl_ref6 = tile[1];",
        "bool __crossgl_cond5 = below(__crossgl_ref6, 1);",
        "tile[1] = __crossgl_ref6;",
        "if (__crossgl_cond5) {",
    )
    assert _follows(
        lines,
        "} else {",
        "int __crossgl_ref8 = tile[2];",
        "bool __crossgl_cond7 = flip(__crossgl_ref8);",
    )


def test_for_initializer_and_condition_get_statements_of_their_own(tmp_path):
    lines = _lines(_translate(tmp_path, HEADERS))
    assert _follows(
        lines,
        "{",
        "int __crossgl_ref10 = tile[2];",
        "int i = take(__crossgl_ref10);",
        "tile[2] = __crossgl_ref10;",
        # The update stays in the header, so a continue still runs it before the next test.
        "for (; ; i = i + 1) {",
        "int __crossgl_ref11 = tile[1];",
        "bool __crossgl_cond9 = below(__crossgl_ref11, 5);",
        "tile[1] = __crossgl_ref11;",
        "if (!__crossgl_cond9) {",
        "break;",
        "}",
        "k = k + i;",
    )


def test_switch_selector_is_evaluated_into_a_scoped_temporary(tmp_path):
    lines = _lines(_translate(tmp_path, HEADERS))
    assert _follows(
        lines,
        "{",
        "int __crossgl_ref13 = tile[3];",
        "int __crossgl_selector12 = take(__crossgl_ref13);",
        "tile[3] = __crossgl_ref13;",
        "switch (__crossgl_selector12) {",
    )


def test_device_element_in_a_while_condition(tmp_path):
    lines = _lines(_translate(tmp_path, DEVICE_WHILE, "shader.hlsl"))
    assert _follows(
        lines,
        "int __crossgl_ref2 = outBuf[1];",
        "bool __crossgl_cond1 = below(__crossgl_ref2, 3);",
        "outBuf[1] = __crossgl_ref2;",
    )


@pytest.mark.parametrize(
    "replacement, expected",
    [
        (
            "do { n = n + 1; } while (below(tile[3], 103));",
            "} while ((__crossgl_ref1 = tile[3], __crossgl_call2 = below(__crossgl_ref1, 103), "
            "tile[3] = __crossgl_ref1, __crossgl_call2));",
        ),
        (
            "for (int j = 0; j < 3; j = j + take(tile[3])) { n = n + 1; }",
            "for (int j = 0; j < 3; j = j + (__crossgl_ref1 = tile[3], __crossgl_call2 = take(__crossgl_ref1), "
            "tile[3] = __crossgl_ref1, __crossgl_call2)) {",
        ),
    ],
    ids=["do-while", "for-update"],
)
def test_headers_without_a_statement_before_each_evaluation_copy_in_place(
    tmp_path, replacement, expected
):
    # A continue reaches these headers from the body, so their copies are made in the header itself.
    source = HEADERS.replace("while (below(tile[3], 103)) { n = n + 1; }", replacement)
    assert expected in _lines(_translate(tmp_path, source))


@pytest.mark.parametrize(
    "source, name",
    [(HEADERS, "shader.cgl"), (DEVICE_WHILE, "shader.hlsl")],
    ids=["threadgroup", "device"],
)
def test_header_copies_compile(tmp_path, source, name):
    code = _translate(tmp_path, source, name)
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
