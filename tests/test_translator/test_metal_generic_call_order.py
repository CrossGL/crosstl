"""Metal writes a generic's specializations before the functions that call them with explicit type arguments.

The parser keeps an explicit call's type arguments in the callee name (``twice<float>``). Metal orders functions so
that callees come first, but it looked the callee up by that whole name, found no function, and so wrote a caller
defined ahead of the generic before ``twice_float``; the Metal compiler then rejected the call as an undeclared
identifier. An implicit call ``twice(x)`` already named the generic and was ordered correctly.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

KERNEL = """
    compute {{
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer float* out_;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            out_[i] = {call};
        }}
    }}
"""

CASES = {
    "explicit": (
        [
            "float caller(float x) { return twice<float>(x); }",
            "generic<T> T twice(T x) { return x + x; }",
        ],
        "caller(inp[i])",
        ["twice_float", "caller"],
    ),
    "implicit": (
        [
            "float caller(float x) { return twice(x); }",
            "generic<T> T twice(T x) { return x + x; }",
        ],
        "caller(inp[i])",
        ["twice_float", "caller"],
    ),
    "generic_to_later_generic": (
        [
            "float caller(float x) { return four<float>(x) + float(twice<int>(int(x))); }",
            "generic<T> T four(T x) { return twice<T>(twice<T>(x)); }",
            "generic<T> T twice(T x) { return x + x; }",
        ],
        "caller(inp[i])",
        ["twice_int", "twice_float", "four_float", "caller"],
    ),
    "stage_entry_calls_later_generic": (
        ["generic<T> T twice(T x) { return x + x; }"],
        "twice<float>(inp[i])",
        ["twice_float"],
    ),
}


def _translate(tmp_path, functions, call):
    body = textwrap.indent("\n".join(functions), "    ")
    source = f"shader main {{\n{body}\n{KERNEL.format(call=call)}}}\n"
    path = tmp_path / "order.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _definition_line(code, name):
    for number, line in enumerate(code.splitlines()):
        signature, brace, _ = line.partition("{")
        if brace and not line.startswith(" ") and f" {name}(" in signature:
            return number
    raise AssertionError(f"{name} is not defined in:\n{code}")


@pytest.mark.parametrize("case", sorted(CASES))
def test_specializations_precede_their_callers(tmp_path, case):
    functions, call, order = CASES[case]
    code = _translate(tmp_path, functions, call)
    assert "<float>" not in code and "<int>" not in code and "<T>" not in code
    lines = [_definition_line(code, name) for name in order]
    assert lines == sorted(lines), code


@pytest.mark.parametrize("case", sorted(CASES))
def test_ordered_output_compiles(tmp_path, case):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    functions, call, _ = CASES[case]
    (tmp_path / "order.metal").write_text(_translate(tmp_path, functions, call))
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "order.metal", "-o", "order.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr
