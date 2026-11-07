"""Metal templates materialized for a target never take a name the source already declares.

Translating Metal to a target without templates materializes each implicit instantiation, such as ``pick(x, 1.0f)``
of ``template <typename T> T pick(T, T)``, as a concrete function named after the template and its arguments. A
source that already declares ``pick_float`` keeps that function; the instantiation takes the next free name.
"""

import re

import pytest

import crosstl

SOURCE = """#include <metal_stdlib>
template <typename T> T pick(T a, T b) { return a > b ? a : b; }
float pick_float(float a, float b) { return a + b; }
kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  o[i] = pick(o[i], 1.0f) + pick_float(o[i], 2.0f);
}
"""

DEFINITION = re.compile(r"\b(pick_float\w*)\s*\([^;{}()]*\)\s*\{")


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_implicit_instantiation_takes_a_free_name(tmp_path, backend):
    source = tmp_path / "names.metal"
    source.write_text(SOURCE)
    code = crosstl.translate(str(source), backend=backend, format_output=False)

    assert sorted(DEFINITION.findall(code)) == ["pick_float", "pick_float_2"]
    definitions = {}
    for name, body in re.findall(
        r"\b(pick_float\w*)\s*\([^;{}()]*\)\s*\{([^}]*)\}", code
    ):
        definitions[name] = " ".join(body.split())
    assert "a > b" in definitions["pick_float_2"]
    assert "a + b" in definitions["pick_float"]
    assert re.search(r"\bpick_float_2\([^;]*?,\s*1\.0f?\)", code)
    assert re.search(r"\bpick_float\([^;]*?,\s*2\.0f?\)", code)
