"""Slang helper parameter directions survive import to CrossGL."""

import pytest

import crosstl
import crosstl.translator as cgl_translator


@pytest.mark.parametrize("direction", ["in", "out", "inout"])
def test_slang_helper_preserves_parameter_direction(tmp_path, direction):
    source = tmp_path / "directions.slang"
    bodies = {
        "in": "return k;",
        "out": "k = 1; return k;",
        "inout": "k = k + 1; return k;",
    }
    body = bodies[direction]
    source.write_text(f"""
RWStructuredBuffer<int> data;
int bump({direction} int k) {{
    {body}
}}
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    int k = data[id.x];
    int r = bump(k);
    data[id.x] = r + k;
}}
""")
    output = crosstl.translate(str(source), backend="cgl", format_output=False)
    assert f"int bump({direction} int k)" in output, output
    cgl_translator.parse(output)


def test_unqualified_helper_parameter_stays_unqualified(tmp_path):
    source = tmp_path / "plain.slang"
    source.write_text(
        "int twice(int k) { return k * 2; }\n"
        '[shader("compute")] void main() { int x = twice(1); }\n'
    )
    output = crosstl.translate(str(source), backend="cgl", format_output=False)
    assert "int twice(int k)" in output, output
    cgl_translator.parse(output)
