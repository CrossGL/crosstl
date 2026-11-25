import re

import crosstl

SHADER = """
shader T {
    const int count = 4;
    const int N = 4;
    float f(int i) {
        float a[count];
        float b[N * 2];
        a[i] = 1.0;
        b[i] = 2.0;
        return a[i] + b[i];
    }
    compute {
        void main() {
            float r = f(1);
        }
    }
}
"""


def _mojo(tmp_path):
    source = tmp_path / "arrays.cgl"
    source.write_text(SHADER)
    return crosstl.translate(str(source), backend="mojo", format_output=False)


def test_local_array_sized_by_a_lowercase_constant_is_a_static_tuple(tmp_path):
    code = _mojo(tmp_path)
    # The CrossGL spelling "float[count]" is not a Mojo type.
    assert "float[count]" not in code
    assert re.search(r"var a = StaticTuple\[Float32, (count|4)\]\(", code), code


def test_local_array_sized_by_a_constant_expression_keeps_its_size(tmp_path):
    code = _mojo(tmp_path)
    # An empty List has no element 'b[i]' can be stored to.
    assert "var b = List[Float32]()" not in code
    assert re.search(r"var b = StaticTuple\[Float32, [^\]]+\]\(", code), code
