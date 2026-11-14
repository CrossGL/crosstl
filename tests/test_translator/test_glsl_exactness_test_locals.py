"""Only floating sums make an equality-tested OpenGL local ``precise``.

The OpenGL generator declares a binary32 local ``precise`` when its single
value is a sum or difference that is later compared with ``==`` or ``!=``,
because Mesa folds a non-precise ``(a + b) == a`` into ``b == 0``.  An integer
``+`` or ``-`` cannot be contracted or reassociated by a floating
optimization, so a local computed from one, such as ``float(n - 8)``, keeps
its plain declaration, and the WebGL generator, which has no ``precise`` in
GLSL ES 3.00, translates it instead of refusing it.
"""

import pytest

from crosstl import translate
from crosstl.translator.codegen.webgl_codegen import WebGLPreciseQualifierError


def _translate(tmp_path, body, backend, uniforms="uniform int n;"):
    source = tmp_path / "probe.cgl"
    source.write_text(
        "shader Probe {\n"
        "    fragment {\n"
        f"        {uniforms}\n"
        "        void main() {\n"
        f"{body}"
        "        }\n"
        "    }\n"
        "}\n"
    )
    return translate(str(source), backend=backend)


INTEGER_DIFFERENCE = """            float f = float(n - 8);
            gl_FragColor = vec4(f == 3.0 ? 1.0 : 0.0);
"""

INTEGER_LOCAL_SUM = """            int k = n * 2;
            uint m = uint(n) + 1u;
            float g = float(k + 1);
            float h = float(m - 2u);
            gl_FragColor = vec4(g == 3.0 ? 1.0 : 0.0, h != 0.0 ? 1.0 : 0.0, 0.0, 1.0);
"""

FLOAT_SUM = """            float f = 1.0 + x;
            float g = float(n - 8) + x;
            gl_FragColor = vec4(f == 1.0 ? 1.0 : 0.0, g == 3.0 ? 1.0 : 0.0, 0.0, 1.0);
"""


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_integer_difference_local_is_not_precise(tmp_path, backend):
    generated = _translate(tmp_path, INTEGER_DIFFERENCE, backend)
    assert "float f = float((n - 8));" in generated
    assert "precise" not in generated


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_integer_local_operands_are_not_precise(tmp_path, backend):
    generated = _translate(tmp_path, INTEGER_LOCAL_SUM, backend)
    assert "float g = float((k + 1));" in generated
    assert "precise" not in generated


def test_floating_sum_local_stays_precise(tmp_path):
    uniforms = "uniform int n;\n        uniform float x;"
    generated = _translate(tmp_path, FLOAT_SUM, "opengl", uniforms)
    assert "precise float f = (1.0 + x);" in generated
    # An integer difference inside a floating sum still leaves a float sum.
    assert "precise float g = (float((n - 8)) + x);" in generated
    with pytest.raises(WebGLPreciseQualifierError):
        _translate(tmp_path, FLOAT_SUM, "webgl", uniforms)
