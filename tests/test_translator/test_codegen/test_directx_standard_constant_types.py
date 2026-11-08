"""Types of the predefined constants the DirectX generator renders as literals.

``M_SQRT1_2_F`` is a ``float`` constant, so ``x * M_SQRT1_2_F`` for a ``float16`` ``x`` is a ``float`` product
(usual arithmetic conversions). The generator used to give the product ``x``'s type and narrow it through the
binary16 widening helper, which DXC rejects for a ``float`` operand.
"""

import crosstl
from crosstl.translator.codegen.standard_constant_types import standard_constant_type

SOURCE = """
shader S {
    compute {
        RWStructuredBuffer<float16> h;
        float wide(float a) { return a; }
        void main() {
            float16 x = buffer_load(h, 0);
            float y = wide(x * M_SQRT1_2_F);
            float z = wide(x * FLT_MAX);
            buffer_store(h, 1, float16(y + z));
        }
    }
}
"""

SHADOWED = """
shader S {
    compute {
        RWStructuredBuffer<float16> h;
        float16 narrow(float16 a) { return a; }
        void main() {
            float16 M_SQRT1_2_F = buffer_load(h, 0);
            float16 x = buffer_load(h, 1);
            buffer_store(h, 2, narrow(x * M_SQRT1_2_F));
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_standard_constant_types():
    assert standard_constant_type("M_SQRT1_2_F") == "float"
    assert standard_constant_type("M_SQRT1_2") == "double"
    assert standard_constant_type("FLT_MAX") == "float"
    assert standard_constant_type("value") is None
    assert standard_constant_type(None) is None


def test_float16_times_float_constant_is_a_float_product(tmp_path):
    hlsl = _translate(tmp_path, SOURCE)
    widened = "__crossgl_binary16_to_float(uint(asuint16(x)))"
    assert f"wide(({widened} * 0.707106781186547524400844362104849039f))" in hlsl
    assert "asuint16((x * " not in hlsl


def test_a_local_shadowing_a_constant_keeps_its_declared_type(tmp_path):
    hlsl = _translate(tmp_path, SHADOWED)
    assert "narrow((x * M_SQRT1_2_F))" in hlsl
