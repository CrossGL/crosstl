"""A shader-scope `let` with no written type takes it from the initializer.

Regression for #2444: Metal printed `constant None scale = 2.0;` because the
global-variable path stringified `var_type` (None) instead of inferring the
type the initializer already carries.
"""

import pytest

from crosstl import translate
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

SOURCE = """
shader main {
    let scale = 2.0;
    compute {
        buffer float data[];
        void main() {
            data[0] = scale;
        }
    }
}
"""


def test_untyped_shader_scope_let_uses_the_initializer_type(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE, encoding="utf-8")

    output = translate(str(path), backend="metal")

    assert "constant float scale = 2.0;" in output
    assert "constant None" not in output


def test_untyped_shader_scope_let_with_no_inferable_type_is_refused(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE.replace("2.0", "missing_symbol"), encoding="utf-8")

    with pytest.raises(
        MetalLoweringRefused,
        match="global declaration: the type of 'scale' is not written and "
        "cannot be inferred from its initializer",
    ):
        translate(str(path), backend="metal")
