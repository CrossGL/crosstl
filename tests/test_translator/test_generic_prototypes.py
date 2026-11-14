"""Generic function prototypes: forward declarations and declaration-only primaries.

``generic<T> T top(T x);`` followed by a definition of ``top`` declares one generic, not two overloads. Collection
treated the prototype as a second overload, so calls named a body-less clone such as ``top_float_overload0`` that
nothing defined, and the explicit specialization of ``top`` was ignored. A prototype that nothing defines is a
declaration-only primary: its explicit specializations are its only bodies, so any other instantiation is refused
instead of emitting a declaration without a definition.
"""

import pytest

import crosstl
from crosstl.translator.codegen.generic_prototypes import (
    GenericDefinitionMissingError,
)

BACKENDS = [
    "directx",
    "opengl",
    "metal",
    "vulkan",
    "wgsl",
    "cuda",
    "hip",
    "mojo",
    "slang",
]

PROTOTYPE = "generic<T> T top(T x);"
DEFINITION = "generic<T> T top(T x) { return x; }"

SOURCE = """
shader S {
    %s
    float top_f(float x) @explicit_specialization(top, float) { return x + 1.0; }
    compute {
        void main() {
            float a = top(1.0);
            int b = top(2);
        }
    }
}
"""


def _translate(tmp_path, backend, declarations):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE % declarations, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "declarations",
    [f"{PROTOTYPE}\n    {DEFINITION}", f"{DEFINITION}\n    {PROTOTYPE}"],
    ids=["prototype_first", "prototype_last"],
)
def test_forward_prototype_declares_the_defined_generic(
    tmp_path, backend, declarations
):
    code = _translate(tmp_path, backend, declarations)

    assert "overload0" not in code and "overload1" not in code
    if backend == "vulkan":
        # main, the float specialization and the int instantiation.
        assert code.count("= OpFunction ") == 3
        return
    assert "top_f(1.0" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_declaration_only_primary_refuses_unspecialized_arguments(tmp_path, backend):
    with pytest.raises(GenericDefinitionMissingError, match="'top'.*declared without"):
        _translate(tmp_path, backend, PROTOTYPE)
