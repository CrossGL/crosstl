"""CrossGL ``@explicit_specialization``: a concrete function that replaces a generic's body for some arguments.

``float twice_f(float x) @explicit_specialization(twice, float)`` plays the part of C++'s
``template <> float twice<float>(float)`` ([temp.expl.spec]/1). Every instantiation of ``twice`` for ``float``,
including one reached only through another generic (``apply<float>``), calls ``twice_f``. Targets with native
generics that cannot select a body per argument refuse the program.
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader S {
    generic<T> T twice(T x) { return x * 2.0; }
    float twice_f(float x) @explicit_specialization(twice, float) { return x + x + 1.0; }
    generic<T> T apply(T x) { return twice(x); }
    compute {
        void main() {
            float a = 2.0;
            float b = apply(a);
            float c = twice(b);
            int n = 3;
            int m = twice(n);
        }
    }
}
"""

MONOMORPHIZING_TARGETS = [
    "directx",
    "opengl",
    "metal",
    "wgsl",
    "cuda",
    "hip",
    "slang",
    "mojo",
]


def _translate(tmp_path, backend, source=SOURCE):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_parser_keeps_the_specialization_as_an_annotation():
    ast = Parser(Lexer(SOURCE).get_tokens()).parse()
    function = next(f for f in ast.functions if f.name == "twice_f")
    assert function.annotations["explicit_specialization"] == ("twice", ("float",))
    assert not any(
        attribute.name == "explicit_specialization" for attribute in function.attributes
    )


def test_malformed_specialization_attribute_is_a_syntax_error():
    source = SOURCE.replace(
        "@explicit_specialization(twice, float)", "@explicit_specialization(twice)"
    )
    with pytest.raises(SyntaxError, match="explicit_specialization"):
        Parser(Lexer(source).get_tokens()).parse()


@pytest.mark.parametrize("backend", MONOMORPHIZING_TARGETS)
def test_instantiations_for_the_specialized_arguments_call_the_specialization(
    tmp_path, backend
):
    code = _translate(tmp_path, backend)
    assert "explicit_specialization" not in code
    # apply<float> and the direct twice(b) both reach the specialization.
    assert code.count("twice_f(") >= 3
    # twice<int> still instantiates the primary's body.
    assert "twice_int" in code or "twice_i32" in code


def test_spirv_runs_the_specialized_body(tmp_path):
    code = _translate(tmp_path, "vulkan", SOURCE.replace("int m = twice(n);", ""))
    # Only the primary's body multiplies; every float instantiation is the specialization.
    assert "OpFMul" not in code


def test_rust_generics_refuse_explicit_specializations(tmp_path):
    from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

    with pytest.raises(RustLoweringRefused, match="explicit specialization twice_f"):
        _translate(tmp_path, "rust")
