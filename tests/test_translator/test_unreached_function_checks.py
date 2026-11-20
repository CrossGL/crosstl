"""Functions no stage entry reaches are written to the output, so the checks read them.

None of the generators drops a function no stage entry reaches, so an undeclared type, identifier or call in an
unreached helper, and in the explicit specialization of a primary nothing calls, reaches the generated output.
The undeclared-name checks therefore read every function body instead of only the reached ones.
"""

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    UndeclaredFunctionError,
    UndeclaredIdentifierError,
    UndeclaredTypeError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.declared_identifiers import (
    generate_refusing_undeclared_identifiers,
    recorded_undeclared_identifier,
)
from crosstl.translator.semantics.declared_types import (
    dead_function_ids,
    undeclared_type_uses,
)

TARGETS = ["metal", "directx", "opengl", "vulkan"]

UNUSED_HELPER_WITH_UNDECLARED_TYPE = """
shader P {
    float unused(float x) { foo f; return x; }
    compute {
        buffer float* outp;
        void main() { outp[0] = outp[3]; }
    }
}
"""

UNUSED_HELPER_WITH_UNDECLARED_IDENTIFIER = """
shader P {
    float unused(float x) { return x + nosuchvar; }
    compute {
        buffer float* outp;
        void main() { outp[0] = outp[3]; }
    }
}
"""

UNUSED_HELPER_WITH_UNDECLARED_CALL = """
shader P {
    float unused(float x) { return nosuch(x); }
    compute {
        buffer float* outp;
        void main() { outp[0] = outp[3]; }
    }
}
"""

UNUSED_PRIMARY_SPECIALIZATION_WITH_UNDECLARED_TYPE = """
shader P {
    generic<T> T twice(T x) { return x * 2.0; }
    float twice_f(float x) @explicit_specialization(twice, float) {
        foo f;
        return x + x;
    }
    compute {
        buffer float* outp;
        void main() { outp[0] = outp[3]; }
    }
}
"""

UNUSED_VALID_HELPER = """
shader P {
    float unused(float x) { float y = x * 2.0; return y; }
    compute {
        buffer float* outp;
        void main() { outp[0] = outp[3]; }
    }
}
"""


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_unreached_helper_is_still_unreachable_but_checked():
    program = _program(UNUSED_HELPER_WITH_UNDECLARED_TYPE)
    assert dead_function_ids(program) != set()
    assert [name for name, _ in undeclared_type_uses(program)] == ["foo"]


def test_unused_primary_specialization_is_checked():
    program = _program(UNUSED_PRIMARY_SPECIALIZATION_WITH_UNDECLARED_TYPE)
    assert [name for name, _ in undeclared_type_uses(program)] == ["foo"]


@pytest.mark.parametrize("target", TARGETS)
def test_translation_refuses_the_unreached_undeclared_type(target):
    with pytest.raises(UndeclaredTypeError):
        apply_rule_carrying(_program(UNUSED_HELPER_WITH_UNDECLARED_TYPE), "cgl", target)


@pytest.mark.parametrize("target", TARGETS)
def test_translation_refuses_the_unreached_undeclared_call(target):
    with pytest.raises(UndeclaredFunctionError):
        apply_rule_carrying(_program(UNUSED_HELPER_WITH_UNDECLARED_CALL), "cgl", target)


@pytest.mark.parametrize("target", TARGETS)
def test_rule_carrying_records_the_unreached_undeclared_identifier(target):
    program = _program(UNUSED_HELPER_WITH_UNDECLARED_IDENTIFIER)
    apply_rule_carrying(program, "cgl", target)
    error = recorded_undeclared_identifier(program)
    assert isinstance(error, UndeclaredIdentifierError)
    assert error.identifier == "nosuchvar"
    with pytest.raises(UndeclaredIdentifierError):
        generate_refusing_undeclared_identifiers(program, lambda: "code")


def test_generation_refuses_the_unreached_undeclared_identifier(tmp_path):
    path = tmp_path / "unused.cgl"
    path.write_text(UNUSED_HELPER_WITH_UNDECLARED_IDENTIFIER)
    with pytest.raises(UndeclaredIdentifierError) as caught:
        crosstl.translate(path, backend="metal")
    assert caught.value.identifier == "nosuchvar"


@pytest.mark.parametrize("target", TARGETS)
def test_unused_primary_specialization_is_refused(target):
    with pytest.raises(UndeclaredTypeError):
        apply_rule_carrying(
            _program(UNUSED_PRIMARY_SPECIALIZATION_WITH_UNDECLARED_TYPE), "cgl", target
        )


@pytest.mark.parametrize("target", TARGETS)
def test_valid_unreached_helper_still_translates(target):
    apply_rule_carrying(_program(UNUSED_VALID_HELPER), "cgl", target)
