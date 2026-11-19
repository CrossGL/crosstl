"""A call to a generic primary keeps its explicit specializations reachable.

The generators route calls to a generic primary through its explicit
specializations, so those specializations are written to the output even when
no call names them directly. Reachability analysis used to mark them dead and
skip the undeclared-name checks, letting undeclared types from a specialization
body reach the generated target.
"""

import pytest

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import UndeclaredTypeError, apply_rule_carrying
from crosstl.translator.semantics.declared_types import (
    dead_function_ids,
    undeclared_type_uses,
)

SPECIALIZATION_WITH_UNDECLARED_TYPE = """
shader P {
    generic<T> T twice(T x) { return x * 2.0; }
    float twice_f(float x) @explicit_specialization(twice, float) {
        foo f;
        return x + x;
    }

    compute {
        buffer float outp[];

        @compute @workgroup_size(1, 1, 1)
        void main() { outp[0] = twice(outp[3]); }
    }
}
"""


def _program(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_used_primary_keeps_its_explicit_specialization_live():
    program = _program(SPECIALIZATION_WITH_UNDECLARED_TYPE)
    assert dead_function_ids(program) == set()


def test_specialization_body_is_checked_for_undeclared_types():
    program = _program(SPECIALIZATION_WITH_UNDECLARED_TYPE)
    assert [name for name, _ in undeclared_type_uses(program)] == ["foo"]


@pytest.mark.parametrize("target", ["metal", "directx", "opengl", "vulkan"])
def test_translation_refuses_reachable_specialization(target):
    with pytest.raises(UndeclaredTypeError):
        apply_rule_carrying(_program(SPECIALIZATION_WITH_UNDECLARED_TYPE), "cgl", target)
