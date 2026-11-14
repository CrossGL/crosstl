"""A malformed ``generic<...>`` declaration is a syntax error, not a dropped declaration.

Before, ``parse_generic_declaration`` caught any ``SyntaxError`` raised inside the wrapper, skipped the balanced
declaration and returned ``None``, so the program silently lost the struct or function. A ``generic`` keyword
without ``<`` or with no declaration after its parameters was skipped the same way.
"""

import pytest

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _in_shader(declaration):
    return f"""
shader S {{
    {declaration}
    compute {{
        layout(local_size_x = 1) in;
        void main() {{}}
    }}
}}
"""


@pytest.mark.parametrize(
    "declaration, message",
    [
        (
            "generic<T> fn apply(value: T, mapper: fn(T) -> T) -> T { return mapper(value); }",
            "Expected RPAREN",
        ),
        ("generic T struct Box { T value; }", "Expected '<' after 'generic'"),
        (
            "generic<T> const int LIMIT = 4;",
            "Expected a struct, enum, function or trait after the generic parameters",
        ),
    ],
)
@pytest.mark.parametrize("scope", ["shader", "module"])
def test_malformed_generic_declaration_raises(declaration, message, scope):
    source = _in_shader(declaration) if scope == "shader" else declaration
    with pytest.raises(SyntaxError, match=message):
        _parse(source)


def test_generic_declaration_truncated_after_its_parameters_raises():
    with pytest.raises(SyntaxError, match="generic declaration"):
        _parse("generic<T>")


def test_well_formed_generic_declarations_still_parse():
    ast = _parse(
        _in_shader(
            "generic<T> struct Box { T value; }\n"
            "    generic<T> T identity(T value) { return value; }"
        )
    )
    (box,) = (struct for struct in ast.structs if struct.name == "Box")
    assert [param.name for param in box.generic_params] == ["T"]
    (identity,) = (func for func in ast.functions if func.name == "identity")
    assert [param.name for param in identity.generic_params] == ["T"]
