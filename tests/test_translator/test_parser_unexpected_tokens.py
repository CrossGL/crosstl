"""A token the CrossGL parser cannot place is a syntax error, not a dropped declaration.

Before, the top-level, shader, stage, structure, trait and enum loops skipped any token they could not read, one
at a time, and the expression parser read any token as an identifier. ``struct S { float a junk; float b; };``
lost ``b``, ``float v = ;`` gave ``v`` the default value ``;``, a structure cut off at the end of input vanished,
and a unit with more than 10,000 declarations was truncated with a printed warning.
"""

import pytest

import crosstl
from crosstl.translator.ast import StructNode
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _shader(declarations="", stage=""):
    return f"""
shader S {{
    {declarations}
    compute {{
        {stage}
        void main() {{}}
    }}
}}
"""


@pytest.mark.parametrize(
    "source, message",
    [
        (
            "+ ; struct S { float a; };",
            "Unexpected PLUS '\\+' in a top-level declaration",
        ),
        (") void main() {}", "Unexpected RPAREN"),
        (_shader("+ ;"), "in a shader declaration"),
        (_shader(stage="+ ;"), "in a compute stage block"),
        ("layout(std140) + ;", "in a layout-qualified declaration"),
        ("struct { float a; } s;", "Expected a structure name after 'struct'"),
        ("struct S float a;", "in structure 'S' before its '{'"),
        ("struct S", "end of input in structure 'S'"),
        ("struct S { float a; ", "Unterminated structure 'S'"),
        ("struct S { float ; };", "before its name"),
        ("struct S { float a, ; };", "before its name"),
        ("struct S { float a junk; float b; };", "in structure member 'a'"),
        ("struct S { a: float junk; };", "in structure member 'a'"),
        ("struct S { generic<T> x; float a; };", "cannot declare a generic"),
        ("struct S { float v = ; };", "Unexpected SEMICOLON ';' in an expression"),
        ("enum E { A { x: float + } };", "in enum variant 'A'"),
        ("trait T { + }", "in trait 'T'"),
        ("trait T { void f(); ", "Unterminated trait 'T'"),
        ("trait { }", "Expected IDENTIFIER"),
    ],
)
def test_unreadable_tokens_are_refused(source, message):
    with pytest.raises(SyntaxError, match=message):
        _parse(source)


def test_an_expression_cannot_start_with_punctuation():
    source = _shader(stage="void f() { float x = ); }")
    with pytest.raises(SyntaxError, match="Unexpected RPAREN"):
        Parser(Lexer(source).get_tokens()).parse()


def test_switch_body_before_the_first_case_is_refused():
    source = _shader(
        stage="void f(int k) { switch (k) { + case 0: break; } }",
    )
    with pytest.raises(SyntaxError, match="switch body before its first case"):
        Parser(Lexer(source).get_tokens()).parse()


def test_unterminated_switch_body_is_refused():
    source = "void f(int k) { switch (k) {"
    with pytest.raises(SyntaxError):
        Parser(Lexer(source).get_tokens()).parse()


def test_empty_declarations_and_forward_declarations_are_accepted():
    ast = _parse(
        "struct S;\n;\nstruct S { float a;; float b; };\n"
        + _shader("struct T { int c; };;", "uniform float scale;;")
    )
    structs = {node.name: node for node in ast.structs if isinstance(node, StructNode)}
    assert [member.name for member in structs["S"].members] == ["a", "b"]
    assert [member.name for member in structs["T"].members] == ["c"]


def test_member_default_values_still_parse():
    (struct,) = _parse("struct S { float v = 1.5; int k; };").structs
    assert [member.name for member in struct.members] == ["v", "k"]
    assert struct.members[0].default_value is not None


def test_keywords_spelled_like_names_are_still_expressions():
    ast = Parser(
        Lexer(_shader(stage="vec4 f(float x) { return vec4(float(x)); }")).get_tokens(),
    ).parse()
    assert ast is not None


def test_annotations_before_let_bindings_are_kept():
    # Before, ``@`` before ``let`` was skipped token by token at shader scope.
    ast = _parse(
        _shader('@ deprecated("old") let pi = 3.14;\n@ doc_hidden let k = 42;')
    )
    annotated = {
        variable.name: [attribute.name for attribute in variable.attributes]
        for variable in ast.global_variables
    }
    assert annotated == {"pi": ["deprecated"], "k": ["doc_hidden"]}


def test_more_than_ten_thousand_declarations_are_all_kept():
    count = 10_050
    source = "".join(f"const int C{index} = {index};\n" for index in range(count))
    assert len(_parse(source).constants) == count


@pytest.mark.parametrize(
    "declarations",
    [
        "struct Rec { float a junk; float b; };",
        "struct Rec { float a = ; };",
        "+ ;",
    ],
)
def test_translation_refuses_unreadable_crossgl(tmp_path, declarations):
    path = tmp_path / "shader.cgl"
    path.write_text(_shader(declarations), encoding="utf-8")
    with pytest.raises(SyntaxError):
        crosstl.translate(str(path), backend="metal")
