"""The WGSL lexer and parser (crosstl/backend/WGSL): tokens, template lists and the statement grammar."""

import pytest

from crosstl.backend.WGSL import WgslAst as A
from crosstl.backend.WGSL.WgslLexer import (
    WgslLexer,
    WgslSyntaxError,
    discover_templates,
)
from crosstl.backend.WGSL.WgslParser import WgslParser


def _kinds(text):
    return [(t.kind, t.text) for t in WgslLexer(text).tokenize()[:-1]]


def _parse(text):
    return WgslParser(WgslLexer(text).tokenize()).parse()


def test_template_lists_close_with_one_greater_than_each():
    tokens = _kinds("var<storage, read_write> a: array<vec3<f32>>;")
    assert [t for t in tokens if t[0].startswith("template")] == [
        ("template_start", "<"),
        ("template_end", ">"),
        ("template_start", "<"),
        ("template_start", "<"),
        ("template_end", ">"),
        ("template_end", ">"),
    ]


def test_comparisons_and_shifts_are_operators():
    assert ("op", "<") in _kinds("let x = a < b && c > d;")
    assert not any(
        kind.startswith("template") for kind, _ in _kinds("let x = a < b && c > d;")
    )
    assert ("op", ">>") in _kinds("let y = a >> 2u;")
    assert ("op", "<<=") in _kinds("x <<= 1u;")
    assert ("op", "<=") in _kinds("let z = a <= b;")


def test_template_discovery_follows_the_specification_examples():
    # WGSL section 3.9: 'a<b || c>d' is not a template list, 'array<i32, select(2, 3, a > b)>' is one.
    starts, ends = discover_templates("let x = a<b || c>d;")
    assert not starts and not ends
    text = "var c: array<i32, select(2, 3, a > b)>;"
    starts, ends = discover_templates(text)
    assert [text[i] for i in sorted(starts)] == ["<"]
    assert sorted(ends) == [len(text) - 2]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("1", "int"),
        ("1i", "int"),
        ("0x7Fu", "int"),
        ("1.5", "float"),
        ("1f", "float"),
        ("1.5h", "float"),
        ("1e-3", "float"),
        (".5", "float"),
        ("2.", "float"),
        ("0x1.8p1", "float"),
        ("0x1p4f", "float"),
    ],
)
def test_numeric_literals_keep_their_spelling(text, kind):
    assert _kinds(text) == [(kind, text)]


def test_block_comments_nest():
    assert _kinds("a /* one /* two */ still */ b // line\nc") == [
        ("ident", "a"),
        ("ident", "b"),
        ("ident", "c"),
    ]


def test_unterminated_comment_is_refused():
    with pytest.raises(WgslSyntaxError, match="unterminated block comment"):
        WgslLexer("/* /* */").tokenize()


def test_loop_with_continuing_and_break_if():
    module = _parse(
        "fn f() { var i = 0; loop { i++; if i > 4 { continue; } continuing { i += 1; break if i >= 8; } } }"
    )
    loop = module.declarations[0].body.statements[1]
    assert isinstance(loop, A.Loop)
    assert isinstance(loop.continuing.statements[0], A.Assign)
    assert isinstance(loop.break_if.condition, A.Binary)


def test_switch_selectors_may_include_default():
    module = _parse("fn f(x: i32) { switch x { case 1, 2: { } case 3, default { } } }")
    clauses = module.declarations[0].body.statements[0].clauses
    assert [len(c.selectors) for c in clauses] == [2, 2]
    assert clauses[1].selectors[1] == "default"


def test_attributes_directives_and_module_declarations():
    module = _parse(
        "enable f16;\n"
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n"
        "override n: u32 = 4u;\n"
        "alias V = vec2<f32>;\n"
        "struct S { @location(0) a: f32, b: V, }\n"
        "@compute @workgroup_size(n, 1) fn main(@builtin(global_invocation_id) id: vec3u) { _ = b[id.x]; }\n"
    )
    assert module.directives[0].names == ["f16"]
    kinds = [type(d).__name__ for d in module.declarations]
    assert kinds == ["GlobalVar", "Override", "Alias", "Struct", "Function"]
    main = module.declarations[-1]
    assert [a.name for a in main.attributes] == ["compute", "workgroup_size"]
    assert main.body.statements[0].lhs is None


def test_for_loop_parts():
    loop = (
        _parse("fn f() { for (var i: i32 = 0; i < 4; i++) { } }")
        .declarations[0]
        .body.statements[0]
    )
    assert isinstance(loop, A.For)
    assert isinstance(loop.init, A.Local) and isinstance(loop.update, A.Increment)


def test_statement_that_is_not_one_is_refused():
    with pytest.raises(WgslSyntaxError, match="expected an assignment"):
        _parse("fn f() { 1 + 2; }")
