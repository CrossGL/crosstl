from hypothesis import assume, given, settings
from hypothesis import strategies as st

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def parse_program_code(code):
    return Parser(Lexer(code).tokens).parse_program()


def import_metadata(shader):
    return [(node.path, node.alias, node.items) for node in shader.imports]


def preprocessor_metadata(shader):
    return [(node.directive, node.content) for node in shader.preprocessors]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    module_segments=st.lists(IDENTIFIER_SUFFIXES, min_size=2, max_size=4),
    item_names=st.lists(IDENTIFIER_SUFFIXES, min_size=2, max_size=4, unique=True),
)
def test_generated_import_forms_preserve_paths_aliases_and_items(
    suffix,
    module_segments,
    item_names,
):
    assume(suffix not in module_segments)
    dotted_path = ".".join(module_segments)
    qualified_path = "::".join(reversed(module_segments))
    alias = f"{suffix}_module"
    file_path = f"shaders/{suffix}.crossgl"
    items = ", ".join(item_names)
    code = f"""
    import {dotted_path} as {alias};
    use {qualified_path};
    from {dotted_path} import {items};
    import "{file_path}" as file_{suffix};

    shader ImportContracts_{suffix} {{
    }}
    """

    ast = parse_code(code)

    assert import_metadata(ast) == [
        (dotted_path, alias, None),
        (qualified_path, None, None),
        (dotted_path, None, item_names),
        (file_path, f"file_{suffix}", None),
    ]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    profile_minor=st.integers(min_value=0, max_value=9),
    define_value=st.integers(min_value=0, max_value=255),
    precision=st.sampled_from(("lowp", "mediump", "highp")),
)
def test_generated_preprocessors_survive_parse_entrypoints(
    suffix,
    profile_minor,
    define_value,
    precision,
):
    code = f"""
    #version 46{profile_minor} core
    precision {precision} float;

    shader PreprocessorContracts_{suffix} {{
        #define LOCAL_{suffix} {define_value}
        precision mediump int;
        const int COUNT_{suffix} = {define_value};
    }}
    """
    expected = [
        ("version", f"46{profile_minor} core"),
        ("precision", f"{precision} float"),
        ("define", f"LOCAL_{suffix} {define_value}"),
        ("precision", "mediump int"),
    ]

    assert preprocessor_metadata(parse_code(code)) == expected
    assert preprocessor_metadata(parse_program_code(code)) == expected
