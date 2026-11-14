"""Declaration, brace-block, using-directive and constexpr-call scans match their previous implementations.

Each reference below is the previous implementation, kept verbatim apart from its name and the ``self`` or
``preprocessor`` argument, and is compared with the current one on generated C++-like text built from the
identifiers, qualifiers, delimiters, comments and literals those scans treat specially.
"""

import random
import re

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor, _MetalTypeAliasBinding
from crosstl.backend.Metal.text_scanners import (
    brace_block_spans,
    find_named_type_declarations,
)
from crosstl.project.pipeline import (
    _metal_block_spans,
    _metal_using_namespace_directives,
)

FRAGMENTS = [
    "S",
    "SS",
    "S2",
    "Foo",
    "Bar",
    "Alias",
    "x",
    "y1",
    "1",
    "1S",
    "_",
    "\u00e9",
    "\u00b2",
    "const",
    "volatile",
    "using",
    "namespace",
    "ns",
    "::",
    "f",
    "g",
    "<",
    ">",
    " ",
    "  ",
    "\n",
    "\t",
    "*",
    "&",
    ".",
    "->",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ";",
    "=",
    "/",
    "//",
    "/*",
    "*/",
    "\\",
    '"',
    "'",
    '"S x;"',
    "'{'",
    "// S x; {\n",
    "/* f(1) } */",
    "{",
    "}",
    "{ S x; }",
    "f(x)",
    "f(",
    "g<1>(S, x)",
    "ns::f (1, (2))",
    "S const* p;",
    "using namespace ns;",
    "using namespace ::ns :: S ;",
    "using namespace",
]


def generated_sources(seed, count=500, length=40):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(0, length)))


def reference_collect_struct_variable_types(
    self, code, struct_names, struct_spans, *, include_indirect=False
):
    declarations = {}
    for struct_name in sorted(struct_names):
        pattern = re.compile(
            rf"\b{re.escape(struct_name)}\b"
            r"(?:\s+(?:const|volatile)\b)*\s*"
            r"(?P<indirection>[*&]+)?\s*"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"(?=[;,)=\[{(])"
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            preceding = match.start() - 1
            while preceding >= 0 and code[preceding].isspace():
                preceding -= 1
            if preceding >= 0 and code[preceding] in ".>":
                continue
            indirection = match.group("indirection") or ""
            if indirection and not include_indirect:
                continue
            declarations.setdefault(match.group("name"), []).append(
                (match.start(), f"{struct_name}{indirection}")
            )
    for entries in declarations.values():
        entries.sort(key=lambda item: item[0])
    return declarations


def reference_collect_aliased_struct_variable_types(
    self, code, type_aliases, struct_spans
):
    declarations = {}
    for alias in sorted(type_aliases):
        pattern = re.compile(
            rf"\b{re.escape(alias)}\b"
            r"(?:\s+(?:const|volatile)\b)*\s*"
            r"(?P<indirection>[*&]+)?\s*"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"(?=[;,)=\[{(])"
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            resolved = self._resolve_struct_type_alias_at(
                type_aliases, alias, match.start()
            )
            if resolved is None:
                continue
            indirection = match.group("indirection") or ""
            declarations.setdefault(match.group("name"), []).append(
                (match.start(), f"{resolved}{indirection}")
            )
    for entries in declarations.values():
        entries.sort(key=lambda item: item[0])
    return declarations


def reference_block_spans(preprocessor, source):
    spans = []
    stack = []
    i = 0
    while i < len(source):
        if source[i] in "\"'":
            _literal, consumed = preprocessor._read_string(source, i)
            i += consumed
            continue
        if source.startswith("//", i):
            end = source.find("\n", i)
            if end == -1:
                break
            i = end + 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        if source[i] == "{":
            stack.append(i)
        elif source[i] == "}" and stack:
            spans.append((stack.pop(), i + 1))
        i += 1
    return spans


def reference_using_namespace_directives(
    preprocessor, source, block_spans, namespace_spans, known_namespaces
):
    from crosstl.project.pipeline import _metal_enclosing_block_span

    directives = []
    i = 0
    while i < len(source):
        if source[i] in "\"'":
            _literal, consumed = preprocessor._read_string(source, i)
            i += consumed
            continue
        if source.startswith("//", i):
            end = source.find("\n", i + 2)
            if end == -1:
                break
            i = end + 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        if not source.startswith("using", i) or (
            i > 0 and (source[i - 1].isalnum() or source[i - 1] == "_")
        ):
            i += 1
            continue
        match = re.match(
            r"using\s+namespace\s+"
            r"(?P<name>(?:::)?[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*;",
            source[i:],
        )
        if match is None:
            i += len("using")
            continue
        directive_scope = _metal_enclosing_block_span(block_spans, i)
        raw_name = re.sub(r"\s+", "", match.group("name"))
        globally_qualified = raw_name.startswith("::")
        name = raw_name.lstrip(":")
        directive_namespace = preprocessor._namespace_at(list(namespace_spans), i)
        candidates = preprocessor._namespace_lookup_names(
            name, directive_namespace, globally_qualified
        )
        resolved = next(
            (candidate for candidate in candidates if candidate in known_namespaces),
            candidates[-1],
        )
        directives.append((i, directive_scope, resolved))
        i += match.end()
    return directives


def reference_find_static_constexpr_calls(self, expression, function_names):
    from crosstl.backend.Metal.preprocessor import _MetalConstexprCall

    calls = []
    i = 0
    while i < len(expression):
        if expression[i] in "\"'":
            _literal, consumed = self._read_string(expression, i)
            i += consumed
            continue
        if expression.startswith("//", i):
            end = expression.find("\n", i)
            i = len(expression) if end == -1 else end + 1
            continue
        if expression.startswith("/*", i):
            end = expression.find("*/", i + 2)
            i = len(expression) if end == -1 else end + 2
            continue
        if not (expression[i].isalpha() or expression[i] == "_"):
            i += 1
            continue

        name, consumed = self._read_identifier(expression, i)
        if name not in function_names:
            i += consumed
            continue
        name_end = i + consumed
        cursor = name_end
        while cursor < len(expression) and expression[cursor].isspace():
            cursor += 1
        template_arguments = None
        if cursor < len(expression) and expression[cursor] == "<":
            angle_end = self._find_matching_angle(expression, cursor)
            if angle_end is None:
                i += consumed
                continue
            template_arguments = self._split_top_level_commas(
                expression[cursor + 1 : angle_end]
            )
            cursor = angle_end + 1
            while cursor < len(expression) and expression[cursor].isspace():
                cursor += 1
        if cursor >= len(expression) or expression[cursor] != "(":
            i += consumed
            continue
        arguments_end = self._find_matching_delimiter(expression, cursor, "(", ")")
        if arguments_end is None:
            i += consumed
            continue
        argument_text = expression[cursor + 1 : arguments_end]
        arguments = (
            self._split_top_level_commas(argument_text) if argument_text.strip() else []
        )
        call_start = self._scoped_identifier_start(expression, i)
        qualified_name = re.sub(
            r"\s*::\s*", "::", expression[call_start:name_end].strip()
        )
        calls.append(
            _MetalConstexprCall(
                name=name,
                qualified_name=qualified_name,
                template_arguments=template_arguments,
                arguments=arguments,
                span=(call_start, arguments_end + 1),
                argument_open=cursor,
            )
        )
        i = arguments_end + 1
    return calls


@pytest.fixture(scope="module")
def preprocessor():
    return MetalPreprocessor()


STRUCT_NAME_SETS = [
    set(),
    {"S"},
    {"S", "SS", "S2", "Foo"},
    {"Bar", "x", "_"},
    {"S", "Foo<S>", "ns::Bar"},
]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("include_indirect", [False, True])
def test_struct_variable_types_match_per_name_scans(
    preprocessor, seed, include_indirect
):
    for index, source in enumerate(generated_sources(seed)):
        names = STRUCT_NAME_SETS[index % len(STRUCT_NAME_SETS)]
        spans = [(0, len(source) // 4)] if index % 3 == 0 else []
        assert preprocessor._collect_struct_variable_types(
            source, names, spans, include_indirect=include_indirect
        ) == reference_collect_struct_variable_types(
            preprocessor, source, names, spans, include_indirect=include_indirect
        ), source


@pytest.mark.parametrize("seed", range(4))
def test_aliased_struct_variable_types_match_per_alias_scans(preprocessor, seed):
    for index, source in enumerate(generated_sources(seed + 10)):
        middle = len(source) // 2
        aliases = {
            "Alias": [_MetalTypeAliasBinding(0, 0, len(source), "S")],
            "Bar": [_MetalTypeAliasBinding(middle, middle, len(source), "Foo")],
        }
        if index % 2:
            aliases["ns::Alias"] = [_MetalTypeAliasBinding(0, 0, len(source), "SS")]
        spans = [(middle, len(source))] if index % 3 == 0 else []
        assert preprocessor._collect_aliased_struct_variable_types(
            source, aliases, spans
        ) == reference_collect_aliased_struct_variable_types(
            preprocessor, source, aliases, spans
        ), source


def test_named_type_declarations_are_whole_words_in_source_order():
    code = "SS a; S b, S2 *c; xS d; S const & e = f; Foo(S g);"
    found = [
        (name, match.group("name"))
        for name, match in find_named_type_declarations(code, {"S", "SS", "Foo"})
    ]
    assert found == [("SS", "a"), ("S", "b"), ("S", "e"), ("S", "g")]


@pytest.mark.parametrize("seed", range(4))
def test_brace_block_spans_match_character_scan(preprocessor, seed):
    for source in generated_sources(seed + 20):
        expected = reference_block_spans(preprocessor, source)
        assert brace_block_spans(source, preprocessor._read_string) == expected, source
        assert _metal_block_spans(preprocessor, source) == expected, source


@pytest.mark.parametrize("seed", range(4))
def test_using_namespace_directives_match_character_scan(preprocessor, seed):
    for source in generated_sources(seed + 30, length=60):
        source = source.replace("namespace", "namespace ns", 1)
        block_spans = reference_block_spans(preprocessor, source)
        namespace_spans = preprocessor._find_namespace_spans(source)
        known = {"ns", "ns::S"}
        assert _metal_using_namespace_directives(
            preprocessor, source, block_spans, namespace_spans, known
        ) == reference_using_namespace_directives(
            preprocessor, source, block_spans, namespace_spans, known
        ), source


def test_using_namespace_directive_after_literal_and_comment(preprocessor):
    source = "char c = '\"'; // using namespace a;\nusing  namespace ::ns ;{ using namespace ns::S; }"
    block_spans = reference_block_spans(preprocessor, source)
    namespace_spans = preprocessor._find_namespace_spans(source)
    directives = _metal_using_namespace_directives(
        preprocessor, source, block_spans, namespace_spans, {"ns", "ns::S"}
    )
    assert [(offset, namespace) for offset, _scope, namespace in directives] == [
        (source.index("using  namespace"), "ns"),
        (source.index("using namespace ns::S"), "ns::S"),
    ]
    assert directives[0][1] is None and directives[1][1] == block_spans[0]


@pytest.mark.parametrize("seed", range(4))
def test_static_constexpr_calls_match_character_scan(preprocessor, seed):
    name_sets = [{"f"}, {"f", "g", "S"}, {"x"}, {"absent"}, set()]
    for index, source in enumerate(generated_sources(seed + 40)):
        names = name_sets[index % len(name_sets)]
        assert preprocessor._find_static_constexpr_calls(
            source, names
        ) == reference_find_static_constexpr_calls(preprocessor, source, names), source
