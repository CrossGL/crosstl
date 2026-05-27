from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayLiteralNode,
    ArrayType,
    ForInNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
    PrimitiveType,
    RangeNode,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@dataclass(frozen=True)
class IntegerLiteralCase:
    source: str
    value: int
    literal_type: str


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@st.composite
def integer_literal_cases(draw):
    value = draw(st.integers(min_value=0, max_value=4095))
    literal_kind = draw(
        st.sampled_from(
            (
                "decimal",
                "decimal_uint",
                "hex",
                "hex_uint",
                "binary",
                "binary_uint",
                "octal",
                "octal_uint",
            )
        )
    )

    if literal_kind == "decimal":
        return IntegerLiteralCase(str(value), value, "int")
    if literal_kind == "decimal_uint":
        return IntegerLiteralCase(f"{value}u", value, "uint")
    if literal_kind == "hex":
        return IntegerLiteralCase(f"0x{value:x}", value, "int")
    if literal_kind == "hex_uint":
        return IntegerLiteralCase(f"0x{value:x}U", value, "uint")
    if literal_kind == "binary":
        return IntegerLiteralCase(f"0b{value:b}", value, "int")
    if literal_kind == "binary_uint":
        return IntegerLiteralCase(f"0b{value:b}u", value, "uint")
    if literal_kind == "octal":
        return IntegerLiteralCase(f"0o{value:o}", value, "int")
    return IntegerLiteralCase(f"0o{value:o}U", value, "uint")


def assert_literal(node, value, literal_type):
    assert isinstance(node, LiteralNode)
    assert node.value == value
    assert isinstance(node.literal_type, PrimitiveType)
    assert node.literal_type.name == literal_type


@settings(max_examples=40, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    literal_case=integer_literal_cases(),
)
def test_generated_integer_literals_preserve_base_value_and_unsigned_suffix(
    suffix,
    literal_case,
):
    code = f"""
    shader IntegerLiteralContracts_{suffix} {{
        int read_{suffix}() {{
            int value_{suffix} = {literal_case.source};
            return value_{suffix};
        }}
    }}
    """

    ast = parse_code(code)
    initializer = ast.functions[0].body.statements[0].initial_value

    assert_literal(initializer, literal_case.value, literal_case.literal_type)


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    whole=st.integers(min_value=0, max_value=128),
    fraction=st.integers(min_value=1, max_value=999),
    use_suffix=st.booleans(),
    enabled=st.booleans(),
    marker=st.characters(min_codepoint=65, max_codepoint=90),
)
def test_generated_scalar_literals_preserve_primitive_literal_types(
    suffix,
    whole,
    fraction,
    use_suffix,
    enabled,
    marker,
):
    float_source = f"{whole}.{fraction:03d}{'f' if use_suffix else ''}"
    expected_float = float(f"{whole}.{fraction:03d}")
    bool_source = "true" if enabled else "false"
    code = f"""
    shader ScalarLiteralContracts_{suffix} {{
        void read_{suffix}() {{
            float weight_{suffix} = {float_source};
            bool enabled_{suffix} = {bool_source};
            string label_{suffix} = "label_{suffix}";
            char marker_{suffix} = '{marker}';
        }}
    }}
    """

    ast = parse_code(code)
    weight, enabled_var, label, marker_var = ast.functions[0].body.statements

    assert_literal(weight.initial_value, expected_float, "float")
    assert_literal(enabled_var.initial_value, enabled, "bool")
    assert_literal(label.initial_value, f"label_{suffix}", "string")
    assert_literal(marker_var.initial_value, marker, "char")


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    start=st.integers(min_value=0, max_value=16),
    span=st.integers(min_value=1, max_value=16),
)
def test_generated_range_and_array_literals_preserve_expression_nodes(
    suffix,
    start,
    span,
):
    end = start + span
    midpoint = start + (span // 2)
    code = f"""
    shader RangeLiteralContracts_{suffix} {{
        int scan_{suffix}() {{
            int values_{suffix}[3] = {{{start}, {midpoint}, {end}}};
            for i_{suffix} in {start}..{end} {{
            }}
            for j_{suffix} in {start}..={end} {{
            }}
            return values_{suffix}[0];
        }}
    }}
    """

    ast = parse_code(code)
    values, exclusive_loop, inclusive_loop = ast.functions[0].body.statements[:3]

    assert isinstance(values.var_type, ArrayType)
    assert_literal(values.var_type.size, 3, "int")
    assert isinstance(values.initial_value, ArrayLiteralNode)
    assert [element.value for element in values.initial_value.elements] == [
        start,
        midpoint,
        end,
    ]

    assert isinstance(exclusive_loop, ForInNode)
    assert exclusive_loop.pattern == f"i_{suffix}"
    assert isinstance(exclusive_loop.iterable, RangeNode)
    assert exclusive_loop.iterable.inclusive is False
    assert_literal(exclusive_loop.iterable.start, start, "int")
    assert_literal(exclusive_loop.iterable.end, end, "int")

    assert isinstance(inclusive_loop, ForInNode)
    assert inclusive_loop.pattern == f"j_{suffix}"
    assert isinstance(inclusive_loop.iterable, RangeNode)
    assert inclusive_loop.iterable.inclusive is True
    assert_literal(inclusive_loop.iterable.start, start, "int")
    assert_literal(inclusive_loop.iterable.end, end, "int")


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    dimension=st.integers(min_value=1, max_value=16),
    uint_value=st.integers(min_value=0, max_value=255),
    enabled=st.booleans(),
)
def test_generated_literal_generic_arguments_preserve_mixed_argument_ir(
    suffix,
    dimension,
    uint_value,
    enabled,
):
    enabled_source = "true" if enabled else "false"
    code = f"""
    shader GenericLiteralContracts_{suffix} {{
        Resource_{suffix}<float, {dimension}, 0x{uint_value:x}u, {enabled_source}, Mode_{suffix}::Read>
            resource_{suffix};
    }}
    """

    ast = parse_code(code)
    resource_type = ast.global_variables[0].var_type

    assert isinstance(resource_type, NamedType)
    assert resource_type.name == f"Resource_{suffix}"
    element_type, size_arg, uint_arg, bool_arg, mode_arg = resource_type.generic_args
    assert isinstance(element_type, PrimitiveType)
    assert element_type.name == "float"
    assert_literal(size_arg, dimension, "int")
    assert_literal(uint_arg, uint_value, "uint")
    assert_literal(bool_arg, enabled, "bool")
    assert isinstance(mode_arg, IdentifierNode)
    assert mode_arg.name == f"Mode_{suffix}::Read"
