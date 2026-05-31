from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    ArrayType,
    EnumNode,
    FunctionNode,
    LiteralNode,
    MatrixType,
    NamedType,
    PrimitiveType,
    StructNode,
    VectorType,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def assert_literal(node, value, literal_type="int"):
    assert isinstance(node, LiteralNode)
    assert node.value == value
    assert isinstance(node.literal_type, PrimitiveType)
    assert node.literal_type.name == literal_type


def attribute_values(attributes):
    return {
        attribute.name: [argument.value for argument in attribute.arguments]
        for attribute in attributes
    }


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    profile_minor=st.integers(min_value=0, max_value=9),
    define_value=st.integers(min_value=0, max_value=255),
)
def test_generated_imports_preprocessors_and_constants_preserve_root_ir(
    suffix,
    profile_minor,
    define_value,
):
    code = f"""
    #version 46{profile_minor} core
    #define FEATURE_{suffix} {define_value}
    import math_{suffix} as mx_{suffix};
    use bindings_{suffix};
    shader DeclarationRoot_{suffix} {{
        const int COUNT_{suffix} = {define_value};
    }}
    """

    ast = parse_code(code)

    assert [(node.directive, node.content) for node in ast.preprocessors] == [
        ("version", f"46{profile_minor} core"),
        ("define", f"FEATURE_{suffix} {define_value}"),
    ]
    assert [(node.path, node.alias, node.items) for node in ast.imports] == [
        (f"math_{suffix}", f"mx_{suffix}", None),
        (f"bindings_{suffix}", None, None),
    ]

    constant = ast.constants[0]
    assert constant.name == f"COUNT_{suffix}"
    assert isinstance(constant.const_type, PrimitiveType)
    assert constant.const_type.name == "int"
    assert_literal(constant.value, define_value)


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    set_index=st.integers(min_value=0, max_value=7),
    binding_index=st.integers(min_value=0, max_value=31),
    texture_slot=st.integers(min_value=0, max_value=31),
    palette_size=st.integers(min_value=1, max_value=8),
)
def test_generated_cbuffer_and_resource_declarations_preserve_metadata_ir(
    suffix,
    set_index,
    binding_index,
    texture_slot,
    palette_size,
):
    code = f"""
    shader ResourceDeclarationContracts_{suffix} {{
        cbuffer Camera_{suffix} @set({set_index}) @binding({binding_index}) {{
            mat4 viewProj_{suffix};
            float exposure_{suffix};
            vec4 palette_{suffix}[{palette_size}];
        }};
        Texture2D<float4> colorTex_{suffix} @texture({texture_slot});
    }}
    """

    ast = parse_code(code)

    cbuffer = ast.cbuffers[0]
    assert cbuffer.name == f"Camera_{suffix}"
    assert cbuffer.is_cbuffer is True
    assert attribute_values(cbuffer.attributes) == {
        "set": [set_index],
        "binding": [binding_index],
    }

    view_proj, exposure, palette = cbuffer.members
    assert view_proj.name == f"viewProj_{suffix}"
    assert isinstance(view_proj.member_type, MatrixType)
    assert view_proj.member_type.rows == 4
    assert view_proj.member_type.cols == 4

    assert exposure.name == f"exposure_{suffix}"
    assert isinstance(exposure.member_type, PrimitiveType)
    assert exposure.member_type.name == "float"

    assert palette.name == f"palette_{suffix}"
    assert isinstance(palette.member_type, ArrayType)
    assert isinstance(palette.member_type.element_type, VectorType)
    assert palette.member_type.element_type.size == 4
    assert_literal(palette.member_type.size, palette_size)

    resource = ast.global_variables[0]
    assert resource.name == f"colorTex_{suffix}"
    assert isinstance(resource.var_type, NamedType)
    assert resource.var_type.name == "Texture2D"
    assert isinstance(resource.var_type.generic_args[0], NamedType)
    assert resource.var_type.generic_args[0].name == "float4"
    assert attribute_values(resource.attributes) == {"texture": [texture_slot]}


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    enum_value=st.integers(min_value=0, max_value=1024),
)
def test_generated_generic_struct_function_and_enum_declarations_preserve_ir(
    suffix,
    enum_value,
):
    code = f"""
    shader GenericDeclarationContracts_{suffix} {{
        struct Payload_{suffix}<T: Scalar_{suffix} + Packable_{suffix}> @packed {{
            T value_{suffix};
            vec4 color_{suffix} : COLOR0;
        }};

        generic<U: Scalar_{suffix} + Packable_{suffix}>
        fn choose_{suffix}(left_{suffix}: U, right_{suffix}: U) -> U {{
            return left_{suffix};
        }}

        enum Mode_{suffix}: uint {{
            Read_{suffix} = {enum_value},
            Tuple_{suffix}(Payload_{suffix}, Error_{suffix}),
            Struct_{suffix} {{ color_{suffix}: vec4, count_{suffix}: int }},
        }}
    }}
    """

    ast = parse_code(code)

    payload = next(node for node in ast.structs if isinstance(node, StructNode))
    assert payload.name == f"Payload_{suffix}"
    assert [parameter.name for parameter in payload.generic_params] == ["T"]
    assert [
        constraint.name for constraint in payload.generic_params[0].constraints
    ] == [f"Scalar_{suffix}", f"Packable_{suffix}"]
    assert [attribute.name for attribute in payload.attributes] == ["packed"]

    value_member, color_member = payload.members
    assert value_member.name == f"value_{suffix}"
    assert isinstance(value_member.member_type, NamedType)
    assert value_member.member_type.name == "T"
    assert color_member.name == f"color_{suffix}"
    assert isinstance(color_member.member_type, VectorType)
    assert [attribute.name for attribute in color_member.attributes] == ["COLOR0"]

    function = ast.functions[0]
    assert function.name == f"choose_{suffix}"
    assert [parameter.name for parameter in function.generic_params] == ["U"]
    assert [
        constraint.name for constraint in function.generic_params[0].constraints
    ] == [f"Scalar_{suffix}", f"Packable_{suffix}"]
    assert isinstance(function.return_type, NamedType)
    assert function.return_type.name == "U"
    assert [parameter.name for parameter in function.parameters] == [
        f"left_{suffix}",
        f"right_{suffix}",
    ]
    assert [parameter.param_type.name for parameter in function.parameters] == [
        "U",
        "U",
    ]

    enum = next(node for node in ast.structs if isinstance(node, EnumNode))
    assert enum.name == f"Mode_{suffix}"
    assert isinstance(enum.underlying_type, PrimitiveType)
    assert enum.underlying_type.name == "uint"

    read_variant, tuple_variant, struct_variant = enum.variants
    assert read_variant.name == f"Read_{suffix}"
    assert_literal(read_variant.value, enum_value)

    assert tuple_variant.name == f"Tuple_{suffix}"
    assert [payload_type.name for payload_type in tuple_variant.data] == [
        f"Payload_{suffix}",
        f"Error_{suffix}",
    ]

    assert struct_variant.name == f"Struct_{suffix}"
    struct_fields = dict(struct_variant.data)
    assert list(struct_fields) == [f"color_{suffix}", f"count_{suffix}"]
    assert isinstance(struct_fields[f"color_{suffix}"], VectorType)
    assert isinstance(struct_fields[f"count_{suffix}"], PrimitiveType)
    assert struct_fields[f"count_{suffix}"].name == "int"


@settings(max_examples=20, deadline=None)
@given(suffix=IDENTIFIER_SUFFIXES)
def test_generated_trait_declarations_preserve_marked_struct_contract(suffix):
    code = f"""
    trait Fetcher_{suffix}<T: Scalar_{suffix}> {{
        fn fetch_{suffix}(self, index_{suffix}: int) -> T;
    }}

    shader TraitDeclarationContracts_{suffix} {{
    }}
    """

    ast = parse_code(code)
    trait = ast.structs[0]

    assert isinstance(trait, StructNode)
    assert trait.name == f"Fetcher_{suffix}"
    assert trait.is_trait is True
    assert [parameter.name for parameter in trait.generic_params] == ["T"]
    assert trait.generic_params[0].constraints[0].name == f"Scalar_{suffix}"

    method = trait.members[0]
    assert isinstance(method, FunctionNode)
    assert method.name == f"fetch_{suffix}"
    assert isinstance(method.return_type, NamedType)
    assert method.return_type.name == "T"
    assert [parameter.name for parameter in method.parameters] == [
        "self",
        f"index_{suffix}",
    ]
    assert isinstance(method.parameters[0].param_type, NamedType)
    assert method.parameters[0].param_type.name == "Self"
    assert isinstance(method.parameters[1].param_type, PrimitiveType)
    assert method.parameters[1].param_type.name == "int"
