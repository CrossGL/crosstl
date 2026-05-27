from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayType,
    LiteralNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)
RESOURCE_TYPES = (
    ("Texture2D<float4>", "Texture2D", "float4"),
    ("RWTexture2D<uint>", "RWTexture2D", "uint"),
    ("StructuredBuffer<float>", "StructuredBuffer", "float"),
    ("RWStructuredBuffer<int>", "RWStructuredBuffer", "int"),
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def assert_named_type(type_node, name):
    assert isinstance(type_node, NamedType)
    assert type_node.name == name


def assert_literal_arg(type_node, index, value):
    argument = type_node.generic_args[index]
    assert isinstance(argument, LiteralNode)
    assert argument.value == value


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    input_control_points=st.integers(min_value=1, max_value=32),
    output_control_points=st.integers(min_value=1, max_value=32),
)
def test_generated_patch_and_stream_generic_types_preserve_literal_arguments(
    suffix,
    input_control_points,
    output_control_points,
):
    code = f"""
    shader GenericTypeIR_{suffix} {{
        struct PatchPayload_{suffix} {{
            vec4 position;
        }};

        void consume(
            InputPatch<PatchPayload_{suffix}, {input_control_points}> inputPatch,
            const OutputPatch<PatchPayload_{suffix}, {output_control_points}> outputPatch,
            inout TriangleStream<PatchPayload_{suffix}> stream
        ) {{
        }}
    }}
    """

    ast = parse_code(code)
    input_patch, output_patch, stream = ast.functions[0].parameters

    assert input_patch.qualifiers == []
    assert_named_type(input_patch.param_type, "InputPatch")
    assert_named_type(input_patch.param_type.generic_args[0], f"PatchPayload_{suffix}")
    assert_literal_arg(input_patch.param_type, 1, input_control_points)

    assert output_patch.qualifiers == ["const"]
    assert_named_type(output_patch.param_type, "OutputPatch")
    assert_named_type(output_patch.param_type.generic_args[0], f"PatchPayload_{suffix}")
    assert_literal_arg(output_patch.param_type, 1, output_control_points)

    assert stream.qualifiers == ["inout"]
    assert_named_type(stream.param_type, "TriangleStream")
    assert_named_type(stream.param_type.generic_args[0], f"PatchPayload_{suffix}")


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    array_size=st.integers(min_value=1, max_value=64),
)
def test_generated_pointer_reference_and_array_types_preserve_wrappers(
    suffix,
    array_size,
):
    code = f"""
    shader TypeWrapperIR_{suffix} {{
        struct Payload_{suffix} {{
            float value;
        }};

        void consume(
            threadgroup Payload_{suffix}* payloadPointer,
            constant Payload_{suffix}& payloadReference,
            device Payload_{suffix}& mut mutablePayloadReference,
            device float values[{array_size}]
        ) {{
        }}
    }}
    """

    ast = parse_code(code)
    (
        payload_pointer,
        payload_reference,
        mutable_payload_reference,
        values,
    ) = ast.functions[0].parameters

    assert payload_pointer.qualifiers == ["threadgroup"]
    assert isinstance(payload_pointer.param_type, PointerType)
    assert_named_type(payload_pointer.param_type.pointee_type, f"Payload_{suffix}")

    assert payload_reference.qualifiers == ["constant"]
    assert isinstance(payload_reference.param_type, ReferenceType)
    assert payload_reference.param_type.is_mutable is False
    assert_named_type(payload_reference.param_type.referenced_type, f"Payload_{suffix}")

    assert mutable_payload_reference.qualifiers == ["device"]
    assert isinstance(mutable_payload_reference.param_type, ReferenceType)
    assert mutable_payload_reference.param_type.is_mutable is True
    assert_named_type(
        mutable_payload_reference.param_type.referenced_type,
        f"Payload_{suffix}",
    )

    assert values.qualifiers == ["device"]
    assert isinstance(values.param_type, ArrayType)
    assert isinstance(values.param_type.element_type, PrimitiveType)
    assert values.param_type.element_type.name == "float"
    assert isinstance(values.param_type.size, LiteralNode)
    assert values.param_type.size.value == array_size


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    array_size=st.integers(min_value=1, max_value=16),
    resource_case=st.sampled_from(RESOURCE_TYPES),
)
def test_generated_typed_resource_arrays_preserve_named_generic_element_ir(
    suffix,
    array_size,
    resource_case,
):
    source_type, expected_name, expected_argument = resource_case
    code = f"""
    shader ResourceTypeIR_{suffix} {{
        {source_type} resources_{suffix}[{array_size}];
    }}
    """

    ast = parse_code(code)
    resource = ast.global_variables[0]

    assert resource.name == f"resources_{suffix}"
    assert isinstance(resource.var_type, ArrayType)
    assert isinstance(resource.var_type.size, LiteralNode)
    assert resource.var_type.size.value == array_size

    resource_type = resource.var_type.element_type
    assert_named_type(resource_type, expected_name)
    assert len(resource_type.generic_args) == 1
    generic_arg = resource_type.generic_args[0]
    if expected_argument == "float4":
        assert_named_type(generic_arg, "float4")
    else:
        assert isinstance(generic_arg, PrimitiveType)
        assert generic_arg.name == expected_argument
