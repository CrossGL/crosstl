from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import (
    ArrayType,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    ExecutionModel,
    FunctionCallNode,
    FunctionNode,
    FunctionType,
    GenericParameterNode,
    GenericType,
    IdentifierNode,
    LambdaNode,
    LiteralNode,
    MatrixType,
    NamedType,
    ParameterNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    ShaderNode,
    VariableNode,
    VectorType,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _float_type():
    return PrimitiveType("float")


def _bool_type():
    return PrimitiveType("bool")


def _id(name):
    return IdentifierNode(name)


def _int(value):
    return LiteralNode(value, _int_type())


def _float(value):
    return LiteralNode(float(value), _float_type())


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    payload_size=st.integers(min_value=1, max_value=16),
    weight=st.integers(min_value=0, max_value=255),
)
def test_generated_callable_expression_nodes_preserve_structural_children(
    suffix,
    payload_size,
    weight,
):
    result_type = NamedType(
        f"Result_{suffix}",
        generic_args=[NamedType(f"Payload_{suffix}"), _int(payload_size)],
    )
    constructor = ConstructorNode(
        result_type,
        [_id(f"value_{suffix}")],
        named_arguments={
            "weight": _float(weight),
            "enabled": _id(f"enabled_{suffix}"),
        },
    )
    cast = CastNode(
        _id(f"raw_{suffix}"),
        VectorType(_float_type(), 4),
    )
    lambda_body = BlockNode(
        [
            ReturnNode(
                BinaryOpNode(
                    _id(f"lhs_{suffix}"),
                    "+",
                    _id(f"rhs_{suffix}"),
                )
            )
        ]
    )
    lambda_node = LambdaNode(
        [
            ParameterNode(f"lhs_{suffix}", _float_type()),
            ParameterNode(f"rhs_{suffix}", _float_type()),
        ],
        lambda_body,
        captures=[f"captured_{suffix}"],
    )
    call = FunctionCallNode(
        _id(f"build_{suffix}"),
        [constructor, cast, lambda_node],
        generic_args=[
            NamedType(f"Payload_{suffix}"),
            ArrayType(_float_type(), _int(payload_size)),
        ],
    )

    _assert_walks_once_with(
        call,
        [
            call.function,
            constructor,
            constructor.constructor_type,
            *constructor.constructor_type.generic_args,
            constructor.arguments[0],
            *constructor.named_arguments.values(),
            cast,
            cast.expression,
            cast.target_type,
            cast.target_type.element_type,
            lambda_node,
            *lambda_node.parameters,
            lambda_node.parameters[0].param_type,
            lambda_node.parameters[1].param_type,
            lambda_body,
            lambda_body.statements[0],
            lambda_body.statements[0].value,
            lambda_body.statements[0].value.left,
            lambda_body.statements[0].value.right,
            *call.generic_args,
            call.generic_args[1].element_type,
            call.generic_args[1].size,
        ],
    )

    call.bind_parent_links()

    assert call.function.parent is call
    assert constructor.parent is call
    assert constructor.constructor_type.parent is constructor
    assert all(
        generic_arg.parent is constructor.constructor_type
        for generic_arg in constructor.constructor_type.generic_args
    )
    assert constructor.arguments[0].parent is constructor
    assert all(
        value.parent is constructor for value in constructor.named_arguments.values()
    )
    assert cast.parent is call
    assert cast.expression.parent is cast
    assert cast.target_type.parent is cast
    assert cast.target_type.element_type.parent is cast.target_type
    assert lambda_node.parent is call
    assert all(parameter.parent is lambda_node for parameter in lambda_node.parameters)
    assert lambda_node.body.parent is lambda_node
    assert lambda_body.statements[0].parent is lambda_body
    assert lambda_body.statements[0].value.parent is lambda_body.statements[0]
    assert all(generic_arg.parent is call for generic_arg in call.generic_args)
    assert call.generic_args[1].element_type.parent is call.generic_args[1]
    assert call.generic_args[1].size.parent is call.generic_args[1]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    rows=st.integers(min_value=2, max_value=4),
    cols=st.integers(min_value=2, max_value=4),
    array_size=st.integers(min_value=1, max_value=8),
)
def test_generated_type_nodes_preserve_nested_callable_type_structure(
    suffix,
    rows,
    cols,
    array_size,
):
    return_type_parameter = GenericType(
        f"T_{suffix}",
        constraints=[NamedType("Copy")],
    )
    pointer_type_parameter = GenericType(
        f"TBuffer_{suffix}",
        constraints=[NamedType("Copy")],
    )
    callback_type_parameter = GenericType(
        f"TCallback_{suffix}",
        constraints=[NamedType("Copy")],
    )
    return_type = NamedType(
        f"Output_{suffix}",
        generic_args=[
            return_type_parameter,
            MatrixType(_float_type(), rows, cols),
        ],
    )
    param_types = [
        PointerType(ArrayType(pointer_type_parameter, _int(array_size))),
        ReferenceType(VectorType(_float_type(), 4), is_mutable=True),
        FunctionType(_bool_type(), [callback_type_parameter, _int_type()]),
    ]
    function_type = FunctionType(return_type, param_types)

    _assert_walks_once_with(
        function_type,
        [
            return_type,
            return_type_parameter,
            return_type_parameter.constraints[0],
            return_type.generic_args[1],
            return_type.generic_args[1].element_type,
            *param_types,
            param_types[0].pointee_type,
            pointer_type_parameter,
            pointer_type_parameter.constraints[0],
            param_types[0].pointee_type.size,
            param_types[1].referenced_type,
            param_types[1].referenced_type.element_type,
            param_types[2].return_type,
            *param_types[2].param_types,
            callback_type_parameter.constraints[0],
        ],
    )

    function_type.bind_parent_links()

    assert return_type.parent is function_type
    assert return_type_parameter.parent is return_type
    assert return_type_parameter.constraints[0].parent is return_type_parameter
    assert return_type.generic_args[1].parent is return_type
    assert (
        return_type.generic_args[1].element_type.parent is return_type.generic_args[1]
    )
    assert param_types[0].parent is function_type
    assert param_types[0].pointee_type.parent is param_types[0]
    assert pointer_type_parameter.parent is param_types[0].pointee_type
    assert pointer_type_parameter.constraints[0].parent is pointer_type_parameter
    assert param_types[0].pointee_type.size.parent is param_types[0].pointee_type
    assert param_types[1].referenced_type.parent is param_types[1]
    assert param_types[2].return_type.parent is param_types[2]
    assert all(
        param_type.parent is param_types[2] for param_type in param_types[2].param_types
    )
    assert callback_type_parameter.constraints[0].parent is callback_type_parameter


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    default_value=st.integers(min_value=0, max_value=64),
)
def test_generated_function_declarations_bind_generic_parameters_and_defaults(
    suffix,
    default_value,
):
    generic_param = GenericParameterNode(
        f"T_{suffix}",
        constraints=[NamedType(f"Scalar_{suffix}"), NamedType(f"Packable_{suffix}")],
        default_type=NamedType(f"DefaultPayload_{suffix}"),
    )
    parameter = ParameterNode(
        f"value_{suffix}",
        NamedType(f"T_{suffix}"),
        default_value=_int(default_value),
        attributes=[AttributeNode("location", [_int(0)])],
    )
    local = VariableNode(
        f"local_{suffix}",
        NamedType(f"T_{suffix}"),
        initial_value=_id(f"value_{suffix}"),
    )
    body = BlockNode([local, ReturnNode(_id(f"value_{suffix}"))])
    function = FunctionNode(
        f"identity_{suffix}",
        NamedType(f"T_{suffix}"),
        [parameter],
        body=body,
        generic_params=[generic_param],
        attributes=[AttributeNode("inline")],
    )
    shader = ShaderNode(
        "CallableDeclarationContract",
        ExecutionModel.GENERAL_PURPOSE,
        functions=[function],
    )

    _assert_walks_once_with(
        shader,
        [
            function,
            function.return_type,
            generic_param,
            *generic_param.constraints,
            generic_param.default_type,
            parameter,
            parameter.param_type,
            parameter.default_value,
            parameter.attributes[0],
            parameter.attributes[0].arguments[0],
            body,
            local,
            local.var_type,
            local.initial_value,
            body.statements[1],
            body.statements[1].value,
            function.attributes[0],
        ],
    )

    shader.bind_parent_links()

    assert function.parent is shader
    assert function.return_type.parent is function
    assert generic_param.parent is function
    assert all(
        constraint.parent is generic_param for constraint in generic_param.constraints
    )
    assert generic_param.default_type.parent is generic_param
    assert parameter.parent is function
    assert parameter.param_type.parent is parameter
    assert parameter.default_value.parent is parameter
    assert parameter.attributes[0].parent is parameter
    assert parameter.attributes[0].arguments[0].parent is parameter.attributes[0]
    assert body.parent is function
    assert local.parent is body
    assert local.var_type.parent is local
    assert local.initial_value.parent is local
    assert body.statements[1].parent is body
    assert body.statements[1].value.parent is body.statements[1]
    assert function.attributes[0].parent is function
