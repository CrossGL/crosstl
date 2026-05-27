from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import (
    ArrayType,
    FunctionType,
    GenericParameterNode,
    GenericType,
    LiteralNode,
    MatrixType,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    VectorType,
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _int_type():
    return PrimitiveType("int")


def _float_type():
    return PrimitiveType("float")


def _bool_type():
    return PrimitiveType("bool")


def _int(value):
    return LiteralNode(value, _int_type())


def _assert_walks_once_with(root, expected_nodes):
    walked = list(root.walk())

    assert len({id(node) for node in walked}) == len(walked)
    for node in expected_nodes:
        assert node in walked


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    array_size=st.integers(min_value=1, max_value=32),
    rows=st.integers(min_value=2, max_value=4),
    cols=st.integers(min_value=2, max_value=4),
)
def test_generated_composite_type_nodes_walk_and_bind_nested_type_children(
    suffix,
    array_size,
    rows,
    cols,
):
    payload_type = GenericType(
        f"TPayload_{suffix}",
        constraints=[NamedType(f"Copy_{suffix}")],
    )
    callback_type = GenericType(
        f"TCallback_{suffix}",
        constraints=[NamedType(f"Callable_{suffix}")],
    )
    matrix_type = MatrixType(_float_type(), rows, cols)
    array_type = ArrayType(payload_type, _int(array_size))
    pointer_type = PointerType(array_type, is_mutable=True)
    reference_type = ReferenceType(VectorType(_float_type(), 4), is_mutable=False)
    callback_signature = FunctionType(_bool_type(), [callback_type, _int_type()])
    result_type = NamedType(
        f"Result_{suffix}",
        generic_args=[
            pointer_type,
            reference_type,
            matrix_type,
            callback_signature,
        ],
    )

    _assert_walks_once_with(
        result_type,
        [
            pointer_type,
            array_type,
            payload_type,
            payload_type.constraints[0],
            array_type.size,
            array_type.size.literal_type,
            reference_type,
            reference_type.referenced_type,
            reference_type.referenced_type.element_type,
            matrix_type,
            matrix_type.element_type,
            callback_signature,
            callback_signature.return_type,
            *callback_signature.param_types,
            callback_type.constraints[0],
        ],
    )

    result_type.bind_parent_links()

    assert all(
        generic_arg.parent is result_type for generic_arg in result_type.generic_args
    )
    assert pointer_type.pointee_type.parent is pointer_type
    assert array_type.element_type.parent is array_type
    assert payload_type.constraints[0].parent is payload_type
    assert array_type.size.parent is array_type
    assert array_type.size.literal_type.parent is array_type.size
    assert reference_type.referenced_type.parent is reference_type
    assert reference_type.referenced_type.element_type.parent is (
        reference_type.referenced_type
    )
    assert matrix_type.element_type.parent is matrix_type
    assert callback_signature.return_type.parent is callback_signature
    assert all(
        param_type.parent is callback_signature
        for param_type in callback_signature.param_types
    )
    assert callback_type.constraints[0].parent is callback_type


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    default_suffix=IDENTIFIER_SUFFIXES,
)
def test_generated_generic_parameter_nodes_bind_constraints_and_defaults(
    suffix,
    default_suffix,
):
    scalar_constraint = NamedType(f"Scalar_{suffix}")
    packable_constraint = NamedType(
        f"Packable_{suffix}",
        generic_args=[NamedType(f"Lane_{suffix}")],
    )
    default_type = NamedType(
        f"DefaultPayload_{default_suffix}",
        generic_args=[VectorType(_float_type(), 4)],
    )
    generic_parameter = GenericParameterNode(
        f"T_{suffix}",
        constraints=[scalar_constraint, packable_constraint],
        default_type=default_type,
    )

    _assert_walks_once_with(
        generic_parameter,
        [
            scalar_constraint,
            packable_constraint,
            packable_constraint.generic_args[0],
            default_type,
            default_type.generic_args[0],
            default_type.generic_args[0].element_type,
        ],
    )

    generic_parameter.bind_parent_links()

    assert scalar_constraint.parent is generic_parameter
    assert packable_constraint.parent is generic_parameter
    assert packable_constraint.generic_args[0].parent is packable_constraint
    assert default_type.parent is generic_parameter
    assert default_type.generic_args[0].parent is default_type
    assert default_type.generic_args[0].element_type.parent is (
        default_type.generic_args[0]
    )


@settings(max_examples=25, deadline=None)
@given(size=st.one_of(st.none(), st.integers(min_value=1, max_value=32)))
def test_generated_array_type_omits_non_ast_dynamic_and_integer_sizes(size):
    array_type = ArrayType(_float_type(), size)

    walked = list(array_type.walk())

    assert array_type in walked
    assert array_type.element_type in walked
    assert len({id(node) for node in walked}) == len(walked)
    assert all(not isinstance(node, int) for node in walked)

    array_type.bind_parent_links()

    assert array_type.parent is None
    assert array_type.element_type.parent is array_type
