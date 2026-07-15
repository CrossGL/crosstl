import pytest

from crosstl.translator.arithmetic_conversions import (
    ArithmeticConversion,
    ArithmeticScalarKind,
    ArithmeticType,
    UnrepresentableArithmeticConversion,
    arithmetic_type,
    promoted_integer_type_name,
    resolve_arithmetic_conversion,
)

INTEGER_WIDTHS = frozenset({32, 64})
FLOATING_WIDTHS = frozenset({32, 64})

BOOLEAN = ArithmeticScalarKind.BOOLEAN
SIGNED = ArithmeticScalarKind.SIGNED_INTEGER
UNSIGNED = ArithmeticScalarKind.UNSIGNED_INTEGER
FLOATING = ArithmeticScalarKind.FLOATING


def _operand(source_type, target_type, kind, bits, lanes=1):
    return ArithmeticType(source_type, target_type, kind, bits, lanes)


BOOL = _operand("bool", "bool", BOOLEAN, 1)
BVEC3 = _operand("bool3", "bvec3", BOOLEAN, 1, 3)
BVEC4 = _operand("bool4", "bvec4", BOOLEAN, 1, 4)
S8 = _operand("int8_t", "int", SIGNED, 8)
U8 = _operand("uint8_t", "uint", UNSIGNED, 8)
S16 = _operand("int16_t", "int", SIGNED, 16)
U16 = _operand("uint16_t", "uint", UNSIGNED, 16)
S32 = _operand("int32_t", "int", SIGNED, 32)
U32 = _operand("uint32_t", "uint", UNSIGNED, 32)
S64 = _operand("int64_t", "int64_t", SIGNED, 64)
U64 = _operand("uint64_t", "uint64_t", UNSIGNED, 64)
F32 = _operand("float", "float", FLOATING, 32)
F64 = _operand("double", "double", FLOATING, 64)

S8V2 = _operand("int8_t2", "ivec2", SIGNED, 8, 2)
U8V4 = _operand("uint8_t4", "uvec4", UNSIGNED, 8, 4)
S16V2 = _operand("int16_t2", "ivec2", SIGNED, 16, 2)
U16V3 = _operand("uint16_t3", "uvec3", UNSIGNED, 16, 3)
S32V2 = _operand("int2", "ivec2", SIGNED, 32, 2)
U32V3 = _operand("uint3", "uvec3", UNSIGNED, 32, 3)
U64V4 = _operand("uint64_t4", "u64vec4", UNSIGNED, 64, 4)
F64V2 = _operand("double2", "dvec2", FLOATING, 64, 2)


def _resolve(
    left,
    right,
    operator,
    *,
    integer_widths=INTEGER_WIDTHS,
    floating_widths=FLOATING_WIDTHS,
):
    return resolve_arithmetic_conversion(
        left,
        right,
        operator,
        supported_integer_widths=integer_widths,
        supported_floating_widths=floating_widths,
    )


def _assert_unrepresentable(
    left,
    right,
    operator,
    reason,
    attempted_common_type,
    *,
    integer_widths=INTEGER_WIDTHS,
    floating_widths=FLOATING_WIDTHS,
):
    with pytest.raises(UnrepresentableArithmeticConversion) as raised:
        _resolve(
            left,
            right,
            operator,
            integer_widths=integer_widths,
            floating_widths=floating_widths,
        )

    assert raised.value.reason == reason
    assert raised.value.attempted_common_type == attempted_common_type
    assert raised.value.args == (reason,)


@pytest.mark.parametrize(
    ("source_type", "target_type", "expected", "promoted_type"),
    [
        (
            "const std::uint8_t &",
            "uint",
            _operand("const std::uint8_t &", "uint", UNSIGNED, 8),
            "int",
        ),
        (
            "volatile metal::packed_short3",
            "ivec3",
            _operand("volatile metal::packed_short3", "ivec3", SIGNED, 16, 3),
            "ivec3",
        ),
        (
            "thread simd_uint4",
            "uvec4",
            _operand("thread simd_uint4", "uvec4", UNSIGNED, 32, 4),
            "uvec4",
        ),
        (
            "constant metal::vector<unsigned short, 2>",
            "uvec2",
            _operand(
                "constant metal::vector<unsigned short, 2>",
                "uvec2",
                UNSIGNED,
                16,
                2,
            ),
            "ivec2",
        ),
    ],
    ids=("qualified-namespace", "packed-vector", "simd-vector", "generic-vector"),
)
def test_source_spelling_normalization_preserves_integer_semantics(
    source_type, target_type, expected, promoted_type
):
    operand = arithmetic_type(source_type, target_type)

    assert operand == expected
    assert promoted_integer_type_name(operand) == promoted_type


@pytest.mark.parametrize(
    ("operand", "expected"),
    [
        (S8, "int"),
        (U8, "int"),
        (S16, "int"),
        (U16, "int"),
        (S32, "int"),
        (U32, "uint"),
        (S64, "int64_t"),
        (U64, "uint64_t"),
    ],
    ids=("s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64"),
)
def test_integer_promotions_cover_signedness_and_width(operand, expected):
    assert promoted_integer_type_name(operand) == expected


@pytest.mark.parametrize(
    ("left", "right", "common_type"),
    [
        (S8, U8, "int"),
        (S16, U16, "int"),
        (S32, U32, "uint"),
        (S64, U32, "int64_t"),
        (S32, U64, "uint64_t"),
        (S64, U64, "uint64_t"),
    ],
    ids=(
        "narrow-8",
        "narrow-16",
        "equal-rank-32",
        "wider-signed",
        "wider-unsigned",
        "equal-rank-64",
    ),
)
def test_usual_integer_conversions_select_the_representable_common_type(
    left, right, common_type
):
    assert _resolve(left, right, "+") == ArithmeticConversion(
        common_type,
        common_type,
        common_type,
        common_type,
    )


@pytest.mark.parametrize(
    ("left", "right", "operator", "expected"),
    [
        (BOOL, BOOL, "*", ArithmeticConversion("int", "int", "int", "int")),
        (BOOL, BOOL, ">>", ArithmeticConversion("int", "int", "int", "int")),
        (
            BVEC3,
            BOOL,
            "+",
            ArithmeticConversion("ivec3", "int", "ivec3", "ivec3"),
        ),
        (
            BVEC3,
            BOOL,
            "<<",
            ArithmeticConversion("ivec3", "int", "ivec3", "ivec3"),
        ),
    ],
    ids=("scalar-arithmetic", "scalar-shift", "vector-arithmetic", "vector-shift"),
)
def test_boolean_operands_promote_to_integer_intermediates(
    left, right, operator, expected
):
    assert _resolve(left, right, operator) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (F32, F64, ArithmeticConversion("double", "double", "double", "double")),
        (S64, F32, ArithmeticConversion("float", "float", "float", "float")),
    ],
    ids=("higher-floating-rank", "floating-dominates-integer"),
)
def test_floating_rank_controls_the_common_type(left, right, expected):
    assert _resolve(left, right, "/") == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            S32,
            U32V3,
            ArithmeticConversion("uint", "uvec3", "uvec3", "uvec3"),
        ),
        (
            U32V3,
            S32,
            ArithmeticConversion("uvec3", "uint", "uvec3", "uvec3"),
        ),
    ],
    ids=("scalar-left", "scalar-right"),
)
def test_arithmetic_broadcast_keeps_scalar_operand_shape(left, right, expected):
    assert _resolve(left, right, "-") == expected


@pytest.mark.parametrize(
    ("left", "right", "operator", "expected"),
    [
        (
            S32,
            U32V3,
            "<",
            ArithmeticConversion("uvec3", "uvec3", "uvec3", "bvec3"),
        ),
        (
            U32V3,
            S32,
            "!=",
            ArithmeticConversion("uvec3", "uvec3", "uvec3", "bvec3"),
        ),
        (
            F32,
            F64V2,
            "?:",
            ArithmeticConversion("dvec2", "dvec2", "dvec2", "dvec2"),
        ),
        (
            BOOL,
            BVEC4,
            "?:",
            ArithmeticConversion("bvec4", "bvec4", "bvec4", "bvec4"),
        ),
    ],
    ids=("relational-scalar-left", "equality-scalar-right", "conditional", "bool"),
)
def test_relational_and_conditional_operators_require_common_shapes(
    left, right, operator, expected
):
    assert _resolve(left, right, operator) == expected


@pytest.mark.parametrize(
    ("left", "right", "operator", "expected"),
    [
        (
            U32,
            S64,
            "<<",
            ArithmeticConversion("uint", "int64_t", "uint", "uint"),
        ),
        (
            U16V3,
            U64,
            ">>",
            ArithmeticConversion("ivec3", "uint64_t", "ivec3", "ivec3"),
        ),
        (
            U64V4,
            U8,
            "<<",
            ArithmeticConversion("u64vec4", "int", "u64vec4", "u64vec4"),
        ),
    ],
    ids=("scalar-ranks", "promoted-vector-left", "wide-vector-left"),
)
def test_shift_operands_are_promoted_independently(left, right, operator, expected):
    assert _resolve(left, right, operator) == expected


@pytest.mark.parametrize(
    ("left", "right", "operator", "reason", "attempted_common_type"),
    [
        (S32V2, U32V3, "+", "vector-width-mismatch", "uint"),
        (S16V2, U8V4, "<<", "vector-width-mismatch", "ivec2"),
        (
            S32,
            U8V4,
            "<<",
            "scalar-left-vector-shift-unsupported",
            "int",
        ),
    ],
    ids=("arithmetic-width", "shift-width", "scalar-left-shift"),
)
def test_incompatible_operand_shapes_report_structured_failures(
    left, right, operator, reason, attempted_common_type
):
    _assert_unrepresentable(
        left,
        right,
        operator,
        reason,
        attempted_common_type,
    )


@pytest.mark.parametrize(
    ("left", "right", "operator", "attempted_common_type"),
    [
        (F32, S32, "%", "float"),
        (S32, F64, "&", "double"),
        (F32, S32, "|", "float"),
        (S32, F64, "^", "double"),
        (F32, S32, "<<", None),
        (S32, F64, ">>", "int"),
    ],
    ids=("remainder", "and", "or", "xor", "float-left-shift", "float-right-shift"),
)
def test_integral_only_operators_reject_floating_operands(
    left, right, operator, attempted_common_type
):
    _assert_unrepresentable(
        left,
        right,
        operator,
        "integer-operands-required",
        attempted_common_type,
    )


@pytest.mark.parametrize(
    (
        "left",
        "right",
        "integer_widths",
        "floating_widths",
        "reason",
        "attempted_common_type",
    ),
    [
        (
            S64,
            S32,
            frozenset({32}),
            FLOATING_WIDTHS,
            "target-integer-width-unsupported",
            "int64_t",
        ),
        (
            U8,
            S8,
            frozenset({64}),
            FLOATING_WIDTHS,
            "target-integer-width-unsupported",
            "int",
        ),
        (
            BOOL,
            BOOL,
            frozenset({64}),
            FLOATING_WIDTHS,
            "target-integer-width-unsupported",
            "int",
        ),
        (
            F64,
            F32,
            INTEGER_WIDTHS,
            frozenset({32}),
            "target-floating-width-unsupported",
            "double",
        ),
    ],
    ids=("integer-64", "promoted-integer-32", "bool-promotion-32", "floating-64"),
)
def test_unsupported_target_widths_report_the_attempted_common_type(
    left,
    right,
    integer_widths,
    floating_widths,
    reason,
    attempted_common_type,
):
    _assert_unrepresentable(
        left,
        right,
        "+",
        reason,
        attempted_common_type,
        integer_widths=integer_widths,
        floating_widths=floating_widths,
    )


@pytest.mark.parametrize("operator", ["**", "&&", "[]"])
def test_unrecognized_operators_decline_to_produce_a_plan(operator):
    assert _resolve(S32, U32, operator) is None
