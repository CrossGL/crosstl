"""Exact bfloat16 vector lowering for HLSL.

DirectX has no bfloat16 type. A source bfloat16 scalar travels as the low
16 bits of a ``uint``; an N-lane bfloat16 vector travels as ``uintN`` with the
same payload in every lane. Lanes are widened to binary32 exactly by shifting
the payload, arithmetic runs in binary32, and every bfloat16 result is rounded
back to nearest-even once, which is what the scalar lowering does.

The mixin only adds lane-aware entry points; ``HLSLCodeGen`` keeps the scalar
lowering and calls into these methods when an operand is a vector. Anything
that a vector payload cannot express exactly raises
``DirectXBFloat16UnsupportedError`` instead of computing on raw payload bits.
"""

import re

_QUALIFIERS = re.compile(
    r"^(?:(?:const|constant|device|thread|threadgroup|volatile)\s+)+"
)
_SCALAR_NAMES = r"(?:bfloat|bfloat16|bfloat16_t)"
_VECTOR_SPELLINGS = (
    re.compile(
        r"(?:metal::)?(?:vec|vector)\s*<\s*" + _SCALAR_NAMES + r"\s*,\s*([234])\s*>"
    ),
    re.compile(r"vec([234])\s*<\s*" + _SCALAR_NAMES + r"\s*>"),
    # ``vecN<bfloat16_t>`` constructors reach codegen spelled ``bfloat16_tN``.
    re.compile(r"(?:bfloat16vec|bfloat16_t|bfloat)([234])"),
)
_NUMERIC_TYPE = re.compile(
    r"(?P<base>float|half|float16_t|min16float|min10float|double|int|uint|"
    r"int64_t|uint64_t|int16_t|uint16_t|min16int|min16uint|min12int|bool)"
    r"(?P<lanes>[234])?"
)
_FLOAT_BASES = frozenset({"float", "half", "float16_t", "min16float", "min10float"})
# Integer sources round once through the exact integer helpers.
_INTEGER_HELPERS = {
    "int": ("int", "from_int", ("from_uint",)),
    "int16_t": ("int", "from_int", ("from_uint",)),
    "min16int": ("int", "from_int", ("from_uint",)),
    "min12int": ("int", "from_int", ("from_uint",)),
    "uint": ("uint", "from_uint", ()),
    "uint16_t": ("uint", "from_uint", ()),
    "min16uint": ("uint", "from_uint", ()),
    "int64_t": ("int64_t", "from_int64", ("from_uint64",)),
    "uint64_t": ("uint64_t", "from_uint64", ()),
}
_VECTOR_HELPER_SOURCES = {
    "vector_from_float": ("float", "from_float"),
    "vector_from_int": ("int", "from_int"),
    "vector_from_uint": ("uint", "from_uint"),
    "vector_from_int64": ("int64_t", "from_int64"),
    "vector_from_uint64": ("uint64_t", "from_uint64"),
}
_LANE_NAMES = ("x", "y", "z", "w")
_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/"})
_COMPARISON_OPERATORS = frozenset({"==", "!=", "<", "<=", ">", ">="})
# Geometric builtins that only exist for vectors: (HLSL name, arity, result).
# "scalar" results are one bfloat16 value; "bfloat" results keep the lanes.
HLSL_BFLOAT16_VECTOR_BUILTIN_CONTRACTS = {
    "cross": ("cross", 2, "bfloat"),
    "distance": ("distance", 2, "scalar"),
    "dot": ("dot", 2, "scalar"),
    "length": ("length", 1, "scalar"),
    "normalize": ("normalize", 1, "bfloat"),
}


def hlsl_bfloat16_vector_helper_sources():
    """HLSL overloads that round each lane through the scalar helper.

    Their keys sort after every scalar helper key, so the scalar definitions
    they call are always emitted first.
    """

    sources = {}
    for key, (source_base, scalar_helper) in _VECTOR_HELPER_SOURCES.items():
        overloads = []
        for lanes in (2, 3, 4):
            calls = ", ".join(
                f"__crossgl_bfloat16_{scalar_helper}(value.{lane})"
                for lane in _LANE_NAMES[:lanes]
            )
            overloads.append(
                f"uint{lanes} __crossgl_bfloat16_{scalar_helper}"
                f"({source_base}{lanes} value) {{\n"
                f"    return uint{lanes}({calls});\n"
                "}\n"
            )
        sources[key] = "\n" + "".join(overloads)
    return sources


class HLSLBfloat16Vectors:
    """Lane-aware bfloat16 lowering mixed into ``HLSLCodeGen``."""

    def hlsl_bfloat16_vector_lanes(self, vtype):
        """Return N for a source N-lane bfloat16 vector type, else ``None``."""

        type_name = self.type_name_string(vtype)
        if not isinstance(type_name, str) or not type_name.strip():
            return None
        aliases = getattr(self, "hlsl_type_aliases", {})
        seen = set()
        normalized = _QUALIFIERS.sub("", type_name.strip())
        while normalized in aliases and normalized not in seen:
            seen.add(normalized)
            normalized = _QUALIFIERS.sub(
                "", str(self.type_name_string(aliases[normalized]) or "").strip()
            )
        for pattern in _VECTOR_SPELLINGS:
            match = pattern.fullmatch(normalized)
            if match is not None:
                return int(match.group(1))
        return None

    def hlsl_bfloat16_lanes(self, vtype):
        """1 for a bfloat16 scalar, N for an N-lane bfloat16 vector."""

        if self.is_hlsl_bfloat16_type(vtype):
            return 1
        return self.hlsl_bfloat16_vector_lanes(vtype)

    def hlsl_bfloat16_vector_payload_type(self, vtype):
        lanes = self.hlsl_bfloat16_vector_lanes(vtype)
        return None if lanes is None else f"uint{lanes}"

    def hlsl_bfloat16_decode(self, rendered, lanes):
        if lanes == 1:
            return self.hlsl_bfloat16_to_float_expression(rendered)
        return self.hlsl_bfloat16_vector_to_float_expression(rendered, lanes)

    def hlsl_bfloat16_encode(self, rendered, lanes):
        if lanes == 1:
            return self.hlsl_float_to_bfloat16_expression(rendered)
        self.require_hlsl_bfloat16_helper("from_float")
        self.require_hlsl_bfloat16_helper("vector_from_float")
        return f"__crossgl_bfloat16_from_float(float{lanes}({rendered}))"

    def hlsl_bfloat16_numeric_shape(self, vtype):
        """``(base, lanes)`` of a non-bfloat numeric scalar or vector type."""

        mapped = self.map_type(vtype) if vtype is not None else None
        if not isinstance(mapped, str):
            return None
        match = _NUMERIC_TYPE.fullmatch(mapped.strip())
        if match is None:
            return None
        return match.group("base"), int(match.group("lanes") or 1)

    def hlsl_bfloat16_vector_refusal(
        self, message, *, operation, source_type, reason, source_location
    ):
        return self.directx_bfloat16_unsupported(
            message,
            operation=operation,
            source_type=source_type,
            reason=reason,
            source_location=source_location,
        )

    def hlsl_bfloat16_encode_numeric(
        self, rendered, source_type, *, operation, source_location
    ):
        """Round a non-bfloat numeric value to a bfloat16 payload of equal lanes."""

        shape = self.hlsl_bfloat16_numeric_shape(source_type)
        if shape is None or shape[0] == "double":
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector conversion requires a 16/32/64-bit "
                "integer, bool or binary16/binary32 source, got "
                f"'{self.type_name_string(source_type)}'",
                operation=operation,
                source_type=source_type,
                reason="unsupported-conversion-source",
                source_location=source_location,
            )
        base, lanes = shape
        if lanes == 1:
            return self.hlsl_bfloat16_conversion_expression(
                rendered, "bfloat16", source_type, source_location=source_location
            )
        if base in _FLOAT_BASES or base == "bool":
            return self.hlsl_bfloat16_encode(rendered, lanes)
        cast_base, helper, prerequisites = _INTEGER_HELPERS[base]
        for prerequisite in prerequisites:
            self.require_hlsl_bfloat16_helper(prerequisite)
        self.require_hlsl_bfloat16_helper(helper)
        self.require_hlsl_bfloat16_helper(f"vector_{helper}")
        return f"__crossgl_bfloat16_{helper}({cast_base}{lanes}({rendered}))"

    def hlsl_bfloat16_vector_conversion_expression(
        self, rendered, expected_type, source_type, *, source_location=None
    ):
        """Contextual conversion where either side is a bfloat16 vector."""

        expected_lanes = self.hlsl_bfloat16_vector_lanes(expected_type)
        source_lanes = self.hlsl_bfloat16_vector_lanes(source_type)
        if expected_lanes is None and source_lanes is None:
            return None
        operation = "bfloat16 vector conversion"
        if expected_lanes is not None:
            if source_lanes is not None or self.is_hlsl_bfloat16_type(source_type):
                lanes = source_lanes or 1
                if lanes == expected_lanes:
                    return rendered
                if lanes == 1:
                    # A payload splat keeps every lane's bits unchanged.
                    return f"((uint{expected_lanes})({rendered}))"
            else:
                shape = self.hlsl_bfloat16_numeric_shape(source_type)
                if shape is not None and shape[1] == 1:
                    payload = self.hlsl_bfloat16_encode_numeric(
                        rendered,
                        source_type,
                        operation=operation,
                        source_location=source_location,
                    )
                    return f"((uint{expected_lanes})({payload}))"
                if shape is not None and shape[1] == expected_lanes:
                    return self.hlsl_bfloat16_encode_numeric(
                        rendered,
                        source_type,
                        operation=operation,
                        source_location=source_location,
                    )
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector conversion requires a scalar or a "
                f"{expected_lanes}-lane source, got "
                f"'{self.type_name_string(source_type)}'",
                operation=operation,
                source_type=source_type,
                reason="bfloat16-vector-lane-mismatch",
                source_location=source_location,
            )

        decoded = self.hlsl_bfloat16_decode(rendered, source_lanes)
        return self.hlsl_bfloat16_vector_decoded_conversion(
            decoded,
            expected_type,
            source_type,
            source_lanes,
            operation=operation,
            source_location=source_location,
        )

    def hlsl_bfloat16_vector_decoded_conversion(
        self,
        decoded,
        expected_type,
        source_type,
        lanes,
        *,
        operation,
        source_location,
    ):
        """Convert binary32 lanes widened from bfloat16 to ``expected_type``."""

        shape = self.hlsl_bfloat16_numeric_shape(expected_type)
        if shape is None or shape[1] != lanes:
            raise self.hlsl_bfloat16_vector_refusal(
                f"DirectX exact bfloat16 vector conversion requires a {lanes}-lane "
                "numeric target, got "
                f"'{self.type_name_string(expected_type)}'",
                operation=operation,
                source_type=source_type,
                reason="unsupported-conversion-target",
                source_location=source_location,
            )
        base = shape[0]
        if base == "float":
            return decoded
        if base == "bool":
            return f"({decoded} != (float{lanes})0.0)"
        if base == "float16_t":
            return self.hlsl_half_conversion_expression(
                decoded, lanes, ("float",), source_location=source_location
            )
        return f"{base}{lanes}({decoded})"

    def hlsl_bfloat16_vector_constructor_call(
        self, func_name, args, *, source_location=None
    ):
        """Lower ``bfloat16vecN(...)`` and conversions out of bfloat16 vectors."""

        lanes = self.hlsl_bfloat16_vector_lanes(func_name)
        if lanes is None:
            if len(args) != 1:
                return None
            source_type = self.hlsl_source_expression_type(args[0])
            source_lanes = self.hlsl_bfloat16_vector_lanes(source_type)
            if (
                source_lanes is None
                or self.hlsl_bfloat16_numeric_shape(func_name) is None
            ):
                return None
            # The conversion consumes the payload lane by lane, so a storage
            # lane-gather view may feed it directly.
            self.hlsl_bfloat16_vector_conversion_depth = (
                getattr(self, "hlsl_bfloat16_vector_conversion_depth", 0) + 1
            )
            try:
                rendered = self.generate_expression_with_expected(args[0], None)
            finally:
                self.hlsl_bfloat16_vector_conversion_depth -= 1
            return self.hlsl_bfloat16_vector_decoded_conversion(
                self.hlsl_bfloat16_decode(rendered, source_lanes),
                func_name,
                source_type,
                source_lanes,
                operation=f"{func_name} constructor",
                source_location=source_location,
            )

        operation = f"{func_name} constructor"
        if not args:
            return f"((uint{lanes})0u)"
        parts, total = [], 0
        for argument in args:
            argument_type = self.hlsl_source_expression_type(argument)
            rendered = self.generate_expression_with_expected(argument, None)
            argument_lanes = self.hlsl_bfloat16_lanes(argument_type)
            if argument_lanes is None:
                shape = self.hlsl_bfloat16_numeric_shape(argument_type)
                if shape is None:
                    raise self.hlsl_bfloat16_vector_refusal(
                        "DirectX exact bfloat16 vector construction requires "
                        "numeric scalar or vector arguments, got "
                        f"'{self.type_name_string(argument_type)}'",
                        operation=operation,
                        source_type=argument_type,
                        reason="unsupported-conversion-source",
                        source_location=source_location,
                    )
                argument_lanes = shape[1]
                rendered = self.hlsl_bfloat16_encode_numeric(
                    rendered,
                    argument_type,
                    operation=operation,
                    source_location=source_location,
                )
            parts.append(rendered)
            total += argument_lanes
        if len(parts) == 1 and total == 1:
            return f"((uint{lanes})({parts[0]}))"
        if total != lanes:
            raise self.hlsl_bfloat16_vector_refusal(
                f"DirectX bfloat16 vector construction of '{func_name}' needs "
                f"{lanes} lanes, got {total}",
                operation=operation,
                source_type=func_name,
                reason="invalid-constructor-arity",
                source_location=source_location,
            )
        if len(parts) == 1:
            return parts[0]
        return f"uint{lanes}({', '.join(parts)})"

    def hlsl_bfloat16_vector_binary_result_type(self, left_type, right_type):
        """Arithmetic result type when a bfloat16 operand meets a vector.

        Mirrors the scalar rule: bfloat16 with bfloat16 stays bfloat16, a
        floating operand decides the result, and an integer operand gives
        binary32. Lane counts follow HLSL scalar-to-vector promotion.
        """

        left_lanes = self.hlsl_bfloat16_lanes(left_type)
        right_lanes = self.hlsl_bfloat16_lanes(right_type)
        if left_lanes is None and right_lanes is None:
            return None
        if left_lanes is not None and right_lanes is not None:
            lanes = max(left_lanes, right_lanes)
            return f"bfloat16vec{lanes}" if lanes > 1 else None
        other_type = right_type if left_lanes is not None else left_type
        shape = self.hlsl_bfloat16_numeric_shape(other_type)
        if shape is None:
            return None
        lanes = max(left_lanes or right_lanes, shape[1])
        if lanes < 2:
            return None
        base = shape[0]
        if base in _FLOAT_BASES or base == "double":
            return f"{base}{lanes}"
        return f"float{lanes}"

    def hlsl_bfloat16_require_matching_lanes(
        self, left, right, *, operation, source_location
    ):
        """Refuse two vector operands of different widths.

        HLSL would silently truncate the wider operand; CrossGL, GLSL and Metal
        reject the expression, so it never reaches the payload arithmetic.
        """

        widths = []
        for lanes, operand_type in (left, right):
            if lanes is None:
                shape = self.hlsl_bfloat16_numeric_shape(operand_type)
                lanes = shape[1] if shape is not None else 1
            widths.append(lanes)
        if min(widths) > 1 and widths[0] != widths[1]:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector lowering requires operands of one "
                f"width, got {widths[0]} and {widths[1]} lanes",
                operation=operation,
                source_type=left[1] if left[0] is not None else right[1],
                reason="bfloat16-vector-lane-mismatch",
                source_location=source_location,
            )

    def hlsl_bfloat16_vector_binary_expression(self, expr, left, right, op):
        """Lower a binary operator with at least one bfloat16 vector operand."""

        left_type = self.hlsl_source_expression_type(getattr(expr, "left", None))
        right_type = self.hlsl_source_expression_type(getattr(expr, "right", None))
        left_lanes = self.hlsl_bfloat16_lanes(left_type)
        right_lanes = self.hlsl_bfloat16_lanes(right_type)
        if left_lanes is None and right_lanes is None:
            return None
        if self.hlsl_bfloat16_vector_binary_result_type(left_type, right_type) is None:
            return None
        source_type = left_type if left_lanes else right_type
        source_location = getattr(expr, "source_location", None)
        if op not in _ARITHMETIC_OPERATORS | _COMPARISON_OPERATORS:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector lowering does not support operator "
                f"'{op}'",
                operation=op,
                source_type=source_type,
                reason="unsupported-bfloat16-operator",
                source_location=source_location,
            )
        self.hlsl_bfloat16_require_matching_lanes(
            (left_lanes, left_type),
            (right_lanes, right_type),
            operation=op,
            source_location=source_location,
        )
        operands = []
        for rendered, lanes, operand_type in (
            (left, left_lanes, left_type),
            (right, right_lanes, right_type),
        ):
            if lanes is not None:
                operands.append(self.hlsl_bfloat16_decode(rendered, lanes))
                continue
            shape = self.hlsl_bfloat16_numeric_shape(operand_type)
            if shape is None or shape[0] not in _FLOAT_BASES | {"int", "uint"}:
                raise self.hlsl_bfloat16_vector_refusal(
                    "DirectX exact bfloat16 vector arithmetic requires binary32, "
                    "binary16 or 32-bit integer operands, got "
                    f"'{self.type_name_string(operand_type)}'",
                    operation=op,
                    source_type=operand_type,
                    reason="unsupported-bfloat16-operand",
                    source_location=source_location,
                )
            operands.append(rendered)
        operation = f"({operands[0]} {op} {operands[1]})"
        if op in _COMPARISON_OPERATORS:
            return operation
        result_lanes = self.hlsl_bfloat16_lanes(self.expression_result_type(expr))
        if result_lanes is None:
            return operation
        return self.hlsl_bfloat16_encode(operation, result_lanes)

    def hlsl_bfloat16_vector_unary_expression(self, expr, operand, op):
        operand_type = self.hlsl_source_expression_type(getattr(expr, "operand", None))
        lanes = self.hlsl_bfloat16_vector_lanes(operand_type)
        if lanes is None:
            return None
        if op == "+":
            return f"(uint{lanes}({operand}) & 0xffffu)"
        if op == "-":
            return f"((uint{lanes}({operand}) ^ 0x8000u) & 0xffffu)"
        if op == "!":
            decoded = self.hlsl_bfloat16_decode(operand, lanes)
            return f"!({decoded} != (float{lanes})0.0)"
        raise self.hlsl_bfloat16_vector_refusal(
            "DirectX exact bfloat16 vector lowering does not support unary "
            f"operator '{op}'",
            operation=op,
            source_type=operand_type,
            reason="unsupported-bfloat16-operator",
            source_location=getattr(expr, "source_location", None),
        )

    def hlsl_bfloat16_vector_compound_assignment(
        self, node, target, value, operator, *, target_type, lhs=None
    ):
        lanes = self.hlsl_bfloat16_vector_lanes(target_type)
        if lanes is None:
            return None
        source_location = getattr(node, "source_location", None)
        if operator == "=":
            return None
        if operator not in {"+=", "-=", "*=", "/="}:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector compound assignment does not "
                f"support '{operator}'",
                operation=operator,
                source_type=target_type,
                reason="unsupported-bfloat16-compound-assignment",
                source_location=source_location,
            )
        if lhs is None and (
            not self.hlsl_expression_is_repeatable(target)
            or self.hlsl_expression_has_observable_side_effects(target)
        ):
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector compound assignment requires a "
                "stable lvalue",
                operation=operator,
                source_type=target_type,
                reason="side-effecting-assignment-target",
                source_location=source_location,
            )
        lhs = lhs or self.generate_expression(target)
        rhs = self.generate_expression_with_expected(value, None)
        rhs_type = self.hlsl_source_expression_type(value)
        rhs_lanes = self.hlsl_bfloat16_lanes(rhs_type)
        if rhs_lanes is not None:
            rhs = self.hlsl_bfloat16_decode(rhs, rhs_lanes)
        else:
            shape = self.hlsl_bfloat16_numeric_shape(rhs_type)
            if shape is None or shape[0] not in _FLOAT_BASES | {"int", "uint"}:
                rhs_lanes = None
            else:
                rhs_lanes = shape[1]
        if rhs_lanes not in {1, lanes}:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector compound assignment requires a "
                f"scalar or {lanes}-lane right operand, got "
                f"'{self.type_name_string(rhs_type)}'",
                operation=operator,
                source_type=rhs_type,
                reason="unsupported-compound-assignment-source",
                source_location=source_location,
            )
        left = self.hlsl_bfloat16_decode(lhs, lanes)
        rounded = self.hlsl_bfloat16_encode(f"({left} {operator[0]} {rhs})", lanes)
        return f"{lhs} = {rounded}"

    def hlsl_bfloat16_vector_builtin_result_type(self, func_name, argument_types):
        """Result type of a builtin over bfloat16 vectors, or ``None``."""

        lanes = [self.hlsl_bfloat16_lanes(value) for value in argument_types]
        if not lanes or None in lanes or max(lanes) < 2 or len(set(lanes)) != 1:
            return None
        contract = HLSL_BFLOAT16_VECTOR_BUILTIN_CONTRACTS.get(
            func_name
        ) or self.HLSL_BFLOAT16_BUILTIN_CONTRACTS.get(func_name)
        if contract is None or len(argument_types) != contract[1]:
            return None
        if contract[2] == "bfloat":
            if func_name == "cross" and lanes[0] != 3:
                return None
            return argument_types[0]
        if contract[2] == "scalar":
            return "bfloat16"
        return f"{contract[2]}{lanes[0]}"

    def generate_hlsl_bfloat16_vector_builtin_call(
        self, func_name, args, argument_types, *, source_location=None
    ):
        lanes = [self.hlsl_bfloat16_lanes(value) for value in argument_types]
        if max((value or 0 for value in lanes), default=0) < 2:
            return None
        result_type = self.hlsl_bfloat16_vector_builtin_result_type(
            func_name, argument_types
        )
        if result_type is None:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector lowering cannot route bfloat16 "
                f"vectors through builtin '{func_name}' without a proven source "
                "contract; every argument must be a bfloat16 vector of one width",
                operation=func_name,
                source_type=next(value for value in argument_types if value),
                reason="unsupported-bfloat16-builtin",
                source_location=source_location,
            )
        contract = HLSL_BFLOAT16_VECTOR_BUILTIN_CONTRACTS.get(
            func_name
        ) or self.HLSL_BFLOAT16_BUILTIN_CONTRACTS.get(func_name)
        rendered = [
            self.hlsl_bfloat16_decode(self.generate_expression(argument), width)
            for argument, width in zip(args, lanes)
        ]
        call = f"{contract[0]}({', '.join(rendered)})"
        result_lanes = self.hlsl_bfloat16_lanes(result_type)
        if result_lanes is None:
            return call
        return self.hlsl_bfloat16_encode(call, result_lanes)

    def hlsl_bfloat16_vector_bitcast_expression(
        self, target_type, source_type, argument_code, *, operation, source_location
    ):
        """``as_type`` where either side is a bfloat16 vector.

        An N-lane bfloat16 vector has the bits of an N-lane uint16 vector, so
        the reinterpretation reuses the uint16 bitcast lowering and only moves
        payloads between ``uintN`` registers and ``uint16_tN`` values.
        """

        target_lanes = self.hlsl_bfloat16_lanes(target_type)
        source_lanes = self.hlsl_bfloat16_lanes(source_type)
        if max(target_lanes or 0, source_lanes or 0) < 2:
            return None
        if target_lanes is not None and target_lanes == source_lanes:
            return argument_code
        stand_in_source = source_type
        if source_lanes is not None:
            stand_in_source = _uint16_type(source_lanes)
            argument_code = f"{stand_in_source}(({argument_code}) & 0xffffu)"
        stand_in_target = (
            target_type if target_lanes is None else _uint16_type(target_lanes)
        )
        try:
            payload = self.hlsl_explicit_bitcast_expression(
                stand_in_target, stand_in_source, argument_code
            )
        except ValueError as error:
            raise self.hlsl_bfloat16_vector_refusal(
                f"DirectX exact bfloat16 vector bitcast is not expressible: {error}",
                operation=operation,
                source_type=source_type,
                reason="bitcast-width-or-type-mismatch",
                source_location=source_location,
            ) from error
        if payload is None:
            raise self.hlsl_bfloat16_vector_refusal(
                "DirectX exact bfloat16 vector bitcasts require a 16- or 32-bit "
                f"integer or floating payload, got "
                f"'{self.type_name_string(source_type)}' to "
                f"'{self.type_name_string(target_type)}'",
                operation=operation,
                source_type=source_type,
                reason="bitcast-width-or-type-mismatch",
                source_location=source_location,
            )
        self.requires_hlsl_native_16_bit_types = True
        if target_lanes is None:
            return payload
        payload_type = "uint" if target_lanes == 1 else f"uint{target_lanes}"
        return f"{payload_type}({payload})"


def _uint16_type(lanes):
    return "uint16_t" if lanes == 1 else f"uint16_t{lanes}"
