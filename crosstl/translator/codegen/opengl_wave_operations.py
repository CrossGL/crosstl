"""Wave operations in the GLSL generator: masked software subgroup execution, structured buffer members, wave indices and arguments, and unsigned shift intervals."""

from ..ast import (
    ArrayAccessNode,
    ExpressionStatementNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from ..validation import expression_debug_name
from ..wave_operations import (
    WAVE_BALLOT_BIT_OPERATIONS,
    wave_extended_result_type,
)
from .array_utils import split_array_type_suffix
from .glsl_affine_relations import related_local_interval
from .glsl_buffer_layout import (
    std430_array_stride,
    std430_value_type_info,
)
from .image_access_contracts import operation_argument_type_error
from .index_range_contracts import (
    INDEX_RANGE_ASSERTED,
    INDEX_RANGE_STATIC,
    IndexTargetProfile,
    IntegerRange,
    parse_index_range_assertions,
)
from .opengl_values import OpenGLStoragePointerError
from .pointer_reinterpret import scalar_storage_layout
from .scalar_return_bounds import scalar_return_interval
from .software_quad_reductions import SOFTWARE_QUAD_REDUCTIONS
from .workgroup_access_contracts import parse_workgroup_access_assertions


class OpenGLWaveOperationMixin:
    """Wave operation members of ``GLSLCodeGen``."""

    def generate_glsl_software_subgroup_masked_if(
        self,
        node,
        indent,
        *,
        additional_condition=None,
    ):
        plan = self.glsl_software_subgroup_masked_if_plans[id(node)]
        statements = self.glsl_software_subgroup_statement_list(node.if_body)
        operation_index = plan["statementIndex"]

        indent_str = "    " * indent
        inner_indent = "    " * (indent + 1)
        code = ""
        hoisted_index = plan.get("hoistedTargetDeclarationIndex")
        prefix = list(statements[:operation_index])
        if hoisted_index is not None:
            code += self.generate_statement(prefix[hoisted_index], indent)
            prefix = [
                statement
                for index, statement in enumerate(prefix)
                if index != hoisted_index
            ]

        reductions = []
        for index in plan.get("statementIndices", (operation_index,)):
            assignment = statements[index]
            if isinstance(assignment, ExpressionStatementNode):
                assignment = assignment.expression
            value = getattr(assignment, "value", getattr(assignment, "right", None))
            wrappers, inner = self.glsl_software_subgroup_conversion_chain(value)
            operation, arguments, operation_node = (
                self.glsl_software_subgroup_direct_operation(inner)
            )
            target = getattr(assignment, "target", getattr(assignment, "left", None))
            value_type = self.glsl_source_expression_type(arguments[0])
            mapped_value_type = (
                self.map_type(value_type) if value_type is not None else None
            )
            target_type = self.glsl_source_expression_type(target)
            mapped_target_type = (
                self.map_type(target_type) if target_type is not None else None
            )
            half_sum = (
                operation == "WaveActiveSum" and self.glsl_half_width(value_type) == 1
            )
            if half_sum and not plan.get("subgroupUniform"):
                # Pairwise binary16 partials round at every step, so a guard that
                # splits a subgroup has no established source participation.
                raise self.glsl_software_subgroup_error(
                    "OpenGL masked binary16 sums require an established source active-lane contract",
                    operation=operation,
                    reason="half-masked-participation-unproven",
                    source_location=getattr(operation_node, "source_location", None),
                )
            if mapped_value_type not in self.GLSL_SOFTWARE_SUBGROUP_VALUE_TYPES or (
                not wrappers and mapped_target_type != mapped_value_type
            ):
                raise self.glsl_software_subgroup_error(
                    "OpenGL masked software subgroup reduction requires a matching "
                    "32-bit numeric scalar payload and assignment target",
                    operation=operation,
                    reason="value-type-unsupported",
                    source_location=getattr(operation_node, "source_location", None),
                )
            reductions.append(
                {
                    "index": index,
                    "wrappers": wrappers,
                    "operation": operation,
                    "arguments": arguments,
                    "operation_node": operation_node,
                    "target": target,
                    "value_type": value_type,
                    "mapped_value_type": mapped_value_type,
                    "half_sum": half_sum,
                }
            )

        active_name = self.glsl_synthetic_local_identifier(
            "crossglSoftwareSubgroupActive"
        )
        for reduction in reductions:
            reduction["input"] = self.glsl_synthetic_local_identifier(
                "crossglSoftwareSubgroupInput"
            )
            reduction["result"] = self.glsl_synthetic_local_identifier(
                "crossglSoftwareSubgroupResult"
            )
        condition = self.generate_glsl_boolean_context(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )
        if additional_condition is not None:
            condition = f"({additional_condition} && ({condition}))"
        code += f"{indent_str}bool {active_name} = {condition};\n"
        previous = None
        for reduction in reductions:
            operation = reduction["operation"]
            mapped_value_type = reduction["mapped_value_type"]
            # -0.0 is the exact float sum identity: x + -0.0 == x for every x,
            # including every rounded binary16 partial, so inactive invocations
            # cannot turn an all -0.0 sum into +0.0.
            identity = {
                "WaveActiveSum": {
                    "float": "uintBitsToFloat(0x80000000u)",
                    "int": "0",
                    "uint": "0u",
                },
                "WaveActiveMin": {
                    "float": "uintBitsToFloat(0x7f800000u)",
                    "int": "2147483647",
                    "uint": "0xffffffffu",
                },
                "WaveActiveMax": {
                    "float": "uintBitsToFloat(0xff800000u)",
                    "int": "(-2147483647 - 1)",
                    "uint": "0u",
                },
            }[operation][mapped_value_type]
            code += (
                f"{indent_str}{mapped_value_type} {reduction['input']} = {identity};\n"
            )
            code += f"{indent_str}if ({active_name}) {{\n"
            if previous is None:
                code += self.generate_scoped_statement_body(prefix, indent + 1)
            else:
                # The previous assignment and the statements up to this
                # reduction keep their source order under the guard.
                code += self.generate_glsl_software_subgroup_masked_assignment(
                    previous,
                    statements[previous["index"] : reduction["index"]],
                    indent + 1,
                )
            payload = self.generate_expression_with_expected(
                reduction["arguments"][0], mapped_value_type
            )
            code += f"{inner_indent}{reduction['input']} = {payload};\n"
            code += f"{indent_str}}}\n"

            helper_type = (
                self.glsl_software_subgroup_payload_type(
                    operation, reduction["value_type"]
                )
                if reduction["half_sum"]
                else mapped_value_type
            )
            self.required_glsl_software_subgroup_helpers.add((operation, helper_type))
            helper = self.glsl_software_subgroup_helper_name(operation, helper_type)
            code += (
                f"{indent_str}{mapped_value_type} {reduction['result']} = "
                f"{helper}({reduction['input']});\n"
            )
            previous = reduction
        code += f"{indent_str}if ({active_name}) {{\n"
        code += self.generate_glsl_software_subgroup_masked_assignment(
            previous, statements[previous["index"] :], indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_glsl_software_subgroup_masked_assignment(
        self, reduction, statements, indent
    ):
        """Emit a predicated reduction's assignment and the statements after it.

        ``statements`` starts with the reduction's source assignment.
        """
        if reduction["wrappers"]:
            # Emit the source assignment, conversions included, with its
            # reduction replaced by the result every invocation computed.
            key = id(reduction["operation_node"])
            self.glsl_software_subgroup_result_overrides[key] = reduction["result"]
            try:
                return self.generate_scoped_statement_body(statements, indent)
            finally:
                self.glsl_software_subgroup_result_overrides.pop(key, None)
        target_text = self.generate_glsl_buffer_block_mutation_target(
            reduction["target"]
        )
        code = f"{'    ' * indent}{target_text} = {reduction['result']};\n"
        return code + self.generate_scoped_statement_body(statements[1:], indent)

    def validate_glsl_software_subgroup_function(
        self, operation, *, source_location=None
    ):
        current_function_name = self.current_glsl_source_function_name
        current_function_id = self.current_glsl_source_function_id
        approved_function_ids = self.glsl_software_subgroup_helper_function_ids
        approved_function = (
            current_function_id == self.glsl_software_subgroup_entry_function_id
            or current_function_id in approved_function_ids
        )
        if not approved_function:
            approved_function = any(
                getattr(specialization, "name", None) == current_function_name
                and getattr(
                    specialization,
                    "_glsl_resource_source_function_id",
                    None,
                )
                in approved_function_ids
                for specialization in (
                    self.glsl_resource_function_specializations.values()
                )
            )
        if not approved_function:
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup barriers require a statically uniform "
                "entry-point or approved helper call",
                operation=operation,
                reason="helper-call-not-uniform",
                source_location=source_location,
            )

    def generate_glsl_software_subgroup_operation(
        self,
        operation,
        arguments,
        *,
        source_location=None,
    ):
        if operation not in self.GLSL_SOFTWARE_SUBGROUP_OPERATIONS:
            raise self.glsl_software_subgroup_error(
                f"OpenGL software subgroup lowering does not support '{operation}'",
                operation=operation,
                reason="operation-unsupported",
                source_location=source_location,
            )
        self.validate_glsl_software_subgroup_function(
            operation, source_location=source_location
        )
        expected_arity = (
            0
            if operation in self.GLSL_SOFTWARE_SUBGROUP_BARRIERS
            else self.GLSL_WAVE_INTRINSIC_ARITIES[operation]
        )
        accepted_arities = self.GLSL_WAVE_INTRINSIC_ALTERNATE_ARITIES.get(
            operation, (expected_arity,)
        )
        if len(arguments) not in accepted_arities:
            raise self.glsl_software_subgroup_error(
                f"OpenGL software subgroup operation '{operation}' requires "
                f"{expected_arity} arguments",
                operation=operation,
                reason="invalid-argument-count",
                source_location=source_location,
            )
        if operation == "subgroupExecutionBarrier":
            return "barrier()"
        if operation == "subgroupBarrier":
            raise self.glsl_software_subgroup_error(
                "OpenGL software subgroup barriers over buffer, image and shared memory "
                "must be statements",
                operation=operation,
                reason="subgroup-memory-barrier-unsupported",
                source_location=source_location,
            )
        if operation == "WaveShuffleAndFillUp":
            diagnostic = self.glsl_wave_validate_shuffle_and_fill_up(
                operation, arguments
            )
            if diagnostic is not None:
                raise self.glsl_software_subgroup_error(
                    "OpenGL software shuffle-and-fill requires matching payload types and scalar integer delta/modulo arguments",
                    operation=operation,
                    reason="argument-type-unsupported",
                    source_location=source_location,
                )
            value_type = self.glsl_source_expression_type(arguments[0])
            mapped_type = self.map_type(value_type)
            layout = scalar_storage_layout(self.glsl_normalized_source_type(value_type))
            # Narrow scalars already occupy canonical 32-bit GLSL registers, so
            # moving them between lanes is lossless.
            valid_type = mapped_type in {"bool", "uvec2"} or (
                mapped_type in self.GLSL_SOFTWARE_SUBGROUP_VALUE_TYPES
                and layout is not None
                and layout.bit_width <= 32
            )
            if not valid_type:
                raise self.glsl_software_subgroup_error(
                    "OpenGL software shuffle-and-fill requires a 32-bit numeric scalar, bool, or two-word uint payload",
                    operation=operation,
                    reason="value-type-unsupported",
                    source_location=source_location,
                )
            self.required_glsl_software_subgroup_helpers.add((operation, mapped_type))
            helper = self.glsl_software_subgroup_helper_name(operation, mapped_type)
            value, fill = (
                self.generate_expression_with_expected(arg, mapped_type)
                for arg in arguments[:2]
            )
            delta = self.generate_expression(arguments[2])
            modulo = (
                self.generate_expression(arguments[3])
                if len(arguments) == 4
                else f"{self.software_subgroup_width}u"
            )
            return f"{helper}({value}, {fill}, uint({delta}), uint({modulo}))"
        value_type = self.glsl_source_expression_type(arguments[0])
        mapped_value_type = (
            self.map_type(value_type) if value_type is not None else None
        )
        vector_shuffle_type = (
            mapped_value_type
            if operation in self.GLSL_SOFTWARE_SUBGROUP_LANE_SHUFFLES
            and mapped_value_type == "uvec2"
            else None
        )
        is_vote = operation in self.GLSL_SOFTWARE_SUBGROUP_VOTES
        valid_type = (
            mapped_value_type == "bool"
            if is_vote
            else mapped_value_type in self.GLSL_SOFTWARE_SUBGROUP_VALUE_TYPES
            or vector_shuffle_type is not None
        )
        narrow_integer_scan = None
        if (
            operation == "WaveActiveProduct"
            or operation in self.GLSL_SOFTWARE_SUBGROUP_SCANS
            or operation in SOFTWARE_QUAD_REDUCTIONS
            or (operation == "WaveReadLaneAt" and vector_shuffle_type is None)
        ):
            layout = scalar_storage_layout(self.glsl_normalized_source_type(value_type))
            exact_width = layout is not None and layout.bit_width == 32
            if (
                operation == "WaveActiveProduct"
                and self.glsl_half_width(value_type) == 1
            ):
                # A binary32 product of two binary16 values is exact; the
                # helper rounds it to binary16 at every node.
                exact_width = True
            if not exact_width and layout is not None and layout.bit_width < 32:
                if operation == "WaveReadLaneAt":
                    # Lane reads move canonical 32-bit registers unchanged.
                    exact_width = True
                elif operation in self.GLSL_SOFTWARE_SUBGROUP_SCANS:
                    if layout.kind == "integer":
                        # Modular sums and products commute with truncation, so a
                        # 32-bit scan rewrapped to the source width is exact.
                        narrow_integer_scan = layout
                        exact_width = True
                    elif self.glsl_half_width(value_type) == 1:
                        exact_width = True
            valid_type = valid_type and exact_width
        if not valid_type:
            raise self.glsl_software_subgroup_error(
                f"OpenGL software subgroup operation '{operation}' requires "
                "scalar bool for votes, a 32-bit numeric scalar for arithmetic, "
                "or an exact two-component uint shuffle payload",
                operation=operation,
                reason="value-type-unsupported",
                source_location=source_location,
            )
        if operation in self.GLSL_SOFTWARE_SUBGROUP_LANE_SHUFFLES:
            delta_type = self.glsl_source_expression_type(arguments[1])
            mapped_delta_type = (
                self.map_type(delta_type) if delta_type is not None else None
            )
            if mapped_delta_type not in {"int", "uint"}:
                raise self.glsl_software_subgroup_error(
                    f"OpenGL software subgroup '{operation}' requires a 32-bit "
                    "integer lane argument",
                    operation=operation,
                    reason="delta-type-unsupported",
                    source_location=source_location,
                )
            if operation == "WaveReadLaneAt":
                layout = scalar_storage_layout(
                    self.glsl_normalized_source_type(delta_type)
                )
                if layout is None or layout.bit_width != 32:
                    raise self.glsl_software_subgroup_error(
                        "OpenGL software subgroup lane reads require a 32-bit integer index",
                        operation=operation,
                        reason="lane-index-type-unsupported",
                        source_location=source_location,
                    )

        if vector_shuffle_type is not None:
            self.required_glsl_software_subgroup_helpers.add((operation, "uint"))
        helper_type = self.glsl_software_subgroup_payload_type(operation, value_type)
        self.required_glsl_software_subgroup_helpers.add((operation, helper_type))
        helper = self.glsl_software_subgroup_helper_name(operation, helper_type)
        value = self.generate_expression_with_expected(arguments[0], mapped_value_type)
        if operation in self.GLSL_SOFTWARE_SUBGROUP_LANE_SHUFFLES:
            delta = self.generate_expression(arguments[1])
            return f"{helper}({value}, uint({delta}))"
        if narrow_integer_scan is not None:
            # bitfieldExtract sign-extends int and zero-extends uint results.
            return (
                f"bitfieldExtract({helper}({value}), 0, "
                f"{narrow_integer_scan.bit_width})"
            )
        return f"{helper}({value})"

    def glsl_floating_bitwise_wave_operation(self, operation, arguments):
        if operation not in self.GLSL_WAVE_BITWISE_VALUE_OPERATIONS:
            return None
        if len(arguments) != 1:
            return None

        value_expr = arguments[0]
        value_type = self.map_type(self.expression_result_type(value_expr))
        component_type = self.vector_component_type(value_type) or value_type
        if component_type != "float":
            return None

        integer_type = "uint"
        if self.is_vector_value_type(value_type):
            integer_type = f"uvec{value_type[-1]}"
        value = self.generate_expression(value_expr)
        integer_value = self.generate_glsl_explicit_bitcast(
            value_type, integer_type, value
        )
        reduced = f"{self.GLSL_WAVE_DIRECT_MAPPINGS[operation]}({integer_value})"
        return self.generate_glsl_explicit_bitcast(integer_type, value_type, reduced)

    def glsl_wave_result_type(self, operation, arguments):
        if operation in WAVE_BALLOT_BIT_OPERATIONS:
            return wave_extended_result_type(operation, None)
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return "uint"
        if operation in {
            "WaveIsFirstLane",
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveAllEqual",
        }:
            return "bool"
        if operation in {
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            return "uint"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uvec4"
        if arguments:
            return self.expression_result_type(arguments[0])
        return None

    def glsl_wave_type_diagnostic(self, operation, args, diagnostic_operation=None):
        diagnostic_operation = diagnostic_operation or operation
        if operation == "WaveShuffleAndFillUp":
            return self.glsl_wave_validate_shuffle_and_fill_up(
                diagnostic_operation, args
            )
        if operation in self.GLSL_WAVE_NUMERIC_OPERATIONS:
            return self.glsl_wave_validate_argument_kind(
                diagnostic_operation,
                args[0],
                "a numeric scalar or vector",
                "value",
                {"float", "double", "int", "uint"},
            )
        if operation in self.GLSL_WAVE_INTEGER_OR_BOOLEAN_OPERATIONS:
            return self.glsl_wave_validate_argument_kind(
                diagnostic_operation,
                args[0],
                "an integer or boolean scalar or vector",
                "value",
                {"int", "uint", "bool"},
            )
        if operation in self.GLSL_WAVE_BOOLEAN_OPERATIONS:
            return self.glsl_wave_validate_argument_type(
                diagnostic_operation,
                args[0],
                "a boolean scalar",
                "value",
                {"bool"},
                allow_vectors=False,
            )
        if operation in self.GLSL_WAVE_SCALAR_OR_VECTOR_OPERATIONS:
            return self.glsl_wave_validate_argument_type(
                diagnostic_operation,
                args[0],
                "a scalar or vector",
                "value",
                {"float", "double", "int", "uint", "bool"},
                allow_vectors=True,
            )
        if operation == "WaveMatch":
            return self.glsl_wave_validate_argument_type(
                diagnostic_operation,
                args[0],
                "a scalar or vector",
                "value",
                {"float", "double", "int", "uint", "bool"},
                allow_vectors=True,
            )
        if operation == "WaveMultiPrefixCountBits":
            diagnostic = self.glsl_wave_validate_argument_type(
                diagnostic_operation,
                args[0],
                "a boolean scalar",
                "value",
                {"bool"},
                allow_vectors=False,
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(diagnostic_operation, args[1])
        if operation in self.GLSL_WAVE_MULTI_PREFIX_NUMERIC_OPERATIONS:
            diagnostic = self.glsl_wave_validate_argument_kind(
                diagnostic_operation,
                args[0],
                "a numeric scalar or vector",
                "value",
                {"float", "double", "int", "uint"},
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(diagnostic_operation, args[1])
        if operation in self.GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS:
            diagnostic = self.glsl_wave_validate_argument_kind(
                diagnostic_operation,
                args[0],
                "an integer scalar or vector",
                "value",
                {"int", "uint"},
            )
            if diagnostic is not None:
                return diagnostic
            return self.glsl_wave_validate_mask_argument(diagnostic_operation, args[1])
        index_argument = self.GLSL_WAVE_LANE_INDEX_ARGUMENTS.get(operation)
        if index_argument is not None:
            return self.glsl_wave_validate_argument_type(
                diagnostic_operation,
                args[index_argument],
                "a scalar integer",
                "lane",
                {"int", "uint"},
                allow_vectors=False,
            )
        return None

    def glsl_wave_validate_shuffle_and_fill_up(self, operation, args):
        for argument_label, arg in (("value", args[0]), ("fill", args[1])):
            diagnostic = self.glsl_wave_validate_argument_type(
                operation,
                arg,
                "a scalar or vector",
                argument_label,
                {"float", "double", "int", "uint", "bool"},
                allow_vectors=True,
            )
            if diagnostic is not None:
                return diagnostic

        value_type = self.expression_result_type(args[0])
        fill_type = self.expression_result_type(args[1])
        mapped_value_type = (
            self.map_type(value_type) if value_type is not None else None
        )
        mapped_fill_type = self.map_type(fill_type) if fill_type is not None else None
        if (
            mapped_value_type is not None
            and mapped_fill_type is not None
            and mapped_value_type != mapped_fill_type
        ):
            return self.glsl_wave_diagnostic_expression(
                operation,
                (
                    "requires value and fill arguments with matching scalar/vector "
                    f"types: {expression_debug_name(args[0])} has type "
                    f"{mapped_value_type}, {expression_debug_name(args[1])} has "
                    f"type {mapped_fill_type}"
                ),
            )

        for index, argument_label in ((2, "delta"), (3, "modulo")):
            if index >= len(args):
                break
            diagnostic = self.glsl_wave_validate_argument_type(
                operation,
                args[index],
                "a scalar integer",
                argument_label,
                {"int", "uint"},
                allow_vectors=False,
            )
            if diagnostic is not None:
                return diagnostic
        return None

    def glsl_wave_validate_argument_kind(
        self, operation, arg, requirement, argument_label, allowed_kinds
    ):
        return self.glsl_wave_validate_argument_type(
            operation,
            arg,
            requirement,
            argument_label,
            allowed_kinds,
            allow_vectors=True,
        )

    def glsl_wave_validate_argument_type(
        self,
        operation,
        arg,
        requirement,
        argument_label,
        allowed_kinds,
        allow_vectors,
    ):
        result_type = self.expression_result_type(arg)
        if result_type is None:
            return None
        mapped_type = self.map_type(result_type)
        if self.is_vector_value_type(mapped_type):
            component_kind = self.vector_component_type(mapped_type)
            if allow_vectors and component_kind in allowed_kinds:
                return None
        elif mapped_type in allowed_kinds:
            return None

        return self.glsl_wave_diagnostic_expression(
            operation,
            (
                f"requires {requirement} {argument_label} argument: "
                f"{expression_debug_name(arg)} has type {mapped_type}"
            ),
        )

    def glsl_wave_validate_mask_argument(self, operation, arg):
        result_type = self.expression_result_type(arg)
        if result_type is None:
            return None
        mapped_type = self.map_type(result_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix or mapped_type != "uvec4":
            return self.glsl_wave_diagnostic_expression(
                operation,
                (
                    "requires a uvec4 partition mask argument: "
                    f"{expression_debug_name(arg)} has type {mapped_type}"
                ),
            )
        return None

    def glsl_wave_diagnostic_expression(self, operation, reason):
        return self.refuse_lowering(
            f"wave intrinsic: {operation} {reason}",
            self.glsl_wave_default_value(operation),
        )

    def glsl_wave_default_value(self, operation):
        expected_type = self.current_expression_expected_type
        if expected_type:
            return self.zero_value_expression(expected_type)
        if operation in {
            "WaveIsFirstLane",
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveAllEqual",
        }:
            return "false"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uvec4(0u)"
        return "0u"

    def generate_interpolation_call(self, func_name, args):
        expected_args = self.GLSL_INTERPOLATION_FUNCTIONS.get(func_name)
        if expected_args is None:
            return None
        if len(args) != expected_args:
            noun = "argument" if expected_args == 1 else "arguments"
            raise OpenGLRefusal(
                f"OpenGL interpolation operation '{func_name}' requires "
                f"{expected_args} {noun}"
            )
        if self.current_stage_entry_type not in (None, "fragment"):
            raise OpenGLRefusal(
                f"OpenGL interpolation operation '{func_name}' is only valid "
                "in fragment stages"
            )

        if func_name == "interpolateAtSample":
            sample_type = self.expression_result_type(args[1])
            if sample_type is not None and not self.is_scalar_integer_type(sample_type):
                raise OpenGLRefusal(
                    operation_argument_type_error(
                        "OpenGL",
                        "interpolation",
                        func_name,
                        "a scalar integer",
                        "sample",
                        expression_debug_name(args[1]),
                        self.type_name_string(sample_type),
                    )
                )

        if func_name == "interpolateAtOffset":
            offset_type = self.expression_result_type(args[1])
            if offset_type is not None and self.map_type(offset_type) != "vec2":
                raise OpenGLRefusal(
                    operation_argument_type_error(
                        "OpenGL",
                        "interpolation",
                        func_name,
                        "a vec2 floating",
                        "offset",
                        expression_debug_name(args[1]),
                        self.type_name_string(offset_type),
                    )
                )

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{func_name}({generated_args})"

    def generate_derivative_call(self, func_name, args):
        expected_args = self.GLSL_DERIVATIVE_FUNCTIONS.get(func_name)
        if expected_args is None:
            return None
        if len(args) != expected_args:
            raise OpenGLRefusal(
                f"OpenGL derivative operation '{func_name}' requires "
                f"{expected_args} argument"
            )
        if self.current_stage_entry_type not in (None, "fragment"):
            raise OpenGLRefusal(
                f"OpenGL derivative operation '{func_name}' is only valid "
                "in fragment stages"
            )

        value_type = self.expression_result_type(args[0])
        component_type = self.vector_component_type(value_type)
        if value_type is not None and not (
            self.is_scalar_floating_type(value_type)
            or component_type in {"float", "double"}
        ):
            raise OpenGLRefusal(
                operation_argument_type_error(
                    "OpenGL",
                    "derivative",
                    func_name,
                    "a floating scalar or vector",
                    "value",
                    expression_debug_name(args[0]),
                    self.type_name_string(value_type),
                )
            )

        generated_arg = self.generate_expression(args[0])
        return f"{func_name}({generated_arg})"

    def ray_query_member_function_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        if func_expr.member not in self.RAY_QUERY_METHOD_MAP:
            return None
        return self.map_ray_query_intrinsic(func_expr.member, func_expr.object, args)

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            self.validate_structured_buffer_access_argument(func_name, args)
            index = self.glsl_index_expression(args[1], args[0])
            value = self.structured_buffer_access_expression(args[0], index)
            element_type = self.glsl_buffer_call_result_type(
                func_name, args, source_type=True
            )
            return self.glsl_apply_narrow_integer_contract(value, element_type)
        if func_name == "buffer_store" and len(args) >= 3:
            self.validate_structured_buffer_access_argument(func_name, args)
            self.validate_glsl_storage_pointer_binding_access(
                self.glsl_storage_pointer_binding(
                    args[0], self.glsl_storage_pointer_aliases()
                ),
                "write",
                args[0],
            )
            index = self.glsl_index_expression(args[1], args[0])
            element_type = self.structured_buffer_source_element_type(
                self.expression_result_type(args[0])
            )
            value = self.generate_expression_with_expected(args[2], element_type)
            value = self.glsl_precise_stored_value(None, args[2], value, element_type)
            value = self.float_atomic_storage.buffer_value(
                args[0], element_type, value, True
            )
            array_access = self.structured_buffer_array_parameter_access(args[0])
            if array_access is not None:
                info, selector = array_access
                return self.structured_buffer_array_select_expression(
                    info,
                    selector,
                    lambda name: f"({name}[{index}] = {value})",
                )
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_append" and len(args) >= 2:
            self.validate_structured_buffer_access_argument(func_name, args)
            element_type = self.structured_buffer_source_element_type(
                self.expression_result_type(args[0])
            )
            value = self.generate_expression_with_expected(args[1], element_type)
            value = self.glsl_precise_stored_value(None, args[1], value, element_type)
            value = self.float_atomic_storage.buffer_value(
                args[0], element_type, value, True
            )
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return self.refuse_lowering(
                    "buffer append: requires an append/consume counter buffer"
                )
            index = f"atomicAdd({counter}, 1u)"
            return (
                f"{self.structured_buffer_access_expression(args[0], index)} = {value}"
            )
        if func_name == "buffer_consume" and args:
            self.validate_structured_buffer_access_argument(func_name, args)
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return self.refuse_lowering(
                    "buffer consume: requires an append/consume counter buffer", "0"
                )
            index = f"(atomicAdd({counter}, uint(-1)) - 1u)"
            value = self.structured_buffer_access_expression(args[0], index)
            element_type = self.structured_buffer_source_element_type(
                self.expression_result_type(args[0])
            )
            value = self.float_atomic_storage.buffer_value(
                args[0], element_type, value, False
            )
            source_element = self.glsl_buffer_call_result_type(
                func_name, args, source_type=True
            )
            return self.glsl_apply_narrow_integer_contract(value, source_element)
        if func_name == "buffer_dimensions" and args:
            length_expr = self.structured_buffer_length_expression(args[0])
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                return f"{target} = {length_expr}"
            return length_expr
        return None

    def glsl_buffer_call_result_type(self, func_name, args, *, source_type=False):
        if func_name not in {"buffer_load", "buffer_consume"} or not args:
            return None

        binding = self.glsl_storage_pointer_binding(
            args[0], self.glsl_storage_pointer_aliases()
        )
        if binding is not None and binding.get("element_type") is not None:
            source_element = binding.get("source_element_type")
            if source_type and (
                self.glsl_narrow_float_width(source_element) is not None
                or self.glsl_narrow_integer_contract(source_element) is not None
            ):
                return source_element
            return binding["element_type"]

        if source_type:
            buffer_type = self.glsl_source_expression_type(args[0])
        else:
            buffer_type = self.expression_result_type(args[0])
        element_type = self.structured_buffer_source_element_type(buffer_type)
        if element_type is not None:
            return element_type

        normalized_type = self.glsl_normalized_source_type(buffer_type)
        if normalized_type and normalized_type.endswith("*"):
            return normalized_type[:-1].rstrip()
        return None

    def glsl_structured_buffer_array_element_member(self, buffer_expr):
        """The block member holding the elements of ``buffer_expr``, one buffer of a declared buffer array.

        An array of structured buffers is declared as an array of blocks (``buffer B { T data[]; } b[N];``), so
        element ``i`` of ``b[k]`` is ``b[k].data[i]``. ``None`` for any other operand.
        """
        if not isinstance(buffer_expr, ArrayAccessNode):
            return None
        base = getattr(buffer_expr, "array", None)
        if not isinstance(base, (IdentifierNode, VariableNode)):
            return None
        name = self.expression_name(base)
        if name in self.local_variable_types:
            return None
        return self.structured_buffer_instance_members.get(name)

    def structured_buffer_access_expression(self, buffer_arg, element_index):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}[{element_index}]",
            )
        storage_pointer_access = self.glsl_storage_pointer_element_expression(
            buffer_arg, element_index
        )
        if storage_pointer_access is not None:
            return storage_pointer_access
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}[{element_index}]"
        return f"{buffer}[{element_index}]"

    def glsl_structured_buffer_get_dimensions(self, func_expr, args):
        """``b.GetDimensions(count, stride)`` on a structured buffer: its element count and std430 stride."""
        if not (
            isinstance(func_expr, MemberAccessNode)
            and func_expr.member == "GetDimensions"
            and len(args or ()) == 2
        ):
            return None
        buffer_expr = func_expr.object
        if not isinstance(buffer_expr, (IdentifierNode, VariableNode)):
            return None
        structured = {"StructuredBuffer", "RWStructuredBuffer"}
        buffer_type = self.expression_result_type(buffer_expr)
        if buffer_type is None or (
            self.structured_buffer_type_name(buffer_type) not in structured
        ):
            # A helper's buffer parameter is an offset into the buffer the
            # specialized helper reads, whose type its binding carries.
            storage_binding = self.glsl_storage_pointer_binding(
                buffer_expr, self.glsl_storage_pointer_aliases()
            )
            if (
                storage_binding is None
                or "pointer_reinterpretation" in storage_binding
                or "byte_offset" in storage_binding
            ):
                return None
            buffer_type = storage_binding.get("type")
        if buffer_type is None or self.structured_buffer_type_name(buffer_type) not in (
            structured
        ):
            return None
        element = self.structured_buffer_element_type(buffer_type)
        info = std430_value_type_info(element)
        element_stride = None if info is None else std430_array_stride(info)
        if element_stride is None:
            return self.refuse_lowering(
                "StructuredBuffer.GetDimensions: the std430 stride of element type "
                f"'{element}' of '{self.expression_name(buffer_expr)}' is not known here",
                "0",
            )
        count = self.generate_expression(args[0])
        stride = self.generate_expression(args[1])
        length = self.structured_buffer_length_expression(buffer_expr)
        # A comma keeps the two writes one expression, so the call stays whole as the
        # body of an unbraced if or for.
        return f"{count} = uint({length}), {stride} = {element_stride}u"

    def structured_buffer_length_expression(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            return self.structured_buffer_array_select_expression(
                info,
                selector,
                lambda name: f"{name}.length()",
            )
        storage_binding = self.glsl_storage_pointer_binding(
            buffer_arg, self.glsl_storage_pointer_aliases()
        )
        if storage_binding is not None:
            root = storage_binding["root"]
            rendered_offset = self.glsl_workgroup_pointer_offset_expression(
                storage_binding
            )
            if (
                "pointer_reinterpretation" in storage_binding
                or "byte_offset" in storage_binding
            ):
                count_expr = self.glsl_reinterpreted_storage_length_expression(
                    root,
                    storage_binding,
                    source_location=getattr(buffer_arg, "source_location", None),
                )
            else:
                count_expr = f"{root}.length()"
            if rendered_offset in {"0", "0u"}:
                return count_expr
            return f"({count_expr} - int({rendered_offset}))"
        buffer = self.generate_expression(buffer_arg)
        buffer_name = self.expression_name(buffer_arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{buffer}.{instance_member}.length()"
        return f"{buffer}.length()"

    def glsl_reinterpreted_storage_length_expression(
        self, root, storage_binding, *, source_location=None
    ):
        """Return the GLSL ``int`` element count for a reinterpreted or byte-offset view."""
        reinterpretation = storage_binding.get("pointer_reinterpretation")
        byte_offset = str(storage_binding.get("byte_offset", "0"))
        if reinterpretation is None:
            if byte_offset in {"0", "0u"}:
                return f"{root}.length()"
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer length() cannot be computed for a "
                "byte-offset view without element layout",
                reason="storage-view-length-byte-offset-unsupported",
                source_location=source_location,
            )

        source_layout = reinterpretation.get("source_layout")
        target_layout = reinterpretation.get("target_layout")
        target_width = reinterpretation.get("target_width", 1)
        if (
            source_layout is None
            or target_layout is None
            or not isinstance(target_width, int)
            or target_width < 1
            or source_layout.byte_width < 1
            or target_layout.byte_width < 1
        ):
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer length() requires known source and "
                "target element layouts",
                reason="storage-view-length-layout-unknown",
                source_location=source_location,
            )

        source_elem_size = source_layout.byte_width
        view_elem_size = target_layout.byte_width * target_width
        if byte_offset in {"0", "0u"}:
            if source_elem_size == view_elem_size:
                return f"{root}.length()"
            if source_elem_size % view_elem_size == 0:
                return f"({root}.length() * {source_elem_size // view_elem_size})"
            if view_elem_size % source_elem_size == 0:
                return f"({root}.length() / {view_elem_size // source_elem_size})"
            return f"(({root}.length() * {source_elem_size}) / {view_elem_size})"

        total_bytes = (
            f"{root}.length()"
            if source_elem_size == 1
            else f"({root}.length() * {source_elem_size})"
        )
        remaining_bytes = f"({total_bytes} - int({byte_offset}))"
        if view_elem_size == 1:
            return remaining_bytes
        return f"({remaining_bytes} / {view_elem_size})"

    def structured_buffer_counter_reference(self, buffer_arg):
        array_access = self.structured_buffer_array_parameter_access(buffer_arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: f"{name}[0]",
                )

        counter_data = self.structured_buffer_counter_data_argument(buffer_arg)
        if counter_data is None:
            return None
        return f"{counter_data}[0]"

    def flattened_stage_member_name(self, expr):
        object_name = self.expression_name(getattr(expr, "object", None))
        if object_name in self.current_stage_inputs:
            return self.current_stage_inputs[object_name].get(expr.member)
        if object_name in self.current_stage_outputs:
            return self.current_stage_outputs[object_name].get(expr.member)
        return None

    def glsl_member_access_name(self, object_expr, member_name):
        object_type = self.type_name_string(self.expression_result_type(object_expr))
        if not object_type:
            return member_name
        struct_name, array_suffix = split_array_type_suffix(object_type)
        if array_suffix and member_name == "length":
            # array.length() is the array's method (GLSL 4.60 section 5.9),
            # not a member of its element structure.
            return member_name
        struct_name = self.glsl_source_type_identifier_name(struct_name)
        member_map = self.glsl_struct_member_name_map(struct_name)
        if not member_map:
            return member_name
        if member_name in member_map:
            return member_map[member_name]
        return self.refuse_lowering(
            f"member '{member_name}' is not declared by structure '{struct_name}'",
            member_name,
        )

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            if expr.name in self.current_identifier_aliases:
                return self.current_identifier_aliases[expr.name]
            return expr.name
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def set_index_range_assertions(self, assertions):
        """Configure explicit source ranges used only to justify index narrowing."""
        self.index_range_assertions = parse_index_range_assertions(assertions)
        return self

    def set_workgroup_access_assertions(self, assertions):
        """Configure explicit absolute ranges for workgroup pointer accesses."""
        self.workgroup_access_assertions = parse_workgroup_access_assertions(assertions)
        return self

    def glsl_index_target_profile(self):
        profile = self.GLSL_INDEX_TARGET_PROFILE
        version_line = str(self.current_glsl_version_line or "").strip()
        if not version_line or self.GLSL_TARGET_DISPLAY_NAME != "OpenGL":
            return profile
        return IndexTargetProfile(
            f"OpenGL {version_line}",
            profile.scalar_types,
        )

    def glsl_index_asserted_range(self, expression, *, function_name=None):
        expression_name = expression_debug_name(expression)
        matches = [
            assertion
            for assertion in self.index_range_assertions
            if assertion.applies_to(
                expression_name,
                function_name or self.current_glsl_source_function_name,
                getattr(self, "current_glsl_indexed_storage", None),
            )
        ]
        if not matches:
            return None
        minimum = max(assertion.value_range.minimum for assertion in matches)
        maximum = min(assertion.value_range.maximum for assertion in matches)
        if minimum > maximum:
            raise OpenGLRefusal(
                "Conflicting index range assertions for source expression "
                f"'{expression_name}'"
            )
        return IntegerRange(
            minimum,
            maximum,
            INDEX_RANGE_ASSERTED,
            ", ".join(sorted({assertion.source for assertion in matches})),
        )

    def glsl_integer_type_domain(self, expression, *, allow_wide=False):
        value_type = self.glsl_source_expression_type(expression)
        info = self.glsl_value_type_info(value_type)
        if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
            return None
        bits = info["narrow"]["bits"] if info["narrow"] is not None else info["bits"]
        if bits > 32 and not allow_wide:
            return None
        if info["family"] == "int":
            return IntegerRange(
                -(1 << (bits - 1)),
                (1 << (bits - 1)) - 1,
                INDEX_RANGE_STATIC,
                f"source type {self.type_name_string(value_type)}",
            )
        return IntegerRange(
            0,
            (1 << bits) - 1,
            INDEX_RANGE_STATIC,
            f"source type {self.type_name_string(value_type)}",
        )

    def glsl_index_flow_range(self, expression, intervals=None):
        if isinstance(expression, ArrayAccessNode):
            bounds = getattr(expression, "_glsl_lowered_pointer_offset_interval", None)
            if bounds is not None:
                return IntegerRange(
                    *bounds, INDEX_RANGE_STATIC, "source pointer-array read"
                )
        if intervals is None:
            intervals = getattr(expression, "_glsl_control_flow_intervals", None)
        if not intervals:
            return None

        returned_interval = scalar_return_interval(
            self, expression, intervals, self.glsl_active_literal_int_constants()
        )
        if returned_interval is not None:
            return IntegerRange(
                *returned_interval, INDEX_RANGE_STATIC, "source-typed helper return"
            )

        element_interval = self.glsl_metadata_element_interval(
            expression, intervals, self.glsl_active_literal_int_constants()
        )
        if element_interval is not None:
            return IntegerRange(
                *element_interval,
                INDEX_RANGE_STATIC,
                "checked immutable metadata elements",
            )

        if isinstance(expression, MemberAccessNode):
            component_index = self.glsl_index_component_index(expression.member)
            key = self.glsl_index_component_interval_key(
                expression.object,
                component_index,
            )
            field_key = self.glsl_declared_struct_member_interval_key(expression)
            if field_key is not None:
                key = field_key
            if key not in intervals:
                key = expression_debug_name(expression)
        elif isinstance(expression, ArrayAccessNode):
            key = self.glsl_private_pointer_interval_key(
                expression, intervals, self.glsl_active_literal_int_constants()
            )
        else:
            key = self.expression_name(expression)
        interval = intervals.get(key) if key is not None else None
        related = related_local_interval(
            self,
            getattr(self, "current_glsl_emitted_function", None),
            expression,
            intervals,
        )
        if related is not None:
            interval = (
                related
                if interval is None
                else (max(interval[0], related[0]), min(interval[1], related[1]))
            )
        if interval is None:
            return None

        candidate = IntegerRange(
            interval[0],
            interval[1],
            INDEX_RANGE_STATIC,
            f"control-flow proof for {expression_debug_name(expression)}",
        )
        domain = self.glsl_integer_type_domain(expression, allow_wide=True)
        return candidate if domain is None or candidate.is_within(domain) else None

    def glsl_unsigned_shift_interval(self, expression, left, right):
        domain = self.glsl_integer_type_domain(expression.left, allow_wide=True)
        if domain is None or domain.minimum != 0:
            return None
        bits = domain.maximum.bit_length()
        # Narrow operands require source-language integer promotion first.
        if bits not in {32, 64} or not (
            0 <= left[0] <= left[1] <= domain.maximum
            and 0 <= right[0] <= right[1] < bits
        ):
            return None
        if self.map_operator(expression.op) == "<<":
            result = left[0] << right[0], left[1] << right[1]
            return result if result[1] <= domain.maximum else None
        if self.map_operator(expression.op) == ">>":
            return left[0] >> right[1], left[1] >> right[0]
        return None
