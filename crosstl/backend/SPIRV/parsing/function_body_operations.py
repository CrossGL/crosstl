"""Extended, atomic, subgroup, arithmetic, call, barrier, store and return instructions of SPIR-V function bodies."""

from ..barrier_import import (
    spirv_barrier_statements,
)
from ..image_texel_atomics import (
    fold_image_texel_atomic,
)
from ..integer_signedness_import import (
    spirv_integer_binary_expression,
)
from ..narrow_integer_import import (
    spirv_narrow_integer_result,
)
from ..ray_tracing_import import (
    SpirvInstructionUnsupportedError,
)
from ..select_import import (
    spirv_select_expression,
)
from ..subgroup_import import (
    OPCODES as SPIRV_SUBGROUP_CALL_OPCODES,
)
from ..subgroup_import import (
    spirv_subgroup_call,
)
from ..struct_result_import import (
    spirv_extended_arithmetic_call,
    spirv_struct_result_call,
)
from ..VulkanAst import (
    AssignmentNode,
    BinaryOpNode,
    DiscardNode,
    FunctionCallNode,
    ReturnNode,
    UnaryOpNode,
)


class VulkanFunctionBodyOperationMixin:
    """Function-body instruction members of ``VulkanParser``."""

    def spirv_assembly_extended_set_instruction(
        self,
        result_id,
        opcode,
        operands,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a extended set instruction of a function body; return True when it is handled.

        Handles ``OpExtInst``.
        """
        if result_id and opcode == "OpExtInst" and len(operands) == 4:
            struct_value = spirv_struct_result_call(
                self,
                result_id,
                operands[0],
                extended_instruction_imports.get(operands[1]),
                operands[2],
                self.spirv_assembly_operand_expression(
                    operands[3], expressions, names, decorations, constants
                ),
                statements,
                names,
                decorations,
                types,
                constants,
            )
            if struct_value is not None:
                expressions[result_id] = struct_value
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and opcode == "OpExtInst" and len(operands) >= 3:
            call = FunctionCallNode(
                self.spirv_ext_inst_function_name(
                    extended_instruction_imports.get(operands[1]),
                    operands[2],
                ),
                [
                    self.spirv_assembly_operand_expression(
                        operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                    for operand in operands[3:]
                ],
            )
            if (
                extended_instruction_imports.get(operands[1]) == "GLSL.std.450"
                and self.SPIRV_GLSL_STD_450_EXT_INST_IDS.get(
                    str(operands[2]), operands[2]
                )
                == "NClamp"
                and len(call.args) == 3
            ):
                # NClamp(x, lo, hi) is NMin(NMax(x, lo), hi).
                call = FunctionCallNode(
                    "fmin",
                    [FunctionCallNode("fmax", call.args[:2]), call.args[2]],
                )
            if self.spirv_type_name(operands[0], types) == "void":
                statements.append(call)
            else:
                record_expression(result_id, operands[0], call)
            return True
        return False

    def spirv_assembly_atomic_and_intersection_instruction(
        self,
        result_id,
        opcode,
        operands,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a atomic and intersection instruction of a function body; return True when it is handled.

        Handles ``OpAtomicIIncrement``, ``OpAtomicIDecrement``, the ``SPIRV_ATOMIC_RMW_FUNCTIONS`` opcodes,
        ``OpReportIntersectionKHR``, ``OpAtomicCompareExchange``, ``OpAtomicCompareExchangeWeak``.
        """
        if result_id and opcode in {
            "OpAtomicIIncrement",
            "OpAtomicIDecrement",
        }:
            if len(operands) >= 4:
                expressions[result_id] = fold_image_texel_atomic(
                    self.spirv_assembly_atomic_increment_expression(
                        opcode,
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    opcode,
                    self.spirv_type_name(operands[0], types),
                )
                expression_type_ids[result_id] = operands[0]
                if result_id not in used_result_ids:
                    statements.append(expressions[result_id])
                return True

        if result_id and opcode in self.SPIRV_ATOMIC_RMW_FUNCTIONS:
            if len(operands) >= 5:
                expressions[result_id] = fold_image_texel_atomic(
                    self.spirv_assembly_atomic_expression(
                        opcode,
                        operands[1],
                        operands[4],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    opcode,
                    self.spirv_type_name(operands[0], types),
                )
                expression_type_ids[result_id] = operands[0]
                if result_id not in used_result_ids:
                    statements.append(expressions[result_id])
                return True

        if result_id and opcode == "OpReportIntersectionKHR":
            if len(operands) < 3:
                raise SpirvInstructionUnsupportedError(
                    opcode, f"needs 3 operands, got {len(operands)}"
                )
            expressions[result_id] = FunctionCallNode(
                "ReportHit",
                [
                    self.spirv_assembly_operand_expression(
                        operand, expressions, names, decorations, constants
                    )
                    for operand in operands[1:3]
                ],
            )
            expression_type_ids[result_id] = operands[0]
            if result_id not in used_result_ids:
                # A discarded result still reports the hit.
                statements.append(expressions[result_id])
            return True

        if result_id and opcode in {
            "OpAtomicCompareExchange",
            "OpAtomicCompareExchangeWeak",
        }:
            if len(operands) >= 7:
                expressions[result_id] = fold_image_texel_atomic(
                    self.spirv_assembly_atomic_compare_exchange_expression(
                        operands[1],
                        operands[5],
                        operands[6],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    opcode,
                    self.spirv_type_name(operands[0], types),
                )
                expression_type_ids[result_id] = operands[0]
                if result_id not in used_result_ids:
                    statements.append(expressions[result_id])
                return True
        return False

    def spirv_assembly_subgroup_instruction(
        self,
        result_id,
        opcode,
        operands,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a subgroup instruction of a function body; return True when it is handled.

        Handles ``OpGroupNonUniformBroadcastFirst``, ``OpGroupNonUniformBroadcast``, the vote, ballot, shuffle,
        quad, rotate and partition instructions of :mod:`..subgroup_import`, and the
        ``SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS`` opcodes.
        """
        if result_id and opcode == "OpGroupNonUniformBroadcastFirst":
            if len(operands) >= 3:
                expressions[result_id] = FunctionCallNode(
                    "subgroupBroadcastFirst",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and opcode == "OpGroupNonUniformBroadcast":
            if len(operands) >= 4:
                expressions[result_id] = FunctionCallNode(
                    "subgroupBroadcast",
                    [
                        self.spirv_assembly_operand_expression(
                            operands[2],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[3],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    ],
                )
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and opcode in SPIRV_SUBGROUP_CALL_OPCODES:
            call = spirv_subgroup_call(
                opcode,
                operands,
                lambda operand: self.spirv_assembly_operand_expression(
                    operand, expressions, names, decorations, constants
                ),
                lambda operand: self.spirv_integer_constant_operand(operand, constants),
            )
            if call is not None:
                expressions[result_id] = call
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and opcode in self.SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS:
            if len(operands) >= 4:
                expressions[result_id] = (
                    self.spirv_assembly_group_non_uniform_reduction_expression(
                        opcode,
                        operands[2],
                        operands[3],
                        operands[4:],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
                expression_type_ids[result_id] = operands[0]
                return True
        return False

    def spirv_assembly_ray_query_and_select_instruction(
        self,
        result_id,
        opcode,
        operands,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a ray query and select instruction of a function body; return True when it is handled.

        Handles ``OpDot``, the ``SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS`` opcodes, ``OpSelect``, ``OpPhi``.
        """
        if result_id and opcode == "OpDot" and len(operands) >= 3:
            expressions[result_id] = FunctionCallNode(
                "dot",
                [
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    self.spirv_assembly_operand_expression(
                        operands[2],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                ],
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if (
            result_id
            and opcode in self.SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS
            and len(operands) >= 2
        ):
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_ray_query_expression(
                    opcode,
                    operands,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            )
            return True

        if result_id and opcode == "OpSelect" and len(operands) >= 4:
            condition, true_value, false_value = (
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands[1:4]
            )
            record_expression(
                result_id,
                operands[0],
                spirv_select_expression(
                    condition,
                    true_value,
                    false_value,
                    expression_type_ids.get(operands[1])
                    or constant_types.get(operands[1]),
                    types,
                ),
            )
            return True

        if result_id and opcode == "OpPhi" and len(operands) >= 3:
            structure.import_phi(
                result_id,
                operands,
                expressions,
                expression_type_ids,
                record_expression,
            )
            return True
        return False

    def spirv_assembly_arithmetic_instruction(
        self,
        result_id,
        opcode,
        operands,
        operation,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a arithmetic instruction of a function body; return True when it is handled.

        Handles the ``SPIRV_CORE_FUNCTIONS`` opcodes, the ``SPIRV_DERIVATIVE_FUNCTIONS`` opcodes, the
        ``SPIRV_UNARY_FUNCTIONS`` opcodes.
        """
        if result_id and len(operands) >= 3:
            struct_value = spirv_extended_arithmetic_call(
                self,
                result_id,
                operands[0],
                operation,
                [
                    self.spirv_assembly_operand_expression(
                        operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                    for operand in operands[1:]
                ],
                statements,
                names,
                decorations,
                types,
            )
            if struct_value is not None:
                expressions[result_id] = struct_value
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and operation in self.SPIRV_CORE_FUNCTIONS and len(operands) >= 2:
            record_expression(
                result_id,
                operands[0],
                FunctionCallNode(
                    self.SPIRV_CORE_FUNCTIONS[operation],
                    [
                        self.spirv_assembly_operand_expression(
                            operand,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for operand in operands[1:]
                    ],
                ),
            )
            return True

        if (
            result_id
            and operation in self.SPIRV_DERIVATIVE_FUNCTIONS
            and len(operands) >= 2
        ):
            record_expression(
                result_id,
                operands[0],
                FunctionCallNode(
                    self.SPIRV_DERIVATIVE_FUNCTIONS[operation],
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                ),
            )
            return True

        if result_id and operation in self.SPIRV_UNARY_FUNCTIONS and len(operands) >= 2:
            record_expression(
                result_id,
                operands[0],
                FunctionCallNode(
                    self.SPIRV_UNARY_FUNCTIONS[operation],
                    [
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                    ],
                ),
            )
            return True
        return False

    def spirv_assembly_remainder_and_specialization_instruction(
        self,
        result_id,
        opcode,
        operands,
        operation,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a remainder and specialization instruction of a function body; return True when it is handled.

        Handles ``OpFMod``, the ``SPIRV_SPEC_CONSTANT_BINARY_OPS`` opcodes, the
        ``SPIRV_SPEC_CONSTANT_UNARY_OPS`` opcodes.
        """
        if result_id and operation == "FMod" and len(operands) >= 3:
            # OpFMod takes the sign of the divisor (x - y * floor(x / y)), which is
            # CrossGL mod; the % operator is the truncated remainder of OpFRem.
            record_expression(
                result_id,
                operands[0],
                FunctionCallNode(
                    "mod",
                    [
                        self.spirv_assembly_operand_expression(
                            operand, expressions, names, decorations, constants
                        )
                        for operand in operands[1:3]
                    ],
                ),
            )
            return True

        if (
            result_id
            and operation in self.SPIRV_SPEC_CONSTANT_BINARY_OPS
            and len(operands) >= 3
        ):
            lhs, rhs = (
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands[1:3]
            )
            expression = spirv_integer_binary_expression(
                operation,
                operands[0],
                lhs,
                expression_type_ids.get(operands[1]) or constant_types.get(operands[1]),
                rhs,
                expression_type_ids.get(operands[2]) or constant_types.get(operands[2]),
                types,
                self.spirv_vector_type_name,
            )
            if expression is None:
                expression = BinaryOpNode(
                    lhs, self.SPIRV_SPEC_CONSTANT_BINARY_OPS[operation], rhs
                )
            expression = spirv_narrow_integer_result(
                operation, operands[0], expression, types
            )
            record_expression(result_id, operands[0], expression)
            return True

        if (
            result_id
            and operation in self.SPIRV_SPEC_CONSTANT_UNARY_OPS
            and len(operands) >= 2
        ):
            expression = UnaryOpNode(
                self.SPIRV_SPEC_CONSTANT_UNARY_OPS[operation],
                self.spirv_assembly_operand_expression(
                    operands[1],
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            )
            record_expression(
                result_id,
                operands[0],
                spirv_narrow_integer_result(operation, operands[0], expression, types),
            )
            return True
        return False

    def spirv_assembly_call_and_barrier_instruction(
        self,
        result_id,
        opcode,
        operands,
        operation,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a call and barrier instruction of a function body; return True when it is handled.

        Handles ``OpFunctionCall``, ``OpAtomicStore``, ``OpImageWrite``, the
        ``SPIRV_GEOMETRY_EMIT_FUNCTIONS`` opcodes, the ``SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS`` opcodes,
        ``OpControlBarrier``, ``OpMemoryBarrier``, ``OpMemoryNamedBarrier``.
        """
        if result_id and opcode == "OpFunctionCall" and len(operands) >= 2:
            call = FunctionCallNode(
                self.spirv_assembly_value_name(
                    operands[1], names, decorations, prefix="function"
                ),
                [
                    self.spirv_assembly_operand_expression(
                        operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                    for operand in operands[2:]
                ],
            )
            if self.spirv_type_name(operands[0], types) == "void":
                statements.append(call)
            elif result_id not in used_result_ids:
                # The call's effects happen even when its result is discarded.
                statements.append(call)
            else:
                expressions[result_id] = call
                expression_type_ids[result_id] = operands[0]
            return True

        if opcode == "OpAtomicStore" and len(operands) >= 4:
            statements.append(
                self.spirv_assembly_atomic_store_statement(
                    operands[0],
                    operands[3],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
            return True

        if opcode == "OpImageWrite" and len(operands) >= 3:
            statements.append(
                self.spirv_assembly_image_write_statement(
                    operands[0],
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
            return True

        if opcode in self.SPIRV_GEOMETRY_EMIT_FUNCTIONS:
            statements.append(
                self.spirv_assembly_geometry_emit_statement(
                    opcode,
                    operands,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
            return True

        if opcode in self.SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS:
            statements.append(
                self.spirv_assembly_ray_query_statement(
                    opcode,
                    operands,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
            return True

        if opcode == "OpControlBarrier" and len(operands) >= 3:
            statements.extend(
                spirv_barrier_statements(
                    opcode,
                    [constants.get(operand) for operand in operands[:3]],
                    self.spirv_module_execution_models,
                )
            )
            return True

        if opcode == "OpMemoryBarrier" and len(operands) >= 2:
            statements.extend(
                spirv_barrier_statements(
                    opcode,
                    [constants.get(operand) for operand in operands[:2]],
                    self.spirv_module_execution_models,
                )
            )
            return True

        if opcode == "OpMemoryNamedBarrier" and len(operands) >= 3:
            statements.append(
                FunctionCallNode(
                    "spirvMemoryNamedBarrier",
                    [
                        self.spirv_assembly_operand_expression(
                            operand,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for operand in operands[:3]
                    ],
                )
            )
            return True
        return False

    def spirv_assembly_store_and_return_instruction(
        self,
        result_id,
        opcode,
        operands,
        operation,
        *,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
        constant_types,
        extended_instruction_imports,
        statements,
        expressions,
        expression_type_ids,
        variables_by_id,
        used_result_ids,
        structure,
        record_expression,
    ):
        """Translate a store and return instruction of a function body; return True when it is handled.

        Handles ``OpStore``, ``OpCopyMemory``, ``OpCopyMemorySized``, ``OpKill``, ``OpTerminateInvocation``,
        ``OpDemoteToHelperInvocation``, ``OpReturnValue``, ``OpReturn``.
        """
        if opcode == "OpStore" and len(operands) >= 2:
            statements.append(
                AssignmentNode(
                    self.spirv_assembly_operand_expression(
                        operands[0],
                        expressions,
                        names,
                        decorations,
                        constants,
                        variables_by_id=variables_by_id,
                        types=types,
                    ),
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                        variables_by_id=variables_by_id,
                        types=types,
                    ),
                )
            )
            return True

        if opcode == "OpCopyMemory" and len(operands) >= 2:
            if operands[0] != operands[1]:
                statements.append(
                    AssignmentNode(
                        self.spirv_assembly_operand_expression(
                            operands[0],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                        self.spirv_assembly_operand_expression(
                            operands[1],
                            expressions,
                            names,
                            decorations,
                            constants,
                        ),
                    )
                )
            return True

        if opcode == "OpCopyMemorySized" and len(operands) >= 3:
            statements.append(
                self.spirv_assembly_copy_memory_sized_statement(
                    operands[0],
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
            return True

        if opcode in {
            "OpKill",
            "OpTerminateInvocation",
            "OpDemoteToHelperInvocation",
        }:
            statements.append(DiscardNode())
            return True

        if opcode == "OpReturnValue" and operands:
            statements.append(
                ReturnNode(
                    self.spirv_assembly_operand_expression(
                        operands[0],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                )
            )
            return True

        if opcode == "OpReturn":
            statements.append(ReturnNode())
            return True
        return False
