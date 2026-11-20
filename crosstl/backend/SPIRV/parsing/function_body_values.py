"""Declaration, conversion, access-chain and composite instructions of SPIR-V function bodies."""

from ..integer_signedness_import import (
    spirv_integer_conversion_expression,
)
from ..VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    VariableNode,
)


class VulkanFunctionBodyValueMixin:
    """Function-body instruction members of ``VulkanParser``."""

    def spirv_assembly_declaration_instruction(
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
        """Translate a declaration instruction of a function body; return True when it is handled.

        Handles ``OpLabel``, ``OpFunctionParameter``, ``OpVariable``, ``OpLoad``.
        """
        if result_id and opcode == "OpLabel":
            structure.block_started(result_id, statements)
            return True

        if result_id and opcode == "OpFunctionParameter":
            expressions[result_id] = VariableNode(
                "",
                self.spirv_assembly_value_name(result_id, names, decorations),
                spirv_id=result_id,
            )
            if operands:
                expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpVariable" and len(operands) >= 2:
            pointer_type = types.get(operands[0], {})
            storage_class = operands[1]
            variable_name = self.spirv_assembly_value_name(
                result_id, names, decorations
            )
            expressions[result_id] = VariableNode("", variable_name, spirv_id=result_id)
            expression_type_ids[result_id] = operands[0]
            if storage_class == "Function":
                variable_type, array_suffix = self.spirv_type_name_and_suffix(
                    pointer_type.get("type_id"),
                    types,
                    constants,
                    names=names,
                )
                variable_type = variable_type or pointer_type.get("type_id") or ""
                declaration = VariableNode(
                    variable_type,
                    f"{variable_name}{array_suffix}",
                    spirv_id=result_id,
                    spirv_type_id=pointer_type.get("type_id"),
                )
                if len(operands) >= 3:
                    statements.append(
                        AssignmentNode(
                            declaration,
                            self.spirv_assembly_operand_expression(
                                operands[2],
                                expressions,
                                names,
                                decorations,
                                constants,
                            ),
                        )
                    )
                else:
                    statements.append(declaration)
            return True

        if result_id and opcode == "OpLoad" and len(operands) >= 2:
            expressions[result_id] = self.spirv_assembly_operand_expression(
                operands[1],
                expressions,
                names,
                decorations,
                constants,
                variables_by_id=variables_by_id,
                types=types,
            )
            expression_type_ids[result_id] = operands[0]
            return True
        return False

    def spirv_assembly_conversion_instruction(
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
        """Translate a conversion instruction of a function body; return True when it is handled.

        Handles ``OpConvert*``, ``OpFConvert``, ``OpSConvert``, ``OpUConvert``, ``OpBitcast``,
        ``OpCopyLogical``, ``OpCopyObject``, ``OpUndef``, ``OpConstantNull``, ``OpTranspose``.
        """
        if (
            result_id
            and (
                opcode.startswith("OpConvert")
                or opcode in {"OpFConvert", "OpSConvert", "OpUConvert"}
            )
            and len(operands) >= 2
        ):
            operand = self.spirv_assembly_operand_expression(
                operands[1],
                expressions,
                names,
                decorations,
                constants,
            )
            expressions[result_id] = spirv_integer_conversion_expression(
                opcode[2:],
                operands[0],
                operand,
                expression_type_ids.get(operands[1]) or constant_types.get(operands[1]),
                types,
                self.spirv_vector_type_name,
            ) or FunctionCallNode(
                self.spirv_type_name(operands[0], types) or operands[0],
                [operand],
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpBitcast" and len(operands) >= 2:
            expressions[result_id] = self.spirv_assembly_bitcast_expression(
                operands[0],
                operands[1],
                expressions,
                expression_type_ids,
                names,
                decorations,
                constants,
                constant_types,
                types,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if (
            result_id
            and opcode in {"OpCopyLogical", "OpCopyObject"}
            and len(operands) >= 2
        ):
            expressions[result_id] = self.spirv_assembly_operand_expression(
                operands[1],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpUndef" and operands:
            expressions[result_id] = self.spirv_assembly_undef_expression(
                operands[0], types
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpConstantNull" and operands:
            expressions[result_id] = self.spirv_assembly_null_expression(
                operands[0], types
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpTranspose" and len(operands) >= 2:
            expressions[result_id] = FunctionCallNode(
                "transpose",
                [
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                ],
            )
            expression_type_ids[result_id] = operands[0]
            return True
        return False

    def spirv_assembly_access_chain_instruction(
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
        """Translate a access chain instruction of a function body; return True when it is handled.

        Handles ``OpAccessChain``, ``OpInBoundsAccessChain``, ``OpPtrAccessChain``,
        ``OpInBoundsPtrAccessChain``, ``OpArrayLength``, ``OpCompositeConstruct``.
        """
        if result_id and opcode in {"OpAccessChain", "OpInBoundsAccessChain"}:
            if len(operands) >= 2:
                access = self.spirv_assembly_access_chain_expression(
                    operands[1],
                    operands[2:],
                    expressions,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    variables_by_id,
                    constants,
                    expression_type_ids=expression_type_ids,
                )
                expressions[result_id] = access
                expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode in {
            "OpPtrAccessChain",
            "OpInBoundsPtrAccessChain",
        }:
            if len(operands) >= 3:
                access = self.spirv_assembly_access_chain_expression(
                    operands[1],
                    operands[2:],
                    expressions,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    variables_by_id,
                    constants,
                )
                expressions[result_id] = access
                expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpArrayLength" and len(operands) >= 3:
            expressions[result_id] = FunctionCallNode(
                "spirvArrayLength",
                [
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    operands[2],
                ],
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpCompositeConstruct" and operands:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_composite_construct_expression(
                    operands[0],
                    operands[1:],
                    expressions,
                    names,
                    decorations,
                    constants,
                    types,
                ),
            )
            return True
        return False

    def spirv_assembly_composite_instruction(
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
        """Translate a composite instruction of a function body; return True when it is handled.

        Handles ``OpCompositeExtract``, ``OpVectorExtractDynamic``, ``OpVectorInsertDynamic``,
        ``OpCompositeInsert``, ``OpVectorShuffle``.
        """
        if result_id and opcode == "OpCompositeExtract" and len(operands) >= 2:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_composite_extract_expression(
                    operands[1],
                    operands[2:],
                    expressions,
                    expression_type_ids,
                    names,
                    decorations,
                    member_names,
                    types,
                    constants,
                    constant_types,
                ),
            )
            return True

        if result_id and opcode == "OpVectorExtractDynamic" and len(operands) >= 3:
            expressions[result_id] = ArrayAccessNode(
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
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpVectorInsertDynamic" and len(operands) >= 4:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_vector_insert_dynamic_expression(
                    operands[0],
                    operands[1],
                    operands[2],
                    operands[3],
                    expressions,
                    names,
                    decorations,
                    constants,
                    types,
                ),
            )
            return True

        if result_id and opcode == "OpCompositeInsert" and len(operands) >= 3:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_composite_insert_expression(
                    operands[0],
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                    types,
                ),
            )
            return True

        if result_id and opcode == "OpVectorShuffle" and len(operands) >= 4:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_vector_shuffle_expression(
                    operands[0],
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    expression_type_ids,
                    names,
                    decorations,
                    constants,
                    types,
                ),
            )
            return True
        return False
