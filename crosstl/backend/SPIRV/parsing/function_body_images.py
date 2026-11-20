"""Image sampling, fetch, gather, query and read instructions of SPIR-V function bodies."""

from ..sampled_images import (
    sampled_image_call,
    sampled_image_image,
)
from ..subpass_import import (
    is_subpass_image,
    subpass_load_expression,
)
from ..VulkanAst import (
    FunctionCallNode,
)


class VulkanFunctionBodyImageMixin:
    """Function-body instruction members of ``VulkanParser``."""

    def spirv_assembly_image_sample_instruction(
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
        """Translate a image sample instruction of a function body; return True when it is handled.

        Handles ``OpImage``, ``OpSampledImage``, ``OpImageSample*``, ``OpImageSparseSample*``.
        """
        if result_id and opcode == "OpImage" and len(operands) >= 2:
            record_expression(
                result_id,
                operands[0],
                sampled_image_image(
                    self.spirv_assembly_operand_expression(
                        operands[1],
                        expressions,
                        names,
                        decorations,
                        constants,
                    )
                ),
            )
            return True

        if result_id and opcode == "OpSampledImage" and len(operands) >= 3:
            record_expression(
                result_id,
                operands[0],
                sampled_image_call(
                    FunctionCallNode,
                    self.spirv_type_name(operands[0], types) or operands[0],
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
                ),
            )
            return True

        if result_id and opcode.startswith("OpImageSample") and len(operands) >= 3:
            record_expression(
                result_id,
                operands[0],
                self.spirv_assembly_image_sample_expression(
                    opcode,
                    operands[1],
                    operands[2],
                    operands[3:],
                    expressions,
                    names,
                    decorations,
                    constants,
                    coordinate_components=self.spirv_sample_coordinate_components(
                        opcode,
                        operands[1],
                        operands[2],
                        expression_type_ids,
                        constant_types,
                        types,
                    ),
                ),
            )
            return True

        if (
            result_id
            and opcode.startswith("OpImageSparseSample")
            and len(operands) >= 3
        ):
            expressions[result_id] = self.spirv_assembly_image_sparse_sample_expression(
                opcode,
                operands[1],
                operands[2],
                operands[3:],
                expressions,
                names,
                decorations,
                constants,
                coordinate_components=self.spirv_sample_coordinate_components(
                    opcode,
                    operands[1],
                    operands[2],
                    expression_type_ids,
                    constant_types,
                    types,
                ),
            )
            expression_type_ids[result_id] = operands[0]
            return True
        return False

    def spirv_assembly_image_fetch_and_gather_instruction(
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
        """Translate a image fetch and gather instruction of a function body; return True when it is handled.

        Handles ``OpImageSparseTexelsResident``, ``OpImageSparseRead``, ``OpImageSparseFetch``,
        ``OpImageSparseGather``, ``OpImageSparseDrefGather``, ``OpImageGather``, ``OpImageDrefGather``,
        ``OpImageFetch``.
        """
        if result_id and opcode == "OpImageSparseTexelsResident":
            if len(operands) >= 2:
                expressions[result_id] = FunctionCallNode(
                    "spirvSparseTexelsResident",
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

        if result_id and opcode == "OpImageSparseRead" and len(operands) >= 3:
            expressions[result_id] = self.spirv_assembly_image_sparse_read_expression(
                operands[1],
                operands[2],
                operands[3:],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpImageSparseFetch" and len(operands) >= 3:
            expressions[result_id] = self.spirv_assembly_image_sparse_fetch_expression(
                operands[1],
                operands[2],
                operands[3:],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode in {
            "OpImageSparseGather",
            "OpImageSparseDrefGather",
        }:
            if len(operands) >= 4:
                expressions[result_id] = (
                    self.spirv_assembly_image_sparse_gather_expression(
                        opcode,
                        operands[1],
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

        if result_id and opcode in {"OpImageGather", "OpImageDrefGather"}:
            if len(operands) >= 4:
                expressions[result_id] = self.spirv_assembly_image_gather_expression(
                    opcode,
                    operands[1],
                    operands[2],
                    operands[3],
                    operands[4:],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                expression_type_ids[result_id] = operands[0]
                return True

        if result_id and opcode == "OpImageFetch" and len(operands) >= 3:
            expressions[result_id] = self.spirv_assembly_image_fetch_expression(
                operands[1],
                operands[2],
                operands[3:],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True
        return False

    def spirv_assembly_image_size_query_instruction(
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
        """Translate a image size query instruction of a function body; return True when it is handled.

        Handles ``OpImageQuerySizeLod``, ``OpImageQuerySize``, ``OpImageQueryLod``, ``OpImageQueryLevels``.
        """
        if result_id and opcode == "OpImageQuerySizeLod" and len(operands) >= 3:
            expressions[result_id] = FunctionCallNode(
                "textureSize",
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

        if result_id and opcode == "OpImageQuerySize" and len(operands) >= 2:
            expressions[result_id] = FunctionCallNode(
                self.spirv_image_query_name(
                    "textureSize",
                    "imageSize",
                    operands[1],
                    expression_type_ids,
                    types,
                ),
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

        if result_id and opcode == "OpImageQueryLod" and len(operands) >= 3:
            expressions[result_id] = FunctionCallNode(
                "textureQueryLod",
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

        if result_id and opcode == "OpImageQueryLevels" and len(operands) >= 2:
            expressions[result_id] = FunctionCallNode(
                "textureQueryLevels",
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

    def spirv_assembly_image_property_query_instruction(
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
        """Translate a image property query instruction of a function body; return True when it is handled.

        Handles ``OpImageQuerySamples``, ``OpImageQueryFormat``, ``OpImageQueryOrder``.
        """
        if result_id and opcode == "OpImageQuerySamples" and len(operands) >= 2:
            expressions[result_id] = FunctionCallNode(
                self.spirv_image_query_name(
                    "textureSamples",
                    "imageSamples",
                    operands[1],
                    expression_type_ids,
                    types,
                ),
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

        if (
            result_id
            and opcode in {
                "OpImageQueryFormat",
                "OpImageQueryOrder",
            }
            and len(operands) >= 2
        ):
            expressions[result_id] = FunctionCallNode(
                {
                    "OpImageQueryFormat": "spirvImageQueryFormat",
                    "OpImageQueryOrder": "spirvImageQueryOrder",
                }[opcode],
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

    def spirv_assembly_image_read_instruction(
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
        """Translate a image read instruction of a function body; return True when it is handled.

        Handles ``OpImageRead``, ``OpImageTexelPointer``, ``OpAtomicLoad``, ``OpNamedBarrierInitialize``.
        """
        if (
            result_id
            and opcode == "OpImageRead"
            and len(operands) >= 3
            and is_subpass_image(operands[1], expression_type_ids, types)
        ):
            parsed_operands = self.spirv_assembly_image_operands(
                operands[3:], expressions, names, decorations, constants
            )
            expressions[result_id] = subpass_load_expression(
                self.spirv_assembly_operand_expression(
                    operands[1], expressions, names, decorations, constants
                ),
                operands[2],
                constants,
                self.spirv_assembly_image_sample_operand(parsed_operands),
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpImageRead" and len(operands) >= 3:
            expressions[result_id] = self.spirv_assembly_image_read_expression(
                operands[1],
                operands[2],
                operands[3:],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpImageTexelPointer" and len(operands) >= 4:
            expressions[result_id] = self.spirv_assembly_image_texel_pointer_expression(
                operands[1],
                operands[2],
                operands[3],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpAtomicLoad" and len(operands) >= 4:
            expressions[result_id] = self.spirv_assembly_atomic_load_expression(
                operands[1],
                expressions,
                names,
                decorations,
                constants,
            )
            expression_type_ids[result_id] = operands[0]
            return True

        if result_id and opcode == "OpNamedBarrierInitialize" and len(operands) >= 2:
            expressions[result_id] = FunctionCallNode(
                "spirvNamedBarrierInitialize",
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
