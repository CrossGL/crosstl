"""Bit casts, undefined and null values, composite insertion, image operands and extended instruction names in the SPIR-V parser."""

from ..integer_signedness_import import spirv_integer_bitcast_expression
from ..VulkanAst import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
)


class VulkanValueMixin:
    """Value members of ``VulkanParser``."""

    def spirv_assembly_bitcast_expression(
        self,
        result_type_id,
        operand_id,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        constant_types,
        types,
    ):
        operand = self.spirv_assembly_operand_expression(
            operand_id, expressions, names, decorations, constants
        )
        source_type_id = expression_type_ids.get(operand_id) or constant_types.get(
            operand_id
        )
        source_family = self.spirv_bitcast_component_family(source_type_id, types)
        result_family = self.spirv_bitcast_component_family(result_type_id, types)

        if source_family == result_family and source_family is not None:
            return operand

        function_name = self.spirv_bitcast_function_name(source_family, result_family)
        if function_name is None:
            reinterpreted = spirv_integer_bitcast_expression(
                operand,
                source_type_id,
                result_type_id,
                types,
                self.spirv_vector_type_name,
            )
            if reinterpreted is not None:
                return reinterpreted
            result_type_name = self.spirv_type_name(result_type_id, types)
            fallback_type = self.spirv_fallback_identifier(
                result_type_name or result_type_id, "type"
            )
            function_name = f"spirvBitcast_{fallback_type}"

        return FunctionCallNode(function_name, [operand])

    def spirv_bitcast_component_family(self, type_id, types):
        type_info = types.get(type_id, {})
        if type_info.get("kind") == "vector":
            return self.spirv_bitcast_component_family(
                type_info.get("component_type"), types
            )

        type_name = self.spirv_type_name(type_id, types)
        if type_name in {"float", "int", "uint"}:
            return type_name
        return None

    def spirv_bitcast_function_name(self, source_family, result_family):
        return {
            ("float", "int"): "floatBitsToInt",
            ("float", "uint"): "floatBitsToUint",
            ("int", "float"): "intBitsToFloat",
            ("int", "uint"): "asuint",
            ("uint", "int"): "asint",
            ("uint", "float"): "uintBitsToFloat",
        }.get((source_family, result_family))

    def spirv_assembly_undef_expression(self, result_type_id, types):
        type_name = self.spirv_type_name(result_type_id, types)
        fallback_type = self.spirv_fallback_identifier(
            type_name or result_type_id, "type"
        )
        return FunctionCallNode(f"spirvUndef_{fallback_type}", [])

    def spirv_is_undef_expression(self, value):
        return (
            isinstance(value, FunctionCallNode)
            and value.name.startswith("spirvUndef_")
            and not value.args
        )

    def spirv_assembly_null_expression(self, result_type_id, types):
        type_name = self.spirv_type_name(result_type_id, types)
        fallback_type = self.spirv_fallback_identifier(
            type_name or result_type_id, "type"
        )
        return FunctionCallNode(f"spirvNull_{fallback_type}", [])

    def spirv_is_null_expression(self, value):
        return (
            isinstance(value, FunctionCallNode)
            and value.name.startswith("spirvNull_")
            and not value.args
        )

    def spirv_is_generated_constant_expression(self, value):
        return self.spirv_is_undef_expression(value) or self.spirv_is_null_expression(
            value
        )

    def spirv_assembly_composite_insert_expression(
        self,
        result_type_id,
        object_operand,
        composite_operand,
        index_operands,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        inserted = self.spirv_assembly_operand_expression(
            object_operand, expressions, names, decorations, constants
        )
        composite = self.spirv_assembly_operand_expression(
            composite_operand, expressions, names, decorations, constants
        )
        result_type_name = self.spirv_type_name(result_type_id, types)
        result_component_count = self.spirv_vector_component_count(
            result_type_id, types
        )

        if len(index_operands) == 1 and result_component_count is not None:
            component_index = self.spirv_vector_shuffle_component_index(
                index_operands[0]
            )
            if (
                component_index is not None
                and 0 <= component_index < result_component_count
            ):
                return FunctionCallNode(
                    result_type_name or result_type_id,
                    [
                        (
                            inserted
                            if index == component_index
                            else self.spirv_composite_component_expression(
                                composite,
                                index,
                                result_component_count,
                                result_type_name,
                            )
                        )
                        for index in range(result_component_count)
                    ],
                )

        return FunctionCallNode(
            "spirvCompositeInsert",
            [composite, inserted, *index_operands],
        )

    def spirv_assembly_vector_insert_dynamic_expression(
        self,
        result_type_id,
        vector_operand,
        component_operand,
        index_operand,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        vector = self.spirv_assembly_operand_expression(
            vector_operand, expressions, names, decorations, constants
        )
        component = self.spirv_assembly_operand_expression(
            component_operand, expressions, names, decorations, constants
        )
        dynamic_index = self.spirv_assembly_operand_expression(
            index_operand, expressions, names, decorations, constants
        )
        result_type_name = self.spirv_type_name(result_type_id, types)
        result_component_count = self.spirv_vector_component_count(
            result_type_id, types
        )
        component_index = self.spirv_integer_constant_operand(index_operand, constants)

        if (
            component_index is not None
            and result_component_count is not None
            and 0 <= component_index < result_component_count
        ):
            return FunctionCallNode(
                result_type_name or result_type_id,
                [
                    (
                        component
                        if component_position == component_index
                        else self.spirv_composite_component_expression(
                            vector,
                            component_position,
                            result_component_count,
                            result_type_name,
                        )
                    )
                    for component_position in range(result_component_count)
                ],
            )

        return FunctionCallNode(
            "spirvVectorInsertDynamic", [vector, component, dynamic_index]
        )

    def spirv_composite_component_expression(
        self, composite, component, component_count, result_type_name
    ):
        if (
            isinstance(composite, FunctionCallNode)
            and composite.name == result_type_name
            and component < len(composite.args)
        ):
            return composite.args[component]

        if component_count <= 4:
            return MemberAccessNode(
                composite, self.spirv_vector_component_name(component)
            )
        return ArrayAccessNode(composite, str(component))

    def spirv_assembly_image_operands(
        self, operands, expressions, names, decorations, constants
    ):
        parsed = {}
        index = 0
        operand_counts = {
            "Bias": 1,
            "ConstOffset": 1,
            "ConstOffsets": 1,
            "Grad": 2,
            "Lod": 1,
            "MakeTexelAvailable": 1,
            "MakeTexelAvailableKHR": 1,
            "MakeTexelVisible": 1,
            "MakeTexelVisibleKHR": 1,
            "MinLod": 1,
            "Offset": 1,
            "Sample": 1,
        }
        while index < len(operands):
            operand = operands[index]
            operand_names = str(operand).split("|")
            if any(name in operand_counts for name in operand_names):
                index += 1
                for name in operand_names:
                    count = operand_counts.get(name, 0)
                    if count == 0:
                        continue
                    values = [
                        self.spirv_assembly_operand_expression(
                            value,
                            expressions,
                            names,
                            decorations,
                            constants,
                        )
                        for value in operands[index : index + count]
                    ]
                    parsed[name] = values
                    index += count
            else:
                index += 1
        return parsed

    def spirv_ext_inst_function_name(self, instruction_set, instruction):
        if instruction_set == "NonSemantic.DebugPrintf" and str(instruction) in {
            "1",
            "DebugPrintf",
        }:
            return "printf"
        if instruction_set == "GLSL.std.450":
            instruction_name = self.SPIRV_GLSL_STD_450_EXT_INST_IDS.get(
                str(instruction), instruction
            )
            mapped_name = self.SPIRV_GLSL_STD_450_EXT_INST_FUNCTIONS.get(
                instruction_name
            )
            if mapped_name is not None:
                return mapped_name
        if not instruction:
            return "spirv_ext_inst"
        sanitized_set = self.spirv_fallback_identifier(instruction_set, "ext_inst")
        sanitized_instruction = self.spirv_fallback_identifier(
            instruction, "instruction"
        )
        return f"spirv_{sanitized_set}_{sanitized_instruction}"
