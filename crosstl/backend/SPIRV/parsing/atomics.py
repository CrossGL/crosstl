"""Atomics, geometry emission, ray queries, group non-uniform reductions and vector shuffles in the SPIR-V parser."""

from ..VulkanAst import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
    UnaryOpNode,
)


class VulkanAtomicMixin:
    """Atomic members of ``VulkanParser``."""

    def spirv_assembly_atomic_expression(
        self,
        opcode,
        pointer_operand,
        value_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        value = self.spirv_assembly_operand_expression(
            value_operand,
            expressions,
            names,
            decorations,
            constants,
        )
        if opcode == "OpAtomicISub":
            value = UnaryOpNode("-", value)

        return FunctionCallNode(
            self.SPIRV_ATOMIC_RMW_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                value,
            ],
        )

    def spirv_assembly_atomic_increment_expression(
        self,
        opcode,
        pointer_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        value = "1"
        if opcode == "OpAtomicIDecrement":
            value = UnaryOpNode("-", value)

        return FunctionCallNode(
            "atomicAdd",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                value,
            ],
        )

    def spirv_assembly_atomic_load_expression(
        self,
        pointer_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicLoad",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            ],
        )

    def spirv_assembly_atomic_store_statement(
        self,
        pointer_operand,
        value_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicStore",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    value_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_geometry_emit_statement(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_GEOMETRY_EMIT_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands
            ],
        )

    def spirv_assembly_ray_query_statement(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_RAY_QUERY_STATEMENT_FUNCTIONS[opcode],
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in operands
            ],
        )

    def spirv_assembly_ray_query_expression(
        self,
        opcode,
        operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            self.SPIRV_RAY_QUERY_EXPRESSION_FUNCTIONS[opcode],
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
        )

    def spirv_assembly_atomic_compare_exchange_expression(
        self,
        pointer_operand,
        value_operand,
        comparator_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "atomicCompSwap",
            [
                self.spirv_assembly_operand_expression(
                    pointer_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    comparator_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    value_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_group_non_uniform_reduction_expression(
        self,
        opcode,
        group_operation,
        value_operand,
        extra_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        operation_name = self.SPIRV_GROUP_NON_UNIFORM_REDUCTION_FUNCTIONS[opcode]
        prefix = self.SPIRV_GROUP_NON_UNIFORM_SCAN_PREFIXES.get(group_operation)
        if prefix is None:
            sanitized_group_operation = self.spirv_fallback_identifier(
                group_operation, "group_operation"
            )
            prefix = f"spirvGroupNonUniform{sanitized_group_operation}"

        function_name = f"{prefix}{operation_name}"
        args = [
            self.spirv_assembly_operand_expression(
                value_operand,
                expressions,
                names,
                decorations,
                constants,
            )
        ]
        if group_operation == "ClusteredReduce" and extra_operands:
            args.append(
                self.spirv_assembly_operand_expression(
                    extra_operands[0],
                    expressions,
                    names,
                    decorations,
                    constants,
                )
            )
        return FunctionCallNode(function_name, args)

    def spirv_assembly_vector_shuffle_expression(
        self,
        result_type_id,
        vector1_id,
        vector2_id,
        component_operands,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        types,
    ):
        source1_count = self.spirv_vector_component_count(
            expression_type_ids.get(vector1_id), types
        )
        components = [
            self.spirv_vector_shuffle_component_index(component)
            for component in component_operands
        ]

        if (
            vector1_id == vector2_id
            and source1_count is not None
            and all(
                component is not None and 0 <= component < source1_count
                for component in components
            )
        ):
            swizzle = "".join(
                self.spirv_vector_component_name(component) for component in components
            )
            source = self.spirv_assembly_operand_expression(
                vector1_id, expressions, names, decorations, constants
            )
            return MemberAccessNode(source, swizzle)

        result_type = self.spirv_type_name(result_type_id, types) or result_type_id
        args = [
            self.spirv_vector_shuffle_component_expression(
                vector1_id,
                vector2_id,
                component,
                source1_count,
                expressions,
                expression_type_ids,
                names,
                decorations,
                constants,
                types,
            )
            for component in components
        ]
        return FunctionCallNode(result_type, args)

    def spirv_vector_shuffle_component_expression(
        self,
        vector1_id,
        vector2_id,
        component,
        source1_count,
        expressions,
        expression_type_ids,
        names,
        decorations,
        constants,
        types,
    ):
        if component is None:
            return "0"

        if source1_count is not None and component >= source1_count:
            source_id = vector2_id
            component -= source1_count
        else:
            source_id = vector1_id

        source = self.spirv_assembly_operand_expression(
            source_id, expressions, names, decorations, constants
        )
        source_count = self.spirv_vector_component_count(
            expression_type_ids.get(source_id), types
        )
        if source_count is not None and 0 <= component < min(source_count, 4):
            return MemberAccessNode(source, self.spirv_vector_component_name(component))
        return ArrayAccessNode(source, str(component))

    def spirv_vector_component_count(self, type_id, types):
        type_info = types.get(type_id, {})
        component_count = type_info.get("component_count")
        if component_count is None:
            return None
        try:
            return int(component_count, 0)
        except ValueError:
            return None

    def spirv_vector_shuffle_component_index(self, component):
        try:
            index = int(component, 0)
        except (TypeError, ValueError):
            return None
        if index in {-1, 0xFFFFFFFF}:
            return None
        return index

    def spirv_vector_component_name(self, component):
        return "xyzw"[component]
