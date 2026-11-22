"""Expression materialization, access chains, composite extraction and construction, and struct member access in the SPIR-V parser."""

from ...common_ast import InitializerListNode
from ..access_chains import (
    chain_base_pointee,
    walk_indices,
)
from ..block_arrays import block_array_element
from ..VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    MethodCallNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)


class VulkanAccessChainMixin:
    """Access chain members of ``VulkanParser``."""

    def spirv_assembly_used_result_ids(self, raw_instructions):
        return {
            operand
            for _result_id, _opcode, operands, _line_number in raw_instructions
            for operand in operands
            if isinstance(operand, str) and operand.startswith("%")
        }

    def spirv_maybe_materialize_assembly_expression(
        self,
        result_id,
        result_type_id,
        expression,
        statements,
        names,
        decorations,
        types,
        constants,
    ):
        if not result_id or not result_type_id:
            return expression
        if (
            self.spirv_expression_complexity(
                expression, self.MAX_SPIRV_MATERIALIZED_EXPRESSION_COMPLEXITY
            )
            <= self.MAX_SPIRV_MATERIALIZED_EXPRESSION_COMPLEXITY
        ):
            return expression

        type_info = types.get(result_type_id, {})
        if type_info.get("kind") in {"pointer", "function"}:
            return expression
        variable_type, array_suffix = self.spirv_type_name_and_suffix(
            result_type_id, types, constants, names=names
        )
        if not variable_type or variable_type == "void":
            return expression

        variable_name = self.spirv_assembly_value_name(
            result_id, names, decorations, prefix="value"
        )
        statements.append(
            AssignmentNode(
                VariableNode(
                    variable_type,
                    f"{variable_name}{array_suffix}",
                    spirv_id=result_id,
                    spirv_type_id=result_type_id,
                ),
                expression,
            )
        )
        return VariableNode("", variable_name, spirv_id=result_id)

    def spirv_expression_complexity(self, expression, limit):
        budget = limit + 1

        def count(node):
            nonlocal budget
            if budget <= 0:
                return 0
            budget -= 1
            if node is None or isinstance(node, (str, int, float, VariableNode)):
                return 1
            if isinstance(node, AssignmentNode):
                return 1 + count(node.left) + count(node.right)
            if isinstance(node, BinaryOpNode):
                return 1 + count(node.left) + count(node.right)
            if isinstance(node, UnaryOpNode):
                return 1 + count(node.operand)
            if isinstance(node, TernaryOpNode):
                return (
                    1
                    + count(node.condition)
                    + count(node.true_expr)
                    + count(node.false_expr)
                )
            if isinstance(node, FunctionCallNode):
                return 1 + sum(count(arg) for arg in node.args)
            if isinstance(node, MethodCallNode):
                return 1 + count(node.object) + sum(count(arg) for arg in node.args)
            if isinstance(node, MemberAccessNode):
                return 1 + count(node.object)
            if isinstance(node, ArrayAccessNode):
                return 1 + count(node.array) + count(node.index)
            if isinstance(node, InitializerListNode):
                return 1 + sum(count(element) for element in node.elements)
            return 1

        return count(expression)

    def spirv_assembly_access_chain_expression(
        self,
        base_operand,
        index_operands,
        expressions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables_by_id,
        constants,
        expression_type_ids=None,
    ):
        struct_member = self.spirv_assembly_struct_member_access_expression(
            base_operand,
            index_operands,
            expressions,
            names,
            decorations,
            member_decorations,
            member_names,
            types,
            variables_by_id,
            constants,
        )
        if struct_member is not None:
            return struct_member

        access = self.spirv_assembly_operand_expression(
            base_operand,
            expressions,
            names,
            decorations,
            constants,
            variables_by_id=variables_by_id,
            types=types,
        )
        pointee = (
            chain_base_pointee(
                base_operand, types, variables_by_id, expression_type_ids
            )
            if expression_type_ids is not None
            else None
        )
        return walk_indices(
            self,
            access,
            pointee,
            index_operands,
            lambda operand: self.spirv_assembly_operand_expression(
                operand,
                expressions,
                names,
                decorations,
                constants,
                variables_by_id=variables_by_id,
                types=types,
            ),
            member_names,
            types,
            constants,
        )

    def spirv_assembly_composite_construct_expression(
        self,
        type_id,
        constituent_ids,
        expressions,
        names,
        decorations,
        constants,
        types,
    ):
        args = [
            self.spirv_assembly_operand_expression(
                operand,
                expressions,
                names,
                decorations,
                constants,
            )
            for operand in constituent_ids
        ]
        if types.get(type_id, {}).get("kind") in {"array", "runtime_array"}:
            return InitializerListNode(args)
        return FunctionCallNode(self.spirv_type_name(type_id, types) or type_id, args)

    def spirv_sparse_image_result_extract_expression(self, expression, member_index):
        if not (
            isinstance(expression, FunctionCallNode)
            and expression.name.startswith(
                (
                    "spirvImageSparseSample",
                    "spirvImageSparseRead",
                    "spirvImageSparseFetch",
                    "spirvImageSparseGather",
                    "spirvImageSparseDrefGather",
                )
            )
        ):
            return None

        if member_index == 0:
            return FunctionCallNode("spirvSparseResidencyCode", [expression])
        if member_index == 1:
            return FunctionCallNode("spirvSparseTexel", [expression])
        return None

    def spirv_composite_index_type_id(self, type_id, index_operand, types):
        type_info = types.get(type_id, {})
        kind = type_info.get("kind")
        if kind in {"array", "runtime_array"}:
            return type_info.get("element_type")
        if kind == "matrix":
            return type_info.get("column_type")
        if kind == "vector":
            return type_info.get("component_type")
        if kind == "struct":
            member_index = self.spirv_integer_constant_operand(index_operand, {})
            member_types = type_info.get("member_types", [])
            if member_index is not None and 0 <= member_index < len(member_types):
                return member_types[member_index]
        return None

    def spirv_assembly_typed_access_chain(
        self,
        base_operand,
        type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """``base.member[i].field``: an access chain walked through its pointee type, struct indices as names."""
        access = self.spirv_assembly_operand_expression(
            base_operand, expressions, names, decorations, constants
        )
        return walk_indices(
            self,
            access,
            type_id,
            index_operands,
            lambda operand: self.spirv_assembly_operand_expression(
                operand, expressions, names, decorations, constants
            ),
            member_names,
            types,
            constants,
        )

    def spirv_assembly_member_tail(
        self,
        access,
        struct_type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """Continue ``access`` (member ``index_operands[0]`` of ``struct_type_id``) through the remaining indices."""
        member_index = self.spirv_integer_constant_operand(index_operands[0], constants)
        member_type_id = types.get(struct_type_id, {}).get("member_types", [])[
            member_index
        ]
        return walk_indices(
            self,
            access,
            member_type_id,
            index_operands[1:],
            lambda operand: self.spirv_assembly_operand_expression(
                operand, expressions, names, decorations, constants
            ),
            member_names,
            types,
            constants,
        )

    def spirv_assembly_storage_block_array_access(
        self,
        base_operand,
        struct_type_id,
        index_operands,
        expressions,
        names,
        decorations,
        member_names,
        types,
        constants,
    ):
        """``v[k][0].m...`` for an access chain into element ``k`` of an array of storage blocks."""
        member_index = self.spirv_integer_constant_operand(index_operands[1], constants)
        member_types = types.get(struct_type_id, {}).get("member_types", [])
        if member_index is None or not 0 <= member_index < len(member_types):
            return None

        def operand(value):
            return self.spirv_assembly_operand_expression(
                value, expressions, names, decorations, constants
            )

        element = ArrayAccessNode(operand(base_operand), operand(index_operands[0]))
        member_name = member_names.get(struct_type_id, {}).get(
            str(member_index), f"member{member_index}"
        )
        access = MemberAccessNode(ArrayAccessNode(element, 0), member_name)
        return self.spirv_assembly_member_tail(
            access,
            struct_type_id,
            index_operands[1:],
            expressions,
            names,
            decorations,
            member_names,
            types,
            constants,
        )

    def spirv_assembly_struct_member_access_expression(
        self,
        base_operand,
        index_operands,
        expressions,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        variables_by_id,
        constants,
    ):
        if not index_operands:
            return None

        variable = variables_by_id.get(base_operand)
        if variable is None:
            return None

        pointer_type = types.get(variable["pointer_type_id"], {})
        if pointer_type.get("kind") != "pointer":
            return None

        storage_class = variable["storage_class"] or pointer_type.get("storage_class")
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if (
            storage_class in self.SPIRV_INTERFACE_STORAGE_CLASSES
            and struct_type.get("kind") in {"array", "runtime_array"}
            and types.get(struct_type.get("element_type"), {}).get("kind") == "struct"
            and len(index_operands) >= 2
        ):
            # Per-vertex interface arrays (gl_in, gl_MeshVerticesEXT, mesh outputs): element, then member names.
            return self.spirv_assembly_typed_access_chain(
                base_operand,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        array_element = block_array_element(struct_type_id, types)
        if (
            array_element is not None
            and len(index_operands) >= 2
            and self.spirv_is_structured_buffer_block_access(
                storage_class, array_element[0], decorations
            )
        ):
            return self.spirv_assembly_storage_block_array_access(
                base_operand,
                array_element[0],
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        if struct_type.get("kind") != "struct":
            return None

        member_index = self.spirv_integer_constant_operand(index_operands[0], constants)
        if member_index is None:
            return None

        member_types = struct_type.get("member_types", [])
        if not 0 <= member_index < len(member_types):
            return None

        member_key = str(member_index)
        if storage_class in self.SPIRV_RAY_STORAGE_QUALIFIERS:
            return self.spirv_assembly_typed_access_chain(
                base_operand,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )
        if storage_class == "Function":
            member_name = member_names.get(struct_type_id, {}).get(
                member_key, f"member_{member_key}"
            )
            access = MemberAccessNode(
                self.spirv_assembly_operand_expression(
                    base_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                member_name,
            )
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if self.spirv_is_flattened_resource_block_access(
            storage_class, struct_type_id, decorations
        ):
            member_name = variable.get("flattened_member_aliases", {}).get(
                member_key
            ) or member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            access = VariableNode("", member_name)
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if self.spirv_is_structured_buffer_block_access(
            storage_class, struct_type_id, decorations
        ):
            member_name = member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            access = MemberAccessNode(
                ArrayAccessNode(
                    self.spirv_assembly_operand_expression(
                        base_operand,
                        expressions,
                        names,
                        decorations,
                        constants,
                    ),
                    0,
                ),
                member_name,
            )
            return self.spirv_assembly_member_tail(
                access,
                struct_type_id,
                index_operands,
                expressions,
                names,
                decorations,
                member_names,
                types,
                constants,
            )

        if storage_class not in self.SPIRV_INTERFACE_STORAGE_CLASSES:
            return None

        member_layout_decorations = [
            (decoration, operands)
            for member, decoration, operands in member_decorations.get(
                struct_type_id, []
            )
            if member == member_key
        ]
        qualifiers = self.spirv_layout_qualifiers(member_layout_decorations)
        block_name = self.spirv_assembly_value_name(base_operand, names, decorations={})
        member_name = self.spirv_struct_member_variable_name(
            struct_type_id,
            member_key,
            block_name,
            qualifiers,
            member_names,
            storage_class=storage_class,
        )
        access = VariableNode("", member_name)
        return self.spirv_assembly_member_tail(
            access,
            struct_type_id,
            index_operands,
            expressions,
            names,
            decorations,
            member_names,
            types,
            constants,
        )
