"""Operand expressions, value names, storage classes, resource block access and function signatures in the SPIR-V parser."""

from ..import_exact_lookups import keyword_safe_identifier
from ..VulkanAst import (
    FunctionCallNode,
    VariableNode,
)


class VulkanOperandMixin:
    """Operand members of ``VulkanParser``."""

    def spirv_assembly_operand_expression(
        self,
        operand,
        expressions,
        names,
        decorations,
        constants,
        variables_by_id=None,
        types=None,
    ):
        if operand in expressions:
            return self.spirv_maybe_non_uniform_expression(
                operand, expressions[operand], decorations
            )

        if operand in constants:
            expression = self.spirv_constant_operand_expression(
                operand, names, constants
            )
            return self.spirv_maybe_non_uniform_expression(
                operand, expression, decorations
            )

        if isinstance(operand, str) and operand.startswith("%"):
            expression = VariableNode(
                "",
                self.spirv_assembly_value_name(
                    operand,
                    names,
                    decorations,
                    storage_class=self.spirv_assembly_variable_storage_class(
                        operand, variables_by_id, types
                    ),
                ),
                spirv_id=operand,
            )
            return self.spirv_maybe_non_uniform_expression(
                operand, expression, decorations
            )

        return operand

    def spirv_maybe_non_uniform_expression(self, operand, expression, decorations):
        if not isinstance(operand, str):
            return expression
        if not self.spirv_has_decoration(decorations.get(operand, []), "NonUniform"):
            return expression
        if (
            isinstance(expression, FunctionCallNode)
            and expression.name == "nonuniformEXT"
        ):
            return expression
        return FunctionCallNode("nonuniformEXT", [expression])

    def spirv_assembly_value_name(
        self,
        value_id,
        names,
        decorations,
        prefix="value",
        storage_class=None,
    ):
        qualifiers = self.spirv_layout_qualifiers(decorations.get(value_id, []))
        builtin_name = self.spirv_builtin_variable_name_from_qualifiers(
            qualifiers, storage_class=storage_class
        )
        if builtin_name:
            return builtin_name
        if value_id in names and names[value_id]:
            return names[value_id]
        fallback = self.spirv_fallback_identifier(value_id, prefix)
        return keyword_safe_identifier(self.SPIRV_VALUE_FALLBACK_RENAMES, fallback)

    def spirv_assembly_variable_storage_class(self, value_id, variables_by_id, types):
        if not variables_by_id or not types:
            return None
        variable = variables_by_id.get(value_id)
        if variable is None:
            return None
        pointer_type = types.get(variable.get("pointer_type_id"), {})
        return variable.get("storage_class") or pointer_type.get("storage_class")

    def spirv_is_flattened_resource_block_access(
        self, storage_class, struct_type_id, decorations
    ):
        struct_decorations = decorations.get(struct_type_id, [])
        if not self.spirv_has_decoration(struct_decorations, "Block"):
            return False
        if storage_class == "PushConstant":
            return True
        if storage_class == "Uniform" and not self.spirv_has_decoration(
            struct_decorations, "BufferBlock"
        ):
            return True
        return False

    def spirv_is_structured_buffer_block_access(
        self, storage_class, struct_type_id, decorations
    ):
        struct_decorations = decorations.get(struct_type_id, [])
        if storage_class == "StorageBuffer":
            return self.spirv_has_decoration(struct_decorations, "Block")
        if storage_class == "Uniform":
            return self.spirv_has_decoration(struct_decorations, "BufferBlock")
        return False

    def spirv_assembly_function_return_type_name(self, type_id, types, constants):
        if type_id is None:
            return "void"
        signature_type = self.spirv_function_signature_type_name(
            type_id, types, constants
        )
        if signature_type is not None:
            return signature_type
        return (
            self.spirv_type_name(type_id, types)
            or self.spirv_fallback_identifier(type_id, "return_type")
            or "void"
        )

    def spirv_function_parameter_type_name(self, type_id, types, constants):
        signature_type = self.spirv_function_signature_type_name(
            type_id, types, constants
        )
        if signature_type is not None:
            return signature_type
        return self.spirv_type_name(type_id, types) or self.spirv_fallback_identifier(
            type_id, "param_type"
        )

    def spirv_function_signature_type_name(self, type_id, types, constants):
        type_info = types.get(type_id, {})
        if type_info.get("kind") == "pointer":
            return self.spirv_function_signature_type_name(
                type_info.get("type_id"), types, constants
            )
        base_type, suffix = self.spirv_type_name_and_suffix(type_id, types, constants)
        if base_type is None:
            return None
        return f"{base_type}{suffix}"

    def spirv_written_pointer_parameter_ids(
        self, parameter_records, raw_instructions, types
    ):
        alias_origins = {
            parameter_id: parameter_id
            for parameter_id, parameter_type_id in parameter_records
            if parameter_id
            and types.get(parameter_type_id, {}).get("kind") == "pointer"
        }
        written_parameter_ids = set()

        for result_id, opcode, operands, _line_number in raw_instructions:
            if (
                result_id
                and opcode in {
                    "OpAccessChain",
                    "OpInBoundsAccessChain",
                    "OpPtrAccessChain",
                    "OpInBoundsPtrAccessChain",
                }
                and len(operands) >= 2
                and operands[1] in alias_origins
            ):
                alias_origins[result_id] = alias_origins[operands[1]]
                continue

            if (
                result_id
                and opcode == "OpCopyObject"
                and len(operands) >= 2
                and operands[1] in alias_origins
            ):
                alias_origins[result_id] = alias_origins[operands[1]]
                continue

            if (
                opcode in {
                    "OpStore",
                    "OpCopyMemory",
                    "OpCopyMemorySized",
                    "OpAtomicStore",
                }
                and operands
            ):
                written_parameter_id = alias_origins.get(operands[0])
                if written_parameter_id:
                    written_parameter_ids.add(written_parameter_id)

        return written_parameter_ids

    def spirv_function_parameter_qualifiers(self, type_id, types, is_written):
        type_info = types.get(type_id, {})
        if is_written and type_info.get("kind") == "pointer":
            return ["inout"]
        return []
