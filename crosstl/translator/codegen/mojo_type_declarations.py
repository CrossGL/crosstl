"""Enums, payload enums, structs and constant buffers emitted by the Mojo generator."""

from ..ast import (
    ArrayNode,
    BinaryOpNode,
    MemberAccessNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MojoRefusal
from .array_utils import get_array_size_from_node
from .enum_utils import (
    enum_struct_fields,
    enum_variant_constructor_name,
    enum_variant_payload_fields,
    generic_enum_specialized_variant_fields,
)


class MojoTypeDeclarationMixin:
    """Type-declaration members of ``MojoCodeGen``."""

    def generate_enum(self, node):
        """Lower a CrossGL enum to Mojo aliases or a tagged payload struct."""
        self.validate_user_defined_type_name(node.name, "enum")
        self.enum_variant_names[node.name] = [
            variant.name for variant in getattr(node, "variants", []) or []
        ]
        if any(
            getattr(variant, "data", None) or getattr(variant, "fields", None)
            for variant in getattr(node, "variants", []) or []
        ):
            return self.generate_payload_enum(node)

        enum_type = self.map_enum_underlying_type(
            getattr(node, "underlying_type", None)
        )
        self.enum_types[node.name] = enum_type

        code = f"comptime {node.name} = {enum_type}\n"
        next_value = 0
        local_aliases = {}
        local_values = {}
        for variant in getattr(node, "variants", []) or []:
            payload = getattr(variant, "data", None) or getattr(variant, "fields", None)
            if payload:
                raise MojoRefusal(
                    "Unsupported enum payload for Mojo codegen; only unit/numeric "
                    f"enum variants are supported: {node.name}.{variant.name}"
                )

            alias_name = self.enum_variant_alias_name(node.name, variant.name)
            value = getattr(variant, "value", None)
            if value is None:
                value_text = str(next_value)
                resolved_value = next_value
            else:
                value_text = self.generate_enum_value_expression(value, local_aliases)
                resolved_value = self.evaluate_enum_integer_value(value, local_values)

            self.register_enum_variant_metadata(
                node.name,
                variant.name,
                alias_name,
                resolved_value,
            )
            local_aliases[variant.name] = alias_name
            local_values[variant.name] = resolved_value
            code += f"comptime {alias_name} = {value_text}\n"

            literal_value = resolved_value
            if literal_value is None:
                literal_value = self.literal_int_value(value_text)
            next_value = (
                literal_value + 1 if literal_value is not None else next_value + 1
            )

        return code + "\n"

    def generate_payload_enum(self, node):
        fields = enum_struct_fields(node)
        if fields is None:
            raise MojoRefusal(
                "Unsupported enum payload for Mojo codegen; payload fields must "
                f"have stable names and compatible types: {node.name}"
            )

        self.enum_types[node.name] = node.name
        self.struct_types[node.name] = {"variant": "int"}
        for field_name, field_type in fields:
            self.struct_types[node.name][field_name] = self.type_name(field_type)

        code = f"@value\nstruct {node.name}:\n"
        code += "    var variant: Int32\n"
        for field_name, field_type in fields:
            code += (
                f"    var {self.mojo_identifier(field_name)}: "
                f"{self.map_type(field_type)}\n"
            )
        code += "\n"

        code += self.generate_payload_enum_constants(node)
        code += self.generate_payload_enum_constructors(node, fields)
        return code

    def generate_payload_enum_constants(self, node):
        code = ""
        next_value = 0
        local_aliases = {}
        local_values = {}
        for variant in getattr(node, "variants", []) or []:
            alias_name = self.enum_variant_alias_name(node.name, variant.name)
            value = getattr(variant, "value", None)
            if value is None:
                value_text = str(next_value)
                resolved_value = next_value
            else:
                value_text = self.generate_enum_value_expression(value, local_aliases)
                resolved_value = self.evaluate_enum_integer_value(value, local_values)

            self.register_enum_variant_metadata(
                node.name,
                variant.name,
                alias_name,
                resolved_value,
            )
            local_aliases[variant.name] = alias_name
            local_values[variant.name] = resolved_value
            code += f"comptime {alias_name} = {value_text}\n"

            literal_value = resolved_value
            if literal_value is None:
                literal_value = self.literal_int_value(value_text)
            next_value = (
                literal_value + 1 if literal_value is not None else next_value + 1
            )
        return code + "\n"

    def register_enum_variant_metadata(
        self, enum_name, variant_name, alias_name, resolved_value
    ):
        for path in self.enum_variant_paths(enum_name, variant_name):
            self.enum_variant_aliases[path] = alias_name
            self.enum_variant_values[path] = resolved_value
            self.enum_variant_result_types[path] = enum_name

    def enum_variant_paths(self, enum_name, variant_name):
        return (f"{enum_name}::{variant_name}", f"{enum_name}.{variant_name}")

    def generate_payload_enum_constructors(self, node, fields):
        code = ""
        all_field_names = {field_name for field_name, _ in fields}
        for variant in getattr(node, "variants", []) or []:
            variant_fields = enum_variant_payload_fields(variant) or []
            constructor_name = (
                f"{self.enum_variant_alias_name(node.name, variant.name)}_make"
            )
            for path in self.enum_variant_paths(node.name, variant.name):
                self.enum_variant_constructors[path] = constructor_name
                self.enum_variant_constructor_fields[path] = variant_fields
                if not variant_fields:
                    self.enum_unit_variant_constructors[path] = constructor_name

            params = ", ".join(
                f"payload{index}: {self.map_type(field_type)}"
                for index, (_field_name, field_type) in enumerate(variant_fields)
            )
            code += f"fn {constructor_name}({params}) -> {node.name}:\n"
            rendered_fields = {
                "variant": self.enum_variant_alias_name(node.name, variant.name)
            }
            active_fields = {field_name for field_name, _ in variant_fields}
            for field_name, field_type in fields:
                if field_name in active_fields:
                    continue
                rendered_fields[field_name] = self.zero_value_for_type(field_type)
            for index, (field_name, _field_type) in enumerate(variant_fields):
                if field_name not in all_field_names:
                    continue
                rendered_fields[field_name] = f"payload{index}"
            field_args = [
                f"{self.mojo_identifier(field_name)}={rendered_fields[field_name]}"
                for field_name in self.struct_types[node.name]
            ]
            code += f"    return {node.name}({', '.join(field_args)})\n\n"
        return code

    def generate_enum_value_expression(self, value, local_aliases):
        previous_aliases = self.current_enum_value_aliases
        self.current_enum_value_aliases = local_aliases
        try:
            return self.generate_expression(value)
        finally:
            self.current_enum_value_aliases = previous_aliases

    def evaluate_enum_integer_value(self, expr, local_values):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return int(expr)
        if isinstance(expr, int):
            return expr
        if hasattr(expr, "value"):
            try:
                return int(expr.value)
            except (TypeError, ValueError):
                return None
        expr_name = getattr(expr, "name", None)
        if expr_name is not None and not isinstance(expr, VariableNode):
            return local_values.get(expr_name)
        if isinstance(expr, MemberAccessNode):
            reference = self.enum_member_reference_name(expr)
            if reference is None:
                return None
            return self.enum_variant_values.get(reference)
        if isinstance(expr, UnaryOpNode):
            operand = self.evaluate_enum_integer_value(expr.operand, local_values)
            if operand is None:
                return None
            op = self.map_operator(expr.op)
            if op == "+":
                return operand
            if op == "-":
                return -operand
            if op == "~":
                return ~operand
            return None
        if isinstance(expr, BinaryOpNode):
            left = self.evaluate_enum_integer_value(expr.left, local_values)
            right = self.evaluate_enum_integer_value(expr.right, local_values)
            if left is None or right is None:
                return None
            op = self.map_operator(expr.op)
            try:
                return self.evaluate_enum_binary_op(left, op, right)
            except ZeroDivisionError:
                return None
        return None

    def evaluate_enum_binary_op(self, left, op, right):
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left // right
        if op == "%":
            return left % right
        if op == "<<":
            return left << right
        if op == ">>":
            return left >> right
        if op == "|":
            return left | right
        if op == "&":
            return left & right
        if op == "^":
            return left ^ right
        return None

    def enum_member_reference_name(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        obj = getattr(expr, "object", None)
        obj_name = getattr(obj, "name", obj if isinstance(obj, str) else None)
        if obj_name is None:
            return None
        return f"{obj_name}.{expr.member}"

    def map_enum_underlying_type(self, underlying_type):
        if underlying_type is None:
            return "Int32"
        mapped = self.map_type(self.convert_type_node_to_string(underlying_type))
        if mapped in {"Int", "Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64"}:
            return mapped
        return "Int32"

    def enum_variant_alias_name(self, enum_name, variant_name):
        return f"{enum_name}_{variant_name}"

    def map_enum_variant_reference(self, name, target_type=None):
        if not isinstance(name, str):
            return name
        constructor = self.enum_unit_variant_constructors.get(name)
        if constructor is not None:
            return f"{constructor}()"
        generic_constructor = self.generic_enum_unit_variant_constructor(
            name, target_type
        )
        if generic_constructor is not None:
            return f"{generic_constructor}()"
        if name in self.current_enum_value_aliases:
            return self.current_enum_value_aliases[name]
        if name in self.enum_variant_aliases:
            return self.enum_variant_aliases[name]
        # Not an enum variant of the program: an identifier, written as declared.
        return name

    def generic_enum_unit_variant_constructor(self, name, target_type):
        variant_fields = self.generic_enum_variant_fields_for_type(name, target_type)
        if variant_fields != []:
            return None
        specialization, variant_name = self.generic_enum_variant_specialization(
            name, target_type
        )
        if specialization is None:
            return None
        return enum_variant_constructor_name(
            specialization["struct_name"], variant_name
        )

    def generic_enum_variant_specialization(self, variant_path, subject_type):
        if "::" in str(variant_path):
            enum_name, variant_name = str(variant_path).split("::", 1)
        elif "." in str(variant_path):
            enum_name, variant_name = str(variant_path).split(".", 1)
        else:
            return None, None

        specialization = self.generic_enum_specialization_for_type(
            subject_type,
            expected_base=enum_name,
        )
        return specialization, variant_name

    def generic_enum_variant_fields_for_type(self, variant_path, subject_type):
        specialization, variant_name = self.generic_enum_variant_specialization(
            variant_path, subject_type
        )
        if specialization is None:
            if "::" in str(variant_path):
                enum_name, variant_name = str(variant_path).split("::", 1)
            elif "." in str(variant_path):
                enum_name, variant_name = str(variant_path).split(".", 1)
            else:
                return None
            specialization = self.generic_enum_pattern_specialization_for_type(
                subject_type,
                expected_base=enum_name,
            )
            if specialization is None:
                return None
        return generic_enum_specialized_variant_fields(
            self,
            specialization,
            variant_name,
        )

    def struct_member_declared_type(self, member):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            element_type = self.explicit_image_format_resource_type(
                self.type_name(element_type), member
            )
            return self.array_type_name(element_type, get_array_size_from_node(member))
        if hasattr(member, "member_type"):
            return self.explicit_image_format_resource_type(
                self.convert_type_node_to_string(member.member_type), member
            )
        if hasattr(member, "vtype"):
            return self.explicit_image_format_resource_type(member.vtype, member)
        return "float"

    def array_declared_type(self, node):
        element_type = getattr(node, "element_type", getattr(node, "vtype", "float"))
        element_type = self.explicit_image_format_resource_type(
            self.type_name(element_type), node
        )
        return self.array_type_name(element_type, get_array_size_from_node(node))

    def register_struct_type_metadata(self, node):
        struct_name = getattr(node, "name", None)
        if not struct_name:
            return
        self.validate_user_defined_type_name(struct_name, "struct")
        self.register_generic_struct_type_metadata(node)
        self.struct_types[struct_name] = {}
        self.struct_member_semantics[struct_name] = {}

        for member in getattr(node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            member_type = self.struct_member_declared_type(member)
            self.struct_types[struct_name][member_name] = member_type
            self.register_struct_resource_access_metadata(
                struct_name, member_name, member, member_type
            )
            semantic = self.node_semantic(member)
            if semantic:
                self.struct_member_semantics[struct_name][member_name] = semantic
                self.validate_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )

    def generate_struct(self, node):
        generic_params = [
            param.name
            for param in getattr(node, "generic_params", []) or []
            if getattr(param, "name", None)
        ]
        generic_suffix = ""
        if generic_params:
            generic_suffix = (
                "[" + ", ".join(f"{param}: Copyable" for param in generic_params) + "]"
            )
        code = f"@value\nstruct {node.name}{generic_suffix}:\n"
        self.register_struct_type_metadata(node)

        members = getattr(node, "members", [])
        for member in members:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            member_type = self.struct_types[node.name][member_name]
            semantic = self.struct_member_semantics.get(node.name, {}).get(member_name)
            semantic_comment = f"  # {self.map_semantic(semantic)}" if semantic else ""
            if self.is_array_type_name(member_type):
                element_type, size = self.parse_array_type_name(member_type)
                code += (
                    f"    var {self.mojo_identifier(member_name)}: "
                    f"{self.array_storage_type(element_type, size)}"
                    f"{semantic_comment}\n"
                )
            else:
                code += (
                    f"    var {self.mojo_identifier(member_name)}: "
                    f"{self.map_type(member_type)}{semantic_comment}\n"
                )

        code += "\n"
        return code

    def get_cbuffer_nodes(self, ast):
        nodes = []
        seen = set()
        for attr in ("cbuffers", "constants"):
            for node in getattr(ast, attr, None) or []:
                if attr == "constants" and not getattr(node, "is_cbuffer", False):
                    continue
                node_id = id(node)
                if node_id in seen:
                    continue
                nodes.append(node)
                seen.add(node_id)
        return nodes

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = self.get_cbuffer_nodes(ast)
        emitted_member_names = set()
        for node in cbuffers:
            code += self.generate_resource_metadata_comment(
                node, getattr(node, "name", None), kind="cbuffer"
            )
            if isinstance(node, StructNode):
                code += self.generate_struct(node)
                code += self.generate_cbuffer_member_placeholders(
                    node, emitted_member_names
                )
        return code

    def generate_cbuffer_member_placeholders(self, node, emitted_member_names=None):
        emitted_member_names = (
            emitted_member_names if emitted_member_names is not None else set()
        )
        code = ""
        members = getattr(node, "members", []) or []
        if not members:
            return code

        for member in members:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in emitted_member_names:
                continue
            member_type = self.struct_types.get(node.name, {}).get(member_name)
            if member_type is None:
                member_type = self.struct_member_declared_type(member)
            self.register_variable_type(member_name, member_type)
            emitted_member_names.add(member_name)
            code += self.generate_zero_initialized_variable(member_name, member_type)
        return code

    def generate_zero_initialized_variable(self, name, type_name):
        emitted_name = self.mojo_identifier(name)
        if self.is_array_type_name(type_name):
            return (
                f"var {emitted_name} = "
                f"{self.array_initial_value_for_type(type_name)}\n"
            )
        if self.is_struct_type_name(type_name):
            return f"var {emitted_name} = {self.zero_value_for_type(type_name)}\n"
        mapped_type = self.map_type(type_name)
        return (
            f"var {emitted_name}: {mapped_type} = "
            f"{self.zero_value_for_type(type_name)}\n"
        )
