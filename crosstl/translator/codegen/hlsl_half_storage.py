"""Bit-preserving storage for logical HLSL binary16 values."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from ..resource_storage import BINARY16_STORAGE, resource_storage_header


class DirectXHalfStorageError(ValueError):
    project_diagnostic_code = "project.translate.directx-half-storage-unsupported"
    missing_capabilities = ("directx.half-storage",)

    def __init__(self, message, *, reason, source_type=None, source_location=None):
        super().__init__(message)
        self.reason = reason
        self.source_type = source_type
        self.source_location = source_location


class HLSLHalfStorage:
    """Keep logical types separate from structured-buffer representations."""

    resource_types = {
        "StructuredBuffer",
        "RWStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
    }

    def __init__(self, generator, ast=None):
        self.generator = generator
        self.structures = {}
        self.resources = {}
        self.updates = {}
        self.constant_types = {}
        self.reserved = (
            {
                node.name
                for node in generator.walk_ast(ast)
                if isinstance(getattr(node, "name", None), str)
            }
            if ast is not None
            else set()
        )
        self.visiting = set()
        self.unchanged_structures = set()

    def name(self, stem):
        candidate = stem
        index = 0
        while candidate in self.reserved:
            index += 1
            candidate = f"{stem}_{index}"
        self.reserved.add(candidate)
        return candidate

    def shape(self, value_type):
        mapped = self.generator.map_type(value_type)
        return re.fullmatch(r"float16_t([234]?)", str(mapped))

    def member_type(self, member):
        g = self.generator
        if isinstance(member, ArrayNode):
            return f"{g.type_name_string(member.element_type)}[{g.expression_to_string(member.size)}]"
        return g.type_name_string(
            getattr(
                member,
                "member_type",
                getattr(member, "var_type", getattr(member, "vtype", None)),
            )
        )

    def physical_type(self, value_type):
        g = self.generator
        mapped = g.map_type(value_type)
        match = re.fullmatch(r"float16_t([234]?)", str(mapped))
        if match:
            return "uint16_t" + match.group(1)
        if "[" in str(mapped):
            base, suffix = str(mapped).split("[", 1)
            return self.physical_type(base) + "[" + suffix
        if mapped in g.hlsl_union_layouts:
            return mapped
        if mapped in self.structures:
            return self.structures[mapped]["physical"]
        declaration = getattr(g, "structs_by_name", {}).get(mapped)
        if (
            declaration is None
            or (mapped, id(declaration)) in self.unchanged_structures
        ):
            return mapped
        if mapped in self.visiting:
            raise DirectXHalfStorageError(
                "Recursive value structures have no finite buffer layout",
                reason="recursive-storage-structure",
                source_type=mapped,
            )
        self.visiting.add(mapped)
        try:
            members = [
                (member.name, self.member_type(member))
                for member in declaration.members
                if not g.hlsl_static_struct_member(member)
            ]
            fields = [
                (name, logical, self.physical_type(logical))
                for name, logical in members
            ]
        finally:
            self.visiting.remove(mapped)
        if all(g.map_type(logical) == physical for _, logical, physical in fields):
            self.unchanged_structures.add((mapped, id(declaration)))
            return mapped
        if any(g.hlsl_struct_member_is_resource(logical) for _, logical, _ in fields):
            raise DirectXHalfStorageError(
                "Half buffer structures cannot contain resource handles",
                reason="resource-storage-member-unsupported",
                source_type=mapped,
            )
        record = {
            "physical": self.name(f"__crossgl_half_storage_{mapped}"),
            "decode": self.name(f"__crossgl_half_load_{mapped}"),
            "encode": self.name(f"__crossgl_half_store_{mapped}"),
            "fields": fields,
        }
        self.structures[mapped] = record
        return record["physical"]

    def lowered(self, value_type):
        if value_type is None:
            return False
        return self.physical_type(value_type) != self.generator.map_type(value_type)

    def homogeneous(self, value_type):
        if self.shape(value_type):
            return True
        self.physical_type(value_type)
        record = self.structures.get(self.generator.map_type(value_type))
        return (
            record is not None
            and bool(record["fields"])
            and all(
                self.generator.map_type(logical) == "float16_t"
                for _, logical, _ in record["fields"]
            )
        )

    def convert(self, value_type, rendered, *, encode=False):
        g = self.generator
        if self.shape(value_type):
            intrinsic = "asuint16" if encode else "asfloat16"
            if (
                g.hlsl_function_name_is_shadowed(intrinsic)
                or intrinsic in g.global_variable_types
            ):
                raise DirectXHalfStorageError(
                    f"Exact half storage requires the unshadowed {intrinsic} intrinsic",
                    source_type=g.type_name_string(value_type),
                    reason="storage-intrinsic-shadowed",
                )
            return f"{intrinsic}({rendered})"
        self.physical_type(value_type)
        record = self.structures.get(g.map_type(value_type))
        if record is not None:
            return f"{record['encode' if encode else 'decode']}({rendered})"
        if self.lowered(value_type):
            raise DirectXHalfStorageError(
                "Half storage arrays require element-wise access or a containing structure value",
                reason="array-storage-value-unsupported",
                source_type=g.type_name_string(value_type),
            )
        return rendered

    def binding_type(self, binding):
        if binding is None or binding.get("kind") == "workgroup-pointer":
            return None
        g = self.generator
        if (
            g.hlsl_resource_type_name(binding.get("resource_type"))
            not in self.resource_types
        ):
            return None
        logical = binding.get("source_element_type") or binding.get("element_type")
        return logical if self.lowered(logical) else None

    def access(self, expression, *, write=False):
        """Return the physical lvalue and its logical type without copying storage."""
        g = self.generator
        if isinstance(expression, PointerAccessNode):
            return self.access(
                MemberAccessNode(
                    UnaryOpNode("*", expression.pointer_expr), expression.member
                ),
                write=write,
            )
        if isinstance(expression, SwizzleNode):
            return self.access(
                MemberAccessNode(expression.vector_expr, expression.components),
                write=write,
            )
        if isinstance(expression, MemberAccessNode):
            base = self.access(expression.object, write=write)
            if base is None:
                return None
            raw, logical = base
            if g.hlsl_union_layout_for_type(logical) is not None:
                return None
            value_type = g.hlsl_source_expression_type(expression)
            if value_type is None:
                record = self.structures.get(g.map_type(logical))
                value_type = (
                    next(
                        (
                            field[1]
                            for field in record["fields"]
                            if field[0] == expression.member
                        ),
                        None,
                    )
                    if record
                    else None
                )
            return (
                (f"{raw}.{expression.member}", value_type)
                if value_type is not None
                else None
            )
        if isinstance(expression, FunctionCallNode):
            name = g.function_call_name(expression)
            if (
                name != "buffer_load"
                or g.hlsl_function_name_is_shadowed(name)
                or len(expression.arguments) != 2
            ):
                return None
            pointer, index = expression.arguments
        elif isinstance(expression, ArrayAccessNode):
            base = self.access(expression.array, write=write)
            if base is not None:
                raw, _ = base
                return (
                    f"{raw}[{g.generate_expression(expression.index)}]",
                    g.hlsl_source_expression_type(expression),
                )
            pointer, index = expression.array, expression.index
        elif isinstance(expression, UnaryOpNode) and expression.op == "*":
            if isinstance(
                expression.operand, UnaryOpNode
            ) and expression.operand.op in {"++", "--"}:
                return None
            pointer, index = expression.operand, 0
        else:
            return None
        binding = g.hlsl_resource_pointer_binding(pointer)
        logical = self.binding_type(binding)
        if logical is None:
            return None
        if isinstance(expression, FunctionCallNode):
            g.validate_buffer_call_access("buffer_load", expression.arguments)
        raw = g.generate_hlsl_resource_pointer_access(
            pointer, index, require_write=write, physical=True
        )
        return (raw, logical) if raw is not None else None

    def read(self, expression):
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            name = self.generator.hlsl_identifier_name(
                expression if isinstance(expression, str) else expression.name
            )
            logical = self.constant_types.get(name)
            return self.convert(logical, name) if logical is not None else None
        if isinstance(expression, UnaryOpNode) and expression.op in {"++", "--"}:
            access = self.access(expression.operand, write=True)
            if access is None:
                return None
            raw, logical = access
            if not self.lowered(logical):
                return None
            self.access(expression.operand)
            if not self.shape(logical):
                raise DirectXHalfStorageError(
                    "Half storage updates require scalar or vector values",
                    reason="update-shape-unsupported",
                    source_type=logical,
                )
            key = (
                self.generator.map_type(logical),
                expression.op,
                bool(getattr(expression, "is_postfix", False)),
            )
            if key not in self.updates:
                self.updates[key] = self.name("__crossgl_half_storage_update")
            return f"{self.updates[key]}({raw})"
        access = self.access(expression)
        return self.convert(access[1], access[0]) if access else None

    def statement(self, expression):
        g = self.generator
        if not isinstance(expression, FunctionCallNode):
            return None
        name = g.function_call_name(expression)
        arguments = expression.arguments
        if (
            name != "buffer_store"
            or g.hlsl_function_name_is_shadowed(name)
            or len(arguments) != 3
        ):
            return None
        if self.binding_type(g.hlsl_resource_pointer_binding(arguments[0])) is None:
            return None
        return g.generate_buffer_call(name, arguments, statement_context=True)

    def assignment(self, node, target, value, operator, *, statement_context):
        access = self.access(target, write=True)
        if access is None:
            return None
        g = self.generator
        raw, logical = access
        if not self.lowered(logical):
            return None
        if operator == "=":
            rhs = g.generate_expression_with_expected(
                value, logical, source_location=getattr(node, "source_location", None)
            )
        else:
            self.access(target)
            binary_operator = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}.get(operator)
            if binary_operator is None or not self.shape(logical):
                raise DirectXHalfStorageError(
                    f"Unsupported half storage assignment {operator}",
                    reason="assignment-operator-unsupported",
                    source_type=logical,
                )
            value_type = g.expression_result_type(value)
            contract = g.hlsl_native_16_bit_arithmetic_contract(
                node, binary_operator, logical, value_type
            )
            if contract is None:
                raise DirectXHalfStorageError(
                    "Unresolved half storage arithmetic operands",
                    reason="arithmetic-type-unresolved",
                    source_type=logical,
                )
            if self.has_side_effects(target) or (
                self.has_side_effects(value)
                and not g.hlsl_minimum_precision_compound_target_is_stable(target)
            ):
                raise DirectXHalfStorageError(
                    "Half compound storage updates require a stable index and pointer",
                    reason="side-effecting-assignment-target",
                    source_type=logical,
                    source_location=getattr(node, "source_location", None),
                )
            left = g.hlsl_native_16_bit_arithmetic_operand(
                self.convert(logical, raw),
                contract["left"],
                contract["left_operation_base"],
            )
            right = g.hlsl_native_16_bit_arithmetic_operand(
                g.generate_expression_with_expected(value, None),
                contract["right"],
                contract["right_operation_base"],
            )
            operation = f"({left} {binary_operator} {right})"
            rhs = (
                g.hlsl_contextual_floating_narrowing_expression(
                    operation,
                    logical,
                    contract["operation_type"],
                    source_location=getattr(node, "source_location", None),
                )
                or f"{g.map_type(logical)}({operation})"
            )
        store = f"{raw} = {self.convert(logical, rhs, encode=True)}"
        return store if statement_context else self.convert(logical, f"({store})")

    def has_side_effects(self, expression):
        g = self.generator
        for node in g.walk_ast(expression):
            if isinstance(node, AssignmentNode) or (
                isinstance(node, UnaryOpNode) and node.op in {"++", "--"}
            ):
                return True
            if isinstance(node, FunctionCallNode):
                name = g.expression_name(node.function)
                if name == "buffer_load" and not g.hlsl_function_name_is_shadowed(name):
                    continue
                return True
        return False

    def register_resource(self, name, logical_resource, *, array=False):
        g = self.generator
        logical = g.hlsl_typed_buffer_element_type(
            logical_resource, self.resource_types
        )
        if (
            not array
            and logical is not None
            and self.lowered(logical)
            and self.homogeneous(logical)
        ):
            self.resources[name] = dict(BINARY16_STORAGE)

    def header(self):
        return resource_storage_header(self.resources) if self.resources else ""

    def declarations(self):
        g = self.generator
        lines = []

        def assignment(logical, lhs, rhs, encode, indent=1):
            mapped = g.map_type(logical)
            match = re.fullmatch(r"([^\[]+)\[([^\]]+)\](.*)", mapped)
            if match:
                index = f"element{indent}"
                lines.append(
                    "    " * indent
                    + f"for (uint {index} = 0; {index} < {match.group(2)}; ++{index}) {{"
                )
                assignment(
                    match.group(1) + match.group(3),
                    f"{lhs}[{index}]",
                    f"{rhs}[{index}]",
                    encode,
                    indent + 1,
                )
                lines.append("    " * indent + "}")
            else:
                lines.append(
                    "    " * indent
                    + f"{lhs} = {self.convert(logical, rhs, encode=encode)};"
                )

        for logical, record in list(self.structures.items()):
            physical = record["physical"]
            lines.append(f"struct {physical} {{")
            for name, _, field_type in record["fields"]:
                base, *suffix = field_type.split("[", 1)
                lines.append(f"    {base} {name}{'[' + suffix[0] if suffix else ''};")
            lines.append("};")
            for encode in (False, True):
                source, target = (logical, physical) if encode else (physical, logical)
                lines.extend(
                    [
                        f"{target} {record['encode' if encode else 'decode']}({source} value) {{",
                        f"    {target} result;",
                    ]
                )
                for name, field_logical, _ in record["fields"]:
                    assignment(field_logical, f"result.{name}", f"value.{name}", encode)
                lines.extend(["    return result;", "}"])
        for (logical, operator, postfix), name in self.updates.items():
            old = self.convert(logical, "storage")
            lines.extend(
                [
                    f"{logical} {name}(inout {self.physical_type(logical)} storage) {{",
                    f"    {logical} previous = {old};",
                    f"    {logical} next = {logical}(previous {operator[0]} {logical}(1));",
                    f"    storage = {self.convert(logical, 'next', encode=True)};",
                    f"    return {'previous' if postfix else 'next'};",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n" if lines else ""
