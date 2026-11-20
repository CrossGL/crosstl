"""Operand precedence, compound literals, casts, storage wrapper reads and decltype."""

import re

from ....translator.codegen.pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    InitializerListNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..MetalLexer import MetalLexer
from ..MetalParser import MetalParser


class MetalOperandMixin:
    """Members of ``MetalToCrossGLConverter``: operand precedence, compound literals, casts, storage wrapper reads and decltype."""

    def expression_precedence(self, expression):
        if isinstance(expression, AssignmentNode):
            return self.assignment_precedence
        if isinstance(expression, TernaryOpNode):
            return self.conditional_precedence
        if isinstance(expression, BinaryOpNode):
            return self.binary_precedence.get(expression.op, 0)
        return None

    def generate_precedence_operand(
        self,
        operand,
        parent_precedence,
        is_main=False,
        parenthesize_equal=False,
    ):
        text = self.generate_expression(operand, is_main)
        operand_precedence = self.expression_precedence(operand)
        if operand_precedence is not None and (
            operand_precedence < parent_precedence
            or (parenthesize_equal and operand_precedence == parent_precedence)
        ):
            return f"({text})"
        return text

    def generate_binary_operand(self, operand, parent_op, is_right, is_main=False):
        parent_precedence = self.binary_precedence.get(parent_op, 0)
        parenthesize_equal = bool(
            is_right
            and isinstance(operand, BinaryOpNode)
            and (parent_op not in {"&&", "||"} or operand.op != parent_op)
        )
        return self.generate_precedence_operand(
            operand,
            parent_precedence,
            is_main,
            parenthesize_equal=parenthesize_equal,
        )

    def generate_postfix_operand(self, operand, is_main=False):
        if isinstance(operand, (CastNode, UnaryOpNode)):
            return f"({self.generate_expression(operand, is_main)})"
        return self.generate_precedence_operand(
            operand,
            self.postfix_precedence,
            is_main,
        )

    def generate_metal_compound_literal(self, expr, is_main):
        """Render ``(T){...}`` as the braced initialization of a ``T`` temporary.

        The Metal compiler accepts the C compound literal ``(float4x4){c0, c1, c2, c3}``
        as a temporary initialized from the braced list, exactly as the declaration
        ``float4x4 m = {c0, c1, c2, c3};`` would initialize it.
        """
        if not isinstance(expr.expression, InitializerListNode):
            return None
        target_type = expr.target_type
        if (
            not target_type
            or "[" in str(target_type)
            or self.metal_pointer_pointee_type_once(
                self.resolve_type_alias(target_type)
            )
            is not None
        ):
            return None
        return self.generate_initializer_value(expr.expression, is_main, target_type)

    def generate_cast_operand(self, operand, rendered_operand):
        if isinstance(operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered_operand})"
        if isinstance(operand, str) and self.cast_literal_operand_pattern.fullmatch(
            rendered_operand
        ):
            return f"({rendered_operand})"
        return rendered_operand

    @staticmethod
    def storage_wrapper_load_parts(expression):
        if isinstance(expression, ArrayAccessNode) and isinstance(
            expression.array, CastNode
        ):
            return expression.array, expression.index
        elif (
            isinstance(expression, UnaryOpNode)
            and expression.op == "*"
            and isinstance(expression.operand, CastNode)
        ):
            return expression.operand, None
        return None

    def generate_storage_wrapper_call_arguments(
        self, function_name, expression, is_main=False
    ):
        candidates = [
            index
            for index, argument in enumerate(expression.args)
            if self.storage_wrapper_load_parts(argument) is not None
        ]
        if not candidates:
            return {}
        selected = self.resolve_transported_metal_source_overload(
            function_name, expression.args, getattr(expression, "source_location", None)
        )
        if selected is None:
            _binding, selected = self.resolve_metal_user_function_overload(
                function_name,
                expression.args,
                source_offset=self.alias_source_offset(expression),
            )
        if selected is None:
            return {}
        rendered = {}
        for index in candidates:
            parameter = selected.params[index]
            if self.reference_parameter(parameter):
                continue
            value = self.generate_storage_wrapper_value_read(
                expression.args[index],
                is_main,
                self.metal_source_overload_parameter_type(parameter),
            )
            if value is not None:
                rendered[index] = value
        return rendered

    def generate_storage_wrapper_value_read(
        self, expression, is_main=False, expected_type=None
    ):
        """Copy an immediate scalar-wrapper load without invoking constructors."""
        parts = self.storage_wrapper_load_parts(expression)
        if parts is None:
            return None
        cast, index = parts
        target_type = self.metal_pointer_pointee_type_once(
            self.resolve_type_alias(cast.target_type)
        )
        if target_type is None:
            return None
        target_type = self.normalized_metal_type(target_type)
        target_type = (
            self.resolve_conditional_type(target_type, require_concrete=True)
            or target_type
        )
        if expected_type:
            destination = self.resolve_type_alias(expected_type)
            if "&" in str(destination) or "*" in str(destination):
                return None
            destination = self.normalized_metal_type(destination)
            destination = (
                self.resolve_conditional_type(destination, require_concrete=True)
                or destination
            )
            if (
                not self.is_plain_metal_auto_type(destination)
                and destination != target_type
            ):
                return None
        declaration = self.struct_declarations.get(target_type)
        if declaration is None:
            return None
        members = [
            member
            for member in declaration.members
            if isinstance(member, VariableNode)
            and "static" not in (getattr(member, "qualifiers", ()) or ())
        ]
        if (
            len(members) != 1
            or getattr(declaration, "aggregate_kind", None) == "union"
            or getattr(declaration, "attributes", None)
            or getattr(declaration, "alignas", None)
        ):
            return None
        member = members[0]
        if (
            getattr(member, "array_sizes", None)
            or getattr(member, "attributes", None)
            or getattr(member, "alignas", None)
            or getattr(member, "bitfield_width", None) is not None
            or getattr(member, "declarator_type_suffix", None)
            or set(getattr(member, "qualifiers", ()) or ()) & {"const", "volatile"}
        ):
            return None
        member_type = self.resolve_type_alias(member.vtype)
        if set(str(member_type).split()) & {"const", "volatile"}:
            return None
        member_layout = scalar_storage_layout(member_type)
        if member_layout is None:
            return None
        constructors = self.metal_constructor_contract(target_type)
        if constructors and self.has_declared_copy_or_move_constructor(
            target_type, constructors[1]
        ):
            return None
        source = cast.expression
        source_type = self.metal_pointer_pointee_type_once(
            self.expression_metal_type(source)
        )
        source_layout = scalar_storage_layout(source_type)
        if source_layout != member_layout or self.metal_concrete_type_layout(
            target_type
        ) != (member_layout.byte_width, member_layout.byte_width):
            return None
        source_qualifiers = self.metal_addressable_storage_qualifiers(source)
        target_qualifiers = set(getattr(cast, "qualifiers", ()) or ())
        address_spaces = {"device", "constant", "thread", "threadgroup"}
        source_spaces = set(source_qualifiers or ()) & address_spaces
        target_spaces = target_qualifiers & address_spaces
        if (
            source_qualifiers is None
            or len(source_spaces) != 1
            or target_spaces != source_spaces
            or "volatile" in set(source_qualifiers) | target_qualifiers
        ):
            raise PointerReinterpretationError(
                "Cannot preserve scalar-wrapper load address space or volatile access",
                source_type=source_type,
                target_type=target_type,
                address_space=" ".join(sorted(target_spaces)) or None,
                access="read",
                reason="wrapper-load-storage-unproven",
                source_location=getattr(expression, "source_location", None),
            )
        if isinstance(source, UnaryOpNode) and source.op == "&" and index is None:
            value = source.operand
        else:
            value = ArrayAccessNode(source, index if index is not None else "0")
        mapped_type = self.map_type(target_type)
        key = (mapped_type, member.name, self.map_type(member_type))
        if key not in self.storage_wrapper_read_helpers:
            base_name = self.sanitize_identifier(
                f"_crosstl_metal_load_value_{target_type}"
            )
            reserved = self.wide_vector_reserved_names | {
                self.sanitize_identifier(name) for name in self.user_function_names
            }
            name = base_name
            while name in reserved:
                name += "_"
            self.wide_vector_reserved_names.add(name)
            field = self.struct_member_name_maps.get(target_type, {}).get(
                member.name, self.sanitize_identifier(member.name)
            )
            self.storage_wrapper_read_helpers[key] = (name, field)
        helper = self.storage_wrapper_read_helpers[key][0]
        return f"{helper}({self.generate_expression(value, is_main)})"

    def generate_storage_wrapper_read_helpers(self):
        code = ""
        for (wrapper, _, scalar), (name, field) in sorted(
            self.storage_wrapper_read_helpers.items()
        ):
            code += (
                f"    {wrapper} {name}({scalar} value) {{\n"
                f"        {wrapper} result;\n"
                f"        result.{field} = value;\n"
                "        return result;\n"
                "    }\n\n"
            )
        return code

    def cast_uses_constructor_syntax(self, mapped_type):
        mapped_text = str(mapped_type).strip()
        if "*" in mapped_text or "&" in mapped_text:
            return False
        if mapped_text in {"float", "double", "int", "uint", "bool"}:
            return True
        if (
            self.crossgl_identifier_pattern.fullmatch(mapped_text)
            and mapped_text[0].isupper()
        ):
            return True
        return bool(
            re.fullmatch(
                r"(?:f16vec[234]|[biu]?vec[234]|dvec[234]|"
                r"mat[234](?:x[234])?|dmat[234](?:x[234])?)",
                mapped_text,
            )
        )

    @staticmethod
    def metal_decltype_expression_is_fully_parenthesized(expression_text):
        text = str(expression_text or "").strip()
        if not text.startswith("("):
            return False
        depth = 0
        for index, character in enumerate(text):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return index == len(text) - 1
                if depth < 0:
                    return False
        return False

    def metal_decltype_expression_is_lvalue(self, expression):
        if isinstance(expression, VariableNode):
            name = getattr(expression, "name", None)
            return bool(
                name
                and (
                    name in self.current_variable_types
                    or name in self.global_variable_types
                )
            )
        if isinstance(expression, ArrayAccessNode):
            return True
        if isinstance(expression, MemberAccessNode):
            return self.metal_decltype_expression_is_lvalue(expression.object)
        return (
            isinstance(expression, UnaryOpNode)
            and getattr(expression, "op", getattr(expression, "operator", None)) == "*"
        )

    def resolve_metal_decltype_type(self, metal_type):
        text = str(metal_type or "").strip()
        prefix = next(
            (
                candidate
                for candidate in ("decltype(", "metal::decltype(")
                if text.startswith(candidate)
            ),
            None,
        )
        if prefix is None:
            return None

        open_index = text.find("(")
        depth = 0
        close_index = None
        for index in range(open_index, len(text)):
            character = text[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index is None:
            return None

        suffix_match = re.fullmatch(
            r"((?:\s*::\s*[A-Za-z_]\w*)*)\s*([*&]*)", text[close_index + 1 :]
        )
        if suffix_match is None:
            return None
        member_suffix = re.sub(r"\s+", "", suffix_match.group(1))
        suffix = member_suffix + suffix_match.group(2)
        expression_text = text[open_index + 1 : close_index].strip()
        if not expression_text or expression_text == "auto":
            return None

        try:
            lexer = MetalLexer(expression_text, preprocess=False)
            parser = MetalParser(lexer.tokenize())
            expression = parser.parse_expression()
            if parser.current_token[0] != "EOF":
                return None
        except (SyntaxError, ValueError, TypeError):
            return None

        inferred_type = self.expression_metal_type(expression)
        if inferred_type is None:
            return None
        self.record_elided_parameter_uses(expression)
        resolved_type = str(self.resolve_type_alias(inferred_type)).strip()
        if (
            self.metal_decltype_expression_is_fully_parenthesized(expression_text)
            and self.metal_decltype_expression_is_lvalue(expression)
            and not resolved_type.endswith("&")
        ):
            resolved_type += "&"
        if member_suffix:
            if re.search(r"[*&\[]", resolved_type):
                # A reference, pointer or array type has no nested members.
                return None
            resolved_type = self.resolve_concrete_struct_member_aliases(
                resolved_type, member_suffix
            )
            if resolved_type is None:
                return None
            suffix = suffix[len(member_suffix) :]
        return f"{resolved_type}{suffix}"

    def resolve_concrete_struct_member_aliases(self, owner, member_suffix):
        """Follow ``::member`` type aliases from a concrete struct, as ``decltype(local)::member`` names them."""
        current = str(owner).strip()
        for member in member_suffix.split("::")[1:]:
            struct_name = self.normalized_metal_type(current)
            if struct_name in self.ambiguous_struct_names:
                return None
            struct_node = self.struct_declarations.get(struct_name)
            aliases = [
                alias
                for alias in getattr(struct_node, "type_aliases", None) or ()
                if getattr(alias, "name", None) == member
            ]
            if len(aliases) != 1:
                return None
            current = str(self.resolve_type_alias(aliases[0].alias_type)).strip()
        return current
