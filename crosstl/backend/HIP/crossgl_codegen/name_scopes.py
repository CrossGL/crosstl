"""Device metadata sources, identifier scopes and name sanitizing in the HIP to CrossGL converter."""

import re

from ...CUDA.array_extents import (
    crossgl_array_extent,
    expression_extent,
    extent_identifier,
    undeclared_extent_names,
)
from ...CUDA.keyword_renames import unused_identifier_name
from ..HipAst import ArrayAccessNode, CastNode, MemberAccessNode, UnaryOpNode


class HipNameScopeMixin:
    """Name scope members of ``HipToCrossGLConverter``."""

    def register_device_property_source(self, name, device_id):
        if not name or device_id is None:
            return
        if not self.device_property_source_scopes:
            self.device_property_source_scopes.append({})
        self.device_property_source_scopes[-1][name] = device_id
        self.clear_member_query_source(name)

    def clear_device_property_source(self, name):
        if not name:
            return
        if not self.device_property_source_scopes:
            self.device_property_source_scopes.append({})
        self.device_property_source_scopes[-1][name] = None

    def lookup_device_property_source(self, name):
        for scope in reversed(self.device_property_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_device_attribute_source(self, name, attribute_name, device_id):
        if not name or attribute_name is None or device_id is None:
            return
        if not self.device_attribute_source_scopes:
            self.device_attribute_source_scopes.append({})
        self.device_attribute_source_scopes[-1][name] = (attribute_name, device_id)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def clear_device_attribute_source(self, name):
        if not name:
            return
        if not self.device_attribute_source_scopes:
            self.device_attribute_source_scopes.append({})
        self.device_attribute_source_scopes[-1][name] = None

    def lookup_device_attribute_source(self, name):
        for scope in reversed(self.device_attribute_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_device_query_source(self, name, query_name, device_id=None):
        if not name or query_name is None:
            return
        if not self.device_query_source_scopes:
            self.device_query_source_scopes.append({})
        self.device_query_source_scopes[-1][name] = (query_name, device_id)
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_member_query_source(name)

    def clear_device_query_source(self, name):
        if not name:
            return
        if not self.device_query_source_scopes:
            self.device_query_source_scopes.append({})
        self.device_query_source_scopes[-1][name] = None

    def lookup_device_query_source(self, name):
        for scope in reversed(self.device_query_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_member_query_sources(self, name, member_queries):
        if not name or not member_queries:
            return
        if not self.member_query_source_scopes:
            self.member_query_source_scopes.append({})
        self.member_query_source_scopes[-1][name] = dict(member_queries)
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)

    def clear_member_query_source(self, name):
        if not name:
            return
        if not self.member_query_source_scopes:
            self.member_query_source_scopes.append({})
        self.member_query_source_scopes[-1][name] = None

    def lookup_member_query_source(self, name):
        for scope in reversed(self.member_query_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def clear_device_metadata_source(self, name):
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def clear_lvalue_metadata_source(self, node):
        self.clear_device_metadata_source(self.get_runtime_pointer_target_name(node))
        root_name = self.get_lvalue_metadata_root_name(node)
        if root_name is not None:
            self.clear_device_metadata_source(root_name)

    def get_lvalue_metadata_root_name(self, node):
        if isinstance(node, CastNode):
            return self.get_lvalue_metadata_root_name(node.expression)
        if isinstance(node, UnaryOpNode):
            return self.get_lvalue_metadata_root_name(node.operand)
        if isinstance(node, MemberAccessNode):
            return self.get_lvalue_metadata_root_name(node.object)
        if isinstance(node, ArrayAccessNode):
            return self.get_lvalue_metadata_root_name(node.array)
        if isinstance(node, str):
            return node
        return None

    def push_identifier_name_scope(self):
        self.identifier_name_scopes.append({})

    def pop_identifier_name_scope(self):
        if len(self.identifier_name_scopes) > 1:
            self.identifier_name_scopes.pop()

    def register_identifier_name(self, name):
        if not self.is_simple_identifier(name):
            return name
        output_name = self.sanitize_identifier_name(name)
        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        self.identifier_name_scopes[-1][name] = output_name
        return output_name

    def register_variable_declaration_name(self, name):
        if self.is_simple_identifier(name):
            return self.register_identifier_name(name)

        output_name = self.sanitize_qualified_variable_name(name)
        if output_name is None:
            return name

        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        self.identifier_name_scopes[-1][name] = output_name
        return output_name

    def output_identifier_name(self, name):
        if self.suppress_identifier_name_rewrite != 0:
            return name
        for scope in reversed(self.identifier_name_scopes):
            if name in scope:
                return scope[name]
        if not self.is_simple_identifier(name):
            return name
        return name

    def sanitize_identifier_name(self, name):
        if name in self.CROSSGL_RESERVED_IDENTIFIERS:
            return unused_identifier_name(name, self.source_identifier_names)
        return name

    def sanitize_crossgl_type_identifier(self, name):
        parts = []
        for char in str(name):
            if char.isascii() and (char.isalnum() or char == "_"):
                parts.append(char)
            elif char == ":":
                if not parts or parts[-1] != "_":
                    parts.append("_")
            elif not parts or parts[-1] != "_":
                parts.append("_")

        raw_name = str(name)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw_name):
            # A plain identifier keeps its underscores, so the declaration and
            # every use of the type agree, as in the CUDA importer.
            sanitized = raw_name
        else:
            sanitized = "".join(parts).strip("_") or "anonymous"
        if sanitized[0].isdigit():
            sanitized = f"type_{sanitized}"
        return self.sanitize_identifier_name(sanitized)

    def convert_hip_record_name_to_crossgl(self, name):
        base_name, template_args = self.parse_cpp_template(name)
        if not template_args:
            return self.sanitize_crossgl_type_identifier(name)

        parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_hip_type_to_crossgl(arg)
            parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in parts if part)

    def format_crossgl_array_extent(self, size):
        """The CrossGL extent of a HIP array extent: its value when it is a constant (``CUDA.array_extents``)."""
        return crossgl_array_extent(
            size, self.integer_constants, self.format_crossgl_expression_extent
        )

    def format_crossgl_expression_extent(self, size):
        """``(K * 2)`` for an extent of names and arithmetic, or a typed import refusal for any other extent."""
        declared = self.integer_constants.declares
        expression = expression_extent(size, declared=declared)
        if expression is not None:
            return expression
        placeholder = self.sanitize_identifier_name(
            extent_identifier(size, "hip_array_extent_")
        )
        undeclared = undeclared_extent_names(size, declared)
        if undeclared and expression_extent(size) is not None:
            names = ", ".join(f"'{name}'" for name in undeclared)
            return self.refuse_import(
                f"HIP array extent '{size}' names {names}, which the program does not declare (a macro or "
                "constant of a header the importer did not read), so no target can size the array",
                placeholder,
            )
        return self.refuse_import(
            f"HIP array extent '{size}' is not an integral constant expression of names, integer "
            "literals and arithmetic, which is all a CrossGL array extent can spell",
            placeholder,
        )

    def sanitize_qualified_variable_name(self, name):
        if not isinstance(name, str) or "::" not in name:
            return None

        stripped = self.strip_cpp_template_arguments(name)
        chars = []
        previous_separator = False
        for char in stripped:
            if char.isalnum() or char == "_":
                chars.append(char)
                previous_separator = False
            elif chars and not previous_separator:
                chars.append("_")
                previous_separator = True

        output_name = "".join(chars)
        if not output_name:
            return None
        if output_name[0].isdigit():
            output_name = f"_{output_name}"
        return self.sanitize_identifier_name(output_name)

    def format_function_declaration_name(self, name):
        operator_name = self.format_cpp_operator_function_name(name)
        if operator_name is not None:
            return operator_name
        base_name, template_args = self.parse_cpp_template(name)
        if template_args:
            return self.format_function_declaration_name(base_name)
        if self.is_simple_identifier(name):
            return self.sanitize_identifier_name(name)
        if not isinstance(name, str) or "::" not in name:
            return self.sanitize_crossgl_type_identifier(name)

        normalized_name = name.replace("::~", "::destructor_")
        return self.sanitize_qualified_variable_name(normalized_name) or name

    def format_cpp_operator_function_name(self, name):
        if not isinstance(name, str):
            return None

        normalized_name = name.replace("::~", "::destructor_")
        parts = normalized_name.split("::")
        raw_operator = parts[-1]
        if not raw_operator.startswith("operator"):
            return None

        operator = raw_operator[len("operator") :]
        if not operator:
            return None

        suffix = self.CPP_OPERATOR_FUNCTION_NAME_PARTS.get(operator)
        if suffix is None:
            suffix = re.sub(r"[^A-Za-z0-9_]+", "_", operator).strip("_")
            suffix = re.sub(r"_+", "_", suffix)
        if not suffix:
            return None

        parts[-1] = f"operator_{suffix}"
        return self.sanitize_qualified_variable_name("::".join(parts)) or parts[-1]

    def format_function_call_name(self, name):
        user_defined_name = self.format_user_defined_function_call_name(name)
        if user_defined_name is not None:
            return user_defined_name
        return name

    def format_user_defined_function_call_name(self, name):
        if self.is_user_defined_function(name):
            return self.format_function_declaration_name(name)

        base_name, template_args = self.parse_cpp_template(name)
        if template_args and self.is_user_defined_function(base_name):
            return self.format_function_declaration_name(base_name)

        return None

    def strip_cpp_template_arguments(self, text):
        stripped = []
        depth = 0
        for char in text:
            if char == "<":
                depth += 1
                continue
            if char == ">" and depth > 0:
                depth -= 1
                continue
            if depth == 0:
                stripped.append(char)
        return "".join(stripped)

    def visit_lvalue_expression(self, node):
        self.suppress_device_property_member_access += 1
        self.suppress_device_attribute_value_access += 1
        self.suppress_device_query_value_access += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_device_property_member_access -= 1
            self.suppress_device_attribute_value_access -= 1
            self.suppress_device_query_value_access -= 1

    def visit_runtime_argument_expression(self, node):
        self.suppress_device_property_member_access += 1
        self.suppress_device_attribute_value_access += 1
        self.suppress_device_query_value_access += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_device_property_member_access -= 1
            self.suppress_device_attribute_value_access -= 1
            self.suppress_device_query_value_access -= 1

    def visit_source_comment_expression(self, node):
        self.suppress_identifier_name_rewrite += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_identifier_name_rewrite -= 1
