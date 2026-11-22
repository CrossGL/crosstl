"""User functions, generated matrix helpers, resource-object type hints and variable-type scopes in the HIP to CrossGL converter."""

import re

from ..HipAst import (
    ArrayAccessNode,
    CastNode,
    FunctionNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
)


class HipResourceHintMixin:
    """Resource hint members of ``HipToCrossGLConverter``."""

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            name = getattr(current, "name", None)
            body = getattr(current, "body", None)
            if name is not None and body is not None:
                if not self.is_generated_matrix_helper_function(current):
                    names.add(name)

            for stmt in getattr(current, "statements", []):
                collect(stmt)
            for function in getattr(current, "functions", []):
                collect(function)
            for kernel in getattr(current, "kernels", []):
                collect(kernel)

        collect(node)
        names.discard(None)
        return names

    def collect_generated_matrix_helper_types(self, node):
        structs = []
        for stmt in getattr(node, "statements", []) or []:
            if isinstance(stmt, StructNode):
                structs.append(stmt)
        structs.extend(getattr(node, "structs", []) or [])
        return {
            struct.name
            for struct in structs
            if self.is_generated_matrix_helper_struct(struct)
        }

    def native_matrix_helper_dimensions(self, type_name):
        match = re.fullmatch(r"(float|double)([234])x([234])", str(type_name))
        if match is None:
            return None
        scalar_type, columns, rows = match.groups()
        return scalar_type, int(columns), int(rows)

    def convert_native_matrix_helper_name_to_crossgl(self, type_name):
        if type_name not in self.generated_matrix_helper_types:
            return None

        dimensions = self.native_matrix_helper_dimensions(type_name)
        if dimensions is None:
            return None

        scalar_type, columns, rows = dimensions
        prefix = "dmat" if scalar_type == "double" else "mat"
        suffix = str(columns) if columns == rows else f"{columns}x{rows}"
        return f"{prefix}{suffix}"

    def is_generated_matrix_helper_struct(self, node):
        dimensions = self.native_matrix_helper_dimensions(getattr(node, "name", ""))
        if dimensions is None:
            return False

        scalar_type, columns, rows = dimensions
        members = getattr(node, "members", []) or []
        member_by_name = {
            getattr(member, "name", None): member
            for member in members
            if getattr(member, "name", None)
        }

        matrix_values = member_by_name.get("m")
        column_count = member_by_name.get("CGL_COLUMNS")
        row_count = member_by_name.get("CGL_ROWS")
        if matrix_values is None or column_count is None or row_count is None:
            return False

        return (
            getattr(matrix_values, "vtype", None) == f"{scalar_type}[{columns * rows}]"
            and getattr(column_count, "vtype", None) == "const int"
            and str(getattr(column_count, "value", "")).strip() == str(columns)
            and "static" in (getattr(column_count, "qualifiers", []) or [])
            and getattr(row_count, "vtype", None) == "const int"
            and str(getattr(row_count, "value", "")).strip() == str(rows)
            and "static" in (getattr(row_count, "qualifiers", []) or [])
        )

    def is_generated_matrix_helper_function(self, node):
        if getattr(node, "name", None) not in {"operator*", "transpose", "inverse"}:
            return False
        if not self.function_references_generated_matrix_helper_type(node):
            return False
        qualifiers = set(getattr(node, "qualifiers", []) or [])
        return {"__host__", "__device__", "inline"}.issubset(qualifiers)

    def function_references_generated_matrix_helper_type(self, node):
        candidate_types = [getattr(node, "return_type", "")]
        for param in getattr(node, "params", []) or []:
            if isinstance(param, dict):
                candidate_types.append(param.get("type", ""))
            else:
                candidate_types.append(getattr(param, "vtype", ""))
        return any(
            self.strip_type_qualifiers(candidate_type).strip()
            in self.generated_matrix_helper_types
            for candidate_type in candidate_types
        )

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def add_resource_object_type_hint(self, hints, name, resource_type):
        if not name or not resource_type:
            return
        if name in hints and hints[name] != resource_type:
            hints[name] = None
            return
        hints[name] = resource_type

    def collect_global_resource_object_type_hints(self, node):
        hints = {}
        global_names = self.collect_global_declared_variable_names(node)
        for stmt in getattr(node, "statements", []):
            if isinstance(stmt, FunctionNode):
                self.collect_global_resource_object_type_hints_from_function(
                    stmt, global_names, hints
                )
        return hints

    def collect_global_declared_variable_names(self, node):
        return {
            stmt.name
            for stmt in getattr(node, "statements", [])
            if isinstance(stmt, VariableNode)
        }

    def get_resource_object_call_hint(self, function_name, args=None):
        base_name, _ = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None
        if base_name in self.HIP_TEXTURE_CALL_TYPE_HINTS:
            return 0, self.HIP_TEXTURE_CALL_TYPE_HINTS[base_name]
        if base_name in self.HIP_SURFACE_CALL_TYPE_HINTS:
            if base_name.endswith("write"):
                return 1, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
            if args and self.is_surface_pointer_output_argument(args[0]):
                return 1, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
            return 0, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
        return None

    def is_surface_pointer_output_argument(self, expression):
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            return True
        if isinstance(expression, CastNode):
            return self.is_surface_pointer_output_argument(expression.expression)
        if isinstance(expression, str):
            return self.is_surface_output_target(expression)
        return False

    def get_resource_object_expression_name(self, expression):
        if isinstance(expression, str):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.get_resource_object_expression_name(expression.array)
        if isinstance(expression, CastNode):
            return self.get_resource_object_expression_name(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.get_resource_object_expression_name(expression.operand)
        return None

    def push_resource_object_hint_scope(self, hints):
        self.resource_object_hint_scopes.append(hints)

    def pop_resource_object_hint_scope(self):
        if self.resource_object_hint_scopes:
            self.resource_object_hint_scopes.pop()

    def lookup_resource_object_type_hint(self, name):
        for scope in reversed(self.resource_object_hint_scopes):
            if name in scope:
                return scope[name]
        return self.global_resource_object_type_hints.get(name)

    def push_variable_type_scope(self):
        self.variable_type_scopes.append({})
        self.integer_constants.push()
        self.push_vector1_name_scope()
        self.device_property_source_scopes.append({})
        self.device_attribute_source_scopes.append({})
        self.device_query_source_scopes.append({})
        self.member_query_source_scopes.append({})
        self.union_active_member_scopes.append({})
        self.union_object_identity_scopes.append({})
        self.union_indirect_object_scopes.append(set())

    def pop_variable_type_scope(self):
        if len(self.variable_type_scopes) > 1:
            self.variable_type_scopes.pop()
            self.integer_constants.pop()
        self.pop_vector1_name_scope()
        if len(self.device_property_source_scopes) > 1:
            self.device_property_source_scopes.pop()
        if len(self.device_attribute_source_scopes) > 1:
            self.device_attribute_source_scopes.pop()
        if len(self.device_query_source_scopes) > 1:
            self.device_query_source_scopes.pop()
        if len(self.member_query_source_scopes) > 1:
            self.member_query_source_scopes.pop()
        if len(self.union_active_member_scopes) > 1:
            self.union_active_member_scopes.pop()
        if len(self.union_object_identity_scopes) > 1:
            self.union_object_identity_scopes.pop()
        if len(self.union_indirect_object_scopes) > 1:
            self.union_indirect_object_scopes.pop()

    def register_union_object_declaration(self, name, definition, declaration):
        if (
            not name
            or definition is None
            or not self.union_tracks_active_member(definition)
        ):
            return
        if name in self.union_object_identity_scopes[-1]:
            self.raise_record_semantic(
                ("union lexical shadowing with flattened scope", definition),
                f"variable '{name}'",
            )
        self.union_object_identity_scopes[-1][name] = f"{name}#{id(declaration)}"

    def register_union_indirect_object(self, name, raw_type, definition):
        if (
            not name
            or definition is None
            or not self.union_tracks_active_member(definition)
        ):
            return
        compact = str(raw_type).replace(" ", "")
        if "*" in compact or "&" in compact or compact.startswith("ptr<"):
            self.union_indirect_object_scopes[-1].add(name)

    def lookup_union_object_identity(self, name):
        for scope in reversed(self.union_object_identity_scopes):
            if name in scope:
                return scope[name]
        return name

    def is_union_indirect_object(self, name):
        return any(
            name in scope for scope in reversed(self.union_indirect_object_scopes)
        )

    def register_variable_type(self, name, type_name):
        if not name or not type_name:
            return
        if not self.variable_type_scopes:
            self.variable_type_scopes.append({})
        self.variable_type_scopes[-1][name] = type_name
        # A declaration hides an integral constant of its name from array extents.
        self.integer_constants.declare(name)
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def lookup_variable_type(self, name):
        for scope in reversed(self.variable_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def push_vector1_name_scope(self):
        self.vector1_name_scopes.append({})

    def pop_vector1_name_scope(self):
        if len(self.vector1_name_scopes) > 1:
            self.vector1_name_scopes.pop()

    def register_vector1_name(self, name, type_name):
        if not name:
            return
        if not self.vector1_name_scopes:
            self.vector1_name_scopes.append({})
        self.vector1_name_scopes[-1][name] = (
            self.hip_vector1_scalar_type(type_name) is not None
        )

    def is_vector1_name(self, name):
        if not isinstance(name, str):
            return False
        for scope in reversed(self.vector1_name_scopes):
            if name in scope:
                return scope[name]
        return False

    def hip_vector1_scalar_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        if "*" in type_name or self.has_array_suffix(type_name):
            return None
        return self.VECTOR1_TYPE_MAPPING.get(type_name)
