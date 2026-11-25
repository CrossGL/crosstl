"""Global resource bindings and record materialization in the HIP to CrossGL converter: record definitions and function-local records."""

import hashlib

from ...device_globals import binds_device_buffer
from ..HipAst import FunctionNode, StructNode, VariableNode


class HipRecordMaterializationMixin:
    """Record materialization members of ``HipToCrossGLConverter``."""

    def prepare_global_resource_bindings(self, ast_node):
        self.global_resource_bindings = {}
        for statement in getattr(ast_node, "statements", []) or []:
            if self.is_global_resource_declaration(statement):
                self.global_resource_bindings[id(statement)] = len(
                    self.global_resource_bindings
                )
        self.global_resource_binding_count = len(self.global_resource_bindings)

    def is_global_resource_declaration(self, statement):
        """Whether a top-level declaration imports as a resource with a binding."""
        return isinstance(statement, VariableNode) and (
            "__constant__" in set(getattr(statement, "qualifiers", []) or [])
            or binds_device_buffer(statement)
        )

    def global_resource_binding_for(self, node):
        binding = self.global_resource_bindings.get(id(node))
        if binding is None:
            binding = self.global_resource_binding_count
            self.global_resource_binding_count += 1
        return binding

    def reset_function_local_record_materializations(self):
        self.function_local_record_definitions = {}
        self.function_local_record_node_ids = set()
        self.function_local_record_declaration_types = {}

    def prepare_record_definitions(self, ast_node):
        self.record_name_stack = []
        self.record_definitions = {}
        self.record_qualified_names = {}
        self.enum_value_aliases = {}
        self.enum_underlying_types = {}
        self.unsupported_record_static_constants = set()
        self.function_definitions = {}
        self.return_type_by_node_id = {}
        for definitions in self.function_local_record_definitions.values():
            for definition in definitions:
                self.collect_record_definition(definition, ())
        for statement in getattr(ast_node, "statements", []) or []:
            if isinstance(statement, StructNode):
                self.collect_record_definition(statement, ())
            elif isinstance(statement, FunctionNode):
                self.function_definitions.setdefault(statement.name, []).append(
                    statement
                )
                self.register_function_return_nodes(statement)

    def materialized_record_type_names(self):
        return {
            self.convert_hip_record_name_to_crossgl(node.name)
            for node in self.record_definitions.values()
            if node is not None and getattr(node, "name", None)
        }

    def collect_record_definition(self, node, owners):
        name = getattr(node, "name", None)
        if not name:
            return
        qualified_name = "::".join((*owners, name))
        self.record_qualified_names[id(node)] = qualified_name
        self.record_definitions[qualified_name] = node
        existing = self.record_definitions.get(name)
        if existing is None and name not in self.record_definitions:
            self.record_definitions[name] = node
        elif existing is not node:
            self.record_definitions[name] = None
        for member in getattr(node, "members", []) or []:
            if isinstance(member, StructNode):
                self.collect_record_definition(member, (*owners, name))

    def prepare_function_local_record_materializations(self, ast_node):
        for statement in getattr(ast_node, "statements", []) or []:
            if not isinstance(statement, FunctionNode):
                continue
            definitions = []
            self.collect_function_local_record_materializations(
                getattr(statement, "body", None),
                statement,
                ("body",),
                definitions,
                set(),
            )
            if definitions:
                self.function_local_record_definitions[id(statement)] = definitions

    def collect_function_local_record_materializations(
        self,
        value,
        function,
        path,
        definitions,
        visited,
    ):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)

        if isinstance(value, list):
            self.collect_function_local_records_from_statement_list(
                value, function, path, definitions
            )
            for index, item in enumerate(value):
                if isinstance(item, StructNode):
                    continue
                self.collect_function_local_record_materializations(
                    item,
                    function,
                    (*path, str(index)),
                    definitions,
                    visited,
                )
            return
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                self.collect_function_local_record_materializations(
                    item,
                    function,
                    (*path, str(index)),
                    definitions,
                    visited,
                )
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                self.collect_function_local_record_materializations(
                    value[key],
                    function,
                    (*path, str(key)),
                    definitions,
                    visited,
                )
            return
        if isinstance(value, (StructNode, FunctionNode)):
            return

        for attribute, child in sorted(
            getattr(value, "__dict__", {}).items(),
            key=lambda item: item[0],
        ):
            self.collect_function_local_record_materializations(
                child,
                function,
                (*path, attribute),
                definitions,
                visited,
            )

    def collect_function_local_records_from_statement_list(
        self, statements, function, path, definitions
    ):
        for index, statement in enumerate(statements):
            if not isinstance(statement, StructNode) or getattr(
                statement, "name", None
            ):
                continue

            prefix = self.nested_record_declarator_prefix(statement)
            declarations = []
            next_index = index + 1
            while next_index < len(statements):
                declaration = statements[next_index]
                if not isinstance(declaration, VariableNode):
                    break
                if not self.nested_record_declarator_matches(declaration, prefix):
                    break
                declarations.append(declaration)
                next_index += 1
            if not declarations:
                continue

            materialized_name = self.function_local_record_name(
                function,
                statement,
                declarations,
                (*path, str(index)),
            )
            materialized = self.clone_ast_node(statement)
            materialized.name = materialized_name
            definitions.append(materialized)
            self.function_local_record_node_ids.add(id(statement))
            for declaration in declarations:
                self.function_local_record_declaration_types[id(declaration)] = (
                    self.rewrite_nested_record_declarator_type(
                        getattr(declaration, "vtype", ""),
                        prefix,
                        materialized_name,
                    )
                )

    def function_local_record_name(self, function, node, declarations, path):
        keyword = "union" if getattr(node, "is_union", False) else "struct"
        function_name = self.sanitize_crossgl_type_identifier(
            getattr(function, "name", "function")
        )
        declarator_name = self.sanitize_crossgl_type_identifier(
            getattr(declarations[0], "name", "local")
        )
        signature = [
            str(getattr(function, "name", "")),
            ".".join(path),
            keyword,
            *(str(getattr(declaration, "name", "")) for declaration in declarations),
            self.function_local_record_signature(node),
        ]
        digest = hashlib.sha256("\0".join(signature).encode("utf-8")).hexdigest()[:12]
        return (
            f"CrossGLFunctionLocal_{function_name}_{declarator_name}_"
            f"{keyword}_{digest}"
        )

    def function_local_record_signature(self, node):
        parts = [
            "union" if getattr(node, "is_union", False) else "struct",
            str(getattr(node, "name", "")),
        ]
        for member in getattr(node, "members", []) or []:
            if isinstance(member, StructNode):
                parts.append(self.function_local_record_signature(member))
                continue
            parts.extend(
                (
                    type(member).__name__,
                    str(getattr(member, "name", "")),
                    str(getattr(member, "vtype", "")),
                    " ".join(
                        str(item) for item in getattr(member, "qualifiers", []) or []
                    ),
                )
            )
        return "\0".join(parts)

    def emit_function_local_record_definitions(self, function):
        for record in self.function_local_record_definitions.get(id(function), ()):
            self.visit_StructNode(record)
            self.emit("")

    def function_local_materialized_variable_type(self, node):
        return self.function_local_record_declaration_types.get(
            id(node), getattr(node, "vtype", "int")
        )
