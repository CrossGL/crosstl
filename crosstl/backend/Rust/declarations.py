"""Rust items, function bodies, let statements, pattern bindings and assignments as CrossGL."""

import re

from crosstl.translator.renames import renamed

from .import_tables import BLOCK_EXPRESSION_NODE_TYPES
from .RustAst import (
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ClosureNode,
    ConstNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LabeledBlockNode,
    LetNode,
    LoopNode,
    MatchBindingPatternNode,
    MatchesMacroNode,
    MatchNode,
    MatchOrPatternNode,
    MatchStructPatternNode,
    ReferenceNode,
    ReturnNode,
    StaticNode,
    TupleNode,
    UseNode,
    WhileNode,
)


class RustDeclarationMixin:
    """Item, function-body, binding and assignment members of ``RustToCrossGLConverter``."""

    def register_type_alias(self, alias):
        if getattr(alias, "name", None):
            self.type_aliases[alias.name] = alias

    def register_associated_consts(self, ast):
        for impl_block in getattr(ast, "impl_blocks", []):
            if getattr(impl_block, "is_negative", False):
                continue

            struct_name = getattr(impl_block, "struct_name", None)
            if not struct_name:
                continue

            for const in getattr(impl_block, "associated_consts", []) or []:
                const_name = getattr(const, "name", None)
                if not const_name or getattr(const, "value", None) is None:
                    continue
                generated_name = self.format_associated_const_identifier(
                    struct_name,
                    const_name,
                )
                for type_name in self.type_lookup_names(struct_name):
                    self.associated_consts[f"{type_name}::{const_name}"] = (
                        generated_name
                    )

    def format_associated_const_identifier(self, type_name, const_name):
        prefix = self.impl_function_prefix(type_name)
        return self.crossgl_identifier(f"{prefix}_{const_name}")

    def generate_impl_associated_const(self, type_name, const, indent=1):
        if getattr(const, "value", None) is None:
            return ""

        indent_str = "    " * indent
        declarator = self.format_typed_declarator(
            const.vtype,
            self.format_associated_const_identifier(type_name, const.name),
        )
        value = self.generate_expression(const.value)
        return f"{indent_str}const {declarator} = {value};\n"

    def register_use_alias(self, use_stmt):
        path = getattr(use_stmt, "path", None)
        items = getattr(use_stmt, "items", None)
        if items:
            self.register_grouped_use_aliases(path, items)
            return

        alias = getattr(use_stmt, "alias", None)
        if not alias or not path or "*" in path or "{" in path:
            return

        self.register_import_alias(alias, path)

    def register_grouped_use_aliases(self, path, items):
        if not path:
            return

        base_path = path.split("::{", 1)[0]
        if not base_path:
            return

        for item in items:
            alias = item.get("alias")
            item_path = item.get("path")
            if not alias or not item_path or "*" in item_path or "{" in item_path:
                continue

            self.register_import_alias(alias, f"{base_path}::{item_path}")

    def register_import_alias(self, alias, path):
        if self.is_imported_type_path(path):
            self.imported_type_aliases[alias] = path
        else:
            self.imported_module_aliases[alias] = path

    def is_imported_type_path(self, path):
        if self.map_builtin_type(path) is not None:
            return True

        type_name = path.rsplit("::", 1)[-1]
        if not type_name:
            return False

        return type_name[0].isupper() or type_name in self.type_map

    def generate_type_alias(self, alias):
        if getattr(alias, "generics", None) or not getattr(alias, "alias_type", None):
            return ""

        alias_type = self.map_type(alias.alias_type)
        # CrossGL spells an alias like C: ``typedef <type> <name>;``.
        return f"    typedef {alias_type} {alias.name};\n"

    def generate_function(self, func, indent=1, struct_name=None):
        """Render one Rust function node as a CrossGL function."""
        code = ""
        indent_str = "    " * indent

        local_binding_names = self.collect_simple_local_binding_names(func.body)
        params_str, param_types, name_aliases = self.prepare_function_parameters(
            func.params,
            struct_name=struct_name,
            extra_forbidden=local_binding_names,
        )
        display_return_type = self.normalize_receiver_type(
            func.return_type, struct_name
        )
        return_type = self.map_function_return_type(display_return_type)

        if struct_name:
            func_name = f"{self.impl_function_prefix(struct_name)}_{func.name}"
        else:
            func_name = func.name

        previous_return_type = self.current_function_return_type
        previous_helpers = self.current_closure_helpers
        self.current_closure_helpers = []
        self.current_function_return_type = func.return_type
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        self.push_name_alias_scope(name_aliases)
        self.local_binding_name_scopes.append(local_binding_names)
        try:
            body_code = self.generate_function_body(
                func.body,
                indent=indent + 1,
                allow_implicit_final_return=True,
            )
            pattern_binding_code = self.generate_function_parameter_pattern_bindings(
                func.params,
                name_aliases,
                indent + 1,
            )
            helper_code = "".join(self.current_closure_helpers)
            code += helper_code
            code += f"{indent_str}{return_type} {func_name}({params_str}) {{\n"
            code += pattern_binding_code
            code += body_code
            code += f"{indent_str}}}\n\n"
        finally:
            self.current_function_return_type = previous_return_type
            self.current_closure_helpers = previous_helpers
            self.local_binding_name_scopes.pop()
            self.pop_name_alias_scope()
            self.pop_local_callable_scope()
            self.pop_value_type_scope()

        return code

    def impl_function_prefix(self, struct_name):
        generic = self.parse_generic_type(struct_name)
        type_name = generic[0] if generic is not None else struct_name
        type_name = self.strip_reference_type(type_name)
        if type_name == "()":
            return "Unit"
        prefix = self.type_lookup_names(type_name)[-1]
        prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_") or "Type"
        if prefix[0].isdigit():
            prefix = f"_{prefix}"
        return prefix

    def generate_shader_stage_function(self, func, shader_type):
        """Render a Rust shader attribute as a named CrossGL stage entry."""
        local_binding_names = self.collect_simple_local_binding_names(func.body)
        params_str, param_types, name_aliases = self.prepare_function_parameters(
            func.params,
            extra_forbidden=local_binding_names,
            shader_type=shader_type,
        )
        return_type = self.map_function_return_type(func.return_type)
        entry_point_name = self.get_entry_point_name_from_attributes(func.attributes)
        stage_name = self.crossgl_identifier(entry_point_name or func.name)
        numthreads = self.get_numthreads_from_attributes(func.attributes)
        numthreads_suffix = (
            f" @numthreads({', '.join(numthreads)})" if numthreads else ""
        )
        stage_header = (
            shader_type if stage_name == "main" else f"{shader_type} {stage_name}"
        )

        previous_return_type = self.current_function_return_type
        previous_helpers = self.current_closure_helpers
        self.current_closure_helpers = []
        self.current_function_return_type = func.return_type
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        self.push_name_alias_scope(name_aliases)
        self.local_binding_name_scopes.append(local_binding_names)
        try:
            body_code = self.generate_function_body(
                func.body,
                indent=3,
                allow_implicit_final_return=True,
            )
            pattern_binding_code = self.generate_function_parameter_pattern_bindings(
                func.params,
                name_aliases,
                3,
            )
            helper_code = "".join(self.current_closure_helpers)
        finally:
            self.current_function_return_type = previous_return_type
            self.current_closure_helpers = previous_helpers
            self.local_binding_name_scopes.pop()
            self.pop_name_alias_scope()
            self.pop_local_callable_scope()
            self.pop_value_type_scope()

        code = helper_code
        code += f"    {stage_header} {{\n"
        for mode_line in self.stage_execution_mode_lines(func, shader_type):
            code += f"        {mode_line}\n"
        code += f"        {return_type} main({params_str}){numthreads_suffix} {{\n"
        code += pattern_binding_code
        code += body_code
        code += "        }\n"
        code += "    }\n\n"
        return code

    def generate_function_parameter_pattern_bindings(
        self, params, name_aliases, indent
    ):
        code = ""
        for param in params:
            pattern = getattr(param, "pattern", None)
            if pattern is None:
                continue

            # ``name_aliases`` holds the parameters the converter renamed.
            subject = renamed(name_aliases, param.name)
            code += self.generate_match_pattern_bindings(subject, pattern, indent)
        return code

    def function_returns_value(self):
        return_type = self.current_function_return_type
        if not return_type:
            return False

        return self.map_type(return_type) != "void"

    def is_implicit_final_return_statement(self, stmt, index, body):
        if index != len(body) - 1 or not self.function_returns_value():
            return False

        return not isinstance(
            stmt,
            (
                ConstNode,
                StaticNode,
                FunctionNode,
                LetNode,
                AssignmentNode,
                ReturnNode,
                ForNode,
                WhileNode,
                BreakNode,
                ContinueNode,
            ),
        )

    def generate_implicit_final_return_statement(
        self,
        stmt,
        indent,
        loop_contexts=None,
    ):
        if isinstance(stmt, MatchesMacroNode):
            return self.generate_matches_expression_return(stmt, indent, loop_contexts)

        return self.generate_expression_result(
            stmt,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_function_body(
        self,
        body,
        indent=1,
        loop_contexts=None,
        allow_implicit_final_return=False,
    ):
        code = ""
        indent_str = "    " * indent
        body = body or []
        loop_contexts = loop_contexts or []
        scoped_aliases = self.local_aliasing_enabled()
        if scoped_aliases:
            self.push_name_alias_scope()
        pushed_callable_scope = False
        if not self.local_callable_scopes:
            self.push_local_callable_scope()
            pushed_callable_scope = True

        try:
            self.predeclare_local_function_items(body)
            for index, stmt in enumerate(body):
                if (
                    allow_implicit_final_return
                    and self.is_implicit_final_return_statement(
                        stmt,
                        index,
                        body,
                    )
                ):
                    code += self.generate_implicit_final_return_statement(
                        stmt,
                        indent,
                        loop_contexts,
                    )
                elif isinstance(stmt, ConstNode):
                    code += self.generate_const_statement(stmt, indent)
                elif isinstance(stmt, StaticNode):
                    code += self.generate_static_statement(stmt, indent)
                elif isinstance(stmt, FunctionNode):
                    code += self.generate_local_function_item(stmt)
                elif isinstance(stmt, UseNode):
                    code += f"{indent_str}// use {stmt.path}\n"
                elif isinstance(stmt, LetNode):
                    code += self.generate_let_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, AssignmentNode):
                    code += self.generate_assignment_statement(
                        stmt, indent, loop_contexts
                    )
                elif isinstance(stmt, ReturnNode):
                    option_value = self.generate_option_unwrap_return_expression(
                        stmt.value
                    )
                    if option_value is not None:
                        code += f"{indent_str}return {option_value};\n"
                    elif isinstance(stmt.value, LoopNode):
                        code += self.generate_loop_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif self.is_block_expression_node(stmt.value):
                        code += self.generate_block_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, IfNode):
                        code += self.generate_if_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, MatchNode):
                        code += self.generate_match_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, MatchesMacroNode):
                        code += self.generate_matches_expression_return(
                            stmt.value, indent, loop_contexts
                        )
                    elif isinstance(stmt.value, AssignmentNode):
                        code += self.generate_assignment_expression_result(
                            stmt.value,
                            indent,
                            self.return_result_target,
                            loop_contexts,
                        )
                    elif self.expression_contains_try(stmt.value):
                        code += self.generate_expression_result(
                            stmt.value,
                            indent,
                            self.return_result_target,
                            loop_contexts,
                        )
                    elif self.expression_contains_inline_result_expression(stmt.value):
                        prelude, value = self.generate_materialized_expression(
                            stmt.value,
                            indent,
                            loop_contexts,
                        )
                        code += prelude
                        code += f"{indent_str}return {value};\n"
                    elif stmt.value:
                        value = self.generate_reparseable_expression(stmt.value)
                        code += f"{indent_str}return {value};\n"
                    else:
                        code += f"{indent_str}return;\n"
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent, loop_contexts)
                elif isinstance(stmt, WhileNode):
                    code += self.generate_while_loop(stmt, indent, loop_contexts)
                elif isinstance(stmt, LoopNode):
                    code += self.generate_loop(
                        stmt, indent, loop_contexts=loop_contexts
                    )
                elif isinstance(stmt, MatchNode):
                    code += self.generate_match_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, BreakNode):
                    code += self.generate_break_statement(stmt, indent, loop_contexts)
                elif isinstance(stmt, ContinueNode):
                    code += self.generate_continue_statement(
                        stmt, indent, loop_contexts
                    )
                elif isinstance(stmt, FunctionCallNode):
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )
                elif isinstance(stmt, BinaryOpNode):
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )
                elif isinstance(stmt, str):
                    code += f"{indent_str}{stmt};\n"
                else:
                    code += self.generate_expression_result(
                        stmt,
                        indent,
                        None,
                        loop_contexts,
                    )

                if self.statement_needs_labeled_control_propagation(
                    stmt,
                    loop_contexts,
                ):
                    code += self.generate_labeled_control_propagation(
                        loop_contexts,
                        indent,
                    )
        finally:
            if scoped_aliases:
                self.pop_name_alias_scope()
            if pushed_callable_scope:
                self.pop_local_callable_scope()

        return code

    def generate_local_function_item(self, func):
        helper_name = self.get_local_function_item_helper_name(func)
        self.add_local_callable(func.name, helper_name)
        helper_func = FunctionNode(
            func.return_type,
            helper_name,
            func.params,
            func.body,
            attributes=func.attributes,
            visibility=func.visibility,
            generics=func.generics,
            where_clauses=func.where_clauses,
            is_async=func.is_async,
            is_unsafe=func.is_unsafe,
            abi=func.abi,
            is_const=func.is_const,
        )
        helper_code = self.generate_function(helper_func, 0)

        if self.current_closure_helpers is not None:
            self.current_closure_helpers.append(helper_code)
            return ""
        return helper_code

    def generate_const_statement(self, node, indent):
        indent_str = "    " * indent
        declarator = self.format_typed_declarator(node.vtype, node.name)
        value = self.generate_expression(node.value)
        return f"{indent_str}const {declarator} = {value};\n"

    def generate_static_statement(self, node, indent):
        indent_str = "    " * indent
        mutability = "mut " if node.is_mutable else ""
        declarator = self.format_typed_declarator(node.vtype, node.name)
        value = self.generate_expression(node.value)
        return f"{indent_str}static {mutability}{declarator} = {value};\n"

    def generate_let_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        type_str = ""

        if getattr(stmt, "else_body", None) is not None:
            return self.generate_let_else_statement(stmt, indent, loop_contexts)

        macro_pattern = self.generate_macro_pattern_let_statement(
            stmt,
            indent,
            loop_contexts,
        )
        if macro_pattern is not None:
            return macro_pattern

        if self.is_nontrivial_let_pattern(stmt.name):
            return self.generate_pattern_let_statement(stmt, indent, loop_contexts)

        if self.is_discard_pattern(stmt.name):
            return self.generate_discard_let_statement(stmt, indent, loop_contexts)
        if self.is_tuple_pattern(stmt.name):
            return self.generate_tuple_pattern_let_statement(
                stmt,
                indent,
                loop_contexts,
            )

        if isinstance(stmt.value, LoopNode):
            return self.generate_loop_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, LabeledBlockNode):
            return self.generate_labeled_block_expression_let(
                stmt, indent, loop_contexts
            )
        if self.is_block_expression_node(stmt.value):
            return self.generate_block_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, IfNode):
            return self.generate_if_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, MatchNode):
            return self.generate_match_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, MatchesMacroNode):
            return self.generate_matches_expression_let(stmt, indent, loop_contexts)
        if isinstance(stmt.value, AssignmentNode):
            code = ""
            target_name = self.declare_local_alias(stmt.name)
            if stmt.vtype:
                code += (
                    f"{indent_str}"
                    f"{self.format_typed_declarator(stmt.vtype, target_name)};\n"
                )
            return code + self.generate_assignment_expression_result(
                stmt.value,
                indent,
                target_name,
                loop_contexts,
            )
        if self.expression_contains_try(stmt.value):
            return self.generate_try_let_statement(stmt, indent, loop_contexts)

        if self.expression_contains_inline_result_expression(stmt.value):
            prelude, value_str = self.generate_materialized_expression(
                stmt.value,
                indent,
                loop_contexts,
            )
            target_name = self.declare_local_alias(stmt.name)
            let_keyword = "let mut" if stmt.is_mutable else "let"
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return prelude + f"{indent_str}{type_str} = {value_str};\n"

            inferred_type = self.infer_value_type(stmt.value)
            if inferred_type:
                self.add_value_type(target_name, inferred_type)
                inferred_declarator = self.inferred_array_declarator(
                    inferred_type,
                    target_name,
                )
                if inferred_declarator is not None:
                    return (
                        prelude + f"{indent_str}{inferred_declarator} = {value_str};\n"
                    )
            return prelude + f"{indent_str}{let_keyword} {target_name} = {value_str};\n"

        if isinstance(stmt.value, ClosureNode):
            helper_name = self.try_generate_closure_helper(
                stmt.value,
                context_name=stmt.name,
            )
            if helper_name is not None:
                target_name = self.declare_local_alias(stmt.name)
                self.add_local_callable(stmt.name, target_name)
                let_keyword = "let mut" if stmt.is_mutable else "let"
                return f"{indent_str}{let_keyword} {target_name} = {helper_name};\n"

        if stmt.vtype:
            type_str = self.format_typed_declarator(stmt.vtype, stmt.name)
        elif stmt.value:
            type_str = ""

        if stmt.value:
            value_str = self.generate_reparseable_expression(stmt.value)
            target_name = self.declare_local_alias(stmt.name)
            let_keyword = "let mut" if stmt.is_mutable else "let"
            if isinstance(stmt.value, ClosureNode):
                self.add_local_callable(stmt.name, target_name)
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return f"{indent_str}{type_str} = {value_str};\n"
            inferred_type = self.infer_value_type(stmt.value)
            if inferred_type:
                self.add_value_type(target_name, inferred_type)
                inferred_declarator = self.inferred_array_declarator(
                    inferred_type,
                    target_name,
                )
                if inferred_declarator is not None:
                    return f"{indent_str}{inferred_declarator} = {value_str};\n"
            return f"{indent_str}{let_keyword} {target_name} = {value_str};\n"
        else:
            target_name = self.declare_local_alias(stmt.name)
            if stmt.vtype:
                self.add_value_type(target_name, stmt.vtype)
                type_str = self.format_typed_declarator(stmt.vtype, target_name)
                return f"{indent_str}{type_str};\n"
            return f"{indent_str}{target_name};\n"

    def generate_macro_pattern_let_statement(self, stmt, indent, loop_contexts=None):
        binding_name = self.size_macro_pattern_binding(stmt.name)
        if binding_name is None:
            return None

        if stmt.value is None:
            return ""

        indent_str = "    " * indent
        value = self.generate_reparseable_expression(stmt.value)
        if stmt.vtype:
            self.add_value_type(binding_name, stmt.vtype)
            declarator = self.format_typed_declarator(stmt.vtype, binding_name)
            return f"{indent_str}{declarator} = {value};\n"

        inferred_type = self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(binding_name, inferred_type)
        return f"{indent_str}let {binding_name} = {value};\n"

    def size_macro_pattern_binding(self, pattern):
        if not isinstance(pattern, FunctionCallNode):
            return None
        if self.rust_macro_base_name(pattern.name) != "size":
            return None
        if len(pattern.args) != 1:
            return None

        binding = pattern.args[0]
        if not self.is_plain_identifier(binding) or self.is_discard_pattern(binding):
            return None
        return binding

    def generate_let_else_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""

        indent_str = "    " * indent
        subject, code = self.generate_let_else_subject(
            stmt.value,
            indent,
            loop_contexts,
        )
        binding_names = self.collect_pattern_binding_names(stmt.name)
        code += self.generate_let_else_binding_declarations(
            stmt,
            binding_names,
            indent,
        )

        matched_flag = self.next_match_chain_flag_name()
        code += f"{indent_str}bool {matched_flag} = false;\n"

        def success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n"

        match_code = self.generate_nested_pattern_match(
            subject,
            stmt.name,
            indent + 1,
            success,
        )
        match_code = self.rewrite_pattern_binding_declarations_as_assignments(
            match_code,
            binding_names,
        )

        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += match_code
        code += f"{indent_str}}}\n"
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_scoped_function_body(
            stmt.else_body,
            indent + 1,
            loop_contexts,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_let_else_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            expression,
            indent,
            loop_contexts,
        )

        if isinstance(subject, str) and not code:
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            subject = subject_name

        return subject, code

    def generate_let_else_binding_declarations(self, stmt, binding_names, indent):
        indent_str = "    " * indent
        code = ""

        for name in binding_names:
            if (
                isinstance(stmt.name, str)
                and stmt.name == name
                and stmt.vtype is not None
            ):
                code += (
                    f"{indent_str}{self.format_typed_declarator(stmt.vtype, name)};\n"
                )
            else:
                code += f"{indent_str}{name};\n"

        return code

    def rewrite_pattern_binding_declarations_as_assignments(self, code, binding_names):
        for name in sorted(binding_names, key=len, reverse=True):
            pattern = rf"(?m)^(\s*)auto {re.escape(name)} ="
            code = re.sub(pattern, rf"\1{name} =", code)
        return code

    def collect_pattern_binding_names(self, pattern):
        names = []

        def add_name(name):
            if not self.is_discard_pattern(name) and name not in names:
                names.append(name)

        def visit(node):
            if isinstance(node, MatchBindingPatternNode):
                add_name(node.name)
                visit(node.pattern)
                return

            if isinstance(node, ReferenceNode):
                visit(node.expression)
                return

            if isinstance(node, MatchOrPatternNode):
                for alternative in node.patterns:
                    visit(alternative)
                return

            if isinstance(node, TupleNode):
                for element in node.elements:
                    visit(element)
                return

            if isinstance(node, ArrayNode):
                for element in node.elements:
                    visit(element)
                return

            if isinstance(node, MatchStructPatternNode):
                for _, field_pattern in node.fields:
                    visit(field_pattern)
                return

            if isinstance(node, FunctionCallNode):
                for arg in node.args:
                    visit(arg)
                return

            if isinstance(node, str) and self.is_simple_pattern_binding(node):
                add_name(node)

        visit(pattern)
        return names

    def is_nontrivial_let_pattern(self, pattern):
        return isinstance(
            pattern,
            (
                ArrayNode,
                FunctionCallNode,
                MatchBindingPatternNode,
                MatchOrPatternNode,
                MatchStructPatternNode,
                ReferenceNode,
            ),
        )

    def generate_pattern_let_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""

        subject, code = self.generate_let_else_subject(
            stmt.value,
            indent,
            loop_contexts,
        )
        binding_names = self.collect_pattern_binding_names(stmt.name)
        code += self.generate_let_else_binding_declarations(
            stmt,
            binding_names,
            indent,
        )

        match_code = self.generate_nested_pattern_match(
            subject,
            stmt.name,
            indent,
            lambda success_indent: "",
        )
        match_code = self.rewrite_pattern_binding_declarations_as_assignments(
            match_code,
            binding_names,
        )
        return code + match_code

    def is_discard_pattern(self, pattern):
        return pattern == "_"

    def is_tuple_pattern(self, pattern):
        return isinstance(pattern, TupleNode)

    def generate_discard_let_statement(self, stmt, indent, loop_contexts=None):
        if stmt.value is None:
            return ""
        return self.generate_discarded_expression(stmt.value, indent, loop_contexts)

    def generate_tuple_pattern_let_statement(self, stmt, indent, loop_contexts=None):
        if not isinstance(stmt.value, TupleNode):
            if stmt.value is None:
                return ""
            return self.generate_tuple_pattern_expression_let(
                stmt,
                indent,
                loop_contexts,
            )

        type_elements = self.get_tuple_pattern_type_elements(
            stmt.vtype,
            len(stmt.name.elements),
        )
        return self.generate_tuple_pattern_bindings(
            stmt.name,
            stmt.value,
            indent,
            loop_contexts,
            type_elements,
        )

    def generate_tuple_pattern_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject_name = self.next_match_tuple_name()
        code = f"{indent_str}auto {subject_name};\n"
        code += self.generate_expression_result(
            stmt.value,
            indent,
            subject_name,
            loop_contexts,
        )

        type_elements = self.get_tuple_pattern_type_elements(
            stmt.vtype,
            len(stmt.name.elements),
        )

        for index, pattern_element in enumerate(stmt.name.elements):
            type_element = type_elements[index] if type_elements is not None else None
            code += self.generate_pattern_binding(
                pattern_element,
                self.generate_tuple_element_accessor(subject_name, index),
                indent,
                loop_contexts,
                type_element,
            )
        return code

    def generate_tuple_pattern_bindings(
        self, pattern, value, indent, loop_contexts=None, type_elements=None
    ):
        if len(pattern.elements) != len(value.elements):
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if type_elements is not None and len(type_elements) != len(pattern.elements):
            type_elements = None

        code = ""
        for index, (pattern_element, value_element) in enumerate(
            zip(pattern.elements, value.elements)
        ):
            type_element = type_elements[index] if type_elements is not None else None
            code += self.generate_pattern_binding(
                pattern_element,
                value_element,
                indent,
                loop_contexts,
                type_element,
            )
        return code

    def generate_pattern_binding(
        self, pattern, value, indent, loop_contexts=None, type_name=None
    ):
        if self.is_discard_pattern(pattern):
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if self.is_tuple_pattern(pattern):
            if isinstance(value, TupleNode):
                type_elements = self.get_tuple_pattern_type_elements(
                    type_name,
                    len(pattern.elements),
                )
                return self.generate_tuple_pattern_bindings(
                    pattern,
                    value,
                    indent,
                    loop_contexts,
                    type_elements,
                )
            return self.generate_discarded_expression(value, indent, loop_contexts)

        if isinstance(pattern, str):
            if type_name:
                return self.generate_typed_pattern_binding(
                    pattern,
                    value,
                    type_name,
                    indent,
                    loop_contexts,
                )
            return self.generate_auto_pattern_binding(
                pattern,
                value,
                indent,
                loop_contexts,
            )

        return self.generate_discarded_expression(value, indent, loop_contexts)

    def generate_auto_pattern_binding(self, name, value, indent, loop_contexts=None):
        indent_str = "    " * indent

        if isinstance(
            value,
            (
                LoopNode,
                IfNode,
                MatchNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ) or self.expression_contains_try(value):
            code = f"{indent_str}auto {name};\n"
            code += self.generate_expression_result(
                value,
                indent,
                name,
                loop_contexts,
            )
            return code

        return f"{indent_str}auto {name} = {self.generate_expression(value)};\n"

    def get_tuple_pattern_type_elements(self, type_name, expected_count):
        elements = self.split_tuple_type(type_name)
        if elements is None or len(elements) != expected_count:
            return None
        return elements

    def split_tuple_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        text = type_name.strip()
        if not (text.startswith("(") and text.endswith(")")):
            return None

        inner = text[1:-1].strip()
        if not inner:
            return []

        return self.split_generic_arguments(inner)

    def generate_typed_pattern_binding(
        self, name, value, type_name, indent, loop_contexts=None
    ):
        indent_str = "    " * indent
        declarator = self.format_typed_declarator(type_name, name)

        if isinstance(
            value,
            (
                LoopNode,
                IfNode,
                MatchNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            code = f"{indent_str}{declarator};\n"
            code += self.generate_expression_result(
                value,
                indent,
                name,
                loop_contexts,
            )
            return code

        if self.expression_contains_try(value):
            code = f"{indent_str}{declarator};\n"
            code += self.generate_expression_result(
                value,
                indent,
                name,
                loop_contexts,
            )
            return code

        return f"{indent_str}{declarator} = {self.generate_expression(value)};\n"

    def generate_assignment_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent

        if stmt.operator == "=" and isinstance(stmt.right, LoopNode):
            return self.generate_loop_expression_assignment(stmt, indent, loop_contexts)
        if stmt.operator == "=" and self.is_block_expression_node(stmt.right):
            return self.generate_block_expression_assignment(
                stmt, indent, loop_contexts
            )
        if stmt.operator == "=" and isinstance(stmt.right, IfNode):
            return self.generate_if_expression_assignment(stmt, indent, loop_contexts)
        if stmt.operator == "=" and isinstance(stmt.right, MatchNode):
            return self.generate_match_expression_assignment(
                stmt, indent, loop_contexts
            )
        if stmt.operator == "=" and isinstance(stmt.right, MatchesMacroNode):
            return self.generate_matches_expression_assignment(
                stmt, indent, loop_contexts
            )
        if self.expression_contains_try(stmt.right):
            return self.generate_try_assignment_statement(stmt, indent, loop_contexts)
        if self.expression_contains_inline_result_expression(stmt.right):
            prelude, value = self.generate_materialized_expression(
                stmt.right,
                indent,
                loop_contexts,
            )
            target = self.generate_expression(stmt.left)
            return prelude + f"{indent_str}{target} {stmt.operator} {value};\n"
        return f"{indent_str}{self.generate_assignment(stmt)};\n"

    def generate_assignment_expression_result(
        self, stmt, indent, result_target, loop_contexts=None
    ):
        indent_str = "    " * indent
        code = self.generate_assignment_statement(stmt, indent, loop_contexts)

        if result_target is self.return_result_target:
            return code + f"{indent_str}return;\n"
        if result_target:
            return code + f"{indent_str}{result_target} = ();\n"
        return code
