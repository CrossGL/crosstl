"""Function return types, stage entry attributes, discarded expressions, function bodies and callbacks."""

import re

from ..aggregate_scalar_conversions import render_aggregate_scalar_conversion
from ..default_samplers import is_default_sampler_local
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CallNode,
    CastNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LambdaNode,
    MemberAccessNode,
    MethodCallNode,
    PostfixOpNode,
    RangeForNode,
    ReturnNode,
    StaticAssertNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)
from ..sparse_color import import_sparse_color_assignment
from ..vote_masks import vote_mask_default_initializer


class MetalFunctionMixin:
    """Members of ``MetalToCrossGLConverter``: function return types, stage entry attributes, discarded expressions, function bodies and callbacks."""

    def format_value_template_parameter_declarations(
        self, func, indent, body=None, bound_value_names=None
    ):
        """Declare non-type (value) template parameters that a body consumes with
        a bitwise/shift operator.

        Entry-point kernels drop the ``generic<...>`` prefix, so value template
        parameters such as ``const int bits`` are referenced in the body with no
        declaration and default to ``float`` - which turns bitwise uses like
        ``bits & (bits - 1)`` into invalid float operations. An uninstantiated
        generic kernel is symbolic (not runnable) regardless, so emitting an
        integer placeholder simply keeps the module well-typed.

        The declaration is emitted ONLY for parameters that appear as an operand
        of a bitwise/shift operator, because those are the uses that require
        integer typing. Value parameters used in other roles - most importantly
        as array extents such as ``shared[tg_mem_size]`` - are left untouched so
        array sizing is not disturbed by an injected runtime local."""
        template_parameters = getattr(func, "template_parameters", None) or []
        bound_value_names = set(bound_value_names or ())
        value_names = []
        seen = set()
        for entry in template_parameters:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            kind, name = entry[0], entry[1]
            if kind != "value" or not name or name in seen or name in bound_value_names:
                continue
            seen.add(name)
            value_names.append(name)
        if not value_names:
            return ""
        if body is None:
            body = getattr(func, "body", []) or []
        bitwise_names = set()
        self.collect_bitwise_operand_identifiers(body, bitwise_names)
        names = [name for name in value_names if name in bitwise_names]
        if not names:
            return ""
        pad = "    " * indent
        return "".join(
            f"{pad}int {self.sanitize_identifier(name)} = 0;\n" for name in names
        )

    def collect_bitwise_operand_identifiers(self, node, found):
        """Record identifier names that appear anywhere inside the operands of a
        bitwise or shift operator within ``node``."""
        bitwise_ops = {"&", "|", "^", "<<", ">>"}
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, BinaryOpNode) and getattr(node, "op", None) in bitwise_ops:
            self.collect_identifier_references(node, found)
        for child in self.iter_ast_children(node):
            self.collect_bitwise_operand_identifiers(child, found)

    def collect_identifier_references(self, node, found):
        """Collect bare identifier names (undeclared VariableNode uses) in a
        subtree - i.e. references rather than typed declarations."""
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if (
            isinstance(node, VariableNode)
            and getattr(node, "name", None)
            and not getattr(node, "vtype", None)
        ):
            found.add(node.name)
        for child in self.iter_ast_children(node):
            self.collect_identifier_references(child, found)

    def map_function_return_type(self, return_type, qualifiers=None):
        pointer_buffer_type = self.map_pointer_return_buffer_type(
            return_type, qualifiers=qualifiers
        )
        if pointer_buffer_type:
            return pointer_buffer_type
        mapped_type = self.map_type(return_type)
        if str(mapped_type).rstrip().endswith("&"):
            return str(mapped_type).rstrip()[:-1].rstrip()
        return mapped_type

    def map_pointer_return_buffer_type(self, return_type, qualifiers=None):
        element_type = self.pointer_element_type(return_type)
        if not element_type:
            return None

        # Metal parsing keeps leading pointer-return qualifiers beside the type.
        # Recombine them before choosing the target's read-only resource form.
        qualifier_names = {str(qualifier).lower() for qualifier in qualifiers or []}
        pointee_contract = str(return_type).split("*", 1)[0]
        qualifier_names.update(
            str(qualifier).lower()
            for qualifier in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pointee_contract)
        )
        element_type = re.sub(
            r"^(?:(?:const|device|thread|threadgroup|constant|volatile|restrict)\s+)+",
            "",
            str(element_type).strip(),
        )
        element_type = self.resolve_type_alias(element_type)
        if qualifier_names & {"thread", "threadgroup"}:
            const = "const " if "const" in qualifier_names else ""
            space = "threadgroup" if "threadgroup" in qualifier_names else "thread"
            return f"{const}{space} {self.map_type(element_type)}*"
        buffer_type = (
            "StructuredBuffer"
            if qualifier_names & {"const", "constant", "readonly"}
            else "RWStructuredBuffer"
        )
        return f"{buffer_type}<{self.map_type(element_type)}>"

    def function_output_name(self, func):
        host_name = self.function_host_name(func)
        if host_name and getattr(func, "qualifier", None) in {
            "vertex",
            "fragment",
            "kernel",
        }:
            return host_name
        return self.metal_source_overload_output_names.get(id(func), func.name)

    def should_emit_kernel_stage_entry(self, func):
        if getattr(func, "name", None) == "main":
            return False
        return self.kernel_stage_entry_builtin_types_supported(func)

    def kernel_stage_entry_builtin_types_supported(self, func):
        # Keep imported kernels on the explicit stage-entry path only when the
        # current Metal generator can validate their builtin parameter shapes.
        # Metal lets the positional / dimension compute builtins be declared as a
        # scalar `uint`, a `uint2`, or a `uint3`; the driver fills the requested
        # component count. MLX's reduction kernels (softmax, rms_norm, layer_norm,
        # logsumexp, ...) use the scalar `uint` form for 1-D dispatches, so
        # restricting these to `uint3` wrongly denied them the explicit
        # stage-entry path and collapsed every materialized kernel to an unnamed
        # "main" entry point (invalid SPIR-V: duplicate entry-point names).
        positional_builtin_types = {
            "uint",
            "uint2",
            "uint3",
            "ushort",
            "ushort2",
            "ushort3",
        }
        index_builtin_types = {"uint", "ushort"}
        expected_types = {
            "thread_position_in_grid": positional_builtin_types,
            "thread_position_in_threadgroup": positional_builtin_types,
            "threadgroup_position_in_grid": positional_builtin_types,
            "thread_index_in_threadgroup": index_builtin_types,
            "threads_per_threadgroup": positional_builtin_types,
            "threadgroups_per_grid": positional_builtin_types,
            "thread_index_in_simdgroup": index_builtin_types,
            "threads_per_simdgroup": index_builtin_types,
        }
        for param in getattr(func, "params", []) or []:
            semantic = self.compute_builtin_attribute_name(param)
            if semantic is None:
                continue
            allowed_types = expected_types.get(semantic)
            if allowed_types is None:
                continue
            param_type = self.normalized_metal_type(getattr(param, "vtype", None))
            if param_type not in allowed_types:
                return False
        return True

    def compute_builtin_attribute_name(self, var):
        compute_builtins = {
            "thread_position_in_grid",
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "thread_index_in_threadgroup",
            "threads_per_threadgroup",
            "threadgroups_per_grid",
            "thread_index_in_simdgroup",
            "threads_per_simdgroup",
        }
        for attr in getattr(var, "attributes", []) or []:
            name = str(getattr(attr, "name", "")).lower()
            if name in compute_builtins:
                return name
        return None

    def function_host_name(self, func):
        for attr in getattr(func, "attributes", []) or []:
            if getattr(attr, "name", None) != "host_name":
                continue
            args = getattr(attr, "args", []) or []
            if not args:
                return None
            raw_name = str(args[0]).strip()
            if (
                len(raw_name) >= 2
                and raw_name[0] in {'"', "'"}
                and raw_name[-1] == raw_name[0]
            ):
                return raw_name[1:-1]
            return raw_name
        return None

    def function_semantic_attributes(self, func):
        return [
            attr
            for attr in getattr(func, "attributes", []) or []
            if getattr(attr, "name", None)
            not in (
                self.fragment_execution_attribute_names
                | self.function_metadata_attribute_names
            )
            and "::" not in str(getattr(attr, "name", ""))
        ]

    def generate_fragment_execution_layouts(self, func):
        layouts = []
        for attr in getattr(func, "attributes", []) or []:
            name = getattr(attr, "name", None)
            if name not in self.fragment_execution_attribute_names:
                continue
            if getattr(attr, "args", None):
                continue
            if name not in layouts:
                layouts.append(name)
        if not layouts:
            return ""
        return f"        layout({', '.join(layouts)}) in;\n"

    def discarded_expression_is_proven_side_effect_free(self, expression):
        if expression is None:
            return True
        if self.discarded_expression_may_read_volatile(expression):
            return False
        if isinstance(expression, (bool, int, float, str)):
            return True
        if isinstance(expression, VariableNode):
            return (
                not bool(getattr(expression, "vtype", None))
                and self.discarded_expression_metal_type(expression) is not None
            )
        if isinstance(expression, MemberAccessNode):
            return self.discarded_member_access_is_proven_data_member(
                expression
            ) and self.discarded_expression_is_proven_side_effect_free(
                expression.object
            )
        if isinstance(expression, ArrayAccessNode):
            return self.discarded_subscript_is_builtin(expression) and all(
                self.discarded_expression_is_proven_side_effect_free(part)
                for part in (expression.array, expression.index)
            )
        if isinstance(expression, CastNode):
            return (
                self.discarded_expression_is_proven_side_effect_free(
                    expression.expression
                )
                and self.metal_type_is_proven_builtin_value(expression.target_type)
                and self.discarded_expression_has_proven_builtin_value_type(
                    expression.expression
                )
            )
        if isinstance(expression, BinaryOpNode):
            return expression.op in {
                ",",
                "||",
                "&&",
                "|",
                "^",
                "&",
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "<<",
                ">>",
                "+",
                "-",
                "*",
                "/",
                "%",
            } and all(
                self.discarded_expression_is_proven_side_effect_free(part)
                and self.discarded_expression_has_proven_builtin_value_type(part)
                for part in (expression.left, expression.right)
            )
        if isinstance(expression, UnaryOpNode):
            if expression.op not in {"+", "-", "!", "~", "&", "*"}:
                return False
            if not self.discarded_expression_is_proven_side_effect_free(
                expression.operand
            ):
                return False
            operand_type = self.discarded_expression_metal_type(expression.operand)
            if expression.op == "*":
                return self.metal_pointer_pointee_type_once(operand_type) is not None
            return self.metal_type_is_proven_builtin_value(operand_type)
        if isinstance(expression, TernaryOpNode):
            return all(
                self.discarded_expression_is_proven_side_effect_free(part)
                and self.discarded_expression_has_proven_builtin_value_type(part)
                for part in (
                    expression.condition,
                    expression.true_expr,
                    expression.false_expr,
                )
            )
        if isinstance(expression, VectorConstructorNode):
            return self.metal_type_is_proven_builtin_value(
                expression.type_name
            ) and all(
                self.discarded_expression_is_proven_side_effect_free(argument)
                for argument in expression.args
            )
        return False

    def discarded_expression_may_read_volatile(self, expression):
        try:
            qualifiers = self.expression_metal_type_qualifiers(expression)
        except (TypeError, ValueError):
            return True
        if "volatile" in set(qualifiers or ()):
            return True
        if isinstance(expression, MemberAccessNode):
            member = self.discarded_member_declaration(expression)
            return member is not None and "volatile" in set(
                self.metal_declaration_type_qualifiers(member)
            )
        return False

    def discarded_expression_metal_type(self, expression):
        try:
            return self.expression_metal_type(expression)
        except ValueError:
            return None

    def discarded_expression_has_proven_builtin_value_type(self, expression):
        return self.metal_type_is_proven_builtin_value(
            self.discarded_expression_metal_type(expression)
        )

    def metal_type_is_proven_builtin_value(self, metal_type):
        if metal_type is None:
            return False
        try:
            value_type = self.metal_source_overload_value_type(metal_type)
            if value_type is None:
                return False
            if self.metal_pointer_pointee_type_once(value_type) is not None:
                return True
            normalized = self.normalized_metal_type(self.resolve_type_alias(value_type))
            if normalized in self.metal_scalar_arithmetic_types:
                return True
            vector = self.metal_vector_component_parts(value_type)
            if vector is not None:
                element_type, width, _width_text = vector
                return width is not None and self.metal_type_is_proven_builtin_value(
                    element_type
                )
            matrix = self.metal_matrix_type_parts(value_type)
            if matrix is not None:
                element_type, columns, rows, _columns_text, _rows_text = matrix
                return (
                    columns is not None
                    and rows is not None
                    and self.metal_type_is_proven_builtin_value(element_type)
                )
        except (TypeError, ValueError):
            return False
        return False

    def discarded_subscript_is_builtin(self, expression):
        try:
            selection = self.metal_indexed_type_selection(expression)
        except (TypeError, ValueError):
            return False
        return selection["kind"] in {"array", "pointer", "vector", "matrix"}

    def discarded_member_declaration(self, expression):
        object_type = self.discarded_expression_metal_type(expression.object)
        if object_type is None:
            return None
        try:
            value_type = self.metal_source_overload_value_type(object_type)
            if getattr(expression, "is_pointer", False):
                value_type = self.metal_pointer_pointee_type_once(value_type)
            if value_type is None:
                return None
            owner = self.normalized_metal_type(self.resolve_type_alias(value_type))
        except (TypeError, ValueError):
            return None
        declaration = self.struct_declarations.get(owner)
        if declaration is None:
            return None
        return next(
            (
                member
                for member in getattr(declaration, "members", []) or []
                if isinstance(member, VariableNode)
                and getattr(member, "name", None) == str(expression.member)
            ),
            None,
        )

    def discarded_member_access_is_proven_data_member(self, expression):
        if self.discarded_member_declaration(expression) is not None:
            return True
        object_type = self.discarded_expression_metal_type(expression.object)
        if object_type is None or getattr(expression, "is_pointer", False):
            return False
        try:
            return (
                self.metal_vector_component_parts(object_type) is not None
                and self.discarded_expression_metal_type(expression) is not None
            )
        except (TypeError, ValueError):
            return False

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, TypeAliasNode):
                # Register a block-local ``using`` or ``typedef`` so later
                # declarations and expressions resolve it in lexical order. The
                # alias itself produces no CrossGL statement.
                self.register_local_type_alias(stmt)
                code = code[: len(code) - 4 * indent]
                continue
            if isinstance(stmt, StructNode):
                self.register_local_aggregate(stmt)
                code = code[: len(code) - 4 * indent]
                continue
            if isinstance(stmt, VariableNode):
                self.local_integral_constant_bindings.pop(stmt.name, None)
                self.current_variable_types[stmt.name] = (
                    self.inferred_metal_declaration_type(stmt)
                )
                self.current_variable_type_qualifiers[stmt.name] = (
                    self.metal_declaration_type_qualifiers(stmt)
                )
                self.current_storage_texture_names.discard(stmt.name)
                self.current_structured_buffer_names.discard(stmt.name)
                if id(stmt) in self.storage_texture_declaration_ids:
                    self.current_storage_texture_names.add(stmt.name)
                if self.structured_buffer_pointer_type(stmt):
                    self.current_structured_buffer_names.add(stmt.name)
                constructor_array = self.generate_local_constructor_array_declaration(
                    stmt,
                    None,
                    is_main,
                    indent,
                )
                if constructor_array is not None:
                    code += f"{constructor_array}\n"
                    continue
                decl = self.format_decl(stmt, include_semantic=False)
                if decl.startswith("sampler ") and is_default_sampler_local(
                    stmt, getattr(self.current_function, "body", None)
                ):
                    # Default-constructed state is fixed state (:mod:`.default_samplers`).
                    code += f"const {decl} = sampler();\n"
                    continue
                if not getattr(stmt, "is_metal_constructor_storage", False):
                    initializer = vote_mask_default_initializer(
                        self, stmt
                    ) or self.generate_default_constructor_initializer(stmt, is_main)
                    if initializer is not None:
                        decl += f" = {initializer}"
                code += f"{decl};\n"
            elif isinstance(stmt, AssignmentNode):
                sparse_color = import_sparse_color_assignment(
                    self, stmt, is_main, indent
                )
                if sparse_color is not None:
                    code += sparse_color
                    continue
                declaration = getattr(stmt, "left", None)
                if isinstance(declaration, VariableNode) and getattr(
                    declaration, "vtype", None
                ):
                    self.local_integral_constant_bindings.pop(declaration.name, None)
                    inferred_type = self.inferred_metal_declaration_type(
                        declaration, stmt.right
                    )
                    inferred_qualifiers = (
                        self.inferred_metal_declaration_type_qualifiers(
                            declaration, stmt.right, inferred_type
                        )
                    )
                    self.current_variable_types[declaration.name] = inferred_type
                    self.current_variable_type_qualifiers[declaration.name] = (
                        inferred_qualifiers
                    )
                    self.current_storage_texture_names.discard(declaration.name)
                    self.current_structured_buffer_names.discard(declaration.name)
                    if id(declaration) in self.storage_texture_declaration_ids:
                        self.current_storage_texture_names.add(declaration.name)
                    if self.structured_buffer_pointer_type(declaration):
                        self.current_structured_buffer_names.add(declaration.name)
                    constructor_array = (
                        self.generate_local_constructor_array_declaration(
                            declaration,
                            stmt.right,
                            is_main,
                            indent,
                        )
                    )
                    if constructor_array is not None:
                        code += f"{constructor_array}\n"
                        continue
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode) and self.generate_metal_object_return(
                stmt
            ):
                code += self.generate_metal_object_return(stmt)
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        if getattr(
                            self.current_function,
                            "is_metal_constructor_factory",
                            False,
                        ):
                            result_name = self.current_function.constructor_result_name
                            code += f"return {self.render_identifier(result_name)};\n"
                        else:
                            code += "return;\n"
                    else:
                        pushed_context = (
                            self.push_materialized_constexpr_expression_context(
                                stmt.value,
                                required=False,
                            )
                        )
                        try:
                            converted_value = render_aggregate_scalar_conversion(
                                self,
                                self.current_function_return_type,
                                stmt.value,
                                "=",
                                is_main,
                            )
                            if converted_value is not None:
                                value = converted_value
                            elif (
                                self.metal_constructor_contract(
                                    self.current_function_return_type
                                )
                                is not None
                                or self.wide_vector_type_info(
                                    self.current_function_return_type,
                                    getattr(stmt, "source_location", None),
                                )
                                is not None
                                or isinstance(stmt.value, ArrayAccessNode)
                                or self.binary16_widening_profile is not None
                            ):
                                value = self.generate_initializer_value(
                                    stmt.value,
                                    is_main,
                                    self.current_function_return_type,
                                )
                            else:
                                value = self.generate_expression(stmt.value, is_main)
                        finally:
                            if pushed_context:
                                self.materialized_constexpr_expression_contexts.pop()
                        code += f"return {value};\n"
            elif isinstance(stmt, BinaryOpNode):
                # One renderer for every binary operator, so ``a = x, b = y;`` keeps the CrossGL comma spelling.
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif (
                isinstance(stmt, CastNode) and self.map_type(stmt.target_type) == "void"
            ):
                if self.discarded_expression_is_proven_side_effect_free(
                    stmt.expression
                ):
                    self.record_elided_parameter_uses(stmt.expression)
                    code = code[: len(code) - 4 * indent]
                else:
                    code += f"{self.generate_expression(stmt.expression, is_main)};\n"
            elif isinstance(stmt, BlockNode):
                code += "{\n"
                code += self.generate_scoped_function_body(
                    stmt.statements, indent + 1, is_main
                )
                code += "    " * indent + "}\n"
            elif isinstance(stmt, RangeForNode):
                code += self.generate_range_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, FunctionCallNode):
                lowered_callback = self.generate_callback_statement(
                    stmt, indent, is_main
                )
                lowered_barrier = self.generate_metal_barrier_statement(stmt, indent)
                if lowered_callback is not None:
                    code += lowered_callback
                elif lowered_barrier is not None:
                    code += lowered_barrier
                else:
                    code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, (MethodCallNode, CallNode)):
                lowered_mesh_call = self.generate_metal_mesh_statement(
                    stmt, indent, is_main
                ) or self.generate_metal_object_statement(stmt, is_main)
                if lowered_mesh_call is not None:
                    code += lowered_mesh_call
                else:
                    code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, PostfixOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
            elif isinstance(stmt, StaticAssertNode):
                cond = self.generate_expression(stmt.condition, is_main)
                if stmt.message is not None:
                    msg = (
                        stmt.message
                        if isinstance(stmt.message, str)
                        else self.generate_expression(stmt.message, is_main)
                    )
                    code += f"static_assert({cond}, {msg});\n"
                else:
                    code += f"static_assert({cond});\n"
            elif isinstance(stmt, EnumNode):
                code += self.generate_local_enum(stmt, indent, is_main)
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                expr = self.generate_expression(stmt, is_main)
                if expr:
                    code += f"{expr};\n"
                else:
                    code += self.refuse_import(
                        f"Metal statement of kind {type(stmt).__name__} has no CrossGL statement form"
                    )
        return code

    def generate_scoped_function_body(self, body, indent=0, is_main=False):
        """Render a nested lexical block without leaking local bindings."""
        previous_variable_types = self.current_variable_types
        previous_variable_type_qualifiers = self.current_variable_type_qualifiers
        previous_storage_texture_names = self.current_storage_texture_names
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_variable_types = dict(previous_variable_types)
        self.current_variable_type_qualifiers = dict(previous_variable_type_qualifiers)
        self.current_storage_texture_names = set(previous_storage_texture_names)
        self.current_structured_buffer_names = set(previous_structured_buffer_names)
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_type_alias_pointee_qualifiers = dict(
            self.type_alias_pointee_qualifiers
        )
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_unresolved_local_type_aliases = dict(
            self.unresolved_local_type_aliases
        )
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        previous_local_integral_constant_bindings = dict(
            self.local_integral_constant_bindings
        )
        self.template_binding_shadow_scopes.append(set())
        try:
            return self.generate_function_body(body, indent, is_main)
        finally:
            self.template_binding_shadow_scopes.pop()
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_type_qualifiers
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.type_alias_pointee_qualifiers = previous_type_alias_pointee_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.unresolved_local_type_aliases = previous_unresolved_local_type_aliases
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.local_integral_constant_bindings = (
                previous_local_integral_constant_bindings
            )

    def generate_callback_statement(self, call, indent, is_main=False):
        helper = str(call.name).rsplit("::", 1)[-1]
        if helper != "dispatch_bool":
            return None

        source_location = getattr(call, "source_location", None)
        if len(call.args) != 2:
            callback = next(
                (arg for arg in call.args if isinstance(arg, LambdaNode)), None
            )
            if callback is None:
                return None
            raise self.callback_lowering_error(
                helper,
                "expected a condition and one lambda callback",
                callback=callback,
                source_location=source_location,
            )
        if not isinstance(call.args[1], LambdaNode):
            return None

        callback = call.args[1]
        if callback.capture != "&":
            raise self.callback_lowering_error(
                helper,
                "callback must use reference-default capture",
                callback=callback,
                source_location=source_location,
            )
        if len(callback.params) != 1 or not callback.params[0].name:
            raise self.callback_lowering_error(
                helper,
                "callback must declare exactly one integral-constant parameter",
                callback=callback,
                source_location=source_location,
            )
        if callback.params[0].vtype != "auto":
            raise self.callback_lowering_error(
                helper,
                "callback parameter must use the deduced integral-constant type",
                callback=callback,
                source_location=source_location,
            )
        if getattr(callback, "return_type", None) not in (None, "void"):
            raise self.callback_lowering_error(
                helper,
                "callback return values cannot be represented by dispatch_bool",
                callback=callback,
                source_location=source_location,
            )
        if self.callback_contains_return(callback):
            raise self.callback_lowering_error(
                helper,
                "callback-local return statements cannot be inlined into the caller",
                callback=callback,
                source_location=source_location,
            )

        condition = self.generate_expression(call.args[0], is_main)
        parameter_name = callback.params[0].name
        true_body = self.generate_integral_constant_callback_body(
            callback.body, parameter_name, True, indent + 1, is_main
        )
        false_body = self.generate_integral_constant_callback_body(
            callback.body, parameter_name, False, indent + 1, is_main
        )
        branch_indent = "    " * indent
        return (
            f"if ({condition}) {{\n"
            f"{true_body}"
            f"{branch_indent}}} else {{\n"
            f"{false_body}"
            f"{branch_indent}}}\n"
        )

    def generate_integral_constant_callback_body(
        self, body, parameter_name, value, indent, is_main
    ):
        previous_variable_types = self.current_variable_types
        previous_variable_type_qualifiers = self.current_variable_type_qualifiers
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_type_alias_pointee_qualifiers = dict(
            self.type_alias_pointee_qualifiers
        )
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_unresolved_local_type_aliases = dict(
            self.unresolved_local_type_aliases
        )
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        previous_local_integral_constant_bindings = dict(
            self.local_integral_constant_bindings
        )
        previous_storage_texture_names = self.current_storage_texture_names
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_variable_types = dict(previous_variable_types)
        self.current_variable_type_qualifiers = dict(previous_variable_type_qualifiers)
        self.current_storage_texture_names = set(previous_storage_texture_names)
        self.current_structured_buffer_names = set(previous_structured_buffer_names)
        self.integral_constant_bindings.append({parameter_name: value})
        self.push_identifier_scope()
        try:
            return self.generate_function_body(body, indent, is_main)
        finally:
            self.pop_identifier_scope()
            self.integral_constant_bindings.pop()
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_type_qualifiers
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.type_alias_pointee_qualifiers = previous_type_alias_pointee_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.unresolved_local_type_aliases = previous_unresolved_local_type_aliases
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.local_integral_constant_bindings = (
                previous_local_integral_constant_bindings
            )
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names

    def callback_contains_return(self, callback):
        def visit(node):
            if isinstance(node, ReturnNode):
                return True
            if isinstance(node, LambdaNode) and node is not callback:
                return False
            return any(visit(child) for child in self.iter_ast_children(node))

        return any(visit(statement) for statement in callback.body)

    def integral_constant_binding(self, name):
        for bindings in reversed(self.integral_constant_bindings):
            if name in bindings:
                return bindings[name]
        return None

    def render_integral_constant_binding(self, name):
        value = self.integral_constant_binding(name)
        if value is None:
            return None
        return "true" if value else "false"
