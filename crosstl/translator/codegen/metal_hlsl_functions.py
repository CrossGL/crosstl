"""Metal emission of HLSL function and program-scope semantics.

HLSL passes ``out``/``inout`` arguments by copy-in/copy-out and value arrays
by value, allows static globals with run-time initializers, and lets programs
use names that Metal parses as keywords. These members lower that to Metal:
reference parameters with argument write-backs, thread-private globals passed
to the functions that use them, value-array parameter copies, renaming of
program identifiers that collide with Metal keywords, and the compute-only
program check.
"""

import re
from copy import deepcopy

from ...backend.common_ast import AssignmentNode as BackendAssignmentNode
from ...backend.common_ast import ReturnNode as BackendReturnNode
from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    ParameterNode,
    PointerType,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import format_c_style_array_declaration, split_array_type_suffix
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class MetalHlslFunctionMixin:
    """HLSL function-call and program-scope members of ``MetalCodeGen``."""

    METAL_VECTOR_COMPONENT_NAMES = frozenset("xyzwrgba")

    def metal_reference_parameter_qualifier(self, call, index):
        """Return ``out``/``inout`` when a user call writes through argument ``index``."""
        func_name = self.function_call_name(call)
        if not isinstance(func_name, str):
            return None
        overloads = self.function_overloads_by_name.get(func_name, [])
        argument_count = len(getattr(call, "args", []) or [])
        qualifiers = set()
        for overload in overloads:
            params = list(
                getattr(overload, "parameters", getattr(overload, "params", [])) or []
            )
            if len(params) != argument_count:
                continue
            names = self.parameter_qualifier_names(params[index])
            qualifiers.add(
                "inout" if "inout" in names else "out" if "out" in names else None
            )
        if len(qualifiers) != 1:
            return None
        return qualifiers.pop()

    def metal_argument_binds_thread_reference(self, expr):
        """Whether an lvalue can bind to a ``thread T&`` parameter directly.

        Locals and parameters live in the thread address space, as do their
        struct fields and array elements. Vector components, buffer elements,
        threadgroup or constant data and resource texels need a temporary.
        """
        node = expr
        while True:
            if isinstance(node, MemberAccessNode):
                member = str(node.member)
                layout_member = self.metal_hlsl_layout_member(node)
                if layout_member is not None and layout_member.converted:
                    return False
                object_type = self.map_type(self.expression_result_type(node.object))
                if (
                    member
                    and set(member) <= self.METAL_VECTOR_COMPONENT_NAMES
                    and self.vector_component_type(object_type) is not None
                    and not self.is_matrix_value_type(object_type)
                ):
                    return False
                node = node.object
            elif isinstance(node, ArrayAccessNode):
                array_type = self.map_type(self.expression_result_type(node.array))
                if self.vector_value_width(array_type) is not None:
                    return False
                node = node.array
            else:
                break
        name = self.expression_name(node)
        if not isinstance(name, str) or name not in self.local_variable_types:
            return False
        local_type = self.type_name_string(self.local_variable_types.get(name))
        return not (
            local_type
            and (
                self.is_structured_buffer_type(local_type)
                or "*" in str(local_type)
                or local_type.startswith(("device ", "constant ", "threadgroup "))
            )
        )

    def metal_expression_is_side_effect_free(self, expr):
        for node in self.iter_ast_nodes(expr):
            if isinstance(node, (FunctionCallNode, AssignmentNode)):
                return False
            if isinstance(node, UnaryOpNode) and str(
                getattr(node, "op", getattr(node, "operator", ""))
            ) in {"++", "--"}:
                return False
        return True

    def metal_statement_evaluated_expressions(self, stmt):
        """Expressions a simple statement evaluates exactly once."""
        if isinstance(stmt, (VariableNode, BackendVariableNode)):
            return [getattr(stmt, "initial_value", None)]
        if isinstance(stmt, (AssignmentNode, BackendAssignmentNode)):
            return [
                getattr(stmt, "target", getattr(stmt, "left", None)),
                getattr(stmt, "value", getattr(stmt, "right", None)),
            ]
        if isinstance(stmt, FunctionCallNode):
            return [stmt]
        if "ExpressionStatementNode" in type(stmt).__name__:
            return [getattr(stmt, "expression", None)]
        if isinstance(stmt, (ReturnNode, BackendReturnNode)):
            return [getattr(stmt, "value", None)]
        return []

    def metal_reference_argument_writebacks(self, stmt):
        """Find reference arguments that need copy-in/copy-out temporaries."""
        if not getattr(self, "function_overloads_by_name", None):
            return []
        writebacks = []
        visited = set()

        def visit(expr, conditional):
            if expr is None or isinstance(expr, (str, int, float, bool)):
                return
            if isinstance(expr, (list, tuple)):
                for item in expr:
                    visit(item, conditional)
                return
            # CrossGL nodes keep legacy alias attributes (``left``/``target``,
            # ``args``/``arguments``); visit each node once.
            if id(expr) in visited:
                return
            visited.add(id(expr))
            if isinstance(expr, BinaryOpNode):
                visit(expr.left, conditional)
                operator = self.map_operator(getattr(expr, "op", ""))
                visit(expr.right, conditional or operator in {"&&", "||"})
                return
            if isinstance(expr, TernaryOpNode):
                visit(expr.condition, conditional)
                visit(expr.true_expr, True)
                visit(expr.false_expr, True)
                return
            if isinstance(expr, FunctionCallNode):
                args = list(getattr(expr, "args", []) or [])
                for index, argument in enumerate(args):
                    qualifier = self.metal_reference_parameter_qualifier(expr, index)
                    parameter = (
                        None
                        if qualifier is None
                        else self.metal_reference_argument_parameter(expr, index)
                    )
                    if parameter is None or self.metal_argument_binds_thread_reference(
                        argument
                    ):
                        visit(argument, conditional)
                        continue
                    if conditional:
                        raise ValueError(
                            "Metal cannot pass a non-thread lvalue to the "
                            f"{qualifier} parameter of "
                            f"'{self.function_call_name(expr)}' in a conditionally "
                            "evaluated expression"
                        )
                    if not self.metal_expression_is_side_effect_free(argument):
                        raise ValueError(
                            "Metal reference argument of "
                            f"'{self.function_call_name(expr)}' must be free of "
                            "side effects to be copied in and out"
                        )
                    writebacks.append((argument, qualifier, parameter))
                    visit(argument, conditional)
                return
            for key, child in getattr(expr, "__dict__", {}).items():
                if key in {"parent", "annotations"}:
                    continue
                if isinstance(child, ASTNode) or isinstance(child, (list, tuple)):
                    visit(child, conditional)

        for expression in self.metal_statement_evaluated_expressions(stmt):
            visit(expression, False)
        if writebacks and isinstance(stmt, (ReturnNode, BackendReturnNode)):
            raise ValueError(
                "Metal cannot copy a reference argument back after a return "
                "statement; assign the call result to a local first"
            )
        return writebacks

    def metal_reference_argument_parameter(self, call, index):
        """Return the parameter that Metal lowers to a ``thread T&`` reference.

        Array and pointer parameters are Metal pointers rather than references:
        the callee writes through the argument, so it needs no temporary.
        """
        overload = self.function_overloads_by_name[self.function_call_name(call)][0]
        parameter = list(
            getattr(overload, "parameters", getattr(overload, "params", [])) or []
        )[index]
        raw_type = self.parameter_raw_type(parameter)
        if isinstance(raw_type, PointerType) or self.is_array_type_node(raw_type):
            return None
        return parameter

    def generate_statement_with_argument_writebacks(self, stmt, indent, writebacks):
        """Emit a statement whose reference arguments use copy-in/copy-out.

        CrossGL ``out``/``inout`` parameters have copy-in/copy-out semantics,
        while Metal lowers them to ``thread T&``, which cannot bind buffer
        elements, threadgroup data or vector components. Each such argument is
        copied into a thread temporary before the statement (``inout``) and
        stored back after it, preserving the source semantics.
        """
        indent_str = "    " * indent
        prologue = ""
        epilogue = ""
        substitutions = {}
        for argument, qualifier, parameter in writebacks:
            self.metal_writeback_temporary_count += 1
            name = f"__crossgl_ref{self.metal_writeback_temporary_count}"
            raw_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            declared_type = self.map_type(self.type_name_string(raw_type))
            source = self.generate_expression(argument)
            if qualifier == "inout":
                prologue += f"{indent_str}{declared_type} {name} = {source};\n"
            else:
                prologue += f"{indent_str}{declared_type} {name};\n"
            epilogue += f"{indent_str}{source} = {name};\n"
            substitutions[id(argument)] = name
        previous = self.metal_argument_substitutions
        self.metal_argument_substitutions = {**previous, **substitutions}
        self.metal_writeback_statement_ids.add(id(stmt))
        try:
            body = self.generate_statement(stmt, indent)
        finally:
            self.metal_writeback_statement_ids.discard(id(stmt))
            self.metal_argument_substitutions = previous
        return prologue + body + epilogue

    def collect_metal_runtime_initialized_constants(self, global_vars):
        """Program constants whose initializers Metal cannot evaluate statically.

        A Metal program-scope ``constant`` needs a constant expression; calls to
        functions (``cos``, ``asfloat``...) would require a global constructor,
        which Metal rejects when the library is linked. Such a constant, and
        any constant initialized from one, becomes an accessor function.
        """
        names = set()
        for node in global_vars or []:
            if not isinstance(node, VariableNode):
                continue
            name = self.metal_variable_name(node)
            initial_value = getattr(node, "initial_value", None)
            if not name or initial_value is None:
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if not qualifiers & {"const", "constant"}:
                continue
            if qualifiers & {"uniform", "extern", "device", "threadgroup"}:
                continue
            if self.is_program_scope_groupshared_global(node):
                continue
            vtype = self.type_name_string(getattr(node, "vtype", None))
            if (
                not vtype
                or "[" in vtype
                or self.is_array_type_node(getattr(node, "vtype", None))
            ):
                continue
            if (
                self.is_texture_or_image_resource_type(vtype)
                or self.is_sampler_type(vtype)
                or self.is_structured_buffer_type(vtype)
            ):
                continue
            if self.metal_initializer_requires_evaluation(initial_value, names):
                names.add(name)
        return names

    def metal_initializer_requires_evaluation(self, expr, runtime_constants):
        for node in self.iter_ast_nodes(expr):
            if isinstance(node, FunctionCallNode):
                callee = self.function_call_name(node)
                if isinstance(callee, str) and (
                    self.is_builtin_value_constructor_type(callee)
                    or callee in self.structs_by_name
                ):
                    continue
                return True
            if isinstance(node, IdentifierNode) and node.name in runtime_constants:
                return True
        return False

    def is_metal_thread_private_global(self, node):
        """A mutable ``static`` program-scope value: one copy per thread."""
        if not isinstance(node, VariableNode):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "static" not in qualifiers or qualifiers & {
            "const",
            "constant",
            "uniform",
            "extern",
        }:
            return False
        if self.is_program_scope_groupshared_global(node):
            return False
        vtype = self.type_name_string(getattr(node, "vtype", None))
        if not vtype:
            return False
        return not (
            self.is_texture_or_image_resource_type(vtype)
            or self.is_sampler_type(vtype)
            or self.is_structured_buffer_type(vtype)
        )

    def collect_metal_lowered_thread_private_globals(
        self, ast, target_stage, global_vars, all_functions
    ):
        """Give every invocation its own copy of each thread-private global.

        Metal program-scope variables live in the constant address space, so a
        per-thread mutable global becomes a value-initialized local of every
        entry that reaches it. Helper functions that reach it, directly or
        through calls, receive the entry's copy as a ``thread`` reference
        parameter appended to their signature and to every call site. Globals
        that a function shadows with a parameter or local of the same name, or
        whose type cannot be passed by reference, stay global and keep their
        diagnostic.
        """
        self.metal_forwarded_thread_private_globals = {}
        candidates = [
            node
            for node in global_vars or []
            if self.is_metal_thread_private_global(node)
        ]
        if not candidates:
            return {}, set()
        entries = self.metal_program_scope_groupshared_stage_entries(ast, target_stage)
        entry_ids = {id(func) for _stage_name, func, _stage_locals in entries}
        helpers = [
            func
            for func in all_functions or []
            if id(func) not in entry_ids and getattr(func, "name", None)
        ]
        helper_calls = {}
        for func in helpers:
            helper_calls.setdefault(func.name, set()).update(
                self.called_user_function_names(func)
            )
        lowered_by_function = {}
        lowered_ids = set()
        for node in candidates:
            name = self.metal_variable_name(node)
            if not name:
                continue
            reaching_helpers = {
                func.name
                for func in helpers
                if self.node_uses_identifier(getattr(func, "body", []), name)
            }
            changed = True
            while changed:
                changed = False
                for helper_name, calls in helper_calls.items():
                    if helper_name not in reaching_helpers and (
                        calls & reaching_helpers
                    ):
                        reaching_helpers.add(helper_name)
                        changed = True
            reference_parameter = None
            if reaching_helpers:
                reference_parameter = self.metal_thread_private_reference_parameter(
                    node, name
                )
                if reference_parameter is None or any(
                    name in self.metal_function_declared_names(func)
                    for func in helpers
                    if func.name in reaching_helpers
                ):
                    continue
            used_entries = []
            blocked = False
            for _stage_name, func, stage_locals in entries:
                if not (
                    self.node_uses_identifier(getattr(func, "body", []), name)
                    or self.called_user_function_names(func) & reaching_helpers
                ):
                    continue
                if name in self.metal_entry_local_names(func, stage_locals):
                    blocked = True
                    break
                used_entries.append(func)
            if blocked:
                continue
            for helper_name in sorted(reaching_helpers):
                self.metal_forwarded_thread_private_globals.setdefault(
                    helper_name, []
                ).append((name, reference_parameter, getattr(node, "vtype", None)))
            lowered = deepcopy(node)
            lowered.qualifiers = [
                qualifier
                for qualifier in getattr(lowered, "qualifiers", []) or []
                if str(qualifier).lower() != "static"
            ]
            if getattr(lowered, "initial_value", None) is None:
                # Source thread-private globals start zeroed for each thread.
                lowered.initial_value = ArrayLiteralNode([])
            for func in used_entries:
                lowered_by_function.setdefault(id(func), []).append(lowered)
            lowered_ids.add(id(node))
        return lowered_by_function, lowered_ids

    def metal_function_declared_names(self, func):
        names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
            if getattr(param, "name", None)
        }
        return names | self.metal_entry_local_names(func)

    def metal_thread_private_reference_parameter(self, node, name):
        """Return a ``thread`` reference declaration for a forwarded global."""
        vtype = getattr(node, "vtype", None)
        sizes = []
        while isinstance(vtype, ArrayType):
            size = getattr(vtype, "size", None)
            if size is None:
                return None
            sizes.append(self.expression_to_string(size))
            vtype = vtype.element_type
        type_name = self.type_name_string(vtype)
        if not type_name or "[" in type_name:
            return None
        mapped_type = self.map_type(type_name)
        if not mapped_type:
            return None
        if sizes:
            suffix = "".join(f"[{size}]" for size in sizes)
            return f"thread {mapped_type} (&{name}){suffix}"
        return f"thread {mapped_type}& {name}"

    def append_required_thread_private_global_parameters(self, params_str, func_name):
        forwarded = getattr(self, "metal_forwarded_thread_private_globals", {}).get(
            func_name, []
        )
        if not forwarded:
            return params_str
        declarations = [declaration for _name, declaration, _vtype in forwarded]
        for name, _declaration, vtype in forwarded:
            self.local_variable_types[name] = self.type_name_string(vtype)
        if params_str:
            return f"{params_str}, {', '.join(declarations)}"
        return ", ".join(declarations)

    def required_function_thread_private_global_argument_names(self, func_name):
        return [
            name
            for name, _declaration, _vtype in getattr(
                self, "metal_forwarded_thread_private_globals", {}
            ).get(func_name, [])
        ]

    def metal_generated_companion_name(self, base_name):
        """A deterministic generated name that no program identifier uses."""
        used = getattr(self, "metal_program_identifier_names", set())
        candidate = base_name
        while candidate in used:
            candidate = f"{candidate}_"
        return candidate

    METAL_RESERVED_PROGRAM_IDENTIFIERS = frozenset(
        {
            "kernel",
            "vertex",
            "fragment",
            "device",
            "constant",
            "thread",
            "threadgroup",
            "threadgroup_imageblock",
            "ray_data",
            "object_data",
        }
    )

    def rename_metal_reserved_program_identifiers(self, ast):
        """Rename program-scope variables spelled like Metal keywords.

        Program-scope resources and values become entry parameters or
        declarations whose names Metal parses as keywords. Every declaration
        and reference with the name is renamed on a copy of the AST.
        """
        reserved = {
            getattr(node, "name", None)
            for node in getattr(ast, "global_variables", []) or []
        } & self.METAL_RESERVED_PROGRAM_IDENTIFIERS
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            for member in getattr(cbuffer, "members", []) or []:
                if (
                    getattr(member, "name", None)
                    in self.METAL_RESERVED_PROGRAM_IDENTIFIERS
                ):
                    reserved.add(member.name)
        if not reserved:
            return ast
        ast = deepcopy(ast)
        used = self.collect_metal_program_identifier_names(ast)
        renames = {}
        for name in sorted(reserved):
            candidate = f"{name}_"
            while candidate in used:
                candidate = f"{candidate}_"
            used.add(candidate)
            renames[name] = candidate
        for node in self.iter_unique_ast_nodes(ast):
            if isinstance(node, (IdentifierNode, VariableNode, ParameterNode)):
                new_name = renames.get(getattr(node, "name", None))
                if new_name is not None:
                    node.name = new_name
        return ast

    def collect_metal_program_identifier_names(self, ast):
        names = set()
        for node in self.iter_unique_ast_nodes(ast):
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
        return names

    # Builtins whose arguments are written through output references.
    METAL_VALUE_ARRAY_WRITING_BUILTINS = frozenset({"frexp", "modf", "sincos"})

    def metal_value_array_parameter_dimensions(self, raw_param_type, node, mapped):
        """Return the dimension texts of a copyable ``in`` array parameter.

        An ``in`` array parameter has value semantics: writes inside the
        function must not reach the caller's array. Metal array parameters
        decay to pointers, so such a parameter is received under another name
        and copied into a local array that keeps the source name.
        """
        if not self.is_array_type_node(raw_param_type):
            return None
        if "in" not in self.parameter_qualifier_names(node):
            return None
        if self.resource_array_parameter(raw_param_type, node) is not None:
            return None
        if self.is_structured_buffer_type(raw_param_type):
            return None
        if self.array_indirect_element_type_node(raw_param_type) is not None:
            return None
        _base_type, suffix = split_array_type_suffix(mapped)
        dimensions = re.findall(r"\[([^\[\]]*)\]", suffix)
        if not dimensions or any(not dimension.strip() for dimension in dimensions):
            return None
        return [dimension.strip() for dimension in dimensions]

    def metal_value_array_parameter_is_mutated(self, func, name):
        """Return whether a function body may write to the array ``name``."""
        for node in self.iter_ast_nodes(getattr(func, "body", None)):
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                if self.assignment_target_root_name(target) == name:
                    return True
            elif isinstance(node, UnaryOpNode):
                if getattr(node, "operator", None) in {
                    "++",
                    "--",
                    "&",
                } and (self.assignment_target_root_name(node.operand) == name):
                    return True
            elif isinstance(node, FunctionCallNode):
                if self.metal_call_may_write_argument_root(node, name):
                    return True
        return False

    def metal_call_may_write_argument_root(self, call, name):
        callee_name = self.function_call_name(call)
        arguments = list(
            getattr(call, "arguments", None) or getattr(call, "args", None) or []
        )
        overloads = (
            self.function_overloads_by_name.get(callee_name, []) if callee_name else []
        )
        for index, argument in enumerate(arguments):
            if self.assignment_target_root_name(argument) != name:
                continue
            if callee_name in self.METAL_VALUE_ARRAY_WRITING_BUILTINS:
                return True
            for callee in overloads:
                parameters = list(
                    getattr(callee, "parameters", getattr(callee, "params", [])) or []
                )
                if index >= len(parameters):
                    continue
                qualifiers = self.parameter_qualifier_names(parameters[index])
                if qualifiers & {"out", "inout"}:
                    return True
                if self.is_array_type_node(
                    self.parameter_raw_type(parameters[index])
                ) and not qualifiers & {"in", "const", "constant"}:
                    # A pointer-like array parameter may write the caller's
                    # elements, which here are the elements of the copy.
                    return True
        return False

    def metal_value_array_input_name(self, name, reserved_names):
        candidate = f"{name}_crosstl_input"
        suffix = 2
        while candidate in reserved_names:
            candidate = f"{name}_crosstl_input{suffix}"
            suffix += 1
        return candidate

    def metal_value_array_copy_statements(self, mapped, name, input_name, dimensions):
        indices = [f"crosstl_{name}_i{level}" for level in range(len(dimensions))]
        subscript = "".join(f"[{index}]" for index in indices)
        statement = f"{name}{subscript} = {input_name}{subscript};"
        for level in reversed(range(len(dimensions))):
            index = indices[level]
            statement = (
                f"for (int {index} = 0; {index} < int({dimensions[level]}); "
                f"++{index}) {{ {statement} }}"
            )
        return [f"{format_c_style_array_declaration(mapped, name)};", statement]

    def generate_function_value_argument(self, func_name, index, arg, param_type):
        """Render a user-function argument with source implicit conversions.

        Only unambiguous calls convert: the function has one overload and the
        parameter is passed by value.
        """
        if (
            param_type is None
            or len(self.function_overloads_by_name.get(func_name, [])) != 1
        ):
            return self.generate_expression(arg)
        if self.metal_reference_parameter_qualifier_for(func_name, index) is not None:
            return self.generate_expression(arg)
        if (
            self.is_metal_bfloat_storage_read(arg)
            or self.is_metal_bfloat_user_call(arg)
        ) and (self.map_type(self.type_name_string(param_type)) == "bfloat"):
            # A stored bfloat value or the result of a user function declared
            # to return bfloat already has the parameter type; only computed
            # values (native math yields float or half) need the explicit
            # construction that expected-type boundaries apply.
            return self.generate_expression(arg)
        return self.generate_expression_with_expected(arg, param_type)

    def is_metal_bfloat_user_call(self, expr):
        """Whether ``expr`` calls a user function whose result is bfloat."""
        if not isinstance(expr, FunctionCallNode):
            return False
        if self.function_call_name(expr) not in self.user_function_names:
            return False
        result_type = self.expression_result_type(expr)
        return result_type is not None and self.map_type(result_type) == "bfloat"

    def metal_reference_parameter_qualifier_for(self, func_name, index):
        overloads = self.function_overloads_by_name.get(func_name, [])
        if len(overloads) != 1:
            return None
        params = list(
            getattr(overloads[0], "parameters", getattr(overloads[0], "params", []))
            or []
        )
        if index >= len(params):
            return None
        names = self.parameter_qualifier_names(params[index])
        if "inout" in names:
            return "inout"
        if "out" in names:
            return "out"
        return None

    def metal_program_is_compute_only(self, ast, target_stage):
        stage_names = set()
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            if getattr(stage, "entry_point", None) is not None:
                stage_names.add(stage_name)
        for func in getattr(ast, "functions", []) or []:
            qualifier_name = function_stage_name(func)
            if not qualifier_name:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                stage_names.add(qualifier_name)
        return stage_names == {"compute"}
