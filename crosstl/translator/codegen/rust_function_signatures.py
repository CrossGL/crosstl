"""Rust function signatures: stage attributes, lifetimes, inferred generic operator bounds, mutable bindings and parameters."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import RustRefusal
from .host_workgroup_size_ids import is_host_sized_workgroup
from .specialized_workgroup_size import require_fixed_workgroup_size
from .stage_utils import compute_local_size
from .rust_exact_lookups import call_names_resolving_to
from .rust_memory_atomics import rust_memory_atomic_location_roots


class RustFunctionSignatureMixin:
    """Function-signature members of ``RustCodeGen``."""

    def is_rust_shared_variable(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return bool(qualifiers & {"shared", "workgroup", "groupshared"})

    def generate_rust_stage_attribute(self, attribute_name):
        """Render a host-compatible Rust shader stage attribute."""
        return f'#[cfg_attr(feature = "crossgl_gpu", {attribute_name})]\n'

    def generate_rust_compute_stage_attribute(self, func, stage_node=None):
        """The rust-gpu compute entry attribute with the workgroup size (``@numthreads`` or the stage layout)."""
        numthreads = self.rust_compute_stage_numthreads(func)
        if numthreads is None:
            config = getattr(stage_node, "execution_config", None)
            require_fixed_workgroup_size(
                config, "Rust", "rust-gpu takes the workgroup size as literals"
            )
            if is_host_sized_workgroup(config):
                raise RustRefusal(
                    "Rust compute stage needs a workgroup size: rust-gpu takes it as literals"
                )
            numthreads = compute_local_size(config)
        threads = ", ".join(str(value) for value in numthreads)
        return f"#[spirv(compute(threads({threads})))]\n"

    def rust_compute_stage_numthreads(self, func):
        """Return CrossGL compute @numthreads dimensions for rust-gpu."""
        arguments = self.rust_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return None
        if len(arguments) > 3:
            raise RustRefusal(
                "Rust compute stage numthreads requires at most three arguments"
            )
        values = [self.attribute_argument_text(arg) for arg in arguments]
        for index, argument in enumerate(arguments):
            value = self.literal_int_value(argument)
            if value is not None and value <= 0:
                raise RustRefusal(
                    f"Rust compute stage numthreads value {index + 1} "
                    "must be positive"
                )
        while len(values) < 3:
            values.append("1")
        return tuple(values[:3])

    def rust_compute_global_invocation_id_parameter_bridge(
        self, parameter, param_type, semantic, emitted_param_names, parameter_names
    ):
        if semantic is None:
            return None
        semantic_key = self.rust_stage_parameter_semantic_key(semantic)
        if semantic_key not in {"gl_globalinvocationid", "SV_DISPATCHTHREADID"}:
            return None

        mapped_type = self.map_type(param_type)
        if mapped_type not in {"u32", "Vec2<u32>"}:
            return None

        source_name = getattr(parameter, "name", None)
        if not isinstance(source_name, str) or not source_name:
            return None

        local_name = self.rust_identifier(source_name)
        used_names = set(emitted_param_names)
        used_names.update(self.rust_identifier(name) for name in parameter_names)
        used_names.update(self.rust_identifier(name) for name in self.variable_types)
        used_names.update(
            self.rust_identifier(name) for name in self.local_variable_names
        )
        native_name = self.rust_unique_local_identifier(
            f"{local_name}_globalInvocationID", used_names
        )
        if mapped_type == "u32":
            alias_value = f"{native_name}.x"
        else:
            alias_value = f"Vec2::<u32>::new({native_name}.x, {native_name}.y)"
        let_keyword = "let mut" if source_name in self.current_mutated_names else "let"
        local_alias = f"{let_keyword} {local_name}: {mapped_type} = {alias_value};"
        return native_name, "Vec3<u32>", local_alias

    def rust_unique_local_identifier(self, candidate, used_names):
        candidate = self.rust_identifier(candidate)
        while candidate in used_names or candidate in self.rust_reserved_identifiers():
            candidate = f"{candidate}_"
        return candidate

    def function_reference_return_lifetime(self, func, return_type, param_types):
        if self.reference_type_parts_for_type(return_type) is None:
            return None

        reference_param_count = sum(
            1
            for param_type in param_types
            if self.reference_type_parts_for_type(param_type) is not None
        )
        if reference_param_count <= 1:
            return None

        generic_names = set(self.generic_param_names(func))
        for lifetime in ("'a", "'b", "'cgl_ref"):
            if lifetime not in generic_names:
                return lifetime
        return "'cgl_ref"

    def format_function_generic_params(
        self,
        node,
        extra_constraints=None,
        lifetime=None,
    ):
        generic_params = self.format_generic_params(
            node,
            extra_constraints=extra_constraints,
        )
        if lifetime is None:
            return generic_params
        if not generic_params:
            return f"<{lifetime}>"
        return f"<{lifetime}, {generic_params[1:-1]}>"

    def map_function_parameter_type_with_lifetime(self, param_type, lifetime):
        if lifetime is not None and self.reference_type_parts_for_type(param_type):
            return self.map_type_with_lifetime(param_type, lifetime)
        return self.map_function_parameter_type(param_type)

    def combined_function_generic_constraints(self, func):
        """Return explicit and inferred generic constraints for one function."""
        getattr(func, "name", None)
        cache_key = id(func)
        if cache_key in self.user_function_generic_constraint_cache:
            return self.user_function_generic_constraint_cache[cache_key]
        if cache_key in self.active_generic_constraint_functions:
            return {}

        self.active_generic_constraint_functions.add(cache_key)
        try:
            constraints = {}
            for param in getattr(func, "generic_params", []) or []:
                name = self.generic_param_name(param)
                if name:
                    constraints[name] = self.generic_param_constraints(param)

            inferred = self.infer_function_generic_operator_constraints(func)
            for name, bounds in inferred.items():
                for bound in bounds:
                    self.add_generic_constraint(constraints, name, bound)
        finally:
            self.active_generic_constraint_functions.remove(cache_key)

        self.user_function_generic_constraint_cache[cache_key] = constraints
        return constraints

    def infer_function_generic_operator_constraints(self, func):
        """Infer Rust operator trait bounds needed by generic function bodies."""
        generic_names = set(self.generic_param_names(func))
        if not generic_names:
            return {}

        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        saved_generic_param_names = self.current_generic_param_names
        constraints = {name: [] for name in generic_names}

        try:
            self.variable_types = {}
            self.local_variable_names = set()
            self.current_generic_param_names = generic_names
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                self.register_variable_type(
                    param.name,
                    self.function_parameter_type(param),
                    scope="local",
                )
            self.collect_generic_operator_constraints(
                getattr(func, "body", []),
                generic_names,
                constraints,
            )
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names
            self.current_generic_param_names = saved_generic_param_names

        return {name: bounds for name, bounds in constraints.items() if bounds}

    def collect_mutated_binding_names(self, node):
        """Collect local binding names that need `mut` because they are assigned."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if hasattr(current, "statements"):
                collect(current.statements)
                return

            if isinstance(current, AssignmentNode):
                root_name = self.assignment_target_root_name(current.target)
                if root_name:
                    names.add(root_name)
                # The target's indices can mutate too: a[k++] = v, a[f(k)] += v with an inout parameter.
                collect(current.target)
                collect(current.value)
                return

            if isinstance(current, VariableNode):
                initial_value = getattr(current, "initial_value", None)
                declared_type = self.get_variable_type(current)
                reference_parts = self.reference_type_parts_for_type(declared_type)
                if reference_parts is not None and reference_parts[0]:
                    names.update(
                        self.mutable_reference_borrow_root_names(initial_value)
                    )
                collect(initial_value)
                return

            if isinstance(current, ExpressionStatementNode):
                collect(current.expression)
                return

            if isinstance(current, ReturnNode):
                collect(current.value)
                return

            if isinstance(current, IfNode):
                collect(current.condition)
                collect(current.then_branch)
                collect(current.else_branch)
                return

            if isinstance(current, ForNode):
                collect(current.init)
                collect(current.condition)
                collect(current.update)
                collect(current.body)
                return

            if isinstance(current, ForInNode):
                collect(current.iterable)
                collect(current.body)
                return

            if isinstance(current, (WhileNode, DoWhileNode)):
                collect(current.condition)
                collect(current.body)
                return

            if isinstance(current, LoopNode):
                collect(current.body)
                return

            if isinstance(current, MatchNode):
                collect(current.expression)
                for arm in getattr(current, "arms", []) or []:
                    collect(getattr(arm, "guard", None))
                    collect(getattr(arm, "body", None))
                return

            if isinstance(current, SwitchNode):
                collect(getattr(current, "expression", None))
                for case in getattr(current, "cases", []) or []:
                    collect(getattr(case, "value", None))
                    collect(getattr(case, "statements", []))
                collect(getattr(current, "default_case", None))
                return

            if isinstance(current, UnaryOpNode):
                operator = self.map_operator(
                    getattr(current, "operator", getattr(current, "op", None))
                )
                if operator in {"++", "--"}:
                    root_name = self.assignment_target_root_name(current.operand)
                    if root_name:
                        names.add(root_name)
                collect(current.operand)
                return

            if isinstance(current, BinaryOpNode):
                collect(current.left)
                collect(current.right)
                return

            if isinstance(current, TernaryOpNode):
                collect(current.condition)
                collect(current.true_expr)
                collect(current.false_expr)
                return

            if isinstance(current, FunctionCallNode):
                func_expr = getattr(current, "function", getattr(current, "name", None))
                func_name = self.function_call_name(func_expr)
                args = getattr(current, "arguments", getattr(current, "args", []))
                for index, argument in enumerate(args or []):
                    param_type = self.user_function_param_type(func_name, index)
                    reference_parts = self.reference_type_parts_for_type(param_type)
                    if reference_parts is None or not reference_parts[0]:
                        continue
                    names.update(self.mutable_reference_borrow_root_names(argument))
                names.update(self.rust_output_argument_roots(func_name, args))
                names.update(rust_memory_atomic_location_roots(self, func_name, args))
                collect(current.function)
                collect(args)
                return

            if isinstance(current, MemberAccessNode):
                collect(current.object_expr)
                return

            if isinstance(current, PointerAccessNode):
                collect(current.pointer_expr)
                return

            if isinstance(current, ArrayAccessNode):
                collect(current.array_expr)
                collect(current.index_expr)
                return

            if isinstance(current, ConstructorNode):
                collect(getattr(current, "arguments", []))
                collect(list((getattr(current, "named_arguments", {}) or {}).values()))
                return

            if isinstance(current, ArrayLiteralNode):
                collect(getattr(current, "elements", []))

        collect(node)
        return names

    def assignment_target_root_name(self, target):
        """Return the binding name mutated by an assignment target."""
        if isinstance(target, IdentifierNode):
            return target.name
        if isinstance(target, VariableNode):
            return target.name
        if isinstance(target, str):
            return target if target.isidentifier() else None
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root_name(target.object_expr)
        if isinstance(target, PointerAccessNode):
            return self.assignment_target_root_name(target.pointer_expr)
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root_name(target.array_expr)
        return None

    def mutable_reference_borrow_root_names(self, expr):
        """Return owned roots that must be mutable for an `&mut` borrow."""
        if expr is None:
            return set()

        source_reference = self.reference_type_parts_for_type(
            self.expression_result_type(expr)
        )
        if source_reference is not None:
            return set()

        if isinstance(expr, TernaryOpNode):
            return self.mutable_reference_borrow_root_names(
                expr.true_expr
            ) | self.mutable_reference_borrow_root_names(expr.false_expr)

        if isinstance(expr, MatchNode):
            names = set()
            subject_type = self.expression_result_type(
                getattr(expr, "expression", None)
            )
            for arm in getattr(expr, "arms", []) or []:
                saved_variable_types = self.variable_types.copy()
                saved_local_variable_names = self.local_variable_names.copy()
                try:
                    self.register_match_pattern_bindings(
                        getattr(arm, "pattern", None),
                        subject_type,
                    )
                    names.update(
                        self.mutable_reference_borrow_root_names(
                            self.match_arm_tail_expression(getattr(arm, "body", None))
                        )
                    )
                finally:
                    self.variable_types = saved_variable_types
                    self.local_variable_names = saved_local_variable_names
            return names

        if isinstance(expr, BlockNode):
            explicit_tail = getattr(expr, "expression", None)
            if explicit_tail is not None:
                return self.mutable_reference_borrow_root_names(explicit_tail)
            statements = self.statement_list(expr)
            if not statements:
                return set()
            tail_expression = self.block_tail_expression(statements[-1])
            if tail_expression is not None:
                return self.mutable_reference_borrow_root_names(tail_expression)
            return set()

        root_name = self.assignment_target_root_name(expr)
        return {root_name} if root_name else set()

    def match_arm_tail_expression(self, body):
        statements = self.statement_list(body)
        if not statements:
            return None
        return self.block_tail_expression(statements[-1])

    def generate_borrow_operand_expression(self, expr):
        """Render a borrow operand without value-context clones."""
        self.assignment_lhs_depth += 1
        try:
            return self.generate_expression(expr)
        finally:
            self.assignment_lhs_depth -= 1

    def collect_generic_operator_constraints(self, node, generic_names, constraints):
        if node is None:
            return

        if isinstance(node, list):
            for item in node:
                self.collect_generic_operator_constraints(
                    item, generic_names, constraints
                )
            return

        if hasattr(node, "statements"):
            for statement in node.statements:
                self.collect_generic_operator_constraints(
                    statement, generic_names, constraints
                )
            return

        if isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", None)
            self.collect_generic_operator_constraints(
                initial_value, generic_names, constraints
            )
            declared_type = self.get_variable_type(node)
            variable_type = declared_type or self.expression_result_type(initial_value)
            if variable_type is not None:
                self.register_variable_type(node.name, variable_type, scope="local")
            return

        if isinstance(node, ExpressionStatementNode):
            self.collect_generic_operator_constraints(
                node.expression, generic_names, constraints
            )
            return

        if isinstance(node, ReturnNode):
            self.collect_generic_operator_constraints(
                node.value, generic_names, constraints
            )
            return

        if isinstance(node, AssignmentNode):
            self.collect_generic_operator_constraints(
                node.target, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.value, generic_names, constraints
            )
            return

        if isinstance(node, IfNode):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.then_branch, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.else_branch, generic_names, constraints
            )
            return

        if isinstance(node, ForNode):
            self.collect_generic_operator_constraints(
                node.init, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.update, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, ForInNode):
            self.collect_generic_operator_constraints(
                node.iterable, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, (WhileNode, DoWhileNode)):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, LoopNode):
            self.collect_generic_operator_constraints(
                node.body, generic_names, constraints
            )
            return

        if isinstance(node, MatchNode):
            subject_expr = getattr(node, "expression", None)
            subject_type = self.expression_result_type(subject_expr)
            self.collect_generic_operator_constraints(
                subject_expr, generic_names, constraints
            )
            for arm in getattr(node, "arms", []) or []:
                saved_variable_types = self.variable_types.copy()
                saved_local_variable_names = self.local_variable_names.copy()
                try:
                    self.register_match_pattern_bindings(
                        getattr(arm, "pattern", None), subject_type
                    )
                    self.collect_generic_operator_constraints(
                        getattr(arm, "guard", None), generic_names, constraints
                    )
                    self.collect_generic_operator_constraints(
                        getattr(arm, "body", None), generic_names, constraints
                    )
                finally:
                    self.variable_types = saved_variable_types
                    self.local_variable_names = saved_local_variable_names
            return

        if isinstance(node, BinaryOpNode):
            self.collect_generic_operator_constraints(
                node.left, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.right, generic_names, constraints
            )
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", None))
            )
            self.add_generic_operator_constraints(
                operator,
                self.expression_result_type(node.left),
                self.expression_result_type(node.right),
                generic_names,
                constraints,
            )
            return

        if isinstance(node, UnaryOpNode):
            self.collect_generic_operator_constraints(
                node.operand, generic_names, constraints
            )
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", None))
            )
            operand_type = self.expression_result_type(node.operand)
            if operator == "-" and operand_type in generic_names:
                self.add_generic_constraint(
                    constraints,
                    operand_type,
                    f"core::ops::Neg<Output = {operand_type}>",
                )
            return

        if isinstance(node, TernaryOpNode):
            self.collect_generic_operator_constraints(
                node.condition, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.true_expr, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.false_expr, generic_names, constraints
            )
            return

        if isinstance(node, FunctionCallNode):
            self.collect_generic_operator_constraints(
                node.function, generic_names, constraints
            )
            for argument in getattr(node, "arguments", []) or []:
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            self.add_builtin_function_call_constraints(node, generic_names, constraints)
            self.add_user_function_call_constraints(node, generic_names, constraints)
            return

        if isinstance(node, MemberAccessNode):
            self.collect_generic_operator_constraints(
                node.object_expr, generic_names, constraints
            )
            return

        if isinstance(node, ConstructorNode):
            for argument in getattr(node, "arguments", []) or []:
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            for argument in (getattr(node, "named_arguments", {}) or {}).values():
                self.collect_generic_operator_constraints(
                    argument, generic_names, constraints
                )
            return

        if isinstance(node, ArrayLiteralNode):
            for element in getattr(node, "elements", []) or []:
                self.collect_generic_operator_constraints(
                    element, generic_names, constraints
                )
            return

        if isinstance(node, ArrayAccessNode):
            self.collect_generic_operator_constraints(
                node.array_expr, generic_names, constraints
            )
            self.collect_generic_operator_constraints(
                node.index_expr, generic_names, constraints
            )

    def add_builtin_function_call_constraints(self, node, generic_names, constraints):
        func_name = self.function_call_name(getattr(node, "function", None))
        if self.is_user_defined_function(func_name):
            return

        if func_name not in call_names_resolving_to(self.function_map, {"sqrt"}):
            return

        arguments = getattr(node, "arguments", []) or []
        if len(arguments) != 1:
            return

        argument_type = self.expression_result_type(arguments[0])
        if argument_type in generic_names:
            self.add_generic_constraint(constraints, argument_type, "CglSqrt")
            self.required_generic_math_traits.add("CglSqrt")

    def add_user_function_call_constraints(self, node, generic_names, constraints):
        func_name = self.function_call_name(getattr(node, "function", None))
        callee = self.user_function_nodes.get(func_name)
        if callee is None:
            return

        substitutions = self.infer_call_generic_substitutions(callee, node)
        if not substitutions:
            return

        callee_constraints = self.combined_function_generic_constraints(callee)
        for callee_param, caller_type in substitutions.items():
            if caller_type not in generic_names:
                continue
            for bound in callee_constraints.get(callee_param, []):
                self.add_generic_constraint(
                    constraints,
                    caller_type,
                    self.substitute_generic_bound(bound, callee_param, caller_type),
                )

    def function_call_name(self, function_expr):
        if isinstance(function_expr, IdentifierNode):
            return function_expr.name
        if isinstance(function_expr, str):
            return function_expr
        return getattr(function_expr, "name", None)

    def infer_call_generic_substitutions(self, callee, call_node):
        callee_generic_names = self.generic_param_names(callee)
        if not callee_generic_names:
            return {}

        substitutions = {}
        param_types = [
            self.function_parameter_type(param)
            for param in getattr(callee, "parameters", getattr(callee, "params", []))
        ]
        arg_types = [
            self.expression_result_type(argument)
            for argument in getattr(call_node, "arguments", []) or []
        ]

        for param_type, arg_type in zip(param_types, arg_types):
            self.collect_generic_substitutions_from_type(
                param_type,
                arg_type,
                set(callee_generic_names),
                substitutions,
            )

        return substitutions

    def collect_generic_substitutions_from_type(
        self, param_type, arg_type, callee_generic_names, substitutions
    ):
        if param_type is None or arg_type is None:
            return
        param_type = (
            self.convert_type_node_to_string(param_type)
            if hasattr(param_type, "name") or hasattr(param_type, "element_type")
            else str(param_type)
        )
        arg_type = (
            self.convert_type_node_to_string(arg_type)
            if hasattr(arg_type, "name") or hasattr(arg_type, "element_type")
            else str(arg_type)
        )

        if param_type in callee_generic_names:
            substitutions.setdefault(param_type, arg_type)
            return

        param_base, param_args = self.generic_type_parts(param_type)
        arg_base, arg_args = self.generic_type_parts(arg_type)
        if param_base != arg_base or len(param_args) != len(arg_args):
            return

        for nested_param, nested_arg in zip(param_args, arg_args):
            self.collect_generic_substitutions_from_type(
                nested_param,
                nested_arg,
                callee_generic_names,
                substitutions,
            )

    def substitute_generic_bound(self, bound, source_name, target_name):
        return str(bound).replace(f"Output = {source_name}", f"Output = {target_name}")

    def add_generic_operator_constraints(
        self, operator, left_type, right_type, generic_names, constraints
    ):
        generic_types = []
        for type_name in (left_type, right_type):
            if type_name in generic_names and type_name not in generic_types:
                generic_types.append(type_name)

        for generic_type in generic_types:
            bound = self.generic_operator_bound(operator, generic_type)
            if bound is not None:
                self.add_generic_constraint(constraints, generic_type, bound)

    def generic_operator_bound(self, operator, generic_name):
        if operator in {"==", "!="}:
            return "PartialEq"
        if operator in {"<", ">", "<=", ">="}:
            return "PartialOrd"

        operator_traits = {
            "+": "Add",
            "-": "Sub",
            "*": "Mul",
            "/": "Div",
            "%": "Rem",
        }
        trait = operator_traits.get(operator)
        if trait is None:
            return None
        return f"core::ops::{trait}<Output = {generic_name}>"

    def add_generic_constraint(self, constraints, generic_name, bound):
        if generic_name not in constraints:
            constraints[generic_name] = []
        if bound not in constraints[generic_name]:
            constraints[generic_name].append(bound)

    def function_parameter_type(self, param):
        if hasattr(param, "param_type"):
            param_type = self.convert_type_node_to_string(param.param_type)
            return self.resource_type_with_attributes(param_type, param)
        if hasattr(param, "vtype"):
            vtype = param.vtype
            if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
                vtype = self.convert_type_node_to_string(vtype)
            return self.resource_type_with_attributes(vtype, param)
        return "float"

    def function_parameter_requires_mut_binding(self, param, generic_constraints):
        qualifiers = {str(q) for q in getattr(param, "qualifiers", []) or []}
        if "mut" in qualifiers or getattr(param, "is_mutable", False):
            return True

        param_type = self.function_parameter_type(param)
        if self.callable_type_uses_trait(param_type, "FnMut"):
            return True

        param_name = self.type_name_string(param_type)
        if not param_name:
            return False

        return any(
            self.callable_type_uses_trait(bound, "FnMut")
            for bound in generic_constraints.get(param_name, [])
        )

    def map_function_parameter_type(self, param_type):
        rust_type = self.map_type(param_type)
        if self.callable_signature_parts(rust_type) is None:
            return rust_type

        if rust_type.startswith(("impl ", "dyn ")):
            return rust_type
        return f"impl {rust_type}"

    def generate_param_attributes(self, param):
        """Generate Rust GPU parameter attributes based on semantic"""
        if not param.semantic:
            return ""

        semantic = param.semantic.lower()
        if "position" in semantic:
            return "#[location(0)] "
        elif "normal" in semantic:
            return "#[location(1)] "
        elif "texcoord" in semantic:
            if "texcoord0" in semantic:
                return "#[location(2)] "
            elif "texcoord1" in semantic:
                return "#[location(3)] "
            else:
                return "#[location(2)] "
        elif "color" in semantic:
            return "#[location(4)] "
        elif "gl_position" in semantic:
            return "#[builtin(position)] "
        elif "gl_fragcolor" in semantic:
            return "#[location(0)] "
        return ""
