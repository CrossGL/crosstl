"""Function candidates and storage buffer access in the SPIR-V generator: scalar and vector compatibility, generic storage function bindings, candidate scores and concrete matches, function order, array dimensions, resource array parameters, diagnostic expressions, image access metadata and arguments, storage buffer access requirements and inline storage buffer functions."""

from typing import Optional

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .ast_walk import walk_ast_values
from .enum_utils import (
    generic_type_parts,
    substitute_generic_type_name,
)
from .generic_call_dependencies import generic_callee_names
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)
from .image_access_contracts import (
    collect_function_image_access_requirements,
    collect_function_parameter_names,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
)
from .spirv_atomic_load_store import spirv_atomic_load_store_access
from .spirv_swizzle_reference_arguments import write_back_swizzle_reference_bindings
from .spirv_value_result_arguments import (
    VALUE_RESULT_WRITE_BACK,
    write_back_value_result_binding,
)
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)
from .stage_utils import (
    order_functions_by_dependencies as order_functions_by_call_dependencies,
)


class SpirvStorageBufferAccessMixin:
    """Storage buffer access members of ``VulkanSPIRVCodeGen``."""

    def scalar_or_vector_type_compatible(
        self, declared_type: Optional[str], actual_type: Optional[str]
    ) -> bool:
        declared_type = self.normalize_signature_type_name(declared_type)
        actual_type = self.normalize_signature_type_name(actual_type)
        if declared_type is None or actual_type is None:
            return True
        if declared_type == actual_type:
            return True

        numeric_types = {
            "float",
            "double",
            "int",
            "uint",
            "i64",
            "u64",
            "bool",
            self.BFLOAT16_TYPE_NAME,
        }
        if declared_type in numeric_types and actual_type in numeric_types:
            return True

        declared_vector = self.vector_component_type_and_count(declared_type)
        actual_vector = self.vector_component_type_and_count(actual_type)
        if declared_vector is not None or actual_vector is not None:
            if declared_vector is None:
                return False
            if actual_vector is None:
                return (
                    declared_vector[0] in numeric_types and actual_type in numeric_types
                )
            return declared_vector == actual_vector

        return False

    def collect_generic_type_bindings(
        self, expected_type, actual_type, generic_params, substitutions
    ) -> None:
        if expected_type is None or actual_type is None:
            return

        expected_type = self.normalize_signature_type_name(expected_type)
        actual_type = self.normalize_signature_type_name(actual_type)
        if expected_type is None or actual_type is None:
            return

        if expected_type in generic_params:
            substitutions.setdefault(expected_type, actual_type)
            return

        expected_pointee = self.pointer_pointee_type_name_from_string(expected_type)
        actual_pointee = self.pointer_pointee_type_name_from_string(actual_type)
        if expected_pointee is not None or actual_pointee is not None:
            if expected_pointee is not None and actual_pointee is not None:
                self.collect_generic_type_bindings(
                    expected_pointee,
                    actual_pointee,
                    generic_params,
                    substitutions,
                )
            return

        expected_base, expected_args = generic_type_parts(expected_type)
        actual_base, actual_args = generic_type_parts(actual_type)
        if expected_base != actual_base or len(expected_args) != len(actual_args):
            return

        for expected_arg, actual_arg in zip(expected_args, actual_args):
            self.collect_generic_type_bindings(
                expected_arg, actual_arg, generic_params, substitutions
            )

    def generic_storage_function_type_bindings(self, function_node, call_args):
        generic_params = set(generic_function_parameters(function_node))
        if not generic_params:
            return {}

        substitutions = {}
        for param, arg in zip(self.function_parameters(function_node), call_args):
            declared_type_name = self.function_parameter_type_name(param)
            actual_type_name = self.call_argument_type_name(arg)
            self.collect_generic_type_bindings(
                declared_type_name, actual_type_name, generic_params, substitutions
            )

        if any(param not in substitutions for param in generic_params):
            return None
        return substitutions

    def substitute_generic_signature_type(self, type_name, substitutions):
        if not substitutions:
            return type_name
        type_name = self.type_name_from_value(type_name)
        if type_name is None:
            return None
        if type_name.startswith("&"):
            referenced = substitute_generic_type_name(
                type_name[1:].strip(), substitutions
            )
            return f"&{referenced}"
        if type_name.endswith("*"):
            pointee = substitute_generic_type_name(
                type_name[:-1].strip(), substitutions
            )
            return f"{pointee}*"
        return substitute_generic_type_name(type_name, substitutions)

    def storage_buffer_candidate_match_score(self, function_node, call_args):
        match, _reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        if match is None:
            return None
        return match["score"]

    def has_matching_non_storage_function_candidate(self, function_name, call_args):
        for function_node in self.function_nodes_by_name.get(function_name, []):
            if getattr(function_node, "_generic_source_name", None) == function_name:
                continue
            if self.function_requires_pointer_preserving_inlining(function_node):
                continue
            parameters, args = self.storage_buffer_effective_call_signature(
                function_node, call_args
            )
            if len(parameters) != len(args):
                continue
            return True
        return False

    def has_matching_concrete_function_candidate(self, function_name, call_args):
        for function_node in self.function_nodes_by_name.get(function_name, []):
            if getattr(function_node, "_generic_source_name", None) == function_name:
                continue
            if generic_function_parameters(function_node):
                continue
            parameters, args = self.storage_buffer_effective_call_signature(
                function_node, call_args
            )
            if len(parameters) != len(args):
                continue
            return True
        return False

    def storage_buffer_overload_diagnostic(
        self,
        function_name,
        call_args,
        candidates,
        reason,
        call_node=None,
        rejection_reasons=None,
    ):
        actual_types = ", ".join(self.call_argument_type_names(call_args))
        actual_signature = f"{function_name}({actual_types})"
        candidate_signatures = ", ".join(
            self.format_function_candidate_signature(candidate)
            for candidate in candidates
        )
        if not candidate_signatures:
            candidate_signatures = "<none>"
        rejection_detail = ""
        if rejection_reasons:
            details = []
            for candidate in candidates:
                signature = self.format_function_candidate_signature(candidate)
                candidate_reason = rejection_reasons.get(id(candidate))
                if candidate_reason is None:
                    candidate_reason = "not rejected"
                details.append(f"{signature}: {candidate_reason}")
            rejection_detail = "; candidate rejection reasons: " + "; ".join(details)
        return UnsupportedSPIRVFeatureError(
            "storage-buffer-function-overload",
            "SPIR-V storage-buffer function inlining could not resolve "
            f"overloaded call '{function_name}': {reason}; actual call "
            f"arity/types: {len(call_args)} ({actual_signature}); candidate "
            f"signatures: {candidate_signatures}{rejection_detail}",
            missing_capabilities=("spirv.storage_buffer_function_overload",),
            source_location=(
                getattr(call_node, "source_location", None)
                if call_node is not None
                else None
            ),
        )

    def order_functions_by_dependencies(self, functions):
        """Order ``functions`` so that every callee precedes its callers.

        A call is resolved against the overloads already emitted, so a call
        depends on every overload of its callee's name, wherever the source
        declares them. Overloads of one name that call each other keep their
        source order.
        """
        return order_functions_by_call_dependencies(
            functions,
            self.walk_ast_nodes,
            self.function_call_name,
            FunctionCallNode,
            include_overloads=True,
            keep_overload_family_order=True,
            # A specialization's calls still spell the generic name (``twice``), not ``twice_float``.
            extra_callee_names=lambda func: generic_callee_names(self, func),
        )

    def walk_ast_nodes(self, root):
        return walk_ast_values(root)

    def array_dimensions(self, type_name: str):
        if not type_name or "[" not in type_name:
            return None

        suffix = type_name[type_name.find("[") :]
        dimensions = []
        offset = 0
        while offset < len(suffix):
            if suffix[offset] != "[":
                return None
            end = suffix.find("]", offset + 1)
            if end == -1:
                return None
            dimensions.append(suffix[offset + 1 : end])
            offset = end + 1
        return dimensions

    def is_unsized_resource_array_type_name(self, type_name: str) -> bool:
        type_name = self.normalize_generic_vector_type(str(type_name))
        array_type = self.split_outer_array_type(type_name)
        return (
            array_type is not None
            and array_type[1] is None
            and self.is_resource_type_name(self.array_base_type_name(type_name))
        )

    def is_fixed_resource_array_type_name(self, type_name: str) -> bool:
        type_name = self.normalize_generic_vector_type(str(type_name))
        array_type = self.split_outer_array_type(type_name)
        return (
            array_type is not None
            and array_type[1] is not None
            and self.is_resource_type_name(self.array_base_type_name(type_name))
        )

    def fixed_type_for_unsized_resource_param(self, declared_type: str, arg_type: str):
        declared_type = self.normalize_generic_vector_type(str(declared_type))
        arg_type = self.normalize_generic_vector_type(str(arg_type))

        if not self.is_unsized_resource_array_type_name(declared_type):
            return None
        if not self.is_fixed_resource_array_type_name(arg_type):
            return None
        if self.array_base_type_name(declared_type) != self.array_base_type_name(
            arg_type
        ):
            return None

        declared_dimensions = self.array_dimensions(declared_type)
        arg_dimensions = self.array_dimensions(arg_type)
        if not declared_dimensions or not arg_dimensions:
            return None
        if len(declared_dimensions) != len(arg_dimensions):
            return None
        if declared_dimensions[0] != "":
            return None
        if declared_dimensions[1:] != arg_dimensions[1:]:
            return None

        return arg_type

    def resource_array_param_type_is_compatible(
        self, declared_type: str, arg_type: str
    ) -> bool:
        declared_type = self.normalize_generic_vector_type(str(declared_type))
        arg_type = self.normalize_generic_vector_type(str(arg_type))

        if not self.is_unsized_resource_array_type_name(declared_type):
            return True

        if self.fixed_type_for_unsized_resource_param(declared_type, arg_type):
            return True

        if not self.is_unsized_resource_array_type_name(arg_type):
            return False

        return self.array_base_type_name(declared_type) == self.array_base_type_name(
            arg_type
        ) and self.array_dimensions(declared_type) == self.array_dimensions(arg_type)

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def diagnostic_expression(self, expr) -> str:
        """Render expression nodes for SPIR-V warnings without leaking reprs."""
        if expr is None:
            return "<unknown>"
        if isinstance(expr, str):
            return expr
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, LiteralNode):
            return str(expr.value)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
            return (
                f"{self.diagnostic_expression(array_expr)}"
                f"[{self.diagnostic_expression(index_expr)}]"
            )
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return f"{self.diagnostic_expression(object_expr)}.{expr.member}"
        if isinstance(expr, FunctionCallNode):
            callee = getattr(expr, "function", getattr(expr, "name", None))
            args = ", ".join(
                self.diagnostic_expression(arg)
                for arg in getattr(expr, "args", getattr(expr, "arguments", []))
            )
            return f"{self.diagnostic_expression(callee)}({args})"
        if isinstance(expr, UnaryOpNode):
            operand = self.diagnostic_expression(expr.operand)
            return f"{operand}{expr.op}" if expr.is_postfix else f"{expr.op}{operand}"
        if isinstance(expr, BinaryOpNode):
            return (
                f"{self.diagnostic_expression(expr.left)} {expr.op} "
                f"{self.diagnostic_expression(expr.right)}"
            )
        name = self.expression_name(expr)
        if name is not None:
            return name
        return type(expr).__name__

    def direct_expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None)
        return None

    def function_call_name(self, call):
        callee = getattr(call, "function", getattr(call, "name", None))
        if hasattr(callee, "name"):
            return callee.name
        if isinstance(callee, str):
            return callee
        return None

    def collect_function_image_access_requirements_for_ast(self, ast):
        functions = self.collect_ast_functions(ast)
        self.function_parameter_names = collect_function_parameter_names(functions)
        return collect_function_image_access_requirements(
            functions,
            self.function_parameter_names,
            self.walk_ast_nodes,
            self.function_call_name,
            self.expression_name,
        )

    def collect_stage_local_function_image_access_metadata(self, ast):
        self.stage_local_function_parameter_names = {}
        self.stage_local_function_image_access_requirements = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            parameter_names = collect_function_parameter_names(visible_functions)
            requirements = collect_function_image_access_requirements(
                visible_functions,
                parameter_names,
                self.walk_ast_nodes,
                self.function_call_name,
                self.expression_name,
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_parameter_names[key] = parameter_names.get(
                    function_name, []
                )
                self.stage_local_function_image_access_requirements[key] = (
                    requirements.get(function_name, {})
                )

    def buffer_operation_access_requirement(self, func_name):
        if (
            func_name == "buffer_load"
            or self.byte_address_helper_load_width(func_name) is not None
        ):
            return "read"
        if (
            func_name == "buffer_store"
            or self.byte_address_helper_store_width(func_name) is not None
        ):
            return "write"
        if func_name == "buffer_append":
            return "write"
        if func_name == "buffer_consume":
            return "read_write"
        if func_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            return "read_write"
        if func_name in self.buffer_atomic_function_names():
            return "read_write"
        return spirv_atomic_load_store_access(func_name)

    def storage_buffer_member_method_access_requirement(self, method_name):
        if method_name in {"Load"} or self.byte_address_method_load_width(method_name):
            return "read"
        if method_name in {"Store"} or self.byte_address_method_store_width(
            method_name
        ):
            return "write"
        if method_name == "Append":
            return "write"
        if method_name == "Consume":
            return "read_write"
        if method_name in {"IncrementCounter", "DecrementCounter"}:
            return "read_write"
        if self.byte_address_method_interlocked_info(method_name) is not None:
            return "read_write"
        return None

    def storage_buffer_parameter_root_name(self, expr, storage_buffer_parameters):
        def parameter_root(name):
            if name not in storage_buffer_parameters:
                return None
            if isinstance(storage_buffer_parameters, dict):
                return storage_buffer_parameters.get(name)
            return name

        if isinstance(expr, str):
            return parameter_root(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = getattr(expr, "name", None)
            return parameter_root(name)
        if isinstance(expr, ArrayAccessNode):
            return self.storage_buffer_parameter_root_name(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                storage_buffer_parameters,
            )
        if isinstance(expr, MemberAccessNode):
            return self.storage_buffer_parameter_root_name(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                storage_buffer_parameters,
            )
        if isinstance(expr, UnaryOpNode) and expr.op == "*":
            return self.storage_buffer_parameter_root_name(
                expr.operand, storage_buffer_parameters
            )
        return None

    def storage_buffer_pointer_root_name(self, expr, storage_buffer_parameters):
        """Resolve a pointer-valued expression to its formal buffer parameter."""
        root_name = self.storage_buffer_parameter_root_name(
            expr, storage_buffer_parameters
        )
        if root_name is not None:
            return root_name
        if isinstance(expr, UnaryOpNode) and expr.op == "&":
            return self.storage_buffer_parameter_root_name(
                expr.operand, storage_buffer_parameters
            )
        if isinstance(expr, BinaryOpNode) and expr.op in {"+", "-"}:
            left_root = self.storage_buffer_pointer_root_name(
                expr.left, storage_buffer_parameters
            )
            right_root = self.storage_buffer_pointer_root_name(
                expr.right, storage_buffer_parameters
            )
            if left_root is not None and right_root is None:
                return left_root
            if expr.op == "+" and right_root is not None and left_root is None:
                return right_root
            return None
        if isinstance(expr, TernaryOpNode):
            true_root = self.storage_buffer_pointer_root_name(
                expr.true_expr, storage_buffer_parameters
            )
            false_root = self.storage_buffer_pointer_root_name(
                expr.false_expr, storage_buffer_parameters
            )
            if true_root == false_root:
                return true_root
        return None

    def function_storage_buffer_alias_roots(
        self, function_node, storage_buffer_parameters
    ):
        """Map local buffer aliases to the formal parameters that own access."""
        roots = {name: name for name in storage_buffer_parameters}
        declarations = [
            node
            for node in self.walk_ast_nodes(getattr(function_node, "body", []))
            if isinstance(node, VariableNode)
            and self.storage_buffer_resource_type_name(
                getattr(node, "var_type", getattr(node, "vtype", None))
            )
            is not None
            and getattr(node, "initial_value", None) is not None
        ]

        changed = True
        while changed:
            changed = False
            for declaration in declarations:
                if declaration.name in roots:
                    continue
                root_name = self.storage_buffer_pointer_root_name(
                    declaration.initial_value, roots
                )
                if root_name is None:
                    continue
                roots[declaration.name] = root_name
                changed = True
        return roots

    def function_storage_buffer_parameter_indices(self, function_node) -> set:
        return {
            index
            for index, param in enumerate(
                getattr(
                    function_node, "parameters", getattr(function_node, "params", [])
                )
            )
            if self.storage_buffer_parameter_type_name(param) is not None
        }

    def scan_storage_buffer_access_path_indices(
        self,
        expr,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if isinstance(expr, ArrayAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_access_path_indices(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
        elif isinstance(expr, MemberAccessNode):
            self.scan_storage_buffer_access_path_indices(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )

    def merge_storage_buffer_access_requirement_for_parameter(
        self, requirements, func_name, parameter_name, required_access
    ):
        if parameter_name is None:
            return
        current = requirements[func_name].get(parameter_name)
        requirements[func_name][parameter_name] = (
            self.merge_resource_access_requirement(current, required_access)
        )

    def scan_storage_buffer_requirement_expression(
        self,
        expr,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        if expr is None or isinstance(expr, (str, int, float, bool)):
            return

        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            if isinstance(callee_expr, MemberAccessNode):
                args = getattr(expr, "arguments", getattr(expr, "args", []))
                required_access = self.storage_buffer_member_method_access_requirement(
                    getattr(callee_expr, "member", None)
                )
                target_name = self.storage_buffer_pointer_root_name(
                    getattr(callee_expr, "object", None),
                    storage_buffer_parameters,
                )
                if required_access is not None and target_name is not None:
                    self.merge_storage_buffer_access_requirement_for_parameter(
                        requirements, func_name, target_name, required_access
                    )
                    self.scan_storage_buffer_access_path_indices(
                        getattr(callee_expr, "object", None),
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                    for arg in args:
                        self.scan_storage_buffer_requirement_node(
                            arg,
                            func_name,
                            storage_buffer_parameters,
                            callee_storage_buffer_parameter_indices,
                            requirements,
                            visited,
                        )
                    return

            callee_name = self.function_call_name(expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            required_access = self.buffer_operation_access_requirement(callee_name)
            if required_access is not None and args:
                target_name = self.storage_buffer_pointer_root_name(
                    args[0], storage_buffer_parameters
                )
                self.merge_storage_buffer_access_requirement_for_parameter(
                    requirements, func_name, target_name, required_access
                )
                self.scan_storage_buffer_access_path_indices(
                    args[0],
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
                for arg in args[1:]:
                    self.scan_storage_buffer_requirement_node(
                        arg,
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                return

            storage_buffer_indices = callee_storage_buffer_parameter_indices.get(
                callee_name, set()
            )
            for arg_index, arg in enumerate(args):
                if (
                    arg_index in storage_buffer_indices
                    and self.storage_buffer_pointer_root_name(
                        arg, storage_buffer_parameters
                    )
                    is not None
                ):
                    self.scan_storage_buffer_access_path_indices(
                        arg,
                        func_name,
                        storage_buffer_parameters,
                        callee_storage_buffer_parameter_indices,
                        requirements,
                        visited,
                    )
                    continue
                self.scan_storage_buffer_requirement_node(
                    arg,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return

        target_name = self.storage_buffer_parameter_root_name(
            expr, storage_buffer_parameters
        )
        if target_name is not None:
            if isinstance(expr, (IdentifierNode, VariableNode)):
                return
            self.merge_storage_buffer_access_requirement_for_parameter(
                requirements, func_name, target_name, "read"
            )
            self.scan_storage_buffer_access_path_indices(
                expr,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(expr, ArrayAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, MemberAccessNode):
            self.scan_storage_buffer_requirement_node(
                getattr(expr, "object", getattr(expr, "object_expr", None)),
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        if isinstance(expr, BinaryOpNode):
            self.scan_storage_buffer_requirement_node(
                expr.left,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            self.scan_storage_buffer_requirement_node(
                expr.right,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, UnaryOpNode):
            self.scan_storage_buffer_requirement_node(
                expr.operand,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return
        if isinstance(expr, TernaryOpNode):
            for child in (expr.condition, expr.true_expr, expr.false_expr):
                self.scan_storage_buffer_requirement_node(
                    child,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )
            return
        if isinstance(expr, ArrayLiteralNode):
            for element in getattr(expr, "elements", []):
                self.scan_storage_buffer_requirement_node(
                    element,
                    func_name,
                    storage_buffer_parameters,
                    callee_storage_buffer_parameter_indices,
                    requirements,
                    visited,
                )

    def scan_storage_buffer_assignment_target_requirement(
        self,
        target,
        operator,
        func_name,
        storage_buffer_parameters,
        callee_storage_buffer_parameter_indices,
        requirements,
        visited,
    ):
        target_name = self.storage_buffer_parameter_root_name(
            target, storage_buffer_parameters
        )
        if target_name is not None:
            if isinstance(target, (str, IdentifierNode, VariableNode)):
                return
            required_access = "write" if operator == "=" else "read_write"
            self.merge_storage_buffer_access_requirement_for_parameter(
                requirements, func_name, target_name, required_access
            )
            self.scan_storage_buffer_access_path_indices(
                target,
                func_name,
                storage_buffer_parameters,
                callee_storage_buffer_parameter_indices,
                requirements,
                visited,
            )
            return

        self.scan_storage_buffer_requirement_node(
            target,
            func_name,
            storage_buffer_parameters,
            callee_storage_buffer_parameter_indices,
            requirements,
            visited,
        )

    def collect_function_storage_buffer_access_requirements_for_functions(
        self, functions
    ):
        functions = self.concrete_function_prepass_list(functions)
        functions_by_name = self.collect_functions_by_name(functions)
        function_parameter_names_by_id = {
            id(func): [
                getattr(param, "name", None)
                for param in self.function_parameters(func)
                if getattr(param, "name", None)
            ]
            for func in functions
            if getattr(func, "name", None)
        }
        parameter_sets = {
            func_key: set(param_names)
            for func_key, param_names in function_parameter_names_by_id.items()
        }
        requirements = {
            id(func): {} for func in functions if getattr(func, "name", None)
        }
        storage_buffer_parameter_roots = {
            id(func): self.run_function_prepass(
                func,
                lambda func=func: self.function_storage_buffer_alias_roots(
                    func, self.function_storage_buffer_parameters(func)
                ),
            )
            for func in functions
            if getattr(func, "name", None)
        }
        storage_buffer_parameter_indices = {}
        for func in functions:
            func_name = self.function_source_name(func)
            if not func_name:
                continue
            storage_buffer_parameter_indices.setdefault(func_name, set()).update(
                self.run_function_prepass(
                    func,
                    lambda func=func: self.function_storage_buffer_parameter_indices(
                        func
                    ),
                )
            )

        for func in functions:
            func_name = self.function_source_name(func)
            if not func_name:
                continue
            func_key = id(func)

            self.run_function_prepass(
                func,
                lambda func=func, func_key=func_key: (
                    self.scan_storage_buffer_requirement_node(
                        getattr(func, "body", []),
                        func_key,
                        storage_buffer_parameter_roots.get(func_key, {}),
                        storage_buffer_parameter_indices,
                        requirements,
                        set(),
                    )
                ),
            )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = self.function_source_name(func)
                if not func_name:
                    continue
                func_key = id(func)

                parameter_set = parameter_sets.get(func_key, set())
                if not parameter_set:
                    continue

                for node in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue

                    callee_name = self.function_call_name(node)
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee_candidates = [
                        candidate
                        for candidate in functions_by_name.get(callee_name, [])
                        if len(self.function_parameters(candidate)) == len(args)
                    ]
                    if not callee_candidates:
                        continue

                    for callee_candidate in callee_candidates:
                        callee_key = id(callee_candidate)
                        callee_requirements = requirements.get(callee_key)
                        if not callee_requirements:
                            continue
                        callee_parameters = function_parameter_names_by_id.get(
                            callee_key, []
                        )
                        for (
                            callee_param,
                            required_access,
                        ) in callee_requirements.items():
                            try:
                                index = callee_parameters.index(callee_param)
                            except ValueError:
                                continue
                            if index >= len(args):
                                continue

                            target_name = self.run_function_prepass(
                                func,
                                lambda: self.storage_buffer_pointer_root_name(
                                    args[index],
                                    storage_buffer_parameter_roots.get(func_key, {}),
                                ),
                            )
                            if target_name not in parameter_set:
                                continue

                            current = requirements[func_key].get(target_name)
                            merged = self.merge_resource_access_requirement(
                                current, required_access
                            )
                            if merged != current:
                                requirements[func_key][target_name] = merged
                                changed = True

        self.function_parameter_names_by_id.update(function_parameter_names_by_id)
        filtered_requirements = {
            func_key: reqs for func_key, reqs in requirements.items() if reqs
        }
        self.function_storage_buffer_access_requirements_by_id.update(
            filtered_requirements
        )
        legacy_requirements = {}
        for func in functions:
            func_name = self.function_source_name(func)
            if not func_name:
                continue
            for param_name, required_access in filtered_requirements.get(
                id(func), {}
            ).items():
                legacy_requirements.setdefault(func_name, {})
                current = legacy_requirements[func_name].get(param_name)
                legacy_requirements[func_name][param_name] = (
                    self.merge_resource_access_requirement(current, required_access)
                )
        return legacy_requirements

    def collect_function_storage_buffer_access_requirements_for_ast(self, ast):
        return self.collect_function_storage_buffer_access_requirements_for_functions(
            self.collect_ast_functions(ast)
        )

    def collect_stage_local_function_storage_buffer_metadata(self, ast):
        self.stage_local_function_storage_buffer_access_requirements = {}
        self.stage_local_inline_storage_buffer_functions = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_inline_functions = []
            for function_node in local_functions:
                if generic_function_parameters(function_node):
                    local_inline_functions.extend(
                        generic_function_emission_list(self, function_node)
                    )
                else:
                    local_inline_functions.append(function_node)

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_global_functions = []
            for function_node in global_functions:
                if getattr(function_node, "name", None) in local_names:
                    continue
                if generic_function_parameters(function_node):
                    visible_global_functions.extend(
                        generic_function_emission_list(self, function_node)
                    )
                else:
                    visible_global_functions.append(function_node)
            visible_functions = visible_global_functions + local_inline_functions
            requirements = (
                self.collect_function_storage_buffer_access_requirements_for_functions(
                    visible_functions
                )
            )

            for function_node in local_inline_functions:
                function_name = self.function_source_name(function_node)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_storage_buffer_access_requirements[key] = (
                    requirements.get(function_name, {})
                )
                self.stage_local_inline_storage_buffer_functions.setdefault(key, [])
                if (
                    self.function_requires_pointer_preserving_inlining(function_node)
                    and getattr(function_node, "body", None) is not None
                ):
                    self.stage_local_inline_storage_buffer_functions[key].append(
                        function_node
                    )

    def merge_resource_access_requirement(self, current, incoming):
        if incoming is None:
            return current
        if current is None or current == incoming:
            return incoming
        return "read_write"

    def storage_image_access_for_expression(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None

        pointer = self.local_variables.get(name) or self.resolve_global_variable(name)
        metadata = (
            self.resource_metadata_for_pointer(pointer) if pointer is not None else None
        )
        if metadata is None:
            return None
        if metadata.get("kind") != "storage_image":
            return None
        if metadata.get("readonly"):
            return "read"
        if metadata.get("writeonly"):
            return "write"
        if metadata.get("readwrite"):
            return "read_write"
        return None

    def storage_buffer_access_for_expression(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None

        pointer = self.local_variables.get(name) or self.resolve_global_variable(name)
        metadata = (
            self.storage_buffer_access_metadata_for_pointer(pointer)
            if pointer is not None
            else None
        )
        if metadata is None:
            return None
        if metadata.get("append_consume"):
            return "read_write"
        if metadata.get("readonly"):
            return "read"
        if metadata.get("writeonly"):
            return "write"
        if metadata.get("readwrite"):
            return "read_write"
        return None

    def expression_debug_name(self, expr) -> str:
        name = self.expression_name(expr)
        return name if name is not None else self.diagnostic_expression(expr)

    def validate_function_image_access_arguments(self, func_name, args) -> bool:
        callee_requirements = self.resolve_function_image_access_requirements(func_name)
        if not callee_requirements:
            return True

        param_names = self.resolve_function_parameter_names(func_name)
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue

            actual_access = self.storage_image_access_for_expression(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue

            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            self.refuse_lowering(
                f"function call '{func_name}' requires {required_label} "
                "storage image access for argument "
                f"{self.expression_debug_name(args[index])} passed to parameter "
                f"{param_name}: got {actual_label}"
            )
            return False

        return True

    def validate_function_storage_buffer_access_arguments(
        self, function_node, args
    ) -> bool:
        func_name = getattr(function_node, "name", "unknown")
        callee_requirements = (
            self.function_storage_buffer_access_requirements_by_id.get(
                id(function_node), {}
            )
        )
        if not callee_requirements:
            return True

        param_names = self.function_parameter_names_by_id.get(
            id(function_node),
            [
                getattr(param, "name", None)
                for param in self.function_parameters(function_node)
            ],
        )
        for index, param, arg in self.storage_buffer_effective_call_bindings(
            function_node, args
        ):
            param_name = (
                param_names[index]
                if index < len(param_names)
                else getattr(param, "name", None)
            )
            required_access = callee_requirements.get(param_name)
            if required_access is None:
                continue

            actual_access = self.storage_buffer_access_for_expression(arg)
            if image_access_satisfies_requirement(required_access, actual_access):
                continue

            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            self.refuse_lowering(
                f"function call '{func_name}' requires {required_label} "
                "storage buffer access for argument "
                f"{self.expression_debug_name(arg)} passed to parameter "
                f"{param_name}: got {actual_label}"
            )
            return False

        return True

    def collect_inline_storage_buffer_functions(self, ast):
        functions = {}
        for func in self.collect_ast_functions(ast):
            function_name = self.function_source_name(func)
            if not function_name:
                continue
            if not self.function_requires_pointer_preserving_inlining(func):
                continue
            if getattr(func, "body", None) is None:
                # A prototype has nothing to inline; its definition is the candidate.
                continue
            functions.setdefault(function_name, []).append(func)
        return functions

    def default_value_for_function(self, function_node) -> Optional[SpirvId]:
        return_type = self.map_crossgl_type(function_node.return_type)
        if return_type.type.base_type == "void":
            return None
        return self.default_value_for_type(return_type)

    def nested_match_inline_return_node(self, function_node):
        """Return the first inline return owned by a match arm, if any."""
        body = getattr(function_node, "body", [])
        for node in self.walk_ast_nodes(body):
            if not isinstance(node, MatchNode):
                continue
            for match_node in self.walk_ast_nodes(node):
                if isinstance(match_node, ReturnNode):
                    return match_node
        return None

    def validate_skipped_storage_buffer_call_arguments(
        self, function_node, call_args, match, call_node=None
    ):
        if match is None:
            return

        arguments = list(call_args or [])
        function_name = getattr(function_node, "name", "unknown")
        for argument_index in match["skipped_arg_indices"]:
            argument = arguments[argument_index]
            if self.side_effect_free_call_argument(argument):
                continue
            source_location = getattr(argument, "source_location", None) or getattr(
                call_node, "source_location", None
            )
            raise UnsupportedSPIRVFeatureError(
                "side-effectful-skipped-storage-buffer-call-argument",
                "SPIR-V ordered argument alignment cannot omit side-effectful "
                f"argument {argument_index + 1} from call to '{function_name}'",
                missing_capabilities=(
                    "spirv.storage_buffer_call_argument_alignment_side_effect",
                ),
                source_location=source_location,
            )

    def inline_storage_buffer_function_call(
        self, function_node, call_args, call_node=None
    ):
        func_name = getattr(function_node, "name", "unknown")
        nested_match_return = self.nested_match_inline_return_node(function_node)
        if nested_match_return is not None:
            source_location = getattr(
                nested_match_return, "source_location", None
            ) or getattr(call_node, "source_location", None)
            raise UnsupportedSPIRVFeatureError(
                "nested-match-return-storage-buffer-function-inline",
                "SPIR-V pointer-preserving function inlining does not yet "
                f"support returns nested in match arms; helper '{func_name}' "
                "requires match-aware return control flow",
                missing_capabilities=(
                    "spirv.nested_match_return_storage_buffer_function",
                ),
                source_location=source_location,
            )
        if not self.validate_function_storage_buffer_access_arguments(
            function_node, call_args
        ):
            return self.default_value_for_function(function_node)

        match, rejection_reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        self.validate_skipped_storage_buffer_call_arguments(
            function_node, call_args, match, call_node
        )
        if match is None:
            parameters, effective_call_args = (
                self.storage_buffer_effective_call_signature(function_node, call_args)
            )
            if len(effective_call_args) > len(parameters):
                raise self.storage_buffer_overload_diagnostic(
                    func_name,
                    call_args,
                    [function_node],
                    "selected candidate has too few parameters",
                    call_node=call_node,
                    rejection_reasons={id(function_node): rejection_reason},
                )
            if len(effective_call_args) < len(parameters):
                self.refuse_lowering(
                    f"function call '{func_name}' requires "
                    f"{len(parameters)} arguments"
                )
                return self.default_value_for_function(function_node)

        function_key = id(function_node)
        if any(
            key == function_key for key, _name in self.inline_storage_buffer_call_stack
        ):
            cycle = [
                name
                for key, name in self.inline_storage_buffer_call_stack
                if key == function_key
            ]
            cycle.append(func_name)
            raise UnsupportedSPIRVFeatureError(
                "recursive-storage-buffer-function-inline",
                "SPIR-V storage-buffer function inlining does not support "
                f"recursive helper calls ({' -> '.join(cycle)})",
                missing_capabilities=("spirv.recursive_storage_buffer_function",),
            )

        generic_type_substitutions = self.generic_storage_function_type_bindings(
            function_node, call_args
        )
        if generic_type_substitutions is None:
            generic_type_substitutions = {}
        prepared_bindings = self.prepare_pointer_preserving_inline_bindings(
            function_node,
            call_args,
            generic_type_substitutions,
        )
        if prepared_bindings is None:
            return self.default_value_for_function(function_node)

        # Argument evaluation can materialize caller variables and pointer aliases.
        # Preserve that post-evaluation state, then overlay every callee parameter
        # together so an early parameter cannot shadow a later argument expression.
        previous_declaration_state = self.local_declaration_scope_state()
        previous_return_type = self.current_return_type
        previous_return_type_source = self.current_return_type_source
        previous_generic_type_substitutions = self.current_generic_type_substitutions
        self.current_generic_type_substitutions = generic_type_substitutions
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        self.current_generic_function_substitutions = (
            getattr(function_node, "_generic_substitutions", {}) or {}
        )
        previous_cooperative_matrix_role_overrides = (
            self.current_cooperative_matrix_role_overrides
        )
        self.current_cooperative_matrix_role_overrides = (
            self.cooperative_matrix_role_overrides_for_function(function_node)
        )
        self.current_return_type = self.map_crossgl_type(function_node.return_type)
        self.current_return_type_source = self.type_name_from_value(
            function_node.return_type
        )
        self.current_function_literal_int_constant_shadows = {
            param.name
            for param in self.function_parameters(function_node)
            if getattr(param, "name", None)
        }
        self.current_function_literal_int_constants = {}
        self.inline_storage_buffer_call_stack.append((function_key, func_name))

        try:
            for binding in prepared_bindings:
                param_name = binding["name"]
                parameter_value = binding["value"]
                if binding["kind"] == "storage_buffer":
                    if binding["copy_pointer"]:
                        parameter_value = self.copy_storage_buffer_pointer_alias(
                            parameter_value,
                            param_name,
                        )
                    self.resource_alias_variables.add(param_name)
                self.local_variables[param_name] = parameter_value

            result = self.inline_function_body(function_node)
            write_back_swizzle_reference_bindings(self, prepared_bindings)
            for binding in prepared_bindings:
                if VALUE_RESULT_WRITE_BACK in binding:
                    write_back_value_result_binding(self, binding)
            if result is not None:
                return result
            return self.default_value_for_function(function_node)
        finally:
            self.inline_storage_buffer_call_stack.pop()
            self.restore_local_declaration_scope_state(previous_declaration_state)
            self.current_return_type = previous_return_type
            self.current_return_type_source = previous_return_type_source
            self.current_generic_type_substitutions = (
                previous_generic_type_substitutions
            )
            self.current_generic_function_substitutions = (
                previous_generic_function_substitutions
            )
            self.current_cooperative_matrix_role_overrides = (
                previous_cooperative_matrix_role_overrides
            )
