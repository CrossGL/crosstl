"""Local and global identifiers in the GLSL generator: scalar array views, dynamic resource statements, fixed array vector pointer views and global identifier names."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    MemberAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    parse_array_type,
)
from .builtin_result_types import (
    GLSL_FAMILY,
    inferred_value_type,
)
from .enum_utils import sanitize_type_name
from .glsl_narrow_integer_literals import widened_integer_literal
from .opengl_values import (
    OpenGLPrivatePointerParameterError,
    OpenGLTemplateTypeError,
)
from .pointer_reinterpret import PointerReinterpretationError


class OpenGLLocalIdentifierMixin:
    """Local identifier members of ``GLSLCodeGen``."""

    def glsl_scalar_array_view_locals(self, function, *, lexical=None):
        lexical = lexical or self.glsl_scalar_array_view_lexical_bindings(function)
        lexical_bindings = lexical["call_bindings"]
        lexical_call_source_types = lexical["call_source_types"]
        all_declarations = lexical["declarations"]
        ordinary_declaration_counts = {}
        for binding in all_declarations.values():
            if binding["kind"] != "local":
                continue
            name = binding["name"]
            ordinary_declaration_counts[name] = (
                ordinary_declaration_counts.get(name, 0) + 1
            )

        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(
                    function,
                    "parameters",
                    getattr(function, "params", []),
                )
                or []
            )
            if getattr(parameter, "name", None)
        }

        def scalar_source_type(type_value):
            if type_value is None or isinstance(
                type_value,
                (ArrayType, PointerType, ReferenceType),
            ):
                return False
            type_name = self.type_name_string(type_value)
            if not type_name or "*" in type_name or "&" in type_name:
                return False
            _element_type, extent_expression = parse_array_type(type_name)
            return extent_expression is None and not self.type_contains_opaque_resource(
                type_value
            )

        def nonlocal_source_type(name):
            parameter = next(
                (
                    candidate
                    for candidate in (
                        getattr(
                            function,
                            "parameters",
                            getattr(function, "params", []),
                        )
                        or []
                    )
                    if getattr(candidate, "name", None) == name
                ),
                None,
            )
            if parameter is not None:
                return getattr(
                    parameter,
                    "param_type",
                    getattr(parameter, "vtype", None),
                )
            return self.global_variable_types.get(name) or self.glsl_constant_types.get(
                name
            )

        views = {}
        for call in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(call, FunctionCallNode):
                continue
            function_name = self.function_call_name(call)
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            bindings = lexical_bindings.get(id(call), {})
            addressed_candidates = []
            for index, argument in enumerate(arguments):
                if not (
                    isinstance(argument, UnaryOpNode)
                    and self.map_operator(argument.op) == "&"
                    and isinstance(argument.operand, (IdentifierNode, VariableNode))
                ):
                    continue
                name = getattr(argument.operand, "name", None)
                binding_id = bindings.get(index)
                binding = all_declarations.get(binding_id)
                binding_type = (
                    binding.get("source_type")
                    if binding is not None
                    else nonlocal_source_type(name)
                )
                if scalar_source_type(binding_type):
                    addressed_candidates.append((index, argument, name, binding_id))
            if not addressed_candidates:
                continue
            resolved_overload = self.glsl_resolve_function_overload_in_scope(
                function,
                function_name,
                arguments,
                call_node=call,
                lexical_source_types=lexical_call_source_types.get(id(call), {}),
            )
            resolved_function = resolved_overload
            if resolved_overload is not None:
                parameter_infos = self.glsl_function_parameter_infos(resolved_overload)
            else:
                overloads = list(
                    getattr(self, "glsl_function_overloads_by_name", {}).get(
                        function_name,
                        (),
                    )
                    or ()
                )
                if len(overloads) > 1:
                    _index, argument, name, _binding_id = addressed_candidates[0]
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL cannot resolve the exact overload for scalar "
                        f"address view '{name}' passed to '{function_name}'",
                        function_name=function_name,
                        parameter_name=name,
                        reason="scalar-array-view-overload-unresolved",
                        source_location=getattr(argument, "source_location", None),
                    )
                if len(overloads) == 1:
                    resolved_function = overloads[0]
                else:
                    resolved_function = self.function_definitions.get(function_name)
                parameter_infos = (
                    self.glsl_function_parameter_infos(resolved_function)
                    if resolved_function is not None
                    else list(
                        self.function_parameter_infos.get(function_name, ()) or ()
                    )
                )
            candidate_indices = {
                index for index, _argument, _name, _binding_id in addressed_candidates
            }
            for index, (argument, parameter_info) in enumerate(
                zip(arguments, parameter_infos)
            ):
                if index not in candidate_indices:
                    continue
                name = getattr(argument.operand, "name", None)
                _parameter_name, parameter_type, parameter_qualifiers = parameter_info
                element_type, extent_expression = parse_array_type(
                    self.type_name_string(parameter_type)
                )
                extent = evaluate_literal_int_expression(
                    extent_expression,
                    self.glsl_active_literal_int_constants(),
                )
                qualifiers = {
                    str(qualifier).lower() for qualifier in parameter_qualifiers or ()
                }
                if extent != 1 or qualifiers.intersection(
                    {
                        "buffer",
                        "constant",
                        "device",
                        "global",
                        "groupshared",
                        "shared",
                        "storage",
                        "threadgroup",
                        "uniform",
                        "workgroup",
                    }
                ):
                    continue
                binding_id = bindings.get(index)
                binding = all_declarations.get(binding_id)
                if binding is None or binding["kind"] not in {"local", "for-in"}:
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL scalar address view requires a lexically visible "
                        f"local declaration for '{name}'",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="scalar-array-view-binding-nonlocal",
                        source_location=getattr(argument, "source_location", None),
                    )
                if binding["kind"] == "local" and (
                    ordinary_declaration_counts.get(name) != 1
                    or name in parameter_names
                ):
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL scalar address view requires one unshadowed local "
                        f"declaration for '{name}'",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="scalar-array-view-binding-ambiguous",
                        source_location=getattr(argument, "source_location", None),
                    )
                declared_type = binding.get("source_type")
                if self.map_type(declared_type) != self.map_type(element_type):
                    raise OpenGLPrivatePointerParameterError(
                        f"OpenGL scalar address view changes element type for '{name}'",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="scalar-array-view-element-type-mismatch",
                        source_location=getattr(argument, "source_location", None),
                    )
                previous = views.get(binding_id)
                if previous is not None and self.map_type(
                    previous["element_type"]
                ) != self.map_type(element_type):
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL scalar address view has conflicting fixed-array "
                        f"element types for '{name}'",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="scalar-array-view-conflicting-element-types",
                        source_location=getattr(argument, "source_location", None),
                    )
                resolved_parameters = list(
                    getattr(
                        resolved_function,
                        "parameters",
                        getattr(resolved_function, "params", []),
                    )
                    or []
                )
                if index < len(resolved_parameters):
                    # ``&local`` denotes caller-visible storage. GLSL's
                    # unqualified array parameters are copy-in only, so retain
                    # source writeback explicitly on the exact resolved helper
                    # and on the helpers it forwards the view to.
                    self.glsl_mark_fixed_array_writeback(
                        resolved_function, _parameter_name
                    )
                views[binding_id] = {
                    "binding_kind": binding["kind"],
                    "element_type": element_type,
                    "extent": 1,
                    "name": name,
                }
        return views

    def generate_glsl_scalar_array_view_declaration(self, node, indent):
        view = self.current_glsl_scalar_array_view_plans.get(id(node))
        if view is None:
            # Any local declaration starts a new lexical binding.  A view from
            # an outer block or for-in pattern must not rewrite this binding;
            # generate_scoped_statement_body restores the outer view on exit.
            self.current_glsl_scalar_array_views.pop(node.name, None)
            return None
        self.current_glsl_scalar_array_views[node.name] = view
        var_type = self.local_variable_declared_type(node)
        local_name = self.glsl_local_identifier_name(node.name)
        self.local_variable_types[node.name] = var_type
        self.local_variable_source_types[node.name] = self.type_name_string(var_type)
        qualifier = self.local_variable_qualifier(node)
        declaration = f"{qualifier}{self.map_type(var_type)} {local_name}[1]"
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return f"{'    ' * indent}{declaration};\n"
        # The one-element view is built by an array constructor, so a literal
        # initializing a 64-bit scalar carries that type's suffix there too.
        initializer = widened_integer_literal(
            self, initial_value, var_type
        ) or self.generate_expression_with_expected(initial_value, var_type)
        return f"{'    ' * indent}{declaration} = {self.map_type(var_type)}[1]({initializer});\n"

    def glsl_scalar_array_view_element_expression(self, name):
        if name not in getattr(self, "current_glsl_scalar_array_views", {}):
            return None
        return f"{self.generate_glsl_identifier_expression(name)}[0]"

    def glsl_fixed_array_vector_pointer_view_plan(self, function):
        """Summarize an exact thread-pointer view over a struct vector array."""

        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        if len(parameters) != 1:
            return None
        receiver = parameters[0]
        receiver_name = getattr(receiver, "name", None)
        receiver_type = self.glsl_normalized_source_type(
            getattr(receiver, "param_type", getattr(receiver, "vtype", None))
        )
        if not receiver_name or not receiver_type:
            return None
        receiver_struct = self.glsl_source_type_identifier_name(receiver_type)
        member_types = self.struct_member_types.get(receiver_struct)
        if member_types is None:
            member_types = self.struct_member_types.get(receiver_type)
        if member_types is None:
            return None

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(receiver, "qualifiers", []) or []
        }
        if "thread" not in qualifiers or "inout" not in qualifiers:
            return None

        body = getattr(function, "body", None)
        statements = getattr(body, "statements", body)
        if not isinstance(statements, (list, tuple)) or len(statements) != 1:
            return None
        statement = statements[0]
        if not isinstance(statement, ReturnNode):
            return None
        reinterpret = getattr(statement, "value", None)
        if not isinstance(reinterpret, PointerReinterpretNode):
            return None

        target_type = getattr(reinterpret, "target_type", None)
        if not isinstance(target_type, PointerType):
            return None
        if str(getattr(target_type, "address_space", "") or "").lower() != "thread":
            return None
        if not bool(getattr(target_type, "is_mutable", False)):
            return None
        scalar_type = self.glsl_normalized_source_type(
            getattr(target_type, "pointee_type", None)
        )
        if not scalar_type:
            return None

        member_access = getattr(reinterpret, "expression", None)
        if not isinstance(member_access, MemberAccessNode):
            return None
        member_object = getattr(
            member_access,
            "object",
            getattr(member_access, "object_expr", None),
        )
        if not isinstance(member_object, (IdentifierNode, VariableNode)):
            return None
        if getattr(member_object, "name", None) != receiver_name:
            return None
        member_name = str(getattr(member_access, "member", "") or "")
        member_type = member_types.get(member_name)
        outer_array = self.glsl_outer_array_type(self.type_name_string(member_type))
        if outer_array is None:
            return None
        extent_expression, vector_type = outer_array
        extent = evaluate_literal_int_expression(
            extent_expression,
            self.glsl_active_literal_int_constants(),
        )
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            return None

        source_vector_type = self.glsl_normalized_source_type(vector_type)
        try:
            mapped_vector_type = self.map_type(source_vector_type)
        except OpenGLTemplateTypeError:
            return None
        vector_match = re.fullmatch(
            r"(vec|dvec|ivec|uvec|bvec)([24])", mapped_vector_type
        )
        if vector_match is None:
            return None
        # Only canonical source vectors have a known tightly packed scalar view.
        # This deliberately rejects half/narrow aliases that map to the same GLSL
        # vector but have a different source storage width.
        if source_vector_type != mapped_vector_type:
            return None
        vector_prefix, vector_width_text = vector_match.groups()
        vector_width = int(vector_width_text)
        component_type = {
            "vec": "float",
            "dvec": "double",
            "ivec": "int",
            "uvec": "uint",
            "bvec": "bool",
        }[vector_prefix]
        if scalar_type != component_type:
            return None

        return {
            "function_id": id(function),
            "function_name": getattr(function, "name", None),
            "receiver_type": receiver_type,
            "receiver_name": receiver_name,
            "member_name": member_name,
            "array_extent": extent,
            "vector_type": source_vector_type,
            "vector_width": vector_width,
            "scalar_type": scalar_type,
            "mutable": True,
        }

    def prepare_glsl_fixed_array_vector_pointer_views(self, root, functions):
        """Bind representable indexed uses before pointer-return helpers emit."""

        self.glsl_fixed_array_vector_pointer_view_helper_ids = set()
        self.glsl_fixed_array_vector_pointer_view_calls = {}
        functions_by_name = {}
        plans_by_name = {}
        for function in functions or []:
            name = getattr(function, "name", None)
            if not name:
                continue
            functions_by_name.setdefault(name, []).append(function)
            plan = self.glsl_fixed_array_vector_pointer_view_plan(function)
            if plan is not None:
                plans_by_name.setdefault(name, []).append(plan)

        nodes = list(self.walk_ast(root))
        direct_accesses_by_call = {}
        calls_by_name = {}
        for node in nodes:
            if isinstance(node, FunctionCallNode):
                name = self.function_call_name(node)
                if name:
                    calls_by_name.setdefault(name, []).append(node)
            if not isinstance(node, ArrayAccessNode):
                continue
            array_expression = getattr(
                node,
                "array",
                getattr(node, "array_expr", None),
            )
            if isinstance(array_expression, FunctionCallNode):
                direct_accesses_by_call.setdefault(id(array_expression), []).append(
                    node
                )

        contract_fields = (
            "receiver_type",
            "member_name",
            "array_extent",
            "vector_type",
            "vector_width",
            "scalar_type",
            "mutable",
        )
        for name, plans in plans_by_name.items():
            named_functions = functions_by_name.get(name, [])
            if len(plans) != len(named_functions):
                continue
            contracts = {
                tuple(plan[field] for field in contract_fields) for plan in plans
            }
            if len(contracts) != 1:
                continue
            plan = plans[0]
            calls = calls_by_name.get(name, [])
            valid = True
            for call in calls:
                accesses = direct_accesses_by_call.get(id(call), [])
                arguments = list(
                    getattr(call, "arguments", getattr(call, "args", [])) or []
                )
                if len(arguments) != 1 or not accesses:
                    valid = False
                    break
                if any(
                    not self.glsl_side_effect_free_expression(
                        getattr(access, "index", getattr(access, "index_expr", None))
                    )
                    for access in accesses
                ):
                    valid = False
                    break
            if not valid:
                continue
            self.glsl_fixed_array_vector_pointer_view_helper_ids.update(
                candidate["function_id"] for candidate in plans
            )
            for call in calls:
                self.glsl_fixed_array_vector_pointer_view_calls[id(call)] = plan
                # Resource specialization deep-copies call sites after this
                # whole-module proof.  Keep the proven plan on the call node so
                # cloned direct subscripts retain the same rewrite contract.
                call._glsl_fixed_array_vector_pointer_view_plan = plan

    def glsl_fixed_array_vector_pointer_view_access_plan(self, expression):
        if not isinstance(expression, ArrayAccessNode):
            return None
        array_expression = getattr(
            expression,
            "array",
            getattr(expression, "array_expr", None),
        )
        if not isinstance(array_expression, FunctionCallNode):
            return None
        plan = getattr(
            array_expression,
            "_glsl_fixed_array_vector_pointer_view_plan",
            None,
        )
        if plan is None:
            plan = self.glsl_fixed_array_vector_pointer_view_calls.get(
                id(array_expression)
            )
        if plan is None:
            return None

        arguments = list(
            getattr(
                array_expression,
                "arguments",
                getattr(array_expression, "args", []),
            )
            or []
        )
        receiver = arguments[0] if len(arguments) == 1 else None
        receiver_type = self.glsl_normalized_source_type(
            self.glsl_source_expression_type(receiver)
        )
        expected_receiver_type = self.glsl_normalized_source_type(plan["receiver_type"])
        if receiver_type != expected_receiver_type:
            raise PointerReinterpretationError(
                "OpenGL cannot apply a fixed-array vector pointer-view rewrite "
                "without the exact proven receiver type",
                source_type=receiver_type,
                target_type=expected_receiver_type,
                address_space="thread",
                access="read-write",
                target_backend="opengl",
                reason="fixed-array-vector-view-receiver-unproven",
                source_location=getattr(array_expression, "source_location", None),
            )
        return plan

    def generate_glsl_fixed_array_vector_pointer_view_access(self, expression):
        plan = self.glsl_fixed_array_vector_pointer_view_access_plan(expression)
        if plan is None:
            return None
        call = getattr(expression, "array", getattr(expression, "array_expr", None))
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        if len(arguments) != 1:
            return None
        receiver = arguments[0]
        receiver_expression = self.generate_expression_with_expected(receiver, None)
        member_name = self.glsl_member_access_name(
            receiver,
            plan["member_name"],
        )
        index_expression = getattr(
            expression,
            "index",
            getattr(expression, "index_expr", None),
        )
        total_extent = plan["array_extent"] * plan["vector_width"]
        rendered_index = self.glsl_index_expression(
            index_expression,
            call,
            indexed_extent=total_extent,
            target_signed=True,
        )
        literal_index = self.literal_int_value(
            index_expression,
            self.glsl_active_literal_int_constants(),
        )
        if isinstance(literal_index, int) and not isinstance(literal_index, bool):
            outer_index = str(literal_index // plan["vector_width"])
            component_index = str(literal_index % plan["vector_width"])
        else:
            outer_index = f"({rendered_index} / {plan['vector_width']})"
            component_index = f"({rendered_index} % {plan['vector_width']})"
        return f"{receiver_expression}.{member_name}[{outer_index}][{component_index}]"

    def glsl_tail_recursive_void_call_ids(self, function):
        return_type = self.type_name_string(getattr(function, "return_type", None))
        if self.map_type(return_type or "void") != "void":
            return set()
        source_name = getattr(function, "_glsl_resource_source_name", None) or getattr(
            function, "name", None
        )
        if not source_name:
            return set()
        recursive_calls = {
            id(node): node
            for node in self.walk_ast(getattr(function, "body", []))
            if isinstance(node, FunctionCallNode)
            and self.function_call_name(node) == source_name
        }
        if not recursive_calls:
            return set()

        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            if getattr(parameter, "name", None)
        }
        if any(
            isinstance(node, (VariableNode, ArrayNode))
            and getattr(node, "name", None) in parameter_names
            for node in self.walk_ast(getattr(function, "body", []))
        ):
            # A same-spelled local can be the recursive argument while the loop
            # state lives in the outer parameter.  Name-only equality cannot
            # distinguish those bindings, and assigning the outer parameter is
            # not expressible from inside the shadowing scope without first
            # renaming it.  Keep the source recursion visible so the existing
            # recursion detector rejects it rather than silently looping with
            # stale state.
            return set()

        lowered = set()
        valid = True

        def body_statements(body):
            if body is None:
                return []
            if isinstance(body, list):
                return body
            statements = getattr(body, "statements", None)
            if statements is not None:
                return list(statements or [])
            return [body]

        def direct_call(statement):
            if isinstance(statement, FunctionCallNode):
                return statement
            if isinstance(statement, ExpressionStatementNode) or (
                hasattr(statement, "__class__")
                and "ExpressionStatementNode" in str(type(statement))
            ):
                expression = getattr(statement, "expression", None)
                return expression if isinstance(expression, FunctionCallNode) else None
            return None

        def contains_recursive_call(value):
            return any(id(node) in recursive_calls for node in self.walk_ast(value))

        def visit_sequence(body, tail_position):
            nonlocal valid
            statements = body_statements(body)
            for index, statement in enumerate(statements):
                visit_statement(
                    statement,
                    tail_position and index == len(statements) - 1,
                )

        def visit_statement(statement, tail_position):
            nonlocal valid
            call = direct_call(statement)
            if call is not None and id(call) in recursive_calls:
                if not tail_position:
                    valid = False
                    return
                lowered.add(id(call))
                return
            if isinstance(statement, BlockNode):
                visit_sequence(statement, tail_position)
                return
            if isinstance(statement, IfNode):
                conditions = [
                    getattr(
                        statement,
                        "condition",
                        getattr(statement, "if_condition", None),
                    ),
                    *(getattr(statement, "else_if_conditions", []) or []),
                ]
                if any(contains_recursive_call(condition) for condition in conditions):
                    valid = False
                    return
                visit_sequence(getattr(statement, "if_body", None), tail_position)
                for branch in getattr(statement, "else_if_bodies", []) or []:
                    visit_sequence(branch, tail_position)
                visit_sequence(getattr(statement, "else_body", None), tail_position)
                return
            if contains_recursive_call(statement):
                valid = False

        visit_sequence(getattr(function, "body", []), True)
        if not valid or lowered != set(recursive_calls):
            return set()
        return lowered

    @staticmethod
    def glsl_tail_recursive_argument_is_same_parameter(argument, parameter_name):
        if isinstance(argument, str):
            return argument.strip() == parameter_name
        if isinstance(argument, (IdentifierNode, VariableNode)):
            return getattr(argument, "name", None) == parameter_name
        return False

    def generate_glsl_tail_recursive_call_statement(self, expression, indent):
        if not isinstance(expression, FunctionCallNode) or id(
            expression
        ) not in getattr(self, "current_glsl_tail_recursive_call_ids", set()):
            return None
        function_name = self.function_call_name(expression)
        arguments = list(
            getattr(expression, "arguments", getattr(expression, "args", [])) or []
        )
        specialized = self.glsl_resource_function_call_specialization(
            function_name,
            arguments,
            call_node=expression,
        )
        if specialized is not None:
            target_function = specialized
            target_name = self.glsl_function_declaration_name(specialized)
            call_arguments = self.glsl_resource_specialized_call_arguments(
                specialized,
                arguments,
            )
        else:
            target_function = self.resolve_glsl_function_overload(
                function_name,
                arguments,
                call_node=expression,
            ) or self.function_definitions.get(function_name)
            target_name = (
                self.glsl_function_declaration_name(target_function)
                if target_function is not None
                else function_name
            )
            call_arguments = arguments
        if target_name != self.glsl_function_graph_name(self.current_glsl_function_key):
            return None

        parameters = list(
            getattr(
                target_function,
                "parameters",
                getattr(target_function, "params", []),
            )
            or []
        )
        skipped = self.skipped_function_parameter_indices(target_name)
        parameters = [
            parameter
            for index, parameter in enumerate(parameters)
            if index not in skipped
        ]
        call_arguments = self.filter_sampler_arguments(target_name, call_arguments)
        if len(parameters) != len(call_arguments):
            return None

        pending = []
        for parameter, argument in zip(parameters, call_arguments):
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                return None
            if self.glsl_tail_recursive_argument_is_same_parameter(
                argument,
                parameter_name,
            ):
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            if qualifiers.intersection(
                {"const", "constant", "readonly", "out", "inout"}
            ):
                return None
            parameter_type = getattr(
                parameter,
                "param_type",
                getattr(parameter, "vtype", None),
            )
            if isinstance(parameter_type, (ArrayType, PointerType, ReferenceType)) or (
                self.type_contains_opaque_resource(parameter_type)
            ):
                return None
            rendered = self.generate_expression_with_expected(argument, parameter_type)
            temporary = self.glsl_synthetic_local_identifier(
                f"{parameter_name}_tail_value"
            )
            self.local_variable_types[temporary] = parameter_type
            self.local_variable_source_types[temporary] = self.type_name_string(
                parameter_type
            )
            pending.append((parameter_name, parameter_type, temporary, rendered))

        indent_str = "    " * indent
        code = ""
        for _parameter_name, parameter_type, temporary, rendered in pending:
            declaration = format_c_style_array_declaration(
                self.map_type(parameter_type),
                temporary,
            )
            code += f"{indent_str}{declaration} = {rendered};\n"
        for parameter_name, _parameter_type, temporary, _rendered in pending:
            target = self.generate_glsl_identifier_expression(parameter_name)
            code += f"{indent_str}{target} = {temporary};\n"
        code += f"{indent_str}continue;\n"
        return code

    def generate_glsl_software_subgroup_barrier_statement(self, expr, indent):
        """A statement ``subgroupBarrier()`` under software subgroups, or ``None``.

        Software subgroups are slices of the workgroup, so their execution barrier is the workgroup
        ``barrier()``. ``barrier()`` orders only shared memory (GLSL 4.60 section 8.16), so full
        ``memoryBarrier()`` fences before and after it also order the buffer and image memory the subgroup
        barrier covers (GL_KHR_shader_subgroup), releasing earlier writes and acquiring them afterwards.
        """
        if (
            self.software_subgroup_width is None
            or not isinstance(expr, FunctionCallNode)
            or self.function_call_name(expr) != "subgroupBarrier"
            or "subgroupBarrier" in self.function_return_types
        ):
            return None
        arguments = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        control = self.generate_glsl_software_subgroup_operation(
            "subgroupExecutionBarrier",
            arguments,
            source_location=getattr(expr, "source_location", None),
        )
        indent_str = "    " * indent
        return "".join(
            f"{indent_str}{line};\n"
            for line in ("memoryBarrier()", control, "memoryBarrier()")
        )

    def glsl_dynamic_resource_call_dispatch_info(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None
        func_name = self.function_call_name(expr)
        if not func_name:
            return None
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        dynamic_info = self.glsl_dynamic_resource_call_info(
            func_name,
            args,
            self.current_resource_aliases,
        )
        if dynamic_info is None:
            return None

        cases = []
        for index in range(dynamic_info["array_size"]):
            static_args = list(args)
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            specialized = self.glsl_resource_function_call_specialization(
                func_name,
                static_args,
            )
            if specialized is None:
                return None
            call_args = self.glsl_resource_specialized_call_arguments(
                specialized,
                static_args,
            )
            target_name = self.glsl_function_declaration_name(specialized)
            rendered_args = ", ".join(
                self.generate_function_call_arguments(target_name, call_args)
            )
            cases.append((index, f"{target_name}({rendered_args})"))

        return {
            "index_expr": dynamic_info["index_expr"],
            "indexed_expr": dynamic_info["array_expr"],
            "cases": cases,
            "return_type": self.expression_result_type(expr),
        }

    def generate_glsl_dynamic_resource_switch_statement(
        self,
        dispatch,
        indent,
        case_statement,
        default_statement=None,
    ):
        indent_str = "    " * indent
        case_indent = "    " * (indent + 1)
        selector = self.glsl_index_expression(
            dispatch["index_expr"],
            dispatch.get("indexed_expr"),
        )
        code = f"{indent_str}switch ({selector}) {{\n"
        for index, call in dispatch["cases"]:
            code += f"{indent_str}case {index}:\n"
            code += f"{case_indent}{case_statement(call)}\n"
            code += f"{case_indent}break;\n"
        code += f"{indent_str}default:\n"
        if default_statement is not None:
            code += f"{case_indent}{default_statement}\n"
        code += f"{case_indent}break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_glsl_dynamic_resource_call_assignment_statement(
        self,
        expr,
        target,
        target_type,
        indent,
    ):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None
        return self.generate_glsl_dynamic_resource_switch_statement(
            dispatch,
            indent,
            lambda call: f"{target} = {call};",
            f"{target} = {self.zero_value_expression(target_type)};",
        )

    def generate_glsl_dynamic_resource_assignment_node(self, stmt, indent):
        left_node = getattr(stmt, "target", getattr(stmt, "left", None))
        right_node = getattr(stmt, "value", getattr(stmt, "right", None))
        op = self.map_operator(getattr(stmt, "operator", getattr(stmt, "op", "=")))
        if op != "=":
            return None
        if self.glsl_dynamic_resource_call_dispatch_info(right_node) is None:
            return None
        self.validate_glsl_buffer_block_assignment_target(left_node, op)
        expected_type = self.glsl_tessellation_factor_assignment_expected_type(
            left_node
        )
        if expected_type is not None:
            self.validate_glsl_tessellation_factor_assignment_value(
                left_node, right_node
            )
        else:
            expected_type = self.expression_result_type(left_node)
        left = self.generate_glsl_buffer_block_mutation_target(left_node)
        return self.generate_glsl_dynamic_resource_call_assignment_statement(
            right_node,
            left,
            expected_type,
            indent,
        )

    def generate_glsl_dynamic_resource_call_expression_statement(self, expr, indent):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None
        return self.generate_glsl_dynamic_resource_switch_statement(
            dispatch,
            indent,
            lambda call: f"{call};",
        )

    def generate_glsl_dynamic_resource_call_return_statement(self, expr, indent):
        dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is None:
            return None

        indent_str = "    " * indent
        case_indent = "    " * (indent + 1)
        selector = self.glsl_index_expression(
            dispatch["index_expr"],
            dispatch.get("indexed_expr"),
        )
        code = f"{indent_str}switch ({selector}) {{\n"
        for index, call in dispatch["cases"]:
            code += f"{indent_str}case {index}:\n"
            code += self.generate_glsl_return_value_statement(call, indent + 1)
        return_type = dispatch.get("return_type") or self.current_function_return_type
        mapped_return_type = self.map_type(return_type)
        code += f"{indent_str}default:\n"
        if mapped_return_type == "void":
            code += f"{case_indent}return;\n"
        else:
            code += self.generate_glsl_return_value_statement(
                self.zero_value_expression(return_type), indent + 1
            )
        code += f"{indent_str}}}\n"
        return code

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        value_expr = getattr(stmt, "expression", None)

        stage_struct_return = self.generate_stage_struct_constructor_return(
            value_expr,
            indent,
        )
        if stage_struct_return is not None:
            return stage_struct_return

        indent_str = "    " * indent
        if self.current_stage_output is not None:
            value = self.generate_expression_with_expected(
                value_expr,
                self.current_function_return_type,
            )
            return (
                f"{indent_str}{self.current_stage_output['name']} = {value};\n"
                f"{indent_str}return;\n"
            )

        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        value = self.generate_expression_with_expected(value_expr, return_type)
        return self.generate_glsl_return_value_statement(value, indent)

    def local_variable_declared_type(self, stmt):
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        var_type_name = self.type_name_string(var_type)
        if var_type_name in {None, "auto"}:
            inferred_type = inferred_value_type(
                self, getattr(stmt, "initial_value", None), GLSL_FAMILY
            )
            if inferred_type is not None:
                return inferred_type
            return self.refuse_lowering(
                f"local declaration: the type of '{getattr(stmt, 'name', '?')}' is not "
                "written and cannot be inferred from its initializer",
                "float",
            )
        return var_type_name

    def glsl_identifier_requires_escape(self, name, extra_reserved=()):
        name = str(name or "")
        return (
            not name
            or name in self.GLSL_RESERVED_IDENTIFIERS
            or name in extra_reserved
            or name.startswith(self.GLSL_RESERVED_IDENTIFIER_PREFIXES)
            or "__" in name
        )

    def glsl_sanitized_identifier_base(
        self,
        name,
        fallback="identifier",
        extra_reserved=(),
    ):
        source_name = str(name or "")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source_name) and not (
            self.glsl_identifier_requires_escape(source_name, extra_reserved)
        ):
            return source_name
        candidate = sanitize_type_name(source_name) or sanitize_type_name(fallback)
        candidate = candidate or "identifier"
        candidate = re.sub(r"_+", "_", candidate)
        if candidate[0].isdigit():
            candidate = f"identifier_{candidate}"
        if source_name.startswith(
            self.GLSL_RESERVED_IDENTIFIER_PREFIXES
        ) or candidate.startswith(self.GLSL_RESERVED_IDENTIFIER_PREFIXES):
            candidate = f"crossgl_{candidate.lstrip('_')}"
            candidate = re.sub(r"_+", "_", candidate)
        if candidate in self.GLSL_RESERVED_IDENTIFIERS or candidate in extra_reserved:
            candidate = f"{candidate}_"
        return candidate

    def glsl_unique_identifier(
        self,
        base_name,
        used_names,
        *,
        extra_reserved=(),
    ):
        base_name = self.glsl_sanitized_identifier_base(
            base_name,
            extra_reserved=extra_reserved,
        )
        candidate = base_name
        suffix = 2
        while candidate in used_names or self.glsl_identifier_requires_escape(
            candidate,
            extra_reserved,
        ):
            separator = "" if base_name.endswith("_") else "_"
            candidate = f"{base_name}{separator}{suffix}"
            suffix += 1
        return candidate

    def glsl_identifier_map(self, names, *, reserved_names=(), extra_reserved=()):
        unique_names = []
        seen = set()
        for name in names or ():
            if not name:
                continue
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            unique_names.append(name)

        external_reserved_names = {
            str(name)
            for name in reserved_names or ()
            if name and not self.glsl_identifier_requires_escape(name, extra_reserved)
        }
        used_names = set(external_reserved_names)
        used_names.update(
            name
            for name in unique_names
            if not self.glsl_identifier_requires_escape(name, extra_reserved)
        )

        identifier_map = {}
        for name in sorted(unique_names):
            if (
                not self.glsl_identifier_requires_escape(name, extra_reserved)
                and name not in external_reserved_names
            ):
                identifier_map[name] = name
                continue
            emitted_name = self.glsl_unique_identifier(
                self.glsl_sanitized_identifier_base(
                    name,
                    extra_reserved=extra_reserved,
                ),
                used_names,
                extra_reserved=extra_reserved,
            )
            identifier_map[name] = emitted_name
            used_names.add(emitted_name)
        return identifier_map

    def prepare_glsl_module_identifier_names(
        self,
        names,
        *,
        type_names=(),
        reserved_names=(),
    ):
        self.glsl_module_identifier_names = self.glsl_identifier_map(
            names,
            reserved_names=reserved_names,
        )
        self.glsl_module_generated_identifier_names = {}
        self.glsl_module_used_identifier_names = {
            str(name)
            for name in reserved_names or ()
            if name and not self.glsl_identifier_requires_escape(name)
        }
        self.glsl_module_used_identifier_names.update(
            self.glsl_module_identifier_names.values()
        )
        self.glsl_type_identifier_names = {
            str(name): self.glsl_module_identifier_names.get(
                str(name),
                self.glsl_sanitized_identifier_base(name),
            )
            for name in type_names or ()
            if name
        }

    def glsl_module_identifier_name(self, name):
        if not name:
            return name
        name = str(name)
        emitted_name = self.glsl_module_identifier_names.get(name)
        if emitted_name is not None:
            return emitted_name
        if name in self.glsl_module_identifier_names.values() or name in (
            self.glsl_module_generated_identifier_names.values()
        ):
            return name

        emitted_name = self.glsl_unique_identifier(
            self.glsl_sanitized_identifier_base(name),
            self.glsl_module_used_identifier_names,
        )
        self.glsl_module_identifier_names[name] = emitted_name
        self.glsl_module_used_identifier_names.add(emitted_name)
        return emitted_name

    def glsl_generated_module_identifier(self, identity, base_name):
        if identity in self.glsl_module_generated_identifier_names:
            return self.glsl_module_generated_identifier_names[identity]
        emitted_name = self.glsl_unique_identifier(
            self.glsl_sanitized_identifier_base(base_name),
            self.glsl_module_used_identifier_names,
        )
        self.glsl_module_generated_identifier_names[identity] = emitted_name
        self.glsl_module_used_identifier_names.add(emitted_name)
        return emitted_name

    def glsl_type_identifier_name(self, name):
        if not name:
            return name
        emitted_name = self.glsl_type_identifier_names.get(str(name))
        if emitted_name is not None:
            return emitted_name
        return self.glsl_module_identifier_name(name)

    def glsl_source_type_identifier_name(self, name):
        for source_name, emitted_name in self.glsl_type_identifier_names.items():
            if emitted_name == name:
                return source_name
        return name

    def glsl_struct_node(self, type_name):
        source_type_name = self.glsl_source_type_identifier_name(type_name)
        return self.structs_by_name.get(source_type_name)

    def glsl_struct_member_types(self, type_name):
        source_type_name = self.glsl_source_type_identifier_name(type_name)
        return self.struct_member_types.get(source_type_name, {})

    def prepare_glsl_local_identifier_names(self, function):
        names = self.collect_glsl_function_local_identifier_names(function)
        global_names = set(self.glsl_global_identifier_names)
        shared_names = names & global_names
        identifier_names = {
            name: self.glsl_module_identifier_name(name) for name in shared_names
        }
        reserved_names = {
            self.glsl_module_identifier_name(name) for name in global_names
        }
        reserved_names.update(self.current_identifier_aliases.values())
        identifier_names.update(
            self.glsl_identifier_map(
                names - shared_names,
                reserved_names=reserved_names,
                extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
            )
        )
        self.current_glsl_local_identifier_names = identifier_names

    def glsl_local_declaration_identifier_name(self, name, reserved_names=()):
        existing = self.current_identifier_aliases.get(name)
        if existing is not None and name in self.current_glsl_local_identifier_names:
            return existing

        preallocated = name in self.current_glsl_local_identifier_names
        emitted_name = self.current_glsl_local_identifier_names.get(name)
        if emitted_name is None:
            emitted_name = self.glsl_sanitized_identifier_base(
                name,
                extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
            )

        used_names = set(self.current_glsl_local_identifier_names.values())
        used_names.update(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS)
        if preallocated:
            used_names.discard(emitted_name)
            # Releasing this local's reservation must not release another alias.
            used_names.update(
                alias
                for source, alias in self.current_identifier_aliases.items()
                if source != name
            )
        # Nor a name the stage's inputs and outputs already declare: a local must not hide the stage output it
        # is written to, as in ``vec4 worldPosition = ...; output.worldPosition = worldPosition.xyz;``.
        used_names.update(reserved_names)
        if emitted_name in used_names or emitted_name in reserved_names:
            emitted_name = self.glsl_unique_identifier(
                emitted_name,
                used_names,
                extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
            )
        self.current_glsl_local_identifier_names[name] = emitted_name
        if emitted_name != name:
            self.current_identifier_aliases[name] = emitted_name
        return emitted_name

    def glsl_parameter_identifier_name(self, name, parameter_names=None):
        reserved_names = set(parameter_names or ())
        reserved_names.discard(name)
        return self.glsl_local_declaration_identifier_name(
            name,
            reserved_names,
        )

    def glsl_local_identifier_name(self, name):
        reserved_names = self.current_stage_declared_names()
        return self.glsl_local_declaration_identifier_name(name, reserved_names)

    def glsl_global_identifier_name(self, name):
        emitted_name = self.glsl_module_identifier_name(name)
        if emitted_name != name:
            self.current_identifier_aliases[name] = emitted_name
        return emitted_name
