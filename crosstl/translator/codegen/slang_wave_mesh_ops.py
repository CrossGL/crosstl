"""Wave, ray-tracing and mesh operations in the Slang generator: lowering, argument checks and refusals."""

from ..ast import (
    IdentifierNode,
    MeshOpNode,
    UnaryOpNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix
from .slang_wave_shuffles import (
    SLANG_WAVE_SHUFFLE_ARITIES,
    render_slang_wave_shuffle_helper,
    slang_wave_shuffle_helper_name,
)


class SlangWaveMeshOpsMixin:
    """Wave and mesh operation members of ``SlangCodeGen``."""

    def generate_slang_ray_tracing_op_expression(self, node):
        if node.operation in self.user_function_names:
            args = self.generate_user_function_arguments(node.operation, node.arguments)
            return f"{self.convert_type(node.operation)}({args})"
        return self.generate_slang_ray_tracing_call_expression(
            node.operation, self.normalized_slang_intrinsic_args(node.arguments)
        )

    def generate_slang_ray_tracing_call_expression(self, operation, arguments):
        result_type = self.slang_ray_tracing_result_type(operation)
        target_reason = self.slang_ray_tracing_target_type_rejection_reason(result_type)
        if target_reason is not None:
            return self.slang_ray_tracing_diagnostic_expression(
                operation, target_reason
            )

        args = [self.generate_expression(arg) for arg in arguments]
        if operation == "TraceRay" and len(args) == 11:
            ray_desc = f"RayDesc({args[6]}, {args[7]}, {args[8]}, {args[9]})"
            args = args[:6] + [ray_desc, args[10]]
        return f"{operation}({', '.join(args)})"

    def generate_slang_wave_shuffle_expression(self, node):
        """A relative wave shuffle as a call of its helper (see :mod:`.slang_wave_shuffles`)."""
        operation = node.operation
        arguments = list(node.arguments)
        if len(arguments) not in SLANG_WAVE_SHUFFLE_ARITIES[operation]:
            expected = " or ".join(
                str(count) for count in sorted(SLANG_WAVE_SHUFFLE_ARITIES[operation])
            )
            return self.unsupported_slang_wave_op_expression(
                operation, f"expects {expected} arguments, got {len(arguments)}"
            )
        value_type = self.slang_wave_argument_mapped_type(arguments[0])
        if value_type is None or not self.is_slang_wave_value_type(value_type):
            return self.unsupported_slang_wave_op_expression(
                operation,
                f"value must be a scalar or vector, got {value_type or 'an unknown type'}",
            )
        if operation == "WaveShuffleAndFillUp":
            fill_type = self.slang_wave_argument_mapped_type(arguments[1])
            if fill_type is not None and fill_type != value_type:
                return self.unsupported_slang_wave_op_expression(
                    operation, f"fill must be {value_type}, got {fill_type}"
                )
        conversion = self.slang_wave_result_conversion(value_type)
        if conversion is None:
            target_reason = self.slang_wave_target_type_rejection_reason(value_type)
            if target_reason is not None:
                return self.unsupported_slang_wave_op_expression(
                    operation, target_reason
                )
        name = slang_wave_shuffle_helper_name(operation, value_type)
        self.helper_functions.setdefault(
            name, render_slang_wave_shuffle_helper(operation, value_type)
        )
        rendered = [self.generate_expression(argument) for argument in arguments]
        if operation == "WaveShuffleAndFillUp":
            modulo = (
                f"uint({rendered[3]})" if len(rendered) == 4 else "WaveGetLaneCount()"
            )
            call = (
                f"{name}({rendered[0]}, {rendered[1]}, uint({rendered[2]}), {modulo})"
            )
        else:
            call = f"{name}({rendered[0]}, uint({rendered[1]}))"
        return f"{conversion}({call})" if conversion else call

    def slang_wave_result_conversion(self, result_type):
        """The type a wave result converts to where it is used, when the use expects another value type.

        Metal converts a returned or assigned value implicitly, as ``bool shuffle(bool b) { return
        simd_shuffle(uint(b), lane); }`` does; the conversion is written out so slangc applies the same one.
        ``None`` when no conversion is needed or the shapes differ.
        """
        expected = self.slang_wave_expected_target_type()
        result = self.type_name_string(result_type)
        if result:
            result = self.convert_type(result)
        if not expected or not result or expected == result:
            return None
        if not (
            self.is_slang_wave_value_type(expected)
            and self.is_slang_wave_value_type(result)
        ):
            return None
        if self.is_vector_value_type(expected) != self.is_vector_value_type(result):
            return None
        if self.is_vector_value_type(expected) and expected[-1] != result[-1]:
            return None
        return expected

    def slang_wave_result_type(self, node):
        operation = getattr(node, "operation", None)
        args = getattr(node, "arguments", []) or []
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return "uint"
        if operation in {"WaveIsFirstLane", "WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "bool"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uint4"
        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "WavePrefixSum",
            "WavePrefixProduct",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            return self.expression_result_type(args[0]) if args else None
        return None

    def slang_wave_argument_rejection_reason(self, operation, args):
        integer_bit_ops = {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
        if operation in integer_bit_ops:
            value_type = self.slang_wave_argument_mapped_type(args[0])
            if value_type is not None and not self.is_slang_wave_integer_value_type(
                value_type
            ):
                return f"value must be scalar or vector int or uint, got {value_type}"

        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveBallot",
        }:
            predicate_type = self.slang_wave_argument_mapped_type(args[0])
            if predicate_type is not None and predicate_type != "bool":
                return f"predicate must be scalar bool, got {predicate_type}"

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            lane_type = self.slang_wave_argument_mapped_type(args[1])
            if lane_type is not None and lane_type not in {"int", "uint"}:
                return f"lane index must be scalar int or uint, got {lane_type}"

        if operation.startswith("WaveMultiPrefix"):
            mask_type = self.slang_wave_argument_mapped_type(args[1])
            if mask_type is not None and mask_type != "uint4":
                return f"partition mask must be uint4, got {mask_type}"

        return None

    def slang_wave_argument_mapped_type(self, arg):
        arg_type = self.expression_result_type(arg)
        if arg_type is None:
            return None
        return self.convert_type(arg_type)

    def slang_wave_target_type_rejection_reason(self, result_type):
        expected_type = self.slang_wave_expected_target_type()
        result_type = self.type_name_string(result_type)
        if result_type:
            result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if not self.is_slang_wave_value_type(result_type):
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def slang_wave_expected_target_type(self):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            return None
        expected_type = self.convert_type(expected_type)
        if expected_type == "auto":
            return None
        if self.is_slang_wave_value_type(expected_type):
            return expected_type
        return None

    def is_slang_wave_value_type(self, type_name):
        return self.is_scalar_value_type(type_name) or self.is_vector_value_type(
            type_name
        )

    def is_slang_wave_integer_value_type(self, type_name):
        return type_name in {
            "int",
            "uint",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
        }

    def unsupported_slang_wave_op_expression(self, operation, reason):
        return self.refuse_lowering(
            f"Slang wave intrinsic: {operation} {reason}",
            self.slang_wave_fallback_value(operation),
        )

    def slang_wave_fallback_value(self, operation):
        default_value = self.slang_wave_default_value(operation)
        expected_type = self.slang_wave_expected_target_type()
        if expected_type and not self.slang_wave_default_matches_expected_type(
            default_value, expected_type
        ):
            return self.zero_value_for_type(expected_type)
        return default_value

    def slang_wave_default_matches_expected_type(self, default_value, expected_type):
        if self.is_vector_value_type(expected_type):
            return default_value.startswith(f"{expected_type}(")
        if expected_type == "bool":
            return default_value == "false"
        if expected_type in {"int", "uint", "float", "double"}:
            return default_value in {"0", "0u"}
        return False

    def slang_wave_default_value(self, operation):
        if operation in {"WaveIsFirstLane", "WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "false"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uint4(0)"
        return "0"

    def generate_slang_mesh_op_expression(self, node):
        expected_arities = self.SLANG_MESH_INTRINSIC_ARITIES.get(node.operation)
        if expected_arities is None:
            return self.unsupported_slang_mesh_op_expression(
                node.operation, "is not recognized by the Slang backend"
            )

        arguments = self.normalized_slang_intrinsic_args(node.arguments)
        actual_arity = len(arguments)
        rejection_reason = None
        if actual_arity not in expected_arities:
            expected = " or ".join(str(arity) for arity in sorted(expected_arities))
            rejection_reason = f"expects {expected} arguments, got {actual_arity}"
        else:
            self.validate_slang_mesh_op_stage(node.operation)
            rejection_reason = self.slang_mesh_argument_rejection_reason(
                node.operation, arguments
            )
            if rejection_reason is None:
                self.validate_slang_dispatch_mesh_payload_argument(node, arguments)
                target_reason = self.slang_mesh_target_type_rejection_reason()
                if target_reason is not None:
                    return self.unsupported_slang_mesh_op_expression(
                        node.operation, target_reason
                    )
                args = ", ".join(self.generate_expression(arg) for arg in arguments)
                return f"{node.operation}({args})"

        if rejection_reason is not None:
            return self.unsupported_slang_mesh_op_expression(
                node.operation, rejection_reason
            )

    def validate_slang_mesh_op_stage_for_shader(self, operation, shader_type):
        if not shader_type:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if operation == "SetMeshOutputCounts" and shader_stage != "mesh":
            raise SlangRefusal(
                f"Slang {shader_type} stage cannot call "
                "SetMeshOutputCounts; SetMeshOutputCounts is only valid in mesh stages"
            )
        if operation == "DispatchMesh" and shader_stage != "amplification":
            raise SlangRefusal(
                f"Slang {shader_type} stage cannot call DispatchMesh; "
                "DispatchMesh is only valid in amplification/task/object stages"
            )

    def validate_slang_mesh_op_stage(self, operation):
        self.validate_slang_mesh_op_stage_for_shader(
            operation, self.current_shader_type
        )

    def slang_mesh_intrinsic_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, MeshOpNode):
                calls.append(
                    (
                        getattr(node, "operation", None),
                        self.normalized_slang_intrinsic_args(
                            getattr(node, "arguments", [])
                        ),
                    )
                )
                continue
            call_name = self.slang_function_call_name(node)
            if call_name in self.SLANG_MESH_INTRINSIC_ARITIES:
                calls.append(
                    (call_name, getattr(node, "arguments", getattr(node, "args", [])))
                )
        return calls

    def validate_slang_mesh_intrinsic_calls(
        self, func, shader_type, visited_helpers=None
    ):
        if visited_helpers is None:
            visited_helpers = set()

        saved_variable_types = self.variable_types.copy()
        try:
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for operation, args in self.slang_mesh_intrinsic_calls(func):
                if operation in self.SLANG_MESH_INTRINSIC_ARITIES:
                    expected_arities = self.SLANG_MESH_INTRINSIC_ARITIES[operation]
                    if len(args) not in expected_arities:
                        continue
                    self.validate_slang_mesh_op_stage_for_shader(operation, shader_type)

            for helper_call in self.slang_user_function_call_nodes(func):
                helper_name = self.slang_function_call_name(helper_call)
                if helper_name in visited_helpers:
                    continue
                helper_func = self.user_functions_by_name.get(helper_name)
                if helper_func is None or helper_func is func:
                    continue
                self.validate_slang_mesh_helper_call_arguments(helper_call, shader_type)
                self.validate_slang_mesh_intrinsic_calls(
                    helper_func, shader_type, visited_helpers | {helper_name}
                )
        finally:
            self.variable_types = saved_variable_types

    def slang_mesh_interface_argument_role(self, operation, args):
        if operation == "DispatchMesh" and len(args) >= 4:
            return args[3], "payload"
        return None, None

    def slang_mesh_interface_parameter_roles(self, func, visited_helpers=None):
        if visited_helpers is None:
            visited_helpers = set()

        func_name = getattr(func, "name", None)
        if func_name in visited_helpers:
            return {}
        if func_name:
            visited_helpers = visited_helpers | {func_name}

        parameters = getattr(func, "parameters", getattr(func, "params", []))
        parameter_names = {
            getattr(parameter, "name", None) for parameter in parameters or []
        }
        parameter_names.discard(None)
        roles = {}
        alias_roots = self.slang_parameter_alias_roots(func, parameter_names)

        for operation, args in self.slang_mesh_intrinsic_calls(func):
            argument, role = self.slang_mesh_interface_argument_role(operation, args)
            if argument is None:
                continue
            root_name = self.slang_expression_root_identifier(argument)
            root_name = self.slang_resolve_alias_root(root_name, alias_roots)
            if root_name in parameter_names and root_name not in roles:
                roles[root_name] = role

        for call_node in self.slang_user_function_call_nodes(func):
            helper_name = self.slang_function_call_name(call_node)
            if helper_name in visited_helpers:
                continue
            helper_func = self.user_functions_by_name.get(helper_name)
            if helper_func is None:
                continue
            helper_roles = self.slang_mesh_interface_parameter_roles(
                helper_func, visited_helpers
            )
            if not helper_roles:
                continue
            helper_params = getattr(
                helper_func, "parameters", getattr(helper_func, "params", [])
            )
            args = getattr(call_node, "arguments", getattr(call_node, "args", []))
            for index, helper_param in enumerate(helper_params or []):
                helper_param_name = getattr(helper_param, "name", None)
                if helper_param_name not in helper_roles or index >= len(args):
                    continue
                root_name = self.slang_expression_root_identifier(args[index])
                root_name = self.slang_resolve_alias_root(root_name, alias_roots)
                if root_name in parameter_names and root_name not in roles:
                    roles[root_name] = helper_roles[helper_param_name]

        return roles

    def validate_slang_mesh_helper_call_arguments(self, call_node, shader_type):
        helper_name = self.slang_function_call_name(call_node)
        helper_func = self.user_functions_by_name.get(helper_name)
        if helper_func is None:
            return

        roles = self.slang_mesh_interface_parameter_roles(helper_func)
        if not roles:
            return

        parameters = getattr(
            helper_func, "parameters", getattr(helper_func, "params", [])
        )
        args = getattr(call_node, "arguments", getattr(call_node, "args", []))
        for index, parameter in enumerate(parameters or []):
            param_name = getattr(parameter, "name", None)
            if param_name not in roles or index >= len(args):
                continue
            role = roles[param_name]
            expected_type = self.slang_parameter_type_name(parameter)
            if not expected_type:
                continue
            expected_compare_type = (
                self.reference_referent_type_name(expected_type) or expected_type
            )
            expected_base, expected_suffix = split_array_type_suffix(
                self.convert_type(expected_compare_type)
            )
            actual_base, actual_suffix = (
                self.slang_expression_mapped_base_and_array_suffix(args[index])
            )
            if actual_base is None:
                continue
            if actual_base == expected_base and actual_suffix == expected_suffix:
                if role == "payload" and self.slang_parameter_requires_lvalue_argument(
                    parameter
                ):
                    self.validate_slang_ray_lvalue_argument(
                        args[index], shader_type, helper_name, role
                    )
                continue
            actual_type = f"{actual_base}{actual_suffix}"
            expected_label = self.convert_type(expected_type)
            raise SlangRefusal(
                f"Slang {shader_type} {helper_name} {role} "
                f"argument type {actual_type} must match parameter type "
                f"{expected_label}"
            )

    def validate_slang_dispatch_mesh_payload_argument(self, node, arguments=None):
        if arguments is None:
            arguments = self.normalized_slang_intrinsic_args(node.arguments)
        if node.operation != "DispatchMesh" or len(arguments) != 4:
            return

        payload_types = sorted(self.slang_mesh_payload_parameter_types)
        if not payload_types and not getattr(
            self, "slang_program_has_mesh_stage", True
        ):
            # A task stage built alone: the mesh stage it launches is another program.
            return
        if not payload_types:
            raise SlangRefusal(
                "Slang DispatchMesh payload argument requires a mesh stage "
                "@mesh_payload parameter"
            )

        payload_argument = arguments[3]
        payload_type = self.slang_dispatch_mesh_payload_argument_type(payload_argument)
        if payload_type is None:
            return

        if payload_type in payload_types:
            if self.slang_expression_is_lvalue(payload_argument):
                return
            raise SlangRefusal(
                "Slang DispatchMesh payload argument must be an lvalue, "
                f"got {payload_type}"
            )

        expected_label = self.slang_dispatch_mesh_payload_type_label(payload_types)
        raise SlangRefusal(
            f"Slang DispatchMesh payload argument type {payload_type} must match "
            f"mesh payload type {expected_label}"
        )

    def slang_dispatch_mesh_payload_argument_type(self, expr):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            expr
        )
        if base_type is None:
            return None
        if array_suffix:
            return f"{base_type}{array_suffix}"
        return base_type

    def slang_dispatch_mesh_payload_type_label(self, payload_types):
        if len(payload_types) == 1:
            return payload_types[0]
        return " or ".join(payload_types)

    def normalized_slang_intrinsic_args(self, args):
        normalized = []
        index = 0
        while index < len(args):
            arg = args[index]
            if self.is_unary_deref_marker(arg) and index + 1 < len(args):
                normalized.append(UnaryOpNode("*", args[index + 1]))
                index += 2
                continue
            normalized.append(arg)
            index += 1
        return normalized

    def is_unary_deref_marker(self, arg):
        return isinstance(arg, IdentifierNode) and getattr(arg, "name", None) == "*"

    def slang_mesh_argument_rejection_reason(self, operation, args):
        if operation == "SetMeshOutputCounts":
            for index, label in enumerate(("vertex count", "primitive count")):
                count_type = self.slang_mesh_argument_mapped_type(args[index])
                if count_type is not None and count_type not in {"int", "uint"}:
                    return f"{label} must be scalar int or uint, got {count_type}"

        if operation == "DispatchMesh":
            for index, label in enumerate(("x count", "y count", "z count")):
                count_type = self.slang_mesh_argument_mapped_type(args[index])
                if count_type is not None and count_type not in {"int", "uint"}:
                    return f"{label} must be scalar int or uint, got {count_type}"
        return None

    def slang_mesh_argument_mapped_type(self, arg):
        arg_type = self.expression_result_type(arg)
        if arg_type is None:
            return None
        return self.convert_type(arg_type)

    def slang_mesh_target_type_rejection_reason(self):
        expected_type = self.slang_mesh_expected_target_type()
        if expected_type is None:
            return None
        return f"returns void but target expects {expected_type}"

    def slang_mesh_expected_target_type(self):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            return None
        expected_type = self.convert_type(expected_type)
        if expected_type in {"auto", "void"}:
            return None
        return expected_type

    def unsupported_slang_mesh_op_expression(self, operation, reason):
        return self.refuse_lowering(
            f"Slang mesh intrinsic: {operation} {reason}",
            self.slang_mesh_fallback_value(),
        )

    def slang_mesh_fallback_value(self):
        expected_type = self.slang_mesh_expected_target_type()
        if expected_type is not None:
            return self.zero_value_for_type(expected_type)
        return "0"
