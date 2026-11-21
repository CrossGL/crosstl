"""Type mapping of the CUDA generator: CrossGL type spellings, stream and patch types, structured-buffer lengths and counters, and user-function call arguments."""

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
)
from .c_reference_parameters import (
    multi_component_swizzle_arguments,
    overlapping_output_arguments,
)
from .cuda_refusals import CudaLoweringRefused
from .cuda_subscripts import matrix_column_output_arguments
from .generic_function_utils import generic_function_specialization_named
from .resource_arrays import format_array_declarator
from .resource_query import ResourceQueryMixin


class CudaTypeMixin:
    """Type members of ``CudaCodeGen``."""

    def cuda_mapped_type_result(self, mapped_type):
        base_type = str(mapped_type).split("[", 1)[0].strip()
        if base_type == "cgl_half3":
            self.require_cuda_half3_helper()
        if base_type == "cgl_half4":
            self.require_cuda_half4_helper()
        if base_type in {
            "CglRayTracingAccelerationStructure",
            "CglRayDesc",
            "CglRayQuery",
            "CglBuiltInTriangleIntersectionAttributes",
        }:
            self.require_cuda_ray_runtime_helpers()
        return mapped_type

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def get_parameter_type(self, param):
        """Return a parameter type with CUDA resource access metadata applied."""
        param_type = ResourceQueryMixin.get_parameter_type(self, param)
        return self.resource_type_with_access(param_type, param)

    def get_variable_node_type(self, node):
        """Return a variable type with CUDA resource access metadata applied."""
        var_type = ResourceQueryMixin.get_variable_node_type(self, node)
        return self.resource_type_with_access(var_type, node)

    def map_type(self, type_name):
        return self.convert_crossgl_type_to_cuda(type_name)

    def cuda_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.convert_crossgl_type_to_cuda(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, generic_arg = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.convert_crossgl_type_to_cuda(generic_arg)

    def cuda_geometry_stream_mapped_type(self, type_name):
        stream_info = self.cuda_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglCuda{stream_name}<{output_type}>"

    def format_cuda_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.cuda_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"{stream_type}& {name}"

    def cuda_tessellation_patch_type_info(self, type_name):
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None
        base_name, args = parts
        if base_name not in {"InputPatch", "OutputPatch"} or len(args) != 2:
            return None
        element_type, point_count = args
        if not element_type or not point_count:
            return None
        mapped_element_type = self.convert_crossgl_type_to_cuda(element_type)
        return base_name, mapped_element_type, point_count

    def cuda_tessellation_patch_mapped_type(self, type_name):
        patch_info = self.cuda_tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        base_name, element_type, point_count = patch_info
        helper_name = {
            "InputPatch": "CglCudaInputPatch",
            "OutputPatch": "CglCudaOutputPatch",
        }[base_name]
        return f"{helper_name}<{element_type}, {point_count}>"

    def strip_cuda_indirect_type_qualifiers(self, type_name):
        """Return the underlying type after CUDA pointer/reference wrappers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        stripped = type_name.strip()
        changed = True
        while changed and stripped:
            changed = False
            if stripped.endswith("&") or stripped.endswith("*"):
                stripped = stripped[:-1].strip()
                changed = True
            if stripped.startswith("const "):
                stripped = stripped[len("const ") :].strip()
                changed = True
            if stripped.endswith(" const"):
                stripped = stripped[: -len(" const")].strip()
                changed = True
        return stripped

    def collect_structured_buffer_length_requirements(self, root):
        """Collect buffers that need explicit length sidecar parameters."""
        functions = self.query_collect_functions(root)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_length_names = set()
        function_param_length_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_length_names[func_name])
                function_param_length_names[func_name].add(resource_name)
                return len(function_param_length_names[func_name]) != before
            before = len(global_length_names)
            global_length_names.add(resource_name)
            return len(global_length_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                buffer_expr = self.structured_buffer_dimensions_target(call)
                if buffer_expr is None:
                    continue
                mark_resource_name(func_name, self.get_expression_name(buffer_expr))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_length_names.get(
                        callee_name, set()
                    )
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_length_names[caller_name])
                            function_param_length_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_length_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_length_names)
                            global_length_names.add(arg_name)
                            changed = changed or len(global_length_names) != before

        return (
            global_length_names,
            {
                func_name: names
                for func_name, names in function_param_length_names.items()
                if names
            },
        )

    def structured_buffer_dimensions_target(self, call):
        """Return the structured-buffer expression queried by a dimensions call."""
        func_name = self.raw_function_call_name(call)
        raw_args = getattr(call, "arguments", getattr(call, "args", []))
        if func_name == "buffer_dimensions" and raw_args:
            return raw_args[0]

        function_expr = getattr(call, "function", getattr(call, "name", None))
        if isinstance(function_expr, MemberAccessNode):
            member = getattr(function_expr, "member", None)
            if member == "GetDimensions":
                return getattr(
                    function_expr,
                    "object_expr",
                    getattr(function_expr, "object", None),
                )
        return None

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def cuda_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to CUDA pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {cuda_element_type}*"
        return f"{cuda_element_type}*"

    def structured_buffer_length_name(self, name):
        """Return the sidecar length parameter/declaration name for a buffer."""
        return f"{name}_length"

    def structured_buffer_requires_length(self, name):
        """Return whether a global structured buffer needs a length sidecar."""
        return bool(name and name in self.structured_buffer_length_names)

    def structured_buffer_parameter_requires_length(
        self, func_name, name, type_name=None
    ):
        """Return whether a function parameter needs a length sidecar."""
        if not name:
            return False
        if name not in self.structured_buffer_length_function_params.get(
            func_name, set()
        ):
            return False
        return type_name is None or self.buffer_type_supports_length(type_name)

    def buffer_type_supports_length(self, type_name):
        """Return whether a resource type can use a CUDA length sidecar."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def structured_buffer_length_declaration(self, name, type_name):
        """Format a global/local sidecar length declaration when required."""
        if not self.structured_buffer_requires_length(name):
            return None
        if not self.buffer_type_supports_length(type_name):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def structured_buffer_length_parameter(self, func_name, name, type_name):
        """Format a sidecar length parameter when required."""
        if not self.structured_buffer_parameter_requires_length(
            func_name, name, type_name
        ):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def format_structured_buffer_length_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        length_name = self.structured_buffer_length_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"const uint* {length_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("const uint*", length_name, array_suffix)

    def structured_buffer_length_data_expression(self, buffer_expr):
        """Return the sidecar length pointer paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_length = self.structured_buffer_length_data_expression(array_expr)
            if base_length is None:
                return None
            return f"{base_length}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None

        length_parameter = self.current_structured_buffer_length_parameters.get(name)
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(name):
            return self.structured_buffer_length_name(name)
        return None

    def structured_buffer_length_expression(self, buffer_expr):
        """Return the scalar length expression for a structured-buffer resource."""
        length_data = self.structured_buffer_length_data_expression(buffer_expr)
        if length_data is None:
            return None
        return f"{length_data}[0]"

    def structured_buffer_counter_name(self, name):
        """Return the sidecar counter parameter/declaration name for a buffer."""
        return f"{name}_counter"

    def structured_buffer_counter_declaration(self, name, type_name):
        """Format a global/local sidecar counter declaration when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def structured_buffer_counter_parameter(self, name, type_name):
        """Format a sidecar counter parameter when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def format_structured_buffer_counter_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        counter_name = self.structured_buffer_counter_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"uint* {counter_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("uint*", counter_name, array_suffix)

    def structured_buffer_counter_expression(self, buffer_expr):
        """Return the sidecar counter expression paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_counter = self.structured_buffer_counter_expression(array_expr)
            if base_counter is None:
                return None
            return f"{base_counter}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None
        return self.structured_buffer_counter_name(name)

    def cuda_reject_unbindable_reference_arguments(self, func_name, params, raw_args):
        """Refuse calls whose out/inout arguments cannot bind to references."""
        overlapping = overlapping_output_arguments(params, raw_args)
        if overlapping:
            raise CudaLoweringRefused(
                [
                    f"overlapping out/inout arguments: {func_name} receives "
                    f"{', '.join(overlapping)} through more than one parameter that "
                    "reaches the caller; CUDA references would not copy out separately"
                ]
            )
        columns = matrix_column_output_arguments(self, params, raw_args)
        if columns:
            raise CudaLoweringRefused(
                [
                    f"matrix column passed to an out/inout parameter: {func_name} receives "
                    + ", ".join(
                        f"a column m[i] as argument {position}" for position in columns
                    )
                    + "; CUDA reads a column as a new vector, which a reference cannot "
                    "bind to; copy it into a local, pass that and assign it back"
                ]
            )
        swizzles = multi_component_swizzle_arguments(
            params, raw_args, self.member_swizzle_components
        )
        if swizzles:
            raise CudaLoweringRefused(
                [
                    f"swizzle passed to an out/inout parameter: {func_name} receives "
                    + ", ".join(
                        f"'.{swizzle}' as argument {position}"
                        for position, swizzle in swizzles
                    )
                    + "; CUDA builds a swizzle of several components as a new vector, "
                    "which a reference cannot bind to; copy it into a local, pass that "
                    "and assign it back"
                ]
            )

    def cuda_user_function_call_arguments(self, func_name, raw_args, args):
        """Expand user calls with CUDA sidecar resource arguments."""
        callee = self.query_functions_by_name.get(func_name)
        if callee is None:
            specialized = generic_function_specialization_named(self, func_name)
            if specialized is not None:
                self.cuda_reject_unbindable_reference_arguments(
                    func_name,
                    getattr(
                        specialized, "parameters", getattr(specialized, "params", [])
                    ),
                    raw_args,
                )
            return args + self.cuda_captured_function_call_args(func_name)

        params = getattr(callee, "parameters", getattr(callee, "params", []))
        self.cuda_reject_unbindable_reference_arguments(func_name, params, raw_args)
        query_params = self.query_metadata_function_params.get(func_name, set())
        expanded_args = []
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue

            param = params[index]
            param_name = getattr(param, "name", None)
            if param_name in query_params:
                metadata_arg = self.query_metadata_expression(raw_args[index])
                if metadata_arg is None:
                    metadata_arg = self.unavailable_query_metadata_argument(
                        param_name,
                        self.get_parameter_type(param),
                    )
                expanded_args.append(metadata_arg)

            param_type = self.get_parameter_type(param)
            if self.structured_buffer_parameter_requires_length(
                func_name, param_name, param_type
            ):
                length_arg = self.structured_buffer_length_data_expression(
                    raw_args[index]
                )
                if length_arg:
                    expanded_args.append(length_arg)

            if self.structured_buffer_requires_counter(param_type):
                counter_arg = self.structured_buffer_counter_expression(raw_args[index])
                if counter_arg:
                    expanded_args.append(counter_arg)
        expanded_args.extend(self.cuda_captured_function_call_args(func_name))
        return expanded_args

    def cuda_captured_function_call_args(self, func_name):
        return [
            self.visit(IdentifierNode(param.name))
            for param in self.cuda_function_capture_params.get(func_name, [])
        ]

    def unavailable_query_metadata_argument(self, param_name, param_type):
        """Return a deterministic zero metadata sidecar for untraceable resources."""
        self.resource_query_info_required = True
        type_name = self.query_type_name(param_type)
        resource_type = self.resource_base_type(type_name) or "unknown resource"
        fallback = "nullptr" if "[" in type_name else "CglResourceQueryInfo{}"
        param_label = f" argument {param_name}" if param_name else " argument"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: metadata unavailable for {resource_type}{param_label}",
            fallback,
        )

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a CUDA fallback expression for unsupported value-producing calls."""
        if type_name is None:
            return "0"
        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "bool":
                zero = "false"
            elif component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            else:
                zero = "0"
            return f"{vector_info['constructor']}({', '.join([zero] * len(vector_info['components']))})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def cuda_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to CUDA byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None
