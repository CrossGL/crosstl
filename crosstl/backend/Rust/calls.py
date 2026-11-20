"""Rust method, associated-function, constructor, resource and builtin calls as CrossGL calls."""

from .RustAst import (
    ArrayNode,
    ClosureNode,
    FunctionCallNode,
    MemberAccessNode,
    TupleNode,
)


class RustCallMixin:
    """Call members of ``RustToCrossGLConverter``."""

    def format_method_call_parts(
        self,
        method_name,
        obj,
        args,
        arg_nodes,
        receiver_type=None,
    ):
        impl_call = self.format_user_impl_method_call(method_name, obj, args)
        if impl_call is not None:
            return impl_call

        byte_addressable_call = self.format_rust_gpu_byte_addressable_buffer_call(
            method_name,
            obj,
            args,
            receiver_type,
        )
        if byte_addressable_call is not None:
            return byte_addressable_call

        resource_call = self.format_resource_method_call(
            method_name,
            obj,
            args,
            receiver_type,
        )
        if resource_call is not None:
            return resource_call

        if method_name == "len" and not args:
            return f"{obj}.length"

        primitive_bitcast_call = self.format_primitive_float_bitcast_method_call(
            method_name,
            obj,
            args,
            receiver_type,
        )
        if primitive_bitcast_call is not None:
            return primitive_bitcast_call

        derivative_method = self.RUST_GPU_DERIVATIVE_METHOD_MAP.get(method_name)
        if derivative_method is not None and not args:
            return f"{derivative_method}({obj})"

        if not args and method_name in self.SCALAR_ZERO_ARG_METHOD_MAP:
            return f"{self.SCALAR_ZERO_ARG_METHOD_MAP[method_name]}({obj})"

        if method_name == "recip" and not args:
            return f"(1.0 / {obj})"

        if len(args) == 1 and method_name in self.SCALAR_ONE_ARG_METHOD_MAP:
            return f"{self.SCALAR_ONE_ARG_METHOD_MAP[method_name]}({obj}, {args[0]})"

        if len(args) == 2 and method_name in self.SCALAR_TWO_ARG_METHOD_MAP:
            return (
                f"{self.SCALAR_TWO_ARG_METHOD_MAP[method_name]}"
                f"({obj}, {args[0]}, {args[1]})"
            )

        if method_name == "is_multiple_of" and len(args) == 1:
            return f"(({obj} % {args[0]}) == 0)"

        if method_name == "extend" and len(args) == 1:
            constructor = self.extended_vector_constructor(receiver_type)
            if constructor is not None:
                return f"{constructor}({obj}, {args[0]})"

        if method_name == "truncate" and not args:
            truncated = self.format_vector_truncate_call(obj, receiver_type)
            if truncated is not None:
                return truncated

        if method_name == "length" and not args:
            return f"length({obj})"

        if (
            method_name == "length_recip"
            and not args
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"(1.0 / length({obj}))"

        if (
            method_name == "length_squared"
            and not args
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"dot({obj}, {obj})"

        if method_name == "normalize" and not args:
            return f"normalize({obj})"

        mapped_receiver_type = self.map_type(receiver_type)
        if (
            method_name == "normalize_or_zero"
            and not args
            and mapped_receiver_type in self.VECTOR_COMPONENT_COUNTS
            and mapped_receiver_type.startswith("vec")
        ):
            zero, _ = self.vector_zero_one_literals(mapped_receiver_type)
            return (
                f"((length({obj}) > 0.0) ? normalize({obj}) : "
                f"{mapped_receiver_type}({zero}))"
            )

        if method_name in {"dot", "cross"} and len(args) == 1:
            return f"{method_name}({obj}, {args[0]})"

        if (
            method_name == "dot_into_vec"
            and len(args) == 1
            and mapped_receiver_type in self.VECTOR_COMPONENT_COUNTS
            and mapped_receiver_type.startswith("vec")
        ):
            return f"{mapped_receiver_type}(dot({obj}, {args[0]}))"

        if (
            method_name == "distance"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"distance({obj}, {args[0]})"

        if (
            method_name == "distance_squared"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            delta = f"({obj} - {args[0]})"
            return f"dot({delta}, {delta})"

        if (
            method_name == "reflect"
            and len(args) == 1
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"reflect({obj}, {args[0]})"

        if (
            method_name == "refract"
            and len(args) == 2
            and self.map_type(receiver_type) in self.VECTOR_COMPONENT_COUNTS
        ):
            return f"refract({obj}, {args[0]}, {args[1]})"

        if (
            method_name == "perp_dot"
            and len(args) == 1
            and self.map_type(receiver_type) == "vec2"
        ):
            return f"(({obj}.x * {args[0]}.y) - ({obj}.y * {args[0]}.x))"

        element_reduction = self.format_vector_element_reduction_call(
            method_name,
            obj,
            receiver_type,
        )
        if element_reduction is not None:
            return element_reduction

        if (
            method_name in {"map", "filter", "for_each", "any", "all"}
            and len(args) == 1
            and isinstance(arg_nodes[0], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                arg_nodes[0],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"{method_name}({obj}, {helper_name})"
            return f"{method_name}({obj}, {args[0]})"

        if (
            method_name == "fold"
            and len(args) == 2
            and isinstance(arg_nodes[1], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                arg_nodes[1],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"fold({obj}, {args[0]}, {helper_name})"
            return f"fold({obj}, {args[0]}, {args[1]})"

        if self.is_swizzle_member(method_name) and not args:
            return f"{obj}.{method_name}"

        return None

    def method_call_may_need_special_format(self, method_name, arg_nodes):
        if not isinstance(method_name, str):
            return False

        base_method_name, _ = self.split_function_type_arguments(method_name)
        arg_count = len(arg_nodes)

        if base_method_name in self.impl_receiver_method_names:
            return True
        if self.is_resource_method_name(base_method_name):
            return True
        if base_method_name in {
            "load",
            "load_unchecked",
            "store",
            "store_unchecked",
            "to_bits",
            "len",
            "recip",
            "is_multiple_of",
            "extend",
            "truncate",
            "length",
            "length_recip",
            "length_squared",
            "normalize",
            "normalize_or_zero",
            "dot",
            "cross",
            "dot_into_vec",
            "distance",
            "distance_squared",
            "reflect",
            "refract",
            "perp_dot",
            "unwrap_or",
        }:
            return True
        if base_method_name in self.RUST_GPU_DERIVATIVE_METHOD_MAP:
            return True
        if base_method_name in self.SCALAR_ZERO_ARG_METHOD_MAP:
            return True
        if base_method_name in self.SCALAR_ONE_ARG_METHOD_MAP:
            return True
        if base_method_name in self.SCALAR_TWO_ARG_METHOD_MAP:
            return True
        if base_method_name in {"max_element", "min_element"}:
            return True
        if base_method_name in {"map", "filter", "for_each", "any", "all"}:
            return arg_count == 1 and isinstance(arg_nodes[0], ClosureNode)
        if base_method_name == "fold":
            return arg_count == 2 and isinstance(arg_nodes[1], ClosureNode)
        if arg_count == 0 and self.is_swizzle_member(base_method_name):
            return True

        return False

    def format_rust_gpu_byte_addressable_buffer_call(
        self,
        method_name,
        obj,
        args,
        receiver_type=None,
    ):
        method_name, type_args = self.split_function_type_arguments(method_name)
        if method_name in {"load", "load_unchecked"} and len(args) == 1:
            if not self.is_byte_address_buffer_receiver(obj, receiver_type):
                return None
            intrinsic = self.byte_address_buffer_intrinsic("buffer_load", type_args)
            return f"{intrinsic}({obj}, {args[0]})"

        if method_name in {"store", "store_unchecked"} and len(args) == 2:
            if not self.is_byte_address_buffer_receiver(
                obj,
                receiver_type,
                writable=True,
            ):
                return None
            intrinsic = self.byte_address_buffer_intrinsic("buffer_store", type_args)
            return f"{intrinsic}({obj}, {args[0]}, {args[1]})"

        return None

    def is_byte_address_buffer_receiver(
        self,
        obj,
        receiver_type=None,
        writable=False,
    ):
        mapped_type = self.map_resource_method_receiver_type(obj, receiver_type)
        if mapped_type is None and receiver_type is not None:
            mapped_type = self.map_type(receiver_type)

        if writable:
            return mapped_type == "RWByteAddressBuffer"
        return mapped_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}

    def byte_address_buffer_intrinsic(self, base_intrinsic, type_args):
        if not type_args:
            return base_intrinsic

        mapped_type = self.map_type(type_args[0])
        component_count = self.VECTOR_COMPONENT_COUNTS.get(mapped_type)
        if component_count in {2, 3, 4}:
            return f"{base_intrinsic}{component_count}"
        return base_intrinsic

    def extended_vector_constructor(self, receiver_type):
        if not receiver_type:
            return None

        mapped_type = self.map_type(receiver_type)
        return self.VECTOR_EXTEND_CONSTRUCTOR_MAP.get(mapped_type)

    def truncated_vector_constructor(self, receiver_type):
        if not receiver_type:
            return None

        mapped_type = self.map_type(receiver_type)
        return self.VECTOR_TRUNCATE_CONSTRUCTOR_MAP.get(mapped_type)

    def format_vector_truncate_call(self, obj, receiver_type):
        constructor = self.truncated_vector_constructor(receiver_type)
        component_count = self.VECTOR_COMPONENT_COUNTS.get(constructor)
        if component_count is None:
            return None
        return f"{obj}.{'xyzw'[:component_count]}"

    def format_vector_element_reduction_call(self, method_name, obj, receiver_type):
        intrinsic = {"max_element": "max", "min_element": "min"}.get(method_name)
        if intrinsic is None:
            return None

        mapped_type = self.map_type(receiver_type)
        component_count = self.VECTOR_COMPONENT_COUNTS.get(mapped_type)
        if component_count is None:
            return None

        components = [f"{obj}.{component}" for component in "xyzw"[:component_count]]
        reduction = components[0]
        for component in components[1:]:
            reduction = f"{intrinsic}({reduction}, {component})"
        return reduction

    def vector_element_scalar_type(self, receiver_type):
        mapped_type = self.map_type(receiver_type)
        if mapped_type.startswith("bvec"):
            return "bool"
        if mapped_type.startswith("ivec"):
            return "int"
        if mapped_type.startswith("uvec"):
            return "uint"
        if mapped_type.startswith("vec"):
            return "float"
        return None

    def format_resource_method_call(self, method_name, obj, args, receiver_type=None):
        if not self.is_resource_method_name(method_name):
            return None

        mapped_resource_type = self.map_resource_method_receiver_type(
            obj,
            receiver_type,
        )
        if mapped_resource_type is None:
            return None

        sample_with_call = self.format_rust_gpu_sample_with_method_call(
            method_name,
            obj,
            args,
            mapped_resource_type,
        )
        if sample_with_call is not None:
            return sample_with_call

        gather_with_call = self.format_rust_gpu_gather_with_method_call(
            method_name,
            obj,
            args,
            mapped_resource_type,
        )
        if gather_with_call is not None:
            return gather_with_call

        fetch_with_call = self.format_rust_gpu_fetch_with_method_call(
            method_name,
            obj,
            args,
            mapped_resource_type,
        )
        if fetch_with_call is not None:
            return fetch_with_call

        mapped = self.function_map.get(method_name)
        if mapped is None:
            mapped = self.map_rust_gpu_resource_method(
                method_name,
                mapped_resource_type,
            )
        if mapped is None:
            return None

        return f"{mapped}({', '.join([obj] + args)})"

    def format_rust_gpu_sample_with_method_call(
        self,
        method_name,
        obj,
        args,
        mapped_resource_type,
    ):
        if not mapped_resource_type.startswith("sampler"):
            return None

        base_intrinsic = self.RUST_GPU_SAMPLE_WITH_METHOD_MAP.get(method_name)
        if base_intrinsic is None or not args:
            return None

        parsed_operand = self.parse_rust_gpu_sample_with_operand(args[-1])
        if parsed_operand is None:
            return None

        operand_name, operand_args = parsed_operand
        intrinsic = self.rust_gpu_sample_with_intrinsic(base_intrinsic, operand_name)
        if intrinsic is None:
            return None

        call_args = [obj] + args[:-1] + operand_args
        return f"{intrinsic}({', '.join(call_args)})"

    def format_rust_gpu_gather_with_method_call(
        self,
        method_name,
        obj,
        args,
        mapped_resource_type,
    ):
        if method_name != "gather_with" or len(args) != 4:
            return None
        if not mapped_resource_type.startswith("sampler"):
            return None

        parsed_operand = self.parse_rust_gpu_sample_with_operand(args[-1])
        if parsed_operand is None:
            return None

        operand_name, operand_args = parsed_operand
        if operand_name != "offset":
            return None

        call_args = [obj] + args[:2] + operand_args + [args[2]]
        return f"textureGatherOffset({', '.join(call_args)})"

    def format_rust_gpu_fetch_with_method_call(
        self,
        method_name,
        obj,
        args,
        mapped_resource_type,
    ):
        if method_name != "fetch_with" or len(args) != 2:
            return None
        if not mapped_resource_type.startswith("sampler"):
            return None

        parsed_operand = self.parse_rust_gpu_sample_index_operand(args[-1])
        if parsed_operand is None:
            return None

        return f"texelFetch({', '.join([obj, args[0], parsed_operand])})"

    def parse_rust_gpu_sample_with_operand(self, operand):
        parsed = self.parse_generated_call_expression(operand.strip())
        if parsed is None:
            return None

        function_name, args = parsed
        function_name, _ = self.split_function_type_arguments(function_name)
        operand_name = function_name.rsplit("::", 1)[-1]
        if operand_name not in {"bias", "lod", "grad", "offset"}:
            return None

        if operand_name in {"bias", "lod", "offset"} and len(args) != 1:
            return None
        if operand_name == "grad" and len(args) != 2:
            return None

        return operand_name, args

    def parse_rust_gpu_sample_index_operand(self, operand):
        parsed = self.parse_generated_call_expression(operand.strip())
        if parsed is None:
            return None

        function_name, args = parsed
        function_name, _ = self.split_function_type_arguments(function_name)
        operand_name = function_name.rsplit("::", 1)[-1]
        if operand_name != "sample_index" or len(args) != 1:
            return None

        return args[0]

    def rust_gpu_sample_with_intrinsic(self, base_intrinsic, operand_name):
        if operand_name == "bias":
            if base_intrinsic.startswith("textureCompare"):
                return None
            return base_intrinsic
        if operand_name == "lod":
            return f"{base_intrinsic}Lod"
        if operand_name == "grad":
            return f"{base_intrinsic}Grad"
        if operand_name == "offset":
            return f"{base_intrinsic}Offset"
        return None

    def is_resource_method_name(self, method_name):
        return isinstance(method_name, str) and (
            method_name.startswith(self.RESOURCE_METHOD_PREFIXES)
            or method_name in self.RESOURCE_METHOD_NAMES
        )

    def is_resource_method_receiver(self, obj, receiver_type=None):
        return self.map_resource_method_receiver_type(obj, receiver_type) is not None

    def map_resource_method_receiver_type(self, obj, receiver_type=None):
        if receiver_type is None:
            receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return None

        mapped_type = self.map_resource_receiver_type(receiver_type)
        if isinstance(mapped_type, str) and mapped_type.startswith(
            self.RESOURCE_TYPE_PREFIXES
        ):
            return mapped_type
        return None

    def map_rust_gpu_resource_method(self, method_name, mapped_resource_type):
        if method_name == "read":
            return "imageLoad"
        if method_name == "write":
            return "imageStore"
        if method_name == "fetch":
            if mapped_resource_type.startswith("sampler"):
                return "texelFetch"
            return "imageLoad"
        if method_name == "fetch_with_lod":
            if mapped_resource_type.startswith("sampler"):
                return "texelFetch"
            return None
        mapped_sample = self.RUST_GPU_SAMPLE_METHOD_MAP.get(method_name)
        if mapped_sample is not None and mapped_resource_type.startswith("sampler"):
            return mapped_sample
        if method_name in {"query_size", "query_size_lod"}:
            if mapped_resource_type.startswith("sampler"):
                return "textureSize"
            return "imageSize"
        if method_name == "query_levels":
            if mapped_resource_type.startswith("sampler"):
                return "textureQueryLevels"
            return None
        if method_name == "query_lod":
            if mapped_resource_type.startswith("sampler"):
                return "textureQueryLod"
            return None
        if method_name == "query_samples":
            if mapped_resource_type.startswith("sampler"):
                return "textureSamples"
            return "imageSamples"
        return None

    def map_resource_receiver_type(self, receiver_type, seen=None):
        mapped_type = self.map_type(receiver_type)
        if isinstance(mapped_type, str) and mapped_type.startswith(
            self.RESOURCE_TYPE_PREFIXES
        ):
            return mapped_type

        alias_target = self.resolve_type_alias_target(receiver_type)
        if alias_target is None:
            return mapped_type

        seen = seen or set()
        if alias_target in seen:
            return mapped_type

        seen.add(alias_target)
        return self.map_resource_receiver_type(alias_target, seen)

    def format_user_impl_method_call(self, method_name, obj, args):
        match = self.lookup_impl_method_receiver_match(obj, method_name)
        if match is None:
            return None

        call_args = [obj] + args
        return f"{match['call_prefix']}_{method_name}({', '.join(call_args)})"

    def format_associated_impl_function_call(self, function_name, args):
        match = self.lookup_impl_associated_function_match(function_name)
        if match is None:
            return None

        call = f"{match['call_prefix']}_{match['method_name']}({', '.join(args)})"
        return_type = self.infer_impl_associated_function_return_type(match, args)
        if return_type is not None:
            self.add_value_type(call, return_type)
        return call

    def format_primitive_float_bitcast_method_call(
        self,
        method_name,
        obj,
        args,
        receiver_type,
    ):
        if method_name != "to_bits" or args:
            return None
        bitcast_info = self.primitive_float_bitcast_info(receiver_type)
        if bitcast_info is None:
            return None
        return f"{bitcast_info['to_bits']}({obj})"

    def format_primitive_float_bitcast_associated_call(self, function_name, args):
        bitcast_info = self.primitive_float_bitcast_associated_info(
            function_name,
            args,
        )
        if bitcast_info is None:
            return None
        return f"{bitcast_info['from_bits']}({args[0]})"

    def format_unresolved_associated_call(self, function_name, args):
        parsed = self.parse_top_level_associated_function_path(function_name)
        if parsed is None:
            return None

        type_name, method_name, method_type_args = parsed
        if not method_name:
            return None
        if "<" not in type_name:
            return None

        prefix_parts = [self.impl_function_prefix(type_name), method_name]
        if method_type_args:
            prefix_parts.extend(
                self.impl_function_prefix(arg) for arg in method_type_args
            )
        function_identifier = self.crossgl_identifier("_".join(prefix_parts))
        return f"{function_identifier}({', '.join(args)})"

    def parse_top_level_associated_function_path(self, function_name):
        if not isinstance(function_name, str) or "::" not in function_name:
            return None

        function_name = self.normalize_associated_type_path(function_name)
        depth = 0
        split_index = -1
        pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
        openers = set(pairs)
        closers = {close: open_ for open_, close in pairs.items()}

        for index, char in enumerate(function_name):
            if char in openers:
                depth += 1
            elif char in closers:
                depth = max(0, depth - 1)
            elif (
                char == ":"
                and depth == 0
                and index + 1 < len(function_name)
                and function_name[index + 1] == ":"
            ):
                split_index = index

        if split_index < 0:
            return None

        type_name = function_name[:split_index]
        method_name = function_name[split_index + 2 :]
        method_name, method_type_args = self.split_function_type_arguments(method_name)
        if not type_name or not method_name:
            return None
        return type_name, method_name, method_type_args

    def infer_primitive_float_bitcast_associated_return_type(
        self,
        function_name,
        args=None,
    ):
        bitcast_info = self.primitive_float_bitcast_associated_info(
            function_name,
            args,
        )
        if bitcast_info is None:
            return None
        return bitcast_info["float_type"]

    def primitive_float_bitcast_associated_info(
        self,
        function_name,
        args=None,
    ):
        if args is not None and len(args) != 1:
            return None

        function_name = self.resolve_imported_module_path(function_name)
        parsed = self.parse_associated_function_path(function_name)
        if parsed is None:
            return None

        type_name, method_name, method_type_args = parsed
        if method_name != "from_bits" or method_type_args:
            return None
        return self.primitive_float_bitcast_info(type_name)

    def primitive_float_bitcast_info(self, receiver_type):
        if not isinstance(receiver_type, str):
            return None

        receiver_type = self.normalize_receiver_type(receiver_type)
        mapped_type = self.map_type(receiver_type)
        if receiver_type in {"f32", "float"} or mapped_type == "float":
            return {
                "to_bits": "floatBitsToUint",
                "from_bits": "uintBitsToFloat",
                "bits_type": "uint",
                "float_type": "float",
            }
        if receiver_type in {"f64", "double"} or mapped_type == "double":
            return {
                "to_bits": "doubleBitsToLong",
                "from_bits": "longBitsToDouble",
                "bits_type": "uint64_t",
                "float_type": "double",
            }
        return None

    def lookup_impl_method_receiver_match(self, obj, method_name):
        base_method_name, _ = self.split_function_type_arguments(method_name)
        if base_method_name not in self.impl_receiver_method_names:
            return None

        receiver_type = self.lookup_value_type(obj)
        if receiver_type is None:
            return None

        for signature in self.impl_method_signatures.values():
            method = signature["methods"].get(method_name)
            if method is None:
                continue
            if not self.method_has_self_parameter(method):
                continue

            substitutions = self.match_impl_receiver_type(
                signature["struct_name"],
                receiver_type,
                signature["generic_names"],
            )
            if substitutions is None:
                continue

            return {
                "receiver_type": receiver_type,
                "call_prefix": signature["call_prefix"],
                "generic_names": signature["generic_names"],
                "method": method,
                "substitutions": substitutions,
            }

        return None

    def lookup_impl_associated_function_match(self, function_name):
        parsed = self.parse_associated_function_path(function_name)
        if parsed is None:
            return None

        type_name, method_name, method_type_args = parsed
        if method_name not in self.impl_associated_function_names:
            return None

        for signature in self.impl_method_signatures.values():
            method = signature["methods"].get(method_name)
            if method is None:
                continue
            if self.method_has_self_parameter(method):
                continue

            receiver_type, substitutions = self.match_associated_impl_type(
                signature["struct_name"],
                type_name,
                signature["generic_names"],
            )
            if receiver_type is None:
                continue

            return {
                "receiver_type": receiver_type,
                "call_prefix": signature["call_prefix"],
                "method_name": method_name,
                "method_type_args": method_type_args,
                "generic_names": signature["generic_names"],
                "method": method,
                "substitutions": substitutions,
            }

        return None

    def parse_associated_function_path(self, function_name):
        if not isinstance(function_name, str) or "::" not in function_name:
            return None

        function_name = self.normalize_associated_type_path(function_name)
        type_name, method_name = function_name.rsplit("::", 1)
        method_name, method_type_args = self.split_function_type_arguments(method_name)
        if not type_name or not method_name:
            return None
        return type_name, method_name, method_type_args

    def normalize_associated_type_path(self, type_name):
        if not isinstance(type_name, str):
            return type_name
        return type_name.replace("::<", "<")

    def match_associated_impl_type(self, pattern_type, call_type, generic_names):
        call_type = self.resolve_imported_module_path(call_type)
        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            pattern_generic = self.parse_generic_type(pattern_candidate)
            for call_candidate in self.generic_type_match_candidates(call_type):
                if self.impl_type_names_match(pattern_candidate, call_candidate):
                    return call_candidate, {}

                call_generic = self.parse_generic_type(call_candidate)
                if pattern_generic is None:
                    if call_generic is None:
                        continue
                    call_base, _ = call_generic
                    if self.generic_type_bases_match(pattern_candidate, call_base):
                        return pattern_candidate, {}
                    continue

                pattern_base, pattern_args = pattern_generic
                if call_generic is None:
                    if self.generic_type_bases_match(pattern_base, call_candidate):
                        return pattern_candidate, {}
                    continue

                call_base, call_args = call_generic
                if not self.generic_type_bases_match(pattern_base, call_base):
                    continue
                if len(pattern_args) != len(call_args):
                    continue

                substitutions = {}
                conflicts = set()
                for pattern_arg, call_arg in zip(pattern_args, call_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        call_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                if conflicts:
                    continue
                return call_candidate, substitutions

        return None, {}

    def match_impl_receiver_type(self, pattern_type, receiver_type, generic_names):
        for pattern_candidate in self.generic_type_match_candidates(pattern_type):
            for receiver_candidate in self.generic_type_match_candidates(receiver_type):
                if self.impl_type_names_match(pattern_candidate, receiver_candidate):
                    return {}

                pattern_generic = self.parse_generic_type(pattern_candidate)
                receiver_generic = self.parse_generic_type(receiver_candidate)
                if pattern_generic is None or receiver_generic is None:
                    continue

                pattern_base, pattern_args = pattern_generic
                receiver_base, receiver_args = receiver_generic
                if not self.generic_type_bases_match(pattern_base, receiver_base):
                    continue
                if len(pattern_args) != len(receiver_args):
                    continue

                substitutions = {}
                conflicts = set()
                for pattern_arg, receiver_arg in zip(pattern_args, receiver_args):
                    self.match_generic_type_parameters(
                        pattern_arg,
                        receiver_arg,
                        generic_names,
                        substitutions,
                        conflicts,
                    )
                if conflicts:
                    continue
                return substitutions

        return None

    def impl_type_names_match(self, left, right):
        return any(
            left_name == right_name
            for left_name in self.type_lookup_names(left)
            for right_name in self.type_lookup_names(right)
        )

    def format_path_constructor_call_parts(self, function_name, args, arg_nodes=None):
        if "::" not in function_name:
            return None

        type_name, constructor_name = function_name.rsplit("::", 1)
        if constructor_name not in {"new", "splat", "from"}:
            return None

        mapped_type = self.map_type(type_name)
        if mapped_type == type_name:
            return None

        if constructor_name == "from":
            return self.format_vector_from_constructor_call(
                mapped_type,
                args,
                arg_nodes,
            )

        if constructor_name == "splat":
            if len(args) != 1 or mapped_type not in self.VECTOR_SPLAT_CONSTRUCTORS:
                return None
            return f"{mapped_type}({args[0]})"

        return f"{mapped_type}({', '.join(args)})"

    def format_vector_from_constructor_call(self, mapped_type, args, arg_nodes=None):
        if (
            len(args) != 1
            or mapped_type not in self.VECTOR_COMPONENT_COUNTS
            or not arg_nodes
            or len(arg_nodes) != 1
        ):
            return None

        arg_node = arg_nodes[0]
        if isinstance(arg_node, TupleNode):
            values = self.split_wrapped_comma_values(args[0], "(", ")")
        elif isinstance(arg_node, ArrayNode) and arg_node.size is None:
            values = self.split_wrapped_comma_values(args[0], "{", "}")
        else:
            return None

        if values is None or len(values) <= 1:
            return None
        return f"{mapped_type}({', '.join(values)})"

    def split_wrapped_comma_values(self, value, opening, closing):
        if not isinstance(value, str):
            return None

        value = value.strip()
        if not (value.startswith(opening) and value.endswith(closing)):
            return None

        inner = value[1:-1].strip()
        if not inner:
            return []

        parts = []
        current = []
        depth = 0
        quote = None
        escaped = False

        for char in inner:
            current.append(char)

            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue

            if char in {'"', "'"}:
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                part = "".join(current[:-1]).strip()
                if part:
                    parts.append(part)
                current = []

        final_part = "".join(current).strip()
        if final_part:
            parts.append(final_part)
        return parts

    def format_vector_associated_constant(self, value):
        if not isinstance(value, str) or "::" not in value:
            return None

        type_name, constant_name = value.rsplit("::", 1)
        mapped_type = self.map_type(type_name)
        components = self.vector_associated_constant_components(
            mapped_type,
            constant_name,
        )
        if components is None:
            return None

        return f"{mapped_type}({', '.join(components)})"

    def vector_associated_constant_components(self, mapped_type, constant_name):
        count = self.VECTOR_COMPONENT_COUNTS.get(mapped_type)
        if count is None:
            return None

        zero, one = self.vector_zero_one_literals(mapped_type)
        if constant_name == "ZERO":
            return [zero] * count
        if constant_name == "ONE":
            return [one] * count
        if constant_name == "NEG_ONE":
            negative_one = self.vector_negative_one_literal(mapped_type)
            if negative_one is None:
                return None
            return [negative_one] * count

        axis_constant_name = constant_name
        axis_value = one
        if constant_name.startswith("NEG_"):
            negative_one = self.vector_negative_one_literal(mapped_type)
            if negative_one is None:
                return None
            axis_constant_name = constant_name[4:]
            axis_value = negative_one

        axis_index = self.VECTOR_AXIS_CONSTANT_INDICES.get(axis_constant_name)
        if axis_index is None or axis_index >= count:
            return None

        components = [zero] * count
        components[axis_index] = axis_value
        return components

    def format_scalar_associated_constant(self, value):
        if not isinstance(value, str):
            return None
        return self.SCALAR_ASSOCIATED_CONSTANTS.get(value)

    def format_impl_associated_const(self, value):
        if not isinstance(value, str) or "::" not in value:
            return None

        value = self.normalize_associated_type_path(value)
        if "::" not in value:
            return None

        direct = self.associated_consts.get(value)
        if direct is not None:
            return direct

        type_name, const_name = value.rsplit("::", 1)
        for type_candidate in self.type_lookup_names(type_name):
            resolved = self.associated_consts.get(f"{type_candidate}::{const_name}")
            if resolved is not None:
                return resolved
        return None

    def vector_zero_one_literals(self, mapped_type):
        if mapped_type.startswith("bvec"):
            return "false", "true"
        if mapped_type.startswith("vec"):
            return "0.0", "1.0"
        return "0", "1"

    def vector_negative_one_literal(self, mapped_type):
        if mapped_type.startswith("vec"):
            return "-1.0"
        if mapped_type.startswith("ivec"):
            return "-1"
        return None

    def format_method_call(self, expr):
        if not isinstance(expr.name, MemberAccessNode):
            return None

        method_name = expr.name.member
        if not self.method_call_may_need_special_format(method_name, expr.args):
            return None

        if method_name == "unwrap_or" and len(expr.args) == 1:
            return self.format_option_unwrap_or_call(expr.name.object, expr.args[0])

        obj = self.generate_expression(expr.name.object)
        receiver_type = self.infer_value_type(expr.name.object)

        if (
            method_name in {"map", "filter", "for_each", "any", "all"}
            and len(expr.args) == 1
            and isinstance(expr.args[0], ClosureNode)
        ):
            helper_name = self.try_generate_closure_helper(
                expr.args[0],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"{method_name}({obj}, {helper_name})"
            return f"{method_name}({obj}, {self.generate_expression(expr.args[0])})"

        if (
            method_name == "fold"
            and len(expr.args) == 2
            and isinstance(expr.args[1], ClosureNode)
        ):
            initial_value = self.generate_expression(expr.args[0])
            helper_name = self.try_generate_closure_helper(
                expr.args[1],
                context_name=method_name,
            )
            if helper_name is not None:
                return f"fold({obj}, {initial_value}, {helper_name})"
            return (
                f"fold({obj}, {initial_value}, "
                f"{self.generate_expression(expr.args[1])})"
            )

        args = [self.generate_expression(arg) for arg in expr.args]
        return self.format_method_call_parts(
            method_name,
            obj,
            args,
            expr.args,
            receiver_type,
        )

    def format_option_unwrap_or_call(self, receiver, fallback):
        if not isinstance(receiver, FunctionCallNode):
            return None
        if not isinstance(receiver.name, str):
            return None

        helper_name = self.ensure_option_unwrap_or_helper(receiver.name)
        if helper_name is None:
            return None

        args = [self.generate_expression(arg) for arg in receiver.args]
        args.append(self.generate_expression(fallback))
        return f"{helper_name}({', '.join(args)})"

    def format_unary_expression_intrinsic_call(self, function_name, args):
        if len(args) != 1 or not isinstance(function_name, str):
            return None

        resolved_name = self.resolve_imported_module_path(function_name)
        if self.is_user_defined_function(resolved_name):
            return None

        current_module_function = self.current_module_user_function_name(resolved_name)
        if current_module_function is not None:
            return None

        if resolved_name == "recip":
            return f"(1.0 / {args[0]})"

        if "::" not in resolved_name:
            return None

        qualifier, intrinsic_name = resolved_name.rsplit("::", 1)
        if intrinsic_name != "recip":
            return None

        if self.is_module_qualified_function_path(qualifier.split("::")):
            return f"(1.0 / {args[0]})"

        return None

    def format_path_constructor_call(self, function_name, args):
        arg_values = [self.generate_expression(arg) for arg in args]
        return self.format_path_constructor_call_parts(
            function_name,
            arg_values,
            args,
        )

    def map_function(self, rust_func):
        stripped_func, type_args = self.split_function_type_arguments(rust_func)
        if type_args and self.lookup_user_function_signature(stripped_func) is not None:
            rust_func = stripped_func
        else:
            rust_func = self.format_function_type_arguments(stripped_func, type_args)

        local_callable = self.lookup_local_callable_name(stripped_func)
        if local_callable is not None:
            return local_callable

        rust_func = self.resolve_imported_module_path(rust_func)
        if self.is_user_defined_function(rust_func):
            return rust_func

        current_module_function = self.current_module_user_function_name(rust_func)
        if current_module_function is not None:
            return current_module_function

        mapped = self.function_map.get(rust_func)
        if mapped is None and type_args:
            mapped = self.function_map.get(stripped_func)
        if mapped is not None:
            return mapped

        if "::" not in rust_func:
            return rust_func

        segments = rust_func.split("::")
        mapped = self.RUST_GPU_DERIVATIVE_METHOD_MAP.get(segments[-1])
        if mapped is not None and self.is_rust_gpu_associated_intrinsic_path(
            segments[:-1]
        ):
            return mapped

        mapped = self.function_map.get(segments[-1])
        if mapped is None:
            return rust_func

        if self.is_module_qualified_function_path(segments[:-1]):
            return mapped
        return rust_func

    def is_rust_gpu_associated_intrinsic_path(self, qualifier_segments):
        if not qualifier_segments:
            return False

        type_segment = qualifier_segments[-1].split("<", 1)[0]
        if type_segment not in self.RUST_GPU_ASSOCIATED_INTRINSIC_TYPES:
            return False

        module_segments = qualifier_segments[:-1]
        return not module_segments or self.is_module_qualified_function_path(
            module_segments
        )

    def current_module_user_function_name(self, rust_func):
        if not isinstance(rust_func, str) or "::" not in rust_func:
            return None

        segments = rust_func.split("::")
        if len(segments) != 2 or segments[0] not in {"self", "crate"}:
            return None

        function_name = segments[-1]
        if self.is_user_defined_function(function_name):
            return function_name
        return None

    def is_module_qualified_function_path(self, qualifier_segments):
        if not qualifier_segments:
            return False

        namespace_roots = {"crate", "self", "super", "std"}
        if qualifier_segments[0] in namespace_roots:
            qualifier_segments = qualifier_segments[1:]

        if not qualifier_segments:
            return True

        return all(
            self.is_module_path_segment(segment) for segment in qualifier_segments
        )

    def is_module_path_segment(self, segment):
        if not segment or "<" in segment or ">" in segment:
            return False
        return not segment[0].isupper()
