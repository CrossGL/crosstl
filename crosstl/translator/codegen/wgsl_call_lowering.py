"""Call lowering in the WGSL generator: foreign-call refusal, packed 8-bit and dot4add calls, bit functions, workgroup uniform loads, bit casts, half conversions, derivatives, structured buffer helpers and mod."""

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
)
from ..refusals import WGSLRefusal
from .wgsl_foreign_calls import (
    WGSLForeignBuiltinCallError,
    is_foreign_wgsl_call,
)
from .wgsl_non_finite_constants import (
    constant_integer_value,
    is_non_finite_f32_bits,
    wgsl_runtime_f32_from_bits,
)


class WGSLCallLoweringMixin:
    """Call lowering members of ``WGSLCodeGen``."""

    def refuse_foreign_wgsl_call(self, function_name, written_name):
        """Refuse a call that would be written with a built-in function name of another language."""
        if self.is_user_function_name(function_name):
            return
        if is_foreign_wgsl_call(function_name, written_name):
            raise WGSLForeignBuiltinCallError(function_name)

    # HLSL 8-bit pack intrinsics and their WGSL builtins (WGSL sections
    # 17.9.3 and 17.9.8): the WGSL argument type each takes, the builtin, and
    # its result type. pack_clamp_u8 takes signed components, while
    # pack4xU8Clamp takes unsigned ones, so negative components are clamped
    # to zero first.
    PACKED_8BIT_FUNCTIONS = {
        "pack_s8": ("vec4<i32>", "pack4xI8", "u32"),
        "pack_u8": ("vec4<u32>", "pack4xU8", "u32"),
        "pack_clamp_s8": ("vec4<i32>", "pack4xI8Clamp", "u32"),
        "pack_clamp_u8": ("vec4<u32>", "pack4xU8Clamp", "u32"),
        "unpack_s8s32": ("u32", "unpack4xI8", "vec4<i32>"),
        "unpack_u8u32": ("u32", "unpack4xU8", "vec4<u32>"),
    }

    def generate_packed_8bit_call(self, node, function_name, mapping):
        """HLSL ``pack_*8``/``unpack_*8*32`` as the WGSL 4x8-bit pack builtins."""
        argument_type, mapped_name, _result_type = mapping
        if len(node.arguments) != 1:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly 1 argument; "
                f"got {len(node.arguments)}"
            )
        argument = node.arguments[0]
        argument_expr = self.generate_expression(argument)
        if argument_type == "u32":
            # HLSL converts an int to the packed type bit for bit, as u32() does.
            if self.integer_scalar_type(self.expression_type(argument)) == "i32":
                argument_expr = f"u32({argument_expr})"
            return f"{mapped_name}({argument_expr})"
        component = self.vector_component_type(self.expression_type(argument))
        if component is not None:
            component = self.map_type_name(component)
        if mapped_name == "pack4xU8Clamp" and component != "u32":
            if component != "i32":
                argument_expr = f"vec4<i32>({argument_expr})"
            argument_expr = f"vec4<u32>(max({argument_expr}, vec4<i32>(0)))"
        elif component != argument_type[len("vec4<") : -1]:
            argument_expr = f"{argument_type}({argument_expr})"
        return f"{mapped_name}({argument_expr})"

    # GLSL bit built-ins whose WGSL forms differ in more than the name:
    # bitCount, findLSB and findMSB return int in GLSL but the argument type in
    # WGSL, and extractBits and insertBits take a u32 offset and count.
    BIT_FUNCTION_NAME_MAP = {
        "bitCount": "countOneBits",
        "findLSB": "firstTrailingBit",
        "findMSB": "firstLeadingBit",
        "bitfieldExtract": "extractBits",
        "bitfieldInsert": "insertBits",
    }

    def generate_bit_function_call(self, node, function_name, mapped_name):
        """GLSL bit built-ins as WGSL ones, with GLSL's argument and result types."""
        arity = {"extractBits": 3, "insertBits": 4}.get(mapped_name, 1)
        if len(node.arguments) != arity:
            noun = "argument" if arity == 1 else "arguments"
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly {arity} "
                f"{noun}; got {len(node.arguments)}"
            )
        args = [self.generate_expression(argument) for argument in node.arguments]
        if arity > 1:
            # The offset and count are the last two arguments.
            for index in (arity - 2, arity - 1):
                offset_type = self.expression_type(node.arguments[index])
                if self.integer_scalar_type(offset_type) != "u32":
                    args[index] = f"u32({args[index]})"
            return f"{mapped_name}({', '.join(args)})"
        call = f"{mapped_name}({args[0]})"
        argument_type = self.expression_type(node.arguments[0])
        scalar = self.integer_scalar_type(argument_type)
        if scalar is not None:
            return f"i32({call})" if scalar == "u32" else call
        vector = self.vector_shape(argument_type)
        if vector is not None and vector[0] in {"i32", "u32"}:
            return f"vec{vector[1]}<i32>({call})" if vector[0] == "u32" else call
        # A u32 result left unconverted is still valid WGSL, but findMSB(x) < 0
        # would never hold and float(findMSB(x)) would give 4294967295.0.
        raise WGSLRefusal(
            f"WGSL target writes {function_name}() as {mapped_name}(), which returns "
            "its argument's type where CrossGL returns int, and cannot tell the type "
            f"of {args[0]}; assign the argument to a local variable first"
        )

    def generate_workgroup_uniform_load_call(self, node):
        """Emit ``workgroupUniformLoad(&x)`` for a workgroup variable ``x``."""
        argument = node.arguments[0]
        arg_expr = self.generate_expression(argument)
        if not arg_expr.startswith("&"):
            arg_expr = f"&{arg_expr}"
        return f"workgroupUniformLoad({arg_expr})"

    # HLSL dot4add_i8packed(a, b, acc) and dot4add_u8packed(a, b, acc) add the
    # packed 4x8-bit dot product of a and b to acc. WGSL has the dot product
    # alone, as dot4I8Packed and dot4U8Packed.
    PACKED_DOT4ADD_FUNCTIONS = {
        "dot4add_i8packed": "dot4I8Packed",
        "dot4add_u8packed": "dot4U8Packed",
    }

    def generate_packed_dot4add_call(self, node, function_name, mapped_name):
        """HLSL ``dot4add_*8packed(a, b, acc)`` as WGSL ``(acc + dot4*8Packed(a, b))``."""
        if len(node.arguments) != 3:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly 3 arguments; "
                f"got {len(node.arguments)}"
            )
        left, right, accumulator = (
            self.generate_expression(argument) for argument in node.arguments
        )
        return f"({accumulator} + {mapped_name}({left}, {right}))"

    def generate_declared_function_call(self, node, function):
        """A call of ``function``, one of the functions the shader declares."""
        dispatch_call = self.generate_sampled_texture_array_dispatch_call(
            node, function
        )
        if dispatch_call is not None:
            return dispatch_call
        function_name = self.expression_name(node.function)
        args = self.generate_call_arguments(
            function_name, node.arguments, function=function
        )
        return f"{self.function_declaration_identifier_name(function)}({args})"

    def is_user_function_name(self, name):
        return bool(getattr(self, "_function_overloads_by_name", {}).get(name))

    def generate_bit_cast_call(self, node, function_name, scalar):
        """HLSL ``asuint``/``asint``/``asfloat`` and GLSL ``floatBitsToUint`` and the like as WGSL ``bitcast``."""
        if len(node.arguments) != 1:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly 1 argument; got "
                f"{len(node.arguments)}"
            )
        argument = node.arguments[0]
        lanes = self.vector_component_count(self.expression_type(argument))
        target = scalar if lanes is None else f"vec{lanes}<{scalar}>"
        rendered = self.generate_expression(argument)
        if target == "f32" and is_non_finite_f32_bits(
            constant_integer_value(self, argument)
        ):
            return wgsl_runtime_f32_from_bits(self, rendered)
        return f"bitcast<{target}>({rendered})"

    def generate_half_conversion_helpers(self):
        """The helper functions used for half float vector packing/unpacking."""
        if not getattr(self, "_half_conversion_helpers", None):
            return None
        return "\n\n".join(self._half_conversion_helpers.values())

    def half_conversion_float_operand(self, argument):
        """``argument`` of ``f32tof16`` as an ``f32`` value or vector, and its width (``None`` for a scalar).

        HLSL converts an integer operand to ``float`` before it packs it, as for any ``float`` parameter; WGSL has no
        implicit conversion, so ``vec2<f32>(k, 0.0)`` with an ``i32`` ``k`` does not compile.
        """
        arg_type = self.expression_type(argument)
        vector = self.vector_shape(arg_type)
        if vector is None:
            return self.generate_expression_for_target(argument, "float"), None
        elem_type, width = vector
        rendered = self.generate_expression(argument)
        if elem_type != "f32":
            rendered = f"vec{width}<f32>({rendered})"
        return rendered, width

    def generate_f32tof16_call(self, node):
        """HLSL ``f32tof16`` as WGSL ``pack2x16float``."""
        if len(node.arguments) != 1:
            raise WGSLRefusal(
                f"WGSL target supports f32tof16() with exactly 1 argument; "
                f"got {len(node.arguments)}"
            )
        arg_expr, width = self.half_conversion_float_operand(node.arguments[0])
        if width is None:
            return f"pack2x16float(vec2<f32>({arg_expr}, 0.0))"
        components = ["x", "y", "z", "w"][:width]
        helper_name = f"crosstl_pack2x16float_vec{width}"
        if helper_name not in self._half_conversion_helpers:
            elements = [f"pack2x16float(vec2<f32>(v.{c}, 0.0))" for c in components]
            self._half_conversion_helpers[helper_name] = (
                f"fn {helper_name}(v: vec{width}<f32>) -> vec{width}<u32> {{\n"
                f"    return vec{width}<u32>({', '.join(elements)});\n"
                f"}}"
            )
        return f"{helper_name}({arg_expr})"

    def generate_f16tof32_call(self, node):
        """HLSL ``f16tof32`` as WGSL ``unpack2x16float``."""
        if len(node.arguments) != 1:
            raise WGSLRefusal(
                f"WGSL target supports f16tof32() with exactly 1 argument; "
                f"got {len(node.arguments)}"
            )
        argument = node.arguments[0]
        if (
            isinstance(argument, FunctionCallNode)
            and self.expression_name(argument.function) == "f32tof16"
            and not self.is_user_function_name("f32tof16")
            and len(argument.arguments) == 1
        ):
            # A round trip through binary16 is WGSL's quantizeToF16 built-in, which naga lowers to OpQuantizeToF16.
            # unpack2x16float(pack2x16float(v)) means the same, but on lavapipe the SPIR-V naga writes for that pair
            # returned v unrounded.
            inner_expr, _width = self.half_conversion_float_operand(
                argument.arguments[0]
            )
            return f"quantizeToF16({inner_expr})"

        arg_type = self.expression_type(argument)
        vector = self.vector_shape(arg_type)
        if vector is None:
            arg_expr = self.generate_expression(argument)
            if self.integer_scalar_type(arg_type) == "i32":
                arg_expr = f"u32({arg_expr})"
            return f"unpack2x16float({arg_expr}).x"
        elem_type, width = vector
        components = ["x", "y", "z", "w"][:width]
        arg_type_name = "i32" if elem_type == "i32" else "u32"
        prefix = "i" if elem_type == "i32" else ""
        helper_name = f"crosstl_unpack2x16float_{prefix}vec{width}"
        if helper_name not in self._half_conversion_helpers:
            elements = []
            for c in components:
                comp = f"v.{c}"
                if elem_type == "i32":
                    comp = f"u32({comp})"
                elements.append(f"unpack2x16float({comp}).x")
            self._half_conversion_helpers[helper_name] = (
                f"fn {helper_name}(v: vec{width}<{arg_type_name}>) -> vec{width}<f32> {{\n"
                f"    return vec{width}<f32>({', '.join(elements)});\n"
                f"}}"
            )
        arg_expr = self.generate_expression(argument)
        return f"{helper_name}({arg_expr})"

    def generate_derivative_call(self, node, mapped_name, function_name):
        if len(node.arguments) != 1:
            raise WGSLRefusal(
                "WGSL target supports derivative intrinsic "
                f"{function_name}() with exactly 1 argument; got "
                f"{len(node.arguments)}"
            )
        return f"{mapped_name}({self.generate_expression(node.arguments[0])})"

    def generate_structured_buffer_free_helper_call(self, node, helper_name):
        if helper_name == "buffer_load":
            if len(node.arguments) != 2:
                raise WGSLRefusal(
                    "WGSL target supports buffer_load() with exactly 2 arguments; "
                    f"got {len(node.arguments)}"
                )
            resource, index = node.arguments
            self.require_structured_buffer_resource(resource, "buffer_load")
            return self.structured_buffer_index_expression(resource, index)

        if len(node.arguments) != 3:
            raise WGSLRefusal(
                "WGSL target supports buffer_store() with exactly 3 arguments; "
                f"got {len(node.arguments)}"
            )
        resource, index, value = node.arguments
        self.require_writable_structured_buffer_resource(resource, "buffer_store")
        return self.structured_buffer_store_expression(resource, index, value)

    def generate_structured_buffer_member_helper_call(self, node, helper_name):
        receiver = node.function.object_expr
        if helper_name == "load":
            if len(node.arguments) != 1:
                raise WGSLRefusal(
                    "WGSL target supports StructuredBuffer.Load() with exactly "
                    f"1 argument; got {len(node.arguments)}"
                )
            self.require_structured_buffer_resource(receiver, "Load")
            return self.structured_buffer_index_expression(receiver, node.arguments[0])

        if len(node.arguments) != 2:
            raise WGSLRefusal(
                "WGSL target supports RWStructuredBuffer.Store() with exactly "
                f"2 arguments; got {len(node.arguments)}"
            )
        self.require_writable_structured_buffer_resource(receiver, "Store")
        return self.structured_buffer_store_expression(receiver, *node.arguments)

    def structured_buffer_store_expression(self, resource, index, value):
        """``buffer[index] = value``, the value converted to the element type as an assignment converts it.

        WGSL has no implicit conversions (section 15.7), so an ``int`` stored into a ``uint`` buffer needs ``u32()``.
        """
        element_type = self.expression_type(ArrayAccessNode(resource, index))
        return (
            f"{self.structured_buffer_index_expression(resource, index)} = "
            f"{self.generate_expression_for_target(value, element_type)}"
        )

    def structured_buffer_load_access(self, node, function_name):
        """``buffer[index]`` for ``buffer_load(buffer, index)`` or ``buffer.Load(index)``, as generation emits it.

        Typing the load as that subscript lets callers such as generic deduction see the element type.
        """
        if isinstance(node.function, MemberAccessNode):
            if (
                self.semantic_key(node.function.member) != "load"
                or len(node.arguments) != 1
            ):
                return None
            resource, index = node.function.object_expr, node.arguments[0]
        elif (
            self.semantic_key(function_name or "") == "buffer_load"
            and len(node.arguments) == 2
            and not self.is_user_function_name(function_name)
        ):
            resource, index = node.arguments
        else:
            return None
        resource_type = self.expression_type(resource)
        if self.structured_buffer_element_type(resource_type) is None and not (
            isinstance(resource, IdentifierNode)
            and self.is_pointer_identifier(resource.name)
        ):
            return None
        return ArrayAccessNode(resource, index)

    def structured_buffer_index_expression(self, resource, index):
        if isinstance(resource, IdentifierNode) and self.is_pointer_identifier(
            resource.name
        ):
            resource_name = self.identifier_name(resource.name)
            return f"(*{resource_name})[{self.generate_expression(index)}]"
        return (
            f"{self.generate_expression(resource)}[{self.generate_expression(index)}]"
        )

    def require_structured_buffer_resource(self, resource, helper_name):
        resource_type = self.expression_type(resource)
        if self.structured_buffer_element_type(resource_type) is None and not (
            isinstance(resource, IdentifierNode)
            and self.is_pointer_identifier(resource.name)
        ):
            raise WGSLRefusal(
                "WGSL target requires "
                f"{helper_name}() to use a StructuredBuffer or RWStructuredBuffer "
                "resource"
            )

    def require_writable_structured_buffer_resource(self, resource, helper_name):
        self.require_structured_buffer_resource(resource, helper_name)
        resource_type = self.expression_type(resource)
        access = self.structured_buffer_access(resource_type)
        if access == "read":
            raise WGSLRefusal(
                "WGSL target cannot store through read-only StructuredBuffer "
                f"resource in {helper_name}()"
            )

    def generate_mod_call(self, node):
        if len(node.arguments) != 2:
            raise WGSLRefusal(
                "WGSL target supports mod() calls with exactly 2 arguments; got "
                f"{len(node.arguments)}"
            )
        left, right = self.convert_integer_call_arguments(
            "mod",
            node.arguments,
            [self.generate_expression(arg) for arg in node.arguments],
        )
        return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"

    def generate_fmod_call(self, node):
        # WGSL has no fmod built-in; its floating `%` is the same truncated remainder
        # (e1 - e2 * trunc(e1 / e2), WGSL section 8.7) and accepts a scalar operand.
        if len(node.arguments) != 2:
            raise WGSLRefusal(
                "WGSL target supports fmod() calls with exactly 2 arguments; got "
                f"{len(node.arguments)}"
            )
        left = self.generate_expression(node.arguments[0])
        right = self.generate_expression(node.arguments[1])
        return f"(({left}) % ({right}))"
