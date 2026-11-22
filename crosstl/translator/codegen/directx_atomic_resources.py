"""Atomic resources and typed buffers in the HLSL generator: texture attributes, registers, image semantics, descriptors and bindings, ternary returns and byte address queries."""

from ..ast import (
    ArrayLiteralNode,
    ArrayNode,
    ArrayType,
    ConstructorNode,
    FunctionCallNode,
    PointerType,
    StructNode,
)
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from ..vulkan_memory_scope import (
    VULKAN_MEMORY_SCOPE_QUALIFIERS,
    needs_device_coherence,
)
from .array_utils import split_array_type_suffix
from .directx_values import (
    DirectXPrivatePointerParameterError,
    DirectXWorkgroupPointerError,
)
from .generic_function_utils import generic_function_call_name
from .hlsl_conservative_depth import hlsl_conservative_depth_semantic
from .hlsl_image_formats import dxc_vulkan_image_format_prefix
from .hlsl_semantic_attributes import (
    hlsl_argument_semantic_name,
    hlsl_parameter_location_semantic,
)
from .hlsl_unrepresentable_memory import hlsl_unrepresentable_memory_qualifier
from .image_access_contracts import (
    explicit_image_format,
    image_format_channel_count,
    image_format_component_type,
    image_format_vector_type,
    is_image_format_attribute,
    is_resource_access_attribute,
    resource_query_scalar_constant_helper_descriptor,
    resource_query_scalar_helper_descriptor,
    resource_query_size_components_descriptor,
    resource_query_size_helper_descriptor,
    supported_image_formats,
    texture_query_levels_multisample_expression,
)
from .memory_qualifier_names import memory_qualifier_name
from .resource_arrays import is_private_pointer_parameter


class DirectXAtomicResourceMixin:
    """Atomic resource members of ``HLSLCodeGen``."""

    def render_hlsl_typed_buffer_atomic_embedded_expression(
        self, expr, expected_type, indent
    ):
        if expr is None or not self.hlsl_expression_contains_typed_buffer_atomic(expr):
            return None

        if self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            temp_type = (
                self.type_name_string(expected_type)
                or self.expression_result_type(expr)
                or "uint"
            )
            temp_name = self.next_hlsl_temp_variable("atomic_ternary")
            indent_str = "    " * indent
            code = f"{indent_str}{self.map_type(temp_type)} {temp_name};\n"
            code += self.generate_hlsl_typed_buffer_atomic_ternary_assignment(
                expr, temp_name, "=", temp_type, indent
            )
            return code, temp_name

        if isinstance(expr, ArrayLiteralNode):
            return self.render_hlsl_typed_buffer_atomic_array_literal(
                expr, expected_type, indent
            )

        if isinstance(expr, ConstructorNode):
            constructor_type = (
                self.expression_result_type(expr)
                or self.type_name_string(getattr(expr, "constructor_type", None))
                or self.type_name_string(expected_type)
            )
            rendered_constructor = (
                self.render_hlsl_typed_buffer_atomic_struct_constructor(
                    constructor_type,
                    getattr(expr, "arguments", []),
                    getattr(expr, "named_arguments", {}),
                    indent,
                )
            )
            if rendered_constructor is not None:
                return rendered_constructor

        if isinstance(expr, FunctionCallNode) or (
            hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__)
        ):
            func_name = self.function_call_name(expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            if isinstance(func_name, str):
                parts = self.hlsl_typed_buffer_atomic_parts(func_name, args)
                if parts is not None:
                    return self.render_hlsl_typed_buffer_atomic_call_value(
                        parts, indent, expected_type
                    )

                if (
                    func_name == "buffer_store"
                    and func_name not in self.function_return_types
                    and len(args) == 3
                ):
                    self.validate_buffer_call_access(func_name, args)
                    resource_type = self.hlsl_buffer_helper_resource_type(args[0])
                    element_type = self.hlsl_typed_buffer_element_type(
                        resource_type,
                        {
                            "RWBuffer",
                            "RWStructuredBuffer",
                            "RasterizerOrderedBuffer",
                            "RasterizerOrderedStructuredBuffer",
                        },
                    )
                    code, value = self.render_hlsl_typed_buffer_atomic_value_expression(
                        args[2], element_type, indent
                    )
                    return code, self.generate_buffer_call(
                        func_name, args, rendered_value=value
                    )

                if self.value_component_count(func_name) is not None:
                    code = ""
                    rendered_args = []
                    changed = False
                    for arg in args:
                        arg_code, rendered_arg = (
                            self.render_hlsl_typed_buffer_atomic_value_expression(
                                arg, None, indent
                            )
                        )
                        code += arg_code
                        rendered_args.append(rendered_arg)
                        changed = changed or bool(arg_code)
                    if changed:
                        return (
                            code,
                            self.hlsl_constructor_expression_from_rendered_args(
                                func_name, args, rendered_args
                            ),
                        )

                rendered_constructor = (
                    self.render_hlsl_typed_buffer_atomic_struct_constructor(
                        func_name, args, {}, indent
                    )
                )
                if rendered_constructor is not None:
                    return rendered_constructor

                if func_name in getattr(self, "function_return_types", {}):
                    code = ""
                    rendered_args = []
                    changed = False
                    parameter_types = self.function_parameter_types.get(func_name) or []
                    for arg in args:
                        index = len(rendered_args)
                        arg_expected_type = (
                            parameter_types[index]
                            if index < len(parameter_types)
                            else self.expression_result_type(arg)
                        )
                        arg_code, rendered_arg = (
                            self.render_hlsl_typed_buffer_atomic_value_expression(
                                arg, arg_expected_type, indent
                            )
                        )
                        code += arg_code
                        rendered_args.append(rendered_arg)
                        changed = changed or bool(arg_code)
                    if changed:
                        specialized_func_name = generic_function_call_name(
                            self, func_name, args
                        )
                        callee = specialized_func_name or func_name
                        call_args = self.generate_call_arguments_from_rendered(
                            func_name, args, rendered_args
                        )
                        return code, f"{callee}({', '.join(call_args)})"

                if (
                    func_name in self.HLSL_BITCAST_FUNCTION_TARGETS
                    and func_name not in self.function_return_types
                    and len(args) == 1
                ):
                    code, value = self.render_hlsl_typed_buffer_atomic_value_expression(
                        args[0], self.expression_result_type(args[0]), indent
                    )
                    bitcast = self.generate_hlsl_bitcast_call(
                        func_name, args, argument_code=value
                    )
                    if bitcast is not None:
                        return code, bitcast

                if (
                    func_name in {"sqrt", "abs", "floor", "ceil", "round", "trunc"}
                    and len(args) == 1
                ):
                    code, value = self.render_hlsl_typed_buffer_atomic_value_expression(
                        args[0], self.expression_result_type(args[0]), indent
                    )
                    return code, f"{func_name}({value})"

        if hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            left_expr = getattr(expr, "left", "")
            right_expr = getattr(expr, "right", "")
            left_code, rendered_left = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    left_expr, self.expression_result_type(left_expr), indent
                )
            )
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "+")))
            if op in {"&&", "||"}:
                temp_name = self.next_hlsl_temp_variable("atomic_condition")
                indent_str = "    " * indent
                code = left_code + f"{indent_str}bool {temp_name} = {rendered_left};\n"
                condition = temp_name if op == "&&" else f"!{temp_name}"
                code += f"{indent_str}if ({condition}) {{\n"
                right_code, rendered_right = (
                    self.render_hlsl_typed_buffer_atomic_value_expression(
                        right_expr, "bool", indent + 1
                    )
                )
                code += (
                    right_code + f"{indent_str}    {temp_name} = {rendered_right};\n"
                )
                code += f"{indent_str}}}\n"
                return code, temp_name
            right_code, rendered_right = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    right_expr, self.expression_result_type(right_expr), indent
                )
            )
            if left_code or right_code:
                op = self.map_operator(
                    getattr(expr, "operator", getattr(expr, "op", "+"))
                )
                return (
                    left_code + right_code,
                    f"({rendered_left} {op} {rendered_right})",
                )

        if hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand_expr = getattr(expr, "operand", "")
            operand_code, rendered_operand = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    operand_expr, self.expression_result_type(operand_expr), indent
                )
            )
            if operand_code:
                op = self.map_operator(
                    getattr(expr, "operator", getattr(expr, "op", "+"))
                )
                if getattr(expr, "is_postfix", False):
                    return operand_code, f"{rendered_operand}{op}"
                return operand_code, f"{op}{rendered_operand}"

        return None

    def render_hlsl_typed_buffer_atomic_value_expression(
        self, expr, expected_type, indent
    ):
        rendered = self.render_hlsl_typed_buffer_atomic_embedded_expression(
            expr, expected_type, indent
        )
        if rendered is not None:
            return rendered

        lifted = self.hlsl_typed_buffer_atomic_lifted_expression(expr, expected_type)
        if lifted is not None:
            lift_statements, rendered_expr = lifted
            code = self.generate_statement_code("\n".join(lift_statements), indent)
            return code, rendered_expr

        if self.hlsl_expression_contains_typed_buffer_atomic(expr):
            raise DirectXRefusal(
                "DirectX typed buffer atomic expression requires statement "
                "lowering in this expression context"
            )

        return "", self.generate_expression_with_expected(expr, expected_type)

    def generate_hlsl_typed_buffer_atomic_value_assignment_statement(
        self, stmt, indent
    ):
        if hasattr(stmt, "target") and hasattr(stmt, "value"):
            target = stmt.target
            value = stmt.value
            op = getattr(stmt, "operator", "=")
        else:
            target = stmt.left
            value = stmt.right
            op = getattr(stmt, "operator", "=")

        if not self.hlsl_expression_contains_typed_buffer_atomic(value):
            return None

        if (
            op == "="
            and (
                isinstance(value, FunctionCallNode)
                or (
                    hasattr(value, "__class__")
                    and "FunctionCall" in str(value.__class__)
                )
            )
            and self.hlsl_typed_buffer_atomic_parts(
                self.function_call_name(value),
                getattr(value, "arguments", getattr(value, "args", [])),
                render_operands=False,
            )
            is not None
        ):
            atomic_statement = self.generate_hlsl_typed_buffer_atomic_statement(
                value,
                self.generate_expression(target),
                self.expression_result_type(target),
            )
            return self.generate_statement_code(atomic_statement, indent)

        return self.generate_hlsl_typed_buffer_atomic_assignment_from_expression(
            value,
            self.generate_expression(target),
            op,
            self.expression_result_type(target),
            indent,
        )

    def generate_hlsl_typed_buffer_atomic_assignment_from_expression(
        self, expr, target, op, expected_type, indent
    ):
        if self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            return self.generate_hlsl_typed_buffer_atomic_ternary_assignment(
                expr, target, op, expected_type, indent
            )

        code, rendered_expr = self.render_hlsl_typed_buffer_atomic_value_expression(
            expr, expected_type, indent
        )
        indent_str = "    " * indent
        code += f"{indent_str}{target} {op} {rendered_expr};\n"
        return code

    def generate_hlsl_typed_buffer_atomic_ternary_assignment(
        self, expr, target, op, expected_type, indent
    ):
        if not self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            return None

        condition = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")
        condition_code, rendered_condition = (
            self.render_hlsl_typed_buffer_atomic_value_expression(
                condition, "bool", indent
            )
        )

        indent_str = "    " * indent
        code = condition_code
        code += f"{indent_str}if ({self.hlsl_strip_wrapping_parentheses(rendered_condition)}) {{\n"
        code += self.generate_hlsl_typed_buffer_atomic_assignment_from_expression(
            true_expr, target, op, expected_type, indent + 1
        )
        code += f"{indent_str}}} else {{\n"
        code += self.generate_hlsl_typed_buffer_atomic_assignment_from_expression(
            false_expr, target, op, expected_type, indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_hlsl_typed_buffer_atomic_ternary_initialization(
        self, expr, declaration, target, expected_type, indent
    ):
        if not self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            return None

        indent_str = "    " * indent
        code = f"{indent_str}{declaration};\n"
        code += self.generate_hlsl_typed_buffer_atomic_ternary_assignment(
            expr, target, "=", expected_type, indent
        )
        return code

    def generate_hlsl_typed_buffer_atomic_ternary_assignment_statement(
        self, stmt, indent
    ):
        if hasattr(stmt, "target") and hasattr(stmt, "value"):
            target = stmt.target
            value = stmt.value
            op = getattr(stmt, "operator", "=")
        else:
            target = stmt.left
            value = stmt.right
            op = getattr(stmt, "operator", "=")

        if not self.hlsl_typed_buffer_atomic_ternary_expression(value):
            return None

        return self.generate_hlsl_typed_buffer_atomic_ternary_assignment(
            value,
            self.generate_expression(target),
            op,
            self.expression_result_type(target),
            indent,
        )

    def generate_hlsl_typed_buffer_atomic_return_from_expression(self, expr, indent):
        if (
            isinstance(expr, FunctionCallNode)
            and self.function_call_name(expr) == "atomicStore"
        ):
            store_return = self.generate_hlsl_typed_buffer_atomic_return(expr, indent)
            if store_return is not None:
                return store_return
        if self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            condition = getattr(expr, "condition", "")
            true_expr = getattr(expr, "true_expr", "")
            false_expr = getattr(expr, "false_expr", "")
            condition_code, rendered_condition = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    condition, "bool", indent
                )
            )
            indent_str = "    " * indent
            code = condition_code
            code += f"{indent_str}if ({self.hlsl_strip_wrapping_parentheses(rendered_condition)}) {{\n"
            code += self.generate_hlsl_typed_buffer_atomic_return_from_expression(
                true_expr, indent + 1
            )
            code += f"{indent_str}}} else {{\n"
            code += self.generate_hlsl_typed_buffer_atomic_return_from_expression(
                false_expr, indent + 1
            )
            code += f"{indent_str}}}\n"
            return code

        code, rendered_expr = self.render_hlsl_typed_buffer_atomic_value_expression(
            expr, self.current_function_return_type, indent
        )
        code += self.generate_hlsl_return_value_statement(rendered_expr, indent)
        return code

    def generate_hlsl_typed_buffer_atomic_ternary_return(self, expr, indent=0):
        if not self.hlsl_typed_buffer_atomic_ternary_expression(expr):
            return None
        return self.generate_hlsl_typed_buffer_atomic_return_from_expression(
            expr, indent
        )

    def hlsl_typed_buffer_atomic_lifted_expression(self, expr, expected_type=None):
        if not self.hlsl_expression_contains_typed_buffer_atomic(expr):
            return None
        statements, expression, changed = self.hlsl_typed_buffer_atomic_lift_expression(
            expr, expected_type
        )
        if not changed:
            return None
        return statements, expression

    def hlsl_typed_buffer_atomic_lift_expression(self, expr, expected_type=None):
        if expr is None:
            return [], "", False

        if isinstance(expr, FunctionCallNode) or (
            hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__)
        ):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            func_name = getattr(func_expr, "name", func_expr)
            if isinstance(func_name, str):
                args = getattr(expr, "arguments", getattr(expr, "args", []))
                parts = self.hlsl_typed_buffer_atomic_parts(func_name, args)
                if parts is not None:
                    original_arg = parts["original_arg"]
                    if original_arg is not None:
                        original = self.generate_expression(original_arg)
                        declaration = []
                    else:
                        temp_type = self.map_type(parts["target_type"])
                        original = self.next_hlsl_temp_variable("atomic_expr")
                        declaration = [f"{temp_type} {original}"]
                    call_args = [*parts["target_args"], *parts["values"], original]
                    return (
                        [
                            *parts["code"].splitlines(),
                            *declaration,
                            f"{parts['intrinsic']}({', '.join(call_args)})",
                        ],
                        original,
                        True,
                    )

                if self.value_component_count(func_name) is not None:
                    statements = []
                    rendered_args = []
                    changed = False
                    for arg in args:
                        arg_statements, rendered_arg, arg_changed = (
                            self.hlsl_typed_buffer_atomic_lift_expression(
                                arg, self.expression_result_type(arg)
                            )
                        )
                        statements.extend(arg_statements)
                        rendered_args.append(rendered_arg)
                        changed = changed or arg_changed
                    if changed:
                        return (
                            statements,
                            self.hlsl_constructor_expression_from_rendered_args(
                                func_name, args, rendered_args
                            ),
                            True,
                        )

            return (
                [],
                self.generate_expression_with_expected(expr, expected_type),
                False,
            )

        if hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            left_statements, left, left_changed = (
                self.hlsl_typed_buffer_atomic_lift_expression(
                    getattr(expr, "left", ""),
                    self.expression_result_type(getattr(expr, "left", "")),
                )
            )
            right_statements, right, right_changed = (
                self.hlsl_typed_buffer_atomic_lift_expression(
                    getattr(expr, "right", ""),
                    self.expression_result_type(getattr(expr, "right", "")),
                )
            )
            if not left_changed and not right_changed:
                return (
                    [],
                    self.generate_expression_with_expected(expr, expected_type),
                    False,
                )
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "+")))
            return (
                [*left_statements, *right_statements],
                f"({left} {op} {right})",
                True,
            )

        if hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            operand_statements, operand, changed = (
                self.hlsl_typed_buffer_atomic_lift_expression(
                    getattr(expr, "operand", ""),
                    self.expression_result_type(getattr(expr, "operand", "")),
                )
            )
            if not changed:
                return (
                    [],
                    self.generate_expression_with_expected(expr, expected_type),
                    False,
                )
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "+")))
            if getattr(expr, "is_postfix", False):
                return operand_statements, f"{operand}{op}", True
            return operand_statements, f"{op}{operand}", True

        return [], self.generate_expression_with_expected(expr, expected_type), False

    def generate_hlsl_typed_buffer_atomic_expression(self, func_name, args):
        if (
            func_name == "atomicCompareExchangeWeak"
            and func_name not in self.function_return_types
        ):
            return self.generate_hlsl_expected_compare_call(args)
        parts = self.hlsl_typed_buffer_atomic_parts(
            func_name, args, render_operands=False
        )
        if parts is None:
            return None
        raise DirectXRefusal(
            f"DirectX typed buffer atomic '{func_name}' requires statement "
            "lowering in this expression context"
        )

    def byteaddress_atomic_operations(self):
        return {
            # A relaxed atomic read is an OR with zero; a relaxed atomic write is an exchange.
            "atomicLoad": ("or", "InterlockedOr", 1),
            "atomicStore": ("exchange", "InterlockedExchange", 2),
            "atomicAdd": ("add", "InterlockedAdd", 2),
            "atomicMin": ("min", "InterlockedMin", 2),
            "atomicMax": ("max", "InterlockedMax", 2),
            "atomicAnd": ("and", "InterlockedAnd", 2),
            "atomicOr": ("or", "InterlockedOr", 2),
            "atomicXor": ("xor", "InterlockedXor", 2),
            "atomicExchange": ("exchange", "InterlockedExchange", 2),
            "atomicCompSwap": ("compare_exchange", "InterlockedCompareExchange", 3),
            "atomicCompareExchange": (
                "compare_exchange",
                "InterlockedCompareExchange",
                3,
            ),
        }

    def byteaddress_atomic_helper_name(self, operation, component_type):
        return f"__crossgl_byteaddress_atomic_{operation}_{component_type}"

    def byteaddress_atomic_uses_uint_bits(self, operation, component_type):
        return component_type == "int" and operation in {
            "add",
            "and",
            "or",
            "xor",
            "exchange",
            "compare_exchange",
        }

    def generate_byteaddress_atomic_helpers(self):
        if not self.required_byteaddress_atomic_helpers:
            return ""

        helpers = []
        for operation, intrinsic, component_type in sorted(
            self.required_byteaddress_atomic_helpers
        ):
            helper_name = self.byteaddress_atomic_helper_name(operation, component_type)
            value_type = self.map_type(component_type)
            use_uint_bits = self.byteaddress_atomic_uses_uint_bits(
                operation, component_type
            )
            original_type = "uint" if use_uint_bits else value_type
            return_value = "asint(original)" if use_uint_bits else "original"
            value_expr = "asuint(value)" if use_uint_bits else "value"
            if operation == "compare_exchange":
                compare_value_expr = (
                    "asuint(compareValue)" if use_uint_bits else "compareValue"
                )
                helpers.append(
                    f"{value_type} {helper_name}(RWByteAddressBuffer buffer, uint offset, {value_type} compareValue, {value_type} value) {{\n"
                    f"    {original_type} original;\n"
                    f"    buffer.{intrinsic}(offset, {compare_value_expr}, {value_expr}, original);\n"
                    f"    return {return_value};\n"
                    "}\n\n"
                )
                continue
            helpers.append(
                f"{value_type} {helper_name}(RWByteAddressBuffer buffer, uint offset, {value_type} value) {{\n"
                f"    {original_type} original;\n"
                f"    buffer.{intrinsic}(offset, {value_expr}, original);\n"
                f"    return {return_value};\n"
                "}\n\n"
            )
        return "".join(helpers)

    def generate_texture_size_helper(self, texture_type):
        descriptor = self.texture_size_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureSize", texture_type, descriptor
        )

    def generate_texture_query_dimension_helper(
        self, helper_name, texture_type, descriptor
    ):
        texture_type = self.resource_base_type(texture_type)
        parameter_type = self.directx_resource_declaration_type(texture_type)
        parameters = [f"{parameter_type} tex"]
        if descriptor["function_params"]:
            parameters.append(descriptor["function_params"])
        declarations = "".join(
            f"    uint {dimension};\n" for dimension in descriptor["dimensions"]
        )
        get_dimensions_args = descriptor["get_dimensions_args"]
        get_dimensions_call = ""
        if get_dimensions_args:
            get_dimensions_call = (
                f"    tex.GetDimensions({', '.join(get_dimensions_args)});\n"
            )
        return (
            f"{descriptor['return_type']} {helper_name}({', '.join(parameters)}) {{\n"
            f"{declarations}"
            f"{get_dimensions_call}"
            f"    return {descriptor['return_expr']};\n"
            "}\n\n"
        )

    def texture_size_helper_descriptor(self, texture_type):
        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "lod")
        return resource_query_size_helper_descriptor(descriptor)

    def texture_query_get_dimensions_descriptor(self, texture_type, lod_arg):
        texture_type = self.resource_base_type(texture_type)
        shape_type = self.sampled_texture_shape_type(texture_type)

        def mip_descriptor(return_type, size_components):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=("levels",),
                function_params="int lod",
                get_dimensions_prefix=(lod_arg,),
            )

        def sample_descriptor(return_type, size_components):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=("samples",),
            )

        if shape_type == "Texture2DMSArray" or texture_type.startswith(
            "RWTexture2DMSArray<"
        ):
            return sample_descriptor("int3", ("width", "height", "elements"))
        if shape_type == "Texture2DMS" or texture_type.startswith("RWTexture2DMS<"):
            return sample_descriptor("int2", ("width", "height"))
        descriptors = {
            "Texture1D": mip_descriptor("int", ("width",)),
            "Texture1DArray": mip_descriptor("int2", ("width", "elements")),
            "Texture2D": mip_descriptor("int2", ("width", "height")),
            "TextureCube": mip_descriptor("int2", ("width", "height")),
            "Texture2DArray": mip_descriptor("int3", ("width", "height", "elements")),
            "TextureCubeArray": mip_descriptor("int3", ("width", "height", "elements")),
            "Texture3D": mip_descriptor("int3", ("width", "height", "depth")),
        }
        return descriptors.get(shape_type)

    def generate_image_size_helper(self, texture_type):
        descriptor = self.image_size_helper_descriptor(texture_type)
        if descriptor is None:
            return ""

        parameter_type = self.directx_resource_declaration_type(texture_type)
        dimensions = descriptor["dimensions"]
        declarations = "".join(f"    uint {dimension};\n" for dimension in dimensions)
        dimension_args = ", ".join(dimensions)
        return (
            f"{descriptor['return_type']} imageSize({parameter_type} image) {{\n"
            f"{declarations}"
            f"    image.GetDimensions({dimension_args});\n"
            f"    return {descriptor['return_expr']};\n"
            "}\n\n"
        )

    def image_size_helper_descriptor(self, texture_type):
        if not self.is_storage_image_resource_type(texture_type):
            return None

        base_type = self.resource_base_type(texture_type)

        def size_descriptor(return_type, size_components, tail_dimensions=()):
            return resource_query_size_components_descriptor(
                return_type,
                size_components,
                tail_dimensions=tail_dimensions,
            )

        descriptors = (
            (
                "RWTexture2DMSArray<",
                size_descriptor(
                    "int3",
                    ("width", "height", "elements"),
                    tail_dimensions=("samples",),
                ),
            ),
            (
                "RWTexture2DMS<",
                size_descriptor(
                    "int2", ("width", "height"), tail_dimensions=("samples",)
                ),
            ),
            (
                "RWTexture2DArray<",
                size_descriptor("int3", ("width", "height", "elements")),
            ),
            (
                "RWTexture1DArray<",
                size_descriptor("int2", ("width", "elements")),
            ),
            (
                "RWTexture1D<",
                size_descriptor("int", ("width",)),
            ),
            (
                "RWTexture3D<",
                size_descriptor("int3", ("width", "height", "depth")),
            ),
            (
                "RWTextureCube<",
                size_descriptor(
                    "int2",
                    ("width", "height"),
                    tail_dimensions=("elements",),
                ),
            ),
        )
        for prefix, descriptor in descriptors:
            if base_type.startswith(prefix):
                return resource_query_size_helper_descriptor(
                    descriptor, include_function_fields=False
                )
        return resource_query_size_helper_descriptor(
            size_descriptor("int2", ("width", "height")),
            include_function_fields=False,
        )

    def generate_texture_query_levels_helper(self, texture_type):
        descriptor = self.texture_query_levels_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureQueryLevels", texture_type, descriptor
        )

    def texture_query_levels_helper_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if self.is_multisample_texture_resource_type(texture_type):
            return resource_query_scalar_constant_helper_descriptor(
                texture_query_levels_multisample_expression()
            )

        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "0")
        return resource_query_scalar_helper_descriptor(descriptor, "int(levels)")

    def generate_texture_samples_helper(self, texture_type):
        descriptor = self.texture_samples_helper_descriptor(texture_type)
        if descriptor is None:
            return ""
        return self.generate_texture_query_dimension_helper(
            "textureSamples", texture_type, descriptor
        )

    def texture_samples_helper_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not (
            self.is_multisample_texture_resource_type(texture_type)
            or self.is_multisample_storage_image_resource_type(texture_type)
        ):
            return None
        descriptor = self.texture_query_get_dimensions_descriptor(texture_type, "lod")
        return resource_query_scalar_helper_descriptor(descriptor, "int(samples)")

    def supported_image_formats(self):
        return supported_image_formats()

    def scalar_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) == 1
        }

    def vector_image_format_components(self):
        return {
            image_format: image_format_vector_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) in {2, 3, 4}
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "group",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "readonly",
            "readwrite",
            "restrict",
            "volatile",
            "writeonly",
            *VULKAN_MEMORY_SCOPE_QUALIFIERS,
        }

    def is_rasterizer_ordered_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "rasterizer_ordered")

    def is_rasterizer_ordered_resource(self, node):
        return any(
            self.is_rasterizer_ordered_attribute(attr)
            for attr in getattr(node, "attributes", []) or []
        )

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def register_space_value(self, value):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            space_index = int(raw_value)
            return None if space_index == 0 else f"space{space_index}"
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            space_index = int(raw_value[5:])
            return None if space_index == 0 else f"space{space_index}"
        return None

    def explicit_resource_binding_index(
        self, node, attribute_names=(), register_prefixes=()
    ):
        override = self.directx_resource_register_override(node, register_prefixes)
        if override is not None:
            return override
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in attribute_names:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], register_prefixes)
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def directx_resource_binding_provenance(self, node):
        if hasattr(node, "get_annotation"):
            provenance = node.get_annotation("directx.binding_provenance", None)
            if provenance:
                return str(provenance)

        attribute_label = self.directx_resource_binding_attribute_label(node)
        if attribute_label is None:
            return None
        declaration_label = self.directx_resource_declaration_label(node)
        if declaration_label:
            return f"{attribute_label} on {declaration_label}"
        return attribute_label

    def directx_resource_binding_attribute_label(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name not in {
                "binding",
                "buffer",
                "register",
                "sampler",
                "texture",
                "uav",
            }:
                continue
            argument_text = ", ".join(
                str(
                    self.attribute_value_to_string(argument)
                    or expression_debug_name(argument)
                    or argument
                )
                for argument in arguments
            )
            return f"@{attr_name}({argument_text})"
        return None

    def directx_resource_declaration_label(self, node):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        if not name:
            return None
        if isinstance(node, StructNode) or getattr(node, "is_cbuffer", False):
            keyword = "tbuffer" if self.is_hlsl_tbuffer_node(node) else "cbuffer"
            return f"{keyword} {name}"
        return f"resource {name}"

    def directx_resource_register_override(self, node, register_prefixes=()):
        overrides = self.directx_resource_register_overrides.get(id(node), {})
        if not overrides:
            return None
        for prefix in register_prefixes:
            if prefix in overrides:
                return overrides[prefix]
        if not register_prefixes and len(overrides) == 1:
            return next(iter(overrides.values()))
        return None

    def explicit_resource_register_space(self, node):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name == "space" and arguments:
                space = self.register_space_value(arguments[0])
                if space is not None:
                    return space
            if attr_name == "register":
                for argument in arguments[1:]:
                    space = self.register_space_value(argument)
                    if space is not None:
                        return space
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            if str(attr_name).lower() == "set":
                return self.register_space_value(arguments[0])
        return None

    def explicit_resource_register_prefix(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or str(attr_name).lower() != "register" or not arguments:
                continue
            value = self.attribute_value_to_string(arguments[0])
            if value is None:
                return None
            value = str(value).strip().lower()
            prefix = ""
            for char in value:
                if not char.isalpha():
                    break
                prefix += char
            return prefix or None
        return None

    def validate_hlsl_acceleration_structure_register(self, node):
        prefix = self.explicit_resource_register_prefix(node)
        if prefix is None or prefix == "t":
            return
        name = getattr(node, "name", getattr(node, "variable_name", "<anonymous>"))
        raise DirectXRefusal(
            "DirectX RaytracingAccelerationStructure resource "
            f"'{name}' must use an SRV t-register, got {prefix}"
        )

    def resource_register_suffix_for_space(self, register_prefix, binding, space=None):
        register = f"{register_prefix}{binding}"
        if space:
            return f" : register({register}, {space})"
        return f" : register({register})"

    def resource_register_suffix(self, register_prefix, binding, node):
        return self.resource_register_suffix_for_space(
            register_prefix,
            binding,
            self.explicit_resource_register_space(node),
        )

    def storage_image_format_prefix(self, mapped_type, node):
        if not self.is_hlsl_rw_texture_type(mapped_type):
            return ""
        image_format = explicit_image_format(node, self.attribute_value_to_string)
        return dxc_vulkan_image_format_prefix(image_format)

    def resource_memory_qualifier(self, mapped_type, node):
        unrepresentable = hlsl_unrepresentable_memory_qualifier(node)
        if unrepresentable is not None:
            return self.refuse_lowering(unrepresentable)
        if not (
            self.is_hlsl_rw_texture_type(mapped_type)
            or str(mapped_type).startswith("RWBuffer")
            or self.is_hlsl_uav_buffer_type(mapped_type)
        ):
            return ""

        qualifiers = {
            memory_qualifier_name(q) for q in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        if qualifiers & {"coherent", "globallycoherent"} or attributes & {
            "coherent",
            "globallycoherent",
        }:
            return "globallycoherent "
        if needs_device_coherence(node):
            # globallycoherent reaches the device, which holds every scope GL_KHR_memory_scope_semantics names.
            return "globallycoherent "
        return ""

    def semantic_from_node(self, node):
        conservative_depth = hlsl_conservative_depth_semantic(node)
        if conservative_depth is not None:
            return conservative_depth
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_rasterizer_ordered_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.hlsl_mesh_parameter_role_attribute_name(attr)
                or self.hlsl_mesh_payload_parameter_attribute_name(attr)
                or self.is_hlsl_resource_array_size_marker(node, attr)
                or self.hlsl_root_signature_attribute(attr)
                or self.is_hlsl_declaration_metadata_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                if self.is_hlsl_interpolation_metadata_name(attr.name):
                    continue
                if str(attr.name).lower() == "location":
                    location_semantic = hlsl_parameter_location_semantic(
                        self, node, attr
                    )
                    if location_semantic is not None:
                        return location_semantic
                return self.semantic_attribute_name(attr)
        return None

    def hlsl_function_return_semantic(self, func):
        conservative_depth = hlsl_conservative_depth_semantic(func)
        if conservative_depth is not None:
            return conservative_depth
        semantic = getattr(func, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(func, "attributes"):
            return None
        for attr in func.attributes:
            if self.hlsl_stage_attribute_name(attr):
                continue
            if self.hlsl_root_signature_attribute(attr):
                continue
            if self.hlsl_waveops_include_helper_lanes_attribute(attr):
                continue
            if self.hlsl_wave_size_attribute(attr):
                continue
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_rasterizer_ordered_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.hlsl_mesh_parameter_role_attribute_name(attr)
                or self.hlsl_mesh_payload_parameter_attribute_name(attr)
                or self.is_hlsl_declaration_metadata_attribute(attr)
            ):
                continue
            location_semantic = self.hlsl_location_return_semantic(attr)
            if location_semantic is not None:
                return location_semantic
            if hasattr(attr, "name"):
                return self.semantic_attribute_name(attr)
        return None

    def semantic_from_array_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_rasterizer_ordered_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.is_hlsl_resource_array_size_marker(node, attr)
                or self.is_hlsl_declaration_metadata_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                if self.is_hlsl_interpolation_metadata_name(attr.name):
                    continue
                return self.semantic_attribute_name(attr)
        return None

    def semantic_attribute_name(self, attr):
        name = getattr(attr, "name", None)
        if name == "gl_FragData":
            arguments = getattr(attr, "arguments", None) or getattr(attr, "args", None)
            if arguments:
                index_arg = arguments[0]
                index = getattr(index_arg, "value", None)
                if index is None:
                    index = getattr(index_arg, "name", None)
                if index is None:
                    index = self.attribute_value_to_string(index_arg)
                if index is not None:
                    return f"gl_FragData[{index}]"
        return hlsl_argument_semantic_name(self, attr)

    def hlsl_location_return_semantic(self, attr):
        index = self.hlsl_location_attribute_index(attr)
        if index is None:
            return None
        return f"SV_Target{index}"

    def hlsl_location_attribute_index(self, node):
        if hasattr(node, "attributes"):
            attributes = getattr(node, "attributes", [])
        else:
            attributes = [node]
        for attr in attributes or []:
            name = getattr(attr, "name", None)
            if str(name).lower() != "location":
                continue
            arguments = getattr(attr, "arguments", None) or getattr(attr, "args", None)
            if not arguments:
                continue
            index_arg = arguments[0]
            index = getattr(index_arg, "value", None)
            if index is None:
                index = getattr(index_arg, "name", None)
            if index is None:
                index = self.attribute_value_to_string(index_arg)
            if isinstance(index, int):
                return index
            if str(index).isdigit():
                return int(index)
        return None

    def semantic_from_struct_member(self, member):
        if isinstance(member, ArrayNode):
            return self.semantic_from_array_node(member)
        return self.semantic_from_node(member)

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)

        metal_buffer_type = self.hlsl_metal_buffer_parameter_resource_type(vtype, node)
        if metal_buffer_type is not None:
            return metal_buffer_type

        constant_reference_type = self.hlsl_constant_reference_parameter_type(
            vtype, node
        )
        if constant_reference_type is not None:
            return constant_reference_type

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)
        lowered_block = self.current_glsl_buffer_block_parameters.get(param_name)
        if lowered_block is not None:
            mapped_type = (
                "ByteAddressBuffer"
                if lowered_block["readonly"]
                else "RWByteAddressBuffer"
            )
            return self.glsl_buffer_block_parameter_type(
                mapped_type, vtype, node, function_hints
            )

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_resource_array_hint_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return (
                    f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"
                )

        attribute_array_size = self.hlsl_resource_array_size_expression(node, vtype)
        if attribute_array_size is not None:
            base_type = self.convert_type_node_to_string(vtype)
            if self.is_resource_array_hint_type(base_type):
                array_size = self.expression_to_string(attribute_array_size)
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return f"{mapped_type}[{array_size}]"

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_resource_array_hint_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return (
                            f"{mapped_type}[{array_size}]"
                            if array_size
                            else f"{mapped_type}[]"
                        )
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def hlsl_private_pointer_fixed_array_candidate(self, parameter):
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        if not isinstance(parameter_type, ArrayType) or isinstance(
            getattr(parameter_type, "element_type", None), (PointerType, ArrayType)
        ):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return bool(qualifiers.intersection({"thread", "private", "function"}))

    def hlsl_private_pointer_parameter(self, parameter, function_name=None):
        if is_private_pointer_parameter(parameter):
            return True
        if not self.hlsl_private_pointer_fixed_array_candidate(parameter):
            return False
        if function_name is None:
            function_name = self.current_function_name
        return (
            function_name,
            getattr(parameter, "name", None),
        ) in self.hlsl_private_pointer_fixed_array_parameters

    def hlsl_private_pointer_parameter_array_type(self, parameter, function_name):
        if not self.hlsl_private_pointer_parameter(parameter, function_name):
            return None
        parameter_name = getattr(parameter, "name", None)
        size = self.current_hlsl_private_pointer_variant.get(parameter_name)
        if size is None:
            size = self.function_private_pointer_array_size_hints.get(
                function_name, {}
            ).get(parameter_name)
        if not size:
            raise DirectXPrivatePointerParameterError(
                "DirectX private pointer parameter "
                f"'{function_name}.{parameter_name}' requires one consistent "
                "fixed array extent",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="missing-fixed-array-extent",
                source_location=getattr(parameter, "source_location", None),
            )
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        pointee_type = (
            parameter_type.element_type
            if isinstance(parameter_type, ArrayType)
            else getattr(parameter_type, "pointee_type", None)
        )
        if pointee_type is None or isinstance(pointee_type, (PointerType, ArrayType)):
            raise DirectXPrivatePointerParameterError(
                "DirectX private pointer parameter "
                f"'{function_name}.{parameter_name}' has an unsupported pointee type",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        word_view = self.hlsl_private_pointer_word_view_for_variant(size)
        if word_view is not None:
            return word_view["parameter_type"]
        if parameter_name in self.function_private_pointer_scalar_parameters.get(
            function_name, set()
        ):
            return self.map_type(pointee_type)
        return f"{self.map_type(pointee_type)}[{size}]"

    def hlsl_workgroup_pointer_parameter_array_type(self, parameter, function_name):
        if not self.hlsl_workgroup_pointer_declaration(parameter):
            return None
        parameter_name = getattr(parameter, "name", None)
        size = self.current_hlsl_workgroup_pointer_variant.get(parameter_name)
        if not isinstance(size, int) or size <= 0:
            raise DirectXWorkgroupPointerError(
                "DirectX workgroup pointer parameter "
                f"'{function_name}.{parameter_name}' requires a concrete "
                "shared-array extent",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="missing-concrete-size",
                source_location=getattr(parameter, "source_location", None),
            )
        element_type = self.hlsl_workgroup_pointer_element_type(parameter)
        if element_type is None or element_type.endswith("*"):
            raise DirectXWorkgroupPointerError(
                "DirectX workgroup pointer parameter "
                f"'{function_name}.{parameter_name}' has an unsupported pointee "
                "type",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        return f"{element_type}[{size}]"
