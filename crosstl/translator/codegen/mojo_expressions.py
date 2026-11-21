"""Expression helpers of the Mojo generator: lambdas, identifier replacement, constructors, casts, swizzles and GPU built-in variables."""

import re

from ..ast import (
    BuiltinVariableNode,
    VariableNode,
)
from ..refusals import MojoRefusal
from .enum_utils import enum_variant_constructor_name
from .mojo_tables import (
    MOJO_BUILTIN_PLACEHOLDERS,
    MOJO_DTYPE_INFO,
    MOJO_GPU_BUILTIN_ALIASES,
    MOJO_SCALAR_DTYPES,
)


class MojoExpressionMixin:
    """Expression members of ``MojoCodeGen``."""

    def generate_lambda_expression(self, args):
        """Materialize supported CrossGL pseudo-lambdas as local Mojo functions."""
        if not args or not self.expression_prelude_stack:
            return None

        params = []
        param_types = {}
        for arg in args[:-1]:
            param = self.generate_lambda_parameter(arg)
            if param is None:
                return None
            type_name, param_name, mapped_type = param
            params.append((type_name, param_name, mapped_type))
            param_types[param_name] = type_name

        body_arg = args[-1]
        return_expr = self.lambda_return_expression(body_arg)
        return_type = self.infer_lambda_return_type(return_expr, param_types)
        if return_type is None:
            return None

        helper_name = self.next_lambda_helper_name()
        signature_params = ", ".join(
            f"{param_name}: {mapped_type}" for _, param_name, mapped_type in params
        )
        return_type_mapped = self.map_type(return_type)

        context = self.expression_prelude_stack[-1]
        indent = context["indent"]
        indent_str = "    " * indent
        body = self.generate_lambda_function_body(body_arg, indent + 1)
        if body is None:
            return None

        context["lines"].append(
            f"{indent_str}def {helper_name}({signature_params}) -> "
            f"{return_type_mapped}:\n{body}"
        )
        return helper_name

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return None

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        if mapped_type is None:
            return None
        return type_name, param_name, mapped_type

    def generate_lambda_function_body(self, arg, indent):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None

        indent_str = "    " * indent
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            if not inner:
                return None
            lines = []
            for statement in inner.split(";"):
                statement = statement.strip()
                if not statement:
                    continue
                if statement.startswith("return "):
                    value = self.translate_lambda_raw_expression(
                        statement[len("return ") :].strip()
                    )
                    lines.append(f"{indent_str}return {value}\n")
                else:
                    lines.append(
                        f"{indent_str}{self.translate_lambda_raw_expression(statement)}\n"
                    )
            return "".join(lines) if lines else None

        expression = self.translate_lambda_raw_expression(raw)
        return f"{indent_str}return {expression}\n"

    def lambda_raw_argument_text(self, arg):
        if hasattr(arg, "name"):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        type_name = self.canonical_lambda_type(type_name)
        if not param_name.isidentifier():
            return None
        if not type_name:
            return None
        return type_name, param_name

    def canonical_lambda_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "".join(type_name.split())
        return type_name

    def lambda_parameter_type(self, type_name):
        if any(char.isspace() for char in type_name):
            return None
        if any(char in type_name for char in "{},;[]()"):
            return None

        mapped_type = self.map_type(type_name)
        if "<" in type_name or ">" in type_name:
            if mapped_type == type_name:
                return None
        return mapped_type

    def lambda_return_expression(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1].strip()
            for statement in inner.split(";"):
                statement = statement.strip()
                if statement.startswith("return "):
                    return statement[len("return ") :].strip()
            return None
        return raw

    def infer_lambda_return_type(self, return_expr, param_types):
        if not return_expr:
            return None

        stripped = self.strip_wrapping_parentheses(return_expr.strip())
        literal_type = self.lambda_literal_type(stripped)
        if literal_type is not None:
            return literal_type

        if re.fullmatch(r"[A-Za-z_]\w*", stripped):
            return param_types.get(stripped) or self.variable_types.get(stripped)

        referenced_types = {
            type_name
            for name, type_name in param_types.items()
            if re.search(rf"\b{re.escape(name)}\b", stripped)
        }
        if len(referenced_types) == 1:
            return next(iter(referenced_types))
        return None

    def lambda_literal_type(self, value):
        if value in {"true", "false", "True", "False"}:
            return "bool"
        if re.fullmatch(r"[+-]?\d+", value):
            return "int"
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", value):
            return "float"
        return None

    def strip_wrapping_parentheses(self, expression):
        while expression.startswith("(") and expression.endswith(")"):
            inner = expression[1:-1].strip()
            if not inner:
                break
            depth = 0
            wraps = True
            for index, char in enumerate(expression):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(expression) - 1:
                        wraps = False
                        break
            if not wraps:
                break
            expression = inner
        return expression

    def translate_lambda_raw_expression(self, expression):
        translated = expression.strip()
        translated = re.sub(r"\btrue\b", "True", translated)
        translated = re.sub(r"\bfalse\b", "False", translated)
        translated = translated.replace("&&", " and ")
        translated = translated.replace("||", " or ")
        return translated

    def next_lambda_helper_name(self):
        while True:
            helper_name = f"_crossgl_lambda_{self.lambda_counter}"
            self.lambda_counter += 1
            if helper_name not in self.function_return_types:
                return helper_name

    def generate_with_identifier_replacements(self, replacements, callback):
        previous_replacements = self.expression_identifier_replacements
        self.expression_identifier_replacements = {
            **previous_replacements,
            **(replacements or {}),
        }
        try:
            return callback()
        finally:
            self.expression_identifier_replacements = previous_replacements

    def generate_expression_with_identifier_replacements(self, expr, replacements):
        return self.generate_with_identifier_replacements(
            replacements, lambda: self.generate_expression(expr)
        )

    def generate_expression_with_expected(self, expr, expected_type):
        return self.generate_expression(expr, expected_type)

    def generate_enum_variant_constructor_call(self, func_name, args, target_type=None):
        constructor = self.enum_variant_constructors.get(func_name)
        if constructor is None:
            return self.generate_generic_enum_variant_constructor_call(
                func_name, args, target_type
            )

        fields = self.enum_variant_constructor_fields.get(func_name, [])
        if len(args) != len(fields):
            raise MojoRefusal(
                f"Enum constructor {func_name} expects {len(fields)} arguments, "
                f"got {len(args)}"
            )

        rendered_args = []
        for index, arg in enumerate(args):
            expected_type = fields[index][1] if index < len(fields) else None
            rendered_args.append(
                self.generate_expression(
                    arg,
                    expected_type,
                    f"enum constructor {func_name} argument {index + 1}",
                )
            )
        return f"{constructor}({', '.join(rendered_args)})"

    def generate_generic_enum_variant_constructor_call(
        self, func_name, args, target_type=None
    ):
        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is None:
            return None
        if len(args) != len(fields):
            raise MojoRefusal(
                f"Enum constructor {func_name} expects {len(fields)} arguments, "
                f"got {len(args)}"
            )

        specialization, variant_name = self.generic_enum_variant_specialization(
            func_name, target_type
        )
        if specialization is None:
            return None

        rendered_args = []
        for index, arg in enumerate(args):
            rendered_args.append(
                self.generate_expression(
                    arg,
                    fields[index][1],
                    f"enum constructor {func_name} argument {index + 1}",
                )
            )
        constructor = enum_variant_constructor_name(
            specialization["struct_name"], variant_name
        )
        return f"{constructor}({', '.join(rendered_args)})"

    def generate_constructor_node(self, expr, target_type=None, target_context=None):
        type_name = self.convert_type_node_to_string(expr.constructor_type)
        mapped_type = self.map_type(type_name)
        target_name = self.type_name(target_type)
        target_struct_type, target_fields = self.struct_type_name_and_fields(
            target_name
        )
        constructor_struct_type, constructor_fields = self.struct_type_name_and_fields(
            type_name
        )
        if target_struct_type is not None:
            struct_type, fields = target_struct_type, target_fields
        elif constructor_struct_type is not None:
            struct_type, fields = constructor_struct_type, constructor_fields
        else:
            struct_type, fields = type_name, {}
        field_items = list(fields.items())
        base_context = target_context or f"constructor {type_name}"
        named_arguments = getattr(expr, "named_arguments", {}) or {}
        if struct_type in self.struct_types:
            self.validate_struct_constructor_field_bindings(
                struct_type,
                field_items,
                len(expr.arguments),
                named_arguments,
                base_context,
            )
        positional_args = []
        rendered_fields = {}
        for index, argument in enumerate(expr.arguments):
            field_type = None
            field_name = None
            if index < len(field_items):
                field_name, field_type = field_items[index]
                field_context = f"{base_context} field {struct_type}.{field_name}"
            else:
                field_context = f"{base_context} field {index + 1}"
            self.validate_expression_target_shape(argument, field_type, field_context)
            rendered_argument = self.generate_expression(
                argument, field_type, field_context
            )
            positional_args.append(rendered_argument)
            if field_name is not None:
                rendered_fields[field_name] = rendered_argument
        named_args = []
        for name, value in named_arguments.items():
            field_type = fields.get(name)
            field_context = (
                f"{base_context} field {struct_type}.{name}"
                if field_type is not None
                else f"{base_context} field {name}"
            )
            self.validate_expression_target_shape(value, field_type, field_context)
            rendered_value = self.generate_expression(value, field_type, field_context)
            named_args.append(f"{self.mojo_identifier(name)}={rendered_value}")
            if field_type is not None:
                rendered_fields[name] = rendered_value

        if struct_type in self.struct_types:
            missing_fields = [
                (field_name, field_type)
                for field_name, field_type in field_items
                if field_name not in rendered_fields
            ]
            if missing_fields:
                for field_name, field_type in missing_fields:
                    rendered_fields[field_name] = self.zero_value_for_type(field_type)
                field_args = [
                    f"{self.mojo_identifier(field_name)}={rendered_fields[field_name]}"
                    for field_name, _ in field_items
                ]
                return f"{mapped_type}({', '.join(field_args)})"

        return f"{mapped_type}({', '.join([*positional_args, *named_args])})"

    def validate_struct_constructor_field_bindings(
        self,
        struct_type,
        field_items,
        positional_count,
        named_arguments,
        base_context,
        constructor_style="braced",
        require_all_fields=False,
    ):
        fields = dict(field_items)
        expected = ", ".join(fields) or "no fields"
        initialized_fields = set()
        for index in range(positional_count):
            if index >= len(field_items):
                raise MojoRefusal(
                    f"Invalid {constructor_style} struct constructor for Mojo "
                    "codegen; "
                    f"too many positional fields in {base_context}: field "
                    f"{index + 1} has no matching field in {struct_type}"
                )
            field_name, _ = field_items[index]
            initialized_fields.add(field_name)

        for field_name in named_arguments:
            if field_name not in fields:
                raise MojoRefusal(
                    f"Invalid {constructor_style} struct constructor for Mojo "
                    "codegen; "
                    f"unknown field {struct_type}.{field_name} in {base_context}; "
                    f"expected one of: {expected}"
                )
            if field_name in initialized_fields:
                raise MojoRefusal(
                    f"Invalid {constructor_style} struct constructor for Mojo "
                    "codegen; "
                    f"duplicate field {struct_type}.{field_name} in {base_context}"
                )
            initialized_fields.add(field_name)

        if require_all_fields:
            for field_name, _ in field_items:
                if field_name not in initialized_fields:
                    raise MojoRefusal(
                        f"Invalid {constructor_style} struct constructor for Mojo "
                        "codegen; "
                        f"missing field {struct_type}.{field_name} in "
                        f"{base_context}; expected fields: {expected}"
                    )

    def generate_struct_function_constructor_call(
        self, expr, type_name, context, target_type=None
    ):
        struct_type, fields = self.struct_type_name_and_fields(target_type)
        if struct_type is None or struct_type != type_name:
            struct_type, fields = self.struct_type_name_and_fields(type_name)
        mapped_type = (
            self.map_type(target_type) if target_type else self.map_type(type_name)
        )
        field_items = list(fields.items())
        context_type = self.type_name(target_type) if target_type else type_name
        base_context = context or f"constructor {context_type}"
        self.validate_struct_constructor_field_bindings(
            struct_type or type_name,
            field_items,
            len(expr.args),
            {},
            base_context,
            "function-style",
            True,
        )
        positional_args = []
        for index, argument in enumerate(expr.args):
            field_type = None
            if index < len(field_items):
                field_name, field_type = field_items[index]
                field_context = (
                    f"{base_context} field {struct_type or type_name}.{field_name}"
                )
            else:
                field_context = f"{base_context} field {index + 1}"
            self.validate_expression_target_shape(
                argument,
                field_type,
                field_context,
            )
            positional_args.append(
                self.generate_expression(argument, field_type, field_context)
            )
        return f"{mapped_type}({', '.join(positional_args)})"

    def generate_cast_node(self, expr):
        target_type = self.convert_type_node_to_string(expr.target_type)
        target_vector = self.vector_type_info(target_type)
        if target_vector is not None:
            source_vector = self.vector_type_info(
                self.expression_result_type(expr.expression)
            )
            target_dtype, target_width, target_storage_width, _ = target_vector
            if source_vector is not None:
                _, source_width, source_storage_width, _ = source_vector
                if (
                    source_width == target_width
                    and source_storage_width == target_storage_width
                ):
                    value = self.generate_expression(expr.expression)
                    return f"{value}.cast[{target_dtype}]()"
            target_name = self.normalize_generic_vector_type_name(target_type)
            if target_name in self.vector_constructor_info:
                return self.generate_vector_constructor(target_name, [expr.expression])

        target_dtype = MOJO_SCALAR_DTYPES.get(self.type_name(target_type))
        if target_dtype is not None:
            value = self.generate_scalar_constructor_argument(
                expr.expression, target_dtype, f"cast to {self.type_name(target_type)}"
            )
            return f"{self.map_type(target_type)}({value})"

        value = self.generate_expression(expr.expression)
        return f"{self.map_type(target_type)}({value})"

    def generate_swizzle_node(self, expr):
        member = expr.components
        swizzle_indices = self.get_swizzle_indices(member)
        if swizzle_indices is None:
            raise MojoRefusal(f"Unsupported Mojo swizzle '{member}'")
        obj = self.generate_expression(expr.vector_expr)
        obj_type = self.expression_result_type(expr.vector_expr)
        return self.generate_swizzle(
            expr.vector_expr, obj, obj_type, member, swizzle_indices
        )

    def generate_builtin_variable_node(self, expr):
        ray_builtin = self.mojo_ray_builtin_expression(expr.builtin_name)
        if ray_builtin is not None:
            return ray_builtin
        gpu_builtin = self.mojo_gpu_builtin_expression(
            expr.builtin_name, getattr(expr, "component", None)
        )
        if gpu_builtin is not None:
            return gpu_builtin
        name = self.mojo_builtin_placeholder_expression(expr.builtin_name)
        if name is None:
            name = self.map_semantic(expr.builtin_name)
        component = getattr(expr, "component", None)
        if not component:
            return name

        swizzle_indices = self.get_swizzle_indices(component)
        if swizzle_indices is None:
            return f"{name}.{component}"
        obj_type = self.variable_types.get(name, "vec4")
        return self.generate_swizzle(expr, name, obj_type, component, swizzle_indices)

    def mojo_builtin_placeholder_expression(self, name):
        placeholder = MOJO_BUILTIN_PLACEHOLDERS.get(name)
        if placeholder is None or name in self.variable_types:
            return None
        placeholder_name, placeholder_type = placeholder
        if placeholder_name not in self.variable_types:
            self.required_builtin_placeholders.add((placeholder_name, placeholder_type))
        return placeholder_name

    def mojo_gpu_builtin_expression(self, name, component=None):
        builtin = MOJO_GPU_BUILTIN_ALIASES.get(name)
        if builtin is None or name in self.variable_types:
            return None
        alias_name, type_name = builtin
        if alias_name not in self.variable_types:
            self.required_gpu_builtin_placeholders.add((alias_name, type_name))
        if component:
            swizzle_indices = self.get_swizzle_indices(component)
            if swizzle_indices is None:
                return f"{alias_name}.{component}"
            return self.generate_gpu_builtin_swizzle(
                alias_name, type_name, swizzle_indices
            )
        return self.generate_gpu_builtin_vector(alias_name, type_name)

    def generate_gpu_builtin_vector(self, alias_name, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is None:
            return alias_name
        dtype, width, storage_width, pad_literal = vector_info
        components = [
            self.gpu_builtin_component_expression(alias_name, index, dtype)
            for index in range(width)
        ]
        while len(components) < storage_width:
            components.append(pad_literal)
        return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

    def generate_gpu_builtin_swizzle(self, alias_name, type_name, swizzle_indices):
        vector_info = self.vector_type_info(type_name)
        dtype = vector_info[0] if vector_info is not None else "DType.uint32"
        if len(swizzle_indices) == 1:
            return self.gpu_builtin_component_expression(
                alias_name, swizzle_indices[0], dtype
            )

        result_type = self.swizzle_result_type(type_name, len(swizzle_indices))
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return alias_name
        result_dtype, _, storage_width, pad_literal = result_info
        components = [
            self.gpu_builtin_component_expression(alias_name, index, result_dtype)
            for index in swizzle_indices
        ]
        while len(components) < storage_width:
            components.append(pad_literal)
        return f"SIMD[{result_dtype}, {storage_width}]({', '.join(components)})"

    def gpu_builtin_component_expression(self, alias_name, index, dtype):
        component_names = ("x", "y", "z")
        if 0 <= index < len(component_names):
            if alias_name in self.variable_types:
                return f"{self.mojo_identifier(alias_name)}[{index}]"
            return f"{alias_name}.{component_names[index]}"
        return MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.uint32"])[2]

    def generate_builtin_member_access(self, expr):
        obj_name = self.identifier_expression_name(getattr(expr, "object", None))
        if obj_name is None:
            return None
        gpu_builtin = self.mojo_gpu_builtin_expression(
            obj_name, getattr(expr, "member", None)
        )
        if gpu_builtin is not None:
            return gpu_builtin
        obj = self.mojo_builtin_placeholder_expression(obj_name)
        if obj is None:
            return None
        member = getattr(expr, "member", None)
        swizzle_indices = self.get_swizzle_indices(member)
        if swizzle_indices is None:
            return f"{obj}.{member}"
        obj_type = MOJO_BUILTIN_PLACEHOLDERS[obj_name][1]
        return self.generate_swizzle(expr, obj, obj_type, member, swizzle_indices)

    def identifier_expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return getattr(expr, "name", None)
        if isinstance(expr, BuiltinVariableNode):
            return getattr(expr, "builtin_name", None)
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return getattr(expr, "name", None)
        return None

    def mojo_ray_builtin_expression(self, name):
        helper = {
            "gl_LaunchIDEXT": "ray_launch_id",
            "gl_LaunchSizeEXT": "ray_launch_size",
            "gl_HitTEXT": "ray_hit_t",
            "gl_HitKindEXT": "ray_hit_kind",
            "gl_WorldRayOriginEXT": "ray_world_origin",
            "gl_WorldRayDirectionEXT": "ray_world_direction",
            "gl_ObjectRayOriginEXT": "ray_object_origin",
            "gl_ObjectRayDirectionEXT": "ray_object_direction",
            "gl_RayTminEXT": "ray_t_min",
            "gl_IncomingRayFlagsEXT": "ray_incoming_flags",
            "gl_InstanceCustomIndexEXT": "ray_instance_custom_index",
            "gl_GeometryIndexEXT": "ray_geometry_index",
        }.get(str(name))
        if helper is None:
            return None
        self.required_ray_helpers.add(helper)
        return f"{helper}()"

    def generate_texture_node(self, expr):
        args = [expr.texture_expr]
        if expr.sampler_expr is not None:
            args.append(expr.sampler_expr)
        args.append(expr.coordinates)

        if expr.level is not None and expr.offset is not None:
            return self.generate_resource_builtin_call(
                [*args, expr.level, expr.offset], "sample_lod_offset", "vec4"
            )
        if expr.level is not None:
            return self.generate_texture_call([*args, expr.level], "sample_lod")
        if expr.offset is not None:
            return self.generate_resource_builtin_call(
                [*args, expr.offset], "sample_offset", "vec4"
            )
        return self.generate_texture_call(args, "sample")

    def texture_op_args(self, expr, include_sampler=True):
        args = [expr.texture_expr]
        if include_sampler and expr.sampler_expr is not None:
            args.append(expr.sampler_expr)
        args.extend(expr.arguments)
        return args
