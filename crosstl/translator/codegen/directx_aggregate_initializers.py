"""Aggregate and struct initializers in the HLSL generator: render targets, array constructors, typed buffer atomics, literals, fixed arrays, conditionals and field assignments."""

import re
from copy import deepcopy

from ..ast import (
    ArrayLiteralNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    TernaryOpNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import DirectXAggregateInitializerError
from .enum_utils import default_value_expression
from .generic_function_utils import generic_function_call_name
from .generic_struct_utils import (
    format_struct_constructor_expression,
    infer_struct_constructor_type,
)
from .image_access_contracts import is_image_atomic_operation


class DirectXAggregateInitializerMixin:
    """Aggregate initializer members of ``HLSLCodeGen``."""

    def materialize_hlsl_aggregate_conditional(
        self, expr, expected_type, indent, *, context
    ):
        aggregate_type = self.hlsl_aggregate_conditional_type(expr, expected_type)
        if aggregate_type is None:
            return None
        temp_name = self.next_hlsl_temp_variable("aggregate_conditional")
        self.local_variable_types[temp_name] = aggregate_type
        declaration = self.hlsl_aggregate_conditional_declaration(
            aggregate_type, temp_name
        )
        indent_str = "    " * indent
        code = f"{indent_str}{declaration};\n"
        code += self.generate_hlsl_aggregate_conditional_assignment(
            expr,
            temp_name,
            "=",
            aggregate_type,
            indent,
            context=context,
        )
        return code, temp_name

    def render_hlsl_aggregate_conditional_value_expression(
        self, expr, expected_type, indent, *, context
    ):
        constructor_type = self.hlsl_effectful_struct_constructor_type(expr)
        if constructor_type is not None:
            if context in {"initializer", "return"}:
                # These statement paths already initialize fields in source order.
                return None
            initializer = self.hlsl_struct_initializer_components(
                constructor_type, expr
            )
            if initializer is None:
                self.hlsl_aggregate_error(
                    expr, expected_type=constructor_type, reason="field-layout-unproven"
                )
            type_name, fields, rendered_args, field_exprs = initializer
            temporary = self.next_hlsl_temp_variable("struct_value")
            self.local_variable_types[temporary] = type_name
            code = f"{'    ' * indent}{type_name} {temporary};\n"
            code += self.render_hlsl_struct_field_assignments(
                temporary, fields, rendered_args, field_exprs, indent
            )
            return code, temporary

        materialized = self.materialize_hlsl_aggregate_conditional(
            expr, expected_type, indent, context=context
        )
        if materialized is not None:
            return materialized

        if not (
            isinstance(expr, FunctionCallNode)
            or (hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__))
        ):
            return None

        func_name = self.function_call_name(expr)
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        parameter_types = self.function_parameter_types.get(func_name) or []
        workgroup_pointer_func_name = func_name
        workgroup_pointer_indices = (
            self.function_hlsl_workgroup_pointer_parameter_indices.get(
                workgroup_pointer_func_name, {}
            )
        )
        if not workgroup_pointer_indices:
            materialized_name = self.hlsl_materialized_function_name(func_name)
            materialized_indices = (
                self.function_hlsl_workgroup_pointer_parameter_indices.get(
                    materialized_name, {}
                )
            )
            if materialized_indices:
                workgroup_pointer_func_name = materialized_name
                workgroup_pointer_indices = materialized_indices
        aggregate_arguments = {}
        for index, arg in enumerate(args):
            if index in workgroup_pointer_indices:
                continue
            argument_type = (
                parameter_types[index] if index < len(parameter_types) else None
            )
            conditional, aggregate_type = self.hlsl_first_aggregate_conditional(
                arg, argument_type
            )
            if conditional is not None:
                aggregate_arguments[index] = (conditional, aggregate_type)

        if not aggregate_arguments:
            return None

        first_index = min(aggregate_arguments)
        first_conditional, first_aggregate_type = aggregate_arguments[first_index]
        for index, arg in enumerate(args):
            if index in aggregate_arguments:
                continue
            if self.hlsl_expression_has_observable_side_effects(arg):
                self.hlsl_aggregate_conditional_error(
                    first_conditional,
                    aggregate_type=first_aggregate_type,
                    context="call-argument",
                    reason="sibling-evaluation-order",
                    detail=(
                        "materializing the aggregate argument would reorder another "
                        "side-effecting argument"
                    ),
                )

        rendered_args = []
        argument_code = ""
        changed_indices = set()

        for index, arg in enumerate(args):
            if index in workgroup_pointer_indices:
                rendered_args.append("")
                continue
            argument_type = (
                parameter_types[index] if index < len(parameter_types) else None
            )
            rendered_argument = self.render_hlsl_aggregate_conditional_value_expression(
                arg,
                argument_type,
                indent,
                context="call-argument",
            )
            if rendered_argument is None:
                rendered_args.append(
                    self.generate_expression_with_expected(arg, argument_type)
                )
                continue

            code, rendered = rendered_argument
            argument_code += code
            rendered_args.append(rendered)
            changed_indices.add(index)

        if not changed_indices:
            return None

        if not isinstance(func_name, str) or func_name not in getattr(
            self, "function_return_types", {}
        ):
            if (
                func_name == "buffer_store"
                and len(args) == 3
                and changed_indices == {2}
            ):
                value_type = self.expression_result_type(args[2])
                if value_type is not None:
                    temporary = self.next_hlsl_temp_variable("store_value")
                    self.local_variable_types[temporary] = value_type
                    argument_code += (
                        f"{'    ' * indent}{self.map_type(value_type)} {temporary} = "
                        f"{rendered_args[2]};\n"
                    )
                    rewritten = deepcopy(expr)
                    rewritten.arguments = [*args[:2], IdentifierNode(temporary)]
                    rewritten.args = rewritten.arguments
                    return argument_code, self.generate_expression(rewritten)
            self.hlsl_aggregate_conditional_error(
                first_conditional,
                aggregate_type=first_aggregate_type,
                context="call-argument",
                reason="unsupported-call-target",
                detail="the callee does not have a materializable user-function contract",
            )

        specialized_func_name = generic_function_call_name(self, func_name, args)
        callee = specialized_func_name or self.hlsl_function_call_name(func_name)
        workgroup_callee = self.hlsl_workgroup_pointer_call_name(func_name, args)
        if workgroup_callee is not None:
            callee = workgroup_callee
        rendered_call_args = self.generate_call_arguments_from_rendered(
            func_name,
            args,
            rendered_args,
            private_pointer_func_name=func_name,
            workgroup_pointer_func_name=workgroup_pointer_func_name,
        )
        return argument_code, f"{callee}({', '.join(rendered_call_args)})"

    def hlsl_expression_contains_typed_buffer_atomic(self, expr):
        for node in self.walk_ast(expr):
            if not (
                isinstance(node, FunctionCallNode)
                or (
                    hasattr(node, "__class__") and "FunctionCall" in str(node.__class__)
                )
            ):
                continue
            func_name = self.function_call_name(node)
            if not isinstance(func_name, str):
                continue
            args = getattr(node, "arguments", getattr(node, "args", []))
            if (
                self.hlsl_typed_buffer_atomic_parts(
                    func_name, args, render_operands=False
                )
                is not None
            ):
                return True
        return False

    def hlsl_expression_contains_image_atomic(self, expr):
        for node in self.walk_ast(expr):
            if not (
                isinstance(node, FunctionCallNode)
                or (
                    hasattr(node, "__class__") and "FunctionCall" in str(node.__class__)
                )
            ):
                continue
            func_name = self.function_call_name(node)
            if isinstance(func_name, str) and is_image_atomic_operation(func_name):
                return True
        return False

    def hlsl_typed_buffer_atomic_ternary_expression(self, expr):
        if not (
            isinstance(expr, TernaryOpNode)
            or (hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__))
        ):
            return False
        return self.hlsl_expression_contains_typed_buffer_atomic(expr)

    def render_hlsl_typed_buffer_atomic_call_value(
        self, parts, indent, expected_type=None
    ):
        self.validate_hlsl_typed_buffer_atomic_result_context(parts, expected_type)
        original_arg = parts["original_arg"]
        indent_str = "    " * indent
        code = self.generate_statement_code(parts["code"], indent)
        if original_arg is not None:
            original = self.generate_expression(original_arg)
        else:
            original = self.next_hlsl_temp_variable("atomic_expr")
            code += f"{indent_str}{self.map_type(parts['target_type'])} {original};\n"

        call_args = [*parts["target_args"], *parts["values"], original]
        code += f"{indent_str}{parts['intrinsic']}({', '.join(call_args)});\n"
        return code, original

    def hlsl_struct_constructor_fields(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        member_types = self.struct_member_types.get(type_name)
        if member_types is None:
            mapped_type = self.map_type(type_name)
            member_types = self.struct_member_types.get(mapped_type)
        if member_types is None:
            return None
        return list(member_types.items())

    def hlsl_struct_initializer_rendered_args(
        self,
        type_name,
        args,
        named_args=None,
        initializer_expr=None,
    ):
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None:
            return None

        positional_args = list(args or [])
        named_args = dict(named_args or {})
        field_names = [field_name for field_name, _field_type in fields]
        if len(positional_args) > len(fields):
            return None
        if set(named_args) - set(field_names):
            return None

        rendered_args = []
        field_exprs = []
        for index, (field_name, field_type) in enumerate(fields):
            if index < len(positional_args):
                field_expr = positional_args[index]
            elif field_name in named_args:
                field_expr = named_args[field_name]
            else:
                field_exprs.append(None)
                mapped_field_type = self.map_type(field_type)
                _base_type, array_suffix = split_array_type_suffix(mapped_field_type)
                if (
                    array_suffix
                    or self.hlsl_struct_constructor_fields(mapped_field_type)
                    is not None
                ):
                    rendered_args.append(
                        self.hlsl_value_initialized_expression(
                            initializer_expr,
                            mapped_field_type,
                        )
                    )
                else:
                    # ``float3(0)`` is no HLSL constructor; the default spells every component.
                    rendered_args.append(self.hlsl_default_value_expression(field_type))
                continue
            field_exprs.append(field_expr)
            rendered_args.append(
                self.generate_expression_with_expected(field_expr, field_type)
            )
        return self.map_type(type_name), fields, rendered_args, field_exprs

    def hlsl_struct_initializer_components(self, expected_type, expr):
        private_scalar_struct = self.hlsl_private_scalar_struct_reinterpret_initializer(
            expected_type,
            expr,
        )
        if private_scalar_struct is not None:
            return private_scalar_struct
        expected_type = self.type_name_string(expected_type)
        if isinstance(expr, ArrayLiteralNode):
            return self.hlsl_struct_initializer_rendered_args(
                expected_type,
                getattr(expr, "elements", []) or [],
                initializer_expr=expr,
            )

        if isinstance(expr, ConstructorNode):
            constructor_type = (
                infer_struct_constructor_type(self, expr)
                or self.type_name_string(getattr(expr, "constructor_type", None))
                or expected_type
            )
            return self.hlsl_struct_initializer_rendered_args(
                constructor_type,
                getattr(expr, "arguments", []) or [],
                getattr(expr, "named_arguments", {}) or {},
                initializer_expr=expr,
            )

        if isinstance(expr, FunctionCallNode) or (
            hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__)
        ):
            func_name = self.function_call_name(expr)
            if func_name in getattr(self, "function_return_types", {}):
                return None
            return self.hlsl_struct_initializer_rendered_args(
                func_name,
                getattr(expr, "arguments", getattr(expr, "args", [])) or [],
                getattr(expr, "named_arguments", {}) or {},
                initializer_expr=expr,
            )

        return None

    def render_hlsl_struct_value_initialization(
        self, target_name, expected_type, expr, indent=0
    ):
        initializer = self.hlsl_struct_initializer_components(expected_type, expr)
        if initializer is None:
            return None
        _type_name, fields, rendered_args, field_exprs = initializer
        return self.render_hlsl_struct_field_assignments(
            target_name, fields, rendered_args, field_exprs, indent
        )

    def render_hlsl_struct_return(self, expr, indent=0):
        return_type = self.type_name_string(self.current_function_return_type)
        initializer = self.hlsl_struct_initializer_components(return_type, expr)
        if initializer is None:
            return None
        type_name, fields, rendered_args, field_exprs = initializer
        temp_name = self.next_hlsl_temp_variable("struct_return")
        self.local_variable_types[temp_name] = type_name
        indent_str = "    " * indent
        code = f"{indent_str}{type_name} {temp_name};\n"
        code += self.render_hlsl_struct_field_assignments(
            temp_name, fields, rendered_args, field_exprs, indent
        )
        code += self.generate_hlsl_return_value_statement(temp_name, indent)
        return code

    def render_hlsl_struct_field_assignments(
        self, target_name, fields, rendered_args, field_exprs, indent=0
    ):
        indent_str = "    " * indent
        code = ""
        for (field_name, field_type), rendered_arg, field_expr in zip(
            fields, rendered_args, field_exprs
        ):
            field_target = f"{target_name}.{field_name}"
            field_type_name = self.type_name_string(field_type)
            if field_expr is not None and self.hlsl_outer_array_type(field_type_name):
                code += self.render_hlsl_aggregate_conditional_branch_assignment(
                    field_expr,
                    field_target,
                    field_type_name,
                    indent,
                    context="struct-member-initializer",
                )
                continue
            if (
                field_expr is not None
                and self.hlsl_struct_constructor_fields(field_type_name) is not None
            ):
                nested_init = self.render_hlsl_struct_value_initialization(
                    field_target, field_type_name, field_expr, indent
                )
                if nested_init is not None:
                    code += nested_init
                    continue
            code += f"{indent_str}{field_target} = {rendered_arg};\n"
        return code

    def render_hlsl_default_struct_field_assignments(
        self, target_name, type_name, indent=0, seen_structs=None
    ):
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None:
            return None

        type_name = self.type_name_string(type_name)
        seen_structs = set(seen_structs or set())
        if type_name in seen_structs:
            return None

        indent_str = "    " * indent
        code = ""
        for field_name, field_type in fields:
            field_target = f"{target_name}.{field_name}"
            field_type_name = self.type_name_string(field_type)
            nested_fields = self.hlsl_struct_constructor_fields(field_type_name)
            if nested_fields is not None:
                nested_code = self.render_hlsl_default_struct_field_assignments(
                    field_target,
                    field_type_name,
                    indent,
                    seen_structs | {type_name},
                )
                if nested_code is not None:
                    code += nested_code
                    continue
            code += (
                f"{indent_str}{field_target} = "
                f"{self.hlsl_default_value_expression(field_type)};\n"
            )
        return code

    def render_hlsl_default_struct_return(self, return_type, indent=0):
        type_name = self.type_name_string(return_type)
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None:
            return None

        mapped_type = self.map_type(type_name)
        temp_name = self.next_hlsl_temp_variable("struct_return")
        self.local_variable_types[temp_name] = mapped_type
        indent_str = "    " * indent
        code = f"{indent_str}{mapped_type} {temp_name};\n"
        code += self.render_hlsl_default_struct_field_assignments(
            temp_name, type_name, indent
        )
        code += self.generate_hlsl_return_value_statement(temp_name, indent)
        return code

    def hlsl_effectful_struct_constructor_type(self, expr):
        if isinstance(expr, ConstructorNode):
            type_name = infer_struct_constructor_type(self, expr)
        elif isinstance(expr, FunctionCallNode):
            type_name = self.function_call_name(expr)
            if type_name in self.function_return_types:
                return None
        else:
            return None
        if self.hlsl_struct_constructor_fields(type_name) is None:
            return None
        if self.hlsl_expression_contains_typed_buffer_atomic(expr):
            return None
        arguments = list(getattr(expr, "arguments", []) or [])
        arguments.extend((getattr(expr, "named_arguments", {}) or {}).values())
        if any(
            self.hlsl_expression_has_observable_side_effects(
                arg, allow_numeric_constructors=True
            )
            for arg in arguments
        ):
            return self.map_type(type_name)
        return None

    def validate_hlsl_struct_constructor_expression_context(self, expr):
        type_name = self.hlsl_effectful_struct_constructor_type(expr)
        if type_name is not None:
            self.hlsl_aggregate_error(
                expr,
                expected_type=type_name,
                reason="field-evaluation-order-unproven",
                detail="effectful constructor fields require ordered statement materialization in this context",
            )

    def format_struct_constructor_expression(self, type_name, rendered_args):
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None or len(fields) != len(rendered_args):
            raise DirectXRefusal(
                f"Struct constructor {type_name} has an unproven field layout"
            )
        if not fields:
            return f"({type_name})0"
        if type_name not in self.hlsl_struct_constructor_names:
            base = "__crossgl_construct_" + re.sub(r"\W", "_", type_name)
            name, suffix = base, 0
            while name in self.hlsl_struct_constructor_reserved_names:
                suffix += 1
                name = f"{base}_{suffix}"
            self.hlsl_struct_constructor_reserved_names.add(name)
            self.hlsl_struct_constructor_names[type_name] = name
        name = self.hlsl_struct_constructor_names[type_name]
        return f"{name}({', '.join(rendered_args)})"

    def generate_hlsl_struct_constructor_helpers(self):
        code = ""
        for type_name, name in sorted(self.hlsl_struct_constructor_names.items()):
            fields = self.hlsl_struct_constructor_fields(type_name)
            parameters = [
                format_c_style_array_declaration(
                    self.map_type(field_type), f"field_{index}"
                )
                for index, (_field_name, field_type) in enumerate(fields)
            ]
            code += f"{type_name} {name}({', '.join(parameters)}) {{\n"
            code += f"    {type_name} value;\n"
            for index, (field_name, _field_type) in enumerate(fields):
                code += f"    value.{field_name} = field_{index};\n"
            code += "    return value;\n}\n\n"
        return code

    def generate_hlsl_struct_constructor_call(self, type_name, args, named_args=None):
        if type_name in getattr(self, "function_return_types", {}):
            return None
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None:
            return None

        positional_args = list(args or [])
        named_args = dict(named_args or {})
        field_names = [field_name for field_name, _field_type in fields]
        if len(positional_args) > len(fields):
            raise DirectXRefusal(
                f"Struct constructor {type_name} expects at most {len(fields)} "
                f"arguments, got {len(positional_args)}"
            )
        unknown_names = sorted(set(named_args) - set(field_names))
        if unknown_names:
            raise DirectXRefusal(
                f"Struct constructor {type_name} has no field "
                f"{', '.join(unknown_names)}"
            )

        if not fields:
            return f"({self.map_type(type_name)})0"
        rendered_args = []
        for index, (field_name, field_type) in enumerate(fields):
            if index < len(positional_args):
                argument = positional_args[index]
                array_value = self.hlsl_array_value_arguments.argument(
                    argument, field_type
                )
                if array_value is not None:
                    rendered_args.append(array_value)
                    continue
                if isinstance(
                    argument, ArrayLiteralNode
                ) and self.hlsl_outer_array_type(field_type):
                    # The helper call cannot take an array literal; only a
                    # declaration initializer can use an initializer list.
                    self.hlsl_aggregate_error(
                        argument,
                        expected_type=self.map_type(field_type),
                        reason="array-argument-unsupported",
                        detail=(
                            "HLSL has no fixed-array literal expression for a "
                            f"{type_name} constructor field outside a declaration "
                            "initializer"
                        ),
                    )
                rendered_args.append(
                    self.generate_expression_with_expected(argument, field_type)
                )
                continue
            if field_name in named_args:
                rendered_args.append(
                    self.generate_expression_with_expected(
                        named_args[field_name],
                        field_type,
                    )
                )
                continue
            rendered_args.append(default_value_expression(self, field_type))

        return format_struct_constructor_expression(
            self,
            self.map_type(type_name),
            rendered_args,
        )

    def render_hlsl_typed_buffer_atomic_struct_constructor(
        self, type_name, args, named_args, indent
    ):
        fields = self.hlsl_struct_constructor_fields(type_name)
        if fields is None:
            return None

        positional_args = list(args or [])
        named_args = dict(named_args or {})
        field_names = [field_name for field_name, _field_type in fields]
        if len(positional_args) > len(fields):
            raise DirectXRefusal(
                f"Struct constructor {type_name} expects at most {len(fields)} "
                f"arguments, got {len(positional_args)}"
            )
        unknown_names = sorted(set(named_args) - set(field_names))
        if unknown_names:
            raise DirectXRefusal(
                f"Struct constructor {type_name} has no field "
                f"{', '.join(unknown_names)}"
            )

        code = ""
        rendered_args = []
        changed = False
        for index, (field_name, field_type) in enumerate(fields):
            if index < len(positional_args):
                field_expr = positional_args[index]
            elif field_name in named_args:
                field_expr = named_args[field_name]
            else:
                rendered_args.append(default_value_expression(self, field_type))
                continue

            field_code, rendered_field = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    field_expr, field_type, indent
                )
            )
            code += field_code
            rendered_args.append(rendered_field)
            changed = changed or bool(field_code)

        if not changed:
            return None

        mapped_type = self.map_type(type_name)
        return (
            code,
            format_struct_constructor_expression(self, mapped_type, rendered_args),
        )

    def hlsl_aggregate_error(
        self,
        expr,
        *,
        expected_type,
        reason,
        expected_element_count=None,
        detail=None,
    ):
        element_count = len(getattr(expr, "elements", []) or [])
        if reason == "expected-type-unresolved":
            message = (
                "DirectX aggregate initializer requires a concrete destination type"
            )
        elif reason == "element-count-mismatch":
            message = (
                f"DirectX aggregate initializer for '{expected_type}' expects at "
                f"most {expected_element_count} elements, got {element_count}"
            )
        elif detail:
            message = (
                f"DirectX aggregate initializer for '{expected_type}' cannot be "
                f"lowered: {detail}"
            )
        else:
            message = (
                f"DirectX aggregate initializer for '{expected_type}' cannot be lowered"
            )
        raise DirectXAggregateInitializerError(
            message,
            expected_type=expected_type,
            element_count=element_count,
            expected_element_count=expected_element_count,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def hlsl_outer_array_type(self, expected_type):
        type_name = self.type_name_string(expected_type)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        if not array_suffix:
            return None
        match = re.fullmatch(r"\[([^\]]*)\](.*)", array_suffix)
        if match is None:
            return None
        extent_text = match.group(1).strip()
        return mapped_type, extent_text, f"{base_type}{match.group(2)}"

    def hlsl_array_literal_element_expected_type(self, expected_type):
        array_type = self.hlsl_outer_array_type(expected_type)
        return array_type[2] if array_type is not None else None

    def hlsl_fixed_array_aggregate_contract(self, expr, expected_type):
        array_type = self.hlsl_outer_array_type(expected_type)
        mapped_type = self.map_type(expected_type)
        if array_type is None:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-type-invalid",
                detail="the array suffix is invalid",
            )

        mapped_type, extent_text, element_type = array_type
        if not extent_text:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="unsized-array",
                detail="an unsized array has no fixed initializer extent",
            )
        extent = evaluate_literal_int_expression(
            extent_text,
            self.hlsl_current_visible_int_constants(),
        )
        if extent is None:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-extent-unresolved",
                detail=f"array extent '{extent_text}' is not a known integer constant",
            )
        if extent <= 0:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-extent-invalid",
                detail=f"array extent '{extent}' must be positive",
            )
        return mapped_type, extent, element_type

    def hlsl_is_zero_initializer_expression(self, expr):
        if isinstance(expr, bool):
            return False
        if isinstance(expr, (int, float)):
            return expr in {0, 0.0}
        if not (hasattr(expr, "__class__") and "Literal" in str(expr.__class__)):
            return False
        value = getattr(expr, "value", None)
        return not isinstance(value, bool) and value in {0, 0.0}

    def hlsl_is_zero_aggregate_initializer(self, expr):
        elements = list(getattr(expr, "elements", []) or [])
        return len(elements) == 1 and self.hlsl_is_zero_initializer_expression(
            elements[0]
        )

    def hlsl_value_initialized_expression(self, expr, expected_type):
        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix:
            _array_type, extent, element_type = (
                self.hlsl_fixed_array_aggregate_contract(expr, mapped_type)
            )
            elements = [
                self.hlsl_value_initialized_expression(expr, element_type)
                for _index in range(extent)
            ]
            return "{" + ", ".join(elements) + "}"

        struct_fields = self.hlsl_struct_constructor_fields(mapped_type)
        if struct_fields is not None:
            for _field_name, field_type in struct_fields:
                self.hlsl_value_initialized_expression(expr, field_type)
            return f"({mapped_type})0"

        if self.is_scalar_value_type(mapped_type):
            return default_value_expression(self, mapped_type)
        if (
            self.is_vector_value_type(mapped_type)
            or self.hlsl_matrix_shape(mapped_type) is not None
        ):
            return f"({mapped_type})0"

        self.hlsl_aggregate_error(
            expr,
            expected_type=mapped_type,
            reason="value-initialization-unsupported",
            detail="the omitted element type has no HLSL value initializer",
        )

    def hlsl_contextual_initializer_element_expression(
        self, element, expected_type, aggregate_expr
    ):
        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        is_aggregate = bool(
            array_suffix
            or self.hlsl_struct_constructor_fields(mapped_type) is not None
            or self.is_vector_value_type(mapped_type)
            or self.hlsl_matrix_shape(mapped_type) is not None
        )
        if isinstance(element, ArrayLiteralNode):
            return self.hlsl_contextual_aggregate_initializer(element, mapped_type)
        if is_aggregate and self.hlsl_is_zero_initializer_expression(element):
            return self.hlsl_value_initialized_expression(aggregate_expr, mapped_type)
        if is_aggregate:
            source_type = self.expression_result_type(element)
            if source_type is None or self.map_type(source_type) != mapped_type:
                self.hlsl_aggregate_error(
                    aggregate_expr,
                    expected_type=mapped_type,
                    reason="aggregate-element-shape-mismatch",
                    detail=(
                        "a nonzero scalar cannot initialize an aggregate element "
                        "without explicit nested braces or a matching typed expression"
                    ),
                )
        return self.generate_expression_with_expected(element, expected_type)

    def hlsl_vector_aggregate_initializer(self, expr, mapped_type):
        width = self.value_component_count(mapped_type)
        component_type = self.vector_component_type(mapped_type)
        if not width or width <= 1 or component_type is None:
            return None
        if self.hlsl_is_zero_aggregate_initializer(expr):
            return self.hlsl_value_initialized_expression(expr, mapped_type)

        elements = list(getattr(expr, "elements", []) or [])
        rendered = []
        provided_components = 0
        for element in elements:
            component_count = self.expression_component_count(element)
            if component_count is None:
                self.hlsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="vector-element-type-unresolved",
                    detail="a vector aggregate element has no known component width",
                )
            if provided_components + component_count > width:
                self.hlsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=width,
                )
            rendered.append(
                self.generate_expression_with_expected(
                    element,
                    component_type if component_count == 1 else None,
                )
            )
            provided_components += component_count
        rendered.extend(
            self.hlsl_value_initialized_expression(expr, component_type)
            for _index in range(provided_components, width)
        )
        return f"{mapped_type}({', '.join(rendered)})"

    def hlsl_struct_aggregate_initializer(self, expr, mapped_type):
        fields = self.hlsl_struct_constructor_fields(mapped_type)
        if fields is None:
            return None
        if self.hlsl_is_zero_aggregate_initializer(expr):
            return self.hlsl_value_initialized_expression(expr, mapped_type)

        elements = list(getattr(expr, "elements", []) or [])
        if len(elements) > len(fields):
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=len(fields),
            )
        if not fields:
            return f"({mapped_type})0"
        rendered = [
            self.hlsl_contextual_initializer_element_expression(
                element,
                fields[index][1],
                expr,
            )
            for index, element in enumerate(elements)
        ]
        rendered.extend(
            self.hlsl_value_initialized_expression(expr, fields[index][1])
            for index in range(len(elements), len(fields))
        )
        return "{" + ", ".join(rendered) + "}"

    def hlsl_fixed_array_aggregate_initializer(self, expr, expected_type):
        mapped_type, extent, element_type = self.hlsl_fixed_array_aggregate_contract(
            expr, expected_type
        )
        elements = list(getattr(expr, "elements", []) or [])
        if self.hlsl_is_zero_aggregate_initializer(expr):
            elements = []
        if len(elements) > extent:
            self.hlsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=extent,
            )
        rendered = [
            self.hlsl_contextual_initializer_element_expression(
                element,
                element_type,
                expr,
            )
            for element in elements
        ]
        rendered.extend(
            self.hlsl_value_initialized_expression(expr, element_type)
            for _index in range(len(elements), extent)
        )
        return "{" + ", ".join(rendered) + "}"

    def hlsl_contextual_aggregate_initializer(self, expr, expected_type):
        expected_type = self.type_name_string(expected_type)
        if not expected_type:
            self.hlsl_aggregate_error(
                expr,
                expected_type=None,
                reason="expected-type-unresolved",
            )
        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix:
            return self.hlsl_fixed_array_aggregate_initializer(expr, mapped_type)

        struct_initializer = self.hlsl_struct_aggregate_initializer(expr, mapped_type)
        if struct_initializer is not None:
            return struct_initializer
        vector_initializer = self.hlsl_vector_aggregate_initializer(expr, mapped_type)
        if vector_initializer is not None:
            return vector_initializer
        matrix_initializer = self.hlsl_matrix_aggregate_initializer(expr, mapped_type)
        if matrix_initializer is not None:
            return matrix_initializer

        self.hlsl_aggregate_error(
            expr,
            expected_type=mapped_type,
            reason="non-aggregate-destination-type",
            detail="the destination is not a struct, fixed array, vector, or matrix",
        )

    def hlsl_repeatable_array_assignment_target(self, target):
        if isinstance(target, str):
            return target.isidentifier()
        if isinstance(target, (IdentifierNode, VariableNode)):
            return True
        if isinstance(target, MemberAccessNode) or (
            hasattr(target, "__class__") and "MemberAccess" in str(target.__class__)
        ):
            owner = getattr(target, "object_expr", getattr(target, "object", None))
            member = getattr(target, "member", None)
            return bool(
                isinstance(member, str)
                and member.isidentifier()
                and self.hlsl_repeatable_array_assignment_target(owner)
            )
        return False

    def hlsl_aggregate_assignment_statements(
        self,
        target,
        initializer,
        expected_type,
        root_expr,
        *,
        value_initialize=False,
    ):
        mapped_type = self.map_type(expected_type)
        array_type = self.hlsl_outer_array_type(mapped_type)
        if array_type is not None:
            contract_expr = (
                initializer if isinstance(initializer, ArrayLiteralNode) else root_expr
            )
            _array_type, extent, element_type = (
                self.hlsl_fixed_array_aggregate_contract(contract_expr, mapped_type)
            )
            if value_initialize or self.hlsl_is_zero_initializer_expression(
                initializer
            ):
                elements = []
            elif isinstance(initializer, ArrayLiteralNode):
                elements = list(getattr(initializer, "elements", []) or [])
                if self.hlsl_is_zero_aggregate_initializer(initializer):
                    elements = []
            else:
                self.hlsl_aggregate_error(
                    root_expr,
                    expected_type=mapped_type,
                    reason="array-assignment-unsupported",
                    detail=(
                        "HLSL fixed arrays require an aggregate literal for "
                        "element-wise assignment"
                    ),
                )
            if len(elements) > extent:
                self.hlsl_aggregate_error(
                    contract_expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=extent,
                )
            statements = []
            for index in range(extent):
                child_initializer = elements[index] if index < len(elements) else None
                statements.extend(
                    self.hlsl_aggregate_assignment_statements(
                        f"{target}[{index}]",
                        child_initializer,
                        element_type,
                        root_expr,
                        value_initialize=index >= len(elements),
                    )
                )
            return statements

        struct_fields = self.hlsl_struct_constructor_fields(mapped_type)
        if struct_fields is not None:
            if (
                value_initialize
                or self.hlsl_is_zero_initializer_expression(initializer)
                or (
                    isinstance(initializer, ArrayLiteralNode)
                    and self.hlsl_is_zero_aggregate_initializer(initializer)
                )
            ):
                elements = []
            elif isinstance(initializer, ArrayLiteralNode):
                elements = list(getattr(initializer, "elements", []) or [])
            else:
                rendered = self.generate_expression_with_expected(
                    initializer, mapped_type
                )
                return [f"{target} = {rendered}"]
            if len(elements) > len(struct_fields):
                self.hlsl_aggregate_error(
                    initializer,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=len(struct_fields),
                )
            statements = []
            for index, (field_name, field_type) in enumerate(struct_fields):
                child_initializer = elements[index] if index < len(elements) else None
                statements.extend(
                    self.hlsl_aggregate_assignment_statements(
                        f"{target}.{field_name}",
                        child_initializer,
                        field_type,
                        root_expr,
                        value_initialize=index >= len(elements),
                    )
                )
            return statements

        if value_initialize:
            rendered = self.hlsl_value_initialized_expression(root_expr, mapped_type)
        elif isinstance(initializer, ArrayLiteralNode):
            rendered = self.hlsl_contextual_aggregate_initializer(
                initializer, mapped_type
            )
        elif (
            self.is_vector_value_type(mapped_type)
            or self.hlsl_matrix_shape(mapped_type) is not None
        ) and self.hlsl_is_zero_initializer_expression(initializer):
            rendered = self.hlsl_value_initialized_expression(root_expr, mapped_type)
        else:
            rendered = self.generate_expression_with_expected(initializer, mapped_type)
        return [f"{target} = {rendered}"]

    def render_hlsl_fixed_array_literal_call_statement(self, expr, indent=0):
        if not (
            isinstance(expr, FunctionCallNode)
            or (hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__))
        ):
            return None

        func_name = self.function_call_name(expr)
        parameter_types = self.function_parameter_types.get(func_name) or []
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        rewritten_args = []
        prelude = ""
        changed = False
        indent_str = "    " * indent
        for index, arg in enumerate(args):
            expected_type = (
                parameter_types[index] if index < len(parameter_types) else None
            )
            if not (
                isinstance(arg, ArrayLiteralNode)
                and self.hlsl_outer_array_type(expected_type)
            ):
                rewritten_args.append(arg)
                continue

            initializer = self.hlsl_fixed_array_aggregate_initializer(
                arg, expected_type
            )
            temp_name = self.next_hlsl_temp_variable("array_arg")
            mapped_type = self.map_type(expected_type)
            declaration = format_c_style_array_declaration(mapped_type, temp_name)
            self.local_variable_types[temp_name] = self.type_name_string(expected_type)
            prelude += f"{indent_str}{declaration} = {initializer};\n"
            rewritten_args.append(IdentifierNode(temp_name))
            changed = True

        if not changed:
            return None

        rewritten_call = deepcopy(expr)
        rewritten_call.arguments = rewritten_args
        if hasattr(rewritten_call, "args"):
            rewritten_call.args = rewritten_args
        return f"{prelude}{indent_str}{self.generate_expression(rewritten_call)};\n"

    def hlsl_array_literal_expression(self, expr, expected_type=None):
        if expected_type is None:
            expected_type = self.current_expression_expected_type
        if expected_type is None:
            self.hlsl_aggregate_error(
                expr,
                expected_type=None,
                reason="expected-type-unresolved",
            )
        mapped_type = self.map_type(expected_type) if expected_type else None
        _base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return self.hlsl_fixed_array_aggregate_initializer(expr, mapped_type)
        struct_initializer = self.hlsl_struct_initializer_components(
            expected_type,
            expr,
        )
        if struct_initializer is not None:
            type_name, _fields, rendered_args, _field_exprs = struct_initializer
            if not _fields:
                return f"({type_name})0"
            return format_struct_constructor_expression(self, type_name, rendered_args)
        vector_initializer = self.hlsl_vector_aggregate_initializer(expr, mapped_type)
        if vector_initializer is not None:
            return vector_initializer
        matrix_initializer = self.hlsl_matrix_aggregate_initializer(expr, mapped_type)
        if matrix_initializer is not None:
            return matrix_initializer
        elements = [self.generate_expression(element) for element in expr.elements]
        return "{" + ", ".join(elements) + "}"

    def render_hlsl_typed_buffer_atomic_array_literal(
        self, expr, expected_type, indent
    ):
        element_type = self.hlsl_array_literal_element_expected_type(expected_type)
        array_contract = None
        if element_type is not None:
            array_contract = self.hlsl_fixed_array_aggregate_contract(
                expr, expected_type
            )
            _mapped_type, extent, _element_type = array_contract
            if len(getattr(expr, "elements", []) or []) > extent:
                self.hlsl_aggregate_error(
                    expr,
                    expected_type=self.map_type(expected_type),
                    reason="element-count-mismatch",
                    expected_element_count=extent,
                )
        code = ""
        rendered_elements = []
        changed = False
        for element in getattr(expr, "elements", []) or []:
            element_code, rendered_element = (
                self.render_hlsl_typed_buffer_atomic_value_expression(
                    element, element_type, indent
                )
            )
            code += element_code
            rendered_elements.append(rendered_element)
            changed = changed or bool(element_code)

        if not changed:
            return None
        if array_contract is not None:
            _mapped_type, extent, _element_type = array_contract
            rendered_elements.extend(
                self.hlsl_value_initialized_expression(expr, element_type)
                for _index in range(len(rendered_elements), extent)
            )
        return code, "{" + ", ".join(rendered_elements) + "}"
