"""HLSL rendering of CrossGL function-call expressions for ``HLSLCodeGen.generate_expression``."""

from ..unhandled import UNHANDLED
from .crossgl_vector_names import (
    crossgl_named_vector_spelling,
)
from .directx_builtin_call_forms import DirectXBuiltinCallFormMixin
from .directx_constructor_names import HLSL_BUILTIN_CONSTRUCTOR_NAMES
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generic_function_call_name,
    reject_unresolved_generic_function_call,
    reject_unresolved_generic_member_call,
)


class DirectXCallExpressionMixin(DirectXBuiltinCallFormMixin):
    """Function-call members of ``HLSLCodeGen.generate_expression``."""

    def generate_hlsl_function_call_expression(
        self,
        expr,
    ):
        """Render a function-call expression."""
        reject_unresolved_generic_member_call(expr, "DirectX")
        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        args = getattr(expr, "arguments", getattr(expr, "args", []))
        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call

        func_name = None
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            func_name = func_expr.name
            callee = self.hlsl_function_call_name(func_name)
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = self.hlsl_function_call_name(func_expr)
        else:
            callee = self.generate_expression(func_expr)

        original_func_name = self.hlsl_builtin_option_constructor_name(func_name)
        original_func_name = crossgl_named_vector_spelling(original_func_name)
        if original_func_name != func_name:
            func_name = original_func_name
            callee = original_func_name

        handled = self.generate_hlsl_storage_and_constructor_call(
            expr, args, func_expr, func_name
        )
        if handled is not UNHANDLED:
            return handled

        handled = self.generate_hlsl_rewritten_intrinsic_call(expr, args, func_name)
        if handled is not UNHANDLED:
            return handled

        call_argument_func_name = self.hlsl_materialized_function_name(func_name)
        specialized_func_name = generic_function_call_name(
            self, call_argument_func_name, args
        )
        if specialized_func_name is not None:
            callee = specialized_func_name
            func_name = specialized_func_name
        else:
            reject_unresolved_generic_function_call(
                self, call_argument_func_name, "HLSL"
            )

        resolved_overload = None
        if func_name in self.hlsl_mapped_overload_names:
            resolved_overload = self.resolve_hlsl_function_overload(
                func_name,
                args,
                call_node=expr,
            )
            if resolved_overload is not None:
                target_name = self.hlsl_function_declaration_name(resolved_overload)
                callee = target_name
                func_name = target_name
                call_argument_func_name = target_name

        constructor_type = self.hlsl_type_constructor_name(func_name)
        if constructor_type is not None:
            return self.hlsl_constructor_expression(constructor_type, args)

        if func_name in HLSL_BUILTIN_CONSTRUCTOR_NAMES:
            return self.hlsl_constructor_expression(func_name, args)
        builtin_call = self.hlsl_builtin_function_call(
            func_name,
            args,
            source_location=getattr(expr, "source_location", None),
        )
        if builtin_call is not None:
            return builtin_call
        builtin_callee = self.hlsl_builtin_function_name(func_name, args)
        if builtin_callee is not None:
            callee = builtin_callee
        self.validate_function_image_access_arguments(func_name, args)
        workgroup_callee = self.hlsl_workgroup_pointer_call_name(
            call_argument_func_name, args
        )
        if workgroup_callee is not None:
            callee = workgroup_callee
        argument_type_func_name = self.hlsl_materialized_function_name(func_name)
        rendered_args, argument_assignments = self.generate_call_arguments(
            call_argument_func_name,
            args,
            argument_type_func_name,
            call_node=expr,
            resolved_overload=resolved_overload,
        )
        call = f"{callee}({', '.join(rendered_args)})"
        return (
            f"({', '.join([*argument_assignments, call])})"
            if argument_assignments
            else call
        )
