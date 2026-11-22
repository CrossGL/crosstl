"""GLSL rendering of CrossGL function-call expressions for ``GLSLCodeGen.generate_expression``."""

from ..ast import (
    MemberAccessNode,
)
from ..semantics.array_values import array_constructor_shape
from ..unhandled import UNHANDLED
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    reject_unresolved_generic_member_call,
)
from .glsl_generic_vectors import (
    glsl_generic_vector_constructor_name,
)
from .opengl_builtin_call_forms import OpenGLBuiltinCallFormMixin
from .opengl_call_resolution import OpenGLCallResolutionMixin


class OpenGLCallExpressionMixin(OpenGLBuiltinCallFormMixin, OpenGLCallResolutionMixin):
    """Function-call members of ``GLSLCodeGen.generate_expression``."""

    def generate_glsl_function_call_expression(
        self,
        expr,
    ):
        """Render a function-call expression."""
        reject_unresolved_generic_member_call(expr, "OpenGL")
        func_expr = getattr(expr, "function", getattr(expr, "name", expr))
        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            expr.args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call

        if (
            isinstance(func_expr, MemberAccessNode)
            and func_expr.member == "length"
            and not expr.args
        ):
            # A buffer parameter or reinterpreted view is rewritten as an
            # offset into the buffer it reads, so its length is computed
            # from that buffer's length, not one element's.
            storage_binding = self.glsl_storage_pointer_binding(
                func_expr.object, self.glsl_storage_pointer_aliases()
            )
            if storage_binding is not None:
                return self.structured_buffer_length_expression(func_expr.object)

        func_name = None
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.generate_expression(func_expr)
        original_func_name = func_name
        original_func_name = self.glsl_builtin_option_constructor_name(
            original_func_name
        )
        original_func_name = glsl_generic_vector_constructor_name(original_func_name)
        if original_func_name != func_name:
            func_name = original_func_name
            callee = original_func_name

        handled = self.generate_glsl_lowered_builtin_call(
            expr, func_expr, original_func_name
        )
        if handled is not UNHANDLED:
            return handled

        handled = self.generate_glsl_rewritten_builtin_call(expr, original_func_name)
        if handled is not UNHANDLED:
            return handled

        if (
            original_func_name not in self.function_return_types
            and func_name in self.function_map
        ):
            func_name = self.function_map[func_name]
        array_constructor = array_constructor_shape(expr)
        if (
            array_constructor is not None
            and original_func_name not in self.function_return_types
        ):
            # ``i32[3](...)`` names its element in CrossGL; GLSL spells it ``int[3](...)``.
            element, size = array_constructor
            func_name = f"{self.map_type(element)}[{size}]"
            callee = func_name
        self.validate_fragment_only_helper_call(original_func_name)

        handled = self.generate_glsl_resource_builtin_call(
            expr, func_name, original_func_name
        )
        if handled is not UNHANDLED:
            return handled

        (
            resource_call_args,
            resolved_overload,
            specialized_source_name,
            resource_lookup_name,
            private_materialized_call_name,
            func_name,
            callee,
        ) = self.resolve_glsl_call_target(expr, func_name, original_func_name, callee)

        handled = self.generate_glsl_constructor_call(
            expr, func_name, original_func_name
        )
        if handled is not UNHANDLED:
            return handled

        return self.emit_glsl_resolved_function_call(
            expr,
            callee,
            func_name,
            original_func_name,
            private_materialized_call_name,
            resolved_overload,
            resource_call_args,
            resource_lookup_name,
            specialized_source_name,
        )
