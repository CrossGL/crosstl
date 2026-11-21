"""Resolution and emission of calls to user functions for ``OpenGLCallExpressionMixin.generate_glsl_function_call_expression``: generic and overload specialization, private-pointer materialization, resource specialization and the call record."""

from .generic_function_utils import (
    generic_function_call_name,
    generic_function_value_arguments,
    reject_unresolved_explicit_generic_call,
)
from .glsl_exact_lookups import (
    glsl_note_call_target,
)
from .opengl_values import (
    OpenGLStoragePointerError,
    OpenGLTemplateTypeError,
    OpenGLWorkgroupPointerError,
)


class OpenGLCallResolutionMixin:
    """User-function call members of ``OpenGLCallExpressionMixin.generate_glsl_function_call_expression``."""

    def resolve_glsl_call_target(
        self,
        expr,
        func_name,
        original_func_name,
        callee,
    ):
        """Resolve the generic, overloaded or materialized function that the call names."""
        specialized_func_name = generic_function_call_name(
            self,
            original_func_name,
            expr.args,
        )
        resource_call_args = list(expr.args)
        resolved_overload = None
        specialized_source_name = specialized_func_name
        if specialized_func_name is not None:
            specialized_definition = self.function_definitions.get(
                specialized_func_name
            )
            func_name = (
                self.glsl_function_declaration_name(specialized_definition)
                if specialized_definition is not None
                else self.glsl_module_identifier_name(specialized_func_name)
            )
            callee = func_name
            resource_call_args = generic_function_value_arguments(
                self,
                original_func_name,
                expr.args,
            )
        else:
            reject_unresolved_explicit_generic_call(self, original_func_name, "OpenGL")
        if (
            specialized_func_name is None
            and original_func_name in getattr(self, "generic_function_definitions", {})
            and original_func_name not in (
                getattr(self, "concrete_function_names", None) or ()
            )
            and getattr(self, "enforce_concrete_glsl_types", False)
        ):
            raise OpenGLTemplateTypeError(
                "OpenGL codegen cannot infer concrete template arguments "
                f"for generic function '{original_func_name}'; provide a "
                "concrete instantiation before GLSL generation"
            )

        resource_lookup_name = specialized_source_name or func_name
        private_materialized_call_name = None
        if specialized_func_name is None:
            resolved_overload = self.resolve_glsl_function_overload(
                original_func_name,
                expr.args,
                call_node=expr,
            )
            if resolved_overload is not None:
                func_name = self.glsl_function_declaration_name(resolved_overload)
                callee = func_name
            else:
                normalized_private_name = (
                    self.glsl_private_pointer_materialized_function_name(
                        original_func_name
                    )
                )
                if normalized_private_name in (
                    self.function_private_pointer_parameters
                ):
                    private_materialized_call_name = normalized_private_name
                    func_name = normalized_private_name
                    callee = normalized_private_name
                    resource_lookup_name = normalized_private_name
        return (
            resource_call_args,
            resolved_overload,
            specialized_source_name,
            resource_lookup_name,
            private_materialized_call_name,
            func_name,
            callee,
        )

    def emit_glsl_resolved_function_call(
        self,
        expr,
        callee,
        func_name,
        original_func_name,
        private_materialized_call_name,
        resolved_overload,
        resource_call_args,
        resource_lookup_name,
        specialized_source_name,
    ):
        """Check the arguments of a resolved user-function call, record it and render it."""
        self.validate_function_structured_buffer_access_arguments(func_name, expr.args)
        self.validate_function_image_parameter_contract_arguments(func_name, expr.args)
        self.validate_function_image_access_arguments(func_name, expr.args)
        resource_specialization = self.glsl_resource_function_call_specialization(
            resource_lookup_name,
            resource_call_args,
            call_node=expr,
        )
        # A specialized generic call takes its arguments as the specialization's concrete parameters.
        argument_func_name = (
            self.glsl_function_declaration_name(resolved_overload)
            if resolved_overload is not None
            else private_materialized_call_name
            or (
                specialized_source_name
                if specialized_source_name in self.function_parameter_infos
                else original_func_name
            )
        )
        argument_parameter_infos = (
            self.glsl_function_parameter_infos(resolved_overload)
            if resolved_overload is not None
            else (
                ()
                if self.is_plain_metal_simd_group_helper_name(original_func_name)
                else None
            )
        )
        if resource_specialization is not None:
            func_name = self.glsl_function_declaration_name(resource_specialization)
            callee = func_name
            argument_func_name = func_name
            argument_parameter_infos = None
            call_args = self.glsl_resource_specialized_call_arguments(
                resource_specialization,
                resource_call_args,
            )
            call_args = self.filter_sampler_arguments(func_name, call_args)
        else:
            callee_definition = resolved_overload or self.function_definitions.get(
                resource_lookup_name
            )
            if (
                callee_definition is not None
                and self.glsl_function_has_workgroup_pointer_parameter(
                    callee_definition
                )
            ):
                raise OpenGLWorkgroupPointerError(
                    "OpenGL helper function "
                    f"'{original_func_name}' requires a workgroup pointer "
                    "argument with a concrete shared-array backing object",
                    function_name=original_func_name,
                    reason="call-backing-unresolved",
                    source_location=getattr(expr, "source_location", None),
                )
            if (
                callee_definition is not None
                and self.glsl_function_has_storage_pointer_parameter(callee_definition)
            ):
                raise OpenGLStoragePointerError(
                    "OpenGL helper function "
                    f"'{original_func_name}' requires a storage pointer "
                    "argument with a concrete buffer backing object",
                    function_name=original_func_name,
                    reason="call-backing-unresolved",
                    source_location=getattr(expr, "source_location", None),
                )
            call_args = self.filter_sampler_arguments(
                argument_func_name, resource_call_args
            )
        call_args = self.glsl_padded_array_constructor_arguments(
            original_func_name, call_args, expr
        )
        args = ", ".join(
            self.generate_function_call_arguments(
                argument_func_name,
                call_args,
                parameter_infos=argument_parameter_infos,
            )
        )
        emitted_call_name = func_name or callee
        call_target_function = resource_specialization or resolved_overload
        if call_target_function is None:
            call_target_function = self.function_definitions.get(
                emitted_call_name
            ) or self.function_definitions.get(original_func_name)
        self.record_glsl_function_call(
            emitted_call_name,
            getattr(expr, "source_location", None),
            target_function=call_target_function,
            source_target_name=original_func_name,
        )
        if call_target_function is None:
            glsl_note_call_target(self, emitted_call_name, len(call_args))
        generated_call = f"{emitted_call_name}({args})"
        if call_target_function is None and emitted_call_name not in {
            "abs",
            "ceil",
            "floor",
            "round",
            "roundEven",
            "sign",
            "trunc",
        }:
            # These intrinsics compute in the physical float type; their
            # logical half result must be rounded before any further use.
            result_type = self.glsl_componentwise_unary_result_type(
                original_func_name, expr.args
            )
            return self.glsl_apply_narrow_float_contract(
                generated_call, result_type, expr
            )
        return generated_call
