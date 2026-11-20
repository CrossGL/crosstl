"""Stage features in the HLSL generator: bfloat16 calls, feedback textures, buffer synchronization, unsupported texture interpolation, derivatives, member maps and input attachments."""

import re
import struct

from ..ast import (
    CooperativeMatrixType,
    ExpressionNode,
    FunctionCallNode,
    PointerType,
    ReferenceType,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    ZeroExtentArrayUnsupportedError,
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    prepend_array_dimension,
    split_array_type_suffix,
)
from .barrier_peephole import HLSL_EXECUTION_BARRIER_PLACEHOLDER
from .buffer_block_diagnostics import BufferBlockUnsupportedError
from .crossgl_vector_names import (
    crossgl_named_vector_spelling,
    crossgl_named_vector_type,
)
from .directx_values import (
    DirectXBFloat16UnsupportedError,
    DirectXCooperativeMatrixUnsupportedError,
    DirectXNative16BitUnsupportedError,
    _directx_cooperative_matrix_contract_type,
    _directx_cooperative_matrix_fragment_contract,
    _directx_cooperative_matrix_fragment_contract_message,
)
from .enum_utils import (
    default_value_expression,
    generic_enum_specialized_type_name,
    generic_type_parts,
)
from .generic_struct_utils import generic_struct_specialized_type_name
from .hlsl_extension_builtins import refuse_unmapped_hlsl_builtin
from .hlsl_ray_queries import hlsl_ray_query_type
from .hlsl_type_name_memo import memoized_bare_hlsl_type
from .image_access_contracts import floating_coordinate_dimension_from_type_name
from .operator_spellings import map_c_family_operator
from .written_type_names import (
    HLSL_TYPE_NAMES,
    written_type_name,
)


class DirectXStageFeatureMixin:
    """Stage feature members of ``HLSLCodeGen``."""

    def hlsl_default_value_expression(self, vtype, seen_structs=None):
        type_name = self.type_name_string(vtype)
        base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
        if array_suffix:
            return self.diagnostic_zero_value_for_type(base_type)

        mapped_type = self.map_type(base_type)
        struct_members = self.struct_member_types.get(base_type)
        if struct_members:
            seen_structs = set(seen_structs or set())
            if base_type in seen_structs:
                return self.diagnostic_zero_value_for_type(base_type)
            nested_seen_structs = seen_structs | {base_type}
            member_values = [
                self.hlsl_default_value_expression(member_type, nested_seen_structs)
                for member_type in struct_members.values()
            ]
            return f"{mapped_type}({', '.join(member_values)})"

        component_count = self.value_component_count(mapped_type)
        if component_count and component_count > 1:
            return self.diagnostic_zero_value_for_type(mapped_type)
        return default_value_expression(self, base_type)

    def is_input_attachment_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        return bool(re.fullmatch(r"[iu]?subpassInput(?:MS)?", str(type_name or "")))

    def unsupported_input_attachment_call(self, func_name):
        if func_name != "subpassLoad":
            return None
        fallback_type = self.current_expression_expected_type or "vec4"
        return self.refuse_lowering(
            "input attachment load: subpassLoad requires Vulkan subpass input lowering",
            self.diagnostic_zero_value_for_type(fallback_type),
        )

    def unsupported_glsl_buffer_block_access_value(self, expr):
        name = self.unsupported_glsl_buffer_block_access_name(expr)
        if not name:
            return None
        raise BufferBlockUnsupportedError(
            "directx", "access", f"{name} has no ByteAddressBuffer lowering"
        )

    def unsupported_glsl_buffer_block_assignment_diagnostic(self, target):
        name = self.unsupported_glsl_buffer_block_access_name(target)
        if not name:
            return None
        raise BufferBlockUnsupportedError(
            "directx", "assignment", f"{name} has no ByteAddressBuffer lowering"
        )

    def hlsl_resource_type_name(self, vtype):
        base_type = self.resource_base_type(vtype)
        return str(base_type).split("<", 1)[0]

    def is_hlsl_rw_texture_type(self, vtype):
        return self.hlsl_resource_type_name(vtype).startswith(
            ("RWTexture", "RasterizerOrderedTexture")
        )

    def is_hlsl_feedback_texture_type(self, vtype):
        return self.hlsl_resource_type_name(vtype).startswith("FeedbackTexture")

    def is_hlsl_constant_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) == "ConstantBuffer"

    def is_hlsl_texture_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) == "TextureBuffer"

    def is_hlsl_readonly_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) in {
            "Buffer",
            "StructuredBuffer",
            "ByteAddressBuffer",
        }

    def is_hlsl_uav_buffer_type(self, vtype):
        return self.hlsl_resource_type_name(vtype) in {
            "RWBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }

    def is_hlsl_interpolation_builtin_call_name(self, callee, shadowed_names=None):
        if callee not in self.HLSL_INTERPOLATION_INTRINSICS:
            return False
        if shadowed_names is None:
            shadowed_names = getattr(self, "function_return_types", {})
        return callee not in shadowed_names

    def collect_hlsl_interpolation_function_names(self, functions):
        named_functions = {
            func.name: func
            for func in functions or []
            if getattr(func, "name", None) is not None
        }
        requirements = {}
        calls_by_function = {}

        for func_name, func in named_functions.items():
            callees = set()
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = self.function_call_name(node)
                if self.is_hlsl_interpolation_builtin_call_name(
                    callee, named_functions
                ):
                    requirements[func_name] = callee
                elif callee in named_functions:
                    callees.add(callee)
            if callees:
                calls_by_function[func_name] = callees

        changed = True
        while changed:
            changed = False
            for func_name, callees in calls_by_function.items():
                if func_name in requirements:
                    continue
                for callee in callees:
                    required_builtin = requirements.get(callee)
                    if required_builtin is not None:
                        requirements[func_name] = required_builtin
                        changed = True
                        break

        return requirements

    def hlsl_interpolation_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.function_call_name(node)
            if self.is_hlsl_interpolation_builtin_call_name(callee):
                calls.append((callee, callee))
            elif callee in self.hlsl_interpolation_function_names:
                calls.append((callee, self.hlsl_interpolation_function_names[callee]))
        return calls

    def validate_hlsl_interpolation_stage_calls(self, func, shader_type):
        if shader_type in {None, "fragment"}:
            return

        calls = self.hlsl_interpolation_calls(func)
        if not calls:
            return

        callee, builtin_name = calls[0]
        intrinsic = self.HLSL_INTERPOLATION_INTRINSICS[builtin_name]
        if callee == builtin_name:
            detail = (
                f"'{builtin_name}' lowers to {intrinsic}, which is only valid "
                "in fragment/pixel stages"
            )
        else:
            detail = (
                f"'{callee}' reaches '{builtin_name}', which lowers to "
                f"{intrinsic} and is only valid in fragment/pixel stages"
            )
        raise DirectXRefusal(
            f"DirectX {shader_type} stage cannot call {callee}; {detail}"
        )

    def is_hlsl_derivative_builtin_call_name(self, callee, shadowed_names=None):
        if callee not in self.HLSL_DERIVATIVE_INTRINSICS:
            return False
        if shadowed_names is None:
            shadowed_names = getattr(self, "function_return_types", {})
        return callee not in shadowed_names

    def collect_hlsl_derivative_function_names(self, functions):
        named_functions = {
            func.name: func
            for func in functions or []
            if getattr(func, "name", None) is not None
        }
        requirements = {}
        calls_by_function = {}

        for func_name, func in named_functions.items():
            callees = set()
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = self.function_call_name(node)
                if self.is_hlsl_derivative_builtin_call_name(callee, named_functions):
                    requirements[func_name] = callee
                elif callee in named_functions:
                    callees.add(callee)
            if callees:
                calls_by_function[func_name] = callees

        changed = True
        while changed:
            changed = False
            for func_name, callees in calls_by_function.items():
                if func_name in requirements:
                    continue
                for callee in callees:
                    required_builtin = requirements.get(callee)
                    if required_builtin is not None:
                        requirements[func_name] = required_builtin
                        changed = True
                        break

        return requirements

    def hlsl_derivative_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.function_call_name(node)
            if self.is_hlsl_derivative_builtin_call_name(callee):
                calls.append((callee, callee))
            elif callee in self.hlsl_derivative_function_names:
                calls.append((callee, self.hlsl_derivative_function_names[callee]))
        return calls

    def validate_hlsl_derivative_stage_calls(self, func, shader_type):
        if shader_type in {None, "fragment"}:
            return

        calls = self.hlsl_derivative_calls(func)
        if not calls:
            return

        callee, builtin_name = calls[0]
        intrinsic = self.HLSL_DERIVATIVE_INTRINSICS[builtin_name]
        if callee == builtin_name:
            detail = (
                f"'{builtin_name}' emits {intrinsic}, which is only valid in "
                "fragment/pixel stages"
            )
        else:
            detail = (
                f"'{callee}' reaches '{builtin_name}', which emits {intrinsic} "
                "and is only valid in fragment/pixel stages"
            )
        raise DirectXRefusal(
            f"DirectX {shader_type} stage cannot call {callee}; {detail}"
        )

    def is_hlsl_synchronization_builtin_call_name(self, callee, shadowed_names=None):
        if callee not in self.HLSL_SYNCHRONIZATION_INTRINSICS:
            return False
        if shadowed_names is None:
            shadowed_names = getattr(self, "function_return_types", {})
        return callee not in shadowed_names

    def collect_hlsl_synchronization_function_names(self, functions):
        named_functions = {
            func.name: func
            for func in functions or []
            if getattr(func, "name", None) is not None
        }
        requirements = {}
        calls_by_function = {}

        for func_name, func in named_functions.items():
            callees = set()
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = self.function_call_name(node)
                if self.is_hlsl_synchronization_builtin_call_name(
                    callee, named_functions
                ):
                    requirements[func_name] = callee
                elif callee in named_functions:
                    callees.add(callee)
            if callees:
                calls_by_function[func_name] = callees

        changed = True
        while changed:
            changed = False
            for func_name, callees in calls_by_function.items():
                if func_name in requirements:
                    continue
                for callee in callees:
                    required_builtin = requirements.get(callee)
                    if required_builtin is not None:
                        requirements[func_name] = required_builtin
                        changed = True
                        break

        return requirements

    def hlsl_synchronization_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.function_call_name(node)
            if self.is_hlsl_synchronization_builtin_call_name(callee):
                calls.append((callee, callee))
            elif callee in self.hlsl_synchronization_function_names:
                calls.append((callee, self.hlsl_synchronization_function_names[callee]))
        return calls

    def hlsl_synchronization_supported_stages(self, builtin_name):
        intrinsic = self.HLSL_SYNCHRONIZATION_INTRINSICS.get(builtin_name)
        compute_like_stages = {"compute", "mesh", "task", "amplification", "object"}
        if intrinsic == "DeviceMemoryBarrier":
            return compute_like_stages | {"fragment"}
        return compute_like_stages

    def hlsl_synchronization_stage_description(self, stages):
        if stages == {"compute", "mesh", "task", "amplification", "object"}:
            return "compute, mesh, or amplification/task stages"
        if stages == {
            "compute",
            "mesh",
            "task",
            "amplification",
            "object",
            "fragment",
        }:
            return "fragment/pixel, compute, mesh, or amplification/task stages"
        return ", ".join(sorted(stages))

    def validate_hlsl_synchronization_stage_calls(self, func, shader_type):
        if shader_type is None:
            return

        calls = self.hlsl_synchronization_calls(func)
        for callee, builtin_name in calls:
            supported_stages = self.hlsl_synchronization_supported_stages(builtin_name)
            if shader_type in supported_stages:
                continue

            intrinsic = self.HLSL_SYNCHRONIZATION_INTRINSICS[builtin_name].replace(
                HLSL_EXECUTION_BARRIER_PLACEHOLDER, "GroupMemoryBarrierWithGroupSync"
            )
            stage_description = self.hlsl_synchronization_stage_description(
                supported_stages
            )
            if callee == builtin_name:
                detail = (
                    f"'{builtin_name}' lowers to {intrinsic}, which is only "
                    f"valid in {stage_description}"
                )
            else:
                detail = (
                    f"'{callee}' reaches '{builtin_name}', which lowers to "
                    f"{intrinsic} and is only valid in {stage_description}"
                )
            raise DirectXRefusal(
                f"DirectX {shader_type} stage cannot call {callee}; {detail}"
            )

    def collect_hlsl_pixel_only_feedback_function_names(self, functions):
        named_functions = {
            func.name: func
            for func in functions or []
            if getattr(func, "name", None) is not None
        }
        requirements = {}
        calls_by_function = {}

        for func_name, func in named_functions.items():
            callees = set()
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = self.function_call_name(node)
                if callee in self.HLSL_PIXEL_ONLY_FEEDBACK_WRITE_HELPERS:
                    requirements[func_name] = callee
                elif callee in named_functions:
                    callees.add(callee)
            if callees:
                calls_by_function[func_name] = callees

        changed = True
        while changed:
            changed = False
            for func_name, callees in calls_by_function.items():
                if func_name in requirements:
                    continue
                for callee in callees:
                    required_helper = requirements.get(callee)
                    if required_helper is not None:
                        requirements[func_name] = required_helper
                        changed = True
                        break

        return requirements

    def hlsl_pixel_only_feedback_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.function_call_name(node)
            if callee in self.HLSL_PIXEL_ONLY_FEEDBACK_WRITE_HELPERS:
                calls.append((callee, callee))
            elif callee in self.hlsl_pixel_only_feedback_function_names:
                calls.append(
                    (callee, self.hlsl_pixel_only_feedback_function_names[callee])
                )
        return calls

    def validate_hlsl_feedback_texture_stage_calls(self, func, shader_type):
        if shader_type in {None, "fragment"}:
            return

        calls = self.hlsl_pixel_only_feedback_calls(func)
        if not calls:
            return

        callee, helper_name = calls[0]
        method_name = self.HLSL_PIXEL_ONLY_FEEDBACK_WRITE_HELPERS[helper_name]
        if callee == helper_name:
            detail = (
                f"{method_name} is only valid in fragment/pixel stages; "
                f"use write_sampler_feedback_grad or write_sampler_feedback_level "
                f"from DirectX {shader_type} stages"
            )
        else:
            detail = (
                f"'{callee}' reaches {method_name} via {helper_name}, which is "
                "only valid in fragment/pixel stages"
            )
        raise DirectXRefusal(
            f"DirectX {shader_type} stage cannot call {callee}; {detail}"
        )

    def generate_hlsl_feedback_write_call(self, func_name, args):
        descriptor = self.HLSL_FEEDBACK_WRITE_HELPERS.get(func_name)
        if descriptor is None:
            return None

        method_name, allowed_arities = descriptor
        if len(args) not in allowed_arities:
            expected = " or ".join(str(count) for count in sorted(allowed_arities))
            raise DirectXRefusal(
                f"DirectX feedback helper '{func_name}' requires {expected} "
                f"argument(s), got {len(args)}"
            )

        self.validate_hlsl_feedback_write_arguments(func_name, method_name, args)
        feedback = self.generate_expression(args[0])
        rendered_args = [self.generate_expression(arg) for arg in args[1:]]
        return f"{feedback}.{method_name}({', '.join(rendered_args)})"

    def validate_hlsl_feedback_write_arguments(self, func_name, method_name, args):
        feedback_type = self.texture_resource_type(args[0])
        feedback_base = self.hlsl_resource_type_name(feedback_type)
        if not feedback_base.startswith("FeedbackTexture"):
            raise DirectXRefusal(
                f"DirectX feedback helper '{func_name}' requires "
                f"FeedbackTexture2D or FeedbackTexture2DArray receiver, got "
                f"{feedback_type or self.type_name_string(self.expression_result_type(args[0]))}"
            )

        sampled_type = self.texture_resource_type(args[1])
        sampled_base = self.hlsl_resource_type_name(sampled_type)
        if sampled_base not in {"Texture2D", "Texture2DArray"}:
            raise DirectXRefusal(
                f"DirectX feedback helper '{func_name}' sampled texture argument "
                f"must be Texture2D or Texture2DArray, got "
                f"{sampled_type or self.type_name_string(self.expression_result_type(args[1]))}"
            )

        expected_sampled_base = (
            "Texture2DArray"
            if feedback_base == "FeedbackTexture2DArray"
            else "Texture2D"
        )
        if sampled_base != expected_sampled_base:
            raise DirectXRefusal(
                f"DirectX feedback helper '{func_name}' receiver "
                f"{feedback_base} requires paired {expected_sampled_base}, got "
                f"{sampled_base}"
            )

        sampler_type = self.expression_result_type(args[2])
        mapped_sampler_type = self.map_type(self.resource_base_type(sampler_type))
        if sampler_type is not None and mapped_sampler_type != "SamplerState":
            raise DirectXRefusal(
                f"DirectX feedback helper '{func_name}' sampler argument must be "
                f"SamplerState, got {mapped_sampler_type}"
            )

        expected_dimension = 3 if sampled_base == "Texture2DArray" else 2
        self.validate_hlsl_feedback_coordinate_dimension(
            func_name, args[3], "location", expected_dimension
        )
        if method_name == "WriteSamplerFeedbackGrad":
            # The location carries the array slice, but the gradients do
            # not: DXC declares ddx and ddy as float2 for both
            # FeedbackTexture2D and FeedbackTexture2DArray.
            self.validate_hlsl_feedback_coordinate_dimension(
                func_name, args[4], "ddx", 2
            )
            self.validate_hlsl_feedback_coordinate_dimension(
                func_name, args[5], "ddy", 2
            )

    def validate_hlsl_feedback_coordinate_dimension(
        self, func_name, argument, role, expected_dimension
    ):
        arg_type = self.expression_result_type(argument)
        dimension = floating_coordinate_dimension_from_type_name(
            self.type_name_string(arg_type), self.map_type
        )
        if dimension is None or dimension == expected_dimension:
            return
        raise DirectXRefusal(
            f"DirectX feedback helper '{func_name}' {role} argument must be "
            f"float{expected_dimension}, got {self.type_name_string(arg_type)}"
        )

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, PointerType) or hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if isinstance(type_node, ReferenceType) or hasattr(
            type_node, "referenced_type"
        ):
            return self.convert_type_node_to_string(type_node.referenced_type)
        if hasattr(type_node, "value"):
            value = type_node.value
            return str(value).lower() if isinstance(value, bool) else str(value)
        if isinstance(type_node, ExpressionNode):
            return self.expression_to_string(type_node)
        generic_args = getattr(type_node, "generic_args", [])
        type_name = getattr(type_node, "name", None)
        if type_name is not None and generic_args:
            args = ", ".join(
                self.convert_type_node_to_string(arg) for arg in generic_args
            )
            return f"{type_name}<{args}>"
        if type_name is not None:
            return type_name
        elif hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            scalar = "double" if element_type in {"double", "f64"} else "float"
            # HLSL floatRxC has R rows and C columns, the same shape as CrossGL matCxR.
            return f"{scalar}{type_node.rows}x{type_node.cols}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            if str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    if isinstance(type_node.size, int):
                        return prepend_array_dimension(element_type, type_node.size)
                    else:
                        size_str = self.expression_to_string(type_node.size)
                        return prepend_array_dimension(element_type, size_str)
                else:
                    return prepend_array_dimension(element_type)
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type == "uint":
                    return f"uint{size}"
                elif element_type == "bool":
                    return f"bool{size}"
                named_vector = crossgl_named_vector_type(element_type, size)
                if named_vector is not None:
                    return named_vector
                return f"{element_type}{size}"
        elif hasattr(type_node, "operator") or hasattr(type_node, "identifier"):
            return self.expression_to_string(type_node)
        else:
            return str(type_node)

    def expression_to_string(self, expr):
        """Convert an expression node to a string representation."""
        if hasattr(expr, "value"):
            return str(expr.value)
        elif getattr(expr, "name", None) is not None:
            return str(expr.name)
        else:
            return self.generate_expression(expr)

    def hlsl_bfloat16_scalar_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        aliases = getattr(self, "hlsl_type_aliases", {})
        seen = set()
        while True:
            normalized = re.sub(
                r"^(?:(?:const|constant|device|thread|threadgroup|volatile)\s+)+",
                "",
                str(type_name).strip(),
            )
            if normalized not in aliases:
                break
            if normalized in seen:
                return None
            seen.add(normalized)
            type_name = self.type_name_string(aliases[normalized])
        normalized = normalized.rsplit("::", 1)[-1]
        return normalized if normalized in self.HLSL_SOURCE_BFLOAT16_TYPES else None

    def is_hlsl_bfloat16_type(self, vtype):
        return self.hlsl_bfloat16_scalar_type_name(vtype) is not None

    def hlsl_type_contains_bfloat16(self, vtype, seen=None):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return False
        if self.is_hlsl_bfloat16_type(type_name):
            return True

        base_type, array_suffix = split_array_type_suffix(str(type_name))
        if array_suffix:
            return self.hlsl_type_contains_bfloat16(base_type, seen=seen)

        generic_base, generic_args = generic_type_parts(base_type)
        if generic_args:
            return any(
                self.hlsl_type_contains_bfloat16(argument, seen=seen)
                for argument in generic_args
            )

        if re.search(r"(?:^|::)(?:bfloat|bfloat16|bfloat16_t)(?:[2-9]|\d)", base_type):
            return True

        seen = set(seen or ())
        if base_type in seen:
            return False
        members = self.struct_member_types.get(base_type)
        if not members:
            return False
        seen.add(base_type)
        return any(
            self.hlsl_type_contains_bfloat16(member_type, seen=seen)
            for member_type in members.values()
        )

    def directx_bfloat16_unsupported(
        self,
        message,
        *,
        operation,
        source_type=None,
        reason,
        source_location=None,
    ):
        return DirectXBFloat16UnsupportedError(
            message,
            target_profile=self.target_profile,
            operation=operation,
            source_type=self.type_name_string(source_type),
            reason=reason,
            source_location=source_location,
        )

    def validate_hlsl_bfloat16_storage_profile(
        self, *, operation, source_type, source_location=None
    ):
        if self.target_profile != "dx11":
            return
        raise self.directx_bfloat16_unsupported(
            "DirectX profile dx11 cannot represent exact 16-bit bfloat16 "
            f"storage for {operation}; use dx12 with Shader Model 6.2 and "
            "dxc -enable-16bit-types",
            operation=operation,
            source_type=source_type,
            reason="target-profile-lacks-native-16bit-storage",
            source_location=source_location,
        )

    def hlsl_bfloat16_groupshared_register_type(
        self,
        vtype,
        *,
        operation="groupshared declaration",
        source_location=None,
    ):
        type_name = self.type_name_string(vtype)
        base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
        if self.is_hlsl_bfloat16_type(base_type):
            return self.hlsl_bfloat16_storage_type(
                vtype,
                operation=operation,
                source_location=source_location,
            )

        def supports_register_aggregate(struct_name, seen=None):
            seen = set(seen or ())
            if struct_name in seen:
                return True
            members = self.struct_member_types.get(struct_name)
            if not members:
                return False
            seen.add(struct_name)
            for member_type in members.values():
                if not self.hlsl_type_contains_bfloat16(member_type):
                    continue
                member_name = self.type_name_string(member_type)
                member_base, _member_array = split_array_type_suffix(
                    str(member_name or "")
                )
                if self.is_hlsl_bfloat16_type(member_base):
                    continue
                if not supports_register_aggregate(member_base, seen):
                    return False
            return True

        if not array_suffix or not supports_register_aggregate(base_type):
            return self.hlsl_bfloat16_storage_type(
                vtype,
                operation=operation,
                source_location=source_location,
            )

        self.validate_hlsl_bfloat16_storage_profile(
            operation=operation,
            source_type=vtype,
            source_location=source_location,
        )
        # The user-defined aggregate itself is internal register state: its
        # logical bfloat members are represented by uint low-bit payloads.  Do
        # not relax ABI-visible resource storage, which still goes through the
        # scalar-only hlsl_bfloat16_storage_type contract.
        self.requires_hlsl_bfloat16_storage = True
        return self.map_type(vtype)

    def hlsl_structured_storage_type(self, vtype, **options):
        mapped = self.hlsl_bfloat16_storage_type(vtype, **options)
        return (
            self.hlsl_half_storage.physical_type(vtype)
            if self.hlsl_half_storage.lowered(vtype)
            else mapped
        )

    def hlsl_bfloat16_storage_type(
        self, vtype, *, operation="storage declaration", source_location=None
    ):
        type_name = self.type_name_string(vtype)
        base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
        scalar_type = base_type if array_suffix else vtype
        if self.is_hlsl_bfloat16_type(scalar_type):
            self.validate_hlsl_bfloat16_storage_profile(
                operation=operation,
                source_type=scalar_type,
                source_location=source_location,
            )
            self.requires_hlsl_bfloat16_storage = True
            return f"uint16_t{array_suffix}"
        if self.hlsl_type_contains_bfloat16(vtype):
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 storage currently supports scalar payload "
                f"elements, not aggregate or vector type '{self.type_name_string(vtype)}'",
                operation=operation,
                source_type=vtype,
                reason="unsupported-storage-shape",
                source_location=source_location,
            )
        return self.map_type(vtype)

    def hlsl_native_16_bit_source_type(self, source_type, mapped_type):
        source_name = str(self.type_name_string(source_type) or "").strip()
        if source_name.startswith(
            ("min10float", "min12int", "min16float", "min16int", "min16uint")
        ):
            return mapped_type
        native_source_match = re.fullmatch(
            r"(?:float16_t|int16_t|uint16_t)(?P<shape>[234](?:x[234])?)?",
            source_name,
        )
        if native_source_match is not None:
            shape = native_source_match.group("shape") or ""
            if self.widen_native_float16 and source_name.startswith("float16_t"):
                return f"float{shape}"
            native_mapped_type = source_name
            if self.target_profile == "dx11":
                raise DirectXNative16BitUnsupportedError(
                    "DirectX profile dx11 cannot preserve exact 16-bit source "
                    f"type '{source_name}'; use directx-12 with Shader Model "
                    "6.2 and dxc -enable-16bit-types, or declare an explicit "
                    "minimum-precision min16 type",
                    target_profile=self.target_profile,
                    source_type=source_name,
                    mapped_type=native_mapped_type,
                    reason="target-profile-lacks-native-16bit-types",
                    source_location=getattr(source_type, "source_location", None),
                )
            self.requires_hlsl_native_16_bit_types = True
            return native_mapped_type
        for minimum_type, native_type in (
            ("half", "float16_t"),
            ("min16int", "int16_t"),
            ("min16uint", "uint16_t"),
        ):
            type_match = re.fullmatch(
                rf"{re.escape(minimum_type)}(?P<shape>[234](?:x[234])?)?",
                str(mapped_type),
            )
            if type_match is not None:
                shape = type_match.group("shape") or ""
                if self.widen_native_float16 and native_type == "float16_t":
                    return f"float{shape}"
                native_mapped_type = native_type + shape
                if self.target_profile == "dx11":
                    raise DirectXNative16BitUnsupportedError(
                        "DirectX profile dx11 cannot preserve exact 16-bit source "
                        f"type '{source_name}'; use directx-12 with Shader Model "
                        "6.2 and dxc -enable-16bit-types, or declare an explicit "
                        "minimum-precision min16 type",
                        target_profile=self.target_profile,
                        source_type=source_name,
                        mapped_type=native_mapped_type,
                        reason="target-profile-lacks-native-16bit-types",
                        source_location=getattr(source_type, "source_location", None),
                    )
                self.requires_hlsl_native_16_bit_types = True
                return native_mapped_type
        return mapped_type

    @staticmethod
    def bfloat16_bits_for_float(value):
        """Round binary32 to bfloat16 using round-to-nearest, ties-to-even."""
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        absolute_bits = bits & 0x7FFFFFFF
        if absolute_bits > 0x7F800000:
            return ((bits >> 16) | 0x40) & 0xFFFF
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF

    def map_type(self, vtype, *, memoized=True):
        """Map types to DirectX equivalents, handling both strings and TypeNode objects.

        ``memoized=False`` skips the bare type-name memo (``hlsl_type_name_memo``).
        """
        if memoized and vtype.__class__ is str:
            return memoized_bare_hlsl_type(self, vtype)
        if vtype is None:
            return "float"

        if isinstance(vtype, CooperativeMatrixType):
            if self.cooperative_matrix_software_lowering:
                return self.directx_cooperative_matrix_fragment_lowering(vtype)[
                    "hlsl_type"
                ]
            fragment_contract = _directx_cooperative_matrix_fragment_contract(vtype)
            fragment_contract_message = (
                _directx_cooperative_matrix_fragment_contract_message(fragment_contract)
            )
            raise DirectXCooperativeMatrixUnsupportedError(
                "DirectX cooperative matrix type lowering is unavailable; "
                "emitting a regular HLSL matrix type would not preserve "
                "distributed matrix semantics; "
                f"{fragment_contract_message}",
                operation="type",
                matrix_type=vtype,
                **fragment_contract,
                reason="unsupported-type",
                source_location=getattr(vtype, "source_location", None),
            )

        if isinstance(vtype, PointerType) or hasattr(vtype, "pointee_type"):
            return f"{self.map_type(vtype.pointee_type)}*"

        if isinstance(vtype, ReferenceType) or hasattr(vtype, "referenced_type"):
            return self.map_type(vtype.referenced_type)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        ray_query_type = hlsl_ray_query_type(vtype_str)
        if ray_query_type is not None:
            return ray_query_type

        bfloat16_vector_payload = self.hlsl_bfloat16_vector_payload_type(vtype_str)
        if bfloat16_vector_payload is not None:
            return bfloat16_vector_payload

        aliases = getattr(self, "hlsl_type_aliases", {})
        if vtype_str in aliases:
            seen = getattr(self, "hlsl_type_alias_resolution_stack", [])
            if vtype_str in seen:
                raise DirectXRefusal(
                    f"Cyclic HLSL type alias: {' -> '.join([*seen, vtype_str])}"
                )
            self.hlsl_type_alias_resolution_stack = [*seen, vtype_str]
            try:
                return self.map_type(aliases[vtype_str])
            finally:
                self.hlsl_type_alias_resolution_stack = seen

        named_vector = crossgl_named_vector_spelling(vtype_str)
        if named_vector != vtype_str:
            return self.map_type(named_vector)

        cooperative_base, cooperative_args = generic_type_parts(vtype_str)
        if (
            cooperative_base == "array"
            and len(cooperative_args) == 2
            and evaluate_literal_int_expression(cooperative_args[1]) == 0
        ):
            raise ZeroExtentArrayUnsupportedError("directx", vtype_str)
        if (
            cooperative_args
            and cooperative_base.rsplit("::", 1)[-1] == "CooperativeMatrix"
        ):
            matrix_type = _directx_cooperative_matrix_contract_type(vtype)
            if self.cooperative_matrix_software_lowering:
                return self.directx_cooperative_matrix_fragment_lowering(matrix_type)[
                    "hlsl_type"
                ]
            fragment_contract = _directx_cooperative_matrix_fragment_contract(
                matrix_type
            )
            fragment_contract_message = (
                _directx_cooperative_matrix_fragment_contract_message(fragment_contract)
            )
            raise DirectXCooperativeMatrixUnsupportedError(
                "DirectX cooperative matrix type lowering is unavailable; "
                f"'{vtype_str}' cannot be represented as a regular HLSL matrix "
                "without changing distributed matrix semantics; "
                f"{fragment_contract_message}",
                operation="type",
                matrix_type=matrix_type,
                **fragment_contract,
                reason="unsupported-type",
                source_location=getattr(vtype, "source_location", None),
            )

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return vtype_str

        if "<" in vtype_str and vtype_str.endswith(">"):
            generic_vector = self.hlsl_generic_vector_type_name(vtype_str)
            if generic_vector is not None:
                return generic_vector

            base_type, generic_args = vtype_str.split("<", 1)
            generic_args = generic_args[:-1].strip()
            resource_base = base_type.rsplit("::", 1)[-1]
            if (
                resource_base in self.HLSL_NATIVE_16_BIT_STORAGE_RESOURCES
                and "," not in generic_args
                and self.hlsl_type_contains_bfloat16(generic_args)
            ):
                storage_type = self.hlsl_bfloat16_storage_type(
                    generic_args,
                    operation=f"{resource_base} element",
                    source_location=getattr(vtype, "source_location", None),
                )
                return f"{base_type}<{storage_type}>"
            if (
                resource_base in self.hlsl_half_storage.resource_types
                and "," not in generic_args
            ):
                return (
                    f"{base_type}<{self.hlsl_half_storage.physical_type(generic_args)}>"
                )
            feedback_texture_types = {
                "feedbackTexture2D": "FeedbackTexture2D",
                "feedbackTexture2DArray": "FeedbackTexture2DArray",
            }
            feedback_texture_type = feedback_texture_types.get(base_type)
            if feedback_texture_type and generic_args:
                return f"{feedback_texture_type}<{generic_args}>"
            mapped_type = self.type_mapping.get(vtype_str)
            if mapped_type is not None:
                return self.hlsl_native_16_bit_source_type(vtype_str, mapped_type)
            if "," not in generic_args:
                return f"{base_type}<{self.map_type(generic_args)}>"

        if vtype_str in self.type_mapping:
            mapped_type = self.type_mapping[vtype_str]
        else:
            mapped_type = written_type_name(
                vtype_str,
                HLSL_TYPE_NAMES,
                self.program_type_names,
                self.written_foreign_type_names,
            )
        return self.hlsl_native_16_bit_source_type(vtype_str, mapped_type)

    def map_struct_member_type(self, struct_name, member_name, vtype):
        mapped_type = self.map_type(vtype)
        if (
            member_name
            and self.is_sampler_type(vtype)
            and self.struct_member_uses_comparison_sampler(struct_name, member_name)
        ):
            return mapped_type.replace("SamplerState", "SamplerComparisonState", 1)
        return mapped_type

    def hlsl_tess_factor_member_declaration(self, member_type, member_name, semantic):
        mapped_semantic = self.hlsl_semantic_name(semantic)
        if str(mapped_semantic).lower() not in {
            "sv_tessfactor",
            "sv_insidetessfactor",
        }:
            return None

        base_type, array_suffix = split_array_type_suffix(str(member_type))
        if array_suffix:
            return format_c_style_array_declaration(member_type, member_name)

        vector_base = None
        vector_size = None
        for scalar_base in ("float", "half", "double", "min16float"):
            if base_type.startswith(scalar_base):
                suffix = base_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    vector_base = scalar_base
                    vector_size = suffix
                    break

        if vector_base is None:
            return f"{member_type} {member_name}"
        return f"{vector_base} {member_name}[{vector_size}]"

    def struct_member_uses_comparison_sampler(self, struct_name, member_name):
        return (
            struct_name,
            member_name,
        ) in self.comparison_sampler_struct_members

    def map_operator(self, op):
        """Spell a CrossGL operator; an operator CrossGL does not define is refused."""
        return map_c_family_operator(op, self.LOWERING_REFUSAL_TYPE.label)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to an HLSL semantic suffix."""
        if semantic:
            mapped = refuse_unmapped_hlsl_builtin(
                self, semantic, self.hlsl_semantic_name(semantic)
            )
            return f": {mapped}"
        else:
            return ""  # Handle None by returning an empty string
