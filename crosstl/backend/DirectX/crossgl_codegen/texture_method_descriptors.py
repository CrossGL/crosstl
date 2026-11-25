"""Descriptors of HLSL texture methods by family: sampler feedback, texel loads and dimension queries, level-of-detail and sample-position queries, gathers and comparison gathers, and sampling methods."""

from ....translator.unhandled import (
    UNHANDLED,
)


class TextureMethodDescriptorMixin:
    """Method-family members of ``HLSLToCrossGLConverter.texture_method_descriptor``."""

    def texture_feedback_method_descriptor(
        self,
        member,
        resource_type,
    ):
        """Describe a sampler-feedback write method."""
        if member in self.feedback_method_map:
            resource_base = self.raw_type_base(resource_type)
            if resource_base.startswith("FeedbackTexture"):
                return {
                    "member": member,
                    "function": self.feedback_method_map[member],
                    "texture_function": self.feedback_method_map[member],
                    "buffer_function": None,
                    "component": None,
                    "usage": None,
                    "buffer_when_max_args": None,
                    "resource": "feedback_texture",
                    "resource_type": resource_type,
                    "operation": {
                        "WriteSamplerFeedback": "write_sampler_feedback",
                        "WriteSamplerFeedbackBias": "write_sampler_feedback_bias",
                        "WriteSamplerFeedbackGrad": "write_sampler_feedback_grad",
                        "WriteSamplerFeedbackLevel": "write_sampler_feedback_level",
                    }[member],
                }
        return UNHANDLED

    def texture_load_dimension_method_descriptor(
        self,
        member,
        arg_count,
        resource_type,
        with_dropped_parameters,
    ):
        """Describe ``Load`` and ``GetDimensions``."""
        if member == "Load":
            texture_function = self.texture_method_map[member]
            buffer_function = self.buffer_method_map[member]
            resource_base = self.raw_type_base(resource_type)
            dropped_parameters = []
            if self.is_byte_address_buffer_type(resource_type):
                if arg_count == 2:
                    dropped_parameters.append("status output")
                return with_dropped_parameters(
                    {
                        "member": member,
                        "function": buffer_function,
                        "texture_function": texture_function,
                        "buffer_function": buffer_function,
                        "component": None,
                        "usage": "regular",
                        "buffer_when_max_args": 1,
                        "resource_type": resource_type,
                        "diagnostic_kind": "tiled_resource_status",
                    },
                    dropped_parameters,
                )
            is_multisample = resource_base in {
                "Texture2DMS",
                "Texture2DMSArray",
                "RWTexture2DMS",
                "RWTexture2DMSArray",
                "RasterizerOrderedTexture2DMS",
                "RasterizerOrderedTexture2DMSArray",
            }
            if self.is_rw_texture_type(resource_type):
                function = "imageLoad"
                if arg_count == (3 if is_multisample else 2):
                    dropped_parameters.append("status output")
            elif self.is_sampled_texel_buffer_type(resource_type):
                function = "texelFetch"
                if arg_count == 2:
                    dropped_parameters.append("status output")
            elif resource_base.startswith(("Texture", "FeedbackTexture")):
                has_offset = arg_count is not None and arg_count >= (
                    3 if is_multisample else 2
                )
                function = "texelFetchOffset" if has_offset else texture_function
                if arg_count == (4 if is_multisample else 3):
                    dropped_parameters.append("status output")
            else:
                function = (
                    buffer_function
                    if arg_count is not None and arg_count <= 1
                    else texture_function
                )
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": function,
                    "texture_function": texture_function,
                    "buffer_function": buffer_function,
                    "component": None,
                    "usage": "regular",
                    "buffer_when_max_args": 1,
                    "resource_type": resource_type,
                    "diagnostic_kind": "tiled_resource_status",
                },
                dropped_parameters,
            )
        if member == "GetDimensions":
            texture_function = self.texture_method_map[member]
            buffer_function = self.buffer_method_map[member]
            use_buffer = self.is_buffer_resource_type(resource_type)
            if resource_type is None:
                use_buffer = arg_count is not None and arg_count <= 1
            return {
                "member": member,
                "function": buffer_function if use_buffer else texture_function,
                "texture_function": texture_function,
                "buffer_function": buffer_function,
                "component": None,
                "usage": "regular" if member == "Load" else None,
                "buffer_when_max_args": 1,
            }
        return UNHANDLED

    def texture_query_method_descriptor(
        self,
        member,
        arg_count,
        resource_type,
    ):
        """Describe the level-of-detail queries and ``GetSamplePosition``."""
        if member in {"CalculateLevelOfDetail", "CalculateLevelOfDetailUnclamped"}:
            resource_base = self.raw_type_base(resource_type)
            diagnostic_reason = None
            if arg_count != 2:
                diagnostic_reason = "expected sampler and coordinate arguments"
            elif resource_base and self.is_multisample_texture_type(resource_type):
                diagnostic_reason = f"{member} is unavailable for multisample textures"
            elif resource_base and (
                self.is_rw_texture_type(resource_type)
                or self.is_buffer_resource_type(resource_type)
                or not resource_base.startswith("Texture")
            ):
                diagnostic_reason = f"{member} is only available on sampled textures"

            descriptor = {
                "member": member,
                "function": "textureQueryLod",
                "texture_function": "textureQueryLod",
                "buffer_function": None,
                "component": None,
                "usage": None if diagnostic_reason else "regular",
                "buffer_when_max_args": None,
                "result_component": (
                    ".y" if member == "CalculateLevelOfDetailUnclamped" else ".x"
                ),
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if diagnostic_reason:
                descriptor["diagnostic_reason"] = diagnostic_reason
                descriptor["fallback_expression"] = "0.0"
            return descriptor
        if member == "GetSamplePosition":
            resource_base = self.raw_type_base(resource_type)
            diagnostic_reason = None
            if arg_count != 1:
                diagnostic_reason = "expected sample-index argument"
            elif resource_base and resource_base not in {
                "Texture2DMS",
                "Texture2DMSArray",
                "RWTexture2DMS",
                "RWTexture2DMSArray",
            }:
                diagnostic_reason = (
                    "GetSamplePosition is only available on multisample textures"
                )

            descriptor = {
                "member": member,
                "function": "textureSamplePosition",
                "texture_function": "textureSamplePosition",
                "buffer_function": None,
                "component": None,
                "usage": None if diagnostic_reason else "regular",
                "buffer_when_max_args": None,
                "diagnostic_label": "texture sample-position query",
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if diagnostic_reason:
                descriptor["diagnostic_reason"] = diagnostic_reason
                descriptor["fallback_expression"] = "vec2(0.0, 0.0)"
            return descriptor
        return UNHANDLED

    def texture_gather_method_descriptor(
        self,
        member,
        arg_count,
        resource_type,
        cube_family_resource,
        with_dropped_parameters,
    ):
        """Describe the per-component gathers and comparison gathers."""
        if member in self.texture_gather_component_map:
            dropped_parameters = []
            if cube_family_resource and arg_count == 3:
                texture_function = "textureGather"
                dropped_parameters.append("status output")
            elif arg_count in {3, 4}:
                texture_function = "textureGatherOffset"
                if arg_count == 4:
                    dropped_parameters.append("status output")
            elif arg_count in {6, 7}:
                texture_function = "textureGatherOffsets"
                if arg_count == 7:
                    dropped_parameters.append("status output")
            else:
                texture_function = "textureGather"
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": texture_function,
                    "texture_function": texture_function,
                    "buffer_function": None,
                    "component": self.texture_gather_component_map[member],
                    "usage": "regular",
                    "buffer_when_max_args": None,
                },
                dropped_parameters,
            )
        if member in self.texture_gather_compare_component_map:
            component = self.texture_gather_compare_component_map[member]
            descriptor = {
                "member": member,
                "function": "textureGatherCompare",
                "texture_function": "textureGatherCompare",
                "buffer_function": None,
                # textureGatherCompare[Offset](t, s, P, ref[, offset], comp): the component goes last, and
                # the red one (0) is the default the call without it compares.
                "component": None if component == "0" else component,
                "usage": "comparison",
                "buffer_when_max_args": None,
                "diagnostic_label": "texture gather-compare component",
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if arg_count in {7, 8}:
                descriptor["diagnostic_reason"] = (
                    f"{member} four-offset overload has no CrossGL helper yet"
                )
                descriptor["fallback_expression"] = "vec4(0.0)"
                return descriptor

            dropped_parameters = []
            resource_base = self.raw_type_base(resource_type)
            if resource_base in {"TextureCube", "TextureCubeArray"} and arg_count == 4:
                dropped_parameters.append("status output")
            elif arg_count in {4, 5}:
                descriptor["function"] = "textureGatherCompareOffset"
                descriptor["texture_function"] = "textureGatherCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            return with_dropped_parameters(descriptor, dropped_parameters)
        return UNHANDLED

    def texture_sampling_method_descriptor(
        self,
        member,
        arg_count,
        cube_family_resource,
        with_dropped_parameters,
    ):
        """Describe the sampling methods of ``texture_method_map``, with their argument-count variants."""
        if member in self.texture_method_map:
            texture_function = self.texture_method_map[member]
            dropped_parameters = []
            # The Shader Model 5 tiled-resource overloads take a minimum-LOD Clamp after the offset (cube
            # textures have none) and an optional status output last; CrossGL carries the clamp as the
            # texture*Clamp operations, so only a status output is still dropped.
            clamp_overloads = {
                # member: (operands before the offset, unclamped, clamped, cube clamped)
                "Sample": (2, "textureOffset", "textureOffsetClamp", "textureClamp"),
                "SampleBias": (
                    3,
                    "textureOffset",
                    "textureOffsetClamp",
                    "textureClamp",
                ),
                "SampleGrad": (
                    4,
                    "textureGradOffset",
                    "textureGradOffsetClamp",
                    "textureGradClamp",
                ),
                "SampleCmp": (
                    3,
                    "textureCompareOffset",
                    "textureCompareOffsetClamp",
                    "textureCompareClamp",
                ),
                # SampleCmpBias (Shader Model 6.8): CrossGL's compare forms take the bias last.
                "SampleCmpBias": (
                    4,
                    "textureCompareOffset",
                    "textureCompareOffsetClamp",
                    "textureCompareClamp",
                ),
                "SampleCmpGrad": (
                    5,
                    "textureCompareGradOffset",
                    "textureCompareGradOffsetClamp",
                    "textureCompareGradClamp",
                ),
            }
            if member in clamp_overloads and arg_count is not None:
                operands, offset_form, clamp_form, cube_clamp_form = clamp_overloads[
                    member
                ]
                if cube_family_resource and arg_count in {operands + 1, operands + 2}:
                    texture_function = cube_clamp_form
                    if arg_count == operands + 2:
                        dropped_parameters.append("status output")
                elif not cube_family_resource and arg_count in {
                    operands + 1,
                    operands + 2,
                    operands + 3,
                }:
                    texture_function = (
                        offset_form if arg_count == operands + 1 else clamp_form
                    )
                    if arg_count == operands + 3:
                        dropped_parameters.append("status output")
            elif member == "SampleLevel" and cube_family_resource and arg_count == 4:
                texture_function = "textureLod"
                dropped_parameters.append("status output")
            elif member == "SampleLevel" and arg_count in {4, 5}:
                texture_function = "textureLodOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            elif member == "SampleCmpLevel" and cube_family_resource and arg_count == 5:
                texture_function = "textureCompareLod"
                dropped_parameters.append("status output")
            elif member == "SampleCmpLevel" and arg_count in {5, 6}:
                texture_function = "textureCompareLodOffset"
                if arg_count == 6:
                    dropped_parameters.append("status output")
            elif (
                member == "SampleCmpLevelZero"
                and cube_family_resource
                and arg_count == 4
            ):
                texture_function = "textureCompareLod"
                dropped_parameters.append("status output")
            elif member == "SampleCmpLevelZero" and arg_count in {4, 5}:
                texture_function = "textureCompareLodOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            elif member == "Gather" and cube_family_resource and arg_count == 3:
                texture_function = "textureGather"
                dropped_parameters.append("status output")
            elif member == "Gather" and arg_count in {3, 4}:
                texture_function = "textureGatherOffset"
                if arg_count == 4:
                    dropped_parameters.append("status output")
            elif member == "GatherCmp" and cube_family_resource and arg_count == 4:
                texture_function = "textureGatherCompare"
                dropped_parameters.append("status output")
            elif member == "GatherCmp" and arg_count in {4, 5}:
                texture_function = "textureGatherCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            usage = (
                "comparison"
                if member in {
                    "SampleCmp",
                    "SampleCmpLevel",
                    "SampleCmpGrad",
                    "SampleCmpBias",
                    "SampleCmpLevelZero",
                    "GatherCmp",
                }
                else "regular"
            )
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": texture_function,
                    "texture_function": texture_function,
                    "buffer_function": None,
                    "component": None,
                    "usage": usage,
                    "buffer_when_max_args": None,
                },
                dropped_parameters,
            )
        return UNHANDLED
