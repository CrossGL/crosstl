"""Sampled-texture calls of the SPIR-V generator: implicit-LOD, depth-compare, offset and gather sampling, texel fetches, size, sample-count, level and LOD queries, and explicit-LOD and gradient sampling."""

from typing import (
    List,
)

from ..unhandled import (
    UNHANDLED,
)
from .spirv_texel_buffers import (
    emit_texel_buffer_fetch,
)
from .spirv_values import (
    SpirvId,
)


class SpirvSamplingCallMixin:
    """Sampled-texture members of ``VulkanSPIRVCodeGen.call_resource_function``."""

    def spirv_implicit_lod_sample_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit implicit-LOD texture sampling."""
        if function_name in {"texture", "texture2D", "textureCube"}:
            sample_args = self.sampled_texture_operands(function_name, args)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 1:
                self.refuse_lowering(
                    f"{function_name} accepts only texture, optional "
                    "sampler, coordinate, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if extra_args:
                bias_operand = self.texture_bias_operand(function_name, extra_args[0])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.default_lod_operand()}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id}"
                    f"{(' ' + bias_operand) if bias_operand else ''}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_depth_compare_sample_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit depth-compare sampling with optional LOD, offset and gradients."""
        if function_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareOffset",
        }:
            extra_arg_count = {
                "textureCompare": 0,
                "textureCompareLod": 1,
                "textureCompareLodOffset": 2,
                "textureCompareGrad": 2,
                "textureCompareGradOffset": 3,
                "textureCompareOffset": 1,
            }[function_name]
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            if compare_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            result_type = self.register_primitive_type("float")
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)

            if function_name in {"textureCompareLod", "textureCompareLodOffset"}:
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {"textureCompareGrad", "textureCompareGradOffset"}:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
            if function_name in {
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
            }:
                offset_index = -1
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[offset_index]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            if function_name == "textureCompare":
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {self.default_lod_operand()}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id}"
                    )
            elif function_name == "textureCompareOffset":
                offset_operand = self.image_offset_operand(extra_args[0])
                if self.requires_explicit_lod_sampling():
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefExplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} "
                        f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                    )
                else:
                    self.emit(
                        f"%{id_value} = OpImageSampleDrefImplicitLod "
                        f"%{result_type.id} %{sampled_image_id.id} "
                        f"%{coord_id.id} %{depth_id.id} {offset_operand}"
                    )
            elif function_name == "textureCompareLod":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Lod %{extra_args[0].id}"
                )
            elif function_name == "textureCompareLodOffset":
                offset_operand = self.image_offset_operand(extra_args[1])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Lod %{extra_args[0].id}', offset_operand)}"
                )
            elif function_name == "textureCompareGrad":
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}"
                )
            else:
                offset_operand = self.image_offset_operand(extra_args[2])
                self.emit(
                    f"%{id_value} = OpImageSampleDrefExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id} "
                    f"{self.image_operands(f'Grad %{extra_args[0].id} %{extra_args[1].id}', offset_operand)}"
                )

            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_offset_sample_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit depth-compare gathers and offset sampling."""
        if function_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            extra_arg_count = 1 if function_name == "textureGatherCompareOffset" else 0
            compare_args = self.shadow_compare_operands(
                function_name, args, extra_arg_count
            )
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 4)
            if compare_args is None:
                return self.default_value_for_type(result_type)

            sampled_image_id, coord_id, depth_id, extra_args = compare_args
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_shadow_compare_depth_operand(function_name, depth_id):
                return self.default_value_for_type(result_type)
            if function_name == "textureGatherCompareOffset":
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[0]
                ):
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operands = (
                f" {self.image_offset_operand(extra_args[0], gather=True)}"
                if function_name == "textureGatherCompareOffset"
                else ""
            )
            self.emit(
                f"%{id_value} = OpImageDrefGather %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} %{depth_id.id}"
                f"{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureOffset":
            sample_args = self.sampled_texture_operands(function_name, args, 1)
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            offset_id = extra_args[0]

            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > 2:
                self.refuse_lowering(
                    "textureOffset accepts only texture, optional "
                    "sampler, coordinate, offset, and optional bias operands"
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_offset_operand(
                function_name, metadata, offset_id
            ):
                return self.default_value_for_type(result_type)

            bias_operand = None
            if len(extra_args) >= 2:
                bias_operand = self.texture_bias_operand(function_name, extra_args[1])
                if bias_operand is None:
                    return self.default_value_for_type(result_type)

            id_value = self.get_id()
            offset_operand = self.image_offset_operand(offset_id)
            if self.requires_explicit_lod_sampling():
                self.emit(
                    f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(self.default_lod_operand(), offset_operand)}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageSampleImplicitLod %{result_type.id} "
                    f"%{sampled_image_id.id} %{coord_id.id} "
                    f"{self.image_operands(offset_operand, bias_operand)}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_gather_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit component gathers and texel-buffer fetches."""
        if function_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            required_extra_count = 0 if function_name == "textureGather" else 1
            sample_args = self.sampled_texture_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            int_type = self.register_primitive_type("int")
            result_type = self.resource_access_result_type(metadata)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureGather":
                if len(extra_args) > 1:
                    self.refuse_lowering(
                        "textureGather accepts only texture, optional "
                        "sampler, coordinate, and optional component operands"
                    )
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[0] if extra_args else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)
                offset_id = None
            elif function_name == "textureGatherOffsets":
                if len(extra_args) not in {1, 2, 4, 5}:
                    self.refuse_lowering(
                        self.sampled_texture_excess_operand_reason(function_name)
                    )
                    return self.default_value_for_type(result_type)
                return self.emit_texture_gather_offsets(
                    sampled_image_id,
                    coord_id,
                    extra_args,
                    metadata,
                    int_type,
                )
            else:
                if len(extra_args) > 2:
                    self.refuse_lowering(
                        "textureGatherOffset accepts only texture, "
                        "optional sampler, coordinate, offset, and optional "
                        "component operands"
                    )
                    return self.default_value_for_type(result_type)
                offset_id = extra_args[0]
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                ):
                    return self.default_value_for_type(result_type)
                component_id = (
                    extra_args[1]
                    if len(extra_args) >= 2
                    else self.register_constant(0, int_type)
                )
                if not self.validate_texture_gather_component_operand(
                    function_name, component_id
                ):
                    return self.default_value_for_type(result_type)

            return self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                offset_id,
            )

        texel_buffer_fetch = emit_texel_buffer_fetch(self, function_name, args)
        if texel_buffer_fetch is not None:
            return texel_buffer_fetch
        return UNHANDLED

    def spirv_texel_fetch_and_size_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit texel fetches and texture and image size queries."""
        if function_name in {"texelFetch", "texelFetchOffset"}:
            required_extra_count = 1 if function_name == "texelFetch" else 2
            # A texel fetch reads the image without filtering, so the sampler it names is irrelevant.
            sample_args = self.texel_fetch_operands(
                function_name, args, required_extra_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            operand_id = extra_args[0]
            offset_id = extra_args[1] if function_name == "texelFetchOffset" else None
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > required_extra_count:
                self.refuse_lowering(
                    self.sampled_texture_excess_operand_reason(function_name)
                )
                return self.default_value_for_type(result_type)

            if metadata.get("dim") == "Cube":
                self.refuse_lowering(f"{function_name} is not valid for cube images")
                return self.default_value_for_type(result_type)

            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id, integer=True
            ):
                return self.default_value_for_type(result_type)
            if metadata.get("multisampled"):
                if not self.validate_sampled_texture_fetch_operand(
                    function_name, operand_id, "sample"
                ):
                    return self.default_value_for_type(result_type)
            elif not self.validate_sampled_texture_fetch_operand(
                function_name, operand_id, "LOD"
            ):
                return self.default_value_for_type(result_type)

            if metadata.get("multisampled") and offset_id is not None:
                self.refuse_lowering(
                    "texelFetchOffset is not valid for multisample images"
                )
                return self.default_value_for_type(result_type)
            if (
                offset_id is not None
                and not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, offset_id
                )
            ):
                return self.default_value_for_type(result_type)

            image_id = self.image_operand_for_query(sampled_image_id, metadata)
            if image_id is None:
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            image_operand = (
                f"Sample %{operand_id.id}"
                if metadata.get("multisampled")
                else f"Lod %{operand_id.id}"
            )
            if offset_id is not None:
                image_operand = self.image_operands(
                    image_operand,
                    self.image_offset_operand(offset_id),
                )
            self.emit(
                f"%{id_value} = OpImageFetch %{result_type.id} "
                f"%{image_id.id} %{coord_id.id} {image_operand}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name in {"textureSize", "imageSize"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image" if function_name == "textureSize" else "storage_image"
            )
            if not metadata or not self.image_query_kind_matches(
                metadata, expected_kind
            ):
                self.refuse_lowering(
                    f"{function_name} requires a {expected_kind} operand"
                )
                return self.resource_query_size_default_value(metadata)

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.resource_query_size_default_value(metadata)

            result_type = self.resource_query_size_result_type(metadata)
            if function_name == "imageSize" and len(args) > 1:
                self.refuse_lowering("imageSize accepts only an image operand")
                return self.default_value_for_type(result_type)

            if function_name == "textureSize" and len(args) > 2:
                self.refuse_lowering(
                    "textureSize accepts only texture and optional " "LOD operands"
                )
                return self.default_value_for_type(result_type)

            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not self.validate_resource_query_lod_operand(function_name, args[1])
            ):
                return self.default_value_for_type(result_type)

            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if (
                function_name == "textureSize"
                and len(args) >= 2
                and not metadata.get("multisampled")
            ):
                self.emit(
                    f"%{id_value} = OpImageQuerySizeLod %{result_type.id} "
                    f"%{image_id.id} %{args[1].id}"
                )
            else:
                self.emit(
                    f"%{id_value} = OpImageQuerySize %{result_type.id} "
                    f"%{image_id.id}"
                )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_sample_count_and_level_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit sample-count and mip-level-count queries."""
        if function_name in {"textureSamples", "imageSamples"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.refuse_lowering(f"{function_name} accepts only an image operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            resource_id = args[0]
            metadata = self.resource_metadata_for_value(resource_id)
            expected_kind = (
                "sampled_image"
                if function_name == "textureSamples"
                else "storage_image"
            )
            if not metadata or not self.image_query_kind_matches(
                metadata, expected_kind
            ):
                self.refuse_lowering(
                    f"{function_name} requires a {expected_kind} operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("dim") != "2D" or not metadata.get("multisampled"):
                self.refuse_lowering(f"{function_name} requires a multisample 2D image")
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(resource_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQuerySamples %{result_type.id} "
                f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "textureQueryLevels":
            if not args:
                self.refuse_lowering("textureQueryLevels requires a texture operand")
                return self.register_constant(0, self.register_primitive_type("int"))

            if len(args) > 1:
                self.refuse_lowering(
                    "textureQueryLevels accepts only a texture operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            sampled_image_id = args[0]
            metadata = self.resource_metadata_for_value(sampled_image_id)
            if not metadata or not self.image_query_kind_matches(
                metadata, "sampled_image"
            ):
                self.refuse_lowering(
                    "textureQueryLevels requires a sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            if metadata.get("multisampled"):
                self.refuse_lowering(
                    "textureQueryLevels requires a non-multisample "
                    "sampled image operand"
                )
                return self.register_constant(0, self.register_primitive_type("int"))

            image_id = self.image_operand_for_query(sampled_image_id, metadata)
            if image_id is None:
                return self.register_constant(0, self.register_primitive_type("int"))

            result_type = self.register_primitive_type("int")
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            self.emit(
                f"%{id_value} = OpImageQueryLevels %{result_type.id} " f"%{image_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_query_lod_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit LOD queries."""
        if function_name == "textureQueryLod":
            if len(args) < 2:
                self.refuse_lowering(
                    "textureQueryLod requires texture and coordinate operands"
                )
                return self.resource_query_lod_default_value()

            sampled_image_id = args[0]
            coord_id = args[1]
            sampler_id = None
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                sampler_id = args[1]
                if len(args) < 3:
                    self.refuse_lowering(
                        "textureQueryLod requires a coordinate operand "
                        "after the sampler"
                    )
                    return self.resource_query_lod_default_value()
                if len(args) > 3:
                    self.refuse_lowering(
                        "textureQueryLod accepts only texture, sampler, "
                        "and coordinate operands"
                    )
                    return self.resource_query_lod_default_value()
                coord_id = args[2]
            elif len(args) > 2:
                self.refuse_lowering(
                    "textureQueryLod accepts only texture and coordinate "
                    "operands unless the second operand is a sampler"
                )
                return self.resource_query_lod_default_value()

            metadata = self.resource_metadata_for_value(sampled_image_id)
            if metadata and metadata.get("kind") == "texture" and sampler_id is None:
                self.refuse_lowering(
                    "textureQueryLod requires a sampler for separate texture operands"
                )
                return self.resource_query_lod_default_value()
            if sampler_id is not None:
                # The computed level depends on the sampler's filters and LOD clamps.
                sampled_image_id = self.pair_resource_with_explicit_sampler(
                    function_name, sampled_image_id, sampler_id, metadata
                )
                if sampled_image_id is None:
                    return self.resource_query_lod_default_value()
                metadata = self.resource_metadata_for_value(sampled_image_id)

            if not metadata or metadata.get("kind") != "sampled_image":
                self.refuse_lowering("textureQueryLod requires a sampled image operand")
                return self.resource_query_lod_default_value()

            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.resource_query_lod_default_value()

            if not self.validate_resource_query_lod_coordinate(
                function_name, metadata, coord_id
            ):
                return self.resource_query_lod_default_value()

            coord_id = self.trim_image_query_lod_coordinate(coord_id, metadata)
            float_type = self.register_primitive_type("float")
            result_type = self.register_vector_type(float_type, 2)
            id_value = self.get_id()
            self.require_capability("ImageQuery")
            if self.requires_explicit_lod_sampling():
                self.require_compute_derivatives()
            self.emit(
                f"%{id_value} = OpImageQueryLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED

    def spirv_explicit_lod_sample_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit explicit-LOD and gradient sampling."""
        if function_name in {
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
        }:
            required_arg_count = {
                "textureLod": 3,
                "textureLodOffset": 4,
                "textureGrad": 4,
                "textureGradOffset": 5,
            }[function_name]
            extra_arg_count = required_arg_count - 2
            sample_args = self.sampled_texture_operands(
                function_name, args, extra_arg_count
            )
            if sample_args is None:
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            sampled_image_id, coord_id, extra_args, metadata = sample_args
            result_type = self.resource_access_result_type(metadata)
            if len(extra_args) > extra_arg_count:
                self.refuse_lowering(
                    self.sampled_texture_excess_operand_reason(function_name)
                )
                return self.default_value_for_type(result_type)
            if not self.validate_non_multisample_sampled_image(function_name, metadata):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            if function_name == "textureLod":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Lod %{extra_args[0].id}"
            elif function_name == "textureLodOffset":
                if not self.validate_sampled_texture_lod_operand(
                    function_name, extra_args, 0
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[1]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Lod %{extra_args[0].id}",
                    self.image_offset_operand(extra_args[1]),
                )
            elif function_name == "textureGrad":
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                image_operands = f"Grad %{extra_args[0].id} %{extra_args[1].id}"
            else:
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[0], "dx"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_gradient_operand(
                    function_name, metadata, extra_args[1], "dy"
                ):
                    return self.default_value_for_type(result_type)
                if not self.validate_sampled_texture_offset_operand(
                    function_name, metadata, extra_args[2]
                ):
                    return self.default_value_for_type(result_type)
                image_operands = self.image_operands(
                    f"Grad %{extra_args[0].id} %{extra_args[1].id}",
                    self.image_offset_operand(extra_args[2]),
                )

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpImageSampleExplicitLod %{result_type.id} "
                f"%{sampled_image_id.id} %{coord_id.id} {image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)
        return UNHANDLED
