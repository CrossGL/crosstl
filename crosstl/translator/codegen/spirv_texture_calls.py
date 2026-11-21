"""Function calls and texture operations of the SPIR-V generator: user function calls, buffer atomic names, LOD queries, gather offsets, image query kinds, shadow compare operands, resource constructors, sampled image types and coordinates, LOD and gradient validation, projected texture operands, and image and buffer atomics."""

from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

from .spirv_atomic_load_store import SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS
from .spirv_values import (
    SpirvId,
    SpirvType,
)
from .texture_lod_clamp import TEXTURE_LOD_CLAMP_INTRINSIC_NAMES


class SpirvTextureCallMixin:
    """Texture call members of ``VulkanSPIRVCodeGen``."""

    def call_function(
        self,
        function_name: str,
        args: List[SpirvId],
        function_reference=None,
    ) -> Optional[SpirvId]:
        """Call a function with arguments."""
        if function_reference is None:
            function_reference = self.resolve_function_reference(function_name, args)
        if function_reference is None:
            return self.call_builtin_function(function_name, args)

        function_id, (return_type, param_types) = function_reference
        self.merge_function_interface_variables_from_callee(
            function_name, function_id.id
        )
        value_array_indices = self.function_value_array_parameter_indices_by_id.get(
            function_id.id, set()
        )
        prepared_args = []
        for index, arg in enumerate(args):
            if index < len(param_types):
                arg = self.prepare_function_call_argument(arg, param_types[index])
                if index in value_array_indices:
                    arg = self.value_array_argument_object(arg, param_types[index])
            prepared_args.append(arg)
        args = prepared_args

        id_value = self.get_id()

        arg_list = " ".join([f"%{arg.id}" for arg in args])
        arg_suffix = f" {arg_list}" if arg_list else ""
        self.emit(
            f"%{id_value} = OpFunctionCall %{return_type.id} "
            f"%{function_id.id}{arg_suffix}"
        )

        spirv_id = SpirvId(id_value, return_type.type)
        self.value_types[id_value] = return_type
        return spirv_id

    def resource_function_names(self):
        names = {
            "subpassLoad",
            "imageLoad",
            "imageStore",
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicExchange",
            "atomicCompSwap",
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
            "buffer_dimensions",
            "buffer_append",
            "buffer_consume",
            "buffer_increment_counter",
            "buffer_decrement_counter",
            "texture",
            "texture2D",
            "textureCube",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareGradOffset",
            "textureCompareOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureLod",
            "textureLodOffset",
            "textureGrad",
            "textureGradOffset",
            "textureOffset",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "texelFetch",
            "texelFetchOffset",
            "textureSize",
            "imageSize",
            "textureSamples",
            "imageSamples",
            "textureQueryLevels",
            "textureQueryLod",
        } | TEXTURE_LOD_CLAMP_INTRINSIC_NAMES
        return names | set(SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS)

    def image_atomic_function_names(self):
        return {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def buffer_atomic_function_names(self):
        return {
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicExchange",
            "atomicCompSwap",
        }

    def buffer_function_names(self):
        return {
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
            "buffer_dimensions",
            "buffer_append",
            "buffer_consume",
            "buffer_increment_counter",
            "buffer_decrement_counter",
        }

    def resource_query_size_result_type(self, metadata) -> SpirvId:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        component_count = {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "Cube": 2,
            "3D": 3,
        }.get(dim, 2)

        if metadata and metadata.get("arrayed"):
            component_count += 1

        component_count = min(max(component_count, 1), 4)
        int_type = self.register_primitive_type("int")
        if component_count <= 1:
            return int_type
        return self.register_vector_type(int_type, component_count)

    def resource_query_lod_coordinate_components(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def resource_query_size_default_value(self, metadata) -> SpirvId:
        if metadata and metadata.get("kind") in {"sampled_image", "storage_image"}:
            return self.default_value_for_type(
                self.resource_query_size_result_type(metadata)
            )
        return self.register_constant(0, self.register_primitive_type("int"))

    def validate_resource_query_lod_operand(
        self, function_name: str, lod_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(lod_id) == 1:
            return True

        self.refuse_lowering(f"{function_name} requires a scalar integer LOD operand")
        return False

    def resource_query_lod_default_value(self) -> SpirvId:
        float_type = self.register_primitive_type("float")
        result_type = self.register_vector_type(float_type, 2)
        return self.default_value_for_type(result_type)

    def floating_value_component_count(self, value_id: SpirvId) -> Optional[int]:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        type_name = (
            value_type.type.base_type
            if value_type is not None
            else value_id.type.base_type
        )
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name in {"float", "double"}:
                return component_count
            return None

        if self.normalize_primitive_name(type_name) in {"float", "double"}:
            return 1
        return None

    def validate_resource_query_lod_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId
    ) -> bool:
        required_count = self.resource_query_lod_coordinate_components(metadata)
        actual_count = self.floating_value_component_count(coord_id)
        if actual_count is not None and actual_count >= required_count:
            return True

        self.refuse_lowering(
            f"{function_name} requires a {required_count}-component "
            "floating-point coordinate operand"
        )
        return False

    def trim_image_query_lod_coordinate(self, coord_id: SpirvId, metadata) -> SpirvId:
        required_count = self.resource_query_lod_coordinate_components(metadata)
        vector_info = self.vector_component_type_and_count(coord_id.type.base_type)
        if vector_info is None:
            return coord_id

        component_type_name, source_count = vector_info
        if source_count <= required_count:
            return coord_id

        component_type = self.register_primitive_type(component_type_name)
        components = [
            self.composite_extract(coord_id, component_type, index)
            for index in range(required_count)
        ]
        if required_count <= 1:
            return components[0]

        result_type = self.register_vector_type(component_type, required_count)
        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} {component_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def texture_gather_offsets_arguments(
        self, extra_args: List[SpirvId]
    ) -> Tuple[List[SpirvId], Optional[SpirvId]]:
        if len(extra_args) >= 4:
            component_id = extra_args[4] if len(extra_args) >= 5 else None
            return extra_args[:4], component_id

        component_id = extra_args[1] if len(extra_args) >= 2 else None
        if not extra_args:
            return [], component_id

        offsets_value = extra_args[0]
        if self.integer_value_component_count(offsets_value) is not None:
            return [offsets_value] * 4, component_id

        offsets_type = self.value_types.get(offsets_value.id)
        array_info = self.array_type_info_from_type(offsets_type)
        if array_info is not None:
            element_type, element_count = array_info
            if element_count != 4:
                return [], component_id
            return [
                self.composite_extract(offsets_value, element_type, index)
                for index in range(4)
            ], component_id

        return [offsets_value] * 4, component_id

    def emit_image_gather(
        self,
        sampled_image_id: SpirvId,
        coord_id: SpirvId,
        component_id: SpirvId,
        result_type: SpirvId,
        image_operand: Optional[Union[SpirvId, str]] = None,
    ) -> SpirvId:
        image_operands = ""
        if isinstance(image_operand, str):
            image_operands = f" {image_operand}"
        elif image_operand is not None:
            image_operands = f" {self.image_offset_operand(image_operand, gather=True)}"

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpImageGather %{result_type.id} "
            f"%{sampled_image_id.id} %{coord_id.id} %{component_id.id}"
            f"{image_operands}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def emit_texture_gather_offsets(
        self,
        sampled_image_id: SpirvId,
        coord_id: SpirvId,
        extra_args: List[SpirvId],
        metadata,
        int_type: SpirvId,
    ) -> SpirvId:
        result_type = self.resource_access_result_type(metadata)
        offsets, component_id = self.texture_gather_offsets_arguments(extra_args)
        if len(offsets) != 4:
            self.refuse_lowering("textureGatherOffsets requires four offset operands")
            return self.default_value_for_type(result_type)

        if component_id is None:
            component_id = self.register_constant(0, int_type)

        for offset_id in offsets:
            if not self.validate_sampled_texture_offset_operand(
                "textureGatherOffsets", metadata, offset_id
            ):
                return self.default_value_for_type(result_type)
        if not self.validate_texture_gather_component_operand(
            "textureGatherOffsets", component_id
        ):
            return self.default_value_for_type(result_type)

        const_offsets_operand = self.image_const_offsets_operand(offsets)
        if const_offsets_operand is not None:
            return self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                const_offsets_operand,
            )

        component_type = self.register_primitive_type(
            metadata.get("component_type", "float")
        )
        gathered_components = []
        for index, offset_id in enumerate(offsets):
            gathered = self.emit_image_gather(
                sampled_image_id,
                coord_id,
                component_id,
                result_type,
                offset_id,
            )
            gathered_components.append(
                self.composite_extract(gathered, component_type, index)
            )

        id_value = self.get_id()
        component_list = " ".join(
            f"%{component.id}" for component in gathered_components
        )
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} " f"{component_list}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def extract_image_from_sampled_image(
        self, sampled_image_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        image_type_id = metadata.get("image_type_id") if metadata else None
        image_type = (
            self.find_registered_type_by_id(image_type_id)
            if image_type_id is not None
            else None
        )
        if image_type is None:
            self.refuse_lowering("Could not determine image type for sampled image")
            return None

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpImage %{image_type.id} %{sampled_image_id.id}")
        self.value_types[id_value] = image_type
        return SpirvId(id_value, image_type.type)

    def image_operand_for_query(
        self, resource_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        if not metadata:
            return None
        if metadata.get("kind") == "sampled_image":
            return self.extract_image_from_sampled_image(resource_id, metadata)
        if metadata.get("kind") in {"storage_image", "texture"}:
            return resource_id
        return None

    @staticmethod
    def image_query_kind_matches(metadata, expected_kind: str) -> bool:
        """Whether a query or fetch that reads a sampled image may read ``metadata``'s resource.

        A separate texture (``texture2D``, HLSL ``Texture2D``) is the image itself, so size, level, sample count
        and texel-fetch queries read it directly, as they read the image of a combined sampler.
        """
        kind = metadata.get("kind")
        if expected_kind == "sampled_image":
            return kind in {"sampled_image", "texture"}
        return kind == expected_kind

    def texel_fetch_operands(
        self, function_name: str, args: List[SpirvId], extra_arg_count: int
    ):
        """Operands of an unfiltered texel read: a separate texture is fetched from directly.

        A sampler named after the texture (HLSL ``Load`` imported with its pair) is irrelevant to the read, so a
        separate texture is not combined with it first.
        """
        metadata = self.resource_metadata_for_value(args[0]) if args else None
        if not metadata or metadata.get("kind") != "texture":
            return self.sampled_texture_operands(
                function_name, args, extra_arg_count, pair_explicit_sampler=False
            )
        coord_index = 1
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2
        if len(args) < coord_index + 1 + extra_arg_count:
            self.refuse_lowering(
                f"{function_name} requires a texture, coordinate, "
                "and operation operands"
            )
            return None
        return args[0], args[coord_index], args[coord_index + 1 :], metadata

    def shadow_compare_operands(
        self, function_name: str, args: List[SpirvId], extra_arg_count: int
    ):
        coord_index = 1
        sampler_id = None
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2
                sampler_id = args[1]

        required_arg_count = coord_index + 2 + extra_arg_count
        if len(args) < required_arg_count:
            self.refuse_lowering(
                f"{function_name} requires a shadow texture, "
                "coordinate, depth, and operation operands"
            )
            return None

        sampled_image_id = args[0]
        coord_id = args[coord_index]
        depth_id = args[coord_index + 1]
        extra_args = args[coord_index + 2 : required_arg_count]
        metadata = self.resource_metadata_for_value(sampled_image_id)
        if sampler_id is not None:
            sampled_image_id = self.pair_resource_with_explicit_sampler(
                function_name, sampled_image_id, sampler_id, metadata
            )
            if sampled_image_id is None:
                return None
            metadata = self.resource_metadata_for_value(sampled_image_id)
        if not metadata or metadata.get("kind") != "sampled_image":
            self.refuse_lowering(
                f"{function_name} requires a shadow sampled image operand"
            )
            return None
        if not self.validate_non_multisample_sampled_image(function_name, metadata):
            return None
        # An explicit sampler carries the comparison (HLSL SampleCmp with a SamplerComparisonState), so the image
        # need not be declared as a depth image: the Vulkan environment ignores the Depth operand of OpTypeImage.
        if sampler_id is None and int(metadata.get("depth", 0)) != 1:
            self.refuse_lowering(
                f"{function_name} requires a shadow sampled image operand"
            )
            return None
        if len(args) > required_arg_count:
            self.refuse_lowering(
                self.shadow_compare_excess_operand_reason(function_name)
            )
            return None

        return sampled_image_id, coord_id, depth_id, extra_args

    def shadow_compare_excess_operand_reason(self, function_name: str) -> str:
        extra_operands = {
            "textureCompare": "",
            "textureCompareOffset": "offset",
            "textureCompareLod": "LOD",
            "textureCompareLodOffset": "LOD and offset",
            "textureCompareGrad": "dx and dy gradient",
            "textureCompareGradOffset": "dx and dy gradient and offset",
            "textureCompareProj": "",
            "textureCompareProjOffset": "offset",
            "textureCompareProjLod": "LOD",
            "textureCompareProjLodOffset": "LOD and offset",
            "textureCompareProjGrad": "dx and dy gradient",
            "textureCompareProjGradOffset": "dx and dy gradient and offset",
            "textureGatherCompare": "",
            "textureGatherCompareOffset": "offset",
        }[function_name]
        if extra_operands:
            return (
                f"{function_name} accepts only texture, optional "
                "sampler, coordinate, depth, and "
                f"{extra_operands} operands"
            )
        return (
            f"{function_name} accepts only texture, optional sampler, "
            "coordinate, and depth operands"
        )

    def sampled_texture_operands(
        self,
        function_name: str,
        args: List[SpirvId],
        extra_arg_count: int = 0,
        *,
        pair_explicit_sampler: bool = True,
    ):
        coord_index = 1
        sampler_id = None
        if len(args) > 1:
            sampler_metadata = self.resource_metadata_for_value(args[1])
            if sampler_metadata and sampler_metadata.get("kind") == "sampler":
                coord_index = 2
                sampler_id = args[1]

        required_arg_count = coord_index + 1 + extra_arg_count
        if len(args) < required_arg_count:
            self.refuse_lowering(
                f"{function_name} requires a texture, coordinate, "
                "and operation operands"
            )
            return None

        sampled_image_id = args[0]
        coord_id = args[coord_index]
        extra_args = args[coord_index + 1 :]
        metadata = self.resource_metadata_for_value(sampled_image_id)
        if metadata and metadata.get("kind") == "texture" and sampler_id is None:
            self.refuse_lowering(
                f"{function_name} requires a sampler for separate texture operands"
            )
            return None
        if sampler_id is not None and (
            pair_explicit_sampler or metadata and metadata.get("kind") == "texture"
        ):
            sampled_image_id = self.pair_resource_with_explicit_sampler(
                function_name, sampled_image_id, sampler_id, metadata
            )
            if sampled_image_id is None:
                return None
            metadata = self.resource_metadata_for_value(sampled_image_id)

        if not metadata or metadata.get("kind") != "sampled_image":
            self.refuse_lowering(f"{function_name} requires a sampled image operand")
            return None

        return sampled_image_id, coord_id, extra_args, metadata

    def pair_resource_with_explicit_sampler(
        self, function_name: str, resource_id: SpirvId, sampler_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        """Return the sampled image an operation with an explicit sampler reads.

        A separate texture is combined with the sampler. A combined resource is sampled through the explicit
        sampler too: its image is extracted and re-paired, because the sampler it was declared with is not the
        one the operation names (two samplers on one HLSL Texture2D must not collapse into one sample).
        """
        if metadata and metadata.get("kind") == "texture":
            return self.combine_texture_and_sampler(
                function_name, resource_id, sampler_id, metadata
            )
        if not metadata or metadata.get("kind") != "sampled_image":
            return resource_id
        image_id = self.extract_image_from_sampled_image(resource_id, metadata)
        if image_id is None:
            return None
        if self.is_non_uniform_value(resource_id):
            self.mark_non_uniform_result(image_id)
        texture_metadata = dict(metadata)
        texture_metadata["kind"] = "texture"
        return self.combine_texture_and_sampler(
            function_name, image_id, sampler_id, texture_metadata
        )

    def call_resource_constructor(
        self, type_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        metadata = self.resource_type_info(type_name)
        if metadata is None or not args:
            return None

        first_metadata = self.resource_metadata_for_value(args[0])
        if metadata.get("kind") == "sampled_image":
            if first_metadata and first_metadata.get("kind") == "sampled_image":
                return args[0]
            if (
                len(args) >= 2
                and first_metadata
                and first_metadata.get("kind") == "texture"
            ):
                return self.combine_texture_and_sampler(
                    type_name, args[0], args[1], first_metadata
                )

        if metadata.get("kind") == "texture":
            if first_metadata and first_metadata.get("kind") in {
                "texture",
                "sampled_image",
            }:
                return args[0]

        return None

    def combine_texture_and_sampler(
        self,
        function_name: str,
        texture_id: SpirvId,
        sampler_id: SpirvId,
        texture_metadata,
    ) -> Optional[SpirvId]:
        sampler_metadata = self.resource_metadata_for_value(sampler_id)
        if not sampler_metadata or sampler_metadata.get("kind") != "sampler":
            self.refuse_lowering(f"{function_name} requires a sampler operand")
            return None

        image_type_id = texture_metadata.get("image_type_id")
        if image_type_id is None:
            image_type_id = self.value_types.get(texture_id.id, texture_id).id
        sampled_image_type = self.register_sampled_image_type_for_image(
            image_type_id, texture_metadata
        )
        if sampled_image_type is None:
            self.refuse_lowering(
                f"{function_name} could not construct a sampled image "
                "from separate texture and sampler operands"
            )
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpSampledImage %{sampled_image_type.id} "
            f"%{texture_id.id} %{sampler_id.id}"
        )
        self.value_types[id_value] = sampled_image_type
        self.resource_type_metadata[id_value] = (
            self.sampled_image_metadata_from_texture(
                sampled_image_type, texture_metadata
            )
        )
        sampled_image = SpirvId(id_value, sampled_image_type.type)
        if self.is_non_uniform_value(texture_id) or self.is_non_uniform_value(
            sampler_id
        ):
            self.mark_non_uniform_result(sampled_image)
        return sampled_image

    def register_sampled_image_type_for_image(
        self, image_type_id: int, texture_metadata
    ) -> Optional[SpirvId]:
        image_type = self.find_registered_type_by_id(image_type_id)
        if image_type is None:
            return None

        type_name = f"{texture_metadata.get('type_name', 'texture')}_sampled"
        cache_key = (
            type_name,
            "sampled_image",
            texture_metadata.get("component_type", "float"),
            texture_metadata.get("format", "Unknown"),
            image_type_id,
        )
        if cache_key in self.resource_types:
            return self.resource_types[cache_key]

        id_value = self.sampled_image_type_id(image_type)
        sampled_image_type = SpirvId(id_value, SpirvType(type_name), type_name)
        self.resource_types[cache_key] = sampled_image_type
        self.resource_type_metadata.setdefault(
            id_value,
            self.sampled_image_metadata_from_texture(
                sampled_image_type, texture_metadata
            ),
        )
        return sampled_image_type

    def sampled_image_type_id(self, image_type: SpirvId) -> int:
        """The OpTypeSampledImage of ``image_type``, declared once: SPIR-V forbids duplicate non-aggregate types."""
        id_value = self.sampled_image_type_ids.get(image_type.id)
        if id_value is None:
            id_value = self.get_id()
            self.emit(f"%{id_value} = OpTypeSampledImage %{image_type.id}")
            self.sampled_image_type_ids[image_type.id] = id_value
        return id_value

    def sampled_image_metadata_from_texture(self, sampled_image_type, texture_metadata):
        metadata = dict(texture_metadata)
        metadata["kind"] = "sampled_image"
        metadata["type_name"] = sampled_image_type.type.base_type
        return metadata

    def sampled_texture_excess_operand_reason(self, function_name: str) -> str:
        operation_operands = {
            "textureLod": "LOD",
            "textureLodOffset": "LOD and offset",
            "textureGrad": "dx and dy gradient",
            "textureGradOffset": "dx and dy gradient and offset",
            "textureGatherOffsets": (
                "one offsets value or four offsets and optional component"
            ),
            "texelFetch": "LOD/sample",
            "texelFetchOffset": "LOD/sample and offset",
        }[function_name]
        return (
            f"{function_name} accepts only texture, optional sampler, "
            "coordinate, and "
            f"{operation_operands} operands"
        )

    def is_scalar_numeric_value(self, value_id: SpirvId) -> bool:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if value_type is None:
            return False
        if self.vector_component_type_and_count(value_type.type.base_type) is not None:
            return False
        return self.normalize_primitive_name(value_type.type.base_type) in {
            "int",
            "uint",
            "float",
            "double",
        }

    def texture_bias_operand(
        self, function_name: str, bias_id: SpirvId
    ) -> Optional[str]:
        if not self.is_scalar_numeric_value(bias_id):
            self.refuse_lowering(f"{function_name} requires a scalar numeric bias")
            return None
        if self.requires_explicit_lod_sampling():
            self.refuse_lowering(
                f"{function_name} bias is not valid for explicit-lod " "SPIR-V sampling"
            )
            return None
        return f"Bias %{bias_id.id}"

    def sampled_texture_axis_component_count(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def sampled_texture_coordinate_component_count(self, metadata) -> int:
        component_count = self.sampled_texture_axis_component_count(metadata)
        if metadata and metadata.get("arrayed"):
            component_count += 1
        return component_count

    def validate_sampled_texture_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId, *, integer: bool = False
    ) -> bool:
        expected_count = self.sampled_texture_coordinate_component_count(metadata)
        actual_count = (
            self.integer_value_component_count(coord_id)
            if integer
            else self.floating_value_component_count(coord_id)
        )
        if actual_count == expected_count:
            return True

        coordinate_kind = "integer" if integer else "floating-point"
        self.refuse_lowering(
            f"{function_name} requires a {expected_count}-component "
            f"{coordinate_kind} coordinate operand"
        )
        return False

    def validate_non_multisample_sampled_image(
        self, function_name: str, metadata
    ) -> bool:
        if metadata and metadata.get("multisampled"):
            self.refuse_lowering(f"{function_name} is not valid for multisample images")
            return False
        return True

    def validate_sampled_texture_lod_operand(
        self, function_name: str, operands: list, index: int = 0
    ) -> bool:
        """Check that ``operands[index]`` is a scalar LOD, converting a scalar integer to ``float`` in place.

        HLSL ``SampleLevel(s, uv, 0)`` and GLSL ``textureLod(t, uv, 0)`` pass an integer level that both languages
        convert implicitly (GLSL 4.60 section 6.1); ``OpImageSampleExplicitLod`` takes a floating-point ``Lod``.
        """
        lod_id = operands[index]
        if self.floating_value_component_count(lod_id) == 1:
            return True
        if self.integer_value_component_count(lod_id) == 1:
            operands[index] = self.convert_value_to_type(
                lod_id, self.register_primitive_type("float")
            )
            return True

        self.refuse_lowering(
            f"{function_name} requires a scalar floating-point LOD operand"
        )
        return False

    def validate_shadow_compare_depth_operand(
        self, function_name: str, depth_id: SpirvId
    ) -> bool:
        if self.floating_value_component_count(depth_id) == 1:
            return True

        self.refuse_lowering(
            f"{function_name} requires a scalar floating-point depth operand"
        )
        return False

    def validate_sampled_texture_fetch_operand(
        self, function_name: str, operand_id: SpirvId, operand_name: str
    ) -> bool:
        if self.integer_value_component_count(operand_id) == 1:
            return True

        self.refuse_lowering(
            f"{function_name} requires a scalar integer {operand_name} operand"
        )
        return False

    def validate_sampled_texture_gradient_operand(
        self, function_name: str, metadata, gradient_id: SpirvId, operand_name: str
    ) -> bool:
        expected_count = self.sampled_texture_axis_component_count(metadata)
        if self.floating_value_component_count(gradient_id) == expected_count:
            return True

        self.refuse_lowering(
            f"{function_name} requires a {expected_count}-component "
            f"floating-point {operand_name} gradient operand"
        )
        return False

    def validate_sampled_texture_offset_operand(
        self, function_name: str, metadata, offset_id: SpirvId
    ) -> bool:
        if metadata and metadata.get("dim") == "Cube":
            self.refuse_lowering(
                f"{function_name} offsets are not valid for cube images"
            )
            return False

        expected_count = self.sampled_texture_axis_component_count(metadata)
        if self.integer_value_component_count(offset_id) == expected_count:
            return True

        self.refuse_lowering(
            f"{function_name} requires a {expected_count}-component "
            "integer offset operand"
        )
        return False

    def validate_texture_gather_component_operand(
        self, function_name: str, component_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(component_id) == 1:
            return True

        self.refuse_lowering(
            f"{function_name} requires a scalar integer component operand"
        )
        return False

    def projected_coordinate_axes(self, metadata) -> int:
        dim = metadata.get("dim", "2D") if metadata else "2D"
        return {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(dim, 2)

    def project_texture_coordinate(
        self, function_name: str, coord_id: SpirvId, metadata
    ) -> Optional[SpirvId]:
        vector_info = self.vector_component_type_and_count(coord_id.type.base_type)
        if vector_info is None:
            self.refuse_lowering(
                f"{function_name} requires a projected vector coordinate"
            )
            return None

        component_type_name, source_count = vector_info
        if component_type_name not in {"float", "double"}:
            self.refuse_lowering(
                f"{function_name} requires a floating-point projected coordinate"
            )
            return None

        axis_count = self.projected_coordinate_axes(metadata)
        output_count = axis_count + (1 if metadata and metadata.get("arrayed") else 0)
        if source_count < output_count + 1:
            self.refuse_lowering(
                f"{function_name} requires a projection component after "
                "the texture coordinate"
            )
            return None

        component_type = self.register_primitive_type(component_type_name)
        projection = self.composite_extract(coord_id, component_type, source_count - 1)
        components = []
        for index in range(axis_count):
            component = self.composite_extract(coord_id, component_type, index)
            components.append(
                self.binary_operation("/", component_type, component, projection)
            )

        if metadata and metadata.get("arrayed"):
            components.append(
                self.composite_extract(coord_id, component_type, axis_count)
            )

        if len(components) == 1:
            return components[0]

        result_type = self.register_vector_type(component_type, len(components))
        return self.composite_construct(result_type, components)

    def requires_explicit_lod_sampling(self) -> bool:
        return self.current_execution_model in {"GLCompute", "MeshEXT", "TaskEXT"}

    def default_lod_operand(self) -> str:
        lod_id = self.register_constant(0.0, self.register_primitive_type("float"))
        return f"Lod %{lod_id.id}"

    def image_atomic_operand_names(
        self, function_name: str, multisampled: bool
    ) -> List[str]:
        operands = ["image", "coordinate"]
        if multisampled:
            operands.append("sample")
        if function_name == "imageAtomicCompSwap":
            operands.append("compare")
        operands.append("value")
        return operands

    def image_atomic_operand_description(
        self, function_name: str, multisampled: bool
    ) -> str:
        operands = self.image_atomic_operand_names(function_name, multisampled)
        return ", ".join(operands[:-1]) + f", and {operands[-1]} operands"

    def call_image_atomic_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if len(args) < 3:
            self.refuse_lowering(
                f"{function_name} requires image, coordinate, " "and value operands"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        image_pointer = args[0]
        coord_id = args[1]
        metadata = self.resource_metadata_for_pointer(image_pointer)
        if not metadata or metadata.get("kind") != "storage_image":
            self.refuse_lowering(
                f"{function_name} requires a storage image pointer operand"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        component_type_name = metadata.get("component_type", "uint")
        result_type = self.register_primitive_type(component_type_name)
        is_multisampled = bool(metadata.get("multisampled"))
        expected_operands = self.image_atomic_operand_names(
            function_name, is_multisampled
        )
        if len(args) > len(expected_operands):
            self.refuse_lowering(
                f"{function_name} accepts only "
                f"{self.image_atomic_operand_description(function_name, is_multisampled)}"
            )
            return self.default_value_for_type(result_type)

        if metadata.get("readonly") or metadata.get("writeonly"):
            self.refuse_lowering(f"{function_name} requires a read-write storage image")
            return self.default_value_for_type(result_type)

        if not self.validate_storage_image_coordinate(
            function_name, metadata, coord_id
        ):
            return self.default_value_for_type(result_type)

        if component_type_name not in {"int", "uint"}:
            self.refuse_lowering(f"{function_name} requires an integer storage image")
            return self.register_constant(0, self.register_primitive_type("uint"))

        if int(metadata.get("component_count", 1)) != 1:
            self.refuse_lowering(
                f"{function_name} requires a scalar storage image format"
            )
            return self.register_constant(0, self.register_primitive_type("uint"))

        value_arg_index = 2
        if is_multisampled:
            if len(args) < 4:
                self.refuse_lowering(f"{function_name} requires a sample operand")
                return self.register_constant(
                    0, self.register_primitive_type(component_type_name)
                )
            sample_id = args[2]
            if not self.validate_storage_image_sample(function_name, sample_id):
                return self.default_value_for_type(result_type)
            value_arg_index = 3
        else:
            sample_id = self.register_constant(0, self.register_primitive_type("uint"))

        if function_name == "imageAtomicCompSwap":
            if len(args) <= value_arg_index + 1:
                self.refuse_lowering(
                    f"{function_name} requires compare and value operands"
                )
                return self.register_constant(
                    0, self.register_primitive_type(component_type_name)
                )
            comparator_id = args[value_arg_index]
            value_id = args[value_arg_index + 1]
        else:
            value_id = args[value_arg_index]
            comparator_id = None

        value_id = self.convert_storage_image_value_operand(
            function_name, "value", value_id, result_type
        )
        if value_id is None:
            return self.default_value_for_type(result_type)
        if comparator_id is not None:
            comparator_id = self.convert_storage_image_value_operand(
                function_name, "compare", comparator_id, result_type
            )
            if comparator_id is None:
                return self.default_value_for_type(result_type)

        pointer_type = self.register_pointer_type(result_type, "Image")
        texel_pointer_id = self.get_id()
        self.emit(
            f"%{texel_pointer_id} = OpImageTexelPointer %{pointer_type.id} "
            f"%{image_pointer.id} %{coord_id.id} %{sample_id.id}"
        )
        self.variable_value_types[texel_pointer_id] = result_type

        atomic_operation = {
            "imageAtomicAdd": "OpAtomicIAdd",
            "imageAtomicAnd": "OpAtomicAnd",
            "imageAtomicOr": "OpAtomicOr",
            "imageAtomicXor": "OpAtomicXor",
            "imageAtomicExchange": "OpAtomicExchange",
        }.get(function_name)
        if atomic_operation is None and function_name == "imageAtomicMin":
            atomic_operation = (
                "OpAtomicSMin" if component_type_name == "int" else "OpAtomicUMin"
            )
        if atomic_operation is None and function_name == "imageAtomicMax":
            atomic_operation = (
                "OpAtomicSMax" if component_type_name == "int" else "OpAtomicUMax"
            )

        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        if function_name == "imageAtomicCompSwap":
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{result_type.id} "
                f"%{texel_pointer_id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{comparator_id.id}"
            )
        else:
            self.emit(
                f"%{id_value} = {atomic_operation} %{result_type.id} "
                f"%{texel_pointer_id} %{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def default_value_for_buffer_atomic_failure(
        self,
        function_name: str,
        args: List[SpirvId],
        target_type: Optional[SpirvId] = None,
    ) -> SpirvId:
        value_index = 2 if function_name == "atomicCompSwap" and len(args) > 2 else 1
        if len(args) > value_index:
            value_type = self.value_types.get(args[value_index].id)
            if value_type is None:
                value_type = self.find_registered_type_by_base(
                    args[value_index].type.base_type
                )
            if value_type is not None:
                return self.default_value_for_type(value_type)

        if target_type is not None:
            return self.default_value_for_type(target_type)

        return self.register_constant(0, self.register_primitive_type("uint"))

    def default_value_for_buffer_load_failure(self, args: List[SpirvId]) -> SpirvId:
        if args:
            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None:
                element_type = metadata.get("element_type")
                if element_type is not None:
                    return self.default_value_for_type(element_type)
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def call_buffer_atomic_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if len(args) < 2:
            self.refuse_lowering(f"{function_name} requires target and value operands")
            return self.register_constant(0, self.register_primitive_type("uint"))

        expected_arg_count = 3 if function_name == "atomicCompSwap" else 2
        if len(args) > expected_arg_count:
            if function_name == "atomicCompSwap":
                self.refuse_lowering(
                    f"{function_name} accepts only target, compare, "
                    "and value operands"
                )
            else:
                self.refuse_lowering(
                    f"{function_name} accepts only target and value operands"
                )
            return self.default_value_for_buffer_atomic_failure(function_name, args)

        target_pointer = args[0]
        target_type = self.variable_value_types.get(target_pointer.id)
        if target_type is None and target_pointer.type.storage_class:
            target_type = self.find_registered_type_by_base(
                target_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if target_type is None:
            self.refuse_lowering(f"{function_name} requires an addressable target")
            return self.default_value_for_buffer_atomic_failure(function_name, args)

        metadata = self.storage_buffer_access_metadata_for_pointer(target_pointer)
        # Group-shared integers are atomic at workgroup scope (Vulkan 1.3 appendix
        # "SPIR-V Environment": atomics on Workgroup storage).
        workgroup = (
            metadata is None and target_pointer.type.storage_class == "Workgroup"
        )
        if metadata is None and not workgroup:
            self.refuse_lowering(f"{function_name} requires a storage buffer target")
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if metadata and (metadata.get("readonly") or metadata.get("writeonly")):
            self.refuse_lowering(
                f"{function_name} requires a read-write storage buffer"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )

        type_name = self.normalize_primitive_name(target_type.type.base_type)
        if self.vector_type_info_from_type(target_type) is not None:
            self.refuse_lowering(
                f"{function_name} requires a scalar int or uint buffer member"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if self.matrix_type_info_from_type(target_type) is not None:
            self.refuse_lowering(
                f"{function_name} requires a scalar int or uint buffer member"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )
        if type_name not in {"int", "uint"}:
            self.refuse_lowering(
                f"{function_name} currently supports only int or uint " "buffer members"
            )
            return self.default_value_for_buffer_atomic_failure(
                function_name, args, target_type
            )

        if function_name == "atomicCompSwap":
            if len(args) < 3:
                self.refuse_lowering(
                    f"{function_name} requires compare and value operands"
                )
                return self.default_value_for_buffer_atomic_failure(
                    function_name, args, target_type
                )
            comparator_id = self.convert_value_to_type(args[1], target_type)
            value_id = self.convert_value_to_type(args[2], target_type)
        else:
            comparator_id = None
            value_id = self.convert_value_to_type(args[1], target_type)

        atomic_operation = {
            "atomicAdd": "OpAtomicIAdd",
            "atomicAnd": "OpAtomicAnd",
            "atomicOr": "OpAtomicOr",
            "atomicXor": "OpAtomicXor",
            "atomicExchange": "OpAtomicExchange",
        }.get(function_name)
        if atomic_operation is None and function_name == "atomicMin":
            atomic_operation = "OpAtomicSMin" if type_name == "int" else "OpAtomicUMin"
        if atomic_operation is None and function_name == "atomicMax":
            atomic_operation = "OpAtomicSMax" if type_name == "int" else "OpAtomicUMax"

        scope = self.spirv_scope_constant("Workgroup" if workgroup else "Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        if function_name == "atomicCompSwap":
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{target_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{comparator_id.id}"
            )
        else:
            self.emit(
                f"%{id_value} = {atomic_operation} %{target_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = target_type
        return SpirvId(id_value, target_type.type)

    def projected_texture_function_names(self):
        return {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }

    def projected_shadow_function_names(self):
        return {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }

    def projected_texture_operand_counts(self, function_name: str) -> int:
        return {
            "textureProj": 0,
            "textureProjOffset": 1,
            "textureProjLod": 1,
            "textureProjLodOffset": 2,
            "textureProjGrad": 2,
            "textureProjGradOffset": 3,
        }[function_name]

    def projected_texture_max_operand_counts(self, function_name: str) -> int:
        return {
            "textureProj": 1,
            "textureProjOffset": 2,
            "textureProjLod": 1,
            "textureProjLodOffset": 2,
            "textureProjGrad": 2,
            "textureProjGradOffset": 3,
        }[function_name]

    def projected_texture_operand_description(self, function_name: str) -> str:
        return {
            "textureProj": "optional bias",
            "textureProjOffset": "offset and optional bias",
            "textureProjLod": "LOD",
            "textureProjLodOffset": "LOD and offset",
            "textureProjGrad": "dx and dy gradient",
            "textureProjGradOffset": "dx and dy gradient and offset",
        }[function_name]

    def projected_shadow_operand_counts(self, function_name: str) -> int:
        return {
            "textureCompareProj": 0,
            "textureCompareProjOffset": 1,
            "textureCompareProjLod": 1,
            "textureCompareProjLodOffset": 2,
            "textureCompareProjGrad": 2,
            "textureCompareProjGradOffset": 3,
        }[function_name]

    def projected_texture_image_operands(
        self, function_name: str, extra_args: List[SpirvId]
    ) -> Optional[str]:
        if function_name == "textureProj":
            if extra_args:
                return self.texture_bias_operand(function_name, extra_args[0])
            if self.requires_explicit_lod_sampling():
                return self.default_lod_operand()
            return ""

        if function_name == "textureProjOffset":
            offset_operand = self.image_offset_operand(extra_args[0])
            if len(extra_args) >= 2:
                bias_operand = self.texture_bias_operand(function_name, extra_args[1])
                if bias_operand is None:
                    return None
                return self.image_operands(offset_operand, bias_operand)
            if self.requires_explicit_lod_sampling():
                return self.image_operands(self.default_lod_operand(), offset_operand)
            return offset_operand

        if function_name == "textureProjLod":
            return f"Lod %{extra_args[0].id}"
        if function_name == "textureProjLodOffset":
            return self.image_operands(
                f"Lod %{extra_args[0].id}", self.image_offset_operand(extra_args[1])
            )
        if function_name == "textureProjGrad":
            return f"Grad %{extra_args[0].id} %{extra_args[1].id}"
        return self.image_operands(
            f"Grad %{extra_args[0].id} %{extra_args[1].id}",
            self.image_offset_operand(extra_args[2]),
        )

    def call_projected_texture_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        sample_args = self.sampled_texture_operands(
            function_name,
            args,
            self.projected_texture_operand_counts(function_name),
        )
        if sample_args is None:
            return self.register_constant(0.0, self.register_primitive_type("float"))

        sampled_image_id, coord_id, extra_args, metadata = sample_args
        result_type = self.resource_access_result_type(metadata)
        if len(extra_args) > self.projected_texture_max_operand_counts(function_name):
            self.refuse_lowering(
                f"{function_name} accepts only texture, optional "
                "sampler, coordinate, and "
                f"{self.projected_texture_operand_description(function_name)} operands"
            )
            return self.default_value_for_type(result_type)
        if not self.validate_non_multisample_sampled_image(function_name, metadata):
            return self.default_value_for_type(result_type)
        if function_name in {"textureProjLod", "textureProjLodOffset"}:
            if not self.validate_sampled_texture_lod_operand(
                function_name, extra_args, 0
            ):
                return self.default_value_for_type(result_type)
        if function_name in {"textureProjGrad", "textureProjGradOffset"}:
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[0], "dx"
            ):
                return self.default_value_for_type(result_type)
            if not self.validate_sampled_texture_gradient_operand(
                function_name, metadata, extra_args[1], "dy"
            ):
                return self.default_value_for_type(result_type)
        offset_indices = {
            "textureProjOffset": 0,
            "textureProjLodOffset": 1,
            "textureProjGradOffset": 2,
        }
        offset_index = offset_indices.get(function_name)
        if (
            offset_index is not None
            and not self.validate_sampled_texture_offset_operand(
                function_name, metadata, extra_args[offset_index]
            )
        ):
            return self.default_value_for_type(result_type)

        projected_coord = self.project_texture_coordinate(
            function_name, coord_id, metadata
        )
        if projected_coord is None:
            return self.default_value_for_type(result_type)

        image_operands = self.projected_texture_image_operands(
            function_name, extra_args
        )
        if image_operands is None:
            return self.default_value_for_type(result_type)

        id_value = self.get_id()
        opcode = (
            "OpImageSampleExplicitLod"
            if image_operands and ("Lod" in image_operands or "Grad" in image_operands)
            else "OpImageSampleImplicitLod"
        )
        self.emit(
            f"%{id_value} = {opcode} %{result_type.id} "
            f"%{sampled_image_id.id} %{projected_coord.id}"
            f"{(' ' + image_operands) if image_operands else ''}"
        )
        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def projected_shadow_image_operands(
        self, function_name: str, extra_args: List[SpirvId]
    ) -> str:
        if function_name == "textureCompareProj":
            return (
                self.default_lod_operand()
                if self.requires_explicit_lod_sampling()
                else ""
            )
        if function_name == "textureCompareProjOffset":
            offset_operand = self.image_offset_operand(extra_args[0])
            if self.requires_explicit_lod_sampling():
                return self.image_operands(self.default_lod_operand(), offset_operand)
            return offset_operand
        if function_name == "textureCompareProjLod":
            return f"Lod %{extra_args[0].id}"
        if function_name == "textureCompareProjLodOffset":
            return self.image_operands(
                f"Lod %{extra_args[0].id}", self.image_offset_operand(extra_args[1])
            )
        if function_name == "textureCompareProjGrad":
            return f"Grad %{extra_args[0].id} %{extra_args[1].id}"
        return self.image_operands(
            f"Grad %{extra_args[0].id} %{extra_args[1].id}",
            self.image_offset_operand(extra_args[2]),
        )
