"""Buffer and storage-image calls of the SPIR-V generator: the calls handed to the atomic, projected-texture and byte-address helpers, append and consume buffers and their counters, buffer dimensions, typed buffer loads and stores, subpass loads, and image loads and stores."""

from typing import (
    List,
)

from ..unhandled import (
    UNHANDLED,
)
from .spirv_atomic_load_store import (
    SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS,
    call_spirv_atomic_load_store,
)
from .spirv_image_texels import (
    emit_four_component_image_read,
    narrow_image_texel,
)
from .spirv_values import (
    SpirvId,
)
from .texture_lod_clamp import (
    is_texture_lod_clamp_operation,
    spirv_texture_lod_clamp_call,
)


class SpirvResourceCallMixin:
    """Buffer and storage-image members of ``VulkanSPIRVCodeGen.call_resource_function``."""

    def spirv_delegated_resource_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Hand LOD-clamped, atomic, projected and byte-address calls to their helpers."""
        if is_texture_lod_clamp_operation(function_name):
            sampler = (
                self.resource_metadata_for_value(args[1]) if len(args) > 1 else None
            )
            return spirv_texture_lod_clamp_call(
                self,
                function_name,
                args,
                bool(sampler) and sampler.get("kind") == "sampler",
            )
        if function_name in self.image_atomic_function_names():
            return self.call_image_atomic_function(function_name, args)
        if function_name in self.buffer_atomic_function_names():
            return self.call_buffer_atomic_function(function_name, args)
        if function_name in SPIRV_ATOMIC_LOAD_STORE_FUNCTIONS:
            return call_spirv_atomic_load_store(self, function_name, args)
        if function_name in self.projected_texture_function_names():
            return self.call_projected_texture_function(function_name, args)
        if function_name in self.projected_shadow_function_names():
            return self.call_projected_shadow_function(function_name, args)

        byte_address_load_width = self.byte_address_helper_load_width(function_name)
        if byte_address_load_width is not None:
            return self.call_byte_address_buffer_load_helper(
                function_name, args, byte_address_load_width
            )

        byte_address_store_width = self.byte_address_helper_store_width(function_name)
        if byte_address_store_width is not None:
            return self.call_byte_address_buffer_store_helper(
                function_name, args, byte_address_store_width
            )
        return UNHANDLED

    def spirv_buffer_counter_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit append, consume, counter and dimension calls on structured buffers."""
        if function_name == "buffer_append":
            if len(args) < 2:
                self.refuse_lowering("buffer_append requires buffer and value operands")
                return None
            if len(args) > 2:
                self.refuse_lowering(
                    "buffer_append accepts only buffer and value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_append requires an AppendStructuredBuffer " "operand"
                )
                return None
            return self.process_structured_buffer_append_call(
                args[0], metadata, [args[1]], "buffer_append"
            )[1]

        if function_name == "buffer_consume":
            if not args:
                self.refuse_lowering("buffer_consume requires a buffer operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )
            if len(args) > 1:
                self.refuse_lowering("buffer_consume accepts only a buffer operand")
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_consume requires a ConsumeStructuredBuffer " "operand"
                )
                return self.default_value_for_buffer_load_failure(args)
            return self.process_structured_buffer_consume_call(
                args[0], metadata, [], "buffer_consume"
            )[1]

        if function_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            if not args:
                self.refuse_lowering(f"{function_name} requires a buffer operand")
                return self.structured_buffer_counter_default_value()
            if len(args) > 1:
                self.refuse_lowering(f"{function_name} accepts only a buffer operand")
                return self.structured_buffer_counter_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    f"{function_name} requires an RWStructuredBuffer " "operand"
                )
                return self.structured_buffer_counter_default_value()
            return self.process_structured_buffer_counter_method_call(
                args[0],
                metadata,
                [],
                function_name,
                function_name == "buffer_increment_counter",
            )[1]

        if function_name == "buffer_dimensions":
            if not args:
                self.refuse_lowering("buffer_dimensions requires a buffer operand")
                return self.structured_buffer_dimensions_default_value()
            if len(args) > 1:
                self.refuse_lowering(
                    "buffer_dimensions expression form accepts only a " "buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_dimensions requires a structured or "
                    "byte-address buffer operand"
                )
                return self.structured_buffer_dimensions_default_value()
            return self.emit_structured_buffer_dimensions(
                args[0], metadata, "buffer_dimensions"
            )
        return UNHANDLED

    def spirv_buffer_element_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit typed buffer element loads and stores."""
        if function_name == "buffer_load":
            if len(args) < 2:
                self.refuse_lowering("buffer_load requires buffer and index operands")
                return self.default_value_for_buffer_load_failure(args)

            if len(args) > 2:
                self.refuse_lowering(
                    "buffer_load accepts only buffer and index operands"
                )
                return self.default_value_for_buffer_load_failure(args)

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is not None and metadata.get("writeonly"):
                self.refuse_lowering("buffer_load requires a readable buffer")
                return self.default_value_for_buffer_load_failure(args)
            if (
                metadata is not None
                and not metadata.get("byte_address")
                and metadata.get("buffer_kind")
                not in {
                    "StructuredBuffer",
                    "RWStructuredBuffer",
                }
            ):
                self.refuse_lowering("buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.refuse_lowering("buffer_load requires a StructuredBuffer operand")
                return self.default_value_for_buffer_load_failure(args)

            element_type = self.variable_value_types[element_pointer.id]
            return self.load_from_variable(element_pointer, element_type)

        if function_name == "buffer_store":
            if len(args) < 3:
                self.refuse_lowering(
                    "buffer_store requires buffer, index, and value operands"
                )
                return None

            if len(args) > 3:
                self.refuse_lowering(
                    "buffer_store accepts only buffer, index, and " "value operands"
                )
                return None

            metadata = self.structured_buffer_metadata_for_pointer(args[0])
            if metadata is None:
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if (
                not metadata.get("byte_address")
                and metadata.get("buffer_kind") != "RWStructuredBuffer"
            ):
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None
            if metadata.get("readonly"):
                self.refuse_lowering("buffer_store requires an RWStructuredBuffer")
                return None

            element_pointer = self.structured_buffer_element_pointer(args[0], args[1])
            if element_pointer is None:
                self.refuse_lowering(
                    "buffer_store requires an RWStructuredBuffer operand"
                )
                return None

            self.store_to_variable(element_pointer, args[2])
            return None
        return UNHANDLED

    def spirv_storage_image_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit subpass loads and storage-image loads and stores."""
        if function_name == "subpassLoad":
            if not args:
                self.refuse_lowering("subpassLoad requires an input attachment operand")
                result_type = self.register_vector_type(
                    self.register_primitive_type("float"), 4
                )
                return self.default_value_for_type(result_type)

            image_id = args[0]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "input_attachment":
                self.refuse_lowering("subpassLoad requires a subpassInput operand")
                result_type = self.register_vector_type(
                    self.register_primitive_type("float"), 4
                )
                return self.default_value_for_type(result_type)

            result_type = self.resource_access_result_type(metadata)
            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 2:
                    self.refuse_lowering(
                        "subpassLoad on subpassInputMS requires " "a sample operand"
                    )
                    return self.default_value_for_type(result_type)
                if not self.validate_storage_image_sample("subpassLoad", args[1]):
                    return self.default_value_for_type(result_type)
                image_operands = f" Sample %{args[1].id}"

            coord_id = self.input_attachment_zero_coordinate()
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpImageRead %{result_type.id} "
                f"%{image_id.id} %{coord_id.id}{image_operands}"
            )
            self.value_types[id_value] = result_type
            return SpirvId(id_value, result_type.type)

        if function_name == "imageLoad":
            if len(args) < 2:
                self.refuse_lowering("imageLoad requires image and coordinate operands")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            image_id, coord_id = args[0], args[1]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.refuse_lowering("imageLoad requires a storage image operand")
                return self.register_constant(
                    0.0, self.register_primitive_type("float")
                )

            result_type = self.resource_access_result_type(metadata)
            if metadata.get("writeonly"):
                self.refuse_lowering("imageLoad requires a readable storage image")
                return self.default_value_for_type(result_type)

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return self.default_value_for_type(result_type)

            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 3:
                    self.refuse_lowering("imageLoad requires a sample operand")
                    return self.default_value_for_type(result_type)
                if not self.validate_storage_image_sample(function_name, args[2]):
                    return self.default_value_for_type(result_type)
                image_operands = f" Sample %{args[2].id}"

            self.require_storage_image_without_format_capability(
                metadata, "StorageImageReadWithoutFormat"
            )
            return emit_four_component_image_read(
                self,
                result_type,
                image_id,
                coord_id,
                image_operands,
                metadata.get("component_count"),
            )

        if function_name == "imageStore":
            if len(args) < 3:
                self.refuse_lowering(
                    "imageStore requires image, coordinate, and value operands"
                )
                return None

            image_id, coord_id, texel_id = args[0], args[1], args[2]
            metadata = self.resource_metadata_for_value(image_id)
            if not metadata or metadata.get("kind") != "storage_image":
                self.refuse_lowering("imageStore requires a storage image operand")
                return None
            if metadata.get("readonly"):
                self.refuse_lowering("imageStore requires a writable storage image")
                return None

            if not self.validate_storage_image_coordinate(
                function_name, metadata, coord_id
            ):
                return None

            texel_type = self.resource_access_result_type(metadata)
            image_operands = ""
            if metadata.get("multisampled"):
                if len(args) < 4:
                    self.refuse_lowering("imageStore requires a sample operand")
                    return None
                sample_id, texel_id = args[2], args[3]
                if not self.validate_storage_image_sample(function_name, sample_id):
                    return None
                image_operands = f" Sample %{sample_id.id}"

            texel_id = self.convert_storage_image_value_operand(
                function_name,
                "value",
                narrow_image_texel(self, texel_id, texel_type),
                texel_type,
            )
            if texel_id is None:
                return None

            self.require_storage_image_without_format_capability(
                metadata, "StorageImageWriteWithoutFormat"
            )
            self.emit(
                f"OpImageWrite %{image_id.id} %{coord_id.id} %{texel_id.id}"
                f"{image_operands}"
            )
            return None
        return UNHANDLED
