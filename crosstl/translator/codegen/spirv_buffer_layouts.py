"""Buffer declarations and layouts in the SPIR-V generator: constant buffers, structured buffer descriptors, GLSL buffer blocks, readonly storage, memory qualifiers, resource metadata, row-major uniforms, specialization-sized members, bfloat16 storage, storage buffer block decorations and storage classes, member offsets, array strides and statement lists."""

from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import VulkanRefusal
from ..semantics.shader_records import (
    is_shader_record_buffer,
    shader_record_memory_layout,
)
from ..vulkan_memory_scope import needs_device_coherence
from .memory_qualifier_names import memory_qualifier_name
from .spirv_exact_lookups import SPIRV_MEMORY_QUALIFIERS
from .spirv_placed_blocks import (
    SCALAR_BLOCK_LAYOUT_SOURCE_EXTENSION,
    decorate_placed_uniform_block,
    placed_member_type,
    placed_uniform_block,
)
from .spirv_push_constants import (
    declare_push_constant_block,
    is_push_constant_block,
    uniform_block_storage_layout,
)
from .spirv_row_major_arrays import (
    row_major_array_has_own_type,
    uniform_block_member_layout_type,
)
from .spirv_values import (
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class SpirvBufferLayoutMixin:
    """Buffer layout members of ``VulkanSPIRVCodeGen``."""

    def process_cbuffer_declaration(self, cbuffer_node: StructNode) -> SpirvId:
        """Emit a CrossGL cbuffer as a SPIR-V Uniform block."""
        members = []
        row_major_members = set()
        for member in getattr(cbuffer_node, "members", []) or []:
            member_type_source = getattr(
                member,
                "member_type",
                getattr(member, "var_type", getattr(member, "vtype", None)),
            )
            if member_type_source is None:
                continue
            if self.uniform_member_is_row_major(member):
                row_major_members.add(len(members))
            members.append((self.map_crossgl_type(member_type_source), member.name))

        if is_push_constant_block(self, cbuffer_node):
            if row_major_members:
                raise UnsupportedSPIRVFeatureError(
                    "row-major push-constant member",
                    f"push-constant block {cbuffer_node.name} has row-major matrix members",
                    missing_capabilities=("spirv.row_major_layout",),
                )
            var_id, members = declare_push_constant_block(self, cbuffer_node, members)
        else:
            placed = placed_uniform_block(self, cbuffer_node)
            layout = (
                placed.clone_key
                if placed is not None
                else uniform_block_storage_layout(self, cbuffer_node, row_major_members)
            )
            # Block members use layout clones so that ArrayStride/Offset
            # decorations never reach the logical types used by Function,
            # Private or Workgroup variables (VUID-StandaloneSpirv-None-10684).
            members = [
                (
                    (
                        placed_member_type(self, member_type, member_name, placed)
                        if placed is not None
                        else uniform_block_member_layout_type(
                            self,
                            member_type,
                            layout,
                            member_index in row_major_members,
                        )
                    ),
                    member_name,
                )
                for member_index, (member_type, member_name) in enumerate(members)
            ]
            cbuffer_type = self.register_struct_type(cbuffer_node.name, members)
            if placed is not None:
                decorate_placed_uniform_block(self, cbuffer_type, placed)
            elif layout == "std140":
                self.decorate_cbuffer_type(cbuffer_type, members, row_major_members)
            else:
                self.decorate_storage_buffer_block_type(
                    cbuffer_type, layout, block_decoration="Block"
                )
                self.require_source_extension(SCALAR_BLOCK_LAYOUT_SOURCE_EXTENSION)

            var_id = self.create_variable(cbuffer_type, "Uniform", cbuffer_node.name)
            descriptor_set, binding = self.resource_descriptor_slot(cbuffer_node)
            self.decorations.append(
                f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
            )
            self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        # HLSL and CrossGL constant-buffer names are not in expression scope: in ``cbuffer ubo { UBO ubo; }`` the
        # identifier ``ubo`` names the member, so the block keeps its name only for reflection.
        if all(member_name != cbuffer_node.name for _, member_name in members):
            self.global_variables[cbuffer_node.name] = var_id
        self.cbuffer_variables[cbuffer_node.name] = var_id
        self.uniform_buffers.append(var_id)
        for member_index, (member_type, member_name) in enumerate(members):
            if member_name in self.cbuffer_members:
                raise VulkanRefusal(f"Ambiguous SPIR-V cbuffer member {member_name}")
            self.cbuffer_members[member_name] = (var_id, member_type, member_index)

        return var_id

    def process_structured_buffer_declaration(
        self, node: VariableNode, type_name: str
    ) -> SpirvId:
        """Emit a StructuredBuffer/RWStructuredBuffer as a Vulkan BufferBlock."""
        metadata = self.structured_buffer_declared_type_info(type_name)
        if metadata is None:
            raise VulkanRefusal(f"Invalid SPIR-V structured buffer type {type_name}")
        memory_flags = self.storage_buffer_memory_flags(
            node, default_readonly=metadata.get("readonly", False)
        )
        if (
            metadata.get("default_writeonly")
            and not memory_flags.get("readonly")
            and not memory_flags.get("readwrite")
        ):
            memory_flags["writeonly"] = True

        element_type = self.storage_layout_type(
            self.map_crossgl_type(metadata["element_type_name"]), "std430"
        )
        if self.type_contains_bfloat16(element_type):
            self.require_bfloat16_storage()
        self.decorate_storage_buffer_nested_type(element_type)
        runtime_array_type = self.register_array_type(element_type, None)
        self.decorations.append(
            f"OpDecorate %{runtime_array_type.id} "
            f"ArrayStride {self.storage_array_stride(element_type)}"
        )

        block_name = f"{node.name}Buffer"
        block_members = [(runtime_array_type, node.name)]
        block_type = self.register_struct_type(
            self.unique_struct_type_name(block_name, block_members),
            block_members,
        )
        self.decorations.append(
            f"OpDecorate %{block_type.id} {self.storage_buffer_block_decoration()}"
        )
        self.decorations.append(f"OpMemberDecorate %{block_type.id} 0 Offset 0")
        # A matrix element needs its column order and stride (Vulkan 1.3 section 15.6.4, "Offset and Stride
        # Assignment"); the runtime array member carries them.
        matrix_stride = self.storage_matrix_stride_for_member(element_type, "std430")
        if matrix_stride is not None:
            self.decorations.append(f"OpMemberDecorate %{block_type.id} 0 ColMajor")
            self.decorations.append(
                f"OpMemberDecorate %{block_type.id} 0 MatrixStride {matrix_stride}"
            )
        self.decorate_storage_buffer_member_memory_qualifiers(block_type, memory_flags)

        variable_type, is_descriptor_array = (
            self.structured_buffer_descriptor_variable_type(type_name, block_type)
        )
        var_id = self.create_variable(
            variable_type, self.storage_buffer_storage_class(), node.name
        )
        descriptor_set, binding = self.resource_descriptor_slot(node)
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        buffer_metadata = {
            **metadata,
            **memory_flags,
            "buffer_variable": var_id,
            "declaration_node": node,
            "declared_type_name": type_name,
            "element_type": element_type,
            "runtime_array_type": runtime_array_type,
            "block_type": block_type,
            "member_index": 0,
        }
        if is_descriptor_array:
            buffer_metadata["descriptor_array"] = True
            buffer_metadata["descriptor_array_type"] = variable_type
        if metadata.get("append_consume"):
            counter_metadata = self.process_structured_buffer_counter_declaration(
                node, type_name
            )
            buffer_metadata.update(counter_metadata)
        self.global_variables[node.name] = var_id
        self.structured_buffer_metadata[var_id.id] = buffer_metadata
        self.structured_buffer_metadata[block_type.id] = buffer_metadata
        return var_id

    def process_structured_buffer_counter_declaration(
        self, node: VariableNode, type_name: str
    ) -> dict:
        """Emit the sidecar counter SSBO used for structured-buffer counters."""
        uint_type = self.register_primitive_type("uint")
        block_name = f"{node.name}CounterBuffer"
        block_type = self.register_struct_type(block_name, [(uint_type, "counter")])
        self.decorations.append(
            f"OpDecorate %{block_type.id} {self.storage_buffer_block_decoration()}"
        )
        self.decorations.append(f"OpMemberDecorate %{block_type.id} 0 Offset 0")

        variable_type, is_descriptor_array = (
            self.structured_buffer_descriptor_variable_type(type_name, block_type)
        )
        counter_name = f"{node.name}Counter"
        var_id = self.create_variable(
            variable_type, self.storage_buffer_storage_class(), counter_name
        )
        descriptor_set = self.resource_descriptor_set(node)
        binding = self.next_available_resource_binding(descriptor_set)
        self.used_resource_bindings.add((descriptor_set, binding))
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        counter_access_metadata = {
            "kind": "structured_buffer_counter",
            "block_type": block_type,
            "member_index": 0,
            "element_type": uint_type,
            "readonly": False,
            "writeonly": False,
        }
        self.storage_buffer_access_metadata[var_id.id] = counter_access_metadata
        self.storage_buffer_access_metadata[block_type.id] = counter_access_metadata
        if is_descriptor_array:
            self.storage_buffer_access_metadata[variable_type.id] = (
                counter_access_metadata
            )

        return {
            "counter_variable": var_id,
            "counter_block_type": block_type,
            "counter_variable_type": variable_type,
            "counter_descriptor_array": is_descriptor_array,
            "counter_member_index": 0,
        }

    def structured_buffer_descriptor_variable_type(
        self, type_name: str, block_type: SpirvId
    ) -> Tuple[SpirvId, bool]:
        array_info = self.split_outer_array_type(type_name)
        if array_info is None:
            return block_type, False

        element_type_name, size = array_info
        element_type, _ = self.structured_buffer_descriptor_variable_type(
            element_type_name, block_type
        )
        if size is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")
        return self.register_array_type(element_type, size), True

    def is_glsl_buffer_block_node(self, node: VariableNode) -> bool:
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }
        return "buffer" in qualifiers or self.has_attribute(node, "glsl_buffer_block")

    def glsl_buffer_block_layout(self, node: VariableNode) -> str:
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "glsl_buffer_block":
                continue
            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                continue
            layout = self.attribute_value_to_string(arguments[0])
            if layout is not None:
                return str(layout).lower()
        return "std430"

    def process_glsl_buffer_block_declaration(
        self,
        node: VariableNode,
        type_name: str,
        *,
        allow_binding_reassignment: bool = False,
    ) -> SpirvId:
        """Emit a GLSL-style buffer block or buffer-qualified array variable."""
        shader_record = is_shader_record_buffer(node)
        layout = (
            shader_record_memory_layout(node)
            if shader_record
            else self.glsl_buffer_block_layout(node)
        )
        base_type_name = self.array_base_type_name(type_name)
        outer_array = self.split_outer_array_type(type_name)
        pointer_element_type_name = self.pointer_pointee_type_name_from_string(
            type_name
        )
        is_pointer_descriptor_array = False
        pointer_descriptor_array_size = None
        if pointer_element_type_name is None and outer_array is not None:
            pointer_element_type_name = self.pointer_pointee_type_name_from_string(
                outer_array[0]
            )
            if pointer_element_type_name is not None:
                is_pointer_descriptor_array = True
                pointer_descriptor_array_size = outer_array[1]
        is_named_block = base_type_name in self.struct_types
        if is_named_block and outer_array is not None and outer_array[1] is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")
        if is_pointer_descriptor_array and pointer_descriptor_array_size is None:
            self.require_capability("RuntimeDescriptorArray")
            self.require_extension("SPV_EXT_descriptor_indexing")

        if is_named_block:
            value_type = self.storage_layout_type(
                self.map_crossgl_type(type_name), layout
            )
            block_type = self.storage_layout_type(
                self.struct_types[base_type_name], layout
            )
            block_members = self.current_struct_members.get(base_type_name, [])
            block_members = self.current_struct_members.get(
                block_type.type.base_type, block_members
            )
            variable_member_name = None
            variable_member_type = None
        else:
            variable_member_name = node.name
            if pointer_element_type_name is not None:
                element_type_name = self.storage_buffer_element_type_name(
                    pointer_element_type_name
                )
                element_type = self.storage_layout_type(
                    self.map_crossgl_type(element_type_name),
                    layout,
                )
                variable_member_type = self.register_array_type(element_type, None)
            else:
                variable_member_type = self.storage_layout_type(
                    self.map_crossgl_type(
                        getattr(node, "var_type", getattr(node, "vtype", "float"))
                    ),
                    layout,
                )
            block_type = self.register_struct_type(
                (
                    f"{node.name}Buffer"
                    if layout == "std430"
                    else f"{node.name}Buffer_{layout}"
                ),
                [(variable_member_type, variable_member_name)],
            )
            value_type = block_type
            block_members = [(variable_member_type, variable_member_name)]

        if is_pointer_descriptor_array:
            value_type = self.register_array_type(
                block_type, pointer_descriptor_array_size
            )

        self.decorate_storage_buffer_block_type(
            block_type, layout, block_decoration="Block" if shader_record else None
        )
        memory_flags = self.storage_buffer_memory_flags(node)
        self.decorate_storage_buffer_member_memory_qualifiers(block_type, memory_flags)

        if shader_record:
            # SPV_KHR_ray_tracing: the shader binding table record, read through no descriptor.
            self.validate_shader_record_buffer(node, value_type, block_type)
            var_id = self.create_variable(
                value_type, "ShaderRecordBufferKHR", node.name
            )
        else:
            var_id = self.create_variable(
                value_type, self.storage_buffer_storage_class(), node.name
            )
            descriptor_set, binding = self.resource_descriptor_slot(
                node, allow_binding_reassignment=allow_binding_reassignment
            )
            self.decorations.append(
                f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
            )
            self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")

        self.global_variables[node.name] = var_id
        self.register_glsl_buffer_access_metadata(node, var_id, value_type, block_type)
        self.register_single_array_storage_buffer_metadata(
            node,
            var_id,
            block_type,
            block_members,
            variable_member_name,
            variable_member_type,
            descriptor_array_type=(value_type if is_pointer_descriptor_array else None),
        )
        return var_id

    def register_single_array_storage_buffer_metadata(
        self,
        node: VariableNode,
        var_id: SpirvId,
        block_type: SpirvId,
        block_members: List[Tuple[SpirvId, str]],
        variable_member_name: Optional[str],
        variable_member_type: Optional[SpirvId],
        descriptor_array_type: Optional[SpirvId] = None,
    ):
        if len(block_members) != 1:
            return

        member_type, member_name = block_members[0]
        array_info = self.array_type_info_from_type(member_type)
        if array_info is None:
            return

        element_type, _ = array_info
        memory_flags = self.storage_buffer_memory_flags(node)
        metadata = {
            "kind": "structured_buffer",
            "buffer_kind": (
                "StructuredBuffer"
                if memory_flags.get("readonly")
                else "RWStructuredBuffer"
            ),
            "element_type_name": element_type.type.base_type,
            **memory_flags,
            "element_type": element_type,
            "runtime_array_type": member_type,
            "block_type": block_type,
            "member_index": 0,
            "member_name": variable_member_name or member_name,
            "declared_type_name": self.type_name_from_value(
                getattr(node, "var_type", getattr(node, "vtype", None))
            ),
        }
        if descriptor_array_type is not None:
            metadata["descriptor_array"] = True
            metadata["descriptor_array_type"] = descriptor_array_type
        self.structured_buffer_metadata[var_id.id] = metadata
        self.structured_buffer_metadata[block_type.id] = metadata
        if variable_member_type is not None:
            self.structured_buffer_metadata[variable_member_type.id] = metadata
        if descriptor_array_type is not None:
            self.structured_buffer_metadata[descriptor_array_type.id] = metadata

    def register_glsl_buffer_access_metadata(
        self,
        node: VariableNode,
        var_id: SpirvId,
        value_type: SpirvId,
        block_type: SpirvId,
    ):
        memory_flags = self.storage_buffer_memory_flags(node)
        metadata = {
            "kind": "glsl_buffer_block",
            **memory_flags,
            "block_type": block_type,
        }
        self.storage_buffer_access_metadata[var_id.id] = metadata
        self.storage_buffer_access_metadata[block_type.id] = metadata
        if value_type.id != block_type.id:
            self.storage_buffer_access_metadata[value_type.id] = metadata

    def storage_buffer_is_readonly(self, node: VariableNode) -> bool:
        return self.storage_buffer_memory_flags(node).get("readonly", False)

    def storage_buffer_is_writeonly(self, node: VariableNode) -> bool:
        return self.storage_buffer_memory_flags(node).get("writeonly", False)

    def resource_memory_qualifier_names(self, node) -> set:
        qualifiers = set()

        for qualifier in getattr(node, "qualifiers", []) or []:
            canonical = SPIRV_MEMORY_QUALIFIERS.get(memory_qualifier_name(qualifier))
            if canonical is not None:
                qualifiers.add(canonical)

        for qualifier in getattr(node, "resource_qualifiers", []) or []:
            canonical = SPIRV_MEMORY_QUALIFIERS.get(memory_qualifier_name(qualifier))
            if canonical is not None:
                qualifiers.add(canonical)

        if {"coherent(system)", "coherent(host)"} & {
            memory_qualifier_name(qualifier)
            for qualifier in [
                *(getattr(node, "qualifiers", []) or []),
                *(getattr(node, "resource_qualifiers", []) or []),
            ]
        }:
            # Vulkan's Coherent reaches the device at most; the CrossDevice scope is not allowed.
            self.refuse_lowering(
                f"resource '{getattr(node, 'name', '<unnamed>')}' is coherent at system "
                "scope: Vulkan SPIR-V has no coherence wider than the device"
            )

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue

            attr_name = str(attr_name).lower()
            canonical = SPIRV_MEMORY_QUALIFIERS.get(attr_name)
            if canonical is not None:
                qualifiers.add(canonical)

            if attr_name != "access":
                continue

            arguments = getattr(attr, "arguments", None)
            if arguments is None:
                arguments = getattr(attr, "args", [])
            if not arguments:
                continue

            access_name = self.attribute_value_to_string(arguments[0])
            if access_name is None:
                continue
            canonical = SPIRV_MEMORY_QUALIFIERS.get(str(access_name).lower())
            if canonical is not None:
                qualifiers.add(canonical)

        if "globallycoherent" in qualifiers or needs_device_coherence(node):
            # Coherent under the GLSL450 memory model is device-wide, which holds every scope
            # GL_KHR_memory_scope_semantics names.
            qualifiers.add("coherent")

        return qualifiers

    def resource_memory_qualifier_flags(self, node) -> dict:
        qualifiers = self.resource_memory_qualifier_names(node)
        readwrite = "readwrite" in qualifiers
        return {
            "readonly": "readonly" in qualifiers and not readwrite,
            "writeonly": "writeonly" in qualifiers and not readwrite,
            "coherent": "coherent" in qualifiers,
            "volatile": "volatile" in qualifiers,
            "restrict": "restrict" in qualifiers,
            "readwrite": readwrite,
        }

    def storage_buffer_memory_flags(
        self, node: VariableNode, default_readonly: bool = False
    ) -> dict:
        flags = self.resource_memory_qualifier_flags(node)
        if (
            default_readonly
            and not flags.get("writeonly")
            and not flags.get("readwrite")
        ):
            flags["readonly"] = True
        return flags

    def resource_memory_decoration_names(self, flags: dict) -> List[str]:
        decorations = []
        if flags.get("readonly"):
            decorations.append("NonWritable")
        if flags.get("writeonly"):
            decorations.append("NonReadable")
        if flags.get("coherent"):
            decorations.append("Coherent")
        if flags.get("volatile"):
            decorations.append("Volatile")
        if flags.get("restrict"):
            decorations.append("Restrict")
        return decorations

    def decorate_storage_buffer_member_memory_qualifiers(
        self, block_type: SpirvId, flags: dict
    ):
        decoration_names = self.resource_memory_decoration_names(flags)
        if not decoration_names:
            return

        members = self.current_struct_members.get(block_type.type.base_type, [])
        for member_index in range(len(members)):
            for decoration_name in decoration_names:
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} "
                    f"{member_index} {decoration_name}"
                )

    def decorate_resource_variable_memory_qualifiers(
        self, var_id: SpirvId, metadata: dict
    ):
        for decoration_name in self.resource_memory_decoration_names(metadata):
            self.decorations.append(f"OpDecorate %{var_id.id} {decoration_name}")

    def metadata_with_resource_memory_qualifiers(self, metadata: dict, node) -> dict:
        return {
            **metadata,
            **self.resource_memory_qualifier_flags(node),
        }

    def resource_metadata_for_declared_type(self, type_id: SpirvId):
        metadata = self.resource_type_metadata.get(type_id.id)
        if metadata is not None:
            return metadata

        element_type = self.array_element_type_from_type(type_id)
        while element_type is not None:
            metadata = self.resource_type_metadata.get(element_type.id)
            if metadata is not None:
                return metadata
            element_type = self.array_element_type_from_type(element_type)

        return None

    def register_declared_resource_metadata(
        self,
        node: VariableNode,
        var_id: SpirvId,
        type_id: SpirvId,
        decorate: bool = True,
    ):
        """Record a resource's metadata; ``decorate`` adds its memory decorations, which need a variable."""
        metadata = self.resource_metadata_for_declared_type(type_id)
        if metadata is None or metadata.get("kind") not in {
            "sampled_image",
            "storage_image",
            "input_attachment",
            "texture",
            "sampler",
        }:
            return

        metadata = self.metadata_with_resource_memory_qualifiers(metadata, node)
        self.resource_type_metadata[var_id.id] = metadata
        if decorate and metadata.get("kind") == "storage_image":
            self.decorate_resource_variable_memory_qualifiers(var_id, metadata)

    def decorate_cbuffer_type(
        self,
        cbuffer_type: SpirvId,
        members: List[Tuple[SpirvId, str]],
        row_major_members=frozenset(),
    ):
        """Decorate a uniform block with std140 offsets (GLSL 4.60 section 7.6.2.2).

        ``row_major_members`` holds the indices of matrix members stored row by row: they are laid out as arrays
        of their rows and decorated ``RowMajor``.
        """
        if self.type_contains_bfloat16(cbuffer_type):
            self.require_bfloat16_storage(uniform=True)
        self.validate_specialization_sized_member_layout(
            cbuffer_type.type.base_type, members
        )
        self.decorations.append(f"OpDecorate %{cbuffer_type.id} Block")
        # The block's own members are decorated below, with their orientation; only nested types here. A row-major
        # member holds only matrices, whose arrays take their row-major strides below.
        decorated_structs = {cbuffer_type.id}
        for member_index, (member_type, _member_name) in enumerate(members):
            if member_index not in row_major_members:
                self.decorate_uniform_nested_type(member_type, decorated_structs)

        offset = 0
        for member_index, (member_type, member_name) in enumerate(members):
            row_major = member_index in row_major_members
            offset = self.align_to(
                offset, self.uniform_layout_alignment(member_type, row_major)
            )
            self.decorations.append(
                f"OpMemberDecorate %{cbuffer_type.id} {member_index} Offset {offset}"
            )
            self.decorate_uniform_matrix_member(
                cbuffer_type, member_index, member_type, row_major, member_name
            )
            self.decorate_uniform_array_strides(member_type, row_major)
            if member_index + 1 < len(members):
                offset += self.uniform_layout_size(member_type, row_major)

    @staticmethod
    def uniform_member_is_row_major(member) -> bool:
        """Whether a block member is declared ``row_major`` (GLSL ``layout(row_major)``, HLSL ``row_major``)."""
        names = {str(q).lower() for q in getattr(member, "qualifiers", None) or []}
        names.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(member, "attributes", None) or []
        )
        return "row_major" in names and "column_major" not in names

    def decorate_uniform_matrix_member(
        self, struct_type, member_index, member_type, row_major, member_name=""
    ):
        """``ColMajor`` or ``RowMajor`` and the ``MatrixStride`` of a uniform member holding matrices."""
        matrix_type = self.uniform_innermost_matrix(member_type)
        if matrix_type is None:
            if row_major:
                raise UnsupportedSPIRVFeatureError(
                    "row-major non-matrix uniform member",
                    f"uniform member {member_name} is declared row_major but holds no matrix",
                    missing_capabilities=("spirv.row_major_layout",),
                )
            return
        if row_major and not row_major_array_has_own_type(self, member_type):
            # The array type is shared with column-major uses, whose element stride differs.
            raise UnsupportedSPIRVFeatureError(
                "row-major array of non-square matrices",
                f"uniform member {member_name} is an array of row-major non-square matrices",
                missing_capabilities=("spirv.row_major_layout",),
            )
        orientation = "RowMajor" if row_major else "ColMajor"
        stride = self.uniform_matrix_stride(matrix_type, row_major)
        self.decorations.append(
            f"OpMemberDecorate %{struct_type.id} {member_index} {orientation}"
        )
        self.decorations.append(
            f"OpMemberDecorate %{struct_type.id} {member_index} MatrixStride {stride}"
        )

    def uniform_innermost_matrix(self, type_id: SpirvId):
        """The matrix type a member or array of members holds, or ``None``."""
        while type_id is not None:
            if self.matrix_type_info_from_type(type_id) is not None:
                return type_id
            array_info = self.array_type_info_from_type(type_id)
            if array_info is None:
                return None
            type_id = array_info[0]
        return None

    def validate_specialization_sized_member_layout(
        self, type_name: str, members: List[Tuple[SpirvId, str]]
    ):
        """Reject variable-size members that would shift later block offsets."""
        for member_type, member_name in members[:-1]:
            if not self.type_contains_specialization_sized_array(member_type):
                continue
            raise UnsupportedSPIRVFeatureError(
                "specialization-sized block member layout",
                "SPIR-V block layout cannot assign stable offsets after "
                f"specialization-sized member {type_name}.{member_name}",
                missing_capabilities=("spirv.specialization_array_layout",),
            )

    def decorate_uniform_nested_type(
        self, type_id: SpirvId, decorated_structs: Optional[set] = None
    ):
        if decorated_structs is None:
            decorated_structs = set()

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            stride = self.uniform_array_stride(element_type)
            self.decorations.append(f"OpDecorate %{type_id.id} ArrayStride {stride}")
            self.decorate_uniform_nested_type(element_type, decorated_structs)
            return

        members = self.current_struct_members.get(type_id.type.base_type)
        if members is None or type_id.id in decorated_structs:
            return

        decorated_structs.add(type_id.id)
        self.validate_specialization_sized_member_layout(
            type_id.type.base_type, members
        )
        # A struct member declared row_major (the row-major copies of semantics/row_major_struct_members.py)
        # is decorated RowMajor, as a block member is; the std140 copy names its source struct.
        row_major_names = self.uniform_struct_row_major_member_names(type_id)
        offset = 0
        for member_index, (member_type, member_name) in enumerate(members):
            row_major = member_name in row_major_names
            offset = self.align_to(
                offset, self.uniform_layout_alignment(member_type, row_major)
            )
            self.decorations.append(
                f"OpMemberDecorate %{type_id.id} {member_index} Offset {offset}"
            )
            self.decorate_uniform_matrix_member(
                type_id, member_index, member_type, row_major, member_name
            )
            if row_major:
                self.decorate_uniform_array_strides(member_type, row_major)
            self.decorate_uniform_nested_type(member_type, decorated_structs)
            if member_index + 1 < len(members):
                offset += self.uniform_layout_size(member_type, row_major)

    def align_to(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def require_bfloat16_storage(self, *, uniform: bool = False):
        """Enable the Vulkan 16-bit storage contract for a block member."""
        self.require_extension("SPV_KHR_16bit_storage")
        self.require_capability(
            "UniformAndStorageBuffer16BitAccess"
            if uniform
            else "StorageBuffer16BitAccess"
        )

    def type_contains_bfloat16(
        self, type_id: Optional[SpirvId], visited: Optional[set] = None
    ) -> bool:
        if type_id is None:
            return False
        if visited is None:
            visited = set()
        if type_id.id in visited:
            return False
        visited.add(type_id.id)

        if (
            self.normalize_primitive_name(type_id.type.base_type)
            == self.BFLOAT16_TYPE_NAME
        ):
            return True
        vector_info = self.vector_component_type_and_count(type_id.type.base_type)
        if vector_info is not None:
            return vector_info[0] == self.BFLOAT16_TYPE_NAME
        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            return self.type_contains_bfloat16(array_info[0], visited)
        return any(
            self.type_contains_bfloat16(member_type, visited)
            for member_type, _ in self.current_struct_members.get(
                type_id.type.base_type, []
            )
        )

    def uniform_layout_alignment(self, type_id: SpirvId, row_major=False) -> int:
        """The std140 base alignment of ``type_id`` (GLSL 4.60 section 7.6.2.2, rules 1-10)."""
        if self.matrix_type_info_from_type(type_id) is not None:
            return self.uniform_matrix_stride(type_id, row_major)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.align_to(
                self.uniform_layout_alignment(element_type, row_major), 16
            )

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            row_major_names = self.uniform_struct_row_major_member_names(type_id)
            return self.align_to(
                max(
                    (
                        self.uniform_layout_alignment(
                            member_type, member_name in row_major_names
                        )
                        for member_type, member_name in struct_members
                    ),
                    default=1,
                ),
                16,
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            component_size = self.uniform_scalar_size(component_type)
            return component_size * (2 if component_count == 2 else 4)

        return self.uniform_scalar_size(type_id)

    def uniform_matrix_vectors(self, type_id: SpirvId, row_major=False):
        """``(component type, components, vectors)`` of the vectors a matrix is stored as: its columns, or its
        rows when it is row-major (rules 5 and 7)."""
        column_type, column_count = self.matrix_type_info_from_type(type_id)
        component_type, row_count = self.vector_type_info_from_type(column_type)
        if row_major:
            return component_type, column_count, row_count
        return component_type, row_count, column_count

    def uniform_matrix_stride(self, type_id: SpirvId, row_major=False) -> int:
        """The std140 stride between the stored vectors of a matrix: an array element's, a multiple of 16."""
        component_type, components, _vectors = self.uniform_matrix_vectors(
            type_id, row_major
        )
        component_size = self.uniform_scalar_size(component_type)
        return self.align_to(component_size * (2 if components == 2 else 4), 16)

    def uniform_layout_size(self, type_id: SpirvId, row_major=False) -> int:
        """The std140 size of ``type_id``; a three-component vector takes three components (rule 3)."""
        if self.matrix_type_info_from_type(type_id) is not None:
            _component, _components, vectors = self.uniform_matrix_vectors(
                type_id, row_major
            )
            return self.uniform_matrix_stride(type_id, row_major) * vectors

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            stride = self.uniform_array_stride(element_type, row_major)
            return stride * self.array_length_default_value(size)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            row_major_names = self.uniform_struct_row_major_member_names(type_id)
            offset = 0
            for member_type, member_name in struct_members:
                member_row_major = member_name in row_major_names
                alignment = self.uniform_layout_alignment(member_type, member_row_major)
                offset = self.align_to(offset, alignment)
                offset += self.uniform_layout_size(member_type, member_row_major)
            return self.align_to(offset, self.uniform_layout_alignment(type_id))

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            return self.uniform_scalar_size(component_type) * component_count

        return self.uniform_scalar_size(type_id)

    def uniform_struct_row_major_member_names(self, type_id: SpirvId) -> set:
        """Names of the members of struct ``type_id`` (or of the struct a layout copy was made from) declared
        row_major, such as the row-major copies of semantics/row_major_struct_members.py.
        """
        source_type = self.layout_struct_source_types.get(type_id.id, type_id)
        metadata = self.struct_member_metadata.get(source_type.type.base_type, {})
        return {
            name
            for name, entry in metadata.items()
            if (entry or {}).get("node") is not None
            and self.uniform_member_is_row_major(entry["node"])
        }

    def uniform_scalar_size(self, type_id: SpirvId) -> int:
        primitive_name = self.normalize_primitive_name(type_id.type.base_type)
        if primitive_name == self.BFLOAT16_TYPE_NAME:
            return 2
        return 8 if primitive_name in {"double", "i64", "u64"} else 4

    def uniform_array_stride(self, element_type: SpirvId, row_major=False) -> int:
        """The std140 array stride: the element size rounded up to its alignment rounded up to 16 (rule 4)."""
        return self.align_to(
            self.uniform_layout_size(element_type, row_major),
            self.align_to(self.uniform_layout_alignment(element_type, row_major), 16),
        )

    def uniform_layout_contains_matrix(self, type_id: SpirvId) -> bool:
        if self.matrix_type_info_from_type(type_id) is not None:
            return True

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.uniform_layout_contains_matrix(element_type)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return any(
                self.uniform_layout_contains_matrix(member_type)
                for member_type, _ in struct_members
            )

        return False

    def decorate_uniform_array_strides(self, type_id: SpirvId, row_major=False):
        array_info = self.array_type_info_from_type(type_id)
        if array_info is None:
            return

        element_type, _ = array_info
        stride = self.uniform_array_stride(element_type, row_major)
        self.decorations.append(f"OpDecorate %{type_id.id} ArrayStride {stride}")
        self.decorate_uniform_array_strides(element_type, row_major)

    def storage_layout_type(self, type_id: SpirvId, layout: str) -> SpirvId:
        if type_id.type.base_type == "bool":
            return self.register_primitive_type("uint")
        bool_vector = self.vector_component_type_and_count(type_id.type.base_type)
        if bool_vector is not None and bool_vector[0] == "bool":
            # Booleans have no externally visible size; blocks hold them as 32-bit uints
            # (GLSL 4.60 section 7.6.2.2), and loads convert back to booleans.
            uint_type = self.register_primitive_type("uint")
            return self.register_vector_type(uint_type, bool_vector[1])

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            layout_element_type = self.storage_layout_type(element_type, layout)
            return self.register_layout_array_type(layout_element_type, size, layout)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            if (
                layout == "std430"
                and type_id.type.base_type in self.std430_runtime_array_struct_types
            ):
                return type_id
            cloned_members = [
                (self.storage_layout_type(member_type, layout), member_name)
                for member_type, member_name in struct_members
            ]
            return self.register_layout_struct_type(type_id, layout, cloned_members)

        return type_id

    def decorate_storage_buffer_block_type(
        self,
        block_type: SpirvId,
        layout: str = "std430",
        block_decoration: Optional[str] = None,
    ):
        if self.type_contains_bfloat16(block_type):
            self.require_bfloat16_storage()
        self.decorations.append(
            f"OpDecorate %{block_type.id} "
            f"{block_decoration or self.storage_buffer_block_decoration()}"
        )
        members = self.current_struct_members.get(block_type.type.base_type, [])
        self.validate_specialization_sized_member_layout(
            block_type.type.base_type, members
        )
        for member_index, (member_type, _) in enumerate(members):
            offset = self.storage_struct_member_offset(block_type, member_index, layout)
            self.decorations.append(
                f"OpMemberDecorate %{block_type.id} {member_index} Offset {offset}"
            )
            matrix_stride = self.storage_matrix_stride_for_member(member_type, layout)
            if matrix_stride is not None:
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} {member_index} ColMajor"
                )
                self.decorations.append(
                    f"OpMemberDecorate %{block_type.id} {member_index} "
                    f"MatrixStride {matrix_stride}"
                )
            self.decorate_storage_buffer_nested_type(member_type, layout)

    def validate_shader_record_buffer(self, node, value_type, block_type):
        """A shader record buffer is one ``Block`` struct with no descriptor slot, read by a ray tracing stage."""
        if value_type is not block_type:
            raise VulkanRefusal(
                f"SPIR-V shader record buffer '{node.name}' must be a single block, not an array of blocks "
                "(GL_EXT_ray_tracing allows one shaderRecordEXT block per stage)"
            )
        if self.explicit_descriptor_binding(node) is not None:
            raise VulkanRefusal(
                f"SPIR-V shader record buffer '{node.name}' cannot declare a binding: the record is read "
                "through no descriptor (GL_EXT_ray_tracing)"
            )

    def uses_storage_buffer_storage_class(self) -> bool:
        """Whether storage buffers are ``StorageBuffer`` + ``Block`` rather than ``Uniform`` + ``BufferBlock``.

        The SPIR-V specification's Decoration table lists ``BufferBlock`` as missing after version 1.3, so a
        module that must be 1.4 or later declares its storage buffers in the ``StorageBuffer`` storage class
        (core since 1.3). The execution models are known before any buffer is declared.
        """
        return (
            self.cooperative_matrix_khr_enabled
            or self.include_resource_interface_variables
            or bool(
                set(getattr(self, "program_execution_models", ()) or ())
                & self.SPIRV_1_4_EXECUTION_MODELS
            )
        )

    def storage_buffer_storage_class(self) -> str:
        return (
            "StorageBuffer" if self.uses_storage_buffer_storage_class() else "Uniform"
        )

    def storage_buffer_block_decoration(self) -> str:
        return "Block" if self.uses_storage_buffer_storage_class() else "BufferBlock"

    def decorate_storage_buffer_nested_type(
        self, type_id: SpirvId, layout: str = "std430"
    ):
        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            self.validate_specialization_sized_member_layout(
                type_id.type.base_type, struct_members
            )
            for member_index, (member_type, _) in enumerate(struct_members):
                offset = self.storage_struct_member_offset(
                    type_id, member_index, layout
                )
                self.decorations.append(
                    f"OpMemberDecorate %{type_id.id} {member_index} Offset {offset}"
                )
                matrix_stride = self.storage_matrix_stride_for_member(
                    member_type, layout
                )
                if matrix_stride is not None:
                    self.decorations.append(
                        f"OpMemberDecorate %{type_id.id} {member_index} ColMajor"
                    )
                    self.decorations.append(
                        f"OpMemberDecorate %{type_id.id} {member_index} "
                        f"MatrixStride {matrix_stride}"
                    )
                self.decorate_storage_buffer_nested_type(member_type, layout)
            return

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            self.decorations.append(
                f"OpDecorate %{type_id.id} "
                f"ArrayStride {self.storage_array_stride(element_type, layout)}"
            )
            self.decorate_storage_buffer_nested_type(element_type, layout)

    def storage_struct_member_offset(
        self,
        struct_type: SpirvId,
        target_member_index: int,
        layout: str = "std430",
    ) -> int:
        offset = 0
        for member_index, (member_type, _) in enumerate(
            self.current_struct_members.get(struct_type.type.base_type, [])
        ):
            offset = self.align_to(
                offset, self.storage_layout_alignment(member_type, layout)
            )
            if member_index == target_member_index:
                return offset
            offset += self.storage_layout_size(member_type, layout)
        return offset

    def storage_layout_alignment(self, type_id: SpirvId, layout: str = "std430") -> int:
        if layout == "std140":
            return self.uniform_layout_alignment(type_id)
        if layout == "scalar":
            return self.scalar_layout_alignment(type_id)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.storage_layout_alignment(column_type, layout)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.storage_layout_alignment(element_type, layout)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return max(
                (
                    self.storage_layout_alignment(member_type, layout)
                    for member_type, _ in struct_members
                ),
                default=1,
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            if self.type_contains_bfloat16(component_type):
                component_size = self.uniform_scalar_size(component_type)
                return component_size * (2 if component_count == 2 else 4)
            if component_count == 2:
                return 8
            if component_count in {3, 4}:
                return 16

        return self.uniform_scalar_size(type_id)

    def storage_layout_size(self, type_id: SpirvId, layout: str = "std430") -> int:
        if layout == "std140":
            return self.uniform_layout_size(type_id)
        if layout == "scalar":
            return self.scalar_layout_size(type_id)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            return self.storage_array_stride(column_type, layout) * column_count

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return self.storage_array_stride(
                element_type, layout
            ) * self.array_length_default_value(size)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            offset = 0
            max_alignment = 1
            for member_type, _ in struct_members:
                alignment = self.storage_layout_alignment(member_type, layout)
                max_alignment = max(max_alignment, alignment)
                offset = self.align_to(offset, alignment)
                offset += self.storage_layout_size(member_type, layout)
            return self.align_to(offset, max_alignment)

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            return self.uniform_scalar_size(component_type) * component_count

        return self.uniform_scalar_size(type_id)

    def storage_array_stride(
        self, element_type: SpirvId, layout: str = "std430"
    ) -> int:
        if layout == "std140":
            return self.uniform_array_stride(element_type)
        if layout == "scalar":
            return self.scalar_array_stride(element_type)

        return self.align_to(
            self.storage_layout_size(element_type, layout),
            self.storage_layout_alignment(element_type, layout),
        )

    def scalar_layout_alignment(self, type_id: SpirvId) -> int:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.scalar_layout_alignment(column_type)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.scalar_layout_alignment(element_type)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return max(
                (
                    self.scalar_layout_alignment(member_type)
                    for member_type, _ in struct_members
                ),
                default=1,
            )

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, _ = vector_info
            return self.uniform_scalar_size(component_type)

        return self.uniform_scalar_size(type_id)

    def scalar_layout_size(self, type_id: SpirvId) -> int:
        """The bytes a scalar-layout value takes before the next member may start.

        GL_EXT_scalar_block_layout reserves no padding after the last element of an array or the last member of a
        struct: glslang puts the member after ``struct { double d; float f; } s`` at 12 bytes past ``s``, although
        an array of that struct has a 16-byte stride.
        """
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, column_count = matrix_info
            return self.scalar_array_stride(column_type) * column_count

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            count = self.array_length_default_value(size)
            if count <= 0:
                return 0
            return self.scalar_array_stride(element_type) * (
                count - 1
            ) + self.scalar_layout_size(element_type)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            offset = 0
            for member_type, _ in struct_members:
                offset = self.align_to(
                    offset, self.scalar_layout_alignment(member_type)
                )
                offset += self.scalar_layout_size(member_type)
            return offset

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, component_count = vector_info
            return self.uniform_scalar_size(component_type) * component_count

        return self.uniform_scalar_size(type_id)

    def scalar_array_stride(self, element_type: SpirvId) -> int:
        return self.align_to(
            self.scalar_layout_size(element_type),
            self.scalar_layout_alignment(element_type),
        )

    def storage_matrix_stride_for_member(
        self, type_id: SpirvId, layout: str = "std430"
    ) -> Optional[int]:
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, _ = matrix_info
            return self.storage_array_stride(column_type, layout)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.storage_matrix_stride_for_member(element_type, layout)

        return None

    def execution_model_hint_for_function(self, function_node, stage=None):
        if stage is not None:
            return self.spirv_execution_model(self.stage_key(stage.stage))

        qualifier = self.get_function_qualifier(function_node)
        if qualifier is not None:
            return self.spirv_execution_model(qualifier)

        execution_models = self.execution_models_for_function(function_node)
        if "GLCompute" in execution_models:
            return "GLCompute"
        if len(execution_models) == 1:
            return next(iter(execution_models))
        return None

    def process_statements(self, statements):
        """Process a list of CrossGL statements."""
        if isinstance(statements, BlockNode):
            self.process_block_statement(statements)
            return
        if hasattr(statements, "statements"):
            stmt_list = statements.statements
        elif isinstance(statements, list):
            stmt_list = statements
        else:
            stmt_list = [statements]

        if self.inline_return_context_stack:
            self.process_inline_statement_sequence(stmt_list)
            return

        for stmt in stmt_list:
            if self.current_block_has_terminator():
                break
            self.process_statement(stmt)

    def process_inline_statement_sequence(self, statements):
        """Guard a statement suffix after its first possible inline return."""
        for index, stmt in enumerate(statements):
            if self.current_block_has_terminator():
                break
            self.process_statement(stmt)
            if not self.statement_contains_return(stmt):
                continue

            remaining = statements[index + 1 :]
            if remaining and not self.current_block_has_terminator():
                self.process_inline_fallthrough_statements(remaining)
            break

    def statement_contains_return(self, statement) -> bool:
        statement_id = id(statement)
        cached = self.inline_return_presence_by_node_id.get(statement_id)
        if cached is not None and cached[0] is statement:
            return cached[1]
        contains_return = any(
            isinstance(node, ReturnNode) or getattr(node, "is_tail_expression", False)
            for node in self.walk_ast_nodes(statement)
        )
        self.inline_return_presence_by_node_id[statement_id] = (
            statement,
            contains_return,
        )
        return contains_return

    def process_inline_fallthrough_statements(self, statements):
        """Run a statement suffix only when the inline call has not returned."""
        context = self.inline_return_context_stack[-1]
        bool_type = self.register_primitive_type("bool")
        returned = self.load_from_variable(context.returned_pointer, bool_type)
        should_continue = self.unary_operation("!", bool_type, returned)
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(should_continue, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.process_inline_statement_sequence(statements)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_statement(self, stmt):
        """Process a single CrossGL statement."""
        if isinstance(stmt, list):
            # A for-loop header keeps ``k = 0, m = 3`` and ``i++, j--`` as lists, run in order.
            for item in stmt:
                if item is not None:
                    self.process_statement(item)
            return
        if isinstance(stmt, AssignmentNode):
            self.process_assignment(stmt)
        elif isinstance(stmt, VariableNode):
            self.process_variable_declaration(stmt)
        elif isinstance(stmt, ReturnNode):
            self.process_return(stmt)
        elif isinstance(stmt, IfNode):
            self.process_if(stmt)
        elif isinstance(stmt, ForNode):
            self.process_for(stmt)
        elif isinstance(stmt, ForInNode):
            self.process_for_in(stmt)
        elif isinstance(stmt, WhileNode):
            self.process_while(stmt)
        elif isinstance(stmt, DoWhileNode):
            self.process_do_while(stmt)
        elif isinstance(stmt, LoopNode):
            self.process_loop(stmt)
        elif isinstance(stmt, SwitchNode):
            self.process_switch(stmt)
        elif isinstance(stmt, MatchNode):
            self.process_match(stmt)
        elif isinstance(stmt, BreakNode):
            self.process_break(stmt)
        elif isinstance(stmt, ContinueNode):
            self.process_continue(stmt)
        elif isinstance(stmt, BlockNode):
            self.process_block_statement(stmt)
        elif isinstance(stmt, FunctionCallNode):
            if self.process_fragment_discard_statement(stmt):
                return
            self.process_expression(stmt)  # Just evaluate and discard result
        elif isinstance(stmt, (UnaryOpNode, BinaryOpNode)):
            self.process_expression(stmt)
        elif hasattr(stmt, "expression"):
            expression = stmt.expression
            if self.process_fragment_discard_statement(expression):
                return
            if isinstance(expression, AssignmentNode):
                self.process_assignment(expression)
            elif (
                getattr(stmt, "is_tail_expression", False)
                and self.current_return_type is not None
                and self.current_return_type.type.base_type != "void"
            ):
                if self.inline_return_context_stack:
                    self.process_inline_return(ReturnNode(expression))
                else:
                    return_value = self.process_expression_with_expected_type(
                        expression, self.current_return_type_source
                    )
                    if return_value is not None:
                        self.create_return_value(return_value)
                    else:
                        self.create_return()
            else:
                self.process_expression(expression)
        elif stmt is not None:
            # Dropping a statement the emitter cannot place would change the program.
            self.refuse_lowering(
                f"statement {type(stmt).__name__} has no SPIR-V lowering"
            )

    def local_declaration_scope_state(self) -> dict:
        """Capture name and pointer-alias state owned by a lexical scope."""
        return {
            "local_variables": self.local_variables.copy(),
            "local_variable_types": self.local_variable_types.copy(),
            "named_constants": self.named_constants.copy(),
            "precise_local_variables": set(self.precise_local_variables),
            "resource_alias_variables": set(self.resource_alias_variables),
            "literal_int_constants": self.current_function_literal_int_constants.copy(),
            "literal_int_constant_shadows": set(
                self.current_function_literal_int_constant_shadows
            ),
            "pointer_offset_aliases": self.storage_buffer_pointer_offset_aliases.copy(),
            "pointer_reinterpretations": self.storage_pointer_reinterpretations.copy(),
        }
