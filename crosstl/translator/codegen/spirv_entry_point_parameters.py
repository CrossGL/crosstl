"""Resource types and entry point parameters in the SPIR-V generator: resource and structured buffer type info, image formats, input attachments, storage image coordinates and values, resource access metadata, semantics and color locations, built-in semantic types, return outputs, resource parameter qualifiers, and entry point descriptor parameters, uniforms and interface names."""

import re
from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    UnaryOpNode,
    VariableNode,
)
from ..refusals import VulkanRefusal
from .image_access_contracts import PACKED_IMAGE_FORMAT_CHANNEL_COUNTS
from .interface_qualifiers import INTERFACE_QUALIFIER_ATTRIBUTES
from .spirv_argument_tables import plan_entry_descriptor_slots
from .spirv_conservative_depth import DEPTH_LAYOUT_ATTRIBUTES, record_depth_layout
from .spirv_extension_builtins import unmapped_builtin_reason
from .spirv_inline_requirements import is_inline_spirv_attribute
from .spirv_interface_locations import (
    plan_entry_input_locations,
    plan_struct_member_locations,
)
from .spirv_interpolation import record_entry_input_variable
from .spirv_invariant_outputs import decorate_invariant_builtin_output
from .spirv_separate_textures import separate_texture_type_info
from .spirv_stage_distances import (
    entry_point_distance_layout,
    entry_point_distance_variable,
    load_entry_point_distance,
)
from .spirv_struct_varyings import (
    flatten_struct_varying,
    load_flattened_struct_varying,
)
from .spirv_texel_buffers import texel_buffer_type_info
from .spirv_unavailable_inputs import unread_unavailable_builtin_input
from .spirv_values import SpirvId


class SpirvEntryPointParameterMixin:
    """Entry point parameter members of ``VulkanSPIRVCodeGen``."""

    def resource_type_info(self, type_str: str):
        type_str = self.normalize_resource_type_name(type_str)
        input_attachment_info = self.input_attachment_type_info(type_str)
        if input_attachment_info is not None:
            return input_attachment_info

        sampler_info = {
            "sampler": {"kind": "sampler"},
            "texture1D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture2D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture3D": {
                "kind": "texture",
                "component_type": "float",
                "dim": "3D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "textureCube": {
                "kind": "texture",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "texture2DArray": {
                "kind": "texture",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "textureCubeArray": {
                "kind": "texture",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler1D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler1DArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "1D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler3D": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "3D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCube": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 1,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DArrayShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 1,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 1,
                "arrayed": 0,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "samplerCubeArrayShadow": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "Cube",
                "depth": 1,
                "arrayed": 1,
                "multisampled": 0,
                "sampled": 1,
                "format": "Unknown",
            },
            "comparison_sampler": {"kind": "sampler"},
            "sampler2DMS": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 0,
                "multisampled": 1,
                "sampled": 1,
                "format": "Unknown",
            },
            "sampler2DMSArray": {
                "kind": "sampled_image",
                "component_type": "float",
                "dim": "2D",
                "depth": 0,
                "arrayed": 1,
                "multisampled": 1,
                "sampled": 1,
                "format": "Unknown",
            },
        }
        if type_str in sampler_info:
            return sampler_info[type_str]

        if self.is_acceleration_structure_type_name(type_str):
            return {"kind": "acceleration_structure"}

        texel_buffer_info = texel_buffer_type_info(type_str)
        if texel_buffer_info is not None:
            return texel_buffer_info

        sampled_image_match = re.fullmatch(
            r"([iu]?sampler)(1D|2D|3D|Cube)(MS)?(Array)?(?:Shadow)?",
            type_str,
        )
        if sampled_image_match:
            prefix, dim, ms_suffix, array_suffix = sampled_image_match.groups()
            if ms_suffix and dim != "2D":
                return None
            component_type = {
                "sampler": "float",
                "isampler": "int",
                "usampler": "uint",
            }[prefix]
            return {
                "kind": "sampled_image",
                "component_type": component_type,
                "dim": dim,
                "depth": 1 if type_str.endswith("Shadow") else 0,
                "arrayed": 1 if array_suffix else 0,
                "multisampled": 1 if ms_suffix else 0,
                "sampled": 1,
                "format": "Unknown",
            }

        texture_info = separate_texture_type_info(type_str)
        if texture_info is not None:
            return texture_info

        image_match = re.fullmatch(
            r"([iu]?image)(1D|2D|3D|Cube)(MS)?(Array)?", type_str
        )
        if image_match:
            prefix, dim, ms_suffix, array_suffix = image_match.groups()
            if ms_suffix and dim != "2D":
                return None
            component_type = {
                "image": "float",
                "iimage": "int",
                "uimage": "uint",
            }[prefix]
            return {
                "kind": "storage_image",
                "component_type": component_type,
                "dim": "Cube" if dim == "Cube" else dim,
                "depth": 0,
                "arrayed": 1 if array_suffix else 0,
                "multisampled": 1 if ms_suffix else 0,
                "sampled": 2,
                "format": "Unknown",
            }

        return None

    def input_attachment_type_info(self, type_str: str):
        match = re.fullmatch(r"([iu]?)(subpassInput)(MS)?", str(type_str))
        if not match:
            return None

        prefix, _base, ms_suffix = match.groups()
        component_type = {
            "": "float",
            "i": "int",
            "u": "uint",
        }[prefix]
        return {
            "kind": "input_attachment",
            "component_type": component_type,
            "dim": "SubpassData",
            "depth": 0,
            "arrayed": 0,
            "multisampled": 1 if ms_suffix else 0,
            "sampled": 2,
            "format": "Unknown",
            "component_count": 4,
        }

    def is_input_attachment_type_name(self, type_str: str) -> bool:
        return (
            self.input_attachment_type_info(self.normalize_resource_type_name(type_str))
            is not None
        )

    def is_resource_type_name(self, type_str: str) -> bool:
        return self.resource_type_info(type_str) is not None

    def is_acceleration_structure_type_name(self, type_str: str) -> bool:
        compact = re.sub(r"\s+", "", str(type_str))
        return compact in {
            "accelerationStructureEXT",
            "AccelerationStructure",
            "RaytracingAccelerationStructure",
            "acceleration_structure",
        }

    def is_ray_query_type_name(self, type_str: str) -> bool:
        compact = re.sub(r"\s+", "", str(type_str))
        return compact in {"RayQuery", "rayQuery", "rayQueryEXT"} or compact.startswith(
            "RayQuery<"
        )

    def structured_buffer_type_info(self, type_str: str):
        type_str = re.sub(r"\s+", "", str(type_str))
        if type_str in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return {
                "kind": "structured_buffer",
                "buffer_kind": type_str,
                "element_type_name": "uint",
                "readonly": type_str == "ByteAddressBuffer",
                "byte_address": True,
            }

        match = re.fullmatch(
            r"(StructuredBuffer|RWStructuredBuffer|AppendStructuredBuffer|"
            r"ConsumeStructuredBuffer)<(.+)>",
            type_str,
        )
        if not match:
            return None

        buffer_kind, element_type_name = match.groups()
        return {
            "kind": "structured_buffer",
            "buffer_kind": buffer_kind,
            "element_type_name": element_type_name,
            "readonly": buffer_kind in {"StructuredBuffer", "ConsumeStructuredBuffer"},
            "default_writeonly": buffer_kind == "AppendStructuredBuffer",
            "append_consume": (
                buffer_kind in {"AppendStructuredBuffer", "ConsumeStructuredBuffer"}
            ),
        }

    def is_structured_buffer_type_name(self, type_str: str) -> bool:
        return self.structured_buffer_type_info(type_str) is not None

    def structured_buffer_declared_type_info(self, type_str: str):
        type_str = re.sub(r"\s+", "", str(type_str))
        metadata = self.structured_buffer_type_info(type_str)
        if metadata is not None:
            return metadata

        base_type = self.array_base_type_name(type_str)
        if base_type == type_str:
            return None

        return self.structured_buffer_type_info(base_type)

    def is_structured_buffer_declared_type_name(self, type_str: str) -> bool:
        return self.structured_buffer_declared_type_info(type_str) is not None

    def spirv_image_format_map(self):
        return {
            "r8": "R8",
            "r8_snorm": "R8Snorm",
            "r8i": "R8i",
            "r8ui": "R8ui",
            "r16": "R16",
            "r16_snorm": "R16Snorm",
            "r16f": "R16f",
            "r16i": "R16i",
            "r16ui": "R16ui",
            "r32f": "R32f",
            "r32i": "R32i",
            "r32ui": "R32ui",
            "rg8": "Rg8",
            "rg8_snorm": "Rg8Snorm",
            "rg8i": "Rg8i",
            "rg8ui": "Rg8ui",
            "rg16": "Rg16",
            "rg16_snorm": "Rg16Snorm",
            "rg16f": "Rg16f",
            "rg16i": "Rg16i",
            "rg16ui": "Rg16ui",
            "rg32f": "Rg32f",
            "rg32i": "Rg32i",
            "rg32ui": "Rg32ui",
            "rgba8": "Rgba8",
            "rgba8_snorm": "Rgba8Snorm",
            "rgba8i": "Rgba8i",
            "rgba8ui": "Rgba8ui",
            "rgba16": "Rgba16",
            "rgba16_snorm": "Rgba16Snorm",
            "rgba16f": "Rgba16f",
            "rgba16i": "Rgba16i",
            "rgba16ui": "Rgba16ui",
            "rgba32f": "Rgba32f",
            "rgba32i": "Rgba32i",
            "rgba32ui": "Rgba32ui",
            "rgb10_a2": "Rgb10A2",
            "rgb10_a2ui": "Rgb10a2ui",
            "r11f_g11f_b10f": "R11fG11fB10f",
        }

    def spirv_image_format_name(self, image_format: Optional[str]) -> Optional[str]:
        if image_format is None:
            return None
        return self.spirv_image_format_map().get(str(image_format).lower())

    def image_format_component_type(self, image_format: str) -> str:
        image_format = str(image_format).lower()
        if image_format.endswith("ui"):
            return "uint"
        if image_format.endswith("i") and not image_format.endswith("_snorm"):
            return "int"
        return "float"

    def image_format_component_count(self, image_format: Optional[str]) -> int:
        if image_format is None:
            return 4

        image_format = str(image_format).lower()
        packed = PACKED_IMAGE_FORMAT_CHANNEL_COUNTS.get(image_format)
        if packed is not None:
            return packed
        if image_format.startswith("rgba"):
            return 4
        if image_format.startswith("rg"):
            return 2
        if image_format.startswith("r"):
            return 1
        return 4

    def resource_access_result_type(self, metadata) -> SpirvId:
        component_type = self.register_primitive_type(
            metadata.get("component_type", "float")
        )
        component_count = int(metadata.get("component_count", 4))
        if component_count <= 1:
            return component_type
        return self.register_vector_type(component_type, component_count)

    def input_attachment_zero_coordinate(self) -> SpirvId:
        int_type = self.register_primitive_type("int")
        coord_type = self.register_vector_type(int_type, 2)
        zero = self.register_constant(0, int_type)
        return self.register_vector_constant(coord_type, [zero, zero])

    def image_coordinate_component_count(self, metadata) -> int:
        component_count = {
            "1D": 1,
            "Buffer": 1,
            "2D": 2,
            "Rect": 2,
            "3D": 3,
            "Cube": 3,
        }.get(metadata.get("dim", "2D"), 2)
        if metadata.get("arrayed"):
            component_count += 1
        return component_count

    def integer_value_component_count(self, value_id: SpirvId) -> Optional[int]:
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
            if component_type_name in {"int", "uint"}:
                return component_count
            return None

        if self.normalize_primitive_name(type_name) in {"int", "uint"}:
            return 1
        return None

    def validate_storage_image_coordinate(
        self, function_name: str, metadata, coord_id: SpirvId
    ) -> bool:
        expected_components = self.image_coordinate_component_count(metadata)
        actual_components = self.integer_value_component_count(coord_id)
        if actual_components == expected_components:
            return True

        self.refuse_lowering(
            f"{function_name} requires a {expected_components}-component "
            f"integer coordinate for {metadata.get('type_name', 'storage image')}"
        )
        return False

    def validate_storage_image_sample(
        self, function_name: str, sample_id: SpirvId
    ) -> bool:
        if self.integer_value_component_count(sample_id) == 1:
            return True

        self.refuse_lowering(
            f"{function_name} requires a scalar integer sample operand"
        )
        return False

    def require_storage_image_without_format_capability(
        self, metadata, capability: str
    ):
        if metadata.get("format", "Unknown") == "Unknown":
            self.require_capability(capability)

    def convert_storage_image_value_operand(
        self,
        function_name: str,
        operand_name: str,
        value_id: SpirvId,
        target_type: SpirvId,
    ) -> Optional[SpirvId]:
        converted = self.convert_value_to_type(value_id, target_type)
        if self.value_has_type(converted, target_type):
            return converted

        self.refuse_lowering(
            f"{function_name} {operand_name} operand does not match "
            f"{target_type.type.base_type} storage image texel type"
        )
        return None

    def resource_metadata_for_value(self, value_id: SpirvId):
        metadata = self.resource_type_metadata.get(value_id.id)
        if metadata is not None:
            return metadata

        result_type = self.value_types.get(value_id.id)
        if result_type is not None:
            metadata = self.resource_type_metadata.get(result_type.id)
            if metadata is not None:
                return metadata

        return None

    def resource_metadata_for_pointer(self, pointer_id: SpirvId):
        metadata = self.resource_type_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            metadata = self.resource_type_metadata.get(pointee_type.id)
            if metadata is not None:
                return metadata

        return self.resource_metadata_for_value(pointer_id)

    def propagate_resource_access_metadata(
        self, source_pointer: SpirvId, access_pointer: SpirvId, access_type: SpirvId
    ):
        target_metadata = self.resource_type_metadata.get(access_type.id)
        if target_metadata is None:
            return

        source_metadata = self.resource_metadata_for_pointer(source_pointer)
        if source_metadata is None or source_metadata.get(
            "kind"
        ) != target_metadata.get("kind"):
            source_metadata = target_metadata

        self.resource_type_metadata[access_pointer.id] = source_metadata

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        return str(value)

    def literal_int_argument(self, value) -> Optional[int]:
        """Return a literal integer value when an AST argument is constant."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, UnaryOpNode) and getattr(value, "op", None) == "-":
            operand = self.literal_int_argument(value.operand)
            return -operand if operand is not None else None
        value_text = self.attribute_value_to_string(value)
        if value_text is None:
            return None
        try:
            return int(str(value_text), 0)
        except ValueError:
            return None

    def semantic_from_node(self, node) -> Optional[str]:
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic

        ignored = {
            "binding",
            "component",
            "domain",
            "format",
            "group",
            "index",
            "input",
            "input_primitive",
            "inputprimitive",
            "instance",
            "invocations",
            "location",
            "local_size",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "localsizex",
            "localsizey",
            "localsizez",
            "max_primitives",
            "max_total_threads_per_threadgroup",
            "max_total_threadgroups_per_mesh_grid",
            "max_vertices",
            "maxprimitives",
            "maxtessfactor",
            "maxvertexcount",
            "maxvertices",
            "maybe_unused",
            "numthreads",
            "output",
            "output_control_points",
            "output_topology",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patch_constant_func",
            "patchconstantfunc",
            "readonly",
            "set",
            "space",
            "stage_entry",
            "compute",
            "fragment",
            "geometry",
            "mesh",
            "object",
            "primitive",
            "vertices",
            "workgroup_size",
            "indices",
            "primitives",
            "task",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
            "writeonly",
            # Interpolation, invariance and precision qualifiers decorate the value; they never name it.
            *INTERFACE_QUALIFIER_ATTRIBUTES,
            # So does a conservative depth layout (spirv_conservative_depth).
            *DEPTH_LAYOUT_ATTRIBUTES,
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = self.normalized_metadata_name(attr_name)
            if normalized in ignored or is_inline_spirv_attribute(attr_name):
                continue
            return str(attr_name)
        return None

    def spirv_semantic_output_kind(self, semantic) -> Optional[str]:
        if semantic is None:
            return None

        semantic_name = self.spirv_output_semantic_alias(str(semantic))
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        input_only_sources = {
            "gl_baseinstance",
            "gl_basevertex",
            "gl_barycoordext",
            "gl_barycoordnoperspext",
            "gl_drawid",
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_instanceindex",
            "gl_invocationid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_sampleid",
            "gl_samplemaskin",
            "gl_tesscoord",
            "gl_vertexid",
            "gl_workgroupid",
        }
        if lower_name in input_only_sources or upper_name in {
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
            return "input_only"

        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name == "gl_samplemask" or upper_name == "SV_COVERAGE":
            return "sample_mask"
        if lower_name == "gl_fragstencilrefext" or upper_name == "SV_STENCILREF":
            return "stencil_ref"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if lower_name == "gl_fragdata":
            return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("COLOR"):
            suffix = upper_name[len("COLOR") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def spirv_return_semantic_builtin_info(
        self, semantic, return_type: SpirvId
    ) -> Optional[Tuple[str, str, SpirvId]]:
        kind = self.spirv_semantic_output_kind(semantic)
        if kind == "position":
            return ("gl_Position", "Position", return_type)
        if kind == "depth":
            return ("gl_FragDepth", "FragDepth", return_type)
        if kind == "sample_mask":
            return ("gl_SampleMask", "SampleMask", self.map_crossgl_type("int[1]"))
        if kind == "stencil_ref":
            return ("gl_FragStencilRefEXT", "FragStencilRefEXT", return_type)
        return None

    def spirv_color_semantic_location(self, semantic, node=None) -> Optional[int]:
        if semantic is None:
            return None

        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()

        if lower_name == "gl_fragdata" and node is not None:
            for attr in getattr(node, "attributes", []) or []:
                if str(getattr(attr, "name", "")).lower() != "gl_fragdata":
                    continue
                arguments = getattr(attr, "arguments", None) or getattr(
                    attr, "args", []
                )
                if arguments:
                    return self.literal_int_argument(arguments[0])
            return 0

        for prefix in ("gl_fragcolor",):
            if lower_name.startswith(prefix):
                suffix = lower_name[len(prefix) :]
                if suffix == "":
                    return 0
                if suffix.isdigit():
                    return int(suffix)

        for prefix in ("SV_TARGET", "COLOR"):
            if upper_name.startswith(prefix):
                suffix = upper_name[len(prefix) :]
                if suffix == "":
                    return 0
                if suffix.isdigit():
                    return int(suffix)

        return None

    def is_spirv_float_vector_width(self, type_name, width: int) -> bool:
        vector_info = self.vector_component_type_and_count(
            self.type_name_string(type_name)
        )
        return vector_info == ("float", width)

    def is_spirv_colour_attachment_type(self, type_name) -> bool:
        """Whether ``type_name`` may be a fragment colour output: a 32-bit float, int or uint scalar or vector.

        A colour attachment need not be a vec4 (Vulkan specification, "Fragment Output Interface"); an output
        with fewer components than the attachment format leaves the remaining components undefined, as in HLSL
        ``float main() : SV_Target`` and MSL ``float depth [[color(3)]]``.
        """
        type_text = self.type_name_string(type_name)
        if self.normalize_primitive_name(type_text) in {"float", "int", "uint"}:
            return True
        vector_info = self.vector_component_type_and_count(type_text)
        return vector_info is not None and vector_info[0] in {"float", "int", "uint"}

    def is_spirv_float_scalar_type(self, type_name) -> bool:
        return (
            self.normalize_primitive_name(self.type_name_string(type_name)) == "float"
        )

    def is_spirv_integer_scalar_type(self, type_name) -> bool:
        return self.normalize_primitive_name(self.type_name_string(type_name)) in {
            "int",
            "uint",
        }

    def is_spirv_sample_mask_type(self, type_name) -> bool:
        type_text = self.type_name_string(type_name)
        if self.normalize_primitive_name(type_text) in {"int", "uint"}:
            return True

        array_info = self.split_outer_array_type(type_text)
        if array_info is None:
            return False
        element_type, _ = array_info
        return self.normalize_primitive_name(element_type) in {"int", "uint"}

    def validate_spirv_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.spirv_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return

        if kind == "position":
            if self.is_spirv_float_vector_width(type_name, 4):
                return
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected vec4-compatible type"
            )

        if kind == "color":
            if self.is_spirv_colour_attachment_type(type_name):
                return
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected a float, int or uint scalar or vector"
            )

        if kind == "depth" and not self.is_spirv_float_scalar_type(type_name):
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected float type"
            )

        if kind == "sample_mask" and not self.is_spirv_sample_mask_type(type_name):
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected int-compatible sample-mask type"
            )

        if kind == "stencil_ref" and not self.is_spirv_integer_scalar_type(type_name):
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "expected integer scalar type"
            )

    def validate_spirv_output_semantic_stage(
        self, execution_model: Optional[str], semantic, context
    ):
        kind = self.spirv_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise VulkanRefusal(
                f"Unsupported {semantic} {context} for SPIR-V codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if execution_model is None:
            return

        allowed_models = {
            "position": {"TessellationEvaluation", "Vertex"},
            "color": {"Fragment"},
            "depth": {"Fragment"},
            "sample_mask": {"Fragment"},
            "stencil_ref": {"Fragment"},
        }[kind]
        if execution_model in allowed_models:
            return

        stage_name = {
            "GLCompute": "compute",
            "MeshEXT": "mesh",
            "TaskEXT": "task",
        }.get(execution_model, execution_model.lower())
        allowed = ", ".join(
            {
                "GLCompute": "compute",
                "MeshEXT": "mesh",
                "TaskEXT": "task",
            }.get(model, model.lower())
            for model in sorted(allowed_models)
        )
        raise VulkanRefusal(
            f"Unsupported {semantic} {context} for SPIR-V {stage_name} stage; "
            f"valid stage is {allowed}"
        )

    def validate_spirv_return_semantic(
        self, execution_model: Optional[str], return_type, semantic
    ):
        if semantic is None:
            return
        if self.type_name_string(return_type) == "void":
            raise VulkanRefusal(
                f"Unsupported {semantic} return semantic for SPIR-V codegen; "
                "void return type"
            )
        self.validate_spirv_output_semantic_stage(
            execution_model, semantic, "return semantic"
        )
        self.validate_spirv_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def direct_return_output_name(self, function_name: str, semantic) -> str:
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(semantic)).strip("_")
        return f"{function_name}_return_{normalized or 'semantic'}"

    def register_direct_return_output(
        self, function_node, semantic, return_type: SpirvId
    ) -> Optional[SpirvId]:
        builtin_info = self.spirv_return_semantic_builtin_info(semantic, return_type)
        if builtin_info is not None:
            name, builtin_name, builtin_type = builtin_info
            cache_key = f"__return_semantic_builtin::{name}"
            if cache_key in self.global_variables:
                variable = self.global_variables[cache_key]
                self.mark_fragment_depth_replacing_if_needed(builtin_name, "Output")
                record_depth_layout(self, variable, function_node, "Output")
                decorate_invariant_builtin_output(
                    self, variable, function_node, "Output"
                )
                self.mark_function_interface_variable(variable)
                return variable

            variable = self.register_builtin_variable(
                name, builtin_type, builtin_name, "Output"
            )
            self.global_variables[cache_key] = variable
            record_depth_layout(self, variable, function_node, "Output")
            decorate_invariant_builtin_output(self, variable, function_node, "Output")
            self.mark_function_interface_variable(variable)
            return variable

        output_node = VariableNode(
            self.direct_return_output_name(function_node.name, semantic),
            function_node.return_type,
            attributes=list(getattr(function_node, "attributes", []) or []),
        )
        location = self.global_interface_location(
            output_node,
            "Output",
            preferred_location=self.spirv_color_semantic_location(
                semantic, function_node
            ),
        )
        variable = self.create_variable(return_type, "Output", output_node.name)
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorate_global_interface_variable(output_node, variable)
        self.mark_function_interface_variable(variable)
        return variable

    def initialize_entry_point_parameters(
        self,
        runtime_parameters: List[Tuple[VariableNode, SpirvId, SpirvId]],
        execution_model: Optional[str],
    ):
        self.entry_input_interface_variables = {}
        self.entry_descriptor_slots = plan_entry_descriptor_slots(
            self,
            [param for param, _param_type, _value_type in runtime_parameters],
            lambda param: self.is_storage_resource_parameter(param)
            or self.is_entry_point_descriptor_parameter(param)
            or self.is_bound_uniform_value_parameter(param, execution_model),
        )
        self.entry_input_locations = plan_entry_input_locations(
            self,
            [
                (param, value_type)
                for param, _param_type, value_type in runtime_parameters
                if not (
                    self.is_storage_resource_parameter(param)
                    or self.is_entry_point_descriptor_parameter(param)
                    or self.is_bound_uniform_value_parameter(param, execution_model)
                    or self.is_mesh_payload_entry_parameter(param, execution_model)
                )
            ],
            execution_model,
        )
        for param, _param_type, param_value_type in runtime_parameters:
            param_name = getattr(param, "name", None) or "param"
            if self.is_mesh_payload_entry_parameter(param, execution_model):
                self.local_variables[param_name] = (
                    self.process_mesh_payload_entry_parameter(param, param_value_type)
                )
                continue
            if self.is_storage_resource_parameter(param):
                variable = self.process_entry_point_storage_buffer_parameter(param)
                self.local_variables[param_name] = variable
                continue
            if self.is_entry_point_descriptor_parameter(param):
                variable = self.process_entry_point_descriptor_parameter(param)
                self.local_variables[param_name] = variable
                continue

            if self.is_bound_uniform_value_parameter(param, execution_model):
                initial_value = self.entry_point_uniform_parameter_value(
                    param, param_value_type
                )
            else:
                initial_value = self.entry_point_parameter_value(
                    param, param_value_type, execution_model
                )
            local_variable = self.create_variable(
                param_value_type, "Function", param_name
            )
            self.local_variables[param_name] = local_variable
            if initial_value is not None:
                self.store_to_variable(local_variable, initial_value)
            self.register_declared_resource_metadata(
                param, local_variable, param_value_type
            )
        self.entry_descriptor_slots = {}
        self.entry_input_locations = {}

    def resource_parameter_qualifier_names(self, param) -> set:
        return {
            str(qualifier).lower()
            for qualifier in getattr(param, "resource_qualifiers", []) or []
        }

    def is_bound_uniform_value_parameter(
        self, param, execution_model: Optional[str]
    ) -> bool:
        qualifiers = self.parameter_qualifier_names(param)
        qualifiers.update(self.resource_parameter_qualifier_names(param))
        if not qualifiers & {"uniform", "constant"}:
            return False

        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        type_name = self.type_name_from_value(param_type)
        type_name = self.normalize_reference_type_name(type_name)
        if type_name is None:
            return False
        return not (
            self.is_resource_type_name(type_name)
            or self.is_structured_buffer_declared_type_name(type_name)
            or self.pointer_pointee_type_name_from_string(type_name) is not None
        )

    def is_storage_resource_parameter(self, param) -> bool:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if param_type is None:
            return False
        type_name = self.type_name_from_value(param_type)
        if self.is_structured_buffer_declared_type_name(type_name):
            return True

        qualifiers = self.resource_parameter_qualifier_names(param)
        parameter_qualifiers = self.parameter_qualifier_names(param)
        if self.pointer_pointee_type_name_from_string(type_name) is not None:
            return bool(
                qualifiers & {"storage", "buffer"}
                or parameter_qualifiers & {"device", "constant", "storage", "buffer"}
            )

        if "storage" not in qualifiers:
            return False
        param_type_id = self.map_crossgl_type(param_type)
        return self.array_type_info_from_type(param_type_id) is not None

    def is_entry_point_descriptor_parameter(self, param) -> bool:
        """Whether an entry parameter is an opaque descriptor (MSL ``texture2d t [[texture(0)]]``, ``sampler s``)."""
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        type_name = self.normalize_reference_type_name(
            self.type_name_from_value(param_type)
        )
        if not type_name:
            return False
        base_type = self.array_base_type_name(type_name)
        return self.is_resource_type_name(
            base_type
        ) or self.is_acceleration_structure_type_name(base_type)

    def process_entry_point_descriptor_parameter(self, param) -> SpirvId:
        """Declare an opaque entry parameter as the ``UniformConstant`` descriptor it is bound to.

        The host binds such a parameter like a module-scope resource; a stage ``Input`` variable cannot hold an
        image, sampler or acceleration structure (``VUID-StandaloneSpirv-OpTypeImage-06924`` forbids storing one into
        the Function-scope copy). MSL keeps textures, samplers and buffers in separate binding namespaces, so a slot
        an earlier resource already holds moves to the next free binding of the set.
        """
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        node = VariableNode(
            getattr(param, "name", "param"),
            param_type,
            attributes=list(getattr(param, "attributes", []) or []),
            qualifiers=list(getattr(param, "qualifiers", []) or []),
        )
        var_type = self.map_resource_type_with_format(param_type, node)
        var_id = self.create_variable(var_type, "UniformConstant", node.name)
        descriptor_set, binding = self.resource_descriptor_slot(
            node, allow_binding_reassignment=True
        )
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
        self.register_declared_resource_metadata(node, var_id, var_type)
        return var_id

    def process_entry_point_storage_buffer_parameter(self, param) -> SpirvId:
        variable = VariableNode(
            getattr(param, "name", "param"),
            getattr(param, "param_type", getattr(param, "vtype", None)),
            attributes=list(getattr(param, "attributes", []) or []),
            qualifiers=list(getattr(param, "qualifiers", []) or []),
        )
        variable.resource_qualifiers = list(
            getattr(param, "resource_qualifiers", []) or []
        )
        type_name = self.type_name_from_value(variable.var_type)
        if self.is_structured_buffer_declared_type_name(type_name):
            return self.process_structured_buffer_declaration(variable, type_name)
        return self.process_glsl_buffer_block_declaration(
            variable, type_name, allow_binding_reassignment=True
        )

    def entry_point_uniform_parameter_value(
        self, param, param_value_type: SpirvId
    ) -> SpirvId:
        param_name = getattr(param, "name", None) or "param"
        member_type = self.storage_layout_type(param_value_type, "std140")
        block_name = f"{self.current_function_name}_{param_name}UniformBlock"
        block_type = self.register_struct_type(block_name, [(member_type, param_name)])
        self.decorate_cbuffer_type(block_type, [(member_type, param_name)])

        var_id = self.create_variable(block_type, "Uniform", f"{param_name}Uniform")
        descriptor_set, binding = self.resource_descriptor_slot(
            param, allow_binding_reassignment=True
        )
        self.decorations.append(
            f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
        )
        self.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
        self.uniform_buffers.append(var_id)

        int_type = self.register_primitive_type("int")
        index = self.register_constant(0, int_type)
        ptr_type = self.register_pointer_type(member_type, "Uniform")
        member_pointer = self.access_chain(var_id, [index], ptr_type)
        self.variable_value_types[member_pointer.id] = member_type
        loaded = self.get_variable_value(member_pointer)
        return self.convert_value_to_type(loaded, param_value_type)

    def entry_point_parameter_value(
        self, param, param_value_type: SpirvId, execution_model: Optional[str]
    ) -> Optional[SpirvId]:
        members = self.current_struct_members.get(param_value_type.type.base_type)
        if members:
            components = []
            metadata = self.struct_member_metadata.get(
                param_value_type.type.base_type, {}
            )
            distances = entry_point_distance_layout(self, members, metadata)
            for member_type, member_name in members:
                member_info = metadata.get(member_name, {})
                member_node = member_info.get("node")
                location_index = self.entry_input_locations.get(
                    (id(param), member_name)
                )
                distance = distances.slot(member_name)
                if distance is not None:
                    array = entry_point_distance_variable(
                        self,
                        distance.builtin,
                        distances.size(distance.builtin),
                        "Input",
                    )
                    components.append(
                        load_entry_point_distance(self, array, distance, member_type)
                    )
                    continue
                if unread_unavailable_builtin_input(
                    self,
                    getattr(self, "spirv_program_ast", None),
                    execution_model,
                    member_node,
                    member_type,
                    member_name,
                ):
                    components.append(self.register_undefined_value(member_type))
                    continue
                interface_name = self.entry_point_interface_name(
                    execution_model,
                    "input",
                    getattr(param, "name", "param"),
                    member_name,
                )
                fields = flatten_struct_varying(
                    self,
                    execution_model,
                    "Input",
                    interface_name,
                    member_type,
                    member_node,
                    planned_location=location_index,
                )
                if fields is not None:
                    components.append(
                        load_flattened_struct_varying(self, member_type, fields)
                    )
                    continue
                variable = self.register_entry_point_interface_variable(
                    execution_model,
                    "Input",
                    interface_name,
                    member_type,
                    member_node,
                    member_name=member_name,
                    member_index=location_index,
                )
                record_entry_input_variable(
                    self, getattr(param, "name", None), member_name, variable
                )
                components.append(
                    self.load_entry_point_interface_value(variable, member_type)
                )
            return self.composite_construct(param_value_type, components)

        variable = self.register_entry_point_interface_variable(
            execution_model,
            "Input",
            self.entry_point_interface_name(
                execution_model, "input", getattr(param, "name", "param")
            ),
            param_value_type,
            param,
            member_index=self.entry_input_locations.get((id(param), None)),
        )
        record_entry_input_variable(self, getattr(param, "name", None), None, variable)
        return self.load_entry_point_interface_value(variable, param_value_type)

    def register_entry_point_return_outputs(
        self,
        function_node,
        return_type: SpirvId,
        execution_model: Optional[str],
    ):
        members = self.current_struct_members.get(return_type.type.base_type)
        if members:
            outputs = []
            distance_outputs = []
            struct_outputs = []
            metadata = self.struct_member_metadata.get(return_type.type.base_type, {})
            distances = entry_point_distance_layout(self, members, metadata)
            planned = plan_struct_member_locations(
                self, members, metadata, distances, execution_model, "Output"
            )
            for member_index, (member_type, member_name) in enumerate(members):
                member_info = metadata.get(member_name, {})
                member_node = member_info.get("node")
                location_index = planned.get(member_name)
                distance = distances.slot(member_name)
                if distance is not None:
                    array = entry_point_distance_variable(
                        self,
                        distance.builtin,
                        distances.size(distance.builtin),
                        "Output",
                    )
                    distance_outputs.append(
                        (member_index, array, member_type, distance)
                    )
                    continue
                interface_name = self.entry_point_interface_name(
                    execution_model,
                    "output",
                    getattr(function_node, "name", "main"),
                    member_name,
                )
                fields = flatten_struct_varying(
                    self,
                    execution_model,
                    "Output",
                    interface_name,
                    member_type,
                    member_node,
                    planned_location=location_index,
                )
                if fields is not None:
                    struct_outputs.append((member_index, member_type, fields))
                    continue
                variable = self.register_entry_point_interface_variable(
                    execution_model,
                    "Output",
                    interface_name,
                    member_type,
                    member_node,
                    member_name=member_name,
                    member_index=location_index,
                )
                outputs.append((member_index, variable, member_type))
            return {
                "kind": "struct",
                "type": return_type,
                "outputs": outputs,
                "distances": distance_outputs,
                "struct_fields": struct_outputs,
            }

        variable = self.register_entry_point_interface_variable(
            execution_model,
            "Output",
            self.entry_point_interface_name(
                execution_model, "output", getattr(function_node, "name", "main")
            ),
            return_type,
            function_node,
        )
        return {"kind": "value", "type": return_type, "variable": variable}

    def entry_point_interface_name(
        self,
        execution_model: Optional[str],
        direction: str,
        base_name: str,
        member_name: Optional[str] = None,
    ) -> str:
        stage = (execution_model or "stage").lower()
        parts = ["_CrossGL", stage, direction, str(base_name)]
        if member_name is not None:
            parts.append(str(member_name))
        return "_".join(part.strip("_") for part in parts if part)

    def register_entry_point_interface_variable(
        self,
        execution_model: Optional[str],
        storage_class: str,
        name: str,
        type_id: SpirvId,
        node,
        member_name: Optional[str] = None,
        member_index: Optional[int] = None,
    ) -> SpirvId:
        semantic = self.entry_point_interface_semantic(
            execution_model, storage_class, node, type_id, member_name
        )
        builtin = self.entry_point_builtin_variable(semantic, storage_class)
        record_depth_layout(self, builtin, node, storage_class)
        if builtin is not None:
            decorate_invariant_builtin_output(self, builtin, node, storage_class)
            return builtin
        unmapped = unmapped_builtin_reason(semantic, storage_class, execution_model)
        if unmapped is not None:
            self.refuse_lowering(unmapped)

        interface_node = node
        if interface_node is None:
            interface_node = VariableNode(name, type_id.type.base_type)
        interface_type = self.lowered_user_defined_interface_type(type_id)
        self.validate_user_defined_interface_type(
            interface_type, storage_class, name, interface_node
        )
        preferred_location = self.entry_point_interface_location(
            semantic, storage_class, node, member_index, execution_model
        )
        # A tessellation evaluation stage reads its control points through its patch parameter, so each other user
        # input is a per-patch value, which Vulkan reads only through a Patch variable (Vulkan 1.3 section 15.1.4).
        patch = storage_class == "Input" and execution_model == "TessellationEvaluation"
        location = self.global_interface_location(
            interface_node,
            storage_class,
            preferred_location=preferred_location,
            patch=patch,
        )
        variable = self.create_variable(interface_type, storage_class, name)
        if interface_type.id != type_id.id:
            self.interface_original_value_types[variable.id] = type_id
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        if patch:
            self.decorations.append(f"OpDecorate %{variable.id} Patch")
        if (
            storage_class == "Input"
            and execution_model == "Fragment"
            and self.interface_type_requires_flat(interface_type)
            and not self.is_per_vertex_interface(interface_node)
        ):
            self.decorations.append(f"OpDecorate %{variable.id} Flat")
        self.decorate_global_interface_variable(interface_node, variable)
        if storage_class == "Input":
            self.inputs.append(variable)
        else:
            self.outputs.append(variable)
        self.mark_function_interface_variable(variable)
        return variable

    def entry_point_builtin_variable(
        self, semantic: Optional[str], storage_class: str
    ) -> Optional[SpirvId]:
        if semantic is None:
            return None
        if storage_class == "Input":
            return self.ensure_builtin_variable(semantic)
        if storage_class == "Output":
            return self.ensure_stage_builtin(semantic)
        return None

    def entry_point_interface_semantic(
        self,
        execution_model: Optional[str],
        storage_class: str,
        node,
        type_id: SpirvId,
        member_name: Optional[str],
    ) -> Optional[str]:
        semantic = self.semantic_from_node(node)
        if semantic is not None:
            return self.spirv_interface_semantic_alias(
                str(semantic), execution_model, storage_class
            )

        normalized_member = str(member_name or getattr(node, "name", "")).lower()
        if (
            storage_class == "Output"
            and execution_model in {"Vertex", "TessellationEvaluation"}
            and normalized_member in {"position", "pos"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_Position"
        if (
            storage_class == "Output"
            and execution_model == "Fragment"
            and normalized_member in {"color", "colour", "fragcolor"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_FragColor"
        if (
            storage_class == "Input"
            and execution_model == "Fragment"
            and normalized_member in {"fragcoord", "frag_coord"}
            and self.vector_component_type_and_count(type_id.type.base_type)
            == ("float", 4)
        ):
            return "gl_FragCoord"
        return None
