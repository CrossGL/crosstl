"""GLSL buffer blocks in the Metal generator: semantics, precision and address-space attributes, resource type mapping with formats, buffer block layouts and lowering failures, buffer block access, scalar, matrix and aggregate loads and stores, unsupported buffer block placeholders and diagnostics."""

import re
from hashlib import sha1

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    VariableNode,
)
from ..refusals import MetalRefusal
from ..semantics.hlsl_ray_payloads import PAYLOAD_ACCESS_ATTRIBUTES
from .array_utils import (
    parse_array_type,
    split_array_type_suffix,
)
from .buffer_block_diagnostics import (
    BufferBlockUnsupportedError,
    record_unsupported_layout_failures,
)
from .buffer_length import (
    COUNTED_BUFFERS,
    BufferLengthUnsupportedError,
)
from .buffer_member_layout import MEMBER_LAYOUT_QUALIFIERS
from .generic_struct_utils import format_struct_constructor_expression
from .glsl_buffer_array_values import glsl_buffer_array_value_type
from .glsl_buffer_block_arrays import (
    glsl_buffer_array_elements,
    glsl_buffer_block_indexed_access,
    glsl_buffer_matrix_column_access,
    glsl_buffer_whole_array_reason,
)
from .glsl_buffer_layout import (
    byte_offset_add,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_block_node_type,
    glsl_buffer_compound_binary_operator,
    matrix_column_offsets,
    vector_component_offsets,
)
from .hlsl_buffer_layout import BUFFER_LAYOUT_ATTRIBUTE
from .image_access_contracts import (
    explicit_image_access,
    explicit_image_format,
    is_image_format_attribute,
    is_resource_access_attribute,
)
from .interface_qualifiers import is_interface_qualifier_attribute
from .metal_array_copies import generate_metal_array_initializer
from .metal_buffer_coherence import memory_helper_token
from .metal_buffer_component_stores import (
    metal_buffer_component_store,
    metal_buffer_component_type,
    metal_glsl_buffer_component_refusal,
)
from .metal_exact_lookups import METAL_MATRIX_ORDER_ATTRIBUTES
from .metal_raster_order_groups import RASTERIZER_ORDERED


class MetalGLSLBufferBlockMixin:
    """GLSL buffer block members of ``MetalCodeGen``."""

    def explicit_resource_set_index(self, node):
        if not hasattr(node, "attributes"):
            return None

        selected_set = None
        selected_description = None
        node_name = self.resource_node_name(node, "<anonymous>")
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in {"set", "space"}:
                source = (
                    self.attribute_value_to_string(arguments[0]) if arguments else None
                )
                set_index = (
                    self.binding_index_value(arguments[0], ("set", "space"))
                    if len(arguments) == 1
                    else None
                )
                description = (
                    f"{attr_name} {source if source is not None else '<missing>'}"
                )
            elif attr_name == "register" and len(arguments) > 1:
                source = self.attribute_value_to_string(arguments[1])
                set_index = self.binding_index_value(arguments[1], ("space", "set"))
                description = (
                    f"register space {source if source is not None else '<missing>'}"
                )
            else:
                continue

            if set_index is None:
                raise MetalRefusal(
                    "Invalid Metal resource binding metadata for "
                    f"'{node_name}': {description} must resolve to a concrete "
                    "integer set"
                )
            if selected_set is None:
                selected_set = set_index
                selected_description = description
                continue
            if selected_set != set_index:
                raise MetalRefusal(
                    "Conflicting Metal resource binding metadata for "
                    f"'{node_name}': {selected_description} differs from "
                    f"{description}"
                )
        return selected_set

    def semantic_from_node(self, node):
        if hasattr(node, "semantic"):
            return node.semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_metal_texture_element_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.is_metal_address_space_attribute(attr)
                or self.is_metal_declaration_qualifier_attribute(attr)
                or self.is_metal_struct_member_abi_attribute(attr)
                or self.normalized_metal_abi_attribute_name(attr) == "function_constant"
                or self.metal_interpolation_attribute_name(attr) is not None
                or self.is_precision_qualifier_attribute(attr)
                or is_interface_qualifier_attribute(attr)
                or str(getattr(attr, "name", "")).lower()
                in (BUFFER_LAYOUT_ATTRIBUTE, "packed")
                or str(getattr(attr, "name", "")).lower() in MEMBER_LAYOUT_QUALIFIERS
                # Matrix order (GLSL 4.60 section 4.4.5, HLSL row_major/column_major) is layout, not a semantic;
                # Metal lays such a matrix out transposed (semantics/row_major_struct_members.py).
                or str(getattr(attr, "name", "")).lower()
                in METAL_MATRIX_ORDER_ATTRIBUTES
                # DXR payload access qualifiers; a Metal payload is a ray_data reference every stage may access.
                or str(getattr(attr, "name", "")).lower() in PAYLOAD_ACCESS_ATTRIBUTES
                # Access ordering, written as [[raster_order_group(0)]] (metal_raster_order_groups.py).
                or str(getattr(attr, "name", "")).lower() == RASTERIZER_ORDERED
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def is_precision_qualifier_attribute(self, attr):
        return str(getattr(attr, "name", attr)).lower() in {
            "lowp",
            "mediump",
            "highp",
            "precise",
        }

    @staticmethod
    def metal_has_precise_qualifier(node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "precise" in qualifiers:
            return True
        return any(
            str(getattr(attribute, "name", "")).lower() == "precise"
            for attribute in getattr(node, "attributes", []) or []
        )

    def metal_function_requires_no_contraction(self, func):
        if self.metal_has_precise_qualifier(func):
            return True
        return any(
            self.metal_has_precise_qualifier(node)
            for node in self.iter_ast_nodes(getattr(func, "body", None))
        )

    def is_metal_address_space_attribute(self, attr):
        name = str(getattr(attr, "name", "")).lower()
        return name in {
            "constant",
            "device",
            "global",
            "groupshared",
            "local",
            "object_data",
            "objectdata",
            "private",
            "ray_data",
            "raydata",
            "shared",
            "storage",
            "thread",
            "threadgroup",
            "workgroup",
        }

    def metal_stage_control_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]

        valid_names = {
            "cw",
            "ccw",
            "domain",
            "equal_spacing",
            "fractional_even_spacing",
            "fractional_odd_spacing",
            "invocations",
            "isolines",
            "layout",
            "line_strip",
            "lines",
            "lines_adjacency",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "max_primitives",
            "maxtessfactor",
            "max_total_threads_per_threadgroup",
            "max_total_threadgroups_per_mesh_grid",
            "max_vertices",
            "maxvertexcount",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
            "point_mode",
            "points",
            "quads",
            "threadgroup_size",
            "threads_per_threadgroup",
            "triangles",
            "triangles_adjacency",
            "triangle_strip",
            "vertices",
        }
        if normalized in valid_names:
            return normalized
        return None

    def function_return_semantic(self, func):
        semantic = getattr(func, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(func, "attributes"):
            return None
        for attr in func.attributes:
            if self.metal_stage_control_attribute_name(attr):
                continue
            if (
                is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_metal_texture_element_attribute(attr)
                or self.is_glsl_buffer_block_attribute(attr)
                or self.is_precision_qualifier_attribute(attr)
                or is_interface_qualifier_attribute(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def map_non_stage_function_semantic(self, semantic):
        if self.is_legacy_glsl_fragment_output_semantic(semantic):
            return ""
        return self.map_semantic(semantic)

    def is_legacy_glsl_fragment_output_semantic(self, semantic):
        if semantic is None:
            return False
        semantic_text = str(semantic)
        return (
            semantic_text == "gl_FragData"
            or semantic_text.startswith("gl_FragData[")
            or semantic_text == "gl_FragColor"
            or re.fullmatch(r"gl_FragColor\d+", semantic_text) is not None
        )

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if isinstance(vtype, (PointerType, ReferenceType)):
            return self.map_type(vtype)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = (
            explicit_image_format(node, self.attribute_value_to_string)
            if node is not None
            else None
        )
        component_type = self.metal_texture_element_attribute_type(node) or (
            self.scalar_image_format_components().get(explicit_format)
            or self.vector_image_format_components().get(explicit_format)
        )
        texture_types = {
            "image1D": "texture1d",
            "iimage1D": "texture1d",
            "uimage1D": "texture1d",
            "image2D": "texture2d",
            "iimage2D": "texture2d",
            "uimage2D": "texture2d",
            "image3D": "texture3d",
            "iimage3D": "texture3d",
            "uimage3D": "texture3d",
            "image1DArray": "texture1d_array",
            "iimage1DArray": "texture1d_array",
            "uimage1DArray": "texture1d_array",
            "image2DArray": "texture2d_array",
            "iimage2DArray": "texture2d_array",
            "uimage2DArray": "texture2d_array",
            "image2DMS": "texture2d_ms",
            "iimage2DMS": "texture2d_ms",
            "uimage2DMS": "texture2d_ms",
            "image2DMSArray": "texture2d_ms_array",
            "iimage2DMSArray": "texture2d_ms_array",
            "uimage2DMSArray": "texture2d_ms_array",
            "imageCube": "texture2d_array",
            "imageBuffer": "texture_buffer",
            "iimageBuffer": "texture_buffer",
            "uimageBuffer": "texture_buffer",
        }
        texture_type = texture_types.get(base_type)
        if texture_type:
            if component_type is None:
                component_type = self.default_image_component_type(base_type)
            access = (
                explicit_image_access(node, self.attribute_value_to_string)
                or "read_write"
            )
            if texture_type in {"texture2d_ms", "texture2d_ms_array"}:
                access = "read"
            return f"{texture_type}<{component_type}, access::{access}>"
        return self.map_type(vtype)

    def is_metal_texture_element_attribute(self, attr):
        return str(getattr(attr, "name", "")).lower() in {
            "metal_texture_element_half",
        }

    def metal_texture_element_attribute_type(self, node):
        if node is None:
            return None
        for attr in getattr(node, "attributes", []) or []:
            if self.is_metal_texture_element_attribute(attr):
                return "half"
        return None

    def default_image_component_type(self, vtype):
        base_type = self.resource_base_type(vtype)
        if base_type.startswith("iimage"):
            return "int"
        if base_type.startswith("uimage"):
            return "uint"
        return "float"

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "restrict",
            "volatile",
        }

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if isinstance(vtype, PointerType):
            return self.resource_base_type(vtype.pointee_type)
        if isinstance(vtype, ReferenceType):
            return self.resource_base_type(vtype.referenced_type)
        if self.is_array_type_node(vtype):
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        arguments = getattr(attr, "arguments", []) if attr is not None else []
        if arguments:
            layout = self.attribute_value_to_string(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "glsl_buffer_block")

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        if self.glsl_buffer_block_attribute(node) is None:
            return False
        type_name = self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, global_vars):
        names = set()
        for node in global_vars:
            node_type = glsl_buffer_block_node_type(node)
            if not self.is_glsl_buffer_block_variable(node, node_type):
                continue
            type_name = self.resource_base_type(node_type)
            names.add(str(type_name))
        return names

    def record_metal_glsl_buffer_block_layout_failures(
        self, nodes, lowered_blocks, var_failures, struct_failures
    ):
        record_unsupported_layout_failures(
            nodes,
            lowered_blocks,
            var_failures,
            struct_failures,
            is_block=self.is_glsl_buffer_block_variable,
            node_type=glsl_buffer_block_node_type,
            layout_of=self.glsl_buffer_block_layout,
            base_type=self.resource_base_type,
            lowering="Metal device pointer offset lowering",
        )

    def collect_lowered_glsl_buffer_block_parameters(self, parameters):
        blocks, var_failures, struct_failures = collect_lowered_glsl_buffer_blocks(
            parameters,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: self.literal_int_value(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="metal_type",
            unsupported_type_message=(
                "type is not supported by Metal pointer/offset lowering"
            ),
            component_refusal=metal_glsl_buffer_component_refusal,
        )
        self.record_metal_glsl_buffer_block_layout_failures(
            parameters, blocks, var_failures, struct_failures
        )
        return blocks, var_failures, struct_failures

    def collect_unsupported_glsl_buffer_block_parameter_names(self, parameters):
        names = set()
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                param_name
                and param_name not in self.current_glsl_buffer_block_parameters
                and self.is_glsl_buffer_block_variable(param, param_type)
            ):
                names.add(param_name)
        return names

    def collect_unsupported_glsl_buffer_block_functions(self, functions):
        unsupported = {}
        omitted_struct_names = set(self.glsl_buffer_block_struct_names)
        if not omitted_struct_names:
            return unsupported

        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            return_type = self.type_name_string(getattr(func, "return_type", "void"))
            return_base = str(self.resource_base_type(return_type))
            if return_base in omitted_struct_names:
                unsupported[func_name] = {
                    "return_type": return_type,
                    "reason": (
                        f"return type references omitted GLSL buffer block struct {return_base}"
                    ),
                }
                continue

            param_list = getattr(func, "parameters", getattr(func, "params", []))
            lowered_params, _, _ = self.collect_lowered_glsl_buffer_block_parameters(
                param_list
            )
            for param in param_list:
                param_name = getattr(param, "name", None)
                param_type = getattr(
                    param,
                    "param_type",
                    getattr(param, "var_type", getattr(param, "vtype", None)),
                )
                param_base = str(self.resource_base_type(param_type))
                if (
                    param_base in omitted_struct_names
                    and param_name not in lowered_params
                ):
                    unsupported[func_name] = {
                        "return_type": return_type,
                        "reason": (
                            f"parameter {param_name} references omitted GLSL "
                            f"buffer block struct {param_base}"
                        ),
                    }
                    break
        return unsupported

    def glsl_buffer_block_lowering_failure_detail(self, type_name, var_name=None):
        if var_name:
            reason = self.current_glsl_buffer_block_parameter_failures.get(var_name)
            if reason:
                return reason
            reason = self.glsl_buffer_block_lowering_failures.get(var_name)
            if reason:
                return reason
        type_name = str(self.resource_base_type(type_name))
        return self.current_glsl_buffer_block_parameter_struct_failures.get(
            type_name
        ) or self.glsl_buffer_block_struct_lowering_failures.get(type_name)

    def glsl_buffer_block_parameter_diagnostics(self, target, parameters, indent=0):
        code = ""
        indent_str = "  " * indent
        for param in parameters or []:
            param_name = getattr(param, "name", None)
            param_type = glsl_buffer_block_node_type(param)
            if (
                not param_name
                or param_name in self.current_glsl_buffer_block_parameters
                or param_name
                not in self.current_unsupported_glsl_buffer_block_parameters
                or not self.is_glsl_buffer_block_variable(param, param_type)
            ):
                continue
            code += indent_str + self.glsl_buffer_block_diagnostic(
                target, param_type, param_name, param, declaration_kind="parameter"
            )
        return code

    def glsl_buffer_block_length_call(self, func_expr, args):
        """``block.member.length()``: a sized array's size; a runtime array is refused, MSL buffers have no size."""
        if not (
            isinstance(func_expr, MemberAccessNode)
            and getattr(func_expr, "member", None) == "length"
            and not args
        ):
            return None
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        if isinstance(object_expr, (IdentifierNode, VariableNode)):
            name = getattr(object_expr, "name", None)
            # A Metal buffer argument is a pointer and carries no size, so
            # length() is refused whether the source spells it as a pointer
            # (buffer float*) or as a typed buffer resource (StructuredBuffer,
            # RWStructuredBuffer, Buffer, ...), which Metal also lowers to a
            # device pointer. Without the resource spelling, the call was
            # copied unchanged onto the pointer, which the Metal compiler
            # rejects.
            local_type = str(self.local_variable_types.get(name, ""))
            if self.current_address_space_variables.get(name) in {
                "device",
                "constant",
            } and (
                local_type.endswith("*")
                or local_type.split("<", 1)[0] in COUNTED_BUFFERS
            ):
                raise BufferLengthUnsupportedError(
                    "metal",
                    f"length() of buffer view '{name}' needs the bound buffer's "
                    "size, which a Metal buffer argument does not carry",
                )
        # AppendStructuredBuffer and ConsumeStructuredBuffer are device
        # pointers too, and so is an element of a structured buffer array:
        # the name of bufs[1] is "bufs", recorded with the element's
        # StructuredBuffer<T> type.
        object_name = self.expression_name(object_expr)
        if self.is_structured_buffer_type(
            self.local_variable_types.get(object_name, "")
        ):
            raise BufferLengthUnsupportedError(
                "metal",
                f"length() of buffer view '{object_name}' needs the bound buffer's "
                "size, which a Metal buffer argument does not carry",
            )
        access = self.glsl_buffer_block_member_access(object_expr)
        if access is None or not access.get("is_array"):
            return None
        if access.get("array_size") is not None:
            return f"int({self.generate_expression(access['array_size'])})"
        raise BufferLengthUnsupportedError(
            "metal",
            f"length() of runtime array '{access.get('member')}' needs the bound "
            "buffer's size, which a Metal buffer argument does not carry",
        )

    def glsl_buffer_block_member_access(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        var_name = self.expression_name(object_expr)
        member_name = getattr(expr, "member", None)
        if var_name:
            block = self.current_glsl_buffer_block_parameters.get(
                var_name
            ) or self.lowered_glsl_buffer_blocks.get(var_name)
            if block is not None:
                buffer_expr = self.generate_expression(object_expr)
                member = block["members"].get(member_name)
                if member is None:
                    return None
                return {
                    "buffer": buffer_expr,
                    "member": member_name,
                    "readonly": block["readonly"],
                    "memory": block.get("memory", ""),
                    **member,
                }

        parent = self.glsl_buffer_block_array_access(
            object_expr
        ) or self.glsl_buffer_block_member_access(object_expr)
        if parent is None or not parent.get("members") or parent.get("subarray"):
            return None
        member = parent["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": parent["buffer"],
            "member": f"{parent['member']}.{member_name}",
            "readonly": parent["readonly"],
            "memory": parent.get("memory", ""),
            **member,
            "offset": byte_offset_add(parent["offset"], member["offset"]),
        }

    def glsl_buffer_block_array_access(self, expr):
        return glsl_buffer_block_indexed_access(self, expr)

    def metal_scalar_load(self, component_type, buffer_name, offset, memory=""):
        # ``memory`` is the block's coherent/volatile prefix: a cast that drops it
        # turns coherent accesses into plain ones (see metal_buffer_coherence).
        if component_type == "bool":
            return (
                f"((*reinterpret_cast<{memory}const device uint*>"
                f"({buffer_name} + {offset})) != 0u)"
            )
        component_type = metal_buffer_component_type(component_type)
        return (
            f"(*reinterpret_cast<{memory}const device {component_type}*>"
            f"({buffer_name} + {offset}))"
        )

    def metal_scalar_store(self, component_type, buffer_name, offset, value, memory=""):
        if component_type == "bool":
            return (
                f"(*reinterpret_cast<{memory}device uint*>"
                f"({buffer_name} + {offset})) = (({value}) ? 1u : 0u)"
            )
        component_type = metal_buffer_component_type(component_type)
        return (
            f"(*reinterpret_cast<{memory}device {component_type}*>"
            f"({buffer_name} + {offset})) = {value}"
        )

    def metal_buffer_load(self, buffer_name, offset, access):
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                column_access = {
                    "components": access["matrix_rows"],
                    "component_type": "float",
                    "metal_type": f"float{access['matrix_rows']}",
                    "memory": access.get("memory", ""),
                }
                columns.append(
                    self.metal_buffer_load(buffer_name, column_offset, column_access)
                )
            return f"{access['metal_type']}({', '.join(columns)})"

        memory = access.get("memory", "")
        if access["components"] == 1:
            return self.metal_scalar_load(
                access["component_type"], buffer_name, offset, memory
            )

        values = []
        for _, component_offset in vector_component_offsets(
            offset, access["components"], access.get("component_size", 4)
        ):
            values.append(
                self.metal_scalar_load(
                    access["component_type"], buffer_name, component_offset, memory
                )
            )
        return f"{access['metal_type']}({', '.join(values)})"

    def next_metal_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.metal_temp_variable_index}"
        self.metal_temp_variable_index += 1
        return name

    def metal_buffer_store(self, buffer_name, offset, value, access):
        if access.get("matrix_columns"):
            return self.metal_matrix_store(buffer_name, offset, value, access)

        if access["components"] == 1:
            if access.get("layout_type") != access.get("type"):
                value = (
                    f"{metal_buffer_component_type(access['component_type'])}({value})"
                )
            return self.metal_scalar_store(
                access["component_type"],
                buffer_name,
                offset,
                value,
                access.get("memory", ""),
            )

        temp_name = self.next_metal_temp_variable("buffer_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for component, component_offset in vector_component_offsets(
            offset, access["components"], access.get("component_size", 4)
        ):
            field = "xyzw"[component]
            lines.append(
                self.metal_scalar_store(
                    access["component_type"],
                    buffer_name,
                    component_offset,
                    f"{temp_name}.{field}",
                    access.get("memory", ""),
                )
            )
        return "\n".join(lines)

    def metal_matrix_store(self, buffer_name, offset, value, access):
        temp_name = self.next_metal_temp_variable("matrix_store")
        lines = [f"{access['metal_type']} {temp_name} = {value}"]
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            for row, element_offset in vector_component_offsets(
                column_offset, access["matrix_rows"]
            ):
                field = "xyzw"[row]
                lines.append(
                    self.metal_scalar_store(
                        "float",
                        buffer_name,
                        element_offset,
                        f"{temp_name}[{column}].{field}",
                        access.get("memory", ""),
                    )
                )
        return "\n".join(lines)

    def metal_aggregate_has_array_member(self, access):
        for member in access["members"].values():
            if member.get("is_array"):
                return True
            if member.get("members") and self.metal_aggregate_has_array_member(member):
                return True
        return False

    def metal_aggregate_helper_suffix(self, access):
        return "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in access["metal_type"]
        )

    def metal_aggregate_layout_signature(self, access):
        parts = []

        def visit(member_name, member):
            fields = [
                member_name,
                str(member.get("type")),
                str(member.get("layout_type")),
                str(member.get("offset")),
                str(member.get("size")),
                str(member.get("align")),
                str(member.get("components")),
                str(member.get("component_type")),
                str(member.get("matrix_columns")),
                str(member.get("matrix_rows")),
                str(member.get("column_stride")),
                str(member.get("is_array")),
                str(member.get("array_count")),
                str(member.get("stride")),
                str(member.get("runtime_array")),
            ]
            if member.get("inner_arrays"):
                fields.append(str(member["inner_arrays"]))
            parts.append(":".join(fields))
            for child_name, child in (member.get("members") or {}).items():
                visit(f"{member_name}.{child_name}", child)

        for field_name, member in access["members"].items():
            visit(field_name, member)
        return sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    def metal_aggregate_load_helper_name(self, access):
        helper_name = (
            f"__crossgl_load_glsl_buffer_"
            f"{memory_helper_token(access.get('memory', ''))}"
            f"{self.metal_aggregate_helper_suffix(access)}_"
            f"{self.metal_aggregate_layout_signature(access)}"
        )
        self.required_glsl_buffer_aggregate_load_helpers[helper_name] = access
        return helper_name

    def metal_aggregate_load_assignments(
        self, target_name, buffer_name, offset, access, indent=1
    ):
        indent_str = "    " * indent
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_target = f"{target_name}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
                "memory": access.get("memory", ""),
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for subscripts, element_offset in glsl_buffer_array_elements(
                    member, member_offset
                ):
                    element_target = f"{member_target}{subscripts}"
                    if member.get("members"):
                        nested_lines = self.metal_aggregate_load_assignments(
                            element_target,
                            buffer_name,
                            element_offset,
                            field_access,
                            indent,
                        )
                        if nested_lines is None:
                            return None
                        lines.extend(nested_lines)
                    else:
                        value = self.metal_buffer_load(
                            buffer_name, element_offset, field_access
                        )
                        lines.append(f"{indent_str}{element_target} = {value};")
                continue
            if member.get("members"):
                nested_lines = self.metal_aggregate_load_assignments(
                    member_target, buffer_name, member_offset, field_access, indent
                )
                if nested_lines is None:
                    return None
                lines.extend(nested_lines)
            else:
                value = self.metal_buffer_load(buffer_name, member_offset, field_access)
                lines.append(f"{indent_str}{member_target} = {value};")
        return lines

    def generate_glsl_buffer_aggregate_load_helpers(self):
        if not self.required_glsl_buffer_aggregate_load_helpers:
            return ""

        helpers = []
        for helper_name, access in sorted(
            self.required_glsl_buffer_aggregate_load_helpers.items()
        ):
            lines = [
                f"static {access['metal_type']} {helper_name}("
                f"{access.get('memory', '')}const device uchar* buffer, uint offset) {{",
                f"    {access['metal_type']} result;",
            ]
            assignments = self.metal_aggregate_load_assignments(
                "result", "buffer", "offset", access
            )
            if assignments is None:
                continue
            lines.extend(assignments)
            lines.extend(["    return result;", "}"])
            helpers.append("\n".join(lines) + "\n\n")
        return "".join(helpers)

    def metal_aggregate_load(self, buffer_name, offset, access):
        if self.metal_aggregate_has_array_member(access):
            helper_name = self.metal_aggregate_load_helper_name(access)
            return f"{helper_name}({buffer_name}, {offset})"

        values = []
        for field_name, member in access["members"].items():
            if member.get("is_array"):
                return None
            member_offset = byte_offset_add(offset, member["offset"])
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
                "memory": access.get("memory", ""),
            }
            if member.get("members"):
                value = self.metal_aggregate_load(
                    buffer_name, member_offset, field_access
                )
            else:
                value = self.metal_buffer_load(buffer_name, member_offset, field_access)
            if value is None:
                return None
            values.append(value)
        return format_struct_constructor_expression(self, access["metal_type"], values)

    def metal_aggregate_store_members(self, buffer_name, offset, value, access):
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_value = f"{value}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
                "memory": access.get("memory", ""),
            }
            if member.get("is_array"):
                array_count = member.get("array_count")
                if member.get("runtime_array") or array_count is None:
                    return None
                for subscripts, element_offset in glsl_buffer_array_elements(
                    member, member_offset
                ):
                    element_value = f"{member_value}{subscripts}"
                    if member.get("members"):
                        nested_stores = self.metal_aggregate_store_members(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if nested_stores is None:
                            return None
                        lines.extend(nested_stores)
                    else:
                        store = self.metal_buffer_store(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if store is None:
                            return None
                        lines.extend(store.splitlines())
                continue
            if member.get("members"):
                nested_stores = self.metal_aggregate_store_members(
                    buffer_name, member_offset, member_value, field_access
                )
                if nested_stores is None:
                    return None
                lines.extend(nested_stores)
            else:
                store = self.metal_buffer_store(
                    buffer_name, member_offset, member_value, field_access
                )
                if store is None:
                    return None
                lines.extend(store.splitlines())
        return lines

    def metal_aggregate_store(self, buffer_name, offset, value, access):
        temp_name = self.next_metal_temp_variable("aggregate_store")
        stores = self.metal_aggregate_store_members(
            buffer_name, offset, temp_name, access
        )
        if stores is None:
            raise BufferBlockUnsupportedError(
                "metal",
                "aggregate store",
                f"storing a whole {access.get('type')} value through a device byte "
                "pointer needs element-wise stores for its array fields, which are "
                "not lowered",
            )
        return "\n".join([f"{access['metal_type']} {temp_name} = {value}", *stores])

    def metal_matrix_compound_store(self, buffer_name, offset, value, op, access):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            raise BufferBlockUnsupportedError(
                "metal",
                "matrix compound store",
                f"operator {op} has no matrix form (GLSL 4.60 section 5.9 defines "
                "+=, -=, *= and /= on matrices)",
            )

        current = self.metal_buffer_load(buffer_name, offset, access)
        rhs = f"({current} {binary_op} {value})"
        return self.metal_matrix_store(buffer_name, offset, rhs, access)

    def metal_buffer_compound_store_diagnostic(self, op, access):
        raise BufferBlockUnsupportedError(
            "metal",
            "compound store",
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members",
        )

    def metal_glsl_buffer_array_return(self, value):
        """``{e0, e1, ...}`` returning a whole buffer block array from an array-valued function, or ``None``.

        A Metal function returns ``array<T, N>``, which a brace list of the element loads initializes; the array
        itself has no Metal expression (glsl_buffer_array_values).
        """
        access = self.glsl_buffer_block_array_access(value)
        indexed = access is not None
        if access is None:
            access = self.glsl_buffer_block_member_access(value)
        if glsl_buffer_whole_array_reason(access, indexed=indexed) is None:
            return None
        return generate_metal_array_initializer(
            self, value, glsl_buffer_array_value_type(access)
        )

    def metal_glsl_buffer_whole_array_check(self, access, *, indexed):
        reason = glsl_buffer_whole_array_reason(access, indexed=indexed)
        if reason is not None:
            raise BufferBlockUnsupportedError("metal", "array value", reason)

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("runtime_array"):
            return None
        self.metal_glsl_buffer_whole_array_check(access, indexed=False)
        if access.get("members"):
            return self.metal_aggregate_load(access["buffer"], access["offset"], access)
        return self.metal_buffer_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        self.metal_glsl_buffer_whole_array_check(access, indexed=True)
        if access.get("members"):
            return self.metal_aggregate_load(
                access["buffer"], access["offset_expr"], access
            )
        return self.metal_buffer_load(access["buffer"], access["offset_expr"], access)

    def generate_glsl_buffer_block_store(self, target, rhs, op):
        component_store = metal_buffer_component_store(self, target, rhs, op)
        if component_store is not None:
            return component_store
        access = self.glsl_buffer_block_array_access(
            target
        ) or glsl_buffer_matrix_column_access(self, target)
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("runtime_array"):
                return None
            self.metal_glsl_buffer_whole_array_check(access, indexed=False)
            offset = access["offset"]
        else:
            self.metal_glsl_buffer_whole_array_check(access, indexed=True)
            offset = access["offset_expr"]

        if access.get("readonly"):
            raise BufferBlockUnsupportedError(
                "metal",
                "store",
                f"{access['buffer']} is a readonly block, and a readonly device "
                "buffer cannot be written",
            )
        if access.get("members"):
            if op != "=":
                raise BufferBlockUnsupportedError(
                    "metal",
                    "aggregate compound store",
                    f"operator {op} does not apply to a {access.get('type')} "
                    "aggregate; assign a full aggregate value explicitly",
                )
            return self.metal_aggregate_store(access["buffer"], offset, rhs, access)

        if access.get("matrix_columns"):
            if op != "=":
                return self.metal_matrix_compound_store(
                    access["buffer"], offset, rhs, op, access
                )
            return self.metal_matrix_store(access["buffer"], offset, rhs, access)

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return self.metal_buffer_compound_store_diagnostic(op, access)
            current = self.metal_buffer_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        return self.metal_buffer_store(access["buffer"], offset, rhs, access)

    def glsl_buffer_block_atomic_access(self, target):
        access = self.glsl_buffer_block_array_access(target)
        if access is not None:
            return access, access["offset_expr"]
        access = self.glsl_buffer_block_member_access(target)
        if access is None or access.get("is_array"):
            return None, None
        return access, access["offset"]

    def unsupported_glsl_buffer_block_atomic_call(
        self, target, operation, reason, access=None
    ):
        raise BufferBlockUnsupportedError(
            "metal", "atomic", f"Metal {operation} {reason}"
        )

    def glsl_buffer_block_diagnostic(
        self, target, type_name, var_name=None, node=None, declaration_kind=None
    ):
        declaration = str(self.resource_base_type(type_name))
        if declaration_kind:
            declaration = f"{declaration_kind} {declaration}"
        if var_name:
            declaration += f" {var_name}"
        details = ""
        if node is not None:
            layout = self.glsl_buffer_block_layout(node)
            binding = self.explicit_resource_binding_index(
                node, {"binding", "buffer"}, ("b", "u", "t")
            )
            details = f" ({layout}"
            if binding is not None:
                details += f", binding = {binding}"
            details += ")"
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(
            type_name, var_name
        )
        reason = f"; {failure_detail}" if failure_detail else ""
        raise BufferBlockUnsupportedError(
            target,
            "declaration",
            f"{declaration}{details}: device pointer offset lowering does not cover this block{reason}",
        )

    def unsupported_glsl_buffer_block_struct_placeholder(self, target, type_name):
        type_name = str(self.resource_base_type(type_name))
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(type_name)
        reason = f": {failure_detail}" if failure_detail else ""
        raise BufferBlockUnsupportedError(
            target,
            "struct",
            f"{type_name} has no device pointer offset lowering{reason}",
        )

    def unsupported_glsl_buffer_block_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        raise BufferBlockUnsupportedError(
            target, "variable", f"{declaration} has no device pointer offset lowering"
        )

    def unsupported_glsl_buffer_block_function_placeholder(
        self, target, func_name, info
    ):
        reason = info.get("reason", "signature references omitted GLSL buffer block")
        raise BufferBlockUnsupportedError(target, "function", f"{func_name}: {reason}")

    def unsupported_glsl_buffer_block_function_call(self, func_name):
        if (
            not func_name
            or func_name not in self.unsupported_glsl_buffer_block_functions
        ):
            return None
        info = self.unsupported_glsl_buffer_block_functions[func_name]
        reason = info.get("reason", "signature references omitted GLSL buffer block")
        raise BufferBlockUnsupportedError(
            "metal", "function call", f"{func_name}: {reason}"
        )

    def is_unsupported_glsl_buffer_block_struct_type(self, vtype):
        return str(self.resource_base_type(vtype)) in (
            self.unsupported_glsl_buffer_block_struct_names
        )

    def unsupported_glsl_buffer_block_local_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        raise BufferBlockUnsupportedError(
            target,
            "local variable",
            f"{declaration} has a block struct type with no device pointer offset lowering",
        )

    def is_unsupported_glsl_buffer_block_name(self, name):
        return name in self.unsupported_glsl_buffer_block_variables or (
            name in self.current_unsupported_glsl_buffer_block_parameters
            or name in self.current_unsupported_glsl_buffer_block_local_variables
        )

    def unsupported_glsl_buffer_block_name_type(self, name):
        if not self.is_unsupported_glsl_buffer_block_name(name):
            return None
        return self.unsupported_glsl_buffer_block_variable_types.get(
            name
        ) or self.local_variable_types.get(name)

    def unsupported_glsl_buffer_block_expression_type(self, expr):
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_type = self.unsupported_glsl_buffer_block_expression_type(
                object_expr
            )
            member_name = str(getattr(expr, "member", ""))
            if object_type and member_name:
                return self.struct_member_types.get(
                    self.type_name_string(object_type), {}
                ).get(member_name)
            return None

        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.unsupported_glsl_buffer_block_expression_type(array_expr)
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type

        expr_name = self.expression_name(expr)
        if expr_name:
            block_type = self.unsupported_glsl_buffer_block_name_type(expr_name)
            if block_type:
                return block_type
        return None

    def unsupported_glsl_buffer_block_access_name(self, expr):
        expr_name = self.expression_name(expr)
        if expr_name and self.is_unsupported_glsl_buffer_block_name(expr_name):
            return expr_name
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.unsupported_glsl_buffer_block_access_name(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.unsupported_glsl_buffer_block_access_name(array_expr)
        return None

    def diagnostic_zero_value_for_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type in {"float", "half", "double", "int"}:
            return "0"
        if (
            mapped_type
            and mapped_type[0].isalpha()
            and any(char.isdigit() for char in mapped_type)
        ):
            return f"{mapped_type}(0)"
        return "0"

    def unsupported_glsl_buffer_block_access_value(self, expr):
        name = self.unsupported_glsl_buffer_block_access_name(expr)
        if not name:
            return None
        raise BufferBlockUnsupportedError(
            "metal", "access", f"{name} has no device pointer offset lowering"
        )

    def unsupported_glsl_buffer_block_assignment_diagnostic(self, target):
        name = self.unsupported_glsl_buffer_block_access_name(target)
        if not name:
            return None
        raise BufferBlockUnsupportedError(
            "metal", "assignment", f"{name} has no device pointer offset lowering"
        )

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def global_resource_shape(self, node):
        resource_count = 1
        if hasattr(node, "var_type"):
            if (
                hasattr(node.var_type, "name")
                or hasattr(node.var_type, "element_type")
                or isinstance(node.var_type, (PointerType, ReferenceType))
            ):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):
                    vtype = self.convert_type_node_to_string(node.var_type.element_type)
                    array_size = (
                        self.expression_to_string(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    )
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
            else:
                vtype = str(node.var_type)
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            vtype = "float"
        return vtype, resource_count

    def resource_node_name(self, node, default=None):
        if node is None:
            return default
        return getattr(
            node,
            "name",
            getattr(node, "variable_name", getattr(node, "type_name", default)),
        )
