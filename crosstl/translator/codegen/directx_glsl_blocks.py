"""GLSL buffer blocks and byte address buffers in the HLSL generator: stores and loads, aggregates, atomics, parameters, arrays and placeholders, layouts and diagnostics."""

from hashlib import sha1

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    MemberAccessNode,
    PointerType,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    parse_array_type,
    split_array_type_suffix,
)
from .buffer_block_diagnostics import BufferBlockUnsupportedError
from .glsl_buffer_array_values import (
    glsl_buffer_array_store_lines,
    glsl_buffer_array_value,
    glsl_buffer_array_value_type,
    glsl_buffer_subscripts_are_repeatable,
)
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
from .hlsl_buffer_component_stores import buffer_component_store
from .hlsl_byteaddress_typed_access import (
    hlsl_byteaddress_component_refusal,
    hlsl_byteaddress_is_typed_access,
    hlsl_byteaddress_typed_load,
    hlsl_byteaddress_typed_store,
)
from .hlsl_image_formats import (
    declared_texel_type,
    normalized_texel_type,
    texel_attribute,
)
from .hlsl_texel_buffers import STORAGE_TEXEL_BUFFER_TYPES
from .image_access_contracts import (
    explicit_image_format,
    image_atomic_result_kind_mismatch,
)
from .storage_image_texels import TEXEL_COMPONENTS


class DirectXGlslBlockMixin:
    """Glsl block members of ``HLSLCodeGen``."""

    def hlsl_metal_buffer_parameter_resource_type(self, vtype, node=None):
        if node is None:
            return None

        attributes = getattr(node, "attributes", []) or []
        has_buffer_attribute = any(
            str(getattr(attr, "name", "")).lower() == "buffer" for attr in attributes
        )

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        resource_qualifiers = qualifiers.intersection({"device", "constant", "const"})
        address_space_qualifiers = qualifiers.intersection({"device", "constant"})
        if not has_buffer_attribute and not address_space_qualifiers:
            return None
        if qualifiers and not resource_qualifiers:
            return None

        if self.hlsl_resource_pointer_array_parameter_type_node(node) is not None:
            return None

        element_type = self.hlsl_pointer_element_type(vtype)
        if not element_type:
            element_type = self.hlsl_array_element_type_name(vtype)
        if not element_type:
            return None

        buffer_type = (
            "StructuredBuffer"
            if qualifiers.intersection({"constant", "const"})
            else "RWStructuredBuffer"
        )
        storage_type = self.hlsl_structured_storage_type(
            element_type,
            operation=f"Metal buffer '{getattr(node, 'name', '<anonymous>')}' element",
            source_location=getattr(node, "source_location", None),
        )
        return f"{buffer_type}<{storage_type}>"

    def hlsl_pointer_pointee_type_once(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        while type_name.endswith("&"):
            type_name = type_name[:-1].strip()
        if not type_name.endswith("*"):
            return None
        pointee_type = type_name[:-1].strip()
        return pointee_type or None

    def hlsl_pointer_element_type(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        pointer_depth = 0
        while type_name.endswith("*") or type_name.endswith("&"):
            if type_name.endswith("*"):
                pointer_depth += 1
            type_name = type_name[:-1].strip()
        if pointer_depth == 0:
            return None

        if type_name.startswith("metal::"):
            type_name = type_name.split("metal::", 1)[1].strip()
        metal_qualifiers = {
            "const",
            "constant",
            "device",
            "thread",
            "threadgroup",
            "volatile",
            "restrict",
        }
        parts = type_name.split()
        while parts and parts[0].lower() in metal_qualifiers:
            parts.pop(0)
        return " ".join(parts) if parts else None

    def hlsl_constant_reference_parameter_type(self, vtype, node=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "constant" not in qualifiers:
            return None

        element_type = self.hlsl_constant_reference_element_type(vtype)
        if element_type is None:
            return None

        referenced_type = self.map_type(element_type)
        # ConstantBuffer<T> in HLSL requires T to be a user-defined struct/class.
        # Scalars, vectors, and matrices -- including the non-32-bit forms such
        # as uint64_t / int64_t / half and the min16* minimum-precision types --
        # must be delivered through a generated cbuffer block; emitting
        # ConstantBuffer<uint64_t> or ConstantBuffer<half> is a DXC compile
        # error ("cannot be used as a type parameter where a struct/class is
        # required"). Returning the plain element type routes these MLX scalar
        # `constant T&` parameters through the scalar cbuffer path instead.
        if (
            self.is_scalar_value_type(element_type)
            or self.is_vector_value_type(element_type)
            or self.hlsl_matrix_shape(element_type) is not None
        ):
            return referenced_type
        return f"ConstantBuffer<{referenced_type}>"

    def glsl_buffer_block_parameter_type(
        self, mapped_type, vtype, node=None, function_hints=None
    ):
        function_hints = function_hints or {}
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            array_size = (
                self.expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )
            return f"{mapped_type}[{array_size}]" if array_size else f"{mapped_type}[]"

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                _, array_suffix = split_array_type_suffix(type_string)
                if array_suffix == "[]":
                    array_size = function_hints.get(param_name, "")
                    return (
                        f"{mapped_type}[{array_size}]"
                        if array_size
                        else f"{mapped_type}[]"
                    )
                return f"{mapped_type}{array_suffix}"

        return mapped_type

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if isinstance(vtype, PointerType):
            vtype_str = self.convert_type_node_to_string(vtype)
        elif hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def storage_parameter_type(self, node):
        """The HLSL type of the storage images a parameter receives, or ``None``."""
        if node is None or texel_attribute(node) is not None:
            return None
        try:
            return self.storage_parameter_texels.parameter_type(
                node,
                lambda declaration: self.map_image_base_type_with_format(
                    declaration.vtype, declaration
                ),
            )
        except ValueError as error:
            return self.refuse_lowering(f"DirectX {error}", None)

    def declared_storage_texel_type(self, node, image_type, explicit_format):
        """The HLSL texel type a storage image's ``@texel`` declares, or ``None``."""
        if node is None:
            return None
        try:
            return declared_texel_type(
                node, image_type, explicit_format, self.attribute_value_to_string
            )
        except ValueError as error:
            return self.refuse_lowering(f"DirectX {error}", None)

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = (
            explicit_image_format(node, self.attribute_value_to_string)
            if node is not None
            else None
        )
        if explicit_format is None:
            parameter_type = self.storage_parameter_type(node)
            if parameter_type is not None:
                return parameter_type
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "RWTexture1D",
            "iimage1D": "RWTexture1D",
            "uimage1D": "RWTexture1D",
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "image1DArray": "RWTexture1DArray",
            "iimage1DArray": "RWTexture1DArray",
            "uimage1DArray": "RWTexture1DArray",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "iimage2DMS": "RWTexture2DMS",
            "uimage2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "iimage2DMSArray": "RWTexture2DMSArray",
            "uimage2DMSArray": "RWTexture2DMSArray",
            "imageCube": "RWTextureCube",
            "RWTexture1D": "RWTexture1D",
            "RWTexture1DArray": "RWTexture1DArray",
            "RWTexture2D": "RWTexture2D",
            "RWTexture2DArray": "RWTexture2DArray",
            "RWTexture3D": "RWTexture3D",
        }
        texture_type = texture_types.get(base_type)
        declared_texel = (
            self.declared_storage_texel_type(node, base_type, explicit_format)
            if texture_type
            else None
        )
        if component_type and texture_type:
            texel_type = normalized_texel_type(explicit_format, component_type)
            mapped_type = f"{texture_type}<{texel_type}>"
        elif declared_texel:
            mapped_type = f"{texture_type}<{declared_texel}>"
        elif texture_type and base_type.startswith(("iimage", "uimage")):
            # An integer storage image without a format is as wide as its texels,
            # except as an image atomic's target, which needs a scalar element.
            component_kind = "int" if base_type.startswith("i") else "uint"
            atomic_target = self.hlsl_atomic_resources.is_atomic_declaration(node)
            width = "" if atomic_target else str(TEXEL_COMPONENTS)
            mapped_type = f"{texture_type}<{component_kind}{width}>"
        elif base_type in STORAGE_TEXEL_BUFFER_TYPES and node is not None:
            # Parameters take the declaration a global of the same name would.
            mapped_type = self.hlsl_texel_buffers.storage_declaration_type(
                getattr(node, "name", None),
                base_type,
                STORAGE_TEXEL_BUFFER_TYPES[base_type],
                node,
            )
        else:
            mapped_type = self.map_type(vtype)
        return self.rasterizer_ordered_resource_type(mapped_type, node)

    def rasterizer_ordered_resource_type(self, mapped_type, node=None):
        if node is None or not self.is_rasterizer_ordered_resource(node):
            return mapped_type

        type_text = str(mapped_type)
        if "[" in type_text and "]" in type_text:
            base_type, array_suffix = split_array_type_suffix(type_text)
        else:
            base_type, array_suffix = type_text, ""

        if "<" in base_type and base_type.endswith(">"):
            resource_name, generic_args = base_type.split("<", 1)
            generic_suffix = f"<{generic_args}"
        else:
            resource_name = base_type
            generic_suffix = ""

        resource_map = {
            "RWTexture1D": "RasterizerOrderedTexture1D",
            "RWTexture1DArray": "RasterizerOrderedTexture1DArray",
            "RWTexture2D": "RasterizerOrderedTexture2D",
            "RWTexture2DArray": "RasterizerOrderedTexture2DArray",
            "RWTexture3D": "RasterizerOrderedTexture3D",
            "RWBuffer": "RasterizerOrderedBuffer",
            "RWStructuredBuffer": "RasterizerOrderedStructuredBuffer",
            "RWByteAddressBuffer": "RasterizerOrderedByteAddressBuffer",
        }
        rasterizer_type = resource_map.get(resource_name)
        if rasterizer_type is None:
            return mapped_type
        return f"{rasterizer_type}{generic_suffix}{array_suffix}"

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return base_type
        return vtype

    def sampled_texture_shape_type(self, texture_type):
        if texture_type is None:
            return ""
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        if not texture_type.startswith("Texture"):
            return texture_type
        if "<" not in texture_type or ">" not in texture_type:
            return texture_type
        shape_type = texture_type.split("<", 1)[0]
        sampled_shapes = {
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture2DMS",
            "Texture2DMSArray",
            "Texture3D",
            "TextureCube",
            "TextureCubeArray",
        }
        return shape_type if shape_type in sampled_shapes else texture_type

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
            target_type_key="hlsl_type",
            unsupported_type_message=(
                "type is not supported by ByteAddressBuffer lowering"
            ),
            component_refusal=lambda info: hlsl_byteaddress_component_refusal(
                self, info
            ),
        )
        self.record_unsupported_glsl_buffer_block_layout_failures(
            parameters, blocks, var_failures, struct_failures
        )
        return blocks, var_failures, struct_failures

    def record_unsupported_glsl_buffer_block_layout_failures(
        self, nodes, lowered_blocks, var_failures, struct_failures
    ):
        for node in nodes or []:
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if not var_name or var_name in lowered_blocks or var_name in var_failures:
                continue
            node_type = glsl_buffer_block_node_type(node)
            if not self.is_glsl_buffer_block_variable(node, node_type):
                continue
            layout = self.glsl_buffer_block_layout(node)
            layout_key = str(layout).lower()
            if layout_key in {"std140", "std430"}:
                continue
            type_name = str(self.resource_base_type(node_type))
            reason = (
                f"unsupported layout {layout}: DirectX ByteAddressBuffer lowering "
                "currently supports std140 and std430 only"
            )
            var_failures[var_name] = reason
            struct_failures.setdefault(type_name, reason)

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
                    **member,
                }

        parent = self.glsl_buffer_block_array_access(
            object_expr
        ) or self.glsl_buffer_block_member_access(object_expr)
        if parent is None or not parent.get("members") or parent.get("subarray"):
            return None
        member_name = getattr(expr, "member", None)
        member = parent["members"].get(member_name)
        if member is None:
            return None
        return {
            "buffer": parent["buffer"],
            "member": f"{parent['member']}.{member_name}",
            "readonly": parent["readonly"],
            **member,
            "offset": byte_offset_add(parent["offset"], member["offset"]),
        }

    def glsl_buffer_block_array_access(self, expr):
        return glsl_buffer_block_indexed_access(self, expr)

    def hlsl_byteaddress_load_method(self, components):
        return "Load" if components == 1 else f"Load{components}"

    def hlsl_byteaddress_store_method(self, components):
        return "Store" if components == 1 else f"Store{components}"

    def hlsl_byteaddress_load(self, buffer_name, offset, access):
        if access.get("matrix_columns"):
            columns = []
            for _, column_offset in matrix_column_offsets(
                offset, access["matrix_columns"], access["column_stride"]
            ):
                load = (
                    f"{buffer_name}."
                    f"{self.hlsl_byteaddress_load_method(access['matrix_rows'])}"
                    f"({column_offset})"
                )
                columns.append(f"asfloat({load})")
            # The buffer holds columns and an HLSL constructor fills rows.
            transposed_type = (
                f"{access['component_type']}{access['matrix_columns']}"
                f"x{access['matrix_rows']}"
            )
            return f"transpose({transposed_type}({', '.join(columns)}))"

        if hlsl_byteaddress_is_typed_access(access):
            return hlsl_byteaddress_typed_load(self, buffer_name, offset, access)

        if access["component_type"] == "bool" and access["components"] > 1:
            values = []
            for _, component_offset in vector_component_offsets(
                offset, access["components"]
            ):
                values.append(f"({buffer_name}.Load({component_offset}) != 0u)")
            return f"{access['hlsl_type']}({', '.join(values)})"

        load = (
            f"{buffer_name}."
            f"{self.hlsl_byteaddress_load_method(access['components'])}({offset})"
        )
        if access["component_type"] == "bool":
            return f"({load} != 0u)"
        if access["component_type"] == "float":
            return f"asfloat({load})"
        if access["component_type"] == "int":
            return f"asint({load})"
        return load

    def hlsl_byteaddress_store_value(self, value, access):
        if access["component_type"] == "bool":
            if access["components"] > 1:
                value_expr = self.hlsl_indexable_expression(value)
                fields = "xyzw"[: access["components"]]
                values = [f"({value_expr}.{field} ? 1u : 0u)" for field in fields]
                return f"uint{access['components']}({', '.join(values)})"
            return f"(({value}) ? 1u : 0u)"
        if access.get("layout_type") != access.get("type"):
            if access["component_type"] == "int":
                return f"asuint(int({value}))"
            if access["component_type"] == "uint":
                if access["components"] == 1:
                    return f"uint({value})"
                return f"uint{access['components']}({value})"
        if access["component_type"] in {"float", "int"}:
            return f"asuint({value})"
        return value

    def next_hlsl_temp_variable(self, prefix):
        name = f"__crossgl_{prefix}_{self.hlsl_temp_variable_index}"
        self.hlsl_temp_variable_index += 1
        return name

    def hlsl_indexable_expression(self, expression):
        expression = str(expression)
        if expression.isidentifier():
            return expression
        if all(part.isidentifier() for part in expression.split(".")):
            return expression
        return f"({expression})"

    def hlsl_byteaddress_matrix_store(self, buffer_name, offset, value, access):
        value_expr = self.hlsl_indexable_expression(value)
        store_method = self.hlsl_byteaddress_store_method(access["matrix_rows"])
        lines = []
        for column, column_offset in matrix_column_offsets(
            offset, access["matrix_columns"], access["column_stride"]
        ):
            # HLSL m[i] is a row; the swizzle selects column ``column``.
            swizzle = "".join(
                f"_m{row}{column}" for row in range(access["matrix_rows"])
            )
            lines.append(
                f"{buffer_name}.{store_method}"
                f"({column_offset}, asuint({value_expr}.{swizzle}))"
            )
        return "\n".join(lines)

    def hlsl_byteaddress_bool_vector_store(self, buffer_name, offset, value, access):
        temp_name = self.next_hlsl_temp_variable("bool_store")
        store_method = self.hlsl_byteaddress_store_method(access["components"])
        store_value = self.hlsl_byteaddress_store_value(temp_name, access)
        return "\n".join(
            [
                f"{access['hlsl_type']} {temp_name} = {value}",
                f"{buffer_name}.{store_method}({offset}, {store_value})",
            ]
        )

    def hlsl_buffer_aggregate_helper_suffix(self, access):
        return "".join(
            char if char.isalnum() or char == "_" else "_"
            for char in access["hlsl_type"]
        )

    def hlsl_buffer_aggregate_layout_signature(self, access):
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

    def hlsl_byteaddress_aggregate_load_helper_name(self, access):
        buffer_type = (
            "ByteAddressBuffer" if access.get("readonly") else "RWByteAddressBuffer"
        )
        kind = "ro" if access.get("readonly") else "rw"
        layout_hash = self.hlsl_buffer_aggregate_layout_signature(access)
        helper_name = (
            f"__crossgl_load_{kind}_glsl_buffer_"
            f"{self.hlsl_buffer_aggregate_helper_suffix(access)}_{layout_hash}"
        )
        self.required_glsl_buffer_aggregate_load_helpers[(helper_name, buffer_type)] = (
            access
        )
        return helper_name

    def hlsl_byteaddress_aggregate_load_assignments(
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
                        nested_lines = self.hlsl_byteaddress_aggregate_load_assignments(
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
                        value = self.hlsl_byteaddress_load(
                            buffer_name, element_offset, field_access
                        )
                        lines.append(f"{indent_str}{element_target} = {value};")
                continue
            if member.get("members"):
                nested_lines = self.hlsl_byteaddress_aggregate_load_assignments(
                    member_target, buffer_name, member_offset, field_access, indent
                )
                if nested_lines is None:
                    return None
                lines.extend(nested_lines)
            else:
                value = self.hlsl_byteaddress_load(
                    buffer_name, member_offset, field_access
                )
                lines.append(f"{indent_str}{member_target} = {value};")
        return lines

    def hlsl_byteaddress_aggregate_load(self, buffer_name, offset, access):
        helper_name = self.hlsl_byteaddress_aggregate_load_helper_name(access)
        return f"{helper_name}({buffer_name}, {offset})"

    def generate_glsl_buffer_aggregate_load_helpers(self):
        if not self.required_glsl_buffer_aggregate_load_helpers:
            return ""

        helpers = []
        for (
            helper_name,
            buffer_type,
        ), access in sorted(self.required_glsl_buffer_aggregate_load_helpers.items()):
            lines = [
                f"{access['hlsl_type']} {helper_name}({buffer_type} buffer, uint offset) {{",
                f"    {access['hlsl_type']} result;",
            ]
            assignments = self.hlsl_byteaddress_aggregate_load_assignments(
                "result", "buffer", "offset", access
            )
            if assignments is None:
                continue
            lines.extend(assignments)
            lines.extend(["    return result;", "}"])
            helpers.append("\n".join(lines) + "\n\n")
        return "".join(helpers)

    def hlsl_byteaddress_leaf_store(self, buffer_name, offset, value, access):
        if access.get("matrix_columns"):
            return self.hlsl_byteaddress_matrix_store(
                buffer_name, offset, value, access
            )
        if hlsl_byteaddress_is_typed_access(access):
            return hlsl_byteaddress_typed_store(
                self, buffer_name, offset, value, access
            )
        if access["component_type"] == "bool" and access["components"] > 1:
            return self.hlsl_byteaddress_bool_vector_store(
                buffer_name, offset, value, access
            )
        store_value = self.hlsl_byteaddress_store_value(value, access)
        store_method = self.hlsl_byteaddress_store_method(access["components"])
        return f"{buffer_name}.{store_method}({offset}, {store_value})"

    def hlsl_byteaddress_aggregate_store_members(
        self, buffer_name, offset, value, access
    ):
        lines = []
        for field_name, member in access["members"].items():
            member_offset = byte_offset_add(offset, member["offset"])
            member_value = f"{value}.{field_name}"
            field_access = {
                **member,
                "buffer": buffer_name,
                "member": f"{access['member']}.{field_name}",
                "readonly": access["readonly"],
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
                        nested_stores = self.hlsl_byteaddress_aggregate_store_members(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if nested_stores is None:
                            return None
                        lines.extend(nested_stores)
                    else:
                        store = self.hlsl_byteaddress_leaf_store(
                            buffer_name, element_offset, element_value, field_access
                        )
                        if store is None:
                            return None
                        lines.extend(store.splitlines())
                continue
            if member.get("members"):
                nested_stores = self.hlsl_byteaddress_aggregate_store_members(
                    buffer_name, member_offset, member_value, field_access
                )
                if nested_stores is None:
                    return None
                lines.extend(nested_stores)
            else:
                store = self.hlsl_byteaddress_leaf_store(
                    buffer_name, member_offset, member_value, field_access
                )
                if store is None:
                    return None
                lines.extend(store.splitlines())
        return lines

    def hlsl_byteaddress_aggregate_store(self, buffer_name, offset, value, access):
        temp_name = self.next_hlsl_temp_variable("aggregate_store")
        stores = self.hlsl_byteaddress_aggregate_store_members(
            buffer_name, offset, temp_name, access
        )
        if stores is None:
            raise BufferBlockUnsupportedError(
                "directx",
                "aggregate store",
                f"storing a whole {access.get('type')} value into a ByteAddressBuffer "
                "needs element-wise stores for its array fields, which are not lowered",
            )
        return "\n".join([f"{access['hlsl_type']} {temp_name} = {value}", *stores])

    def hlsl_byteaddress_matrix_compound_store(
        self, buffer_name, offset, value, op, access, *, value_is_matrix=False
    ):
        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }
        binary_op = compound_ops.get(op)
        if binary_op is None:
            raise BufferBlockUnsupportedError(
                "directx",
                "matrix compound store",
                f"operator {op} has no matrix form (GLSL 4.60 section 5.9 defines "
                "+=, -=, *= and /= on matrices)",
            )

        temp_name = self.next_hlsl_temp_variable("matrix_store")
        current = self.hlsl_byteaddress_load(buffer_name, offset, access)
        if binary_op == "*" and value_is_matrix:
            # CrossGL m *= n is the linear-algebra product; HLSL * is componentwise.
            updated = f"mul({current}, {value})"
        else:
            updated = f"({current} {binary_op} {value})"
        temp = f"{access['hlsl_type']} {temp_name} = {updated}"
        stores = self.hlsl_byteaddress_matrix_store(
            buffer_name, offset, temp_name, access
        )
        return f"{temp}\n{stores}"

    def hlsl_byteaddress_compound_store_diagnostic(self, op, access):
        raise BufferBlockUnsupportedError(
            "directx",
            "compound store",
            f"operator {op} is not supported for "
            f"{access['component_type']} buffer members",
        )

    def hlsl_glsl_buffer_array_value(self, expr, access):
        """The whole block array ``access`` read as one value (glsl_buffer_array_values)."""
        if not glsl_buffer_subscripts_are_repeatable(expr):
            raise BufferBlockUnsupportedError(
                "directx",
                "array value",
                f"{access.get('member')} is read element by element, and its "
                "subscripts have side effects",
            )

        def load_element(element):
            if element.get("members"):
                return self.hlsl_byteaddress_aggregate_load(
                    element["buffer"], element["offset"], element
                )
            return self.hlsl_byteaddress_load(
                element["buffer"], element["offset"], element
            )

        return glsl_buffer_array_value(
            access,
            load_element=load_element,
            value_struct=self.hlsl_array_value_arguments.value_struct,
            map_type=self.map_type,
        )

    def hlsl_glsl_buffer_whole_array_target(self, target):
        """Whether ``target`` is a whole array of a lowered GLSL buffer block."""
        access = self.glsl_buffer_block_array_access(target)
        if access is not None:
            return glsl_buffer_whole_array_reason(access, indexed=True) is not None
        access = self.glsl_buffer_block_member_access(target)
        return glsl_buffer_whole_array_reason(access, indexed=False) is not None

    def hlsl_glsl_buffer_array_store(self, target, access, offset, value, op):
        """Statements storing array ``value`` into the whole block array ``access``."""
        if op != "=":
            raise BufferBlockUnsupportedError(
                "directx",
                "array value",
                f"operator {op} does not apply to the array {access.get('member')}",
            )
        if not glsl_buffer_subscripts_are_repeatable(target):
            raise BufferBlockUnsupportedError(
                "directx",
                "array value",
                f"{access.get('member')} is written element by element, and its "
                "subscripts have side effects",
            )
        value_type = glsl_buffer_array_value_type(access)
        rhs = None
        if isinstance(value, ArrayLiteralNode):
            rhs = self.hlsl_array_value_arguments.argument(value, value_type)
        if rhs is None:
            rhs = self.generate_expression_with_expected(value, value_type)

        def store_element(element, element_offset, element_value):
            if element.get("members"):
                lines = self.hlsl_byteaddress_aggregate_store_members(
                    element["buffer"], element_offset, element_value, element
                )
                if lines is None:
                    raise BufferBlockUnsupportedError(
                        "directx",
                        "array value",
                        f"the elements of {access.get('member')} hold runtime arrays",
                    )
                return lines
            return self.hlsl_byteaddress_leaf_store(
                element["buffer"], element_offset, element_value, element
            ).splitlines()

        return "\n".join(
            glsl_buffer_array_store_lines(
                access,
                offset,
                rhs,
                self.next_hlsl_temp_variable("array_store"),
                store_element=store_element,
                map_type=self.map_type,
            )
        )

    def generate_glsl_buffer_block_member_load(self, expr):
        access = self.glsl_buffer_block_member_access(expr)
        if access is None or access.get("runtime_array"):
            return None
        if glsl_buffer_whole_array_reason(access, indexed=False) is not None:
            return self.hlsl_glsl_buffer_array_value(expr, access)
        if access.get("members"):
            return self.hlsl_byteaddress_aggregate_load(
                access["buffer"], access["offset"], access
            )
        return self.hlsl_byteaddress_load(access["buffer"], access["offset"], access)

    def generate_glsl_buffer_block_array_load(self, expr):
        access = self.glsl_buffer_block_array_access(expr)
        if access is None:
            return None
        if glsl_buffer_whole_array_reason(access, indexed=True) is not None:
            return self.hlsl_glsl_buffer_array_value(expr, access)
        if access.get("members"):
            return self.hlsl_byteaddress_aggregate_load(
                access["buffer"], access["offset_expr"], access
            )
        return self.hlsl_byteaddress_load(
            access["buffer"], access["offset_expr"], access
        )

    def generate_glsl_buffer_block_store(self, target, value, op):
        access = self.glsl_buffer_block_array_access(
            target
        ) or glsl_buffer_matrix_column_access(self, target)
        indexed = access is not None
        if access is None:
            access = self.glsl_buffer_block_member_access(target)
            if access is None or access.get("runtime_array"):
                return buffer_component_store(self, target, value, op)
            offset = access["offset"]
        else:
            offset = access["offset_expr"]
        whole_array = glsl_buffer_whole_array_reason(access, indexed=indexed)

        if access.get("readonly"):
            raise BufferBlockUnsupportedError(
                "directx",
                "store",
                f"{access['buffer']} is a readonly block, and a readonly "
                "ByteAddressBuffer cannot be written",
            )
        if whole_array is not None:
            return self.hlsl_glsl_buffer_array_store(target, access, offset, value, op)
        if access.get("members"):
            if op != "=":
                raise BufferBlockUnsupportedError(
                    "directx",
                    "aggregate compound store",
                    f"operator {op} does not apply to a {access.get('type')} "
                    "aggregate; assign a full aggregate value explicitly",
                )
            rhs = self.generate_expression_with_expected(value, access["type"])
            return self.hlsl_byteaddress_aggregate_store(
                access["buffer"], offset, rhs, access
            )

        rhs = self.generate_expression_with_expected(value, access["type"])
        if access.get("matrix_columns"):
            if op != "=":
                return self.hlsl_byteaddress_matrix_compound_store(
                    access["buffer"],
                    offset,
                    rhs,
                    op,
                    access,
                    value_is_matrix=(
                        self.hlsl_matrix_orientation.expression_dimensions(value)
                        is not None
                    ),
                )
            return self.hlsl_byteaddress_matrix_store(
                access["buffer"], offset, rhs, access
            )

        if op != "=":
            binary_op = glsl_buffer_compound_binary_operator(
                op, access["component_type"]
            )
            if binary_op is None:
                return self.hlsl_byteaddress_compound_store_diagnostic(op, access)
            current = self.hlsl_byteaddress_load(access["buffer"], offset, access)
            rhs = f"({current} {binary_op} {rhs})"

        if access["component_type"] == "bool" and access["components"] > 1:
            return self.hlsl_byteaddress_bool_vector_store(
                access["buffer"], offset, rhs, access
            )

        return self.hlsl_byteaddress_leaf_store(access["buffer"], offset, rhs, access)

    def glsl_buffer_block_atomic_access(self, target):
        access = self.glsl_buffer_block_array_access(target)
        if access is not None:
            return access, access["offset_expr"]
        access = self.glsl_buffer_block_member_access(target)
        if access is None or access.get("runtime_array"):
            return None, None
        return access, access["offset"]

    def unsupported_glsl_buffer_block_atomic_call(
        self, target, operation, reason, access=None
    ):
        raise BufferBlockUnsupportedError("directx", "atomic", f"{operation} {reason}")

    def generate_glsl_buffer_block_atomic_call(self, func_name, args):
        operations = self.byteaddress_atomic_operations()
        operation_info = operations.get(func_name)
        if (
            operation_info is None
            or func_name in getattr(self, "function_return_types", {})
            or not args
        ):
            return None

        operation, intrinsic, expected_args = operation_info
        target = args[0]
        access, offset = self.glsl_buffer_block_atomic_access(target)
        if access is None:
            return None
        self.validate_hlsl_byteaddress_atomic_argument_count(
            func_name, args, expected_args
        )
        if access.get("readonly"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                (
                    "cannot read readonly ByteAddressBuffer through an interlocked operation"
                    if func_name == "atomicLoad"
                    else "cannot write readonly ByteAddressBuffer"
                ),
                access,
            )
        if access.get("components") != 1 or access.get("matrix_columns"):
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "requires a scalar int or uint buffer member",
                access,
            )
        if access.get("component_type") not in {"int", "uint"}:
            return self.unsupported_glsl_buffer_block_atomic_call(
                target,
                func_name,
                "currently supports only int or uint buffer members",
                access,
            )

        if func_name != "atomicLoad":
            self.validate_hlsl_byteaddress_atomic_value_arguments(
                func_name, args, access, operation
            )
        self.validate_hlsl_byteaddress_atomic_result_context(func_name, access)

        component_type = access["component_type"]
        helper_name = self.byteaddress_atomic_helper_name(operation, component_type)
        self.required_byteaddress_atomic_helpers.add(
            (operation, intrinsic, component_type)
        )

        if operation == "compare_exchange":
            compare_value = self.hlsl_byteaddress_atomic_value(args[1], access)
            replacement = self.hlsl_byteaddress_atomic_value(args[2], access)
            return (
                f"{helper_name}({access['buffer']}, {offset}, "
                f"{compare_value}, {replacement})"
            )

        if func_name == "atomicLoad":
            value = "0u" if component_type == "uint" else "0"
        else:
            value = self.hlsl_byteaddress_atomic_value(args[1], access)
        return f"{helper_name}({access['buffer']}, {offset}, {value})"

    def hlsl_byteaddress_atomic_value(self, arg, access):
        value = self.generate_expression_with_expected(arg, access["type"])
        if (
            access.get("component_type") == "uint"
            and self.scalar_expression_kind(arg) == "int"
        ):
            return f"uint({value})"
        return value

    def validate_hlsl_byteaddress_atomic_argument_count(
        self, func_name, args, expected_args
    ):
        if len(args) == expected_args:
            return
        raise DirectXRefusal(
            f"DirectX GLSL buffer block atomic '{func_name}' requires "
            f"{expected_args} argument(s), got {len(args)}"
        )

    def validate_hlsl_byteaddress_atomic_value_arguments(
        self, func_name, args, access, operation
    ):
        expected_kind = access.get("component_type")
        if expected_kind not in {"int", "uint"}:
            return

        if operation == "compare_exchange":
            value_roles = ((1, "compare value"), (2, "replacement"))
        else:
            value_roles = ((1, "value"),)
        for index, role in value_roles:
            value_kind = self.scalar_expression_kind(args[index])
            if value_kind is None or value_kind == expected_kind:
                continue
            if (value_kind, expected_kind) == ("int", "uint"):
                # GLSL converts an int argument to a uint parameter implicitly (GLSL 4.60 section 4.1.10); the
                # value is passed through an explicit uint conversion.
                continue
            raise DirectXRefusal(
                f"DirectX GLSL buffer block atomic '{func_name}' {role} "
                f"argument must be scalar {expected_kind}, got {value_kind}"
            )

    def validate_hlsl_byteaddress_atomic_result_context(self, func_name, access):
        component_type = access.get("component_type")
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_type
        )
        if expected_kind is None:
            return

        member_type = self.type_name_string(access.get("type")) or component_type
        raise DirectXRefusal(
            f"DirectX GLSL buffer block atomic '{func_name}' requires "
            f"{component_type} result context for {member_type} buffer member: "
            f"expected {expected_kind}"
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
                node, {"binding", "buffer", "texture", "uav"}, ("b", "t", "u")
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
            f"{declaration}{details}: ByteAddressBuffer offset lowering does not cover this block{reason}",
        )

    def unsupported_glsl_buffer_block_struct_placeholder(self, target, type_name):
        type_name = str(self.resource_base_type(type_name))
        failure_detail = self.glsl_buffer_block_lowering_failure_detail(type_name)
        reason = f": {failure_detail}" if failure_detail else ""
        raise BufferBlockUnsupportedError(
            target,
            "struct",
            f"{type_name} has no ByteAddressBuffer lowering{reason}",
        )

    def unsupported_glsl_buffer_block_variable_placeholder(
        self, target, type_name, var_name
    ):
        declaration = str(self.resource_base_type(type_name))
        if var_name:
            declaration += f" {var_name}"
        raise BufferBlockUnsupportedError(
            target, "variable", f"{declaration} has no ByteAddressBuffer lowering"
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
            "directx", "function call", f"{func_name}: {reason}"
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
            f"{declaration} has a block struct type with no ByteAddressBuffer lowering",
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

        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
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
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.unsupported_glsl_buffer_block_access_name(array_expr)
        return None

    def diagnostic_zero_value_for_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type in {"float", "double", "int"}:
            return "0"
        component_count = self.value_component_count(mapped_type)
        component_type = self.vector_component_type(mapped_type)
        if component_count and component_count > 1 and component_type:
            zero_value = "0"
            if component_type in {
                "float",
                "double",
                "half",
                "min16float",
                "min10float",
            }:
                zero_value = "0.0"
            elif component_type in {"uint", "uint16_t", "min16uint"}:
                zero_value = "0u"
            elif component_type == "bool":
                zero_value = "false"
            return f"{mapped_type}({', '.join([zero_value] * component_count)})"
        if (
            mapped_type
            and mapped_type[0].isalpha()
            and any(char.isdigit() for char in mapped_type)
        ):
            return f"{mapped_type}(0)"
        return "0"

    def default_value_expression_for_type(self, field_type):
        mapped_type = self.map_type(field_type)
        if self.hlsl_struct_constructor_fields(mapped_type) is not None:
            return f"({mapped_type})0"
        return None
