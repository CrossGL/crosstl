"""GLSL buffer blocks in the Slang generator: block discovery, scalar layout members and the Slang resource types that replace them."""

from ..ast import ArrayNode
from ..semantics.shader_records import is_shader_record_buffer
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .glsl_buffer_layout import (
    align_to,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_block_node_type,
    std430_layout_type_name,
)
from .hlsl_byteaddress_typed_access import hlsl_byteaddress_component_refusal


class SlangBufferBlockTypesMixin:
    """GLSL buffer-block type members of ``SlangCodeGen``."""

    def collect_slang_glsl_buffer_blocks(self, node):
        declarations = self.collect_slang_resource_declaration_nodes(node)
        (
            self.lowered_glsl_buffer_blocks,
            self.glsl_buffer_block_lowering_failures,
            self.glsl_buffer_block_struct_lowering_failures,
        ) = collect_lowered_glsl_buffer_blocks(
            declarations,
            structs_by_name=self.structs_by_name,
            is_glsl_buffer_block_variable=self.is_glsl_buffer_block_variable,
            resource_base_type=self.resource_base_type,
            glsl_buffer_block_layout=self.glsl_buffer_block_layout,
            convert_type_node_to_string=self.convert_type_node_to_string,
            literal_int_value=lambda expr: evaluate_literal_int_expression(
                expr, self.literal_int_constants
            ),
            map_type=self.map_type,
            target_type_key="slang_type",
            unsupported_type_message=(
                "type is not supported by Slang ByteAddressBuffer lowering"
            ),
            component_refusal=lambda info: hlsl_byteaddress_component_refusal(
                self, info
            ),
        )
        scalar_blocks, scalar_failures, scalar_struct_failures = (
            self.collect_scalar_glsl_buffer_blocks(declarations)
        )
        self.lowered_glsl_buffer_blocks.update(scalar_blocks)
        self.glsl_buffer_block_lowering_failures.update(scalar_failures)
        self.glsl_buffer_block_struct_lowering_failures.update(scalar_struct_failures)
        self.decorate_slang_lowered_glsl_buffer_blocks(declarations)
        self.glsl_buffer_block_struct_names = {
            block["type_name"]
            for block in self.lowered_glsl_buffer_blocks.values()
            if block.get("type_name")
        }

    def decorate_slang_lowered_glsl_buffer_blocks(self, declarations):
        for node in declarations:
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            block = self.lowered_glsl_buffer_blocks.get(var_name)
            if block is None:
                continue
            access = self.explicit_resource_access(node)
            if access is None:
                access = "readonly" if block.get("readonly") else "readwrite"
            block["access"] = access
            block["readonly"] = access == "readonly"
            block["writeonly"] = access == "writeonly"

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

    def collect_scalar_glsl_buffer_blocks(self, declarations):
        blocks = {}
        var_failures = {}
        struct_failures = {}
        for node in declarations:
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            node_type = glsl_buffer_block_node_type(node)
            if (
                not var_name
                or str(self.glsl_buffer_block_layout(node)).lower() != "scalar"
                or not self.is_glsl_buffer_block_variable(node, node_type)
            ):
                continue

            type_name = str(self.resource_base_type(node_type))
            struct = self.structs_by_name.get(type_name)
            if struct is None:
                continue

            members, failure_reason = self.scalar_glsl_buffer_struct_members(struct)
            if not members:
                if failure_reason:
                    var_failures[var_name] = failure_reason
                    struct_failures.setdefault(type_name, failure_reason)
                continue

            runtime_array = next(
                (
                    name
                    for name, member in members.items()
                    if member.get("runtime_array")
                ),
                None,
            )
            blocks[var_name] = {
                "type_name": type_name,
                "layout": "scalar",
                "readonly": False,
                "members": members,
                "runtime_array": runtime_array,
            }
        return blocks, var_failures, struct_failures

    def scalar_glsl_buffer_struct_members(self, struct):
        offset = 0
        members = {}
        struct_members = getattr(struct, "members", []) or []
        for index, member in enumerate(struct_members):
            member_name = getattr(member, "name", None)
            member_info = self.scalar_glsl_buffer_member_info(member)
            if member_info is None:
                return {}, (
                    f"unsupported member {member_name or '<unnamed>'}: "
                    "type is not supported by Slang scalar ByteAddressBuffer lowering"
                )
            if not member_name:
                return {}, "unsupported unnamed buffer block member"

            if member_info["is_array"]:
                offset = align_to(offset, member_info["align"])
                stride = align_to(member_info["size"], member_info["align"])
                if member_info["array_size"] is None:
                    if index != len(struct_members) - 1:
                        return {}, (
                            f"unsupported member {member_name}: runtime arrays "
                            "must be the final buffer block member"
                        )
                    members[member_name] = {
                        **member_info,
                        "offset": offset,
                        "stride": stride,
                        "runtime_array": True,
                    }
                    continue

                array_count = evaluate_literal_int_expression(
                    member_info["array_size"], self.literal_int_constants
                )
                if array_count is None:
                    return {}, (
                        f"unsupported member {member_name}: fixed array size "
                        "must be a literal integer"
                    )
                members[member_name] = {
                    **member_info,
                    "offset": offset,
                    "stride": stride,
                    "array_count": array_count,
                    "runtime_array": False,
                }
                offset += stride * array_count
                continue

            offset = align_to(offset, member_info["align"])
            members[member_name] = {
                **member_info,
                "offset": offset,
                "runtime_array": False,
            }
            offset += member_info["size"]

        return members, None

    def scalar_glsl_buffer_member_info(self, member, type_stack=()):
        member_type = getattr(member, "member_type", None)
        is_array = False
        array_size = None
        if member_type is not None:
            if str(type(member_type)).find("ArrayType") != -1:
                is_array = True
                array_size = member_type.size
                member_type = member_type.element_type
            type_name = self.convert_type_node_to_string(member_type)
        elif isinstance(member, ArrayNode):
            is_array = True
            array_size = member.size
            type_name = getattr(member, "element_type", getattr(member, "vtype", None))
        elif hasattr(member, "vtype"):
            type_name = member.vtype
        else:
            return None

        type_name = str(type_name)
        layout_type_name = std430_layout_type_name(type_name)
        if layout_type_name is None:
            # Not a value type: a structure is laid out, and named, as declared.
            layout_type_name = type_name
            type_info = self.scalar_glsl_buffer_struct_type_info(type_name, type_stack)
        else:
            type_info = self.scalar_glsl_buffer_type_info(layout_type_name, type_stack)
        if type_info is None:
            return None
        return {
            "type": type_name,
            "layout_type": layout_type_name,
            **type_info,
            "slang_type": self.map_type(type_name),
            "is_array": is_array,
            "array_size": array_size,
        }

    def scalar_glsl_buffer_type_info(self, type_name, type_stack=()):
        scalar_types = {
            "bool": "bool",
            "float": "float",
            "int": "int",
            "uint": "uint",
        }
        if type_name in scalar_types:
            return {
                "size": 4,
                "align": 4,
                "components": 1,
                "component_type": scalar_types[type_name],
            }

        for prefix, component_type in (
            ("bvec", "bool"),
            ("vec", "float"),
            ("ivec", "int"),
            ("uvec", "uint"),
        ):
            if type_name.startswith(prefix):
                suffix = type_name[len(prefix) :]
                if suffix in {"2", "3", "4"}:
                    components = int(suffix)
                    return {
                        "size": components * 4,
                        "align": 4,
                        "components": components,
                        "component_type": component_type,
                    }

        matrix_info = self.scalar_glsl_buffer_matrix_type_info(type_name)
        if matrix_info is not None:
            return matrix_info

        return self.scalar_glsl_buffer_struct_type_info(type_name, type_stack)

    def scalar_glsl_buffer_matrix_type_info(self, type_name):
        for columns in range(2, 5):
            for rows in range(2, 5):
                names = {f"mat{columns}x{rows}", f"float{rows}x{columns}"}
                if columns == rows:
                    names.add(f"mat{columns}")
                if type_name not in names:
                    continue
                column_stride = rows * 4
                return {
                    "size": columns * column_stride,
                    "align": 4,
                    "matrix_columns": columns,
                    "matrix_rows": rows,
                    "column_stride": column_stride,
                    "component_type": "float",
                }
        return None

    def scalar_glsl_buffer_struct_type_info(self, type_name, type_stack):
        if type_name in type_stack:
            return None
        struct = self.structs_by_name.get(type_name)
        if struct is None:
            return None

        members, failure_reason = self.scalar_glsl_buffer_struct_members_with_layout(
            struct, (*type_stack, type_name)
        )
        if failure_reason or not members:
            return None

        offset = 0
        max_align = 0
        for member in members.values():
            offset = max(offset, member["offset"])
            if member.get("runtime_array"):
                return None
            if member.get("is_array"):
                offset += member["stride"] * member["array_count"]
            else:
                offset += member["size"]
            max_align = max(max_align, member["align"])
        if max_align == 0:
            return None
        return {
            "size": align_to(offset, max_align),
            "align": max_align,
            "members": members,
            "is_struct": True,
        }

    def scalar_glsl_buffer_struct_members_with_layout(self, struct, type_stack):
        offset = 0
        members = {}
        struct_members = getattr(struct, "members", []) or []
        for member in struct_members:
            member_name = getattr(member, "name", None)
            member_info = self.scalar_glsl_buffer_member_info(member, type_stack)
            if member_info is None or not member_name:
                return {}, "unsupported nested scalar buffer block member"
            offset = align_to(offset, member_info["align"])
            if member_info["is_array"]:
                if member_info["array_size"] is None:
                    return {}, "unsupported nested runtime array member"
                array_count = evaluate_literal_int_expression(
                    member_info["array_size"], self.literal_int_constants
                )
                if array_count is None:
                    return {}, "unsupported nested nonliteral array size"
                stride = align_to(member_info["size"], member_info["align"])
                members[member_name] = {
                    **member_info,
                    "offset": offset,
                    "stride": stride,
                    "array_count": array_count,
                    "runtime_array": False,
                }
                offset += stride * array_count
                continue
            members[member_name] = {
                **member_info,
                "offset": offset,
                "runtime_array": False,
            }
            offset += member_info["size"]
        return members, None

    def is_glsl_buffer_block_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        return bool(attr_name and str(attr_name).lower() == "glsl_buffer_block")

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        # A shader record is a constant buffer of its struct (:mod:`.slang_shader_records`), not a lowered block.
        if self.glsl_buffer_block_attribute(node) is None or is_shader_record_buffer(
            node
        ):
            return False
        type_name = self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, declarations):
        names = set()
        for node in declarations:
            node_type = glsl_buffer_block_node_type(node)
            if self.is_glsl_buffer_block_variable(node, node_type):
                names.add(str(self.resource_base_type(node_type)))
        return names

    def is_glsl_buffer_array_declaration(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" not in qualifiers:
            return False
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        if not attributes.intersection({"std140", "std430", "scalar"}):
            return False
        node_type = glsl_buffer_block_node_type(node)
        if getattr(node_type, "__class__", None).__name__ == "ArrayType":
            return True
        type_name = self.type_name_string(node_type)
        return isinstance(type_name, str) and type_name.endswith("[]")

    def glsl_buffer_array_element_type(self, node):
        node_type = glsl_buffer_block_node_type(node)
        if getattr(node_type, "__class__", None).__name__ == "ArrayType":
            return self.convert_type_node_to_string(node_type.element_type)
        type_name = self.type_name_string(node_type)
        if not isinstance(type_name, str) or not type_name.endswith("[]"):
            return None
        return type_name[:-2]

    def slang_glsl_buffer_array_resource_type(self, node):
        if not self.is_glsl_buffer_array_declaration(node):
            return None
        element_type = self.glsl_buffer_array_element_type(node)
        if not element_type:
            return None
        access = self.explicit_resource_access(node) or "readwrite"
        resource_type = (
            "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
        )
        return f"{resource_type}<{self.convert_type(element_type)}>"

    def slang_glsl_buffer_block_resource_type(self, node):
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        block = self.lowered_glsl_buffer_blocks.get(var_name)
        if block is None:
            return None
        resource_type = (
            "ByteAddressBuffer" if block.get("readonly") else "RWByteAddressBuffer"
        )
        _base_type, array_suffix = split_array_type_suffix(
            self.type_name_string(glsl_buffer_block_node_type(node))
        )
        return f"{resource_type}{array_suffix or ''}"

    def slang_storage_buffer_resource_type(self, node, vtype=None):
        """Return the structured-buffer type a CrossGL storage buffer declares.

        ``buffer T* name`` names bound storage, so it becomes ``RWStructuredBuffer<T>``
        (``StructuredBuffer<T>`` when it is declared ``readonly``). Slang has no pointer to
        bound memory: a global pointer is an implicit shader parameter, one
        ``PhysicalStorageBuffer`` address per buffer, which a host cannot bind as a
        storage buffer on any slangc target (#2921).
        """
        if node is None:
            return None

        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        if "buffer" not in qualifiers:
            return None
        if self.glsl_buffer_block_attribute(node) is not None:
            return None

        if vtype is None:
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if vtype is None:
                vtype = getattr(node, "param_type", None)
        if vtype is None:
            return None

        element_type = self.slang_storage_buffer_element_type(vtype)
        if element_type is None:
            return None

        resource_type = (
            "StructuredBuffer"
            if self.explicit_resource_access(node) == "readonly"
            else "RWStructuredBuffer"
        )
        return f"{resource_type}<{element_type}>"

    def slang_storage_buffer_element_type(self, vtype):
        """Return the element type ``buffer T* name`` declares, or ``None``.

        ``buffer BlockName instance`` (an interface block instance, whose members are read directly) is not this
        form: it keeps the emitter's block lowering.
        """
        element_type = self.pointer_pointee_type_name(vtype)
        if element_type is None:
            return None

        element_type = str(element_type).strip()
        if element_type.startswith("atomic<") and element_type.endswith(">"):
            element_type = element_type[len("atomic<") : -1].strip()
        return self.convert_type(element_type) if element_type else None

    def slang_resource_declaration_type(self, node):
        return (
            self.slang_glsl_buffer_array_resource_type(node)
            or self.slang_glsl_buffer_block_resource_type(node)
            or self.slang_storage_buffer_resource_type(node)
            or self.variable_declaration_type(node)
        )
