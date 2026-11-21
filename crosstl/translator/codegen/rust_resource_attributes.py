"""Rust resource types, access, GLSL buffer blocks and explicit binding and set attributes."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import RustRefusal
from .array_utils import prepend_array_dimension
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .rust_exact_lookups import rust_resource_kind_of_spelling
from .rust_gpu_interface import (
    BUILTIN_INPUT_KIND,
    STORAGE_SLICE_KIND,
    UNIFORM_VALUE_KIND,
    WORKGROUP_KIND,
    is_uniform_value,
    storage_slice_parts,
)
from .rust_gpu_spec_constants import SPEC_CONSTANT_KIND, spec_constant_parameter


class RustResourceAttributeMixin:
    """Resource-attribute and binding members of ``RustCodeGen``."""

    def storage_image_type_with_attributes(self, type_name, attributes):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            image_type = self.storage_image_type_with_attributes(base_type, attributes)
            if image_type is None:
                return None
            return self.format_c_array_type(image_type, sizes)

        mapped_type = self.type_mapping.get(type_name)
        if mapped_type is None or not mapped_type.startswith("Image"):
            return None

        format_name = self.storage_image_format_attribute(attributes)
        if format_name is None:
            return None

        element_type = self.storage_image_format_element_type(format_name)
        if element_type is None:
            return None

        runtime_base = mapped_type.split("<", 1)[0]
        return f"{runtime_base}<{element_type}>"

    def resource_type_with_attributes(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        glsl_buffer_type = self.glsl_buffer_array_resource_type(type_name, node)
        if glsl_buffer_type is not None:
            return glsl_buffer_type

        buffer_type = self.buffer_type_with_access(type_name, node)
        image_type = self.storage_image_type_with_attributes(
            buffer_type,
            getattr(node, "attributes", []),
        )
        return image_type or buffer_type

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

    def glsl_buffer_array_resource_type(self, type_name, node):
        if not self.is_glsl_buffer_array_declaration(node):
            return None
        base_type, sizes = self.c_array_type_parts(type_name)
        if not sizes or len(sizes) != 1 or sizes[0] is not None:
            return None
        access = self.explicit_resource_access(node) or "readwrite"
        resource_type = (
            "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
        )
        return f"{resource_type}<{base_type}>"

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
            layout = self.attribute_argument_text(arguments[0])
            if layout:
                return layout
        return "std430"

    def is_glsl_buffer_block_variable(self, node):
        return self.glsl_buffer_block_attribute(node) is not None

    def register_glsl_buffer_block_metadata(self, node):
        if not self.is_glsl_buffer_block_variable(node):
            return
        name = getattr(node, "name", None)
        if not name:
            return
        self.glsl_buffer_block_accesses[name] = (
            self.explicit_resource_access(node) or "readwrite"
        )
        self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)

    def glsl_buffer_block_metadata_comment(self, node):
        if not self.is_glsl_buffer_block_variable(node):
            return ""
        name = getattr(node, "name", "")
        access = self.explicit_resource_access(node) or "readwrite"
        layout = self.glsl_buffer_block_layout(node)
        return (
            f"// CrossGL resource metadata: name={name} "
            f"kind=glsl_buffer_block layout={layout} access={access}\n"
        )

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None)
        if isinstance(expr, str):
            return expr
        if isinstance(expr, MemberAccessNode):
            return self.glsl_buffer_block_root_name(
                getattr(expr, "object_expr", getattr(expr, "object", None))
            )
        if isinstance(expr, ArrayAccessNode):
            return self.glsl_buffer_block_root_name(
                getattr(expr, "array_expr", getattr(expr, "array", None))
            )
        return None

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if not root_name:
            return None
        return self.glsl_buffer_block_accesses.get(root_name)

    def unsupported_glsl_buffer_block_expression(self, operation, reason, result_type):
        return self.refuse_lowering(
            f"GLSL buffer block {operation}: {reason}",
            self.rust_default_value_for_type(result_type),
        )

    def unsupported_glsl_buffer_block_statement(self, operation, reason):
        return self.refuse_lowering(f"GLSL buffer block {operation}: {reason}")

    def rust_default_value_for_type(self, type_name):
        rust_type = self.map_type(type_name)
        scalar = self.rust_scalar_default_literal(rust_type)
        if scalar is not None:
            return scalar
        return "Default::default()"

    def explicit_resource_access(self, node):
        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "") or "").lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                access = access_names.get(
                    str(self.attribute_argument_text(arguments[0])).lower()
                )
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def buffer_type_with_access(self, type_name, node):
        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type, sizes = (
            self.c_array_type_parts(type_name)
            if self.is_array_type_name(type_name)
            else (type_name, [])
        )
        generic_base, generic_args = self.generic_type_parts(base_type)
        mapped_base = None

        if generic_base in {"StructuredBuffer", "RWStructuredBuffer"}:
            mapped_base = (
                "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
            )
        elif generic_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )

        if mapped_base is None:
            return type_name

        if generic_args:
            mapped = f"{mapped_base}<{', '.join(generic_args)}>"
        else:
            mapped = mapped_base
        return self.format_c_array_type(mapped, sizes) if sizes else mapped

    def storage_image_format_attribute(self, attributes):
        for attr in attributes or []:
            name = str(getattr(attr, "name", "") or "")
            if name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_argument_text(arguments[0])
                continue
            if self.storage_image_format_element_type(name) is not None:
                return name
        return None

    def literal_int_value(self, value):
        """Return an integer value for literal integer AST nodes/text."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, UnaryOpNode) and not getattr(value, "is_postfix", False):
            operator = getattr(value, "operator", getattr(value, "op", None))
            if operator not in {"+", "-"}:
                return None
            operand_value = self.literal_int_value(value.operand)
            if operand_value is None:
                return None
            return -operand_value if operator == "-" else operand_value
        if hasattr(value, "value"):
            return self.literal_int_value(value.value)
        if hasattr(value, "name"):
            return None

        text = str(value).strip()
        if not text:
            return None
        text = re.sub(r"[uUlL]+$", "", text)
        try:
            return int(text, 0)
        except ValueError:
            return None

    def attribute_argument_text(self, argument):
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return str(argument)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
        raw_value = self.attribute_argument_text(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def resource_register_space_value(self, value):
        raw_value = self.attribute_argument_text(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_rust_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_rust_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if attr_name in {"set", "group"} and arguments:
                set_index = self.resource_binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.resource_register_space_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.resource_register_space_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def rust_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_argument_text(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def rust_explicit_resource_binding_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return attr_name
        return None

    def rust_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            count = 1
            for size in sizes:
                if size is not None:
                    count *= int(size)
            return base_type, count
        return type_name, 1

    def rust_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_variable(node):
            return "glsl_buffer_block"
        if node is not None and storage_slice_parts(node) is not None:
            return STORAGE_SLICE_KIND
        if node is not None and is_uniform_value(
            node, self.type_name_string(type_name)
        ):
            return UNIFORM_VALUE_KIND

        base_type, _count = self.rust_resource_base_type_and_count(type_name)
        generic_base, _generic_args = self.generic_type_parts(base_type)
        base_name = generic_base.rsplit("::", 1)[-1]
        if base_name in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "Buffer",
            "RwBuffer",
            "AppendBuffer",
            "ConsumeBuffer",
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RwByteAddressBuffer",
        }:
            return "buffer"

        mapped_type = self.type_mapping.get(base_name)
        if mapped_type is None:
            # Already a Rust runtime spelling, or a type that is not a resource.
            return rust_resource_kind_of_spelling(base_name)
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        return rust_resource_kind_of_spelling(mapped_base)

    def rust_resource_binding_namespace(self, kind):
        if kind in {
            "cbuffer",
            "glsl_buffer_block",
            STORAGE_SLICE_KIND,
            UNIFORM_VALUE_KIND,
        }:
            return "buffer"
        return kind

    def rust_resource_binding_validation_count(self, node, resource_kind, count):
        if resource_kind in {STORAGE_SLICE_KIND, UNIFORM_VALUE_KIND}:
            # One buffer or uniform block, whatever its type's shape.
            return 1
        if (
            resource_kind == "sampler"
            and count != 1
            and self.rust_explicit_resource_binding_attribute(node) == "binding"
        ):
            return 1
        return count

    def rust_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.rust_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_rust_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.rust_resource_binding_cursors.get(key, 0)
        while self.rust_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.rust_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_rust_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.rust_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise RustRefusal(
                    "Conflicting Rust resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        ranges.append((binding, end, name))
        self.rust_resource_binding_cursors[key] = max(
            self.rust_resource_binding_cursors.get(key, 0), end + 1
        )

    def rust_resource_binding_info(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.rust_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return None

        cache_key = (id(node), resource_kind)
        cached_info = self.rust_resource_binding_infos.get(cache_key)
        if cached_info is not None:
            return cached_info

        _base_type, count = self.rust_resource_base_type_and_count(type_name)
        validation_count = self.rust_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.rust_resource_binding_namespace(resource_kind)
        set_index = self.explicit_rust_resource_set_index(node)
        binding = self.explicit_rust_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_rust_resource_binding(
                namespace, set_index, validation_count
            )
            binding_source = "automatic"
            self.reserve_rust_resource_binding(
                namespace, set_index, binding, validation_count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_rust_resource_binding(
                namespace, set_index, binding, validation_count, name
            )

        info = {
            "name": name,
            "kind": resource_kind,
            "set_index": set_index,
            "binding": binding,
            "binding_source": binding_source,
            "count": count,
            "register": self.rust_resource_register_metadata(node),
        }
        self.rust_resource_binding_infos[cache_key] = info
        return info

    def rust_resource_metadata_comment(self, node, type_name, kind=None):
        info = self.rust_resource_binding_info(node, type_name, kind=kind)
        if info is None:
            return ""

        parts = [
            "// CrossGL resource metadata:",
            f"name={info['name']}",
            f"kind={info['kind']}",
        ]
        if info["kind"] == "glsl_buffer_block":
            parts.append(f"layout={self.glsl_buffer_block_layout(node)}")
            parts.append(f"access={self.explicit_resource_access(node) or 'readwrite'}")
        parts.extend(
            [
                f"set={info['set_index']}",
                f"binding={info['binding']}",
                f"binding_source={info['binding_source']}",
            ]
        )
        if info["count"] != 1:
            parts.append(f"count={info['count']}")
        if info["register"]:
            parts.append(f"register={info['register']}")
        return " ".join(parts) + "\n"

    def rust_resource_parameter_attribute(self, node, type_name):
        info = self.rust_resource_binding_info(node, type_name)
        if info is None:
            return ""

        spirv_args = []
        if info["kind"] in {"buffer", "glsl_buffer_block", STORAGE_SLICE_KIND}:
            spirv_args.append("storage_buffer")
        elif info["kind"] == UNIFORM_VALUE_KIND:
            spirv_args.append("uniform")
        spirv_args.extend(
            [
                f"descriptor_set = {info['set_index']}",
                f"binding = {info['binding']}",
            ]
        )
        return f"#[spirv({', '.join(spirv_args)})] "

    def generate_rust_resource_parameter(self, resource_info):
        """One entry-point parameter: a bound resource, workgroup memory or a built-in input (rust_gpu_interface)."""
        node = resource_info["node"]
        type_name = resource_info["type_name"]
        name = self.rust_identifier(node.name)
        kind = resource_info.get("kind")
        if kind == BUILTIN_INPUT_KIND:
            return f"#[spirv({resource_info['builtin']})] {name}: {self.map_type(type_name)}"
        if kind == SPEC_CONSTANT_KIND:
            return spec_constant_parameter(name, resource_info)
        if kind == WORKGROUP_KIND:
            return f"#[spirv(workgroup)] {name}: {self.map_type(type_name)}"
        attribute = self.rust_resource_parameter_attribute(node, type_name)
        if kind == UNIFORM_VALUE_KIND:
            return f"{attribute}{name}: &{self.map_type(type_name)}"
        return f"{attribute}{name}: {self.map_function_parameter_type(type_name)}"

    def reserve_explicit_rust_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.rust_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_rust_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.rust_resource_base_type_and_count(type_name)
        validation_count = self.rust_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.rust_resource_binding_namespace(resource_kind)
        set_index = self.explicit_rust_resource_set_index(node)
        self.reserve_rust_resource_binding(
            namespace, set_index, binding, validation_count, name
        )

    def reserve_explicit_rust_resource_bindings(self, ast):
        def reserve_variable(node):
            if isinstance(node, ArrayNode):
                element_type = self.convert_type_node_to_string(
                    getattr(node, "element_type", "float")
                )
                size = self.format_array_size(getattr(node, "size", None))
                type_name = prepend_array_dimension(element_type, size)
            else:
                type_name = self.get_variable_type(node) or "float"
            type_name = self.resource_type_with_attributes(type_name, node)
            self.reserve_explicit_rust_resource_binding(node, type_name)

        for node in getattr(ast, "global_variables", []) or []:
            reserve_variable(node)
        for cbuffer in self.get_cbuffer_nodes(ast):
            self.reserve_explicit_rust_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        for stage in (
            getattr(ast, "stages", {}).values() if hasattr(ast, "stages") else []
        ):
            for node in getattr(stage, "local_variables", []) or []:
                reserve_variable(node)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_rust_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )
