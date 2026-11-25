"""Resource types, binding indices, register spaces and binding reservation in the Mojo generator."""

from ..refusals import MojoRefusal
from .image_access_contracts import (
    explicit_image_format,
    image_format_channel_count,
    image_format_component_kind,
)
from .mojo_exact_lookups import mojo_resource_kind_of_spelling
from .mojo_tables import (
    MOJO_BUFFER_RESOURCE_TYPES,
    MOJO_BYTE_ADDRESS_BUFFER_TYPES,
    MOJO_HLSL_BUFFER_TYPE_ALIASES,
    MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES,
    MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING,
    MOJO_IMAGE_RESOURCE_BASE_TYPES,
    MOJO_RESERVED_RESOURCE_TYPE_NAMES,
    MOJO_RESOURCE_ACCESS_QUALIFIERS,
    MOJO_RESOURCE_COMPARE_COORDS,
    MOJO_RESOURCE_MEMORY_QUALIFIERS,
    MOJO_RESOURCE_OFFSET_COORDS,
    MOJO_RESOURCE_PROJECTED_COORDS,
    MOJO_RESOURCE_SAMPLE_COORDS,
    MOJO_RESOURCE_SIZE_RETURNS,
    MOJO_RESOURCE_TEXEL_COORDS,
    MOJO_RESOURCE_TYPE_MAPPING,
    MOJO_SCALAR_DTYPES,
    MOJO_TEXTURE_RESOURCE_BASE_TYPES,
    MOJO_TYPED_BUFFER_RESOURCE_TYPES,
    MOJO_TYPED_IMAGE_DTYPE_SUFFIX,
    MOJO_TYPED_IMAGE_SUFFIX_INFO,
    MOJO_TYPED_TEXTURE_DTYPE_SUFFIX,
    MOJO_TYPED_TEXTURE_SUFFIX_INFO,
)


class MojoResourceTypeMixin:
    """Resource-type members of ``MojoCodeGen``."""

    def buffer_resource_info(self, type_name):
        generic = self.parse_generic_type_name(self.type_name(type_name))
        if generic is None:
            type_name = self.canonical_buffer_resource_type(type_name)
            if type_name in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
                return type_name, None
            return None

        base_type, generic_args = generic
        base_type = self.canonical_buffer_resource_type(base_type)
        if base_type not in MOJO_TYPED_BUFFER_RESOURCE_TYPES or not generic_args:
            return None
        return base_type, generic_args[0]

    def canonical_buffer_resource_type(self, base_type):
        base_name = self.type_name(base_type)
        if base_name in MOJO_HLSL_BUFFER_TYPE_ALIASES:
            return MOJO_HLSL_BUFFER_TYPE_ALIASES[base_name]
        # Already canonical, or not a buffer; callers check membership in MOJO_TYPED_BUFFER_RESOURCE_TYPES.
        return base_name

    def resource_type_alias(self, type_name):
        type_name = self.type_name(type_name)
        rw_alias = self.writable_texture_resource_alias(type_name)
        if rw_alias is not None:
            return rw_alias

        mapped_type = MOJO_RESOURCE_TYPE_MAPPING.get(type_name)
        if mapped_type is not None:
            return mapped_type

        generic = self.parse_generic_type_name(type_name)
        if generic is None:
            return None

        base_type, generic_args = generic
        rw_alias = self.writable_texture_resource_alias(
            base_type, generic_args[0] if generic_args else None
        )
        if rw_alias is not None:
            return rw_alias

        mapped_base_type = MOJO_RESOURCE_TYPE_MAPPING.get(base_type)
        if mapped_base_type is None and self.is_mojo_resource_type(base_type):
            mapped_base_type = base_type
        if mapped_base_type is not None and self.is_texture_resource_type(
            mapped_base_type
        ):
            if generic_args:
                element_info = self.resource_element_value_info(generic_args[0])
                if element_info is not None:
                    dtype, source_width, _ = element_info
                    typed_texture = self.typed_texture_resource_type(
                        mapped_base_type, dtype, source_width
                    )
                    if typed_texture != mapped_base_type:
                        return typed_texture
            return mapped_base_type
        return None

    def writable_texture_resource_alias(self, base_type, element_type=None):
        base_name = self.type_name(base_type)
        if base_name in MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES:
            raise MojoRefusal(
                "Unsupported writable texture resource for Mojo codegen; "
                f"multisample writable texture alias is not supported: {base_name}"
            )

        image_aliases = MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING.get(base_name)
        if image_aliases is None:
            return None

        element_info = self.resource_element_value_info(element_type)
        dtype = element_info[0] if element_info is not None else None
        if dtype in {"DType.uint16", "DType.uint32"}:
            image_alias = image_aliases[2]
        elif dtype in {"DType.int16", "DType.int32"}:
            image_alias = image_aliases[1]
        else:
            image_alias = image_aliases[0]

        if element_info is None:
            return image_alias

        _, source_width, _ = element_info
        if source_width == 1 and dtype in {"DType.int32", "DType.uint32"}:
            return image_alias
        return self.typed_image_resource_type(image_alias, dtype, source_width)

    def resource_element_dtype(self, element_type):
        element_info = self.resource_element_value_info(element_type)
        if element_info is not None:
            return element_info[0]
        return None

    def resource_element_value_info(self, element_type):
        if element_type is None:
            return None

        type_name = self.normalize_generic_vector_type_name(
            self.type_name(element_type)
        )
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            dtype, source_width, _, _ = vector_info
            return dtype, source_width, self.map_type(type_name)

        dtype = MOJO_SCALAR_DTYPES.get(type_name)
        if dtype is None:
            return None
        return dtype, 1, self.map_type(type_name)

    def typed_image_resource_type(self, base_image_type, dtype, source_width):
        suffix = MOJO_TYPED_IMAGE_DTYPE_SUFFIX.get(dtype)
        if suffix is None:
            return base_image_type
        if source_width != 1:
            suffix = f"{suffix}{source_width}"
        return f"{base_image_type}{suffix}"

    def typed_texture_resource_type(self, base_texture_type, dtype, source_width):
        suffix = MOJO_TYPED_TEXTURE_DTYPE_SUFFIX.get(dtype)
        if suffix is None:
            return base_texture_type
        if source_width != 4:
            suffix = f"{suffix}{source_width}"
        return f"{base_texture_type}{suffix}"

    def typed_image_resource_info(self, resource_type):
        if not isinstance(resource_type, str):
            return None
        for base_image_type in MOJO_IMAGE_RESOURCE_BASE_TYPES:
            if not resource_type.startswith(base_image_type):
                continue
            suffix = resource_type[len(base_image_type) :]
            if not suffix:
                return None
            suffix_info = MOJO_TYPED_IMAGE_SUFFIX_INFO.get(suffix)
            if suffix_info is not None:
                dtype, source_width = suffix_info
                return base_image_type, dtype, source_width
        return None

    def typed_texture_resource_info(self, resource_type):
        if not isinstance(resource_type, str):
            return None
        for base_texture_type in MOJO_TEXTURE_RESOURCE_BASE_TYPES:
            if not resource_type.startswith(base_texture_type):
                continue
            suffix = resource_type[len(base_texture_type) :]
            if not suffix:
                return None
            suffix_info = MOJO_TYPED_TEXTURE_SUFFIX_INFO.get(suffix)
            if suffix_info is not None:
                dtype, source_width = suffix_info
                return base_texture_type, dtype, source_width
        return None

    def image_resource_base_type(self, resource_type):
        typed_info = self.typed_image_resource_info(resource_type)
        if typed_info is not None:
            return typed_info[0]
        return self.texture_resource_base_type(resource_type)

    def texture_resource_base_type(self, resource_type):
        typed_info = self.typed_texture_resource_info(resource_type)
        if typed_info is not None:
            return typed_info[0]
        return resource_type

    def explicit_image_format_resource_type(self, type_name, node):
        image_format = explicit_image_format(node, self.attribute_value_to_string)
        if image_format is None:
            return type_name

        type_name = self.type_name(type_name)
        if self.is_array_type_name(type_name):
            element_type, size = self.parse_array_type_name(type_name)
            formatted_element_type = self.explicit_image_format_resource_type(
                element_type, node
            )
            return self.array_type_name(formatted_element_type, size)

        mapped_type = self.resource_type_alias(type_name) or self.type_mapping.get(
            type_name
        )
        if mapped_type is None and self.is_mojo_resource_type(type_name):
            mapped_type = type_name
        if mapped_type is None or not self.is_image_resource_type(mapped_type):
            return type_name

        component_kind = image_format_component_kind(image_format)
        dtype = {
            "float": "DType.float32",
            "int": "DType.int32",
            "uint": "DType.uint32",
        }.get(component_kind)
        if dtype is None:
            return type_name

        channel_count = image_format_channel_count(image_format)
        if channel_count is None:
            return type_name

        source_width = 4 if channel_count == 1 else channel_count
        return self.typed_image_resource_type(mapped_type, dtype, source_width)

    def resource_texel_coord_type(self, resource_type):
        return MOJO_RESOURCE_TEXEL_COORDS.get(
            self.image_resource_base_type(resource_type)
        )

    def resource_size_return_type(self, resource_type):
        return MOJO_RESOURCE_SIZE_RETURNS.get(
            self.image_resource_base_type(resource_type)
        )

    def resource_sample_coord_type(self, resource_type):
        return MOJO_RESOURCE_SAMPLE_COORDS.get(
            self.texture_resource_base_type(resource_type)
        )

    def resource_offset_coord_type(self, resource_type):
        return MOJO_RESOURCE_OFFSET_COORDS.get(
            self.texture_resource_base_type(resource_type)
        )

    def resource_compare_coord_type(self, resource_type):
        return MOJO_RESOURCE_COMPARE_COORDS.get(
            self.texture_resource_base_type(resource_type)
        )

    def resource_projected_coord_type(self, resource_type):
        return MOJO_RESOURCE_PROJECTED_COORDS.get(
            self.texture_resource_base_type(resource_type)
        )

    def mapped_buffer_type(self, buffer_type, element_type=None):
        buffer_type = self.canonical_buffer_resource_type(buffer_type)
        if buffer_type in MOJO_TYPED_BUFFER_RESOURCE_TYPES and element_type is not None:
            return f"{buffer_type}[{self.map_type(element_type)}]"
        return buffer_type

    def register_buffer_resource_type(self, buffer_type):
        """Refuse an HLSL buffer object the generator would spell as a placeholder type.

        A structured or byte-address buffer declared outside a function is a pointer argument of the kernel
        (mojo_buffer_objects) and never reaches here; the placeholder types and helpers the generator wrote for
        every other buffer object did not compile.
        """
        buffer_type = self.canonical_buffer_resource_type(buffer_type)
        if buffer_type in MOJO_BUFFER_RESOURCE_TYPES:
            self.refuse_mojo_buffer_object_type(buffer_type)

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        return str(value)

    def binding_index_value(self, value, prefixes=()):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def register_space_index_value(self, value):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            if (
                attr_name in {"binding", "buffer", "sampler", "texture", "uav"}
                or attr_name == "register"
            ):
                binding = self.binding_index_value(arguments[0], ("b", "s", "t", "u"))
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name in {"set", "group"} and arguments:
                set_index = self.binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.register_space_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.register_space_index_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_value_to_string(argument)
                for argument in getattr(attr, "arguments", []) or []
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def normalized_resource_access_metadata(self, node):
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = MOJO_RESOURCE_ACCESS_QUALIFIERS.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    access = MOJO_RESOURCE_ACCESS_QUALIFIERS.get(
                        str(self.attribute_value_to_string(arguments[0])).lower()
                    )
                    if access is not None:
                        return access
            access = MOJO_RESOURCE_ACCESS_QUALIFIERS.get(attr_name)
            if access is not None:
                return access
        return None

    def resource_memory_qualifiers(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
            if str(qualifier).lower() in MOJO_RESOURCE_MEMORY_QUALIFIERS
        }
        qualifiers.update(
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
            if str(getattr(attr, "name", "")).lower() in MOJO_RESOURCE_MEMORY_QUALIFIERS
        )
        return sorted(qualifiers)

    def resource_base_type_and_count(self, type_name):
        type_name = self.type_name(type_name)
        if self.is_array_type_name(type_name):
            base_type, size = self.parse_array_type_name(type_name)
            return base_type, self.resource_binding_count(size)
        return type_name, 1

    def resource_binding_count(self, size):
        if size in (None, ""):
            return 1
        count = self.literal_int_value(size)
        if count is None:
            raise MojoRefusal(
                "Mojo resource array binding count must resolve to a "
                f"compile-time integer, got {size!r}"
            )
        if count < 1:
            raise MojoRefusal(
                "Mojo resource array binding count must be positive, " f"got {count}"
            )
        return count

    def user_defined_type_base(self, type_name):
        base_type, _ = self.resource_base_type_and_count(type_name)
        generic = self.parse_generic_type_name(base_type)
        if generic is not None:
            return generic[0]
        return base_type

    def is_user_defined_type_name(self, type_name):
        base_type = self.user_defined_type_base(type_name)
        return base_type in self.struct_types or base_type in self.enum_types

    def reserved_resource_type_name(self, type_name):
        base_type = self.user_defined_type_base(type_name)
        if (
            base_type in MOJO_RESERVED_RESOURCE_TYPE_NAMES
            or self.typed_image_resource_info(base_type) is not None
        ):
            return base_type
        return None

    def validate_user_defined_type_name(self, type_name, declaration_kind):
        reserved_name = self.reserved_resource_type_name(type_name)
        if reserved_name is None:
            return
        raise MojoRefusal(
            f"Mojo {declaration_kind} name '{type_name}' conflicts with "
            f"reserved resource type '{reserved_name}'"
        )

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def mojo_resource_kind(self, type_name, forced_kind=None, node=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_node(node):
            return "glsl_buffer_block"
        base_type, _ = self.resource_base_type_and_count(type_name)
        if self.is_user_defined_type_name(base_type):
            return None
        if self.buffer_resource_info(base_type) is not None:
            return "buffer"
        mapped_type = self.resource_type_alias(base_type) or self.type_mapping.get(
            str(base_type)
        )
        if mapped_type is None:
            # Not mapped: a Mojo resource spelling already, or not a resource at all.
            mapped_type = str(base_type)
        kind = mojo_resource_kind_of_spelling(mapped_type)
        if kind is not None:
            return kind
        if self.typed_image_resource_info(mapped_type) is not None:
            return "image"
        if self.typed_texture_resource_info(mapped_type) is not None:
            return "texture"
        return None

    def mojo_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def mojo_resource_register_class(self, node):
        """Return the HLSL register class letter from an explicit ``register`` attribute."""
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            raw_value = self.attribute_value_to_string(arguments[0]) or ""
            raw_value = str(raw_value).strip().lower()
            if (
                raw_value
                and raw_value[0] in ("b", "s", "t", "u")
                and raw_value[1:].isdigit()
            ):
                return raw_value[0]
        return None

    def mojo_resource_binding_namespace_with_class(self, kind, node):
        namespace = self.mojo_resource_binding_namespace(kind)
        register_class = self.mojo_resource_register_class(node)
        if register_class is not None:
            return f"{namespace}:{register_class}"
        return namespace

    def resource_binding_family_keys(self, base_namespace, set_index):
        """Yield the unscoped key and all class-scoped keys for a base namespace."""
        yield (base_namespace, set_index)
        for cls in ("b", "s", "t", "u"):
            yield (f"{base_namespace}:{cls}", set_index)

    def next_available_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.mojo_resource_binding_cursors.get(key, 0)
        while self.resource_binding_family_conflicts(
            namespace, set_index, binding, count
        ):
            binding += 1
        self.mojo_resource_binding_cursors[key] = binding + count
        return binding

    def resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _ in self.mojo_used_resource_bindings.get(key, []):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def resource_binding_family_conflicts(
        self, base_namespace, set_index, binding, count
    ):
        for key in self.resource_binding_family_keys(base_namespace, set_index):
            if self.resource_binding_range_conflicts(key, binding, count):
                return True
        return False

    def reserve_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        base_namespace = namespace.split(":", 1)[0] if ":" in namespace else namespace

        # Register-class resources only conflict with the same class, but still
        # conflict with classless resources in the unscoped namespace. Classless
        # resources conflict with every class-scoped range in the same family.
        if ":" in namespace:
            keys_to_check = [(namespace, set_index), (base_namespace, set_index)]
        else:
            keys_to_check = self.resource_binding_family_keys(base_namespace, set_index)

        for check_key in keys_to_check:
            for used_start, used_end, used_name in self.mojo_used_resource_bindings.get(
                check_key, []
            ):
                if binding <= used_end and used_start <= end:
                    raise MojoRefusal(
                        "Conflicting Mojo resource binding for "
                        f"'{name}': {base_namespace} set {set_index} binding "
                        f"{binding}-{end} overlaps '{used_name}' binding "
                        f"{used_start}-{used_end}"
                    )

        self.mojo_used_resource_bindings.setdefault(key, []).append(
            (binding, end, name)
        )
        self.mojo_resource_binding_cursors[key] = max(
            self.mojo_resource_binding_cursors.get(key, 0), end + 1
        )

    def resource_attribute_names(self, node):
        return {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }

    def has_resource_qualifier(self, node, qualifier_name):
        qualifier_name = str(qualifier_name).lower()
        return qualifier_name in {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }

    def uses_source_descriptor_array_binding(self, node, count, resource_kind=None):
        if count <= 1:
            return False
        attr_names = self.resource_attribute_names(node)
        if "binding" not in attr_names or "register" in attr_names:
            return False
        return (
            self.has_resource_qualifier(node, "uniform") or resource_kind == "sampler"
        )

    def explicit_binding_is_target_native(self, node, count, resource_kind=None):
        return not self.uses_source_descriptor_array_binding(node, count, resource_kind)

    def reserve_source_resource_binding(
        self, namespace, set_index, binding, count, name
    ):
        key = (namespace, set_index)
        end = binding + count - 1
        for used_start, used_end, used_name in self.mojo_source_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                raise MojoRefusal(
                    "Conflicting Mojo source resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        self.mojo_source_resource_bindings.setdefault(key, []).append(
            (binding, end, name)
        )
