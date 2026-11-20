"""Type names in the WGSL generator: WGSL spellings, uniform array wrappers and strides, sizes and alignments, and resource element types."""

import re

from ..ast import (
    ArrayType,
    GenericType,
    LiteralNode,
    MatrixType,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    VectorType,
)
from ..refusals import WGSLRefusal
from ..semantics.structured_layouts import StructuredLayoutUnsupportedError
from .wgsl_narrow_storage import narrow_storage_word_type
from .wgsl_numeric_types import refuse_unsupported_wgsl_numeric_type
from .wgsl_pointer_types import WGSLPointerTypeUnsupportedError


class WGSLTypeNameMixin:
    """Type name members of ``WGSLCodeGen``."""

    def type_name_string(self, vtype, allow_storage_resources=False):
        if vtype is None:
            return "void"
        if isinstance(vtype, PrimitiveType):
            refuse_unsupported_wgsl_numeric_type(vtype.name)
            return self.PRIMITIVE_TYPE_MAP.get(vtype.name.lower(), vtype.name)
        if isinstance(vtype, VectorType):
            return f"vec{vtype.size}<{self.type_name_string(vtype.element_type)}>"
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            self.validate_not_resource_array(vtype)
            element = self.type_name_string(vtype.element_type)
            if vtype.size is None:
                return f"array<{element}>"
            size = (
                self.generate_expression(vtype.size)
                if hasattr(vtype.size, "__class__")
                and not isinstance(vtype.size, (str, int))
                else str(vtype.size)
            )
            return f"array<{element}, {size}>"
        if isinstance(vtype, NamedType):
            atomic_type = self.atomic_type_name(vtype)
            if atomic_type is not None:
                return atomic_type
            if not vtype.generic_args and vtype.name in self._enum_type_names:
                return "i32"
            option_type = self.builtin_option_specialized_type_name(vtype)
            if option_type is not None:
                return option_type
            storage_element = self.structured_buffer_element_type(vtype)
            if storage_element is not None:
                if not allow_storage_resources:
                    raise WGSLRefusal(
                        "WGSL target only supports StructuredBuffer resources as "
                        "module-scope storage bindings"
                    )
                word = narrow_storage_word_type(storage_element)
                return f"array<{self.type_name_string(word or storage_element)}>"
            generic_vector = self.generic_vector_type_name(vtype)
            if generic_vector is not None:
                return generic_vector
            if vtype.generic_args:
                if vtype.name in self._generic_struct_definitions:
                    return self.generic_struct_specialized_type_name(vtype)
                if vtype.name == "ptr":
                    raise WGSLPointerTypeUnsupportedError(self.generic_type_key(vtype))
                raise WGSLRefusal(
                    "WGSL target does not support generic named type "
                    f"{self.generic_type_key(vtype)}; only concrete generic "
                    "struct specializations can be emitted"
                )
            if vtype.name in self._generic_struct_definitions:
                definition = self._generic_struct_definitions[vtype.name]
                expected = len(getattr(definition, "generic_params", []) or [])
                raise WGSLRefusal(
                    "WGSL target cannot use generic struct "
                    f"{vtype.name} without concrete type arguments; expected "
                    f"{expected} argument{'s' if expected != 1 else ''}"
                )
            return self.map_type_name(vtype.name)
        if isinstance(vtype, GenericType):
            raise WGSLRefusal("WGSL target does not support generic types yet")
        if isinstance(vtype, PointerType):
            raise WGSLRefusal(
                "WGSL target only supports pointer types for buffer/storage "
                "resources and helper parameters"
            )
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        if isinstance(vtype, str):
            option_type = self.builtin_option_specialized_type_name(vtype)
            if option_type is not None:
                return option_type
            return self.map_type_name(vtype)
        return str(vtype)

    def uniform_buffer_member_type_name(self, vtype):
        if self.is_bool_type(vtype):
            return "u32"
        wrapper_name = self.uniform_scalar_array_wrapper_name(vtype)
        if wrapper_name is None:
            return self.type_name_string(vtype)
        return self.array_type_name(wrapper_name, vtype)

    def array_type_name(self, element_type_name, vtype):
        if not isinstance(vtype, ArrayType):
            raise WGSLRefusal("WGSL target expected an array type")
        if vtype.size is None:
            return f"array<{element_type_name}>"
        size = (
            self.generate_expression(vtype.size)
            if hasattr(vtype.size, "__class__")
            and not isinstance(vtype.size, (str, int))
            else str(vtype.size)
        )
        return f"array<{element_type_name}, {size}>"

    def uniform_scalar_array_wrapper_name(self, vtype):
        scalar_type = self.uniform_scalar_array_element_type_name(vtype)
        if scalar_type is None:
            return None
        return self._uniform_scalar_array_wrappers.get(scalar_type)

    def uniform_scalar_array_element_type_name(self, vtype):
        if not isinstance(vtype, ArrayType) or vtype.size is None:
            return None
        scalar_type = self.scalar_type_name(vtype.element_type)
        if scalar_type in {"f32", "i32", "u32"}:
            return scalar_type
        return None

    def is_opaque_resource_type(self, vtype):
        leaf = self.array_element_type(vtype)
        res_name = self.resource_type_name(leaf)
        if res_name is not None and self.is_resource_type_name(res_name):
            return True
        if self.is_sampler_type(leaf):
            return True
        if self.sampled_texture_type(leaf) is not None:
            return True
        if self.storage_texture_type(leaf, None) is not None:
            return True
        return False

    def is_unsupported_uniform_array_stride(self, vtype, variable_name=None):
        if not isinstance(vtype, ArrayType):
            return False
        if self.is_opaque_resource_type(vtype):
            return False
        stride = self.uniform_array_element_stride(
            vtype.element_type, variable_name=variable_name
        )
        return stride % 16 != 0

    def wgsl_type_size_and_align(self, vtype):
        """The host-shareable size and alignment of ``vtype`` in bytes, or ``(None, None)`` when unknown.

        WGSL's layout rules: a vector of 3 is aligned as a vector of 4, a matrix is an array of its column vectors,
        an array's stride is its element size rounded up to the element alignment, and a structure is aligned to
        its widest member. ``(None, None)`` makes the caller refuse the declaration.
        """
        scalar_bytes = {
            "bool": (1, 1),
            "i8": (1, 1),
            "u8": (1, 1),
            "f16": (2, 2),
            "i16": (2, 2),
            "u16": (2, 2),
            "f32": (4, 4),
            "i32": (4, 4),
            "u32": (4, 4),
            "f64": (8, 8),
            "i64": (8, 8),
            "u64": (8, 8),
        }
        if isinstance(vtype, ArrayType):
            count = self.resolve_integer_expression(vtype.size)
            element_size, element_align = self.wgsl_type_size_and_align(
                vtype.element_type
            )
            if count is None or element_size is None:
                return None, None
            stride = (
                (element_size + element_align - 1) // element_align
            ) * element_align
            return count * stride, element_align

        type_name = self.type_name_string(vtype) if vtype is not None else None
        if type_name in scalar_bytes:
            return scalar_bytes[type_name]

        if type_name:
            m_vec = re.fullmatch(r"vec([234])<([^>]+)>", type_name)
            if m_vec:
                lanes = int(m_vec.group(1))
                elem_name = m_vec.group(2)
                if elem_name not in scalar_bytes:
                    return None, None
                elem_size, elem_align = scalar_bytes[elem_name]
                if lanes == 2:
                    return elem_size * 2, elem_align * 2
                elif lanes == 3:
                    return elem_size * 3, elem_align * 4
                else:
                    return elem_size * 4, elem_align * 4

            m_mat = re.fullmatch(r"mat([234])x([234])<([^>]+)>", type_name)
            if m_mat:
                cols = int(m_mat.group(1))
                rows = int(m_mat.group(2))
                elem_name = m_mat.group(3)
                if elem_name not in scalar_bytes:
                    return None, None
                elem_size, elem_align = scalar_bytes[elem_name]
                col_align = elem_align * (4 if rows > 2 else 2)
                col_size = elem_size * rows
                col_stride = ((col_size + col_align - 1) // col_align) * col_align
                return cols * col_stride, col_align

        struct_name = type_name or (
            getattr(vtype, "name", None) if isinstance(vtype, NamedType) else None
        )
        struct_node = self._structs_by_name.get(struct_name)
        if struct_node is not None:
            struct_align = 1
            offset = 0
            for member in getattr(struct_node, "members", []) or []:
                m_type = getattr(member, "member_type", None) or getattr(
                    member, "vtype", None
                )
                m_size, m_align = self.wgsl_type_size_and_align(m_type)
                if m_size is None or m_align is None:
                    return None, None
                offset = ((offset + m_align - 1) // m_align) * m_align
                offset += m_size
                struct_align = max(struct_align, m_align)
            struct_size = ((offset + struct_align - 1) // struct_align) * struct_align
            return struct_size, struct_align

        return None, None

    def uniform_array_element_stride(self, element_type, variable_name=None):
        # Recursively return the leaf element stride on purpose: any non-16
        # leaf stride is invalid in uniform space, regardless of outer array size.
        if isinstance(element_type, ArrayType):
            return self.uniform_array_element_stride(
                element_type.element_type, variable_name=variable_name
            )

        if self.is_opaque_resource_type(element_type):
            return None

        size, align = self.wgsl_type_size_and_align(element_type)
        if size is None or align is None:
            type_name = getattr(element_type, "name", None) or str(element_type)
            decl_name = f"'{variable_name}'" if variable_name else "declaration"
            raise StructuredLayoutUnsupportedError(
                f"WGSL target cannot determine uniform array stride for {decl_name} "
                f"with unresolvable element type {type_name!r}"
            )

        return ((size + align - 1) // align) * align

    def generate_uniform_scalar_array_wrappers(self):
        sections = []
        for scalar_type, wrapper_name in sorted(
            self._uniform_scalar_array_wrappers.items()
        ):
            sections.append(
                f"struct {wrapper_name} {{\n"
                f"    @size(16) value: {scalar_type},\n"
                "};"
            )
        return "\n\n".join(sections)

    def map_type_name(self, type_name):
        normalized = type_name.strip()
        lower = normalized.lower()
        if lower in self.PRIMITIVE_TYPE_MAP:
            return self.PRIMITIVE_TYPE_MAP[lower]
        if normalized not in self._type_identifier_names:
            refuse_unsupported_wgsl_numeric_type(normalized)
        if self.is_resource_type_name(lower):
            raise WGSLRefusal(
                "WGSL target does not support CrossGL resource type "
                f"{normalized} yet; split texture/sampler/storage bindings are required"
            )

        wgsl_vector_match = re.fullmatch(r"vec([234])<\s*([^>]+)\s*>", normalized, re.I)
        if wgsl_vector_match:
            size = wgsl_vector_match.group(1)
            element = wgsl_vector_match.group(2)
            return f"vec{size}<{self.map_type_name(element)}>"

        float16_vector_match = self.FLOAT16_VECTOR_TYPE_RE.match(lower)
        if float16_vector_match:
            return f"vec{float16_vector_match.group(1)}<f32>"

        vector_match = self.VECTOR_TYPE_RE.match(lower)
        if vector_match:
            size = vector_match.group(1)
            element = "f32"
            if lower.startswith(("int", "ivec")):
                element = "i32"
            elif lower.startswith(("uint", "uvec")):
                element = "u32"
            elif lower.startswith(("bool", "bvec")):
                element = "bool"
            return f"vec{size}<{element}>"

        matrix_match = self.MATRIX_TYPE_RE.match(lower)
        if matrix_match:
            columns = matrix_match.group(1)
            rows = matrix_match.group(2) or columns
            return f"mat{columns}x{rows}<f32>"

        return self.type_identifier_name(normalized)

    def generic_vector_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        if str(vtype.name).lower() not in {"vec", "vector"}:
            return None
        if len(vtype.generic_args) < 2:
            return None
        size = self.literal_integer_value(vtype.generic_args[1])
        if size not in {2, 3, 4}:
            return None
        return f"vec{size}<{self.type_name_string(vtype.generic_args[0])}>"

    def literal_integer_value(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, LiteralNode):
            return self.literal_integer_value(value.value)
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                return None
        return None

    def is_resource_type_name(self, lower_type_name):
        return lower_type_name in self.RESOURCE_TYPE_NAMES

    def structured_buffer_element_type(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.STRUCTURED_BUFFER_TYPE_NAMES:
            return None
        if len(vtype.generic_args) != 1:
            raise WGSLRefusal(
                "WGSL target requires StructuredBuffer resources to declare one "
                "element type"
            )
        return vtype.generic_args[0]

    def structured_buffer_access(self, vtype):
        if self.structured_buffer_element_type(vtype) is None:
            return None
        base_name = str(vtype.name).lower()
        if base_name in self.WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES:
            return "read_write"
        return "read"

    def constant_buffer_element_type(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.CONSTANT_BUFFER_TYPE_NAMES:
            return None
        if len(vtype.generic_args) != 1:
            raise WGSLRefusal(
                "WGSL target requires ConstantBuffer resources to declare one "
                "element type"
            )
        return vtype.generic_args[0]

    def stage_uniform_parameter_type(self, parameter):
        constant_buffer_type = self.constant_buffer_element_type(parameter.param_type)
        if constant_buffer_type is not None:
            return constant_buffer_type

        if not self.has_binding_attribute(parameter):
            return None

        qualifier_names = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "constant" not in qualifier_names:
            return None

        param_type = getattr(parameter, "param_type", None)
        if isinstance(param_type, ReferenceType):
            return param_type.referenced_type
        if isinstance(param_type, PointerType):
            pointee_type = param_type.pointee_type
            if self.struct_type_name(pointee_type) in self._structs_by_name:
                return pointee_type
            return None
        return param_type

    def stage_resource_module_type(self, parameter):
        uniform_type = self.stage_uniform_parameter_type(parameter)
        if uniform_type is not None:
            return uniform_type
        return getattr(parameter, "param_type", None)

    def validate_uniform_binding_type(self, uniform_type):
        struct_name = self.struct_type_name(self.array_element_type(uniform_type))
        struct = self._structs_by_name.get(struct_name)
        if struct is None:
            return
        for member in getattr(struct, "members", []) or []:
            self.validate_cbuffer_member(struct, member)

    def unsupported_storage_buffer_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES:
            return None
        return str(vtype.name)

    def storage_buffer_like_type_name(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if (
            base_name in self.STRUCTURED_BUFFER_TYPE_NAMES
            or base_name in self.UNSUPPORTED_STORAGE_BUFFER_TYPE_NAMES
        ):
            return str(vtype.name)
        return None

    def validate_not_resource_array(self, vtype):
        resource_type = self.resource_array_element_type_name(vtype)
        if resource_type is None:
            return
        raise WGSLRefusal(
            "WGSL target does not support resource arrays of "
            f"{resource_type}; WebGPU/WGSL requires texture, sampler, image, "
            "and storage-buffer resources to be declared as individual "
            "module-scope bindings"
        )

    def resource_array_element_type_name(self, vtype):
        if not isinstance(vtype, ArrayType):
            return None
        element_type = self.array_element_type(vtype)
        resource_type = self.resource_type_name(element_type)
        if resource_type is not None and self.is_resource_type_name(resource_type):
            return self.type_display_name(element_type)
        return self.storage_buffer_like_type_name(element_type)

    def type_display_name(self, vtype):
        if isinstance(vtype, NamedType):
            return str(vtype.name)
        if isinstance(vtype, str):
            return vtype.strip()
        return str(vtype)
