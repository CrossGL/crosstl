"""Resource bindings in the GLSL generator: type mapping, binding formats, image attributes, explicit bindings and precision qualifiers."""

import re

from ..ast import (
    CooperativeMatrixType,
    PointerType,
    ReferenceType,
)
from ..refusals import OpenGLRefusal
from .array_utils import (
    ZeroExtentArrayUnsupportedError,
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .crossgl_vector_names import crossgl_named_vector_spelling
from .enum_utils import (
    generic_enum_specialized_type_name,
    generic_type_parts,
    sanitize_type_name,
    substitute_generic_type_name,
)
from .generic_struct_utils import generic_struct_specialized_type_name
from .glsl_fragment_interlock import INTERLOCK_LAYOUTS
from .image_access_contracts import (
    is_image_format_attribute,
    is_resource_access_attribute,
    supported_image_formats,
)
from .opengl_values import (
    OpenGLCooperativeMatrixError,
    OpenGLPrivatePointerParameterError,
    OpenGLTemplateTypeError,
    _opengl_cooperative_matrix_contract_type,
    _opengl_cooperative_matrix_fragment_contract,
    _opengl_cooperative_matrix_fragment_contract_message,
)
from .resource_arrays import is_private_pointer_parameter
from .written_type_names import (
    GLSL_TYPE_NAMES,
    sized_float_spelling,
    sized_vector_parts,
    written_type_name,
)


class OpenGLResourceBindingMixin:
    """Resource binding members of ``GLSLCodeGen``."""

    def map_type(self, vtype):
        """Map types to GLSL equivalents, handling both strings and TypeNode objects."""
        if vtype is None:
            return "float"

        if isinstance(vtype, CooperativeMatrixType):
            if self.cooperative_matrix_software_lowering:
                return self.glsl_cooperative_matrix_fragment_contract(
                    vtype,
                    source_location=getattr(vtype, "source_location", None),
                )["type_name"]
            type_name = self.convert_type_node_to_string(vtype)
            fragment_contract = _opengl_cooperative_matrix_fragment_contract(vtype)
            fragment_contract_message = (
                _opengl_cooperative_matrix_fragment_contract_message(fragment_contract)
            )
            raise OpenGLCooperativeMatrixError(
                "OpenGL cooperative-matrix lowering is not available for type "
                f"'{type_name}'; substituting an ordinary GLSL matrix would change "
                "distributed matrix semantics; "
                f"{fragment_contract_message}",
                operation="type",
                matrix_type=vtype,
                **fragment_contract,
                reason="unsupported-type",
                source_location=getattr(vtype, "source_location", None),
            )

        if isinstance(vtype, PointerType):
            return f"{self.map_type(vtype.pointee_type)}*"
        if isinstance(vtype, ReferenceType):
            return self.map_type(vtype.referenced_type)

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)
        cooperative_base, cooperative_args = generic_type_parts(vtype_str)
        if (
            cooperative_base == "array"
            and len(cooperative_args) == 2
            and evaluate_literal_int_expression(cooperative_args[1]) == 0
        ):
            raise ZeroExtentArrayUnsupportedError("opengl", vtype_str)
        if (
            cooperative_args
            and cooperative_base.rsplit("::", 1)[-1] == "CooperativeMatrix"
        ):
            matrix_type = _opengl_cooperative_matrix_contract_type(vtype)
            if self.cooperative_matrix_software_lowering:
                return self.glsl_cooperative_matrix_fragment_contract(
                    matrix_type,
                    source_location=getattr(vtype, "source_location", None),
                )["type_name"]
            fragment_contract = _opengl_cooperative_matrix_fragment_contract(
                matrix_type
            )
            fragment_contract_message = (
                _opengl_cooperative_matrix_fragment_contract_message(fragment_contract)
            )
            raise OpenGLCooperativeMatrixError(
                "OpenGL cooperative-matrix lowering is not available for type "
                f"'{vtype_str}'; substituting an ordinary GLSL matrix would change "
                "distributed matrix semantics; "
                f"{fragment_contract_message}",
                operation="type",
                matrix_type=matrix_type,
                **fragment_contract,
                reason="unsupported-type",
                source_location=getattr(vtype, "source_location", None),
            )
        if vtype_str.rstrip().endswith("&"):
            return self.map_type(vtype_str.rstrip()[:-1].rstrip())

        resolved_alias = self.resolve_glsl_type_alias(vtype_str)
        if resolved_alias != vtype_str:
            return self.map_type(resolved_alias)

        substitutions = (
            getattr(self, "current_generic_function_substitutions", {}) or {}
        )
        if substitutions:
            substituted_type = substitute_generic_type_name(vtype_str, substitutions)
            if substituted_type != vtype_str:
                return self.map_type(substituted_type)

        named_vector = crossgl_named_vector_spelling(vtype_str)
        if named_vector != vtype_str:
            return self.map_type(named_vector)

        vector_base, vector_args = generic_type_parts(vtype_str)
        if (
            len(vector_args) == 2
            and vector_base.rsplit("::", 1)[-1] == "vec"
            and vector_args[1].strip() in {"2", "3", "4"}
        ):
            component_type = self.map_type(vector_args[0])
            vector_type = self.glsl_vector_type_for_component(
                component_type, int(vector_args[1])
            )
            if vector_type is not None:
                return vector_type

        if self.is_ray_query_type_name(vtype_str):
            return "rayQueryEXT"

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_type(base_type)
            return f"{base_mapped}{array_suffix}"

        canonical_type = self.canonical_resource_type(vtype_str)
        if canonical_type is not None:
            return canonical_type

        unresolved_template_type = self.unresolved_template_placeholder_type(vtype_str)
        if unresolved_template_type is not None:
            raise OpenGLTemplateTypeError(
                "OpenGL codegen cannot emit unresolved template type "
                f"'{unresolved_template_type}' from '{vtype_str}'; "
                "provide a concrete instantiation before GLSL generation",
                unresolved_parameter=unresolved_template_type,
                enclosing_type=vtype_str,
            )

        generic_enum_type = generic_enum_specialized_type_name(self, vtype_str)
        if generic_enum_type is not None:
            return self.glsl_type_identifier_name(generic_enum_type)

        generic_struct_type = generic_struct_specialized_type_name(self, vtype_str)
        if generic_struct_type is not None:
            return self.glsl_type_identifier_name(generic_struct_type)

        materialized_generic_type = self.materialized_generic_type_name(vtype_str)
        if materialized_generic_type is not None:
            return self.glsl_type_identifier_name(materialized_generic_type)

        if vtype_str in getattr(self, "enum_type_names", set()):
            return "int"

        if vtype_str in getattr(self, "enum_struct_type_names", set()):
            return self.glsl_type_identifier_name(vtype_str)

        emulated_matrix_type = self.glsl_emulated_matrix_mapped_type(vtype_str)
        if emulated_matrix_type is not None:
            return emulated_matrix_type

        mapped_type = self.type_mapping.get(vtype_str)
        if (
            mapped_type in (None, vtype_str)
            and vtype_str in self.glsl_type_identifier_names
        ):
            return self.glsl_type_identifier_name(vtype_str)
        if mapped_type is None:
            if sized_float_spelling(vtype_str) != vtype_str:
                return self.map_type(sized_float_spelling(vtype_str))
            sized_vector = sized_vector_parts(vtype_str)
            if sized_vector is not None:
                return self.map_type(f"vec<{sized_vector[0]}, {sized_vector[1]}>")
            return written_type_name(
                vtype_str,
                GLSL_TYPE_NAMES,
                self.program_type_names,
                self.written_foreign_type_names,
            )
        return mapped_type

    def resolve_glsl_type_alias(self, type_name):
        aliases = getattr(self, "current_type_aliases", {})
        resolved = type_name
        seen = []
        while resolved in aliases:
            if resolved in seen:
                cycle = " -> ".join([*seen, resolved])
                raise OpenGLRefusal(f"Cyclic OpenGL type alias: {cycle}")
            seen.append(resolved)
            resolved = self.type_name_string(aliases[resolved])
        return resolved

    def materialized_generic_type_name(self, type_name):
        base_type, array_suffix = split_array_type_suffix(str(type_name or "").strip())
        if array_suffix:
            return None
        base_name, generic_args = generic_type_parts(base_type)
        if not base_name or not generic_args:
            return None
        suffix_parts = [sanitize_type_name(arg) for arg in generic_args]
        suffix_parts = [part for part in suffix_parts if part]
        if not suffix_parts:
            return None
        suffix = "_".join(suffix_parts)
        if base_name.endswith(f"_{suffix}"):
            return base_name
        return None

    def unresolved_template_placeholder_type(self, type_name):
        if not getattr(self, "enforce_concrete_glsl_types", False):
            return None

        type_text = str(type_name or "").strip()
        if not type_text:
            return None
        base_type, _array_suffix = split_array_type_suffix(type_text)
        base_type = base_type.strip().rstrip("*&").strip()
        base_name, generic_args = generic_type_parts(base_type)
        if generic_args:
            for generic_arg in generic_args:
                unresolved = self.unresolved_template_placeholder_type(generic_arg)
                if unresolved is not None:
                    return unresolved
            return None

        if self.is_known_glsl_type_name(base_name):
            return None
        if base_name in getattr(self, "generic_type_parameter_names", set()):
            return base_name
        if self.looks_like_template_placeholder_type(base_name):
            return base_name
        return None

    def is_known_glsl_type_name(self, type_name):
        if not type_name:
            return True
        if type_name in self.type_mapping or type_name in self.type_mapping.values():
            return True
        if self.canonical_resource_type(type_name) is not None:
            return True
        if type_name in getattr(self, "structs_by_name", {}):
            return True
        if type_name in getattr(self, "enum_type_names", set()):
            return True
        if type_name in getattr(self, "enum_struct_type_names", set()):
            return True
        if type_name in {
            "void",
            "bool",
            "int",
            "uint",
            "float",
            "double",
            "int64_t",
            "uint64_t",
            "sampler",
            "atomic_uint",
            "atomic_int",
            "rayQueryEXT",
        }:
            return True
        return False

    def looks_like_template_placeholder_type(self, type_name):
        if not isinstance(type_name, str) or not type_name:
            return False
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", type_name):
            return False
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,2}", type_name):
            return True
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*T", type_name):
            return True
        if re.fullmatch(r"[A-Z][A-Z0-9_]*_T", type_name):
            return True
        return False

    def is_ray_query_type(self, vtype):
        if vtype is None:
            return False
        return self.is_ray_query_type_name(self.type_name_string(vtype))

    def is_acceleration_structure_type(self, vtype):
        if vtype is None:
            return False
        return self.resource_base_type(vtype) == "accelerationStructureEXT"

    def is_ray_query_type_name(self, type_name):
        type_name = str(type_name)
        return (
            type_name == "rayQueryEXT"
            or type_name == "rayQuery"
            or type_name == "RayQuery"
            or type_name.startswith("RayQuery<")
        )

    def map_resource_parameter_type_with_hint(
        self, vtype, node=None, function_name=None
    ):
        if vtype is None:
            return self.map_type(vtype)
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" in qualifiers and isinstance(vtype, PointerType):
            return f"{self.map_type(vtype.pointee_type)}[]"
        if self.is_structured_buffer_type(vtype):
            return f"{self.structured_buffer_element_type(vtype)}[]"

        function_hints = self.function_resource_array_size_hints.get(function_name, {})
        param_name = getattr(node, "name", None)

        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if self.is_inferable_resource_array_type(base_type):
                array_size = (
                    self.expression_to_string(vtype.size)
                    if vtype.size is not None
                    else function_hints.get(param_name, "")
                )
                mapped_type = self.map_image_base_type_with_format(base_type, node)
                return f"{mapped_type}[{array_size or 1}]"

        if not (hasattr(vtype, "name") or hasattr(vtype, "element_type")):
            type_string = str(vtype)
            if "[" in type_string and "]" in type_string:
                base_type, array_suffix = split_array_type_suffix(type_string)
                if self.is_inferable_resource_array_type(base_type):
                    mapped_type = self.map_image_base_type_with_format(base_type, node)
                    if array_suffix == "[]":
                        array_size = function_hints.get(param_name, "")
                        return f"{mapped_type}[{array_size or 1}]"
                    return f"{mapped_type}{array_suffix}"

        return self.map_resource_type_with_format(vtype, node)

    def glsl_private_pointer_parameter_array_type(self, parameter, function_name):
        if not is_private_pointer_parameter(parameter):
            return None
        parameter_name = getattr(parameter, "name", None)
        size = self.current_glsl_private_pointer_variant.get(parameter_name)
        if size is None:
            size = self.function_private_pointer_array_size_hints.get(
                function_name, {}
            ).get(parameter_name)
        if size is None or size == "":
            display_name = self.glsl_private_pointer_display_name(function_name)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer parameter "
                f"'{display_name}.{parameter_name}' requires a provable bounded "
                "array span",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="missing-fixed-array-extent",
                source_location=getattr(parameter, "source_location", None),
            )
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        pointee_type = getattr(parameter_type, "pointee_type", None)
        if pointee_type is None or isinstance(pointee_type, PointerType):
            display_name = self.glsl_private_pointer_display_name(function_name)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer parameter "
                f"'{display_name}.{parameter_name}' has an unsupported pointee type",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        if self.glsl_private_pointer_byte_view_variant(size):
            return size[2]
        mapped_type = self.map_type(pointee_type)
        if size in {0, "0"}:
            return mapped_type
        return f"{mapped_type}[{size}]"

    def map_resource_type_with_format(self, vtype, node=None):
        if vtype is None:
            return self.map_type(vtype)

        if (
            isinstance(vtype, (PointerType, ReferenceType))
            or hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
        ):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        if "[" in vtype_str and "]" in vtype_str:
            base_type, array_suffix = split_array_type_suffix(vtype_str)
            base_mapped = self.map_image_base_type_with_format(base_type, node)
            return f"{base_mapped}{array_suffix}"

        return self.map_image_base_type_with_format(vtype_str, node)

    def invalid_glsl_image_format_message(self, node, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Invalid OpenGL image format metadata for "
            f"'{node_name}': format {source} is not a supported storage image format"
        )

    def glsl_image_format_choice_description(self, attr_name, source):
        if attr_name == "format":
            source = source if source is not None else "<missing>"
            return f"format {source}"
        return f"@{attr_name}"

    def duplicate_glsl_image_format_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        description = self.glsl_image_format_choice_description(attr_name, source)
        return (
            f"Duplicate OpenGL image format metadata for '{node_name}': {description}"
        )

    def conflicting_glsl_image_format_message(
        self, node, previous_attr, previous_source, attr_name, source
    ):
        node_name = self.resource_node_name(node, "<unnamed>")
        previous = self.glsl_image_format_choice_description(
            previous_attr, previous_source
        )
        current = self.glsl_image_format_choice_description(attr_name, source)
        return (
            "Conflicting OpenGL image format metadata for "
            f"'{node_name}': {previous} differs from {current}"
        )

    def explicit_glsl_image_format(self, node):
        if node is None or not hasattr(node, "attributes"):
            return None
        supported_formats = supported_image_formats()
        choices = []
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                choices.append((attr_name, attr_name, attr_name))
                continue
            if attr_name != "format":
                continue

            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            if source is None:
                raise OpenGLRefusal(
                    self.invalid_glsl_image_format_message(node, source)
                )
            format_name = str(source).lower()
            if format_name == self.GLSL_UNKNOWN_IMAGE_FORMAT:
                choices.append((attr_name, source, format_name))
                continue
            if format_name not in supported_formats:
                raise OpenGLRefusal(
                    self.invalid_glsl_image_format_message(node, source)
                )
            choices.append((attr_name, source, format_name))

        if not choices:
            return None

        seen_by_attribute = {}
        for attr_name, source, format_name in choices:
            previous = seen_by_attribute.get(attr_name)
            if previous is not None:
                previous_source, previous_format = previous
                if format_name == previous_format:
                    raise OpenGLRefusal(
                        self.duplicate_glsl_image_format_message(
                            node, attr_name, source
                        )
                    )
                raise OpenGLRefusal(
                    self.conflicting_glsl_image_format_message(
                        node, attr_name, previous_source, attr_name, source
                    )
                )
            seen_by_attribute[attr_name] = (source, format_name)

        first_attr, first_source, first_format = choices[0]
        for attr_name, source, format_name in choices[1:]:
            if format_name == first_format:
                continue
            raise OpenGLRefusal(
                self.conflicting_glsl_image_format_message(
                    node, first_attr, first_source, attr_name, source
                )
            )
        return first_format

    def validate_glsl_image_format_target(self, node, vtype, explicit_format):
        if explicit_format is None or self.is_storage_image_type(vtype):
            return
        node_name = self.resource_node_name(node, "<unnamed>")
        type_name = self.type_name_string(self.resource_base_type(vtype))
        raise OpenGLRefusal(
            "OpenGL image format metadata for "
            f"'{node_name}' applies only to storage image resources, "
            f"got {type_name}"
        )

    def map_image_base_type_with_format(self, vtype, node=None):
        base_type = self.resource_base_type(vtype)
        explicit_format = self.explicit_glsl_image_format(node)
        self.validate_glsl_image_format_target(node, base_type, explicit_format)
        if explicit_format in {
            "r8",
            "r8_snorm",
            "r16",
            "r16_snorm",
            "r16f",
            "r32f",
            "r8i",
            "r16i",
            "r32i",
            "r8ui",
            "r16ui",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg8i",
            "rg16i",
            "rg8ui",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba32f",
            "rgba8i",
            "rgba16i",
            "rgba32i",
            "rgba8ui",
            "rgba16ui",
            "rgba32ui",
            "rgb10_a2",
            "rgb10_a2ui",
            "r11f_g11f_b10f",
        }:
            if explicit_format in {
                "r8",
                "r8_snorm",
                "r16",
                "r16_snorm",
                "r16f",
                "rg8",
                "rg8_snorm",
                "rg16",
                "rg16_snorm",
                "rg16f",
                "rg32f",
                "rgba8",
                "rgba8_snorm",
                "rgba16",
                "rgba16_snorm",
                "rgba16f",
                "rgba32f",
                "rgb10_a2",
                "r11f_g11f_b10f",
            }:
                format_class = "r32f"
            elif explicit_format.endswith("ui"):
                format_class = "r32ui"
            elif explicit_format.endswith("i"):
                format_class = "r32i"
            else:
                format_class = explicit_format
            format_type_map = {
                "image1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "iimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "uimage1D": {
                    "r32f": "image1D",
                    "r32i": "iimage1D",
                    "r32ui": "uimage1D",
                },
                "image2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "iimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "uimage2D": {
                    "r32f": "image2D",
                    "r32i": "iimage2D",
                    "r32ui": "uimage2D",
                },
                "image3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "iimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "uimage3D": {
                    "r32f": "image3D",
                    "r32i": "iimage3D",
                    "r32ui": "uimage3D",
                },
                "image1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "iimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "uimage1DArray": {
                    "r32f": "image1DArray",
                    "r32i": "iimage1DArray",
                    "r32ui": "uimage1DArray",
                },
                "image2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "iimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "uimage2DArray": {
                    "r32f": "image2DArray",
                    "r32i": "iimage2DArray",
                    "r32ui": "uimage2DArray",
                },
                "image2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "iimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "uimage2DMS": {
                    "r32f": "image2DMS",
                    "r32i": "iimage2DMS",
                    "r32ui": "uimage2DMS",
                },
                "image2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "iimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "uimage2DMSArray": {
                    "r32f": "image2DMSArray",
                    "r32i": "iimage2DMSArray",
                    "r32ui": "uimage2DMSArray",
                },
                "imageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "iimageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "uimageCube": {
                    "r32f": "imageCube",
                    "r32i": "iimageCube",
                    "r32ui": "uimageCube",
                },
                "imageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
                "iimageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
                "uimageCubeArray": {
                    "r32f": "imageCubeArray",
                    "r32i": "iimageCubeArray",
                    "r32ui": "uimageCubeArray",
                },
            }
            mapped_type = format_type_map.get(base_type, {}).get(format_class)
            if mapped_type:
                return mapped_type

        return self.map_type(vtype)

    def is_opaque_resource_type(self, vtype):
        return vtype in {
            "texture1D",
            "texture1DArray",
            "texture2D",
            "texture3D",
            "textureCube",
            "texture2DArray",
            "textureCubeArray",
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "samplerExternalOES",
            "__samplerExternal2DY2YEXT",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "samplerCubeArray",
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "sampler2DRect",
            "samplerBuffer",
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler1D",
            "isampler1DArray",
            "isampler2D",
            "isampler2DArray",
            "isampler3D",
            "isamplerCube",
            "isamplerCubeArray",
            "isampler2DRect",
            "isamplerBuffer",
            "isampler2DMS",
            "isampler2DMSArray",
            "usampler1D",
            "usampler1DArray",
            "usampler2D",
            "usampler2DArray",
            "usampler3D",
            "usamplerCube",
            "usamplerCubeArray",
            "usampler2DRect",
            "usamplerBuffer",
            "usampler2DMS",
            "usampler2DMSArray",
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DArray",
            "image2DMS",
            "image2DMSArray",
            "imageBuffer",
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "iimageBuffer",
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
            "uimageBuffer",
            "atomic_uint",
            "accelerationStructureEXT",
        }

    def supported_image_formats(self):
        return supported_image_formats()

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "volatile",
            "restrict",
            "readonly",
            "writeonly",
            "readwrite",
        }

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
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

    def invalid_resource_binding_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        if attr_name == "register":
            requirement = "must use b/s/t/u register syntax or an integer binding"
        else:
            requirement = "must resolve to a concrete integer binding"
        return (
            "Invalid OpenGL resource binding metadata for "
            f"'{node_name}': {attr_name} {source} {requirement}"
        )

    def unsupported_resource_space_message(self, node, attr_name, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Unsupported OpenGL resource binding metadata for "
            f"'{node_name}': {attr_name} {source} is not supported by OpenGL GLSL"
        )

    def resource_register_prefix(self, source):
        if source is None:
            return None
        raw_source = str(source).strip().lower()
        if len(raw_source) >= 2 and raw_source[0] in {"b", "s", "t", "u"}:
            suffix = raw_source[1:]
            if suffix.isdigit():
                return raw_source[0]
        return None

    def expected_resource_register_prefix(self, node):
        if (
            hasattr(node, "members")
            and not hasattr(node, "var_type")
            and not hasattr(node, "param_type")
        ):
            return "b", "uniform buffer binding"

        vtype = self.resource_node_type(node)

        if self.is_glsl_buffer_block_variable(node, vtype):
            return "u", "buffer binding"

        if self.is_structured_buffer_type(vtype):
            if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
                return "t", "buffer binding"
            return "u", "buffer binding"

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if self.is_opaque_resource_type(mapped_type):
            if self.is_storage_image_type(vtype):
                return "u", "image binding"
            return "t", "texture binding"

        if mapped_type == "sampler":
            return "s", "sampler binding"

        return None, None

    def accepts_hlsl_sampler_register_for_texture_binding(self, node):
        vtype = self.resource_node_type(node)
        mapped_type = self.map_resource_type_with_format(vtype, node)
        return (
            self.is_opaque_resource_type(mapped_type)
            and not self.is_storage_image_type(vtype)
            and mapped_type.startswith(("sampler", "isampler", "usampler"))
        )

    def validate_resource_register_prefix(self, node, source):
        actual_prefix = self.resource_register_prefix(source)
        if actual_prefix is None:
            return

        expected_prefix, namespace = self.expected_resource_register_prefix(node)
        if expected_prefix is None or actual_prefix == expected_prefix:
            return
        if (
            expected_prefix == "t"
            and actual_prefix == "s"
            and self.accepts_hlsl_sampler_register_for_texture_binding(node)
        ):
            return

        node_name = self.resource_node_name(node, "<unnamed>")
        raise OpenGLRefusal(
            "Incompatible OpenGL resource register metadata for "
            f"'{node_name}': register {source} uses {actual_prefix}-register, "
            f"expected {expected_prefix}-register for {namespace}"
        )

    def has_descriptor_set_attribute(self, node):
        if not hasattr(node, "attributes"):
            return False
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() in {"set", "space"}:
                return True
        return False

    def has_explicit_opengl_target_binding_attribute(self, node):
        if not hasattr(node, "attributes"):
            return False
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() in {
                "buffer",
                "register",
                "sampler",
                "texture",
            }:
                return True
        return False

    def source_descriptor_binding_index(self, node):
        if not self.has_descriptor_set_attribute(node):
            return None
        choices = self.explicit_resource_binding_choices(node)
        binding_choices = [
            binding for attr_name, _, binding in choices if attr_name == "binding"
        ]
        if not binding_choices:
            return None
        self.explicit_resource_binding_index(node)
        return binding_choices[0]

    def raw_descriptor_binding_index(self, node):
        """The ``binding`` of ``node`` inside its descriptor set, before OpenGL binding points are assigned."""
        choices = self.explicit_resource_binding_choices(node, flatten_sets=False)
        return next(
            (binding for attr_name, _, binding in choices if attr_name == "binding"),
            choices[0][2] if choices else None,
        )

    def should_remap_descriptor_array_target_binding(self, node, resource_count):
        return (
            resource_count > 1
            and self.source_descriptor_binding_index(node) is not None
            and not self.has_explicit_opengl_target_binding_attribute(node)
        )

    def invalid_resource_access_message(self, node, source):
        node_name = self.resource_node_name(node, "<unnamed>")
        source = source if source is not None else "<missing>"
        return (
            "Invalid OpenGL resource access metadata for "
            f"'{node_name}': access({source}) must be readonly, writeonly, "
            "or readwrite"
        )

    def explicit_resource_binding_choices(self, node, flatten_sets=True):
        choices = []
        if not hasattr(node, "attributes"):
            return choices
        descriptor_set = 0
        descriptor_set_description = None
        node_name = self.resource_node_name(node, "<unnamed>")
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name not in {"set", "space"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            resource_space = (
                self.binding_index_value(arguments[0]) if arguments else None
            )
            if resource_space is None:
                raise OpenGLRefusal(
                    self.invalid_resource_binding_message(node, attr_name, source)
                )
            description = f"{attr_name} {source if source is not None else '<missing>'}"
            if descriptor_set_description is not None:
                if resource_space == descriptor_set:
                    raise OpenGLRefusal(
                        "Duplicate OpenGL resource binding metadata for "
                        f"'{node_name}': {description}"
                    )
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {descriptor_set_description} differs from "
                    f"{description}"
                )
            descriptor_set = resource_space
            descriptor_set_description = description

        if descriptor_set_description is None:
            descriptor_set = self.register_space_index(node)

        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in {"set", "space"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            source = self.attribute_value_to_string(arguments[0]) if arguments else None
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture"}:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], ("b", "s", "t", "u"))
                self.validate_resource_register_prefix(node, source)
            else:
                continue
            if binding is None:
                raise OpenGLRefusal(
                    self.invalid_resource_binding_message(node, attr_name, source)
                )
            if flatten_sets:
                binding = self.opengl_set_binding(node, descriptor_set, binding)
            choices.append((attr_name, source, binding))
        return choices

    def explicit_resource_binding_choice_description(self, attr_name, source, binding):
        if attr_name == "binding":
            source = source if source is not None else binding
            return f"binding {source}"
        source = source if source is not None else binding
        return f"{attr_name} {source} binding {binding}"

    def explicit_resource_binding_index(self, node):
        planned = (getattr(self, "glsl_register_target_bindings", None) or {}).get(
            id(node)
        )
        if planned is not None:
            return planned
        choices = self.explicit_resource_binding_choices(node)
        if not choices:
            return None

        node_name = self.resource_node_name(node, "<unnamed>")
        seen_by_attribute = {}
        for attr_name, source, binding in choices:
            description = self.explicit_resource_binding_choice_description(
                attr_name, source, binding
            )
            previous = seen_by_attribute.get(attr_name)
            if previous is not None:
                previous_description, previous_binding = previous
                if binding == previous_binding:
                    raise OpenGLRefusal(
                        "Duplicate OpenGL resource binding metadata for "
                        f"'{node_name}': {description}"
                    )
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {previous_description} differs from "
                    f"{description}"
                )
            seen_by_attribute[attr_name] = (description, binding)

        first_name, first_source, first_binding = choices[0]
        first_description = self.explicit_resource_binding_choice_description(
            first_name, first_source, first_binding
        )
        for attr_name, source, binding in choices[1:]:
            if binding != first_binding:
                current_description = self.explicit_resource_binding_choice_description(
                    attr_name, source, binding
                )
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource binding metadata for "
                    f"'{node_name}': {first_description} differs from "
                    f"{current_description}"
                )
        return first_binding

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic is not None:
            return semantic
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            if self.glsl_stage_control_attribute_name(attr):
                continue
            if (
                self.is_glsl_stage_io_metadata_attribute(attr)
                or str(getattr(attr, "name", "")).lower()
                in {"maybe_unused", "source_resource"}
                or is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_hlsl_metadata_attribute(attr)
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def glsl_stage_control_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]

        valid_names = (
            {
                "cw",
                "ccw",
                "compute",
                "domain",
                "early_fragment_tests",
                "equal_spacing",
                "fragment",
                "fractional_even_spacing",
                "fractional_odd_spacing",
                "hlsl_wavesize",
                "instance",
                "invocations",
                "inputtopology",
                "isolines",
                "layout",
                "local_size",
                "line_strip",
                "lines",
                "lines_adjacency",
                "local_size_x",
                "local_size_y",
                "local_size_z",
                "max_vertices",
                "max_primitives",
                "max_total_threads_per_threadgroup",
                "max_total_threadgroups_per_mesh_grid",
                "maxprimitivecount",
                "maxvertexcount",
                "numthreads",
                "outputcontrolpoints",
                "outputtopology",
                "partitioning",
                "patchconstantfunc",
                "point_mode",
                "points",
                "quads",
                "stage_entry",
                "stream",
                "triangles",
                "triangles_adjacency",
                "triangle_strip",
                "vertex",
                "vertices",
                "workgroup_size",
                "wavesize",
            }
            | self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES
            | INTERLOCK_LAYOUTS
        )
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
            if self.glsl_stage_control_attribute_name(attr):
                continue
            if (
                self.is_glsl_stage_io_metadata_attribute(attr)
                or is_image_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or is_resource_access_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.is_hlsl_metadata_attribute(attr)
                or (
                    getattr(attr, "name", None)
                    and str(getattr(attr, "name")).lower() == "glsl_buffer_block"
                )
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def is_hlsl_metadata_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {"rootsignature", "hlsl_rootsignature"}

    def image_format_qualifier(self, vtype, node=None):
        explicit_format = self.explicit_glsl_image_format(node)
        self.validate_glsl_image_format_target(node, vtype, explicit_format)
        if explicit_format == self.GLSL_UNKNOWN_IMAGE_FORMAT:
            return self.glsl_unknown_image_format_qualifier(vtype, node)
        if explicit_format:
            return explicit_format
        if vtype in {"imageBuffer", "iimageBuffer", "uimageBuffer"}:
            # Texel buffers take the format of the bound buffer view.
            return self.glsl_unknown_image_format_qualifier(vtype, node)
        if vtype in {
            "image1D",
            "image1DArray",
            "image2D",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DArray",
        }:
            return "rgba32f"
        if vtype in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
        }:
            return "r32i"
        if vtype in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
        }:
            return "r32ui"
        return None

    GLSL_ES_READ_WRITE_IMAGE_FORMATS = frozenset({"r32f", "r32i", "r32ui"})

    def glsl_es_image_access_qualifier(self, vtype, node, memory_qualifiers):
        """The access qualifier a GLSL ES storage image needs and lacks, or ``""``.

        GLSL ES 3.10 (section 4.10) lets only ``r32f``, ``r32i`` and ``r32ui`` images be both read and
        written, so an image of any other format must be declared ``readonly`` or ``writeonly``. An image
        that is only loaded from or size-queried is ``readonly``, and one that is only stored to is
        ``writeonly``. One that is both, or that is passed to a program function, has no valid declaration.
        """
        if not self.glsl_es_target or not self.is_storage_image_type(vtype):
            return ""
        if {"readonly", "writeonly"} & set(memory_qualifiers.split()):
            return ""
        image_format = self.image_format_qualifier(vtype, node)
        if not image_format or image_format in self.GLSL_ES_READ_WRITE_IMAGE_FORMATS:
            return ""
        names = {self.resource_node_name(node)}
        if getattr(node, "name", None):
            names.add(self.glsl_global_identifier_name(node.name))
        usages = set()
        for name in names:
            usages |= self.glsl_image_usages.get(name, set())
        if not usages - {"load"}:
            return "readonly"
        if usages == {"store"}:
            return "writeonly"
        raise OpenGLRefusal(
            f"GLSL ES storage image '{self.resource_node_name(node)}' has format "
            f"{image_format} and is both read and written or passed to a function; "
            "GLSL ES allows read-write access only to r32f, r32i and r32ui images"
        )

    def opaque_resource_layout(self, vtype, binding, node=None):
        image_format = self.image_format_qualifier(vtype, node)
        return self.glsl_resource_layout_prefix(
            image_format, binding=binding, node=node
        )

    def resource_precision_qualifiers(self, node):
        qualifiers = [
            qualifier
            for qualifier in self.glsl_variable_qualifier_prefix(node).split()
            if qualifier in self.GLSL_PRECISION_QUALIFIERS
        ]
        return " ".join(qualifiers)
