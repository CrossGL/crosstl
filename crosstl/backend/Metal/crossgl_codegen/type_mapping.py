"""Metal to CrossGL type mapping, type aliases and resource types."""

import re

from ....translator.codegen.array_utils import prepend_array_dimension
from ..conditional_types import split_template_arguments
from ..generic_local_aliases import refuse_unresolved_local_alias_type
from ..simd_aliases import SIMD_TYPE_ALIASES
from ..sparse_color import refuse_sparse_color_type
from ..texture_types import (
    METAL_DEPTH_TEXTURE_TYPES,
    METAL_SAMPLED_TEXTURE_TYPES,
)
from .support import MetalScalarAliasResolutionError


class MetalTypeMappingMixin:
    """Members of ``MetalToCrossGLConverter``: Metal to CrossGL type mapping, type aliases and resource types."""

    def map_type(self, metal_type):
        """Map a Metal type name to the closest CrossGL type name."""
        if not metal_type:
            return metal_type
        sparse_color = refuse_sparse_color_type(self, metal_type)
        if sparse_color is not None:
            return sparse_color

        metal_type = self.substitute_template_type_text(metal_type)
        metal_type = self.substitute_template_value_text(metal_type)

        decltype_type = self.resolve_metal_decltype_type(metal_type)
        if decltype_type is not None and decltype_type != str(metal_type).strip():
            return self.map_type(decltype_type)

        materialized_alias = self.materialize_alias_template_type(metal_type)
        if materialized_alias != str(metal_type).strip():
            return self.map_type(materialized_alias)

        resolved_local_type = self.resolve_local_type_aliases(metal_type)
        if resolved_local_type != str(metal_type).strip():
            return self.map_type(resolved_local_type)
        refuse_unresolved_local_alias_type(self, metal_type)

        empty_array = self.zero_extent_metal_array_type_parts(metal_type)
        if empty_array is not None:
            return f"array<{self.map_standard_array_storage_type(empty_array[0])}, 0>"

        alias_base = str(metal_type).strip()
        alias_suffix = ""
        while alias_base.endswith("*") or alias_base.endswith("&"):
            alias_suffix = alias_base[-1] + alias_suffix
            alias_base = alias_base[:-1].strip()
        scalar_alias = self.scalar_alias_binding(alias_base)
        if scalar_alias is not None and any(
            self.scalar_alias_needs_inline(alias)
            for alias in self.scalar_alias_declarations(alias_base)
        ):
            return f"{self.map_type(scalar_alias[0])}{alias_suffix}"
        resolved_alias = self.resolve_type_alias(alias_base)
        if resolved_alias != alias_base:
            wide_vector_alias = self.wide_vector_type_info(resolved_alias)
            if wide_vector_alias is not None:
                return f"{wide_vector_alias['type_name']}{alias_suffix}"
            canonical_struct = self.canonical_concrete_struct_alias_target(
                resolved_alias
            )
            if canonical_struct is not None:
                return f"{self.map_type(canonical_struct)}{alias_suffix}"

        array_type = self.metal_array_type_parts(metal_type)
        if array_type:
            element_type, size = array_type
            return prepend_array_dimension(
                self.map_type(element_type), self.format_array_extent(size)
            )

        base = metal_type.strip()
        if base.endswith("..."):
            base = base[:-3].strip()
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()
        for tag_prefix in ("struct ", "enum "):
            if base.startswith(tag_prefix):
                base = base[len(tag_prefix) :].strip()
                break
        base = self.resolve_contextual_nested_struct_type(base)

        uniform_payload_type = self.uniform_value_payload_type(base)
        if uniform_payload_type is not None:
            return f"{self.map_type(uniform_payload_type)}{suffix}"

        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        if base.startswith("raytracing::"):
            base = base.split("raytracing::", 1)[1]

        cooperative_matrix = self.map_metal_cooperative_matrix_type(base)
        if cooperative_matrix is not None:
            return f"{cooperative_matrix}{suffix}"

        conditional_type = self.resolve_conditional_type(base)
        if conditional_type is not None:
            return f"{conditional_type}{suffix}"

        atomic_element = self.atomic_element_type(base)
        if atomic_element is not None:
            return f"{self.map_type(atomic_element)}{suffix}"

        function_table_alias = self.function_table_type_alias(base)
        if function_table_alias:
            return f"{function_table_alias}{suffix}"

        pixel_payload_type = self.pixel_data_payload_type(base)
        if pixel_payload_type:
            return f"{self.map_type(pixel_payload_type)}{suffix}"

        vector_type = self.metal_vector_type_parts(base)
        if vector_type:
            element_type, size = vector_type
            return f"{self.map_generic_vector_type(element_type, size)}{suffix}"

        named_vector_type = self.metal_vector_component_parts(base)
        if named_vector_type is not None:
            element_type, width, _width_text = named_vector_type
            if width is not None and 1 < width <= 4:
                return f"{self.map_generic_vector_type(element_type, width)}{suffix}"

        # Normalize Metal resource access qualifiers without dropping dimensions or
        # other non-resource generic arguments, e.g. matrix<bfloat, 4, 4>.
        if "<" in base and base.endswith(">"):
            base_name, inner = base.split("<", 1)
            inner = inner[:-1]
            generic_args = self.split_generic_arguments(inner)
            if self.should_elide_resource_access_qualifier(base_name, generic_args):
                base = f"{base_name}<{generic_args[0].strip()}>"
            elif (
                not self.is_metal_resource_type_name(base_name)
                and "::" not in base_name
            ):
                mapped_base_name = self.sanitize_identifier(base_name.strip())
                mapped_args = ",".join(
                    self.map_generic_type_argument(arg) for arg in generic_args
                )
                return f"{mapped_base_name}<{mapped_args}>{suffix}"

        sampled_resource_type = self.map_sampled_texture_type(base)
        if sampled_resource_type:
            return f"{sampled_resource_type}{suffix}"

        if base.startswith("xhalf"):
            # The xhalf names are a sample convention for half, not MSL types:
            # a typedef the program declares, such as `typedef float xhalf;`,
            # decides what they mean.
            declared = self.resolve_type_alias(base)
            if declared != base:
                return f"{self.map_type(declared)}{suffix}"
        mapped = self.type_map.get(base)
        if mapped is not None:
            return f"{mapped}{suffix}"
        mapped = self.struct_name_map.get(base)
        if mapped is not None:
            return f"{mapped}{suffix}"
        mapped = self.map_scoped_type_name(base)
        return f"{mapped}{suffix}"

    def canonical_concrete_struct_alias_target(self, metal_type):
        candidate = str(metal_type or "").strip()
        array_type = self.metal_array_type_parts(candidate)
        if array_type:
            element_type, size = array_type
            canonical_element = self.canonical_concrete_struct_alias_target(
                element_type
            )
            if canonical_element is None:
                return None
            return prepend_array_dimension(
                canonical_element, self.format_array_extent(size)
            )
        suffix = ""
        while candidate.endswith(("*", "&")):
            suffix = candidate[-1] + suffix
            candidate = candidate[:-1].strip()
        candidate = re.sub(
            r"^(?:(?:const|device|thread|threadgroup|constant|volatile|restrict)\s+)+",
            "",
            candidate,
        )
        for tag_prefix in ("struct ", "union "):
            if candidate.startswith(tag_prefix):
                candidate = candidate[len(tag_prefix) :].strip()
                break
        if (
            candidate in self.struct_name_map
            or candidate in self.struct_name_map.values()
        ):
            return f"{candidate}{suffix}"
        if "::" not in candidate:
            return None
        unqualified = candidate.rsplit("::", 1)[-1]
        if (
            unqualified in self.struct_name_map
            and unqualified not in self.ambiguous_struct_names
        ):
            return f"{unqualified}{suffix}"
        return None

    def resolve_conditional_type(self, base, *, require_concrete=False):
        """Resolve ``conditional_t<C, A, B>`` / ``conditional<C, A, B>::type``.

        Materialized functions and concrete kernels can make ``C`` provable from
        template bindings or a preceding local integral ``constexpr``.  Select
        that exact branch first.  Constructor selection passes
        ``require_concrete=True`` so it never chooses a constructor from an
        unresolved symbolic branch.  Other uninstantiated generic emission keeps
        the legacy integer-safe else branch used by existing diagnostics and
        parsing contracts.
        """
        # ``typename metal::conditional_t<...>`` names the same selection.
        candidate = re.sub(
            r"^(?:typename\s+)?(?:::\s*)?(?:metal\s*::\s*)?", "", str(base).strip()
        )
        type_accessor = False
        if candidate.endswith("::type"):
            candidate = candidate[: -len("::type")].strip()
            type_accessor = True
        if "<" not in candidate or not candidate.endswith(">"):
            return None
        name, inner = candidate.split("<", 1)
        name = name.strip()
        if name not in ("conditional_t", "conditional"):
            return None
        # ``conditional_t`` is the alias form (no ``::type``); the bare
        # ``conditional`` metafunction only yields a type through ``::type``.
        if name == "conditional" and not type_accessor:
            return None
        # Conditions such as ``flag && (N < 4)`` hold angle brackets that are
        # operators, so split with the template-argument rules.
        args = split_template_arguments(inner[:-1])
        if args is None or len(args) != 3:
            return None

        bindings = {
            name: value
            for scope in self.template_value_bindings
            for name, value in scope.items()
            if not self.template_value_binding_is_shadowed(name)
        }
        bindings.update(self.current_function_materialization_bindings)
        condition = self.substitute_template_value_text(
            args[0].strip(),
            bindings=bindings,
            honor_shadowing=False,
        )
        condition = self.substitute_local_integral_constant_text(condition)
        condition_value = self.evaluate_value_template_constant_expression(
            condition, constants={"true": 1, "false": 0}
        )
        if condition_value is None and require_concrete:
            return None
        selected_branch = (
            args[1] if condition_value is not None and condition_value else args[2]
        )
        selected_branch = selected_branch.strip()
        if not selected_branch:
            return None
        # A branch may itself be a selection, as in nested conditional_t.
        nested = self.resolve_conditional_type(
            selected_branch, require_concrete=require_concrete
        )
        if nested is not None:
            return nested
        return self.map_type(selected_branch)

    def map_generic_type_argument(self, argument):
        argument = str(argument).strip()
        if not argument:
            return argument
        if self.crossgl_identifier_pattern.fullmatch(argument):
            return self.sanitize_identifier(argument)
        base_name, generic_args = self.generic_type_parts(argument)
        if (
            base_name
            and generic_args
            and "::" not in base_name
            and argument.endswith(">")
        ):
            mapped_base_name = self.sanitize_identifier(base_name.strip())
            mapped_args = ",".join(
                self.map_generic_type_argument(arg) for arg in generic_args
            )
            return f"{mapped_base_name}<{mapped_args}>"
        return argument

    def format_type_alias_declaration(self, alias):
        if (
            self.is_template_alias_declaration(alias)
            or alias.name in self.callable_type_aliases
            or getattr(alias, "is_function_type", False)
            or self.is_resource_type_alias(alias)
        ):
            return None
        if (
            self.wide_vector_type_info(
                alias.alias_type, getattr(alias, "source_location", None)
            )
            is not None
        ):
            return None

        scalar_binding = self.scalar_alias_binding(alias.name, context=alias)
        if scalar_binding is not None and self.scalar_alias_needs_inline(alias):
            # Namespace-local primitive aliases are resolved at each use, so
            # flattening namespaces cannot merge unrelated typedef names.
            return None
        mapped_type = self.crossgl_typedef_source_type(self.map_type_alias(alias))
        alias_name = self.sanitize_identifier(alias.name)
        target = str(alias.alias_type).strip()
        if (
            not mapped_type
            and re.fullmatch(r"[A-Za-z_]\w*", target)
            and target not in self.type_map
            and target not in self.struct_name_map
            and target not in self.metal_enum_arithmetic_types
            and not self.scalar_alias_declarations(target, context=alias)
            and not getattr(alias, "array_sizes", None)
            and not getattr(alias, "declarator_type_suffix", "")
        ):
            raise MetalScalarAliasResolutionError(
                alias.name,
                f"target '{target}' has no concrete visible declaration",
                source_location=getattr(alias, "source_location", None),
            )
        if not mapped_type or not alias_name or mapped_type == alias_name:
            return None
        return f"typedef {mapped_type} {alias_name};"

    def map_type_alias(self, alias):
        storage_type = self.map_storage_texture_type(alias.alias_type)
        if storage_type:
            return storage_type
        scalar_binding = self.scalar_alias_declaration_binding(alias)
        target = scalar_binding[0] if scalar_binding is not None else alias.alias_type
        if self.normalized_metal_type(target) == "half":
            return "f16"
        return self.map_type(target)

    def crossgl_typedef_source_type(self, mapped_type):
        mapped_type = str(mapped_type).strip()
        if not mapped_type or "::" in mapped_type:
            return None
        if mapped_type in {
            info["type_name"] for info in self.wide_vector_types.values()
        }:
            return mapped_type
        if any(char in mapped_type for char in "*&[]"):
            return None

        normalized_type = self.normalize_crossgl_typedef_source_type(mapped_type)
        if normalized_type is not None:
            return normalized_type
        if re.fullmatch(r"(?:matrix|vec|vector)<[^{};]+>", mapped_type):
            return mapped_type
        if mapped_type in self.crossgl_typedef_source_types():
            return mapped_type
        return None

    def normalize_crossgl_typedef_source_type(self, mapped_type):
        scalar_aliases = {
            "int8": "i8",
            "uint8": "u8",
            "int16": "i16",
            "uint16": "u16",
            "int64": "i64",
            "uint64": "u64",
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
        }
        if mapped_type in scalar_aliases:
            return scalar_aliases[mapped_type]

        vector_match = re.fullmatch(r"(f16|i8|u8|i16|u16)vec([2-4])", mapped_type)
        if vector_match:
            element_type, size = vector_match.groups()
            return f"vector<{element_type},{size}>"

        matrix_match = re.fullmatch(r"f16mat([2-4])(?:x([2-4]))?", mapped_type)
        if matrix_match:
            columns, rows = matrix_match.groups()
            rows = rows or columns
            return f"matrix<f16,{columns},{rows}>"
        return None

    def crossgl_typedef_source_types(self):
        return {
            "bool",
            "int",
            "uint",
            "float",
            "double",
            "f16",
            "bfloat",
            "bfloat16",
            "i8",
            "i16",
            "i32",
            "i64",
            "u8",
            "u16",
            "u32",
            "u64",
            "f32",
            "f64",
            "half",
            "char",
            "string",
            "void",
            "vec2",
            "vec3",
            "vec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
        }

    def map_scoped_type_name(self, type_name):
        if "::" not in str(type_name):
            text = str(type_name)
            if self.crossgl_identifier_pattern.match(text):
                return self.sanitize_identifier(text)
            return text

        base_name, _generic_args = self.generic_type_parts(type_name)
        if base_name and "::" in base_name:
            return type_name

        return self.sanitize_identifier(str(type_name).split("::")[-1])

    def is_resource_type_alias(self, alias):
        return self.is_metal_resource_type(alias.alias_type)

    def resolve_type_alias(self, metal_type):
        if not metal_type:
            return metal_type

        base = str(metal_type).strip()
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        if base in SIMD_TYPE_ALIASES:
            # <simd/simd.h> reserves these names for the Metal types they alias.
            return f"{SIMD_TYPE_ALIASES[base]}{suffix}"
        if (
            "<" not in base
            and ":" not in base
            and base not in self.scalar_alias_names
            and base not in self.type_aliases
        ):
            # A plain name absent from both alias tables cannot resolve further.
            return f"{base}{suffix}"
        scalar_binding = self.scalar_alias_binding(base)
        if scalar_binding is not None:
            return f"{scalar_binding[0]}{suffix}"
        base = self.materialize_alias_template_type(base)
        seen = set()
        previous_context = self.current_type_resolution_context
        local = True
        try:
            while True:
                active_binding = local and (
                    base in self.local_type_alias_names
                    or any(base in scope for scope in self.template_type_bindings)
                )
                declarations = (
                    [] if active_binding else self.scalar_alias_declarations(base)
                )
                if declarations:
                    declaration = declarations[0]
                    key = id(declaration)
                    aliased = str(declaration.alias_type).strip()
                    self.current_type_resolution_context = declaration
                    local = False
                elif base in self.type_aliases and (
                    active_binding or base not in self.scalar_alias_names
                ):
                    key = ("local", base)
                    aliased = str(self.type_aliases[base]).strip()
                else:
                    break
                if key in seen:
                    break
                seen.add(key)
                alias_suffix = ""
                while aliased.endswith("*") or aliased.endswith("&"):
                    alias_suffix = aliased[-1] + alias_suffix
                    aliased = aliased[:-1].strip()
                base = self.materialize_alias_template_type(aliased)
                suffix = alias_suffix + suffix
        finally:
            self.current_type_resolution_context = previous_context
        if base in SIMD_TYPE_ALIASES:
            # A user alias may resolve to a <simd/simd.h> name such as float4.
            base = SIMD_TYPE_ALIASES[base]
        return f"{base}{suffix}"

    def atomic_element_type(self, metal_type):
        # Metal atomics lower to their plain element type in CrossGL; atomicity is
        # carried by the atomic_* intrinsics, and the GLSL SSBO / DirectX UAV /
        # SPIR-V backends store atomics as the underlying scalar (an
        # atomic-wrapped element type is not a valid buffer element). Handles both
        # the generic `atomic<T>` form and Metal's `atomic_int`-style aliases.
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name == "atomic" and len(generic_args) == 1:
            return generic_args[0].strip()
        return {
            "atomic_int": "int",
            "atomic_uint": "uint",
            "atomic_bool": "bool",
            "atomic_ulong": "uint64_t",
            "atomic_float": "float",
        }.get(str(metal_type).strip())

    def uniform_value_payload_type(self, metal_type):
        """Return the value type carried by Metal's ``uniform<T>`` wrapper."""
        base_name, generic_args = self.generic_type_parts(metal_type)
        if len(generic_args) != 1:
            return None
        if base_name == "metal::uniform":
            return generic_args[0].strip()
        if base_name != "uniform" or base_name in self.struct_name_map:
            return None
        return generic_args[0].strip()

    def function_table_type_alias(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if (
            base_name in {"visible_function_table", "intersection_function_table"}
            and generic_args
        ):
            return base_name
        return None

    def pixel_data_payload_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name in self.pixel_data_type_wrappers and len(generic_args) == 1:
            return generic_args[0].strip()
        return None

    def map_sampled_texture_type(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if not base_name or not generic_args:
            return None

        resource_type = METAL_SAMPLED_TEXTURE_TYPES.get(
            base_name, METAL_DEPTH_TEXTURE_TYPES.get(base_name)
        )
        if resource_type is None:
            return None
        if base_name.startswith("depth"):
            return resource_type

        integer_prefix = self.integer_resource_prefix(generic_args[0])
        if integer_prefix is None:
            if self.is_float_resource_element(generic_args[0]):
                return resource_type
            return None
        return f"{integer_prefix}{resource_type}"

    def integer_resource_prefix(self, metal_type):
        mapped = self.map_type(str(metal_type).strip())
        if mapped in {"uint", "uint8", "uint16", "uint64"}:
            return "u"
        if mapped in {"int", "int8", "int16", "int64"}:
            return "i"
        return None

    def is_float_resource_element(self, metal_type):
        return self.map_type(str(metal_type).strip()) in {
            "float",
            "float16",
            "double",
        }

    def should_elide_resource_access_qualifier(self, base_name, generic_args):
        if len(generic_args) < 2 or not self.is_metal_resource_type_name(base_name):
            return False
        return self.normalized_access_qualifier(generic_args[1]).startswith("access::")

    def is_metal_resource_type_name(self, base_name):
        base = str(base_name).strip()
        while base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        return base in {
            "texture1d",
            "texture1d_array",
            "texture2d",
            "texture2d_array",
            "texture2d_ms",
            "texture2d_ms_array",
            "texture3d",
            "texturecube",
            "texturecube_array",
            "texture_buffer",
            "depth2d",
            "depth2d_array",
            "depth2d_ms",
            "depth2d_ms_array",
            "depthcube",
            "depthcube_array",
        }

    def is_metal_resource_type(self, metal_type):
        resolved_type = self.resolve_type_alias(metal_type)
        base_name, _generic_args = self.generic_type_parts(resolved_type)
        if not base_name:
            base_name = self.normalized_metal_type(resolved_type)
        return self.is_metal_resource_type_name(base_name)
