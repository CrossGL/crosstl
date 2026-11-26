"""Type conversion in the CUDA to CrossGL converter: variable, member, resource, template, pointer and array types, and built-in function names."""

import re

from ..libm_names import LIBM_BINARY32_FUNCTIONS


class CudaImportTypeMixin:
    """Type members of ``CudaToCrossGLConverter``."""

    def convert_cuda_variable_type_to_crossgl(self, cuda_type, name):
        """Convert CUDA variable types, using call-site hints for resource handles."""
        resource_type = self.convert_cuda_resource_object_type(cuda_type, name)
        if resource_type is not None:
            return resource_type
        return self.convert_cuda_type_to_crossgl(cuda_type)

    def convert_cuda_struct_member_type_to_crossgl(self, struct_name, cuda_type, name):
        hint = self.struct_resource_member_hints.get((struct_name, name))
        if hint is not None:
            resource_type = self.convert_cuda_resource_object_type_with_hint(
                cuda_type, hint
            )
            if resource_type is not None:
                return resource_type
        return self.convert_cuda_type_to_crossgl(cuda_type)

    def convert_cuda_resource_object_type(self, cuda_type, name):
        hint = self.lookup_resource_object_type_hint(name)
        if hint is None:
            return None
        return self.convert_cuda_resource_object_type_with_hint(cuda_type, hint)

    def convert_cuda_resource_object_type_with_hint(self, cuda_type, hint):
        cuda_type = self.strip_type_qualifiers(cuda_type)

        if self.has_array_suffix(cuda_type):
            base_type = cuda_type.split("[", 1)[0].strip()
            mapped_type = self.convert_cuda_resource_object_type_with_hint(
                base_type, hint
            )
            if mapped_type is None:
                return None
            return self.wrap_mapped_cuda_array_type(cuda_type, mapped_type)

        if "*" in cuda_type:
            pointer_depth = cuda_type.count("*")
            base_type = cuda_type.replace("*", "").strip()
            mapped_type = self.convert_cuda_resource_object_base_type(base_type, hint)
            if mapped_type is None:
                return None
            for _ in range(pointer_depth):
                mapped_type = f"ptr<{mapped_type}>"
            return mapped_type

        return self.convert_cuda_resource_object_base_type(cuda_type, hint)

    def convert_cuda_resource_object_base_type(self, cuda_type, hint):
        cuda_type = self.strip_type_qualifiers(cuda_type)
        if cuda_type == "cudaTextureObject_t" and hint.startswith("sampler"):
            return hint
        if cuda_type == "cudaSurfaceObject_t" and "image" in hint:
            return hint
        return None

    def wrap_mapped_cuda_array_type(self, cuda_type, mapped_type):
        base_type = cuda_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = cuda_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def unqualified_leading_cuda_std_type_name(self, cuda_type):
        normalized_type = str(cuda_type)
        prefix = "::cuda::std::"
        if normalized_type.startswith(prefix):
            return normalized_type[len(prefix) :]
        return None

    def convert_nested_scoped_template_type_to_crossgl(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not template_args or "::" not in str(cuda_type):
            return None
        if not any(
            self.template_arg_contains_scoped_template(arg) for arg in template_args
        ):
            return None

        return self.convert_scoped_template_type_to_crossgl_generic(
            base_name, template_args
        )

    def convert_scoped_template_type_to_crossgl_generic(self, base_name, template_args):
        converted_args = [
            self.convert_scoped_template_argument_to_crossgl(arg)
            for arg in template_args
        ]
        return (
            f"{self.sanitize_crossgl_type_identifier(base_name)}"
            f"<{', '.join(converted_args)}>"
        )

    def convert_scoped_template_argument_to_crossgl(self, type_name):
        base_name, template_args = self.parse_cpp_template(type_name)
        if template_args and "::" in base_name:
            return self.convert_scoped_template_type_to_crossgl_generic(
                base_name, template_args
            )
        return self.convert_cuda_type_to_crossgl(type_name)

    def template_arg_contains_scoped_template(self, type_name):
        base_name, template_args = self.parse_cpp_template(type_name)
        return bool(template_args and "::" in base_name)

    def strip_dependent_template_disambiguators(self, type_name):
        return str(type_name).replace("::template ", "::")

    def convert_cuda_expression_template_type_to_crossgl(self, cuda_type):
        _, template_args = self.parse_cpp_template(cuda_type)
        if not template_args:
            return None
        if not any(
            self.is_crossgl_unsafe_template_argument(arg) for arg in template_args
        ):
            return None
        return self.convert_unknown_cuda_template_type_to_crossgl(cuda_type)

    def is_crossgl_unsafe_template_argument(self, arg):
        return re.search(r"[()+\-*/%~!=|&?]", str(arg)) is not None

    def is_decltype_type_name(self, type_name):
        return isinstance(type_name, str) and type_name.strip().startswith("decltype(")

    def convert_cooperative_group_type(self, cuda_type):
        metadata = self.cooperative_group_metadata_from_type(cuda_type)
        if metadata is None:
            return None
        group_kind = metadata["kind"]
        if group_kind == "thread_block_tile" and metadata.get("tile_size"):
            return f"cooperative_groups_thread_block_tile_{metadata['tile_size']}"
        return f"cooperative_groups_{group_kind}"

    def format_crossgl_type_syntax(self, type_name):
        return str(type_name).replace(">>", "> >")

    def crossgl_safe_generic_type_argument(self, type_name):
        if "::" in str(type_name):
            return self.sanitize_crossgl_type_identifier(type_name)
        return type_name

    def convert_cuda_resource_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name == "texture" and len(template_args) >= 2:
            return self.CUDA_TEXTURE_TYPE_MAPPING.get(template_args[1])
        if base_name == "surface" and len(template_args) >= 2:
            return self.CUDA_SURFACE_TYPE_MAPPING.get(template_args[1])
        return None

    def convert_unique_ptr_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not self.is_unique_ptr_base_name(base_name) or not template_args:
            return None

        target_type, _ = self.unwrap_array_template_type(template_args[0])
        return f"ptr<{self.convert_cuda_type_to_crossgl(target_type)}>"

    def convert_cuda_span_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name not in {"cuda::std::span", "std::span"} or not template_args:
            return None

        element_type = self.convert_cuda_type_to_crossgl(template_args[0])
        if len(template_args) >= 2 and not self.is_dynamic_span_extent(
            template_args[1]
        ):
            extent = self.format_crossgl_array_extent(template_args[1])
            return f"array<{element_type}, {extent}>"
        return f"array<{element_type}>"

    def convert_cuda_std_array_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name.startswith("::"):
            base_name = base_name[2:]
        if base_name != "cuda::std::array" or not template_args:
            return None

        element_type = self.convert_cuda_type_to_crossgl(template_args[0])
        if len(template_args) >= 2 and template_args[1].strip():
            extent = self.format_crossgl_array_extent(template_args[1])
            return f"array<{element_type}, {extent}>"
        return f"array<{element_type}>"

    def convert_enable_if_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name in {"std::enable_if_t", "::std::enable_if_t"} and template_args:
            return "void"
        return None

    def convert_cuda_wmma_type(self, cuda_type):
        normalized_type = self.resolve_namespace_alias_name(str(cuda_type).strip())
        base_name, template_args = self.parse_cpp_template(normalized_type)
        if not self.is_cuda_wmma_type_name(base_name):
            return None

        if not template_args:
            return self.sanitize_crossgl_type_identifier(normalized_type)

        converted_parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_cuda_type_to_crossgl(arg)
            converted_parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in converted_parts if part)

    def is_cuda_wmma_type_name(self, type_name):
        return str(type_name).startswith(("wmma::", "nvcuda::wmma::"))

    def is_dynamic_span_extent(self, extent):
        extent = self.resolve_namespace_alias_name(str(extent).strip())
        if extent.startswith("::"):
            extent = extent[2:]
        return extent in {
            "dynamic_extent",
            "std::dynamic_extent",
            "cuda::std::dynamic_extent",
        }

    def is_unique_ptr_type_name(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        type_name = self.resolve_type_alias(type_name)
        base_name, template_args = self.parse_cpp_template(type_name)
        return self.is_unique_ptr_base_name(base_name) and bool(template_args)

    def is_unique_ptr_base_name(self, base_name):
        return self.is_std_unique_ptr_base_name(base_name)

    def is_std_unique_ptr_base_name(self, base_name):
        return base_name in {"unique_ptr", "std::unique_ptr"}

    def is_std_make_unique_base_name(self, base_name):
        return base_name in {"make_unique", "std::make_unique"}

    def has_array_suffix(self, type_name):
        depth = 0
        for char in str(type_name):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "[" and depth == 0:
                return True
        return False

    def unwrap_array_template_type(self, type_name):
        type_name = type_name.strip()
        if type_name.endswith("[]"):
            return type_name[:-2].strip(), True
        return type_name, False

    def parse_cpp_template(self, text):
        if not isinstance(text, str):
            return str(text), []

        start = text.find("<")
        if start == -1 or not text.endswith(">"):
            return self.resolve_namespace_alias_name(text), []

        base_name = self.resolve_namespace_alias_name(text[:start].strip())
        args = self.split_cpp_template_args(text[start + 1 : -1])
        return base_name, args

    def resolve_namespace_alias_name(self, name):
        if not isinstance(name, str) or not self.namespace_aliases:
            return name

        resolved = name
        for _ in range(len(self.namespace_aliases)):
            next_resolved = self.resolve_namespace_alias_name_once(resolved)
            if next_resolved == resolved:
                break
            resolved = next_resolved
        return resolved

    def resolve_namespace_alias_name_once(self, name):
        leading_scope = name.startswith("::")
        lookup_name = name[2:] if leading_scope else name
        alias, separator, suffix = lookup_name.partition("::")
        if not separator:
            return name

        target = self.namespace_aliases.get(alias)
        if target is None:
            return name

        resolved = f"{target}::{suffix}"
        if leading_scope and not resolved.startswith("::"):
            resolved = f"::{resolved}"
        return resolved

    def split_cpp_template_args(self, args_text):
        args = []
        depth = 0
        start = 0

        for index, char in enumerate(args_text):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(args_text[start:index].strip())
                start = index + 1

        tail = args_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    def convert_cuda_pointer_type(self, cuda_type):
        pointer_depth = cuda_type.count("*")
        base_type = cuda_type.replace("*", "").strip()
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_cuda_type_to_crossgl(base_type)

        for _ in range(pointer_depth):
            mapped_type = self.wrap_crossgl_pointer_type(mapped_type)

        return mapped_type

    def convert_cuda_pointer_element_type(self, cuda_type):
        pointer_depth = cuda_type.count("*")
        base_type = cuda_type.replace("*", "").strip()
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_cuda_type_to_crossgl(base_type)

        for _ in range(max(0, pointer_depth - 1)):
            mapped_type = self.wrap_crossgl_pointer_type(mapped_type)

        return mapped_type

    def strip_function_pointer_parameter_list(self, type_name):
        """Keep imported C++ function-pointer types reparsable in CrossGL."""
        return re.sub(r"\s*\(\s*\)\s*$", "", str(type_name)).strip()

    def wrap_crossgl_pointer_type(self, mapped_type):
        element_type = self.crossgl_safe_generic_type_argument(mapped_type)
        return f"ptr<{element_type}>"

    def strip_type_qualifiers(self, type_name):
        qualifiers = {
            "const",
            "volatile",
            "__restrict__",
            "__restrict",
            "restrict",
            "__grid_constant__",
            "typename",
            "&",
            "&&",
        }
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def convert_cuda_array_type(self, cuda_type, type_mapping):
        base_type = cuda_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = cuda_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        mapped_type = type_mapping.get(base_type)
        if mapped_type is None:
            mapped_type = self.convert_cuda_type_to_crossgl(base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_cuda_builtin_function(self, func_name):
        function_mapping = {
            "sqrtf": "sqrt",
            "powf": "pow",
            # The accurate sinf and cosf are bounded over the whole binary32 range, which the libm rule
            # carries; CrossGL's sin and cos have the precision of the target.
            "sinf": "__cgl_libm_sin",
            "cosf": "__cgl_libm_cos",
            "tanf": "tan",
            "sinhf": "sinh",
            "coshf": "cosh",
            "tanhf": "tanh",
            **LIBM_BINARY32_FUNCTIONS,
            "asinf": "asin",
            "acosf": "acos",
            "atanf": "atan",
            "logf": "log",
            "log2f": "log2",
            "expf": "exp",
            "__expf": "exp",
            "exp2f": "exp2",
            "fabsf": "abs",
            "rsqrtf": "inversesqrt",
            # C11 7.12.9.6 round rounds halfway cases away from zero, which the round_away rule carries;
            # rint and nearbyint round to nearest even in CUDA's only rounding mode (7.12.9.3-4).
            "roundf": "__cgl_round_away",
            "round": "__cgl_round_away",
            "rintf": "roundEven",
            "nearbyintf": "roundEven",
            "truncf": "trunc",
            "atan2f": "atan2",
            "fmaf": "fma",
            "fmodf": "fmod",
            "fminf": "fmin",
            "fmaxf": "fmax",
            "lerp": "mix",
            "floorf": "floor",
            "ceilf": "ceil",
            "sqrt": "sqrt",
            "pow": "pow",
            "sin": "__cgl_libm_sin",
            "cos": "__cgl_libm_cos",
            "tan": "tan",
            "log": "log",
            "exp": "exp",
            "fabs": "abs",
            "rsqrt": "inversesqrt",
            "rint": "roundEven",
            "nearbyint": "roundEven",
            "fmod": "fmod",
            "fmin": "fmin",
            "fmax": "fmax",
            "__threadfence": "memoryBarrier",
            "__threadfence_block": "memoryBarrier",
            "__threadfence_system": "memoryBarrier",
            "__popc": "bitCount",
            "__popcll": "bitCount",
            "floor": "floor",
            "ceil": "ceil",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "int8_t": "i8",
            "uint8_t": "u8",
            "int16_t": "i16",
            "uint16_t": "u16",
            "int32_t": "i32",
            "uint32_t": "u32",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "__half": "f16",
            "__half2": "vec2<f16>",
            "__nv_bfloat16": "bfloat16",
            "__nv_bfloat162": "vec2<bfloat16>",
            "float": "f32",
            "double": "f64",
            "size_t": "u64",
            "ptrdiff_t": "i64",
            # clock_t is long on the LP64 hosts CUDA and ROCm compile device code for (C11 7.27.1).
            "clock_t": "i64",
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
        }

        normalized_func_name = self.normalize_cuda_builtin_function_name(func_name)
        matrix_constructor = self.convert_native_matrix_helper_name_to_crossgl(
            normalized_func_name
        )
        if matrix_constructor is not None:
            return matrix_constructor
        return function_mapping.get(normalized_func_name, func_name)

    def normalize_cuda_builtin_function_name(self, func_name):
        if not isinstance(func_name, str):
            return func_name
        normalized_name = self.resolve_namespace_alias_name(func_name)
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]
        for namespace in ("cuda::std::", "std::"):
            if normalized_name.startswith(namespace):
                return normalized_name[len(namespace) :]
        return normalized_name
