"""Type conversion, built-in function names and enums in the HIP to CrossGL converter."""

import re


class HipTypeConversionMixin:
    """Type conversion members of ``HipToCrossGLConverter``."""

    def convert_hip_variable_type_to_crossgl(self, hip_type, name):
        """Convert HIP variable types, using call-site hints for resource handles."""
        resource_type = self.convert_hip_resource_object_type(hip_type, name)
        if resource_type is not None:
            return resource_type
        return self.convert_hip_type_to_crossgl(hip_type)

    def convert_hip_resource_object_type(self, hip_type, name):
        hint = self.lookup_resource_object_type_hint(name)
        if hint is None:
            return None
        return self.convert_hip_resource_object_type_with_hint(hip_type, hint)

    def convert_hip_resource_object_type_with_hint(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)

        if self.has_array_suffix(hip_type):
            base_type = hip_type.split("[", 1)[0].strip()
            mapped_type = self.convert_hip_resource_object_type_with_hint(
                base_type, hint
            )
            if mapped_type is None:
                return None
            return self.wrap_mapped_hip_array_type(hip_type, mapped_type)

        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        if pointer_depth:
            mapped_type = self.convert_hip_resource_object_base_type(base_type, hint)
            if mapped_type is None:
                return None
            for _ in range(pointer_depth):
                mapped_type = f"ptr<{mapped_type}>"
            return mapped_type

        return self.convert_hip_resource_object_base_type(hip_type, hint)

    def convert_hip_resource_object_base_type(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)
        if hip_type == "hipTextureObject_t" and hint.startswith("sampler"):
            return hint
        if hip_type == "hipSurfaceObject_t" and "image" in hint:
            return hint
        return None

    def wrap_mapped_hip_array_type(self, hip_type, mapped_type):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

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

    def canonical_cpp_scalar_type(self, type_name):
        """``type_name`` with a namespace-qualified ``<cstdint>`` spelling (``std::uint32_t``, ``hip::std::size_t``)
        replaced by its plain spelling; any other name is already plain and is returned unchanged.
        """
        if type_name in self.CPP_SCALAR_TYPE_ALIASES:
            return self.CPP_SCALAR_TYPE_ALIASES[type_name]
        return type_name

    def is_decltype_type_name(self, type_name):
        return isinstance(type_name, str) and type_name.strip().startswith("decltype(")

    def strip_elaborated_record_type_keyword(self, hip_type):
        if not isinstance(hip_type, str):
            return hip_type

        for keyword in ("struct", "class", "union"):
            prefix = f"{keyword} "
            if not hip_type.startswith(prefix):
                continue

            record_type = hip_type[len(prefix) :].strip()
            if not record_type.startswith("<anonymous>"):
                return record_type
            if keyword == "union":
                suffix = record_type[len("<anonymous>") :]
                return f"hip_anonymous_union{suffix}"
            return hip_type

        return hip_type

    def convert_hip_resource_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if base_name == "texture" and len(template_args) >= 2:
            return self.HIP_TEXTURE_TYPE_MAPPING.get(template_args[1])
        if base_name == "surface" and len(template_args) >= 2:
            return self.HIP_SURFACE_TYPE_MAPPING.get(template_args[1])
        return None

    def convert_unique_ptr_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if not self.is_unique_ptr_base_name(base_name) or not template_args:
            return None

        target_type, _ = self.unwrap_array_template_type(template_args[0])
        return f"ptr<{self.convert_hip_type_to_crossgl(target_type)}>"

    def convert_std_container_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if self.is_std_vector_base_name(base_name) and template_args:
            element_type = self.convert_hip_type_to_crossgl(template_args[0])
            return f"array<{element_type}>"
        if self.is_std_array_base_name(base_name) and len(template_args) >= 2:
            element_type = self.convert_hip_type_to_crossgl(template_args[0])
            size = self.format_crossgl_array_extent(template_args[1])
            return f"array<{element_type}, {size}>"
        if self.is_std_pair_base_name(base_name) and len(template_args) >= 2:
            first_type = self.convert_hip_type_to_crossgl(template_args[0])
            second_type = self.convert_hip_type_to_crossgl(template_args[1])
            return f"pair<{first_type}, {second_type}>"
        if self.is_std_tuple_base_name(base_name) and template_args:
            element_types = [
                self.convert_hip_type_to_crossgl(arg) for arg in template_args
            ]
            return f"tuple<{', '.join(element_types)}>"
        return None

    def convert_enable_if_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if base_name in {"std::enable_if_t", "::std::enable_if_t"} and template_args:
            return "void"
        return None

    def is_unique_ptr_type_name(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        type_name = self.resolve_type_alias(type_name)
        base_name, template_args = self.parse_cpp_template(type_name)
        return self.is_unique_ptr_base_name(base_name) and bool(template_args)

    def is_unique_ptr_base_name(self, base_name):
        return self.is_std_unique_ptr_base_name(base_name)

    def is_std_unique_ptr_base_name(self, base_name):
        return base_name in {"unique_ptr", "std::unique_ptr"}

    def is_std_vector_base_name(self, base_name):
        return base_name in {"vector", "std::vector"}

    def is_std_array_base_name(self, base_name):
        return base_name in {"array", "std::array"}

    def is_std_pair_base_name(self, base_name):
        return base_name in {"pair", "std::pair"}

    def is_std_tuple_base_name(self, base_name):
        return base_name in {"tuple", "std::tuple"}

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

    def split_pointer_declarators(self, type_name):
        base_chars = []
        pointer_depth = 0
        template_depth = 0

        for char in str(type_name):
            if char == "<":
                template_depth += 1
                base_chars.append(char)
                continue
            if char == ">":
                template_depth = max(0, template_depth - 1)
                base_chars.append(char)
                continue
            if char == "*" and template_depth == 0:
                pointer_depth += 1
                continue
            base_chars.append(char)

        return "".join(base_chars).strip(), pointer_depth

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

    def convert_hip_pointer_type(self, hip_type):
        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(pointer_depth):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def convert_hip_pointer_element_type(self, hip_type):
        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(max(0, pointer_depth - 1)):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def is_readonly_pointer_type(self, hip_type):
        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        if pointer_depth <= 0:
            return False
        qualifiers = set(re.findall(r"\b(?:const|readonly)\b", str(base_type)))
        return bool(qualifiers)

    def strip_function_pointer_parameter_list(self, type_name):
        """Keep imported C++ function-pointer types reparsable in CrossGL."""
        return re.sub(r"\s*\(\s*\)\s*$", "", str(type_name)).strip()

    def strip_type_qualifiers(self, type_name):
        qualifiers = {"const", "volatile", "__restrict__", "restrict", "&", "&&"}
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def strip_variadic_type_marker(self, type_name):
        return " ".join(part for part in str(type_name).split() if part != "...")

    def convert_hip_array_type(self, hip_type, type_mapping):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        mapped_type = type_mapping.get(base_type)
        if mapped_type is None:
            mapped_type = self.convert_hip_type_to_crossgl(base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_hip_builtin_function(self, func_name):
        function_mapping = {
            # Math functions
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
            "asinhf": "asinh",
            "acoshf": "acosh",
            "atanhf": "atanh",
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
            # rint and nearbyint round to nearest even in the default rounding mode (7.12.9.3-4).
            "roundf": "__cgl_round_away",
            "round": "__cgl_round_away",
            "rintf": "roundEven",
            "rint": "roundEven",
            "nearbyintf": "roundEven",
            "nearbyint": "roundEven",
            "truncf": "trunc",
            "atan2f": "atan2",
            "fmaf": "fma",
            "__fmaf_rn": "fma",
            "__fma_rn": "fma",
            "fmodf": "fmod",
            "fminf": "fmin",
            "fmaxf": "fmax",
            "lerp": "mix",
            "floorf": "floor",
            "ceilf": "ceil",
            # Double precision variants
            "sqrt": "sqrt",
            "pow": "pow",
            "sin": "__cgl_libm_sin",
            "cos": "__cgl_libm_cos",
            "tan": "tan",
            "log": "log",
            "exp": "exp",
            "fabs": "abs",
            "rsqrt": "inversesqrt",
            "fmod": "fmod",
            "fmin": "fmin",
            "fmax": "fmax",
            "floor": "floor",
            "ceil": "ceil",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "__int64": "i64",
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
            "float": "f32",
            "double": "f64",
            "size_t": "u64",
            "ptrdiff_t": "i64",
            # Vector functions
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
            # Sync functions
            "__syncthreads": "workgroupBarrier",
            "__threadfence": "memoryBarrier",
            "__threadfence_block": "memoryBarrier",
            "__threadfence_system": "memoryBarrier",
            # Population count intrinsics
            "__builtin_popcount": "bitCount",
            "__builtin_popcountl": "bitCount",
            "__builtin_popcountll": "bitCount",
            "__popcnt16": "bitCount",
            "__popcnt": "bitCount",
            "__popcnt64": "bitCount",
            # Atomic functions
            "atomicAdd": "atomicAdd",
            "hipAtomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "hipAtomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "hipAtomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "hipAtomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "hipAtomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
            "hipAtomicCAS": "atomicCompareExchange",
            "atomicAnd": "atomicAnd",
            "hipAtomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "hipAtomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "hipAtomicXor": "atomicXor",
            "atomicInc": "atomicInc",
            "hipAtomicInc": "atomicInc",
            "atomicDec": "atomicDec",
            "hipAtomicDec": "atomicDec",
        }

        normalized_func_name = self.normalize_hip_builtin_function_name(func_name)
        matrix_constructor = self.convert_native_matrix_helper_name_to_crossgl(
            normalized_func_name
        )
        if matrix_constructor is not None:
            return matrix_constructor
        return function_mapping.get(normalized_func_name, func_name)

    def normalize_hip_builtin_function_name(self, func_name):
        if not isinstance(func_name, str):
            return func_name

        normalized_name = self.resolve_namespace_alias_name(func_name)
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]

        for namespace in ("hip::std::", "cuda::std::", "std::"):
            if normalized_name.startswith(namespace):
                return normalized_name[len(namespace) :]

        return normalized_name

    def visit_EnumNode(self, node):
        # An enum named ``let`` is declared as ``let_``, the name its uses get.
        name = (
            self.sanitize_identifier_name(node.name) or self.next_anonymous_enum_name()
        )
        underlying = getattr(node, "underlying_type", None)
        if getattr(node, "name", None):
            self.enum_underlying_types[node.name] = underlying or "int"
        suffix = (
            f" : {self.convert_hip_type_to_crossgl(underlying)}" if underlying else ""
        )
        self.emit(f"enum {name}{suffix} {{")
        self.indent_level += 1

        members = getattr(node, "members", None) or getattr(node, "variants", [])
        for member in members:
            if isinstance(member, tuple):
                member_name, member_value = member
            else:
                member_name = getattr(member, "name", str(member))
                member_value = getattr(member, "value", None)

            if member_value is not None:
                value = self.visit(member_value)
                member_name = self.sanitize_enum_member_name(member_name)
                self.emit(f"{member_name} = {value},")
            else:
                self.emit(f"{self.sanitize_enum_member_name(member_name)},")

        self.indent_level -= 1
        self.emit("};")

    def sanitize_enum_member_name(self, name):
        return self.sanitize_identifier_name(name)

    def next_anonymous_enum_name(self):
        name = f"anonymous_enum_{self.anonymous_enum_count}"
        self.anonymous_enum_count += 1
        return name
