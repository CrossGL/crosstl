"""Function call names, wave arguments and the identity and transport of Metal source overloads."""

import re

from ..MetalAst import (
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    VariableNode,
)
from ..namespaced_functions import (
    METAL_LIBRARY_MATH_FUNCTIONS,
    declared_in_several_namespaces,
)
from ..using_declared_library_names import using_declared_library_names
from .support import (
    _PLAIN_TYPE_NAME,
    MetalAliasTemplateResolutionError,
    MetalSourceOverloadResolutionError,
    MetalStructAliasResolutionError,
)


class MetalSourceOverloadMixin:
    """Members of ``MetalToCrossGLConverter``: function call names, wave arguments and the identity and transport of Metal source overloads."""

    def map_function_call_name(self, name, args=None, *, source_offset=None):
        source_name = name
        name = str(name).lstrip(":")
        match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", name)
        if not match:
            if str(name).rsplit("::", 1)[-1] == "copysign":
                metal_math_name = self.map_metal_math_function_name(name, args)
                if metal_math_name is not None:
                    return metal_math_name
                return self.sanitize_identifier(name)
            precise_math_name = self.map_metal_precise_math_function_name(name, args)
            if precise_math_name is not None:
                return precise_math_name
            metal_type_constructor = self.map_metal_type_constructor_name(name)
            if metal_type_constructor is not None:
                return metal_type_constructor
            metal_bit_name = self.map_metal_bit_function_name(name)
            if metal_bit_name is not None:
                return metal_bit_name
            metal_math_name = self.map_metal_math_function_name(name, args)
            if metal_math_name is not None:
                return metal_math_name
            metal_wave_name = self.map_metal_wave_function_name(
                source_name, args, source_offset=source_offset
            )
            if metal_wave_name is not None:
                return metal_wave_name
            generic_call_name = self.map_generic_function_call_name(name)
            if generic_call_name is not None:
                return generic_call_name
            return self.sanitize_identifier(name)

        target_type = self.normalized_metal_type(
            self.resolve_type_alias(match.group(1))
        )
        mapped_type = self.map_type(target_type)
        alias_name = None
        if target_type.startswith("float") or mapped_type in {"float", "double"}:
            alias_name = "asfloat"
        elif target_type.startswith("uint") or mapped_type.startswith("uvec"):
            alias_name = "asuint"
        elif target_type.startswith("int") or mapped_type.startswith("ivec"):
            alias_name = "asint"

        if alias_name is not None:
            source_type = self.expression_mapped_type(args[0]) if args else None
            # These aliases preserve only 32-bit component layouts. Equal lane
            # counts alone do not make a narrow or wide scalar interchangeable.
            if (
                re.fullmatch(r"(?:float|int|uint|[iu]?vec[234])", mapped_type)
                and re.fullmatch(r"(?:float|int|uint|[iu]?vec[234])", str(source_type))
                and self.crossgl_type_shape(source_type)
                == self.crossgl_type_shape(mapped_type)
            ):
                return alias_name
        target_name = match.group(1).strip()
        inline_alias = target_name in self.local_type_alias_names or any(
            self.scalar_alias_needs_inline(alias)
            for alias in self.scalar_alias_declarations(target_name)
        )
        # Metal long and ulong keep 64-bit components, while targets read the
        # bare spellings as 32-bit aliases. Carry the fixed-width CrossGL type.
        wide_alias = re.fullmatch(r"u?long[234]?", target_type) is not None
        if (
            alias_name is not None
            or wide_alias
            or (
                inline_alias
                and self.crossgl_typedef_source_type(mapped_type) is not None
            )
        ):
            return f"as_type<{mapped_type}>"
        return name

    def metal_numeric_limits_expression(self, name, args=None):
        """Lower a concrete Metal numeric_limits call to an exact CrossGL value."""
        if args:
            return None
        match = re.fullmatch(
            r"(?:metal::)?numeric_limits<(.+)>::"
            r"(max|min|lowest|infinity|quiet_NaN|signaling_NaN|epsilon|denorm_min)",
            str(name),
        )
        if match is None:
            return None

        source_type, operation = match.groups()
        resolved_type = self.normalized_metal_type(
            self.resolve_type_alias(source_type.strip())
        )
        mapped_type = self.map_type(resolved_type)

        float_bits = {
            "max": "0x7f7fffffu",
            "min": "0x00800000u",
            "lowest": "0xff7fffffu",
            "infinity": "0x7f800000u",
            "quiet_NaN": "0x7fc00000u",
            "signaling_NaN": "0x7f800001u",
            "epsilon": "0x34000000u",
            "denorm_min": "0x00000001u",
        }
        if resolved_type in {"float", "float32_t"}:
            return f"asfloat({float_bits[operation]})"

        if resolved_type in {"half", "xhalf", "float16_t"}:
            half_values = {
                "max": "65504.0",
                "min": "0.00006103515625",
                "lowest": "-65504.0",
                "infinity": f"asfloat({float_bits['infinity']})",
                "quiet_NaN": f"asfloat({float_bits['quiet_NaN']})",
                "signaling_NaN": f"asfloat({float_bits['signaling_NaN']})",
                "epsilon": "0.0009765625",
                "denorm_min": "0.000000059604644775390625",
            }
            return f"{mapped_type}({half_values[operation]})"

        if resolved_type in {"bfloat", "bfloat16", "bfloat16_t"}:
            bfloat_bits = {
                "max": "0x7f7f0000u",
                "min": "0x00800000u",
                "lowest": "0xff7f0000u",
                "infinity": "0x7f800000u",
                "quiet_NaN": "0x7fc00000u",
                "signaling_NaN": "0x7f810000u",
                "epsilon": "0x3c000000u",
                "denorm_min": "0x00010000u",
            }
            return f"{mapped_type}(asfloat({bfloat_bits[operation]}))"

        if resolved_type in {"double", "float64_t"}:
            double_values = {
                "max": "1.7976931348623157e308",
                "min": "2.2250738585072014e-308",
                "lowest": "-1.7976931348623157e308",
                "infinity": f"double(asfloat({float_bits['infinity']}))",
                "quiet_NaN": f"double(asfloat({float_bits['quiet_NaN']}))",
                "signaling_NaN": f"double(asfloat({float_bits['signaling_NaN']}))",
                "epsilon": "2.2204460492503131e-16",
                "denorm_min": "4.9406564584124654e-324",
            }
            value = double_values[operation]
            return value if value.startswith("double(") else f"double({value})"

        integer_ranges = {
            "char": ("-128", "127"),
            "int8_t": ("-128", "127"),
            "short": ("-32768", "32767"),
            "int16_t": ("-32768", "32767"),
            "int": ("-2147483648", "2147483647"),
            "int32_t": ("-2147483648", "2147483647"),
            "long": ("-9223372036854775808", "9223372036854775807"),
            "int64_t": ("-9223372036854775808", "9223372036854775807"),
            "uchar": ("0", "255"),
            "uint8_t": ("0", "255"),
            "ushort": ("0", "65535"),
            "uint16_t": ("0", "65535"),
            "uint": ("0", "4294967295"),
            "uint32_t": ("0", "4294967295"),
            "ulong": ("0", "18446744073709551615"),
            "uint64_t": ("0", "18446744073709551615"),
            "size_t": ("0", "18446744073709551615"),
        }
        if resolved_type in integer_ranges and operation in {"min", "lowest", "max"}:
            minimum, maximum = integer_ranges[resolved_type]
            value = maximum if operation == "max" else minimum
            if resolved_type in {"ulong", "uint64_t", "size_t"}:
                value += "ul"
            elif resolved_type in {"long", "int64_t"}:
                # The positive magnitude of LONG_MIN is not a signed literal.
                value = maximum + "l" if operation == "max" else f"(-{maximum}l - 1l)"
            return f"{mapped_type}({value})"
        if resolved_type == "bool" and operation in {"min", "lowest", "max"}:
            return "true" if operation == "max" else "false"
        return None

    def crossgl_type_shape(self, type_name):
        vector_match = re.fullmatch(r"(?:[a-zA-Z0-9_]*vec)([234])", str(type_name))
        return int(vector_match.group(1)) if vector_match else 1

    def map_metal_bit_function_name(self, name):
        text = str(name)
        if text.startswith("metal::"):
            return self.metal_bit_intrinsics.get(text.split("::")[-1])
        if text in self.user_function_names:
            return None
        return self.metal_bit_intrinsics.get(text)

    def generate_metal_wave_argument(
        self, source_name, mapped_name, expression, index, is_main=False
    ):
        argument = expression.args[index]
        if self.binds_reference_argument(source_name, expression, index):
            rendered = self.generate_with_structured_buffer_lvalue(
                argument, lambda: self.generate_expression(argument, is_main)
            )
        else:
            rendered = self.generate_expression(argument, is_main)
        source_name = str(source_name).lstrip(":")
        if source_name.startswith("metal::"):
            source_name = source_name[len("metal::") :]
        if mapped_name != self.metal_wave_intrinsics.get(
            source_name
        ) or index not in self.metal_wave_ushort_arguments.get(source_name, ()):
            return rendered
        argument_type = self.metal_source_overload_value_type(
            self.expression_metal_type(argument)
        )
        descriptor = self.metal_source_overload_type_descriptor(argument_type)
        if (
            self.metal_scalar_arithmetic_type_info(argument_type) is None
            or descriptor is None
            or descriptor[0] in {"pointer", "vector"}
        ):
            raise MetalSourceOverloadResolutionError(
                str(expression.name),
                [
                    self.expression_metal_type(arg) or "<unknown>"
                    for arg in expression.args
                ],
                (),
                f"builtin argument {index + 1} requires scalar conversion to ushort",
                getattr(expression, "source_location", None),
            )
        # Retain the source conversion even on targets that widen ushort storage.
        return f"(uint({rendered}) & 65535u)"

    def map_metal_wave_function_name(self, name, args=None, *, source_offset=None):
        text = str(name).lstrip(":")
        if text.startswith("metal::"):
            return self.metal_wave_intrinsics.get(text.split("::")[-1])
        mapped = self.metal_wave_intrinsics.get(text)
        if mapped is None:
            return None
        binding, function = self.resolve_metal_user_function_overload(
            name,
            args or [],
            allow_wave_lane_conversion=True,
            source_offset=source_offset,
        )
        if binding == "user" and self.is_materialized_metal_stdlib_wrapper(function):
            return mapped
        if binding in {"user", "unknown"}:
            return None
        return mapped

    def metal_source_overload_parameter_type(self, parameter):
        metal_type = self.metal_declaration_expression_type(parameter)
        if metal_type is None:
            return None
        context = self.parameter_type_contexts.get(id(parameter))
        if context is not None:
            base = str(metal_type).rstrip("*&").strip()
            suffix = str(metal_type)[len(base) :]
            binding = self.scalar_alias_binding(base, context=context)
            if binding is not None:
                return f"{binding[0]}{suffix}"
        resolved = self.resolve_local_type_aliases(metal_type)
        resolved = self.substitute_template_type_text(resolved)
        resolved = self.resolve_type_alias(resolved)
        return re.sub(r"\s+", " ", str(resolved).strip())

    def metal_source_overload_value_type(self, metal_type):
        if metal_type is None:
            return None
        plain = str(metal_type).strip()
        if (
            _PLAIN_TYPE_NAME.fullmatch(plain)
            and plain not in self.local_type_alias_names
            and plain not in self.type_aliases
            and plain not in self.scalar_alias_names
            and not any(plain in scope for scope in self.template_type_bindings)
        ):
            # No alias table or template binding can rewrite this name.
            return None if plain == "auto" else plain
        resolved = self.resolve_local_type_aliases(metal_type)
        resolved = self.substitute_template_type_text(resolved)
        resolved = self.resolve_type_alias(resolved)
        text = re.sub(r"\s+", " ", str(resolved).strip())
        while text.endswith("&"):
            text = text[:-1].strip()
        text = re.sub(
            r"^(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)+",
            "",
            text,
        )
        if not text or text == "auto":
            return None
        if "::" in text:
            # ``ops::P`` names the struct a parameter declared inside ``namespace ops`` spells ``P``; the struct
            # table knows it by that unqualified name when no other namespace declares one.
            text = self.canonical_concrete_struct_alias_target(text) or text
        return text

    def metal_source_overload_type_descriptor(self, metal_type):
        text = self.metal_source_overload_value_type(metal_type)
        if text is None:
            return None

        indirection = 0
        while True:
            array_match = re.fullmatch(r"(.+?)\s*\[[^\[\]]*\]", text)
            if array_match is None:
                break
            indirection += 1
            text = array_match.group(1).strip()
        while text.endswith("*"):
            indirection += 1
            text = text[:-1].strip()

        if indirection:
            pointee = self.metal_source_overload_type_descriptor(text)
            return ("pointer", indirection, pointee or ("object", text))

        vector_parts = self.metal_small_vector_type_parts(text)
        if vector_parts is not None:
            element_type, width = vector_parts
            element = self.metal_source_overload_type_descriptor(element_type)
            return ("vector", width, element or ("object", element_type))

        normalized = self.normalized_metal_type(text)
        if normalized in self.metal_source_bfloat_types:
            return ("scalar", "floating", True, 16)
        scalar = self.metal_scalar_arithmetic_types.get(normalized)
        if scalar is not None:
            family, signed, bits = scalar
            return ("scalar", family, signed, bits)
        return ("object", normalized)

    def metal_source_overload_type_identity(self, metal_type):
        """Return a canonical source identity without erasing float formats."""
        text = self.metal_source_overload_value_type(metal_type)
        if text is None:
            return None

        indirection = 0
        while True:
            array_match = re.fullmatch(r"(.+?)\s*\[[^\[\]]*\]", text)
            if array_match is None:
                break
            indirection += 1
            text = array_match.group(1).strip()
        while text.endswith("*"):
            indirection += 1
            text = text[:-1].strip()

        if indirection:
            pointee = self.metal_source_overload_type_identity(text)
            return ("pointer", indirection, pointee or ("object", text))

        vector_parts = self.metal_small_vector_type_parts(text)
        if vector_parts is not None:
            element_type, width = vector_parts
            element = self.metal_source_overload_type_identity(element_type)
            return ("vector", width, element or ("object", element_type))

        normalized = self.normalized_metal_type(text)
        bfloat_vector = re.fullmatch(
            r"(?:bfloat|bfloat16|bfloat16_t)([234])", normalized
        )
        if bfloat_vector is not None:
            return (
                "vector",
                int(bfloat_vector.group(1)),
                ("scalar", "bfloat", True, 16),
            )
        if normalized in self.metal_source_bfloat_types:
            return ("scalar", "bfloat", True, 16)
        if normalized == "bool":
            return ("scalar", "bool", None, 1)

        descriptor = self.metal_source_overload_type_descriptor(text)
        return descriptor or ("object", normalized)

    def metal_source_overload_portable_type_identity(self, metal_type):
        """Return the type shape shared by baseline HLSL and OpenGL ABIs."""
        identity = self.metal_source_overload_type_identity(metal_type)
        if identity is None:
            return None
        return self.metal_source_overload_portable_identity(identity)

    @classmethod
    def metal_source_overload_portable_identity(cls, identity):
        if not identity:
            return identity
        if identity[0] in {"pointer", "vector"}:
            return (
                identity[0],
                identity[1],
                cls.metal_source_overload_portable_identity(identity[2]),
            )
        if identity[0] != "scalar":
            return identity
        _kind, family, signed, bits = identity
        if family == "bool":
            return identity
        if family in {"floating", "bfloat"}:
            return ("scalar", "floating", True, 32 if bits <= 32 else bits)
        if family == "integer":
            return ("scalar", family, signed, 32 if bits <= 32 else bits)
        return identity

    def metal_source_overload_signature(self, function):
        signature = []
        for parameter in getattr(function, "params", []) or []:
            parameter_type = self.metal_source_overload_parameter_type(parameter)
            descriptor = self.metal_source_overload_type_descriptor(parameter_type)
            signature.append(descriptor or ("unknown", parameter_type or ""))
        return tuple(signature)

    def metal_source_overload_parameter_identity(self, parameter):
        parameter_type = self.metal_source_overload_parameter_type(parameter)
        descriptor = self.metal_source_overload_type_identity(parameter_type)
        descriptor = descriptor or ("unknown", parameter_type or "")
        reference_depth = 0
        type_without_references = str(parameter_type or "").rstrip()
        while type_without_references.endswith("&"):
            reference_depth += 1
            type_without_references = type_without_references[:-1].rstrip()
        qualifiers = self.metal_declaration_type_qualifiers(parameter)
        if descriptor[0] != "pointer" and reference_depth == 0:
            qualifiers = ()
        return descriptor, reference_depth, qualifiers

    def metal_source_overload_portable_parameter_identity(self, parameter):
        parameter_type = self.metal_source_overload_parameter_type(parameter)
        descriptor = self.metal_source_overload_portable_type_identity(parameter_type)
        descriptor = descriptor or ("unknown", parameter_type or "")
        reference_depth = 0
        type_without_references = str(parameter_type or "").rstrip()
        while type_without_references.endswith("&"):
            reference_depth += 1
            type_without_references = type_without_references[:-1].rstrip()
        qualifiers = self.metal_declaration_type_qualifiers(parameter)
        if descriptor[0] != "pointer" and reference_depth == 0:
            qualifiers = ()
        return descriptor, reference_depth, qualifiers

    def metal_source_overload_declaration_signature(self, function):
        return tuple(
            self.metal_source_overload_parameter_identity(parameter)
            for parameter in getattr(function, "params", []) or []
        )

    def metal_source_overload_portable_signature(self, function):
        return tuple(
            self.metal_source_overload_portable_parameter_identity(parameter)
            for parameter in getattr(function, "params", []) or []
        )

    def metal_source_overload_candidate_signature(self, function):
        parameters = []
        for parameter in getattr(function, "params", []) or []:
            parameter_type = (
                self.metal_source_overload_parameter_type(parameter) or "<unknown>"
            )
            qualifiers = self.metal_declaration_type_qualifiers(parameter)
            parameters.append(" ".join((*qualifiers, parameter_type)))
        return f"{function.name}({', '.join(parameters)})"

    def metal_source_overload_set_has_portable_collision(self, signature_groups):
        portable_signatures = {}
        for source_signature, declarations in signature_groups.items():
            portable_signature = self.metal_source_overload_portable_signature(
                declarations[0]
            )
            portable_signatures.setdefault(portable_signature, set()).add(
                source_signature
            )
        return any(
            len(source_signatures) > 1
            for source_signatures in portable_signatures.values()
        )

    def metal_source_overload_set_requires_transport(self, signature_groups):
        functions = [declarations[0] for declarations in signature_groups.values()]
        if self.metal_source_overload_set_has_portable_collision(signature_groups):
            return True

        by_arity = {}
        for function in functions:
            by_arity.setdefault(len(getattr(function, "params", []) or []), []).append(
                function
            )
        for candidates in by_arity.values():
            if len(candidates) < 2:
                continue
            signatures = [
                self.metal_source_overload_signature(candidate)
                for candidate in candidates
            ]
            first_signature = signatures[0]
            common_pointer_parameter = any(
                descriptor[0] == "pointer"
                and all(
                    candidate_signature[parameter_index] == descriptor
                    for candidate_signature in signatures[1:]
                )
                for parameter_index, descriptor in enumerate(first_signature)
            )
            if not common_pointer_parameter:
                continue
            for index in range(len(first_signature)):
                shapes = {signature[index][0] for signature in signatures}
                if {"scalar", "vector"}.issubset(shapes):
                    return True
        return False

    def qualified_metal_math_operation(self, name):
        name = str(name).lstrip(":")
        for prefix in self.metal_math_namespace_prefixes:
            if name.startswith(prefix):
                operation = name[len(prefix) :]
                return operation if operation in self.metal_math_intrinsics else None
        return None

    def metal_qualified_math_call_names(self, root):
        calls = set()
        seen = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if id(node) in seen:
                return
            seen.add(id(node))
            if isinstance(node, FunctionCallNode):
                name = str(node.name).lstrip(":")
                operation = self.qualified_metal_math_operation(name)
                if operation is not None:
                    calls.add(operation)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return calls

    def metal_library_function_names(self):
        """Unqualified names of the Metal library functions the importer lowers.

        User overloads of these names extend the library overload set (``metal::simd_shuffle_down(bfloat, ushort)``
        next to the library template), so a call that no user overload matches still binds to the library function.
        Splitting them per namespace would hide the library candidates, so they keep the library-collision path.
        """
        return (
            set(self.metal_math_intrinsics)
            | set(self.metal_bit_intrinsics)
            | set(self.metal_wave_intrinsics)
            | set(self.metal_atomic_intrinsics)
            | METAL_LIBRARY_MATH_FUNCTIONS
        )

    def prepare_metal_source_overload_transport(self, ast):
        """Assign internal names where portable target ABIs erase source identity."""
        self.metal_source_overload_groups = {}
        self.metal_source_overload_output_names = {}
        self.metal_builtin_name_collision_groups = set()
        qualified_math_calls = self.metal_qualified_math_call_names(ast)
        # These portable names are emitted by as_type and arithmetic helpers,
        # even without a qualified builtin call in the source AST.
        builtin_names = qualified_math_calls | {"asfloat", "asint", "asuint"}
        self.metal_using_declared_library_names = using_declared_library_names(
            ast, self.metal_library_function_names()
        )
        builtin_names |= self.metal_using_declared_library_names
        used_names = set(self.wide_vector_reserved_names)
        used_names.update(
            self.sanitize_identifier(name)
            for name in getattr(self, "existing_function_names", set())
        )

        for function_name, overloads in self.user_function_overloads_by_name.items():
            builtin_collision = (
                function_name in builtin_names
                and any(
                    not self.is_materialized_metal_stdlib_wrapper(function)
                    for function in overloads
                )
                or function_name not in self.metal_library_function_names()
                and declared_in_several_namespaces(overloads)
            )
            if builtin_collision:
                self.metal_builtin_name_collision_groups.add(function_name)
            signature_groups = {}
            for function in overloads:
                signature = self.metal_source_overload_declaration_signature(function)
                if builtin_collision:
                    signature = (
                        str(getattr(function, "qualified_name", function.name)),
                        signature,
                    )
                signature_groups.setdefault(signature, []).append(function)
            if len(signature_groups) < 2 and not builtin_collision:
                continue
            portable_collision = self.metal_source_overload_set_has_portable_collision(
                signature_groups
            )
            if (
                not builtin_collision
                and not self.metal_source_overload_set_requires_transport(
                    signature_groups
                )
            ):
                continue

            self.metal_source_overload_groups[function_name] = signature_groups
            ordered_groups = (
                list(signature_groups.items())
                if portable_collision
                else sorted(signature_groups.items(), key=lambda item: repr(item[0]))
            )
            for ordinal, (_signature, declarations) in enumerate(
                ordered_groups, start=1
            ):
                if ordinal == 1 and not portable_collision and not builtin_collision:
                    output_name = function_name
                else:
                    base = (
                        f"{self.sanitize_identifier(function_name)}"
                        f"__metal_overload_{ordinal}"
                    )
                    output_name = base
                    suffix = 2
                    while output_name in used_names:
                        output_name = f"{base}_{suffix}"
                        suffix += 1
                sanitized_output_name = self.sanitize_identifier(output_name)
                used_names.add(sanitized_output_name)
                self.wide_vector_reserved_names.add(sanitized_output_name)
                self.existing_function_names.add(sanitized_output_name)
                self.user_function_names.add(output_name)
                for declaration in declarations:
                    self.metal_source_overload_output_names[id(declaration)] = (
                        output_name
                    )

    def metal_source_overload_pointer_qualifier_rank(self, argument, parameter):
        actual_qualifiers = set(self.expression_metal_type_qualifiers(argument))
        parameter_qualifiers = set(self.metal_declaration_type_qualifiers(parameter))

        def address_space(qualifiers):
            selected = qualifiers & self.metal_source_overload_address_spaces
            return selected or {"thread"}

        if address_space(actual_qualifiers) != address_space(parameter_qualifiers):
            return None
        actual_cv = actual_qualifiers & {"const", "volatile"}
        parameter_cv = parameter_qualifiers & {"const", "volatile"}
        if not actual_cv.issubset(parameter_cv):
            return None
        return 3 if actual_cv == parameter_cv else 2

    @classmethod
    def metal_source_overload_identity_contains_bfloat(cls, identity):
        if not identity:
            return False
        if identity[0] in {"pointer", "vector"}:
            return cls.metal_source_overload_identity_contains_bfloat(identity[2])
        return identity[0] == "scalar" and identity[1] == "bfloat"

    def metal_source_overload_initializer_list_match_rank(self, argument, parameter):
        parameter_type = self.metal_source_overload_parameter_type(parameter)
        expected = self.metal_source_overload_type_descriptor(parameter_type)
        if expected is None or expected[0] != "vector":
            return None

        elements = list(argument.elements)
        expected_width = expected[1]
        if len(elements) != expected_width or any(
            isinstance(element, (InitializerListNode, DesignatedInitializerNode))
            for element in elements
        ):
            return None

        parameter_value_type = self.metal_source_overload_value_type(parameter_type)
        vector_parts = self.metal_small_vector_type_parts(parameter_value_type)
        if vector_parts is None or vector_parts[1] != expected_width:
            return None
        component_type, _width = vector_parts
        component_parameter = VariableNode(
            component_type,
            "_crosstl_initializer_component",
        )

        ranks = []
        for element in elements:
            element_type = self.metal_source_overload_value_type(
                self.expression_metal_type(element)
            )
            if element_type is None:
                return None
            rank = self.metal_source_overload_argument_match_rank(
                element,
                element_type,
                component_parameter,
            )
            if rank is None:
                return None
            ranks.append(rank)
        return min(ranks)

    def metal_source_overload_argument_match_rank(
        self, argument, actual_type, parameter
    ):
        if isinstance(argument, InitializerListNode):
            return self.metal_source_overload_initializer_list_match_rank(
                argument,
                parameter,
            )
        actual = self.metal_source_overload_type_descriptor(actual_type)
        parameter_type = self.metal_source_overload_parameter_type(parameter)
        expected = self.metal_source_overload_type_descriptor(parameter_type)
        if actual is None or expected is None:
            return None
        actual_identity = self.metal_source_overload_type_identity(actual_type)
        expected_identity = self.metal_source_overload_type_identity(parameter_type)
        if actual_identity == expected_identity:
            if actual[0] == "pointer":
                pointer_rank = self.metal_source_overload_pointer_qualifier_rank(
                    argument, parameter
                )
                return pointer_rank + 1 if pointer_rank is not None else None
            return 4
        # Native Metal does not implicitly convert half/float values to bfloat,
        # even though those formats share a 16-bit descriptor after portable
        # normalization. Keep bfloat destination overloads exact; a bfloat
        # source may still widen to float, matching the native language.
        if self.metal_source_overload_identity_contains_bfloat(expected_identity):
            return None
        if actual == expected:
            if actual[0] == "pointer":
                return self.metal_source_overload_pointer_qualifier_rank(
                    argument, parameter
                )
            return 3
        if actual[0] != "scalar" or expected[0] != "scalar":
            return None

        _actual_kind, actual_family, _actual_signed, actual_bits = actual
        _expected_kind, expected_family, expected_signed, expected_bits = expected
        if actual_family == expected_family == "integer":
            if actual_bits < 32 and expected_bits == 32 and expected_signed:
                return 2
            return 1
        if (
            actual_family == expected_family == "floating"
            and actual_bits == 32
            and expected_bits == 64
        ):
            return 2
        if actual_family in {"integer", "floating"} and expected_family in {
            "integer",
            "floating",
        }:
            return 1
        return None

    def metal_lowered_readonly_scalar_conversion(self, metal_type):
        """Return one generated readonly aggregate-to-scalar conversion helper.

        Struct-member lowering transports a Metal conversion operator into a
        free helper named ``Struct__operator_*``. Once the member is removed,
        built-in expressions must call that helper explicitly. Admit only one
        exact receiver helper with a readonly thread reference and a native
        arithmetic scalar result; ambiguous or mutating conversions remain
        unsupported.
        """

        candidates = self.metal_lowered_readonly_scalar_conversions(metal_type)
        return candidates[0] if len(candidates) == 1 else None

    def metal_lowered_readonly_scalar_conversions(self, metal_type):
        """Return every generated readonly aggregate-to-scalar conversion helper.

        Each entry is ``(function, result_type)`` for one lowered
        ``Struct__operator_*`` helper that takes the aggregate by readonly
        thread reference and returns a native arithmetic scalar, in source
        declaration order.
        """

        value_type = self.metal_source_overload_value_type(metal_type)
        if value_type is None:
            return []
        resolved_type = self.normalized_metal_type(value_type)
        if resolved_type not in self.struct_member_types:
            return []
        prefix = f"{self.sanitize_identifier(resolved_type)}__operator_"
        candidates = []
        for function_name, overloads in self.user_function_overloads_by_name.items():
            if not str(function_name).startswith(prefix):
                continue
            for function in overloads:
                parameters = list(getattr(function, "params", []) or [])
                if len(parameters) != 1:
                    continue
                parameter = parameters[0]
                parameter_type = self.metal_source_overload_value_type(
                    self.metal_source_overload_parameter_type(parameter)
                )
                if self.normalized_metal_type(parameter_type) != resolved_type:
                    continue
                qualifiers = {
                    str(qualifier).lower()
                    for qualifier in getattr(parameter, "qualifiers", []) or []
                }
                if "const" not in qualifiers or "thread" not in qualifiers:
                    continue
                result_type = self.metal_source_overload_value_type(
                    getattr(function, "return_type", None)
                )
                result_descriptor = self.metal_source_overload_type_descriptor(
                    result_type
                )
                if result_descriptor is None or result_descriptor[0] != "scalar":
                    continue
                candidate_name = str(getattr(function, "name", function_name))
                conversion_suffix = (
                    candidate_name[len(prefix) :]
                    if candidate_name.startswith(prefix)
                    else ""
                )
                result_spellings = {
                    str(result_type or ""),
                    self.normalized_metal_type(result_type),
                    self.normalized_metal_type(self.resolve_type_alias(result_type)),
                }
                expected_suffixes = {
                    re.sub(r"[^A-Za-z0-9_]+", "_", spelling).strip("_")
                    for spelling in result_spellings
                    if spelling
                }
                if conversion_suffix not in expected_suffixes:
                    previous_context = self.current_type_resolution_context
                    self.current_type_resolution_context = function
                    try:
                        alias_result = self.resolve_dependent_alias_type(
                            resolved_type,
                            conversion_suffix,
                            required=False,
                        )
                    except (
                        MetalAliasTemplateResolutionError,
                        MetalStructAliasResolutionError,
                    ):
                        continue
                    finally:
                        self.current_type_resolution_context = previous_context
                    if alias_result is None or (
                        self.metal_source_overload_type_identity(alias_result)
                        != self.metal_source_overload_type_identity(result_type)
                    ):
                        continue
                candidates.append((function, result_type))
        return candidates

    def metal_builtin_lowered_conversion_plan(
        self,
        operator,
        arguments,
        argument_types=None,
    ):
        """Prove a native scalar binary operation after one lowered conversion."""

        if len(arguments) != 2 or operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "<<",
            ">>",
            "&",
            "|",
            "^",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            return None
        if argument_types is None:
            argument_types = [
                self.metal_source_overload_value_type(
                    self.expression_metal_type(argument)
                )
                for argument in arguments
            ]
        if any(argument_type is None for argument_type in argument_types):
            return None

        aggregate_indices = [
            index
            for index, argument_type in enumerate(argument_types)
            if self.normalized_metal_type(argument_type) in self.struct_member_types
        ]
        if len(aggregate_indices) != 1:
            return None
        aggregate_index = aggregate_indices[0]
        scalar_index = 1 - aggregate_index
        scalar_descriptor = self.metal_source_overload_type_descriptor(
            argument_types[scalar_index]
        )
        if scalar_descriptor is None or scalar_descriptor[0] != "scalar":
            return None
        conversion = self.metal_lowered_readonly_scalar_conversion(
            argument_types[aggregate_index]
        )
        if conversion is None:
            return None
        _function, converted_type = conversion
        converted_types = list(argument_types)
        converted_types[aggregate_index] = converted_type
        if operator in {"==", "!=", "<", "<=", ">", ">="}:
            if any(
                (descriptor := self.metal_source_overload_type_descriptor(operand_type))
                is None
                or descriptor[0] != "scalar"
                for operand_type in converted_types
            ):
                return None
        elif (
            self.metal_scalar_binary_result_type(
                operator,
                converted_types[0],
                converted_types[1],
            )
            is None
        ):
            return None
        helpers = [None, None]
        helpers[aggregate_index] = conversion[0]
        return tuple(helpers), tuple(converted_types)

    def metal_free_operator_helper_label(self, operator, arity):
        if operator == "-" and arity == 1:
            return "negate"
        return self.metal_free_operator_helper_labels.get(operator)

    def metal_free_operator_candidates(self, operator, arity):
        label = self.metal_free_operator_helper_label(operator, arity)
        if label is None:
            return []
        prefix = f"{self.metal_free_operator_helper_prefix}{label}__"
        candidates = []
        overloads_by_name = getattr(self, "user_function_overloads_by_name", {})
        for function_name in self.metal_free_operator_helper_names(
            overloads_by_name, prefix
        ):
            candidates.extend(
                function
                for function in overloads_by_name[function_name]
                if len(getattr(function, "params", []) or []) == arity
            )
        return candidates

    def metal_free_operator_helper_names(self, overloads_by_name, prefix):
        """Return the overload names that start with ``prefix``, in mapping order.

        Every binary and unary operator asks for its helper candidates, and a
        large translation unit has thousands of functions, so scanning every
        name per operator dominated code generation. Names are only ever added
        to the overload mapping (or the mapping is replaced), so the mapping's
        identity and size identify its key set and key the cached lists.
        """
        cache = getattr(self, "_metal_free_operator_helper_name_cache", None)
        if (
            cache is None
            or cache[0] is not overloads_by_name
            or cache[1] != len(overloads_by_name)
        ):
            helper_prefix = self.metal_free_operator_helper_prefix
            cache = (
                overloads_by_name,
                len(overloads_by_name),
                [
                    name
                    for name in overloads_by_name
                    if str(name).startswith(helper_prefix)
                ],
                {},
            )
            self._metal_free_operator_helper_name_cache = cache
        names = cache[3].get(prefix)
        if names is None:
            names = cache[3][prefix] = [
                name for name in cache[2] if str(name).startswith(prefix)
            ]
        return names
