"""Auto type inference and array, vector, matrix and texture type parts."""

import re

from ..dependent_builtins import (
    dependent_subscript_selection,
    unbound_template_type_names,
)
from ..generic_call_return_types import deduced_call_return_type
from ..MetalAst import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
    VariableNode,
)
from ..type_layout import metal_type_layout
from .support import (
    MetalAutoTypeInferenceError,
    MetalIndexedComponentTypeResolutionError,
    MetalWideVectorLoweringError,
)


class MetalDeclarationTypeMixin:
    """Members of ``MetalToCrossGLConverter``: auto type inference and array, vector, matrix and texture type parts."""

    def metal_declaration_expression_type(self, declaration):
        metal_type = getattr(declaration, "vtype", None)
        if metal_type is None:
            return None
        suffix = "".join(
            "[]" if size is None else f"[{self.format_array_extent(size)}]"
            for size in getattr(declaration, "array_sizes", []) or []
        )
        return f"{metal_type}{suffix}"

    def metal_declaration_type_qualifiers(self, declaration):
        qualifiers = set(self.resolved_declaration_qualifiers(declaration))
        if getattr(declaration, "is_const", False):
            qualifiers.add("const")
        return tuple(
            qualifier
            for qualifier in self.metal_source_overload_type_qualifiers
            if qualifier in qualifiers
        )

    def inferred_metal_declaration_type_qualifiers(
        self, declaration, initializer=None, inferred_type=None
    ):
        qualifiers = set(self.metal_declaration_type_qualifiers(declaration))
        if (
            self.is_metal_auto_declaration_type(getattr(declaration, "vtype", None))
            and initializer is not None
            and self.metal_pointer_pointee_type_once(inferred_type) is not None
        ):
            if not self.metal_auto_pointer_depth(declaration.vtype):
                # ``const auto p = ptr`` deduces ``T* const`` ([dcl.type.auto.deduct]): the const qualifies the
                # local pointer object, which CrossGL locals need not spell, not the storage it points to.
                qualifiers.difference_update({"const", "volatile"})
            qualifiers.update(self.expression_metal_type_qualifiers(initializer))
            if self.metal_auto_pointer_depth(declaration.vtype):
                return self.normalized_metal_address_qualifiers(
                    qualifiers,
                    getattr(declaration, "source_location", None),
                    inferred_type,
                )
        return tuple(
            qualifier
            for qualifier in self.metal_source_overload_type_qualifiers
            if qualifier in qualifiers
        )

    @staticmethod
    def is_plain_metal_auto_type(metal_type):
        return str(metal_type or "").strip() == "auto"

    @staticmethod
    def is_metal_auto_declaration_type(metal_type):
        return (
            re.fullmatch(r"auto(?:\s*\*)*", str(metal_type or "").strip()) is not None
        )

    @staticmethod
    def metal_auto_pointer_depth(metal_type):
        text = str(metal_type or "").strip()
        return text.count("*") if re.fullmatch(r"auto(?:\s*\*)+", text) else 0

    def validate_auto_pointer_initializer(self, declaration, inferred_type):
        depth = self.metal_auto_pointer_depth(declaration.vtype)
        if not depth:
            return
        remaining = inferred_type
        for _ in range(depth):
            remaining = self.metal_pointer_pointee_type_once(remaining)
        reason = None
        if remaining is None:
            reason = f"the initializer must provide at least {depth} pointer level(s)"
        elif self.is_metal_auto_declaration_type(remaining):
            reason = "the initializer pointee type remains auto"
        elif getattr(declaration, "indirection_qualifiers", ()):
            reason = "qualified auto pointer objects require distinct pointer-level qualifiers"
        if reason is not None:
            raise MetalAutoTypeInferenceError(
                declaration.name,
                None,
                inferred_type or "<unknown>",
                reason,
                getattr(declaration, "source_location", None),
            )

    def selected_metal_callable(self, expression):
        if not isinstance(expression, FunctionCallNode):
            return None
        if str(expression.name) in {"decltype", "metal::decltype"}:
            return None
        if re.fullmatch(r"(?:metal::)?as_type<(.+)>", str(expression.name)):
            return None
        if self.metal_constructor_result_type(expression.name) is not None:
            return None

        lowered_static = self.resolve_lowered_static_struct_method_call(expression)
        if lowered_static is not None:
            return lowered_static
        lowered_method = self.resolve_lowered_struct_method_call(expression)
        if lowered_method is not None:
            function, _transported = lowered_method
            return function
        source_overload = self.resolve_transported_metal_source_overload(
            str(expression.name),
            expression.args,
            getattr(expression, "source_location", None),
        )
        if source_overload is not None:
            return source_overload
        binding, function = self.resolve_metal_user_function_overload(
            str(expression.name), expression.args
        )
        return function if binding == "user" else None

    def selected_metal_callable_return_type(self, function):
        return_type = getattr(function, "return_type", None)
        if return_type is None:
            return None
        return_type = self.substitute_template_type_text(return_type)
        return_type = self.substitute_template_value_text(return_type)
        binding = self.scalar_alias_binding(return_type, context=function)
        return binding[0] if binding is not None else return_type

    def unresolved_selected_return_parameters(self, function, return_type, arguments):
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", str(return_type or "")))
        selected_parameters = {
            name
            for _kind, name in getattr(function, "template_parameters", None) or []
            if name
        }
        deduced_parameters = set()
        for parameter, argument in zip(
            getattr(function, "params", None) or [], arguments
        ):
            if self.expression_metal_type(argument) is None:
                continue
            parameter_type = self.metal_declaration_expression_type(parameter)
            parameter_identifiers = set(
                re.findall(r"\b[A-Za-z_]\w*\b", str(parameter_type or ""))
            )
            deduced_parameters.update(parameter_identifiers & selected_parameters)
        return tuple(sorted((identifiers & selected_parameters) - deduced_parameters))

    def validate_selected_auto_return_type(
        self, declaration, initializer, function, return_type, inferred_type
    ):
        if inferred_type is None:
            reason = (
                "the selected callable return type is auto"
                if self.is_plain_metal_auto_type(return_type)
                else "the selected callable has no value return type"
            )
            raise MetalAutoTypeInferenceError(
                declaration.name,
                function.name,
                return_type or "<unknown>",
                reason,
                getattr(declaration, "source_location", None)
                or getattr(initializer, "source_location", None),
            )
        if inferred_type == "void":
            raise MetalAutoTypeInferenceError(
                declaration.name,
                function.name,
                inferred_type,
                "a void return cannot initialize an auto local",
                getattr(declaration, "source_location", None)
                or getattr(initializer, "source_location", None),
            )

        unresolved = self.unresolved_selected_return_parameters(
            function, return_type, initializer.args
        )
        if unresolved:
            raise MetalAutoTypeInferenceError(
                declaration.name,
                function.name,
                return_type,
                "the selected callable return type remains dependent on "
                + ", ".join(unresolved),
                getattr(declaration, "source_location", None)
                or getattr(initializer, "source_location", None),
                unresolved,
            )

    def inferable_metal_auto_value_type(self, metal_type):
        if self.metal_array_type_parts(metal_type) is not None:
            return True
        descriptor = self.metal_source_overload_type_descriptor(metal_type)
        if descriptor is None:
            return False
        if descriptor[0] == "pointer":
            return True
        if descriptor[0] not in {"scalar", "vector", "object"}:
            return False
        if descriptor[0] == "object" and self.is_metal_resource_type(metal_type):
            return False
        return True

    def inferred_metal_declaration_type(self, declaration, initializer=None):
        declared_type = self.metal_declaration_expression_type(declaration)
        if not self.is_metal_auto_declaration_type(declared_type):
            return declared_type
        if initializer is None:
            self.validate_auto_pointer_initializer(declaration, None)
            return declared_type
        selected_callable = self.selected_metal_callable(initializer)
        return_type = (
            self.selected_metal_callable_return_type(selected_callable)
            if selected_callable is not None
            else self.expression_metal_type(initializer)
        )
        if selected_callable is not None:
            return_type = deduced_call_return_type(
                self, declaration, initializer, selected_callable, return_type
            )
        inferred_type = self.metal_source_overload_value_type(return_type)
        if self.metal_auto_pointer_depth(declared_type):
            array_element = self.split_outer_metal_declarator_array_type(inferred_type)
            if array_element is not None and "[" not in array_element:
                inferred_type = f"{array_element}*"
            self.validate_auto_pointer_initializer(declaration, inferred_type)
        if selected_callable is not None:
            self.validate_selected_auto_return_type(
                declaration,
                initializer,
                selected_callable,
                return_type,
                inferred_type,
            )
        if self.inferable_metal_auto_value_type(inferred_type):
            return inferred_type
        if self.metal_matrix_type_parts(inferred_type) is not None:
            return inferred_type
        if self.metal_array_type_parts(inferred_type) is not None:
            return inferred_type
        if self.split_outer_metal_declarator_array_type(inferred_type) is not None:
            return inferred_type
        if self.normalized_metal_type(inferred_type) in self.struct_member_types:
            return inferred_type
        return declared_type

    def metal_type_contains_wide_vector(self, metal_type, seen_structs=None):
        if metal_type is None:
            return False
        resolved_type = str(self.resolve_type_alias(metal_type)).strip()
        while True:
            array_element = self.split_outer_metal_declarator_array_type(resolved_type)
            if array_element is None:
                break
            resolved_type = array_element
        while resolved_type.endswith("*") or resolved_type.endswith("&"):
            resolved_type = resolved_type[:-1].strip()
        if self.wide_vector_type_info(resolved_type) is not None:
            return True

        struct_name = self.normalized_metal_type(resolved_type)
        if struct_name not in self.struct_member_types:
            return False
        seen_structs = set(seen_structs or set())
        if struct_name in seen_structs:
            return False
        seen_structs.add(struct_name)
        return any(
            self.metal_type_contains_wide_vector(member_type, seen_structs)
            for member_type in self.struct_member_types[struct_name].values()
        )

    def reject_abi_visible_wide_vector_declaration(self, declaration):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(declaration, "qualifiers", []) or []
        }
        if not qualifiers.intersection({"device", "constant"}):
            return
        metal_type = self.metal_declaration_expression_type(declaration)
        if not self.metal_type_contains_wide_vector(metal_type):
            return
        raise MetalWideVectorLoweringError(
            self.resolve_type_alias(metal_type),
            "aggregate lowering does not preserve Metal ABI alignment in "
            "device- or constant-address-space storage",
            getattr(declaration, "source_location", None),
            operation="resource-layout",
        )

    def split_outer_metal_declarator_array_type(self, metal_type):
        if metal_type is None:
            return None
        match = re.fullmatch(r"(.+?)(\[[^\[\]]*\])((?:\[[^\[\]]*\])*)", str(metal_type))
        if match is None:
            return None
        base_type, _outer_extent, remaining_extents = match.groups()
        return f"{base_type}{remaining_extents}"

    def metal_array_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if not self.is_metal_array_type_name(base_name) or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def map_standard_array_storage_type(self, metal_type):
        element = self.resolve_type_alias(metal_type)
        nested = self.metal_array_type_parts(element)
        if nested is not None:
            value_type, extent = nested
            return f"array<{self.map_standard_array_storage_type(value_type)}, {self.format_array_extent(extent)}>"
        # Storage elements must retain packing rather than use value carriers.
        if element.startswith("packed_") and metal_type_layout(element) is not None:
            return element
        return self.map_type(element)

    def zero_extent_metal_array_type_parts(self, metal_type):
        resolved = self.resolve_type_alias(str(metal_type or "").strip())
        base_name, arguments = self.generic_type_parts(resolved)
        if base_name not in {"array", "metal::array"} or len(arguments) != 2:
            return None
        extent = self.substitute_template_value_text(arguments[1])
        if self.evaluate_concrete_array_extent(extent) != 0:
            return None
        return arguments[0].strip(), "0"

    def is_metal_array_type_name(self, base_name):
        return base_name in {"array", "metal::array"}

    def metal_vector_type_parts(self, metal_type):
        base_name, generic_args = self.generic_type_parts(metal_type)
        if base_name not in {"vec", "vector"} or len(generic_args) < 2:
            return None
        return generic_args[0].strip(), generic_args[1].strip()

    def map_generic_vector_type(self, element_type, size):
        size = str(size).strip()
        element_type = self.resolve_type_alias(
            self.resolve_local_type_aliases(element_type)
        )
        mapped_element = self.map_type(element_type)
        prefixes = {
            "float": "vec",
            "float16": "f16vec",
            "bfloat16": "bfloat16vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "int64": "i64vec",
            "uint64": "u64vec",
            "int16": "i16vec",
            "uint16": "u16vec",
            "int8": "i8vec",
            "uint8": "u8vec",
            "bool": "bvec",
        }
        prefix = prefixes.get(mapped_element)
        if prefix and size in {"2", "3", "4"}:
            return f"{prefix}{size}"
        wide_vector = self.wide_vector_type_info_from_parts(element_type, size)
        if wide_vector is not None:
            return wide_vector["type_name"]
        return f"vec<{mapped_element}, {size}>"

    def normalized_metal_type(self, metal_type):
        if not metal_type:
            return ""
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        while base.endswith("*") or base.endswith("&"):
            base = base[:-1].strip()
        return base

    def access_qualified_texture_parts(self, metal_type):
        metal_type = self.resolve_type_alias(metal_type)
        array_type = self.metal_array_type_parts(metal_type)
        type_to_check = array_type[0] if array_type else metal_type
        base_name, generic_args = self.generic_type_parts(type_to_check)
        if len(generic_args) < 2:
            return None, generic_args
        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None, generic_args
        generic_args = [*generic_args]
        generic_args[1] = access
        return base_name, generic_args

    def normalized_access_qualifier(self, access):
        access = str(access).replace(" ", "")
        while access.startswith("metal::"):
            access = access.split("metal::", 1)[1]
        return access

    def is_access_qualified_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        return bool(
            generic_args
            and base_name in {
                "texture1d",
                "texture1d_array",
                "texture2d",
                "texture2d_array",
                "texture_buffer",
                "texture3d",
            }
        )

    def map_storage_texture_type(self, metal_type):
        base_name, generic_args = self.access_qualified_texture_parts(metal_type)
        if not generic_args:
            return None
        return self.map_access_qualified_texture_type(base_name, generic_args)

    def storage_texture_access_attribute(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if len(generic_args) < 2:
            return ""

        access = self.normalized_access_qualifier(generic_args[1])
        access_attributes = {
            "access::read": "@readonly",
            "access::write": "@writeonly",
            "access::read_write": "@readwrite",
        }
        return access_attributes.get(access, "")

    def storage_texture_format_attributes(self, var):
        if not (
            id(var) in self.storage_texture_declaration_ids
            or getattr(var, "name", None) in self.current_storage_texture_names
            or getattr(var, "name", None) in self.global_storage_texture_names
        ):
            return ""

        _, generic_args = self.access_qualified_texture_parts(
            getattr(var, "vtype", None)
        )
        if not generic_args:
            return ""

        element_type = self.normalized_metal_type(generic_args[0])
        if element_type == "half":
            return "@rgba16f @metal_texture_element_half"
        return ""

    def split_generic_arguments(self, inner):
        args = []
        current = []
        depth = 0
        for char in inner:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            if char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            args.append("".join(current).strip())
        return args

    def map_access_qualified_texture_type(self, base_name, generic_args):
        if len(generic_args) < 2:
            return None

        access = self.normalized_access_qualifier(generic_args[1])
        if access not in self.storage_texture_accesses:
            return None

        image_type = {
            "texture1d": "image1D",
            "texture1d_array": "image1DArray",
            "texture2d": "image2D",
            "texture2d_array": "image2DArray",
            "texture_buffer": "imageBuffer",
            "texture3d": "image3D",
        }.get(base_name)
        if image_type is None:
            return None

        element_type = generic_args[0].strip()
        integer_prefix = self.integer_resource_prefix(element_type)
        if integer_prefix == "u":
            return f"u{image_type}"
        if integer_prefix == "i":
            return f"i{image_type}"
        return image_type

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.expression_base_name(expr.object)
        return None

    def metal_vector_component_parts(self, metal_type):
        candidate = self.metal_source_overload_value_type(metal_type)
        if candidate is None or candidate.endswith("*"):
            return None
        candidate = self.substitute_template_value_text(candidate)

        generic_parts = self.metal_vector_type_parts(candidate)
        if generic_parts is not None:
            element_type, size = generic_parts
            return (
                self.resolve_type_alias(element_type),
                self.concrete_generic_vector_width(size),
                str(size).strip(),
            )

        match = re.fullmatch(
            r"(?:(?:packed|simd|vector)_)?"
            r"(bool|bfloat|xhalf|half|float|double|char|uchar|short|ushort|"
            r"int|uint|long|ulong)([234])",
            self.normalized_metal_type(candidate),
        )
        if match is None:
            return None
        return match.group(1), int(match.group(2)), match.group(2)

    def metal_small_vector_type_parts(self, metal_type):
        parts = self.metal_vector_component_parts(metal_type)
        if parts is None:
            return None
        element_type, width, _width_text = parts
        if width is None or not 1 < width <= 4:
            return None
        return element_type, width

    def metal_matrix_type_parts(self, metal_type):
        candidate = self.metal_source_overload_value_type(metal_type)
        if candidate is None or candidate.endswith("*"):
            return None
        candidate = self.substitute_template_value_text(candidate)
        base_name, generic_args = self.generic_type_parts(candidate)
        if base_name and base_name.rsplit("::", 1)[-1] == "matrix":
            if len(generic_args) != 3:
                return None
            element_type, columns, rows = generic_args
            return (
                self.resolve_type_alias(element_type),
                self.concrete_generic_vector_width(columns),
                self.concrete_generic_vector_width(rows),
                str(columns).strip(),
                str(rows).strip(),
            )

        match = re.fullmatch(
            r"(?:(?:simd|matrix)_)?(bfloat|xhalf|half|float|double)" r"([234])x([234])",
            self.normalized_metal_type(candidate),
        )
        if match is None:
            return None
        return (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            match.group(2),
            match.group(3),
        )

    def metal_index_expression_text(self, expression):
        if isinstance(expression, (str, int, float, bool)):
            return str(expression)
        name = getattr(expression, "name", None)
        if name:
            return str(name)
        value = getattr(expression, "value", None)
        if value is not None:
            return str(value)
        return None

    def metal_indexed_type_selection(self, expr):
        base_type = self.expression_metal_type(expr.array)
        source_location = getattr(expr, "source_location", None)
        index_expression = self.metal_index_expression_text(expr.index)
        dependent = dependent_subscript_selection(
            expr.array,
            (
                None
                if base_type is None
                else self.metal_source_overload_value_type(base_type)
            ),
            self.expression_metal_type,
            unbound_template_type_names(
                self.current_function,
                set(self.type_aliases).union(*self.template_type_bindings),
            ),
            self.struct_declarations.values(),
        )
        if dependent is not None:
            return dependent
        if base_type is None:
            raise MetalIndexedComponentTypeResolutionError(
                None,
                "subscript",
                "the indexed base expression type could not be inferred",
                source_location,
                index_expression,
            )

        resolved_type = self.metal_source_overload_value_type(base_type)
        if resolved_type is None:
            raise MetalIndexedComponentTypeResolutionError(
                base_type,
                "subscript",
                "the indexed base does not have a concrete value type",
                source_location,
                index_expression,
            )

        declarator_element_type = self.split_outer_metal_declarator_array_type(
            resolved_type
        )
        if declarator_element_type is not None:
            return {
                "kind": "array",
                "base_type": resolved_type,
                "selected_type": declarator_element_type,
            }

        pointer_element_type = self.metal_pointer_pointee_type_once(resolved_type)
        if pointer_element_type is not None:
            return {
                "kind": "pointer",
                "base_type": resolved_type,
                "selected_type": pointer_element_type,
            }

        array_parts = self.metal_array_type_parts(resolved_type)
        if array_parts is not None:
            return {
                "kind": "array",
                "base_type": resolved_type,
                "selected_type": array_parts[0],
            }

        vector_parts = self.metal_vector_component_parts(resolved_type)
        if vector_parts is not None:
            element_type, width, width_text = vector_parts
            if not element_type:
                raise MetalIndexedComponentTypeResolutionError(
                    resolved_type,
                    "vector-subscript",
                    "the vector element type is empty",
                    source_location,
                    index_expression,
                )
            return {
                "kind": "vector",
                "base_type": resolved_type,
                "selected_type": element_type,
                "element_type": element_type,
                "width": width,
                "width_text": width_text,
            }

        matrix_parts = self.metal_matrix_type_parts(resolved_type)
        if matrix_parts is not None:
            element_type, columns, rows, columns_text, rows_text = matrix_parts
            if rows is None:
                raise MetalIndexedComponentTypeResolutionError(
                    resolved_type,
                    "matrix-subscript",
                    f"the matrix row count '{rows_text}' is not concrete",
                    source_location,
                    index_expression,
                )
            return {
                "kind": "matrix",
                "base_type": resolved_type,
                "selected_type": self.metal_vector_type_from_element(
                    element_type, rows
                ),
                "element_type": element_type,
                "columns": columns,
                "columns_text": columns_text,
                "rows": rows,
                "rows_text": rows_text,
            }

        base_name, generic_args = self.generic_type_parts(resolved_type)
        generic_name = base_name.rsplit("::", 1)[-1] if base_name else None
        if generic_name in {"vec", "vector", "matrix"}:
            raise MetalIndexedComponentTypeResolutionError(
                resolved_type,
                f"{generic_name}-subscript",
                "the aggregate template arguments do not prove a component type",
                source_location,
                index_expression,
            )

        if self.metal_scalar_arithmetic_type_info(resolved_type) is not None:
            raise MetalIndexedComponentTypeResolutionError(
                resolved_type,
                "subscript",
                "a scalar value has no indexed component",
                source_location,
                index_expression,
            )

        # Resource tables and user aggregates can define their own subscript
        # contract. Preserve those expressions unless a known aggregate layer is
        # available to consume here.
        return {
            "kind": "aggregate",
            "base_type": resolved_type,
            "selected_type": None,
        }

    def metal_vector_type_from_element(self, element_type, width):
        if self.map_type(self.resolve_type_alias(element_type)) == "bfloat16":
            return f"bfloat{width}"
        info = self.metal_scalar_arithmetic_type_info(element_type)
        if info is None:
            return f"vec<{element_type}, {width}>"
        family, signed, bits = info
        if family == "floating":
            base = {16: "half", 32: "float"}.get(bits)
        elif bits == 1:
            base = "bool"
        elif signed:
            base = {8: "char", 16: "short", 32: "int", 64: "long"}.get(bits)
        else:
            base = {8: "uchar", 16: "ushort", 32: "uint", 64: "ulong"}.get(bits)
        return f"{base}{width}" if base is not None else f"vec<{element_type}, {width}>"
