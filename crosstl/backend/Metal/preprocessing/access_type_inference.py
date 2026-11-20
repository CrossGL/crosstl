"""Type inference for subscripts, member access and literals, span removal and kernel instantiation discovery."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from .support import (
    IDENTIFIER_RE,
    _MetalBufferTypeView,
    _MetalStructDefinition,
    _MLXKernelInstantiation,
)


class MetalAccessTypeInferenceMixin:
    """Members of ``MetalPreprocessor``: type inference for subscripts, member access and literals, span removal and kernel instantiation discovery."""

    def _infer_subscript_base_element_type(
        self,
        base: str,
        buffer_element_types: _MetalBufferTypeView,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Element type of the subscript base `base[...]`:
        #   * bare name: a buffer/array element type (`buf[i]`, `totals[i]`).
        #   * member access `obj.member` / `obj->member`: the element type of the
        #     struct field `member`.
        if IDENTIFIER_RE.fullmatch(base):
            return self._buffer_element_type(buffer_element_types, base)
        cast_type = self._c_style_pointer_cast_type(base)
        if cast_type is not None:
            return self._pointer_pointee_value_type(cast_type)
        field_type = self._struct_member_field_type(base, struct_field_types)
        if field_type is None:
            return None
        return self._subscript_declared_element_type(field_type)

    def _c_style_pointer_cast_type(self, expression: str) -> Optional[str]:
        expression = self._strip_enclosing_parens(expression.strip())
        if not expression.startswith("("):
            return None
        close = self._find_matching_delimiter(expression, 0, "(", ")")
        if close is None:
            return None
        pointer = self._normalize_known_address_space_pointer_type(expression[1:close])
        operand = expression[close + 1 :].strip()
        # Admit one complete operand, not a cast followed by a binary tail.
        if pointer is None or not operand:
            return None
        if IDENTIFIER_RE.fullmatch(operand) or (
            operand.startswith("(")
            and self._find_matching_delimiter(operand, 0, "(", ")") == len(operand) - 1
        ):
            return pointer
        return None

    def _subscript_declared_element_type(self, type_text: str) -> Optional[str]:
        text = self._normalize_template_argument_text(type_text)
        # Declarator dimensions enclose the named type, even when that type is
        # itself a standard array or an address-space-qualified pointer.
        dimensions = re.fullmatch(r"(.+?)\[[^\[\]]*\]((?:\[[^\[\]]*\])*)", text)
        if dimensions is not None:
            return self._normalize_template_argument_text(
                dimensions.group(1) + dimensions.group(2)
            )
        value_type = re.sub(
            r"^(?:(?:const|constexpr|volatile|mutable|device|constant|thread|threadgroup)\s+)+",
            "",
            text,
        )
        array = re.match(r"(?:metal::|c10::metal::)?array\s*<", value_type)
        if array is not None:
            start = value_type.find("<")
            end = self._find_matching_angle(value_type, start)
            if end != len(value_type) - 1:
                return None
            arguments = self._split_top_level_commas(value_type[start + 1 : end])
            if len(arguments) != 2 or not all(item.strip() for item in arguments):
                return None
            return self._normalize_template_argument_text(arguments[0])
        return self._pointer_pointee_value_type(text)

    def _infer_group_builtin_call_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # `builtin(arg0, ...)` for a SIMD/quad group built-in that returns arg0's
        # type -> the inferred type of arg0. Conservative: only a recognized
        # built-in with a fully balanced trailing call and an inferable first
        # argument yields a type. `metal::`/`::metal::` name the same built-in.
        match = re.match(
            r"(?:(?:::\s*)?metal\s*::\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr
        )
        if match is None:
            return None
        if match.group(1) not in self._METAL_FIRST_ARG_TYPED_GROUP_BUILTINS:
            return None
        paren_start = expr.find("(")
        paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
        if paren_end != len(expr) - 1:
            return None
        arguments = [
            argument
            for argument in self._split_top_level_commas(
                expr[paren_start + 1 : paren_end]
            )
            if argument.strip()
        ]
        if not arguments:
            return None
        return self._infer_argument_type(
            arguments[0],
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )

    def _infer_member_access_type(
        self,
        expr: str,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        field_type = self._struct_member_field_type(expr, struct_field_types)
        if field_type is None:
            return None
        return self._normalize_inferred_expression_type(field_type)

    def _infer_vector_member_access_type(
        self, expr: str, local_variable_types: Dict[str, str]
    ) -> Optional[str]:
        match = re.fullmatch(
            r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
            r"(?P<swizzle>[A-Za-z_][A-Za-z0-9_]*)",
            expr,
        )
        if match is None:
            return None
        vector_type = self._normalize_inferred_type(
            local_variable_types.get(match.group("obj"), "")
        )
        vector_match = re.fullmatch(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?P<width>[234])", vector_type
        )
        if vector_match is not None and vector_type in self._METAL_SCALAR_VECTOR_TYPES:
            base_type, width = vector_match.group("base"), int(
                vector_match.group("width")
            )
        else:
            # `metal::vec<float, 2>` is `float2`.
            generic = (
                self._scalar_and_width(vector_type)
                if re.match(r"(?:::)?(?:metal::)?vec<", vector_type)
                else None
            )
            if generic is None or generic[1] not in (2, 3, 4):
                return None
            base_type, width = generic

        swizzle = match.group("swizzle")
        if not 1 <= len(swizzle) <= 4:
            return None
        component_sets = ("xyzw", "rgba")
        components = next(
            (
                component_set
                for component_set in component_sets
                if all(ch in component_set for ch in swizzle)
            ),
            None,
        )
        if components is None:
            return None
        if any(components.index(component) >= width for component in swizzle):
            return None
        return base_type if len(swizzle) == 1 else f"{base_type}{len(swizzle)}"

    def _struct_member_field_type(
        self,
        access: str,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Resolve a single-level `obj.member` or `obj->member` access to the
        # member's declared field type. Pointer bindings are stored under an
        # `obj->` key so the access operator must match the declaration shape.
        if not struct_field_types:
            return None
        match = re.fullmatch(
            r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<access>\.|->)\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)",
            access,
        )
        if match is None:
            return None
        lookup = match.group("obj")
        if match.group("access") == "->":
            lookup += "->"
        fields = struct_field_types.get(lookup)
        if not fields:
            return None
        return fields.get(match.group("member"))

    def _infer_literal_type(self, expr: str) -> Optional[str]:
        # Recognize the simple numeric/boolean literal forms required by the
        # conservative inference contract.
        if expr in {"true", "false"}:
            return "bool"
        # half literal: `1h`, `2.5h`.
        if re.fullmatch(r"[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?h", expr):
            return "half"
        # float literal: `1.0`, `2.5f`, `1.0f`, `3.f`, `1e3f`.
        if re.fullmatch(r"[0-9]*\.[0-9]*(?:[eE][+-]?[0-9]+)?f?", expr) and any(
            c.isdigit() for c in expr
        ):
            return "float"
        if re.fullmatch(r"[0-9]+(?:[eE][+-]?[0-9]+)?f", expr):
            return "float"
        if re.fullmatch(r"[0-9]+[eE][+-]?[0-9]+", expr):
            return "float"
        # unsigned integer literal: `1u`, `42U`.
        if re.fullmatch(r"[0-9]+[uU]", expr):
            return "uint"
        # plain integer literal: `1`, `42` (and hex).
        if re.fullmatch(r"[0-9]+", expr) or re.fullmatch(r"0[xX][0-9A-Fa-f]+", expr):
            return "int"
        return None

    def _remove_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        if not spans:
            return text
        result: List[str] = []
        pos = 0
        for start, end in sorted(spans):
            if start < pos:
                continue
            result.append(text[pos:start])
            pos = end
        result.append(text[pos:])
        return "".join(result)

    def _collapse_blank_lines(self, text: str) -> str:
        # Collapse runs of blank lines left behind by removed method bodies so
        # the data-only struct stays tidy without altering meaningful content.
        return re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", text)

    def _find_mlx_kernel_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        return self._find_project_template_instantiations(code)
