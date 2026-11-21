"""Concrete scalar types, sizeof, type traits, template constraint evaluation and constrained overload selection."""

import operator
import re
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..category_traits import (
    CATEGORY_TRAITS,
    evaluate_category_trait,
)
from ..trait_alias_templates import (
    expand_trait_alias_templates,
    find_trait_alias_templates,
)
from ..trait_operand_types import (
    METAL_MATRIX_TYPE_NAMES,
    canonical_matrix_type,
    typedef_aggregate_names,
)
from ..type_layout import metal_type_size
from ..value_traits import simdgroup_type_trait
from .support import (
    _C_INTEGER_SPELLINGS,
    IDENTIFIER_RE,
    MetalTypeTraitError,
    _MetalBooleanVariableTemplate,
    _MetalNamespaceVisibility,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTypeAliasBinding,
    _strip_global_metal_qualifiers,
)


class MetalTypeTraitMixin:
    """Members of ``MetalPreprocessor``: concrete scalar types, sizeof, type traits, template constraint evaluation and constrained overload selection."""

    def _split_return_type_sfinae(self, return_type: str) -> Tuple[Optional[str], str]:
        # Detect a return-type SFINAE wrapper `[metal::]enable_if_t<COND, TYPE>`
        # (the second SFINAE layer used by Sum/Min/Max's `simd_reduce_impl`). On a
        # match return (COND, TYPE) — the enabling condition and the unwrapped
        # value type; otherwise (None, return_type) unchanged. Only a return type
        # that is EXACTLY an `enable_if_t<...>` template-id is unwrapped, so an
        # ordinary `T` / `float4` return is left intact (never guess).
        text = _strip_global_metal_qualifiers(return_type.strip())
        match = re.match(r"^(?:typename\s+)?(?:metal\s*::\s*)?enable_if_t\s*<", text)
        if match is None:
            return None, return_type
        angle_start = text.find("<", match.end() - 1)
        # Use the SFINAE-aware angle/split so a condition containing a comparison
        # (`sizeof(T) < 8`) balances correctly.
        angle_end = self._find_matching_template_param_angle(text, angle_start)
        if angle_end is None or text[angle_end + 1 :].strip():
            # Trailing tokens after the angle (e.g. a pointer/qualifier) are not a
            # simple wrapper; leave it for clean-fail rather than mis-parse.
            return None, return_type
        arguments = self._split_template_parameter_list(
            text[angle_start + 1 : angle_end]
        )
        if len(arguments) != 2:
            return None, return_type
        condition = arguments[0].strip()
        value_type = arguments[1].strip()
        if not condition or not value_type:
            return None, return_type
        return condition, value_type

    # ------------------------------------------------------------------ #
    # SFINAE constraint evaluation for overload selection.               #
    # ------------------------------------------------------------------ #
    # Only the two constraint families MLX `reduce` needs are understood:
    # `sizeof(T) <cmp> N` and `[!]is_integral_v<T>`. Any other constraint shape
    # is a clean-fail (the overload set is left unresolved so the existing
    # diagnostic fires) — the contract is to NEVER guess or mis-select.

    # Byte sizes of the scalar element types (vectors scale by component count).
    _METAL_SCALAR_TYPE_SIZES: Dict[str, int] = {
        "bool": 1,
        "char": 1,
        "uchar": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "short": 2,
        "ushort": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "half": 2,
        "float16_t": 2,
        "bfloat16_t": 2,
        "int": 4,
        "uint": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "float": 4,
        "long": 8,
        "ulong": 8,
        "int64_t": 8,
        "uint64_t": 8,
        "double": 8,
        "size_t": 8,
    }
    # Integral scalar element types for `is_integral_v<T>`.
    _METAL_INTEGRAL_SCALAR_TYPES: Set[str] = {
        "bool",
        "char",
        "uchar",
        "int8_t",
        "uint8_t",
        "short",
        "ushort",
        "int16_t",
        "uint16_t",
        "int",
        "uint",
        "int32_t",
        "uint32_t",
        "long",
        "ulong",
        "int64_t",
        "uint64_t",
        "size_t",
    }
    _METAL_SIGNED_SCALAR_TYPES: Set[str] = {
        "char",
        "int8_t",
        "short",
        "int16_t",
        "int",
        "int32_t",
        "long",
        "int64_t",
        "half",
        "float16_t",
        "bfloat16_t",
        "float",
        "double",
    }
    _METAL_NATIVE_SCALAR_SPELLINGS: Dict[str, str] = {
        "bfloat": "bfloat16_t",
        "bfloat16": "bfloat16_t",
    }
    # Every spelling of a Metal scalar type, to the name `_METAL_SCALAR_TYPE_SIZES` gives it; a spelling missing
    # here is no Metal scalar.
    _METAL_SCALAR_BY_SPELLING: Dict[str, str] = {
        **dict(zip(_METAL_SCALAR_TYPE_SIZES, _METAL_SCALAR_TYPE_SIZES)),
        **_METAL_NATIVE_SCALAR_SPELLINGS,
    }
    _METAL_FLOATING_SCALAR_TYPES: Set[str] = {
        "half",
        "float16_t",
        "bfloat16_t",
        "float",
        "double",
    }
    _METAL_TYPE_TRAIT_ARITIES: Dict[str, int] = {
        "is_same": 2,
        "is_convertible": 2,
        "is_integral": 1,
        "is_floating_point": 1,
        "is_signed": 1,
        "is_unsigned": 1,
        "is_const": 1,
        "is_pointer": 1,
        "is_scalar": 1,
        "is_vector": 1,
        "is_arithmetic": 1,
    }
    _METAL_TYPE_TRAIT_START_RE = re.compile(
        r"(?<![A-Za-z0-9_:])"
        r"(?:(?:::)?metal\s*::\s*)?"
        r"(?P<trait>is_same_v|is_convertible_v|is_integral_v|"
        r"is_floating_point_v|is_signed_v|is_unsigned_v|is_const_v|"
        r"is_pointer_v|is_scalar_v|is_vector_v|is_arithmetic_v|"
        r"is_same|is_convertible|is_integral|"
        r"is_floating_point|is_signed|is_unsigned|is_const|is_pointer|"
        r"is_scalar|is_vector|is_arithmetic)\s*<"
    )

    # Metal SIMD-group / quad-group built-ins that return the SAME type as their
    # first argument (a shuffle/broadcast moves a value between lanes; a prefix /
    # reduction combines values of the operand type). Recognizing them lets an
    # argument like `simd_shuffle_and_fill_up(val, init, i)` be typed from `val`,
    # so a SFINAE member method that threads such a call through a sibling method
    # (scan's `operator()(val, simd_shuffle_and_fill_up(val, ...))`) can resolve.
    _METAL_FIRST_ARG_TYPED_GROUP_BUILTINS: Set[str] = {
        "simd_shuffle",
        "simd_shuffle_up",
        "simd_shuffle_down",
        "simd_shuffle_xor",
        "simd_shuffle_rotate_up",
        "simd_shuffle_rotate_down",
        "simd_shuffle_and_fill_up",
        "simd_shuffle_and_fill_down",
        "simd_broadcast",
        "simd_broadcast_first",
        "simd_prefix_inclusive_sum",
        "simd_prefix_exclusive_sum",
        "simd_prefix_inclusive_product",
        "simd_prefix_exclusive_product",
        "simd_sum",
        "simd_product",
        "simd_prod",
        "simd_min",
        "simd_max",
        "simd_and",
        "simd_or",
        "simd_xor",
        "quad_shuffle",
        "quad_shuffle_up",
        "quad_shuffle_down",
        "quad_shuffle_xor",
        "quad_shuffle_and_fill_up",
        "quad_shuffle_and_fill_down",
        "quad_broadcast",
        "quad_sum",
        "quad_prefix_inclusive_sum",
        "quad_prefix_exclusive_sum",
    }

    def _scalar_and_width(self, type_text: str) -> Optional[Tuple[str, int]]:
        # Decompose a scalar/vector type into (scalar base, component width). A
        # trailing 2/3/4 on a known scalar denotes a vector; concrete
        # `metal::vec<T, N>` spellings carry the same information. Returns None
        # when the base or vector extent is not concrete and recognized.
        text = self._normalize_inferred_type(type_text)
        if not text or " " in text:
            return None
        generic_match = re.fullmatch(r"(?:::)?(?:metal::)?vec<(.+)>", text)
        if generic_match is not None:
            arguments = self._split_top_level_commas(generic_match.group(1))
            if len(arguments) != 2:
                return None
            base = self._METAL_SCALAR_BY_SPELLING.get(
                self._normalize_inferred_type(arguments[0])
            )
            if base is None:
                return None
            width_text = self._proven_integral_constant_value("int", arguments[1])
            if width_text is None:
                return None
            width = int(width_text)
            return (base, width) if width > 0 else None
        match = re.fullmatch(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_]*?)(?P<width>[234])?", text
        )
        if match is None:
            return None
        # `bfloat` and `bfloat16` spell the scalar `_METAL_SCALAR_TYPE_SIZES` calls `bfloat16_t`, including in
        # vectors.
        base = self._METAL_SCALAR_BY_SPELLING.get(match.group("base"))
        if base is None:
            # A width suffix that is actually part of the name (e.g. an unknown
            # type) — treat the whole token as the base.
            if text in self._METAL_SCALAR_TYPE_SIZES:
                return text, 1
            return None
        width = int(match.group("width")) if match.group("width") else 1
        return base, width

    def _sizeof_concrete_type(
        self,
        type_text: str,
        *,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        position: int = 0,
    ) -> Optional[int]:
        scalar_size = metal_type_size(type_text)
        if scalar_size is not None or not structs:
            return scalar_size
        layout = self._concrete_aggregate_type_layout(
            type_text,
            list(structs),
            type_aliases or {},
            position,
        )
        return layout[0] if layout is not None else None

    def _is_integral_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_INTEGRAL_SCALAR_TYPES

    def _is_signed_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_SIGNED_SCALAR_TYPES

    def _is_floating_point_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_FLOATING_SCALAR_TYPES

    def _parse_metal_type_trait_expression(
        self,
        expression: str,
    ) -> Optional[Tuple[str, List[str]]]:
        text = expression.strip()
        match = self._METAL_TYPE_TRAIT_START_RE.match(text)
        if match is None:
            return None
        angle_start = text.find("<", match.start("trait"), match.end())
        angle_end = self._find_matching_angle(text, angle_start)
        if angle_end is None:
            return None
        trait = match.group("trait")
        suffix = text[angle_end + 1 :].strip()
        if trait.endswith("_v"):
            if suffix:
                return None
            trait = trait[:-2]
        elif re.fullmatch(r"::\s*value", suffix) is None:
            return None
        operands = self._split_top_level_commas(text[angle_start + 1 : angle_end])
        return trait, operands

    def _canonical_type_trait_operand(
        self,
        type_text: str,
        resolver,
    ) -> Optional[str]:
        resolved = resolver(type_text)
        if resolved is None:
            return None
        normalized = self._normalize_template_argument_text(resolved)
        normalized = re.sub(r"^(?:typename|struct|class|enum)\b\s*", "", normalized)
        normalized = re.sub(
            r"(?<![A-Za-z0-9_])(?:::)?metal::(?=[A-Za-z_])",
            "",
            normalized,
        )
        normalized = self._normalize_template_argument_text(normalized)
        # A multi-word C integer spelling names a one-word Metal scalar. Every other operand already has its
        # canonical spelling; the trait evaluator classifies it.
        if normalized in _C_INTEGER_SPELLINGS:
            normalized = _C_INTEGER_SPELLINGS[normalized]
        if not re.search(r"[*&\[\]]", normalized):
            qualifiers = {
                token
                for token in IDENTIFIER_RE.findall(normalized)
                if token in {"const", "volatile"}
            }
            unqualified = re.sub(r"\b(?:const|volatile)\b", " ", normalized)
            unqualified = self._normalize_template_argument_text(unqualified)
            decomposed = self._scalar_and_width(unqualified)
            if decomposed is not None:
                base, width = decomposed
                normalized = base if width == 1 else f"{base}{width}"
                if qualifiers:
                    normalized = " ".join((*sorted(qualifiers), normalized))
            else:
                matrix = canonical_matrix_type(unqualified)
                if matrix is not None:
                    normalized = " ".join((*sorted(qualifiers), matrix))
        return normalized or None

    def _type_trait_operand_is_concrete(
        self,
        type_text: str,
        known_type_names: Set[str],
        *,
        allow_unknown_named_types: bool,
    ) -> bool:
        normalized = self._normalize_template_argument_text(type_text)
        if not normalized or re.search(
            r"\b(?:auto|decltype|typename)\b|\bis_[A-Za-z_]*\s*<",
            normalized,
        ):
            return False

        stripped = re.sub(
            r"\b(?:const|volatile|restrict|device|constant|thread|threadgroup)\b",
            " ",
            normalized,
        )
        stripped = re.sub(r"(?:&&|[&*])+", " ", stripped)
        stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
        stripped = self._normalize_template_argument_text(stripped)
        if not stripped:
            return False
        if self._scalar_and_width(stripped) is not None or stripped == "void":
            return True

        angle_start = stripped.find("<")
        if angle_start != -1:
            angle_end = self._find_matching_angle(stripped, angle_start)
            if angle_end != len(stripped) - 1:
                return False
            owner = stripped[:angle_start].split("::")[-1]
            if owner not in known_type_names and not allow_unknown_named_types:
                return False
            arguments = self._split_top_level_commas(
                stripped[angle_start + 1 : angle_end]
            )
            return bool(arguments) and all(
                self._evaluate_static_integral_expression(argument)[0]
                or self._type_trait_operand_is_concrete(
                    argument,
                    known_type_names,
                    allow_unknown_named_types=allow_unknown_named_types,
                )
                for argument in arguments
            )

        if not re.fullmatch(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", stripped):
            return False
        return (
            allow_unknown_named_types
            or stripped in known_type_names
            or (stripped.split("::")[-1] in known_type_names)
        )

    def _evaluate_metal_type_trait(
        self,
        trait: str,
        operands: Sequence[str],
        *,
        resolver,
        known_type_names: Set[str],
        allow_unknown_named_types: bool = False,
    ) -> Tuple[Optional[bool], Tuple[str, ...], str]:
        expected_arity = self._METAL_TYPE_TRAIT_ARITIES.get(trait)
        resolved_operands = tuple(
            self._canonical_type_trait_operand(operand, resolver) or operand.strip()
            for operand in operands
        )
        if expected_arity is None or len(operands) != expected_arity:
            return None, resolved_operands, "argument-count"
        concrete = tuple(
            self._type_trait_operand_is_concrete(
                operand,
                known_type_names,
                allow_unknown_named_types=allow_unknown_named_types,
            )
            for operand in resolved_operands
        )
        if not all(concrete):
            return None, resolved_operands, "operand-unresolved"

        if trait in CATEGORY_TRAITS:
            category = evaluate_category_trait(self, trait, resolved_operands[0])
            if category is None:
                return None, resolved_operands, "operand-category-unsupported"
            return category, resolved_operands, "evaluated"

        if trait == "is_same":
            return (
                resolved_operands[0] == resolved_operands[1],
                resolved_operands,
                "evaluated",
            )
        if trait == "is_convertible":
            source, destination = resolved_operands
            if source == destination:
                return True, resolved_operands, "evaluated"
            source_scalar = self._scalar_and_width(source)
            destination_scalar = self._scalar_and_width(destination)
            if source_scalar is None or destination_scalar is None:
                return False, resolved_operands, "evaluated"
            source_base, _source_width = source_scalar
            destination_base, _destination_width = destination_scalar
            # Metal exposes bfloat as explicitly constructible, not as an
            # implicit half/float destination. It may still widen to ordinary
            # floating-point types, matching native overload selection.
            if destination_base == "bfloat16_t":
                return False, resolved_operands, "evaluated"
            source_numeric = (
                source_base in self._METAL_INTEGRAL_SCALAR_TYPES
                or source_base in self._METAL_FLOATING_SCALAR_TYPES
            )
            destination_numeric = (
                destination_base in self._METAL_INTEGRAL_SCALAR_TYPES
                or destination_base in self._METAL_FLOATING_SCALAR_TYPES
            )
            return (
                source_numeric and destination_numeric,
                resolved_operands,
                "evaluated",
            )

        operand = resolved_operands[0]
        if trait == "is_pointer":
            without_reference = re.sub(r"\s*(?:&&|&)\s*$", "", operand)
            if without_reference != operand:
                return False, resolved_operands, "evaluated"
            without_top_level_cv = re.sub(r"\s+(?:const|volatile)\s*$", "", operand)
            if without_top_level_cv.endswith("*"):
                return True, resolved_operands, "evaluated"
            return False, resolved_operands, "evaluated"

        if trait == "is_const":
            if re.search(r"(?:&&|&)\s*$", operand):
                return False, resolved_operands, "evaluated"
            pointer = operand.rfind("*")
            qualifier_region = operand[pointer + 1 :] if pointer != -1 else operand
            if re.search(r"\bconst\b", qualifier_region):
                return True, resolved_operands, "evaluated"
            return False, resolved_operands, "evaluated"

        value_type = re.sub(r"\b(?:const|volatile)\b", " ", operand)
        value_type = self._normalize_template_argument_text(value_type)
        if re.search(r"[*&\[\]]", value_type):
            return False, resolved_operands, "evaluated"
        if trait == "is_integral":
            result = self._is_integral_concrete_type(value_type)
        elif trait == "is_floating_point":
            result = self._is_floating_point_concrete_type(value_type)
        elif trait == "is_signed":
            result = self._is_signed_concrete_type(value_type)
        else:
            integral = self._is_integral_concrete_type(value_type)
            signed = self._is_signed_concrete_type(value_type)
            result = (
                None if integral is None or signed is None else integral and not signed
            )
        if result is None:
            if concrete[0]:
                return False, resolved_operands, "evaluated"
            return None, resolved_operands, "operand-category-unsupported"
        return result, resolved_operands, "evaluated"

    def _concrete_type_trait_names(
        self,
        code: str,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
    ) -> Set[str]:
        names = set(self._METAL_SCALAR_VECTOR_TYPES)
        names.update(self._METAL_SCALAR_TYPE_SIZES)
        names.update({"void", "atomic_bool", "atomic_int", "atomic_uint"})
        for struct in structs or self._find_concrete_struct_definitions(code):
            names.add(struct.name)
            if struct.qualified_name:
                names.add(struct.qualified_name)
        names.update(template.name for template in self._find_template_structs(code))
        names.update(METAL_MATRIX_TYPE_NAMES)
        template_spans = self._find_template_declaration_spans(code)
        ignored_spans = sorted(
            [*template_spans, *self._find_comment_and_literal_spans(code)]
        )
        names.update(
            typedef_aggregate_names(
                code,
                lambda text, start: self._find_matching_delimiter(
                    text, start, "{", "}"
                ),
                lambda position: self._containing_span(position, ignored_spans)
                is not None,
            )
        )
        for match in re.finditer(
            r"\benum(?:\s+class)?\s+(?P<name>[A-Za-z_]\w*)",
            code,
        ):
            if self._containing_span(match.start(), ignored_spans) is None:
                names.add(match.group("name"))
        return names

    def _concrete_type_trait_owner(self, code: str, position: int) -> Optional[str]:
        template_spans = self._find_template_declaration_spans(code)
        candidates: List[Tuple[int, str]] = []
        for function in self._find_non_template_function_definitions(
            code,
            template_spans,
        ):
            if function.span[0] <= position < function.span[1]:
                candidates.append((function.span[1] - function.span[0], function.name))
        for struct in self._find_concrete_struct_definitions(code):
            if struct.span[0] <= position < struct.span[1]:
                candidates.append(
                    (
                        struct.span[1] - struct.span[0],
                        struct.qualified_name or struct.name,
                    )
                )
        return min(candidates)[1] if candidates else None

    def _fold_initializer_type_traits(
        self,
        initializer: str,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        position: int,
    ) -> str:
        """Fold type traits over concrete operands in one constexpr initializer, e.g. ``is_same_v<float, bool>``.

        A trait whose operands do not all resolve to concrete supported types is left in place, so the constant
        stays unknown and any template argument that names it is still refused.
        """
        if not any(name in initializer for name in self._METAL_TYPE_TRAIT_ARITIES):
            return initializer
        replacements: List[Tuple[int, int, str]] = []
        for match in self._METAL_TYPE_TRAIT_START_RE.finditer(initializer):
            angle_start = initializer.find("<", match.start("trait"), match.end())
            angle_end = self._find_matching_angle(initializer, angle_start)
            if angle_end is None:
                continue
            spelling = match.group("trait")
            trait = spelling[:-2] if spelling.endswith("_v") else spelling
            expression_end = angle_end + 1
            if not spelling.endswith("_v"):
                suffix = re.match(r"\s*::\s*value\b", initializer[expression_end:])
                if suffix is None:
                    continue
                expression_end += suffix.end()
            value, _operands, _reason = self._evaluate_metal_type_trait(
                trait,
                self._split_top_level_commas(initializer[angle_start + 1 : angle_end]),
                resolver=lambda operand: self._canonicalize_type_aliases_at(
                    operand, type_aliases or {}, position
                ),
                known_type_names=set(),
            )
            if value is not None:
                replacements.append(
                    (match.start(), expression_end, "true" if value else "false")
                )
        if not replacements:
            return initializer
        return self._apply_text_replacements(initializer, replacements)

    def _fold_concrete_type_trait_values(
        self, code: str, *, require_resolution: bool = False
    ) -> str:
        if not any(name in code for name in self._METAL_TYPE_TRAIT_ARITIES):
            return code
        template_spans = self._find_template_declaration_spans(code)
        ignored_spans = sorted(
            [*template_spans, *self._find_comment_and_literal_spans(code)]
        )
        aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=template_spans,
        )
        structs = self._find_concrete_struct_definitions(code)
        known_type_names = self._concrete_type_trait_names(code, structs)
        owner_name_counts: Dict[str, int] = {}
        for struct in structs:
            owner_name_counts[struct.name] = owner_name_counts.get(struct.name, 0) + 1
        qualified_alias_targets: Dict[str, str] = {}
        for struct in structs:
            owner_names = {struct.qualified_name} if struct.qualified_name else set()
            if owner_name_counts[struct.name] == 1:
                owner_names.add(struct.name)
            for alias, target in struct.type_aliases.items():
                for owner_name in owner_names:
                    qualified_alias_targets[f"{owner_name}::{alias}"] = target

        # Alias templates (`Int<1>` for `integral_constant<int, 1>`) name the template-id they expand to
        # (trait_alias_templates.py).
        alias_templates = find_trait_alias_templates(self, code)

        def resolve_operand(operand: str, position: int) -> Optional[str]:
            operand = expand_trait_alias_templates(self, operand, alias_templates)
            resolved = self._canonicalize_type_aliases_at(operand, aliases, position)
            if resolved is None:
                return None
            for _ in range(len(qualified_alias_targets) + 1):
                previous = resolved
                resolved = re.sub(r"\btypename\s+", "", resolved)
                resolved = re.sub(r"::template\s+", "::", resolved)
                resolved = expand_trait_alias_templates(self, resolved, alias_templates)
                for qualified_alias, target in qualified_alias_targets.items():
                    resolved = re.sub(
                        rf"(?<![A-Za-z0-9_:]){re.escape(qualified_alias)}\b",
                        target,
                        resolved,
                    )
                resolved = self._normalize_template_argument_text(resolved)
                if resolved == previous:
                    break
            return resolved

        replacements: List[Tuple[int, int, str]] = []
        for match in self._METAL_TYPE_TRAIT_START_RE.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            angle_start = code.find("<", match.start("trait"), match.end())
            angle_end = self._find_matching_angle(code, angle_start)
            if angle_end is None:
                continue
            trait_spelling = match.group("trait")
            trait = (
                trait_spelling[:-2] if trait_spelling.endswith("_v") else trait_spelling
            )
            expression_end = angle_end + 1
            if not trait_spelling.endswith("_v"):
                suffix = re.match(r"\s*::\s*value\b", code[expression_end:])
                if suffix is None:
                    continue
                expression_end += suffix.end()
            operands = self._split_top_level_commas(code[angle_start + 1 : angle_end])
            value, resolved_operands, reason = self._evaluate_metal_type_trait(
                trait,
                operands,
                resolver=lambda operand, position=match.start(): resolve_operand(
                    operand,
                    position,
                ),
                known_type_names=known_type_names,
            )
            if value is not None:
                replacements.append(
                    (match.start(), expression_end, "true" if value else "false")
                )
                continue
            if not (require_resolution or self._fail_closed_static_assertions):
                continue
            expression = re.sub(
                r"\s+", " ", code[match.start() : expression_end]
            ).strip()
            suggested_action = (
                "materialize every trait operand to a concrete supported type before "
                "target translation"
            )
            owner = self._concrete_type_trait_owner(code, match.start())
            owner_detail = f" in '{owner}'" if owner else ""
            raise MetalTypeTraitError(
                f"Metal type trait '{expression}'{owner_detail} could not be evaluated "
                f"because {reason.replace('-', ' ')}. Suggested action: "
                f"{suggested_action}.",
                trait=trait,
                expression=expression,
                operands=tuple(operand.strip() for operand in operands),
                resolved_operands=resolved_operands,
                reason=reason,
                owner=owner,
                source_location=self._source_location_for_offsets(
                    code,
                    match.start(),
                    expression_end,
                ),
                suggested_action=suggested_action,
            )
        return self._apply_text_replacements(code, replacements)

    def _evaluate_template_constraint(
        self,
        constraint: str,
        bindings: Dict[str, str],
        *,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        position: int = 0,
        boolean_templates: Optional[
            Dict[str, List[_MetalBooleanVariableTemplate]]
        ] = None,
        namespace_visibility: Optional[_MetalNamespaceVisibility] = None,
        lookup_position: Optional[int] = None,
    ) -> bool:
        # Evaluate a single SFINAE constraint for the concrete `bindings` and
        # return whether the overload is ENABLED. Raises _UnrecognizedConstraint
        # for any constraint shape outside the recognized set so the caller can
        # clean-fail rather than guess.
        text = _strip_global_metal_qualifiers(constraint.strip())
        # Unwrap an `[metal::]enable_if_t<COND, type>` enabler down to COND; a
        # bare `enable_if_t<COND>` (one argument) unwraps to COND too. The
        # presence of the alias means "enabled iff COND".
        enable_match = re.match(
            r"^(?:typename\s+)?(?:metal\s*::\s*)?enable_if_t\s*<", text
        )
        if enable_match is not None:
            angle_start = text.find("<", enable_match.end() - 1)
            # The enabler's condition may embed a comparison `<` (`sizeof(T) < 8`),
            # so balance the angle and split its arguments with the SFINAE-aware
            # helpers rather than the generic ones.
            angle_end = self._find_matching_template_param_angle(text, angle_start)
            if angle_end is None or text[angle_end + 1 :].strip():
                raise self._UnrecognizedConstraint(constraint)
            arguments = self._split_template_parameter_list(
                text[angle_start + 1 : angle_end]
            )
            if not arguments:
                raise self._UnrecognizedConstraint(constraint)
            return self._evaluate_boolean_constraint(
                arguments[0].strip(),
                bindings,
                structs=structs,
                type_aliases=type_aliases,
                position=position,
                boolean_templates=boolean_templates,
                namespace_visibility=namespace_visibility,
                lookup_position=lookup_position,
            )
        return self._evaluate_boolean_constraint(
            text,
            bindings,
            structs=structs,
            type_aliases=type_aliases,
            position=position,
            boolean_templates=boolean_templates,
            namespace_visibility=namespace_visibility,
            lookup_position=lookup_position,
        )

    def _evaluate_boolean_constraint(
        self,
        expression: str,
        bindings: Dict[str, str],
        *,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        position: int = 0,
        boolean_templates: Optional[
            Dict[str, List[_MetalBooleanVariableTemplate]]
        ] = None,
        namespace_visibility: Optional[_MetalNamespaceVisibility] = None,
        lookup_position: Optional[int] = None,
        variable_stack: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    ) -> bool:
        expr = _strip_global_metal_qualifiers(
            self._replace_identifiers(expression, bindings).strip()
        )
        if not expr:
            raise self._UnrecognizedConstraint(expression)

        def evaluate(nested_expression: str) -> bool:
            return self._evaluate_boolean_constraint(
                nested_expression,
                bindings,
                structs=structs,
                type_aliases=type_aliases,
                position=position,
                boolean_templates=boolean_templates,
                namespace_visibility=namespace_visibility,
                lookup_position=lookup_position,
                variable_stack=variable_stack,
            )

        # Strip a single fully-enclosing paren group.
        while (
            expr.startswith("(")
            and self._find_matching_delimiter(expr, 0, "(", ")") == len(expr) - 1
        ):
            expr = expr[1:-1].strip()
            if not expr:
                raise self._UnrecognizedConstraint(expression)

        if expr == "true":
            return True
        if expr == "false":
            return False

        disjunction = self._split_top_level_boolean_constraint(expr, ("||", "|"))
        if disjunction is not None:
            return any(evaluate(part) for part in disjunction)
        conjunction = self._split_top_level_boolean_constraint(expr, ("&&", "&"))
        if conjunction is not None:
            return all(evaluate(part) for part in conjunction)

        # `a == b` and `a != b` over Boolean operands (`is_complex_v<T> ==
        # is_complex_v<U>`). Equality binds tighter than `&&` and looser than
        # `!`. Operands that are not Boolean constraints, such as
        # `sizeof(T) == 4`, fall through to the integer comparisons below.
        for equality_operator in ("==", "!="):
            operands = self._split_top_level_boolean_constraint(
                expr, (equality_operator,)
            )
            if operands is None or len(operands) != 2:
                continue
            try:
                left_value = evaluate(operands[0])
                right_value = evaluate(operands[1])
            except self._UnrecognizedConstraint:
                break
            if equality_operator == "==":
                return left_value == right_value
            return left_value != right_value

        if expr.startswith("!"):
            return not evaluate(expr[1:])

        combinator = re.match(
            r"^(?:metal\s*::\s*)?(_disjunction|_conjunction)\s*<", expr
        )
        if combinator is not None:
            angle_start = expr.find("<", combinator.start())
            angle_end = self._find_matching_angle(expr, angle_start)
            if angle_end is None or not re.fullmatch(
                r"\s*::\s*value", expr[angle_end + 1 :]
            ):
                raise self._UnrecognizedConstraint(expression)
            arguments = self._split_top_level_commas(expr[angle_start + 1 : angle_end])

            # These internal Metal combinators consume trait classes, not Boolean
            # values. Their lazy evaluation must retain the source operand order.
            def trait_value(argument: str) -> bool:
                return evaluate(f"{argument.strip()}::value")

            if combinator.group(1) == "_disjunction":
                return any(trait_value(argument) for argument in arguments)
            return all(trait_value(argument) for argument in arguments)

        named = re.match(r"^(?P<name>(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*<", expr)
        if named is not None and boolean_templates:
            name = named.group("name")
            if name.rsplit("::", 1)[-1] in boolean_templates:
                angle_start = expr.find("<", named.start())
                angle_end = self._find_matching_angle(expr, angle_start)
                suffix = "" if angle_end is None else expr[angle_end + 1 :].strip()
                member = re.fullmatch(r"::\s*(value)", suffix)
                if angle_end is None or (suffix and member is None):
                    raise self._UnrecognizedConstraint(expression)
                return self._evaluate_boolean_variable_template(
                    name,
                    self._split_top_level_commas(expr[angle_start + 1 : angle_end]),
                    bindings,
                    boolean_templates,
                    variable_stack,
                    structs=structs,
                    type_aliases=type_aliases,
                    position=position,
                    namespace_visibility=namespace_visibility,
                    lookup_position=lookup_position,
                    member=member.group(1) if member else "",
                )

        simdgroup_trait = simdgroup_type_trait(
            self,
            expr,
            lambda operand: self._canonical_type_trait_operand(
                operand,
                lambda text: self._canonicalize_type_aliases_at(
                    self._resolve_constraint_type(text, bindings),
                    type_aliases or {},
                    position if lookup_position is None else lookup_position,
                ),
            ),
            {struct.name for struct in structs or ()},
        )
        if simdgroup_trait is not None:
            if simdgroup_trait[0] is None:
                raise self._UnrecognizedConstraint(expression)
            return simdgroup_trait[0]

        parsed_trait = self._parse_metal_type_trait_expression(expr)
        if parsed_trait is not None:
            trait, operands = parsed_trait
            result, _resolved_operands, _reason = self._evaluate_metal_type_trait(
                trait,
                operands,
                resolver=lambda operand: self._canonicalize_type_aliases_at(
                    self._resolve_constraint_type(operand, bindings),
                    type_aliases or {},
                    position if lookup_position is None else lookup_position,
                ),
                known_type_names=set(),
                allow_unknown_named_types=True,
            )
            if result is None:
                raise self._UnrecognizedConstraint(expression)
            return result

        # `is_integral_v<T>` (optionally `metal::`).
        integral_match = re.fullmatch(
            r"(?:metal\s*::\s*)?is_integral_v\s*<\s*(?P<arg>[^<>]+?)\s*>", expr
        )
        if integral_match is not None:
            concrete = self._resolve_constraint_type(
                integral_match.group("arg"), bindings
            )
            result = self._is_integral_concrete_type(concrete)
            if result is None:
                raise self._UnrecognizedConstraint(expression)
            return result

        # `is_signed_v<T>` (optionally `metal::`).
        signed_match = re.fullmatch(
            r"(?:metal\s*::\s*)?is_signed_v\s*<\s*(?P<arg>[^<>]+?)\s*>", expr
        )
        if signed_match is not None:
            concrete = self._resolve_constraint_type(
                signed_match.group("arg"), bindings
            )
            result = self._is_signed_concrete_type(concrete)
            if result is None:
                raise self._UnrecognizedConstraint(expression)
            return result

        # `sizeof(T) <cmp> N`.
        sizeof_match = re.fullmatch(
            r"sizeof\s*\(\s*(?P<arg>[^()]+?)\s*\)\s*"
            r"(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>[0-9]+)",
            expr,
        )
        if sizeof_match is not None:
            concrete = self._resolve_constraint_type(
                sizeof_match.group("arg"), bindings
            )
            size = self._sizeof_concrete_type(
                concrete,
                structs=structs,
                type_aliases=type_aliases,
                position=position,
            )
            if size is None:
                raise self._UnrecognizedConstraint(expression)
            rhs = int(sizeof_match.group("rhs"))
            return self._compare(size, sizeof_match.group("op"), rhs)

        raise self._UnrecognizedConstraint(expression)

    def _split_top_level_boolean_constraint(
        self, expression: str, operators: Tuple[str, ...]
    ) -> Optional[List[str]]:
        parts: List[str] = []
        depth = 0
        angle_depth = 0
        start = 0
        i = 0
        while i < len(expression):
            ch = expression[i]
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if ch == "<":
                angle_depth += 1
                i += 1
                continue
            if ch == ">":
                angle_depth = max(0, angle_depth - 1)
                i += 1
                continue
            if depth == 0 and angle_depth == 0:
                for operator_text in operators:
                    if expression.startswith(operator_text, i):
                        parts.append(expression[start:i].strip())
                        i += len(operator_text)
                        start = i
                        break
                else:
                    i += 1
                continue
            i += 1
        if not parts:
            return None
        parts.append(expression[start:].strip())
        if any(not part for part in parts):
            raise self._UnrecognizedConstraint(expression)
        return parts

    def _resolve_constraint_type(self, arg: str, bindings: Dict[str, str]) -> str:
        # Resolve a constraint's type operand (`T`) to its concrete binding. An
        # operand that is no template parameter already names a type, which the
        # trait evaluators classify or refuse with `_UnrecognizedConstraint`.
        text = arg.strip()
        if text in bindings:
            return bindings[text]
        return text

    def _compare(self, left: int, op: str, right: int) -> bool:
        comparisons = {
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        return comparisons[op](left, right)

    def _select_constrained_overload(
        self,
        candidates: List[Tuple[_MetalStructMethod, Dict[str, str]]],
        *,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        position: int = 0,
    ) -> Optional[Tuple[_MetalStructMethod, Dict[str, str]]]:
        # Pick the unique overload whose SFINAE constraints ALL evaluate true for
        # that overload's bindings. Returns None when zero or more than one
        # overload is enabled, or when any constraint is unrecognized — every
        # such case is a clean failure (never guess / never mis-select). An
        # overload with NO constraints is considered always-enabled.
        enabled: List[Tuple[_MetalStructMethod, Dict[str, str]]] = []
        for overload, bindings in candidates:
            try:
                if all(
                    self._evaluate_template_constraint(
                        constraint,
                        bindings,
                        structs=structs,
                        type_aliases=type_aliases,
                        position=position,
                    )
                    for constraint in overload.template_constraints
                ):
                    enabled.append((overload, bindings))
            except self._UnrecognizedConstraint:
                return None
        if len(enabled) == 1:
            return enabled[0]
        return None
