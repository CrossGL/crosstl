"""Type calls to source functions whose every declaration states one return type.

Constrained overload resolution needs the type of each call argument. A call
``helper(i, v)`` to a function declared in the translation unit has the return
type of the overload that overload resolution selects ([over.match]). When every
declaration of that name, template or not and in every namespace, states the same
concrete, non-dependent return type, that type is the call's type whichever
overload is selected, so it is known without resolving the call.

Names are refused, and the call stays untyped, when any declaration has a
dependent, deduced (``auto``/``decltype``), reference, pointer or ``void`` return
type; when two declarations disagree; when a declaration the definition scan does
not account for exists (a prototype or an explicit instantiation); and when the
name is also a Metal standard library function, whose overloads are not in the
source. Member functions are not candidates: calls are typed only in free
function bodies, where unqualified lookup does not find members, and a qualified
callee must be qualified by source namespaces only.
"""

from __future__ import annotations

import functools
import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, Mapping, Sequence

from .alias_template_types import (
    expand_alias_template_types,
    find_alias_templates,
    fold_expression_decltypes,
)

__all__ = [
    "CallResultScope",
    "CallResultTypes",
    "collect_call_result_types",
    "infer_call_result_type",
    "metal_library_function_names",
]

_CALLEE_RE = re.compile(
    r"(?P<callee>(?:::\s*)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)\s*(?=[<(])"
)

# Metal Shading Language Specification functions that the importer's intrinsic
# tables do not list (integer functions, section 6.4; synchronization, 6.9;
# geometric, 6.6; reinterpretation, 2.19) and families named by prefix.
_EXTRA_LIBRARY_FUNCTIONS = frozenset(
    {
        "abs_diff",
        "addsat",
        "as_type",
        "hadd",
        "mad24",
        "madhi",
        "madsat",
        "mul24",
        "mulhi",
        "refract",
        "rhadd",
        "rotate",
        "simdgroup_barrier",
        "subsat",
        "threadgroup_barrier",
    }
)
_LIBRARY_PREFIXES = ("atomic_", "pack_", "quad_", "simd_", "unpack_")


@functools.lru_cache(maxsize=1)
def metal_library_function_names() -> frozenset[str]:
    """Unqualified names of Metal standard library functions."""
    # Imported lazily: the converter module imports the preprocessor.
    from .MetalCrossGLCodeGen import MetalToCrossGLConverter
    from .namespaced_functions import METAL_LIBRARY_MATH_FUNCTIONS

    names = set(METAL_LIBRARY_MATH_FUNCTIONS) | _EXTRA_LIBRARY_FUNCTIONS
    for table in (
        MetalToCrossGLConverter.metal_math_intrinsics,
        MetalToCrossGLConverter.metal_bit_intrinsics,
        MetalToCrossGLConverter.metal_wave_intrinsics,
        MetalToCrossGLConverter.metal_atomic_intrinsics,
    ):
        names.update(table)
    return frozenset(names)


def _is_library_name(name: str) -> bool:
    return name in metal_library_function_names() or name.startswith(_LIBRARY_PREFIXES)


@dataclass(frozen=True)
class CallResultTypes:
    """Return types by unqualified function name, and the source namespaces."""

    types: Mapping[str, str]
    namespaces: frozenset[str]


EMPTY_CALL_RESULT_TYPES = CallResultTypes(types={}, namespaces=frozenset())


def _declared_return_type(
    preprocessor, name: str, source: str, template_parameters: Sequence[str]
) -> str | None:
    # The materialized-helper rule already extracts one concrete, non-dependent
    # value return type from a declaration header; an empty substitution keeps
    # every template parameter unbound, so a dependent spelling is refused.
    declaration = SimpleNamespace(
        name=name, source=source, template_parameters=list(template_parameters)
    )
    declared = preprocessor._materialized_template_function_return_type(declaration, {})
    if declared is None and not template_parameters and "enable_if" in source:
        # A materialized instantiation keeps its return-type enabler until the end of preprocessing; once its
        # traits fold it names one type (return_type_enablers.py). An enabler that does not fold stays untyped.
        from .return_type_enablers import fold_concrete_enablers

        try:
            folded = fold_concrete_enablers(
                preprocessor, preprocessor._fold_concrete_type_trait_values(source)
            )
        except ValueError:
            return None
        if folded != source:
            declaration.source = folded
            declared = preprocessor._materialized_template_function_return_type(
                declaration, {}
            )
    if declared is None and not template_parameters and "decltype" in source:
        # A dependent result type leaves a decltype of built-in operands in each instantiation
        # (dependent_result_templates.py) until preprocessing folds it.
        folded = fold_expression_decltypes(preprocessor, source)
        if folded != source:
            declaration.source = folded
            declared = preprocessor._materialized_template_function_return_type(
                declaration, {}
            )
    return declared


def _definition_header_start(definition, namespace_spans) -> int:
    # A declaration scan stops at the previous `;` or `}`, so the first
    # definition in a namespace starts after the namespace's opening brace.
    start = definition.span[0]
    for namespace_start, namespace_end, _name in namespace_spans:
        if namespace_start <= definition.body_span[0] < namespace_end:
            start = max(start, namespace_start)
    return start


def collect_call_result_types(preprocessor, code: str) -> CallResultTypes:
    """Index the agreed return type of every free function declared in ``code``."""
    templates = preprocessor._find_template_functions(code)
    aggregate_spans = sorted(
        [struct.span for struct in preprocessor._find_concrete_struct_definitions(code)]
        + [struct.span for struct in preprocessor._find_template_structs(code)]
    )
    template_spans = preprocessor._find_template_declaration_spans(code)
    definitions = preprocessor._find_non_template_function_definitions(
        code, template_spans
    )
    namespace_visibility = preprocessor._metal_namespace_visibility(code)
    namespace_spans = list(namespace_visibility.namespace_spans)

    declared: dict[str, list[str | None]] = {}
    template_bodies: list[tuple[int, int]] = []
    alias_templates = None
    for template in templates:
        template_bodies.append((template.body_start, template.span[1]))
        if preprocessor._containing_span(template.span[0], aggregate_spans):
            continue
        declared.setdefault(template.name, []).append(
            _declared_return_type(
                preprocessor,
                template.name,
                template.source,
                template.template_parameters,
            )
        )
    for definition in definitions:
        start = _definition_header_start(definition, namespace_spans)
        header = code[start : definition.body_span[0] - 1]
        return_type = _declared_return_type(
            preprocessor, definition.name, header + "{}", ()
        )
        if return_type is None and "<" in header:
            # A materialized instantiation can name its type through an alias template (alias_template_types.py).
            if alias_templates is None:
                alias_templates = find_alias_templates(preprocessor, code)
            expanded = expand_alias_template_types(
                preprocessor, header, alias_templates
            )
            if expanded != header:
                return_type = _declared_return_type(
                    preprocessor, definition.name, expanded + "{}", ()
                )
        declared.setdefault(definition.name, []).append(return_type)

    names = sorted(
        name
        for name, return_types in declared.items()
        if None not in return_types
        and len(set(return_types)) == 1
        and not _is_library_name(name)
    )
    if not names:
        return CallResultTypes(types={}, namespaces=_namespace_names(namespace_spans))

    # Every declaration of a name, prototypes included, must be one counted
    # above; otherwise an overload whose return type was not read could exist.
    # An explicit specialization declares a specialization of its primary
    # template, return type included ([temp.expl.spec]/1), so it adds none.
    explicit_spans = [
        specialization["span"]
        for specialization in (
            preprocessor._find_explicit_template_function_specializations(code).values()
        )
    ]
    references = preprocessor._find_static_constexpr_calls(code, set(names))
    declarations = preprocessor._find_free_function_overload_declarations(
        code,
        references,
        excluded_spans=sorted(
            aggregate_spans
            + [definition.body_span for definition in definitions]
            + template_bodies
            + explicit_spans
        ),
        template_spans=template_spans,
        namespace_visibility=namespace_visibility,
    )
    types = {
        name: declared[name][0]
        for name in names
        if len(declarations.get(name, [])) == len(declared[name])
    }
    return CallResultTypes(types=types, namespaces=_namespace_names(namespace_spans))


def _namespace_names(namespace_spans) -> frozenset[str]:
    names = set()
    for _start, _end, name in namespace_spans:
        names.update(part for part in str(name or "").split("::") if part)
    return frozenset(names)


@dataclass
class CallResultScope:
    """The source whose function calls argument inference may type.

    The index is built on first use, so a pass that never meets an untyped call
    argument does not scan the source's declarations. A dataclass, so that state
    memoization can compare preprocessors that hold one.
    """

    source: str | None = None
    indexed: CallResultTypes | None = None

    @contextmanager
    def active(self, source: str) -> Iterator[None]:
        previous = (self.source, self.indexed)
        self.source, self.indexed = source, None
        try:
            yield
        finally:
            self.source, self.indexed = previous

    def types(self, preprocessor) -> CallResultTypes:
        if self.source is None:
            return EMPTY_CALL_RESULT_TYPES
        if self.indexed is None:
            self.indexed = collect_call_result_types(preprocessor, self.source)
        return self.indexed


def _call_shape(preprocessor, expr: str) -> list[str] | None:
    """The callee's ``::`` components when ``expr`` is exactly one call."""
    match = _CALLEE_RE.match(expr)
    if match is None:
        return None
    position = match.end()
    if expr[position] == "<":
        angle_end = preprocessor._find_matching_template_param_angle(expr, position)
        if angle_end is None:
            return None
        position = angle_end + 1
        while position < len(expr) and expr[position].isspace():
            position += 1
        if position >= len(expr) or expr[position] != "(":
            return None
    paren_end = preprocessor._find_matching_delimiter(expr, position, "(", ")")
    if paren_end != len(expr) - 1:
        return None
    parts = [part.strip() for part in match.group("callee").split("::")]
    return [part for part in parts if part]


def infer_call_result_type(
    preprocessor,
    expr: str,
    scope: CallResultScope,
    local_variable_types: Mapping[str, str],
    structs_by_name: Mapping[str, object] | None = None,
) -> str | None:
    """The type of ``expr`` when it is one call to an indexed source function."""
    parts = _call_shape(preprocessor, expr)
    if not parts:
        return None
    name = parts[-1]
    if len(parts) == 1 and name in local_variable_types:
        return None
    if structs_by_name and name in structs_by_name:
        return None
    call_result_types = scope.types(preprocessor)
    return_type = call_result_types.types.get(name)
    if return_type is None:
        return None
    if any(qualifier not in call_result_types.namespaces for qualifier in parts[:-1]):
        return None
    return preprocessor._normalize_inferred_expression_type(return_type)
