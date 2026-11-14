"""User function templates that overload a Metal SIMD-group or quad-group function.

Metal sources overload the library's SIMD-group functions for their own aggregates, as in::

    template <typename U>
    Pair<U> simd_shuffle_down(Pair<U> data, ushort delta) {
      return Pair<U>{simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)};
    }

C++ resolves every call of that name per instantiation. A ``Pair<float>`` argument deduces the template, while
``data.index`` (``uint``) and ``data.val`` (``float``) fail its deduction and reach ``metal::simd_shuffle_down``
(MSL 4, section 6.9.2; [temp.deduct]/8: deduction failure removes the candidate). The CrossGL generic path decides a
name once, for every instantiation: it maps a library spelling to the canonical ``Wave*`` operation and keeps a user
spelling as a call of the user function. A call whose overload set holds both is bound wrongly either way, so the
aggregate call reached the library operation and the calls in the template body called the template again.

The constrained call pass already decides each call of an ``enable_if``-constrained template per instantiation: it
deduces the candidates, materializes the one that applies and leaves a call whose candidates all fail deduction to
the library. A template of a SIMD-group function's name joins that pass as a template without constraints, so its
deduction is its only condition (``overloads_simd_group_function``). It does so only when preprocessing the source
the CrossGL importer reads. The project materializer instantiates such templates for the targets without templates
itself and records each specialization, so its own runs of the pass leave them alone. Where the pass cannot type an
argument, it leaves the call and the templates it can see for a later binder (``only_simd_group_overloads``).

When such an enabled template competes with the library function, ``simd_group_library_overload_selected`` ranks
the two by the library's declarations (``metal_simdgroup`` and ``metal_quadgroup`` of the Metal toolchain, compiler
32023). Each one is a template ``T f(T data, ushort delta)`` that is enabled only for a scalar or vector of ``half``,
``float``, ``double``, ``char``, ``uchar``, ``short``, ``ushort``, ``int`` or ``uint`` (only the integral ones for
``and``, ``or`` and ``xor``). ``all``, ``any`` and ``ballot`` are non-template functions of ``bool``, and
``active_threads_mask``, ``is_first`` and ``is_helper_thread`` non-template functions without parameters.

* An argument of any other type, such as a struct or ``bool`` data, leaves no viable library function, so the user
  template is selected.
* Otherwise the arguments are compared one by one ([over.match.best]/2.1): the library takes its data arguments with
  the identity conversion and the others as declared. The function that takes some argument by identity where the
  other converts it, and converts none that the other takes by identity, is the better one. An argument both take as
  the same type ties.
* When every argument ties, a non-template library function wins ([over.match.best]/2.4) and two templates are
  ordered by partial ordering, which is not modeled. That, mixed outcomes, references and conversions to different
  types leave the precedence unproven, and the call is refused. The Metal compiler reports ``simd_sum(x)`` beside a
  user ``template <typename T> T simd_sum(T)`` as ambiguous.

The pass types the arguments in a materialized body from that body's own declarations. Template materialization
spells a struct template-id such as ``Pair<float>`` as written, while the source already declares its materialized
struct ``Pair_float`` with the same members, so ``canonicalize_struct_template_ids`` spells each such template-id as
the materialized struct before its body is scanned.
"""

import re
from typing import Any, Collection, List, Mapping, Optional, Sequence, Tuple

from .static_member_constants import materialized_struct_for_template_id

__all__ = [
    "SIMD_GROUP_FUNCTION_NAMES",
    "canonicalize_struct_template_ids",
    "only_simd_group_overloads",
    "overloads_simd_group_function",
    "simd_group_library_overload_selected",
]

# MSL 4, section 6.9.2 (SIMD-group functions) and section 6.9.3 (quad-group functions).
_GROUP_OPERATIONS = (
    "active_threads_mask",
    "all",
    "and",
    "any",
    "ballot",
    "broadcast",
    "broadcast_first",
    "is_first",
    "is_helper_thread",
    "max",
    "min",
    "or",
    "prefix_exclusive_product",
    "prefix_exclusive_sum",
    "prefix_inclusive_product",
    "prefix_inclusive_sum",
    "product",
    "shuffle",
    "shuffle_and_fill_down",
    "shuffle_and_fill_up",
    "shuffle_down",
    "shuffle_rotate_down",
    "shuffle_rotate_up",
    "shuffle_up",
    "shuffle_xor",
    "sum",
    "xor",
)

SIMD_GROUP_FUNCTION_NAMES = frozenset(
    f"{prefix}_{operation}"
    for prefix in ("simd", "quad")
    for operation in _GROUP_OPERATIONS
)

_DATA = "T"
_DATA_ELEMENTS = frozenset(
    {"half", "float", "double", "char", "uchar", "short", "ushort", "int", "uint"}
)
_INTEGRAL_ELEMENTS = frozenset({"char", "uchar", "short", "ushort", "int", "uint"})
_SCALAR_ELEMENTS = _DATA_ELEMENTS | {"bool", "long", "ulong", "bfloat"}
_DATA_WIDTHS = frozenset({1, 2, 3, 4, 8, 16})
_INTEGRAL_OPERATIONS = frozenset({"and", "or", "xor"})
_BOOLEAN_OPERATIONS = ("all", "any", "ballot")
_NULLARY_OPERATIONS = ("active_threads_mask", "is_first", "is_helper_thread")
_NON_TEMPLATE_OPERATIONS = frozenset(_BOOLEAN_OPERATIONS + _NULLARY_OPERATIONS)
_SHUFFLES = ("broadcast", "shuffle", "shuffle_down", "shuffle_up", "shuffle_xor")
_SIGNATURES = {
    **{
        operation: ((_DATA,),)
        for operation in (
            "and",
            "broadcast_first",
            "max",
            "min",
            "or",
            "prefix_exclusive_product",
            "prefix_exclusive_sum",
            "prefix_inclusive_product",
            "prefix_inclusive_sum",
            "product",
            "sum",
            "xor",
        )
    },
    **{
        operation: ((_DATA, "ushort"),)
        for operation in (*_SHUFFLES, "shuffle_rotate_down", "shuffle_rotate_up")
    },
    **{
        operation: ((_DATA, _DATA, "ushort"), (_DATA, _DATA, "ushort", "ushort"))
        for operation in ("shuffle_and_fill_down", "shuffle_and_fill_up")
    },
    **{operation: (("bool",),) for operation in _BOOLEAN_OPERATIONS},
    **{operation: ((),) for operation in _NULLARY_OPERATIONS},
}

_TEMPLATE_ID_START_RE = re.compile(r"(?<![A-Za-z0-9_:.])[A-Za-z_]\w*\s*<")


def overloads_simd_group_function(template: Any) -> bool:
    """Whether a free function template without constraints overloads a SIMD-group or quad-group function."""
    return (
        not getattr(template, "template_constraints", None)
        and template.name in SIMD_GROUP_FUNCTION_NAMES
    )


def only_simd_group_overloads(candidates: Sequence[Any]) -> bool:
    """Whether every candidate is such a template, so no ``enable_if`` contract depends on binding the call here."""
    return bool(candidates) and all(
        overloads_simd_group_function(template) for template in candidates
    )


def _scalar_and_width(spelling: str) -> Tuple[str, int]:
    match = re.fullmatch(r"([a-z]+?)(\d*)", spelling)
    if match is None or not match.group(2):
        return spelling, 1
    return match.group(1), int(match.group(2))


def _library_takes(name: str, kind: str, spelling: Optional[str]) -> Optional[bool]:
    """Whether the library parameter ``kind`` takes an argument spelled ``spelling``; ``None`` when unknown."""
    if spelling is None:
        # A class could convert to a fixed parameter type; data is deduced and must be a valid type.
        return False if kind == _DATA else None
    element, width = _scalar_and_width(spelling)
    if kind == _DATA:
        elements = (
            _INTEGRAL_ELEMENTS
            if name.split("_", 1)[1] in _INTEGRAL_OPERATIONS
            else _DATA_ELEMENTS
        )
        return element in elements and width in _DATA_WIDTHS
    # A scalar converts to bool and to ushort; a vector converts to neither.
    return width == 1 and element in _SCALAR_ELEMENTS


def _compare(
    name: str,
    kinds: Sequence[str],
    arguments: Sequence[Optional[str]],
    template_parameters: Sequence[Optional[str]],
) -> str:
    """``"template"``, ``"library"`` or ``"unproven"`` for one viable library declaration."""
    template_better = library_better = False
    for kind, argument, parameter in zip(kinds, arguments, template_parameters):
        library_identity = kind == _DATA or kind == argument
        if parameter is None or parameter.endswith("&") or argument is None:
            if library_identity and parameter is None and argument is not None:
                library_better = True
                continue
            return "unproven"
        template_identity = parameter == argument
        if template_identity and not library_identity:
            template_better = True
        elif library_identity and not template_identity:
            library_better = True
        elif not template_identity and parameter != kind:
            return "unproven"
    if template_better and not library_better:
        return "template"
    if library_better and not template_better:
        return "library"
    if not template_better and name.split("_", 1)[1] in _NON_TEMPLATE_OPERATIONS:
        return "library"
    return "unproven"


def simd_group_library_overload_selected(
    preprocessor: Any,
    code: str,
    call: Any,
    argument_types: Sequence[Optional[str]],
    template: Any,
    bindings: Mapping[str, str],
) -> Optional[bool]:
    """Whether the library function, not ``template``, is selected; ``None`` when the template is no such overload."""
    if not overloads_simd_group_function(template):
        return None
    arguments = [
        (preprocessor._library_argument_types([argument_type]) or [None])[0]
        for argument_type in argument_types
    ]
    parameters = preprocessor._template_function_parameter_text(template) or ""
    substitutions = {name: str(value) for name, value in dict(bindings).items()}
    template_parameters = [
        preprocessor._library_parameter_type(
            preprocessor._normalize_function_parameter_type_text(
                preprocessor._replace_identifiers(parameter, substitutions)
            )
            or ""
        )
        for parameter in preprocessor._split_top_level_commas(parameters)
        if parameter.strip()
    ]
    operation = call.name.split("_", 1)[1]
    selected = False
    for kinds in _SIGNATURES.get(operation, ()):
        if len(kinds) != len(arguments):
            continue
        takes = [
            _library_takes(call.name, kind, argument)
            for kind, argument in zip(kinds, arguments)
        ]
        data = {argument for kind, argument in zip(kinds, arguments) if kind == _DATA}
        if False in takes or len(data) > 1:
            # Not viable: a rejected argument, or data arguments that deduce different types.
            continue
        outcome = (
            "unproven"
            if None in takes or len(template_parameters) != len(arguments)
            else _compare(call.name, kinds, arguments, template_parameters)
        )
        if outcome == "unproven":
            preprocessor._raise_constrained_free_function_error(
                code,
                call,
                f"the template competes with the Metal library's '{call.name}', and which one "
                "overload resolution selects is unproven",
            )
        selected = selected or outcome == "library"
    return selected


def canonicalize_struct_template_ids(
    preprocessor: Any, text: str, struct_names: Collection[str]
) -> str:
    """Spell each template-id in ``text`` that names a materialized struct in ``struct_names`` as that struct."""
    if "<" not in text:
        return text
    ignored = preprocessor._find_comment_and_literal_spans(text)
    replacements: List[Tuple[int, int, str]] = []
    for match in _TEMPLATE_ID_START_RE.finditer(text):
        if preprocessor._containing_span(match.start(), ignored) is not None:
            continue
        angle_close = preprocessor._find_matching_angle(text, match.end() - 1)
        if angle_close is None:
            continue
        concrete = materialized_struct_for_template_id(
            preprocessor, text[match.start() : angle_close + 1]
        )
        if concrete is not None and concrete in struct_names:
            replacements.append((match.start(), angle_close + 1, concrete))
    if not replacements:
        return text
    return preprocessor._apply_text_replacements(text, replacements)
