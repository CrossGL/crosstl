"""The value type of a call to a source function template, from the template arguments the call deduces.

``auto q = twice(v);`` calls ``template <typename T> T twice(T x)``. The local has the type of the selected
specialization's return type ([dcl.spec.auto]/4), which is the declared return type with the deduced template
arguments substituted ([temp.deduct]/5). The importer selects the template and records the argument types, but the
declared spelling still names the callee's parameter, so ``T q`` would land in a function that has no ``T``.

``deduced_call_return_type`` deduces a template parameter from each function parameter declared as exactly that
parameter (``T``, ``const T`` or ``thread const T&``), as [temp.deduct.call]/2-3 does once the top-level cv-qualifiers
and the reference are dropped, and substitutes the deduced types into the return type. A ``decltype`` of built-in
arithmetic that the substitution makes concrete is folded to its type (``alias_template_types.py``).

The return type is refused, as dependent, when it still names a parameter of the callee that this deduction did not
bind, or when two arguments deduce different types for one parameter ([temp.deduct.call]/4 allows no conversion
there).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .alias_template_types import fold_expression_decltypes

__all__ = [
    "deduce_call_return_type",
    "deduced_call_return_type",
    "expression_call_return_type",
    "same_arity_overloads",
]

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_PARAMETER_QUALIFIERS_RE = re.compile(r"\b(?:const|volatile|thread)\b")
_ARGUMENT_QUALIFIERS_RE = re.compile(r"\b(?:const|volatile)\b")


def _template_parameter_names(function: Any) -> tuple[str, ...]:
    return tuple(
        name
        for _kind, name in getattr(function, "template_parameters", None) or ()
        if name
    )


def _bare_parameter(parameter_type: str | None, names: Sequence[str]) -> str | None:
    """The template parameter a function parameter is declared as, ignoring cv-qualifiers and ``&``."""

    if not parameter_type:
        return None
    text = _PARAMETER_QUALIFIERS_RE.sub(" ", str(parameter_type)).strip()
    if text.endswith("&") and not text.endswith("&&"):
        text = text[:-1].strip()
    return text if text in names else None


def _argument_value_type(argument_type: str | None) -> str | None:
    if not argument_type:
        return None
    text = _ARGUMENT_QUALIFIERS_RE.sub(" ", str(argument_type)).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or any(token in text for token in ("&", "*", "[", "auto")):
        return None
    return text


def _deduced_bindings(
    converter: Any, function: Any, arguments: Sequence[Any], names: Sequence[str]
) -> tuple[dict[str, str], bool]:
    """Bindings deduced from bare parameters, and whether two arguments disagree."""

    bindings: dict[str, str] = {}
    for parameter, argument in zip(getattr(function, "params", None) or (), arguments):
        name = _bare_parameter(
            converter.metal_declaration_expression_type(parameter), names
        )
        if name is None:
            continue
        argument_type = _argument_value_type(converter.expression_metal_type(argument))
        if argument_type is None:
            continue
        previous = bindings.setdefault(name, argument_type)
        if converter.normalized_metal_type(previous) != converter.normalized_metal_type(
            argument_type
        ):
            return bindings, True
    return bindings, False


def _explicit_template_arguments(converter: Any, function: Any, call: Any) -> list[str]:
    """The explicit template-argument list of ``call`` when it names ``function`` ([temp.arg.explicit])."""

    text = str(getattr(call, "name", "") or "").strip()
    if "<" not in text or not text.endswith(">"):
        return []
    base = text[: text.find("<")].strip().rsplit("::", 1)[-1].strip()
    if base != str(getattr(function, "name", "")).rsplit("::", 1)[-1].strip():
        return []
    return [
        str(argument).strip()
        for argument in converter.split_generic_arguments(text[text.find("<") + 1 : -1])
    ]


def _call_bindings(
    converter: Any,
    function: Any,
    arguments: Sequence[Any],
    names: Sequence[str],
    call: Any,
) -> tuple[dict[str, str], bool]:
    """Template arguments of a call: explicit, then deduced, then declaration defaults ([temp.deduct]/2-5).

    An explicitly specified parameter is not deduced, so its function parameters accept converted arguments. A default
    may name earlier parameters ([temp.param]/14) and is substituted with the bindings made so far. An explicit list
    longer than the parameter list, or one for a template with a pack, binds nothing explicitly.
    """

    kinds = [
        str(kind)
        for kind, name in getattr(function, "template_parameters", None) or ()
        if name
    ]
    explicit = (
        _explicit_template_arguments(converter, function, call)
        if call is not None
        else []
    )
    if len(explicit) > len(names) or any(kind.endswith("...") for kind in kinds):
        explicit = []
    bindings = dict(zip(names, explicit))
    deduced, conflicting = _deduced_bindings(
        converter, function, arguments, [name for name in names if name not in bindings]
    )
    bindings.update(deduced)
    if conflicting:
        return bindings, True
    defaults = getattr(function, "template_parameter_defaults", None) or {}
    for name in names:
        if name not in bindings and defaults.get(name):
            bindings[name] = converter_preprocessor(converter)._replace_identifiers(
                str(defaults[name]), bindings
            )
    return bindings, False


def same_arity_overloads(converter: Any, call: Any) -> list[Any]:
    """The source functions ``call`` can name, qualified or with explicit template arguments, of its arity."""

    name = str(getattr(call, "name", "") or "").strip()
    candidates = converter.metal_user_function_overloads(name)
    if not candidates and "<" in name and name.endswith(">"):
        candidates = [
            candidate
            for candidate in converter.metal_user_function_overloads(
                name[: name.find("<")].strip()
            )
            if _template_parameter_names(candidate)
        ]
    arity = len(getattr(call, "args", None) or ())
    return [
        candidate
        for candidate in candidates
        if len(getattr(candidate, "params", None) or ()) == arity
    ]


def _decltype_parameters(function: Any, return_type: str) -> tuple[Any, ...]:
    """The callee's function parameters a ``decltype`` in ``return_type`` names, as a trailing return type can."""

    if "decltype" not in return_type:
        return ()
    words = set(_IDENTIFIER_RE.findall(return_type))
    return tuple(
        parameter
        for parameter in getattr(function, "params", None) or ()
        if getattr(parameter, "name", None) in words
    )


def _callee_scope_type(
    converter: Any, function: Any, return_type: str, bindings: dict[str, str]
) -> str | None:
    """``return_type`` resolved where the callee's parameters, not the caller's names, are in scope ([basic.scope.param])."""

    scope: dict[str, str] = {}
    for parameter in getattr(function, "params", None) or ():
        declared = converter.metal_declaration_expression_type(parameter)
        name = getattr(parameter, "name", None)
        if not declared or not name:
            return None
        scope[str(name)] = converter_preprocessor(converter)._replace_identifiers(
            str(declared), bindings
        )
    saved_types = converter.current_variable_types
    saved_parameters = getattr(converter, "current_parameter_scope", None)
    converter.current_variable_types = {**converter.global_variable_types, **scope}
    converter.current_parameter_scope = None
    try:
        resolved = converter.resolve_metal_decltype_type(return_type)
    finally:
        converter.current_variable_types = saved_types
        converter.current_parameter_scope = saved_parameters
    if resolved is None or "decltype" in resolved:
        return None
    return resolved


def deduce_call_return_type(
    converter: Any,
    function: Any,
    arguments: Sequence[Any],
    return_type: str | None,
    call: Any = None,
) -> tuple[str | None, tuple[str, ...], bool]:
    """``return_type`` with the template arguments of a call, the parameters left unbound, and a conflict.

    The arguments are those ``call`` lists explicitly, then the ones deduced from ``arguments``, then the defaults
    (``_call_bindings``).

    The type is None when a parameter the return type names stays unbound or two arguments disagree. A ``decltype``
    of the callee's own parameters is resolved with those parameters in scope; one that does not resolve leaves them
    unbound.
    """

    if not return_type:
        return return_type, (), False
    names = _template_parameter_names(function)
    scoped = _decltype_parameters(function, return_type)
    referenced = [
        name for name in names if re.search(rf"\b{re.escape(name)}\b", return_type)
    ]
    for parameter in scoped:
        declared = str(converter.metal_declaration_expression_type(parameter) or "")
        referenced.extend(
            name
            for name in names
            if name not in referenced and re.search(rf"\b{re.escape(name)}\b", declared)
        )
    if not referenced and not scoped:
        return return_type, (), False
    bindings, conflicting = _call_bindings(converter, function, arguments, names, call)
    unresolved = tuple(name for name in referenced if name not in bindings)
    if conflicting or unresolved:
        return None, unresolved, conflicting
    substituted = converter_preprocessor(converter)._replace_identifiers(
        return_type, bindings
    )
    if scoped:
        resolved = _callee_scope_type(converter, function, substituted, bindings)
        if resolved is None:
            return None, tuple(str(parameter.name) for parameter in scoped), False
        return resolved, (), False
    return (
        fold_expression_decltypes(converter_preprocessor(converter), substituted),
        (),
        False,
    )


def expression_call_return_type(
    converter: Any,
    function: Any,
    arguments: Sequence[Any],
    return_type: str | None,
    call: Any = None,
) -> str | None:
    """The value type of a call expression, or None while it depends on the callee's parameters."""

    return deduce_call_return_type(converter, function, arguments, return_type, call)[0]


def deduced_call_return_type(
    converter: Any,
    declaration: Any,
    initializer: Any,
    function: Any,
    return_type: str | None,
) -> str | None:
    """The type an ``auto`` local initialized by a call to ``function`` has; refused while it stays dependent."""

    deduced, unresolved, conflicting = deduce_call_return_type(
        converter,
        function,
        getattr(initializer, "args", None) or (),
        return_type,
        initializer,
    )
    if not conflicting and not unresolved:
        return deduced
    from .MetalCrossGLCodeGen import MetalAutoTypeInferenceError

    reason = (
        "the call deduces conflicting types for one template parameter"
        if conflicting
        else "the selected callable return type remains dependent on "
        + ", ".join(unresolved)
    )
    raise MetalAutoTypeInferenceError(
        declaration.name,
        function.name,
        return_type,
        reason,
        getattr(declaration, "source_location", None)
        or getattr(initializer, "source_location", None),
        unresolved,
    )


def converter_preprocessor(converter: Any) -> Any:
    """The Metal preprocessor whose type rules fold ``decltype`` for the converter."""

    preprocessor = getattr(converter, "_decltype_folding_preprocessor", None)
    if preprocessor is None:
        from .preprocessor import MetalPreprocessor

        preprocessor = converter._decltype_folding_preprocessor = MetalPreprocessor()
    return preprocessor
