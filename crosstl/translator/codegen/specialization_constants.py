"""Target-neutral analysis of CrossGL specialization constants.

A CrossGL ``const T name @constant_id(N) = value;`` is a specialization constant: the host may replace ``value``
when it creates the pipeline (SPIR-V ``Decoration SpecId N``, GLSL ``layout(constant_id = N)``). A program constant
whose initializer reads one, such as ``const int doubled = scale * 2 + 1;``, follows the specialized value too, as
``OpSpecConstantOp`` does in SPIR-V; it is *dependent*.

The generators for targets with their own specialization model share this analysis:

* :func:`specialization_constant_ids` reads the id of every specialization constant and refuses a malformed, negative,
  out-of-range or duplicated one;
* :func:`specialization_dependent_constants` names the constants that follow a specialization constant;
* :func:`array_sizes_reading` and :func:`case_labels_reading` find the places a target may require a constant
  expression that its specialization constants are not.

Each refusal is a :class:`SpecializationConstantError`, a typed diagnostic naming the target and the constant.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from ..ast import ArrayType, ASTNode, CaseNode, IdentifierNode, LiteralNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "SPECIALIZATION_ID_ATTRIBUTES",
    "SpecializationConstantError",
    "array_sizes_reading",
    "case_labels_reading",
    "declares_specialization_id",
    "identifier_names",
    "walk_nodes",
    "specialization_constant_ids",
    "specialization_dependent_constants",
]

# The CrossGL attribute and its Metal spelling; a ``glsl_``, ``metal_`` or ``msl_`` prefix names the same attribute.
SPECIALIZATION_ID_ATTRIBUTES = frozenset({"constant_id", "function_constant"})
_ATTRIBUTE_PREFIXES = ("glsl_", "metal_", "msl_")


class SpecializationConstantError(TranslationDiagnostic):
    """A specialization constant that a target cannot express, or that the source misstates."""

    code = CODE_PREFIX + "specialization-constant-unsupported"

    def __init__(
        self,
        reason: str,
        *,
        target: str,
        constant_name: str | None = None,
        specialization_id: int | None = None,
        kind: str = "unsupported",
    ) -> None:
        self.constant_name = constant_name
        self.specialization_id = specialization_id
        self.kind = kind
        self.missing_capabilities = (
            f"{target.lower()}.specialization-constant-{kind}",
        )
        super().__init__(reason, target=target)


def _attribute_name(attribute) -> str:
    name = str(getattr(attribute, "name", attribute) or "").lower().replace("-", "_")
    for prefix in _ATTRIBUTE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def declares_specialization_id(node) -> bool:
    """Whether ``node`` carries a specialization id attribute, whatever its arguments are."""
    return any(
        _attribute_name(attribute) in SPECIALIZATION_ID_ATTRIBUTES
        for attribute in getattr(node, "attributes", None) or []
    )


def _integer_id(argument) -> int | None:
    raw = argument.value if isinstance(argument, LiteralNode) else argument
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        text = raw.strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def _constant_id(node, target: str, max_id: int) -> int | None:
    name = getattr(node, "name", None) or "<unnamed>"
    found = None
    for attribute in getattr(node, "attributes", None) or []:
        if _attribute_name(attribute) not in SPECIALIZATION_ID_ATTRIBUTES:
            continue
        arguments = getattr(attribute, "arguments", getattr(attribute, "args", None))
        arguments = list(arguments or [])
        spec_id = _integer_id(arguments[0]) if len(arguments) == 1 else None
        if spec_id is None:
            raise SpecializationConstantError(
                f"specialization constant '{name}' needs one non-negative integer id",
                target=target,
                constant_name=name,
                kind="invalid-id",
            )
        if spec_id > max_id:
            raise SpecializationConstantError(
                f"specialization constant '{name}' has id {spec_id}; {target} specialization ids are at "
                f"most {max_id}",
                target=target,
                constant_name=name,
                specialization_id=spec_id,
                kind="id-out-of-range",
            )
        if found is not None and found != spec_id:
            raise SpecializationConstantError(
                f"specialization constant '{name}' names two ids, {found} and {spec_id}",
                target=target,
                constant_name=name,
                specialization_id=spec_id,
                kind="invalid-id",
            )
        found = spec_id
    return found


def specialization_constant_ids(
    constants: Iterable, target: str, max_id: int
) -> dict[str, int]:
    """Map the name of each specialization constant among ``constants`` to its id, in declaration order.

    A pipeline sets a specialization constant by id, so two constants with one id cannot be set apart and are
    refused, as glslang and naga refuse them.
    """
    ids: dict[str, int] = {}
    owners: dict[int, str] = {}
    for node in constants or []:
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            continue
        spec_id = _constant_id(node, target, max_id)
        if spec_id is None:
            continue
        if spec_id in owners:
            raise SpecializationConstantError(
                f"specialization constants '{owners[spec_id]}' and '{name}' share id {spec_id}",
                target=target,
                constant_name=name,
                specialization_id=spec_id,
                kind="duplicate-id",
            )
        owners[spec_id] = name
        ids[name] = spec_id
    return ids


def walk_nodes(value) -> Iterator:
    """Every AST node under ``value``, which may be a node or a list of them; parent links are not followed."""
    if isinstance(value, ASTNode):
        yield from value.walk()
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_nodes(item)


def identifier_names(value) -> set[str]:
    """The names that the identifiers in the expression ``value`` read."""
    return {
        node.name
        for node in walk_nodes(value)
        if isinstance(node, IdentifierNode) and isinstance(node.name, str)
    }


def specialization_dependent_constants(
    constants: Iterable, is_specialization_constant: Callable[[object], bool]
) -> set[str]:
    """The names of the program constants whose value follows a specialization constant.

    A specialization constant depends on itself; any other constant depends on one when its initializer reads a
    dependent constant declared before it.
    """
    dependent: set[str] = set()
    for node in constants or []:
        name = getattr(node, "name", None)
        if not isinstance(name, str):
            continue
        if is_specialization_constant(node) or (
            identifier_names(getattr(node, "value", None)) & dependent
        ):
            dependent.add(name)
    return dependent


def _size_names(size) -> set[str]:
    if isinstance(size, str):
        return {size}
    if isinstance(size, int):
        return set()
    return identifier_names(size)


def array_sizes_reading(
    root, names: set[str], allowed: Iterable[ArrayType] = ()
) -> Iterator[tuple[ArrayType, str]]:
    """Yield ``(array type, name)`` for each array type under ``root`` whose size reads one of ``names``.

    The array types in ``allowed`` (by identity) are skipped; their own element types are not.
    Each name is yielded once, at its first array in walk order.
    """
    if not names:
        return
    skipped = {id(node) for node in allowed}
    reported: set[str] = set()
    for node in walk_nodes(root):
        if not isinstance(node, ArrayType) or node.size is None or id(node) in skipped:
            continue
        for name in sorted((_size_names(node.size) & names) - reported):
            reported.add(name)
            yield node, name


def case_labels_reading(root, names: set[str]) -> Iterator[tuple[CaseNode, str]]:
    """Yield ``(case, name)`` for each ``switch`` case label under ``root`` that reads one of ``names``."""
    if not names:
        return
    for node in walk_nodes(root):
        if not isinstance(node, CaseNode):
            continue
        for name in sorted(identifier_names(node.value) & names):
            yield node, name
