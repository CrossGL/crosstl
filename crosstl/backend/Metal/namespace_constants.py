"""Qualified names of namespace-scope integral constants in Metal static assertions.

A ``static_assert`` at file scope may read a constant declared in a namespace through its qualified name, as
PyTorch's MPS headers do with ``INNER_TG_SIZE % ::c10::metal::simdgroup_size == 0`` after
``namespace c10 { namespace metal { constant constexpr unsigned simdgroup_size = 32; } }``. The assertion
evaluator resolved only names visible unqualified at the assertion, so the condition stayed unresolved and the
kernel was refused with ``MetalStaticAssertionError``.

A namespace-scope binding's scope is the namespace body, so the bindings the evaluator already collects identify
these constants: :class:`NamespaceConstantValues` keys each one by its namespace's qualified name, and
:meth:`NamespaceConstantValues.substitute` replaces ``ns::name`` and ``::ns::name`` with the value of a constant
declared before the assertion ([basic.lookup.qual]). Every other name is left for the evaluator, which still
refuses what it cannot resolve.
"""

from __future__ import annotations

import bisect
import re
from typing import Iterable

__all__ = ["NamespaceConstantValues"]

_QUALIFIED_NAME_RE = re.compile(
    r"(?<![\w:.])(?:::\s*)?(?P<name>[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)+)\b(?!\s*[(<])"
)


class NamespaceConstantValues:
    """The values of namespace-scope integral constants, by qualified name and declaration position."""

    def __init__(self, namespace_spans: Iterable[tuple[int, int, str]], bindings):
        namespaces = {
            (start, end): name for start, end, name in namespace_spans if name
        }
        entries: list[tuple[int, str, str]] = []
        for name, candidates in bindings.items():
            for binding in candidates:
                namespace = namespaces.get((binding.scope_start, binding.scope_end))
                if namespace is None or binding.type_alias or binding.value is None:
                    continue
                entries.append(
                    (
                        binding.declaration_position,
                        f"{namespace}::{name}",
                        binding.value,
                    )
                )
        entries.sort()
        self._positions = [entry[0] for entry in entries]
        self._entries = entries

    def visible_at(self, position: int) -> dict[str, str]:
        """The constants declared before ``position``, by qualified name."""
        end = bisect.bisect_right(self._positions, position)
        return {name: value for _position, name, value in self._entries[:end]}

    def substitute(self, condition: str, position: int) -> str:
        """``condition`` with each qualified reference to a constant declared before ``position`` replaced."""
        if not self._entries or "::" not in condition:
            return condition
        values = self.visible_at(position)

        def replace(match: re.Match) -> str:
            name = re.sub(r"\s+", "", match.group("name"))
            value = values.get(name)
            return match.group(0) if value is None else f"({value})"

        return _QUALIFIED_NAME_RE.sub(replace, condition)
