"""Which constrained free-function templates the constrained call pass may remove.

The pass binds each call in a reachable concrete body to the ``enable_if``-constrained template that overload
resolution selects, and then removes the templates it consulted, so that no later stage can instantiate one of them
without its constraint. Calls the pass does not bind can still need those templates:

* calls in template bodies, which name the template once their own template is instantiated later, for example
  ``wrap<T>(a, b) { return max(a, b); }`` beside a kernel that also calls ``max`` directly;
* calls in concrete functions no entry point reaches yet, which a later instantiation may make reachable.

A template such a call can see stays declared, as it would if no reachable call had consulted it, so a later pass
binds that call to it. Removing it would let the call silently bind to a different function, typically the Metal
standard library function of the same name, whose semantics differ.

A call the pass defers from the instantiated body of a carrier template (``constrained_call_carriers``) needs the
templates lookup from that body can see in the same way, although the carrier's own declaration goes.

A template consulted by a call whose every constrained candidate was disabled is still removed: that call is left
for a native overload, which a residual unconstrained declaration must not capture.

The name and parameter list of a function definition, such as ``template <> half twice(half v) { ... }`` or a
concrete overload ``float twice(float v) { ... }``, read like a call but name no function, so they are no unbound
reference (``defines_function``).
"""

import re
from typing import Any, Callable, Iterable, List, NamedTuple, Sequence, Set, Tuple

__all__ = [
    "DeferredConstrainedCall",
    "defines_function",
    "removable_constrained_templates",
]

Span = Tuple[int, int]

# What may stand between the parameter list of a function definition and its body.
_DECLARATOR_SUFFIX_RE = re.compile(r"(?:\s+|\bconst\b|\bnoexcept\b|\[\[[^\[\]]*\]\])*")


def defines_function(masked_code: str, reference_end: int) -> bool:
    """Whether the call-shaped text ending at ``reference_end`` is the declarator of a function definition.

    ``masked_code`` is the source with comments and literals masked. In an expression, a parenthesized argument
    list is never followed by ``{``; after a declarator's parameter list only cv and exception qualifiers and
    attributes can precede the body.
    """
    suffix = _DECLARATOR_SUFFIX_RE.match(masked_code, reference_end)
    position = suffix.end() if suffix else reference_end
    return masked_code.startswith("{", position)


class DeferredConstrainedCall(NamedTuple):
    """A call the pass left for a later pass, looked up from ``span[0]``."""

    name: str
    qualified_name: str
    position: int

    @property
    def span(self) -> Span:
        return (self.position, self.position)


def removable_constrained_templates(
    handled_spans: Iterable[Span],
    templates: Sequence[Any],
    unbound_references: Sequence[Any],
    native_fallback_names: Set[str],
    sees: Callable[[Any, Any], bool],
    containing_span: Callable[[int, List[Span]], Any],
    deferred_calls: Sequence[DeferredConstrainedCall] = (),
) -> List[Span]:
    """The spans of the handled templates that no unbound reference still needs, in source order.

    ``sees(template, reference)`` tells whether lookup from the reference can find the template. A reference inside a
    template that is itself removed needs nothing, unlike a ``deferred_calls`` entry.
    """
    by_span = {template.span: template for template in templates}
    removed = set(handled_spans)
    changed = True
    while changed:
        changed = False
        removed_spans = sorted(removed)
        for span in removed_spans:
            template = by_span[span]
            if template.name in native_fallback_names:
                continue
            if any(
                reference.name == template.name
                and containing_span(reference.span[0], removed_spans) is None
                and sees(template, reference)
                for reference in unbound_references
            ) or any(
                call.name == template.name and sees(template, call)
                for call in deferred_calls
            ):
                removed.discard(span)
                changed = True
                break
    return sorted(removed)
