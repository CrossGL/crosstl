"""Metal function-scope samplers declared without an initializer.

``sampler s;`` default-constructs a sampler with the default state (normalized coordinates, clamp to edge, nearest
filtering, no mipmapping, no comparison; Metal Shading Language Specification, section 2.10, table 2.7), exactly as
``constexpr sampler s;`` does. The importer emitted it as an uninitialized ``sampler`` local, which no other target
can bind: the SPIR-V generator failed with an untyped ``ValueError`` and the HLSL generator emitted an uninitialized
``SamplerState`` local, which DXC rejects. CrossGL spells fixed default state ``const sampler s = sampler();``
(:mod:`crosstl.translator.semantics.sampler_states`), which every target then carries or refuses.

Metal also lets such a local be assigned later (``s = boundSampler;``), and then it is not fixed state, so
:func:`is_default_sampler_local` accepts only a scalar declaration whose name is never assigned or has its address
taken anywhere in the function.
"""

from __future__ import annotations

__all__ = ["is_default_sampler_local"]

_SAMPLER_TYPES = frozenset({"sampler", "metal::sampler"})


def _children(node):
    if isinstance(node, (list, tuple)):
        return list(node)
    if isinstance(node, dict):
        return list(node.values())
    if hasattr(node, "__dict__"):
        return list(vars(node).values())
    return []


def _name(node):
    if isinstance(node, str):
        return node
    if getattr(node, "vtype", None) in (None, ""):
        name = getattr(node, "name", None)
        return name if isinstance(name, str) else None
    return None


def _modifies(body, name) -> bool:
    """Whether ``name`` is assigned, incremented or has its address taken anywhere in ``body``."""
    seen = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if (
            node is None
            or isinstance(node, (str, int, float, bool))
            or id(node) in seen
        ):
            continue
        seen.add(id(node))
        kind = type(node).__name__
        if kind == "AssignmentNode" and _name(getattr(node, "left", None)) == name:
            return True
        if (
            kind == "UnaryOpNode"
            and str(getattr(node, "op", "")) in {"&", "++", "--"}
            and _name(getattr(node, "operand", None)) == name
        ):
            return True
        stack.extend(_children(node))
    return False


def is_default_sampler_local(declaration, function_body) -> bool:
    """Whether a function-scope ``sampler`` declaration is a default-state sampler the body never reassigns."""
    if getattr(declaration, "value", None) is not None:
        return False
    if getattr(declaration, "array_sizes", None):
        return False
    if str(getattr(declaration, "vtype", "") or "").strip() not in _SAMPLER_TYPES:
        return False
    qualifiers = {
        str(q).lower() for q in getattr(declaration, "qualifiers", None) or []
    }
    if qualifiers & {"extern", "static", "threadgroup"}:
        return False
    name = getattr(declaration, "name", None)
    if not isinstance(name, str) or function_body is None:
        return False
    return not _modifies(function_body, name)
