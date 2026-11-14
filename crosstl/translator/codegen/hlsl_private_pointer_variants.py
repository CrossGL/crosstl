"""Which backing variants of a DirectX private pointer helper to emit.

HLSL has no pointers into private arrays, so a helper's ``thread T*``
parameter becomes a fixed array, and a helper called with differently
shaped backings is emitted once per backing as an overload. Each call
binds a backing: a local array of some extent, a word or byte view of a
local record, or the caller's own parameter. The helper's default backing,
the span it requires itself, is a candidate too.

A variant is reached when some call binds it, following calls from the
variants their callers reach in turn. Functions no recorded call reaches
keep their own variants, and a helper with a call the analysis did not
record, such as one in a branch it proved dead, keeps its default for it.

Two variants whose HLSL arrays have the same element types and extents are
one overload to HLSL, because ``in`` and ``inout`` do not distinguish
overloads: emitting both makes every call ambiguous. An unreached candidate
is therefore emitted only when no emitted variant has its signature, and
two reached variants with one signature are reported, so the caller can
refuse them instead of emitting ambiguous HLSL.
"""


def reached_variants(variants, view_calls, pointer_parameters, callee_variant):
    """The variants of every helper that some call chain reaches.

    ``variants`` maps each helper to all its candidate variants, defaults
    included, and ``view_calls`` maps each caller to its recorded calls.
    ``callee_variant(caller_variant, call)`` gives the variant a call binds
    from one variant of its caller, or ``None`` when it binds none. A helper
    that calls reach in no variant keeps all its candidates.
    """

    called = {call["callee"] for calls in view_calls.values() for call in calls}
    reached = {name: set() for name in variants}
    pending = []
    for name, function_variants in variants.items():
        if name not in called:
            reached[name].update(function_variants)
            pending.extend((name, variant) for variant in function_variants)
    pending.extend((caller, ()) for caller in view_calls if caller not in variants)
    visited = set()
    while pending:
        caller, caller_variant = pending.pop()
        if (caller, caller_variant) in visited:
            continue
        visited.add((caller, caller_variant))
        for call in view_calls.get(caller, ()):
            variant = callee_variant(caller_variant, call)
            callee = call["callee"]
            if variant is None or callee not in reached:
                continue
            if variant not in reached[callee]:
                reached[callee].add(variant)
                pending.append((callee, variant))
    return {
        name: reached[name] if reached[name] else set(function_variants)
        for name, function_variants in variants.items()
    }


def emitted_variants(candidates, reached, signature):
    """``(variants, collisions)`` of one helper.

    ``variants`` holds every reached variant and each unreached candidate
    whose HLSL signature, ``signature(variant)``, no other emitted variant
    has. ``collisions`` lists ``(first, second)`` pairs of reached variants
    that share one signature. Both are independent of set order.
    """

    emitted = {}
    collisions = []
    for variant in sorted(reached, key=repr):
        first = emitted.setdefault(signature(variant), variant)
        if first != variant:
            collisions.append((first, variant))
    for variant in sorted(set(candidates) - set(reached), key=repr):
        emitted.setdefault(signature(variant), variant)
    return set(emitted.values()) | set(reached), collisions
