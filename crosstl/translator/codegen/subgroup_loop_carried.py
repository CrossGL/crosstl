"""Uniformity facts for locals that a uniform loop carries between iterations.

Software subgroup lowering proves that branch conditions and helper arguments
are uniform across a subgroup. Lexical facts only cover locals that are never
written, and ``subgroup_reassignments`` follows writes outside loops. A loop
may instead advance a local on every trip, as a block-mask offset

    short count = factor;
    for (; iterations > 0; iterations--) {
        if (mask[offset]) { ... }
        count--;
        offset += count == 0 ? step : 0;
        count = count == 0 ? factor : count;
    }

Each trip writes ``count`` and ``offset`` from uniform values, so every
invocation of a subgroup holds the same values on every trip.

``loop_carried_candidates`` selects the locals whose every write is a plain,
compound or increment write of the local itself inside a loop. A loop carries
the candidates it writes. ``carried_fixpoint`` keeps the largest set of them
that is uniform where the loop starts and still uniform after the body and
update of one trip, assuming the set at the start of the trip. The facts hold
on every trip and after the loop only when every invocation runs the same
trips, so the caller must prove the loop's trip count uniform. A loop body
that may leave a trip early (``break``, ``continue`` or ``return``) carries
nothing.
"""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    LoopNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

_LOOPS = (ForNode, WhileNode, DoWhileNode, LoopNode, ForInNode)
_STEPS = frozenset(
    {
        "++",
        "--",
        "PRE_INCREMENT",
        "POST_INCREMENT",
        "PRE_DECREMENT",
        "POST_DECREMENT",
    }
)
_ADDRESS = frozenset({"&", "ADDRESS_OF"})
_COMPOUND = frozenset({"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="})


def _operator(node, map_operator):
    operator = getattr(node, "operator", getattr(node, "op", None))
    mapped = map_operator(operator)
    return mapped if mapped is not None else operator


def carried_write(node, candidates, map_operator):
    """Return ``(kind, name, value)`` for a write of a carried candidate.

    ``kind`` is ``"assign"`` for ``name = value``, ``"compound"`` for
    ``name op= value`` and ``"step"`` for ``name++`` or ``--name`` (value
    ``None``). Any other node gives ``None``.
    """
    if isinstance(node, ExpressionStatementNode):
        node = getattr(node, "expression", None)
    if isinstance(node, AssignmentNode):
        target = getattr(node, "target", getattr(node, "left", None))
        if not isinstance(target, IdentifierNode) or target.name not in candidates:
            return None
        operator = _operator(node, map_operator)
        value = getattr(node, "value", getattr(node, "right", None))
        if operator == "=":
            return "assign", target.name, value
        if operator in _COMPOUND:
            return "compound", target.name, value
        return None
    if isinstance(node, UnaryOpNode):
        operand = getattr(node, "operand", None)
        if (
            isinstance(operand, IdentifierNode)
            and operand.name in candidates
            and _operator(node, map_operator) in _STEPS
        ):
            return "step", operand.name, None
    return None


def loop_carried_candidates(
    function,
    mutable_names,
    call_mutations,
    *,
    walk,
    map_operator,
    identifier_names,
    value_type,
    volatile,
    write_roots=None,
    excluded=frozenset(),
):
    """Return the locals whose writes a loop-carried flow can follow.

    Each is a plain value declared once with an initializer outside the
    parameters, and every write to it is ``name = value``, ``name op= value``
    or an increment or decrement of ``name`` itself, inside some loop's body.
    Element and member writes, writes outside loops, address-taking, pointer
    or reference views and calls that may write it disqualify a name.
    ``excluded`` names (followed by another analysis) never qualify.
    ``write_roots`` gives the names an assignment target writes; by default
    the identifier at the root of its element and member accesses.
    """
    write_roots = write_roots or _write_roots
    body = getattr(function, "body", None)
    parameters = {
        parameter.name
        for parameter in getattr(function, "parameters", []) or []
        if getattr(parameter, "name", None)
    }
    counts = {}
    declarations = {}
    for node in walk(body):
        if isinstance(node, VariableNode) and getattr(node, "name", None):
            counts[node.name] = counts.get(node.name, 0) + 1
            declarations[node.name] = node
    candidates = {
        name
        for name, declaration in declarations.items()
        if counts[name] == 1
        and name not in parameters
        and name not in excluded
        and name in mutable_names
        and getattr(declaration, "initial_value", None) is not None
        and value_type(declaration.var_type)
        and not volatile(declaration)
        and not volatile(declaration.var_type)
    }
    if not candidates:
        return set()
    in_loop = set()
    for loop in walk(body):
        if isinstance(loop, _LOOPS):
            for node in walk(getattr(loop, "body", None)):
                in_loop.add(id(node))
            update = getattr(loop, "update", None)
            for node in walk(update):
                in_loop.add(id(node))
    written = set()
    for node in walk(body):
        write = carried_write(node, candidates, map_operator)
        if write is not None:
            written.add(write[1])
            if id(node) not in in_loop:
                candidates.discard(write[1])
            continue
        if isinstance(node, AssignmentNode):
            target = getattr(node, "target", getattr(node, "left", None))
            candidates.difference_update(write_roots(target))
        elif isinstance(node, UnaryOpNode):
            operator = _operator(node, map_operator)
            if operator in _STEPS or operator in _ADDRESS:
                candidates.difference_update(identifier_names(node.operand))
        elif isinstance(node, FunctionCallNode):
            candidates.difference_update(call_mutations.get(id(node), set()))
        elif isinstance(node, VariableNode) and isinstance(
            getattr(node, "var_type", None), (PointerType, ReferenceType)
        ):
            candidates.difference_update(identifier_names(node.initial_value))
    return candidates & written


def _write_roots(target):
    while True:
        if isinstance(target, ArrayAccessNode):
            target = target.array_expr
        elif isinstance(target, MemberAccessNode):
            target = target.object_expr
        elif isinstance(target, UnaryOpNode):
            target = target.operand
        else:
            break
    return {target.name} if isinstance(target, IdentifierNode) else set()


def loop_carried_names(loop, candidates, walk, map_operator):
    """The candidates ``loop``'s body or update writes, at any depth."""
    names = set()
    for root in (getattr(loop, "body", None), getattr(loop, "update", None)):
        for node in walk(root):
            write = carried_write(node, candidates, map_operator)
            if write is not None:
                names.add(write[1])
    return frozenset(names)


def leaves_trip_early(loop, walk):
    """Whether ``loop``'s body may end a trip early."""
    return any(
        isinstance(node, (BreakNode, ContinueNode, ReturnNode))
        for node in walk(getattr(loop, "body", None))
    )


def follow_carried_write(node, candidates, names, uniform_value, map_operator):
    """Update ``names`` for a carried write; ``None`` when ``node`` is not one.

    A plain write is uniform when its value is. A compound write or a step
    keeps a uniform local uniform when its value is uniform.
    """
    write = carried_write(node, candidates, map_operator)
    if write is None:
        return None
    kind, name, value = write
    if kind == "assign":
        uniform = uniform_value(value)
    elif kind == "compound":
        uniform = name in names and uniform_value(value)
    else:
        uniform = name in names
    if uniform:
        names.add(name)
    else:
        names.discard(name)
    return uniform


def carried_fixpoint(carried, names, follow_trip):
    """The carried names uniform on every trip of a uniform-trip loop.

    ``names`` holds the facts where the loop starts. ``follow_trip(trial)``
    updates ``trial`` in place across the loop's body and update for one
    trip. The result shrinks from the carried names uniform at the start
    until one trip preserves all of them.
    """
    kept = set(carried) & set(names)
    while kept:
        trial = (set(names) - set(carried)) | kept
        follow_trip(trial)
        preserved = kept & trial
        if preserved == kept:
            break
        kept = preserved
    return frozenset(kept)
