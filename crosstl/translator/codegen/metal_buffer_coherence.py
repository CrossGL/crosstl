"""Coherence and ``volatile`` of Metal helper buffer parameters.

MSL treats ``coherent(scope)`` and ``volatile`` as part of a pointer or reference type. A conversion may add
them but never drop them: ``device uint*`` converts to ``coherent(device) device uint*``, the reverse is "would
lose coherent(device) qualifier", and ``coherent(threadgroup)`` widens to ``coherent(device)`` and on to
``coherent(system)`` but not back. Plain ``coherent`` is ``coherent(device)``.

So a helper parameter that receives a coherent or volatile buffer has to state the widest coherence and every
``volatile`` among the arguments its callers pass, or the call does not compile. Widening a parameter is safe:
callers that pass plain buffers convert implicitly, and wider coherence only narrows the set of values a shader
may observe.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)

# Rank of each MSL coherence scope; plain ``coherent`` (scope None) is device coherence.
COHERENCE_RANK = {"threadgroup": 1, None: 2, "device": 2, "system": 3}


def merge_memory_contracts(own, inherited):
    """Return ``own`` widened by ``inherited`` ``(kind, scope)`` contracts.

    The result keeps ``own`` when ``inherited`` adds nothing. Otherwise it holds ``volatile`` when either list
    does, followed by the widest coherence of both (the first one listed on a tie). An unknown scope is kept as
    listed so that the declaration's own diagnostic reports it.
    """
    own = list(own)
    combined = own + [contract for contract in inherited if contract not in own]
    if len(combined) == len(own):
        return own
    if any(
        kind == "coherent" and scope not in COHERENCE_RANK for kind, scope in combined
    ):
        return combined
    merged = []
    if any(kind == "volatile" for kind, _scope in combined):
        merged.append(("volatile", None))
    widest = None
    for kind, scope in combined:
        if kind != "coherent":
            continue
        if widest is None or COHERENCE_RANK[scope] > COHERENCE_RANK[widest[1]]:
            widest = (kind, scope)
    if widest is not None:
        merged.append(widest)
    return merged


def memory_helper_token(memory):
    """Name part that keeps helpers for coherent or volatile buffers apart: ``"coherent "`` -> ``"coherent_"``."""
    words = [
        "".join(character if character.isalnum() else "_" for character in word)
        for word in str(memory or "").split()
    ]
    return "".join(f"{word}_" for word in words)


def coherent_array_parameter_declaration(type_name, name):
    """Declare an array parameter as the pointer it decays to.

    MSL rejects ``coherent`` on a parameter written with array syntax ("parameter may not be qualified with
    coherent") but accepts it on the equivalent pointer: ``uint[]`` and ``uint[4]`` become ``uint *name`` and
    ``uint[][4]`` becomes ``uint (*name)[4]``.
    """
    type_name = str(type_name)
    start = type_name.find("[")
    if start < 0:
        return f"{type_name} {name}"
    depth = 0
    end = len(type_name)
    for index in range(start, len(type_name)):
        if type_name[index] == "[":
            depth += 1
        elif type_name[index] == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    base = type_name[:start].rstrip()
    rest = type_name[end:]
    return f"{base} (*{name}){rest}" if rest else f"{base} *{name}"


def buffer_argument_root_name(expression):
    """Name the variable whose storage ``expression`` addresses, or None.

    Element, member, address-of, cast and pointer-offset expressions all keep the storage of their root.
    """
    while True:
        if isinstance(expression, ArrayAccessNode):
            expression = expression.array_expr
        elif isinstance(expression, MemberAccessNode):
            expression = expression.object_expr
        elif isinstance(expression, CastNode):
            expression = expression.expression
        elif isinstance(expression, UnaryOpNode) and expression.operator in {"&", "*"}:
            expression = expression.operand
        elif isinstance(expression, BinaryOpNode) and expression.operator in {
            "+",
            "-",
        }:
            expression = expression.left
        else:
            break
    if isinstance(expression, IdentifierNode):
        return expression.name
    return None


def _local_declarations(body, iter_nodes):
    declarations = {}
    for node in iter_nodes(body):
        if isinstance(node, VariableNode) and getattr(node, "name", None):
            declarations.setdefault(node.name, node)
    return declarations


def collect_inherited_buffer_contracts(
    global_variables, functions, *, contracts_of, call_name, iter_nodes
):
    """Return ``{(function name, parameter name): contracts}`` that helper parameters inherit from callers.

    ``contracts_of(node)`` lists a global, parameter or local declaration's own ``(kind, scope)`` memory
    contracts, ``call_name(call)`` names a call's callee and ``iter_nodes(body)`` walks a function body.
    Inheritance follows calls through any chain of helpers until nothing changes. Overloads that share a name
    share their parameter keys, which only widens them.
    """
    global_contracts = {}
    for node in global_variables:
        name = getattr(node, "name", None)
        if name and name not in global_contracts:
            global_contracts[name] = list(contracts_of(node))

    overloads = {}
    scopes = []
    for function in functions:
        function_name = getattr(function, "name", None)
        body = getattr(function, "body", None)
        if not function_name:
            continue
        overloads.setdefault(function_name, []).append(function)
        if body is None:
            continue
        parameters = {
            parameter.name: parameter
            for parameter in getattr(function, "parameters", None) or []
            if getattr(parameter, "name", None)
        }
        calls = [
            node for node in iter_nodes(body) if isinstance(node, FunctionCallNode)
        ]
        if calls:
            locals_ = _local_declarations(body, iter_nodes)
            scopes.append((function_name, parameters, locals_, calls))

    inherited = {}

    def effective(function_name, parameter):
        key = (function_name, parameter.name)
        return merge_memory_contracts(contracts_of(parameter), inherited.get(key, []))

    changed = True
    while changed:
        changed = False
        for function_name, parameters, locals_, calls in scopes:
            for call in calls:
                callees = overloads.get(call_name(call), [])
                arguments = getattr(call, "arguments", None) or []
                for callee in callees:
                    callee_parameters = getattr(callee, "parameters", None) or []
                    for argument, parameter in zip(arguments, callee_parameters):
                        source = buffer_argument_root_name(argument)
                        if source is None:
                            continue
                        if source in parameters:
                            contracts = effective(function_name, parameters[source])
                        elif source in locals_:
                            contracts = list(contracts_of(locals_[source]))
                        else:
                            contracts = global_contracts.get(source, [])
                        if not contracts:
                            continue
                        key = (callee.name, parameter.name)
                        known = inherited.setdefault(key, [])
                        for contract in contracts:
                            if contract not in known:
                                known.append(contract)
                                changed = True
    return {key: contracts for key, contracts in inherited.items() if contracts}
