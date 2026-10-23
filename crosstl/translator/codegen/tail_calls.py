"""Self tail calls rewritten as loops, for targets that forbid recursion.

A function whose every call to itself is a tail call computes the same result as an unconditional loop (``LoopNode``)
that rebinds its parameters and starts the body again. :func:`lower_self_tail_calls` performs that rewrite on the
functions of a :class:`~.call_graph.CallGraph` before :func:`~.call_graph.validate_recursion_free` refuses the cycles
that remain:

.. code-block:: text

    void step(inout S s, int n) {          void step(inout S s, int n_entry) {
        ...                                    int n = n_entry;
        if (c) { step(s, n - 1); }     ->      loop {
    }                                              ...
                                                   if (c) { n = n - 1; continue; }
                                                   return;
                                               }
                                           }

A function with a result keeps it in a local: each ``return e;`` becomes ``result = e; break;`` and the loop is
followed by ``return result;``, so the function visibly ends in a ``return`` for validators that do not analyse loop
exits (Naga). Such a function is lowered only when no loop or ``switch`` in its body holds a ``return``, since a
``break`` there would leave the inner statement instead.

A self call is a tail call when it is a statement that ends the body of a ``void`` function or is followed by
``return;``, or when it is the value of a ``return``, provided no loop or ``switch`` encloses it (``continue`` must
reach the new loop). Every self call of the function must be a tail call and resolve to the function alone.

A parameter that a tail call passes unchanged keeps its binding, whatever its type. A parameter that receives
another value must be a scalar, vector, matrix or structure passed by value; the parameter is renamed and copied
into a mutable local of the original name, so targets whose parameters are immutable (WGSL) can rebind it. The
function is left unchanged, and still refused as recursive, when a parameter that would be rebound is an array,
pointer, reference, resource or ``out``/``inout`` parameter, or when a declaration in the body shadows it.
"""

from __future__ import annotations

from .. import ast as A
from ..semantics.hub_types import type_from_node
from ..semantics.rewrite import child_slots, get_child

__all__ = ["lower_self_tail_calls"]

# Parameter qualifiers that keep a by-value binding; any other qualifier marks a binding the loop cannot rebind.
_VALUE_QUALIFIERS = frozenset({"in", "const"})


def lower_self_tail_calls(graph) -> bool:
    """Rewrite each function of ``graph`` whose self calls are all lowerable tail calls; ``True`` if any changed."""
    structures = _structure_names(graph)
    changed = False
    for index, function in enumerate(graph.functions):
        calls = graph.self_calls(index)
        if not calls or getattr(function, "body", None) is None:
            continue
        rewrite = _TailRewrite(function, calls, structures)
        if rewrite.plan():
            rewrite.apply()
            changed = True
    return changed


def _structure_names(graph):
    names = set()
    program = graph.program
    for struct in getattr(program, "structs", None) or []:
        names.add(getattr(struct, "name", None))
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        for struct in getattr(stage, "local_structs", None) or []:
            names.add(getattr(struct, "name", None))
    names.discard(None)
    return names


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _statements(body):
    if body is None:
        return []
    if isinstance(body, A.BlockNode):
        return body.statements
    if isinstance(body, list):
        return body
    return None


def _statement_call(statement):
    if isinstance(statement, A.FunctionCallNode):
        return statement
    if isinstance(statement, A.ExpressionStatementNode) and isinstance(
        statement.expression, A.FunctionCallNode
    ):
        return statement.expression
    return None


def _is_void(function):
    returned = getattr(function, "return_type", None)
    if returned is None:
        return True
    name = returned if isinstance(returned, str) else getattr(returned, "name", None)
    return isinstance(returned, (A.PrimitiveType, A.NamedType, str)) and name == "void"


def _same_parameter(argument, name):
    return isinstance(argument, A.IdentifierNode) and argument.name == name


class _TailRewrite:
    """The tail-call sites of one function, the parameters they rebind, and the rewrite that turns them into a loop."""

    def __init__(self, function, calls, structures):
        self.function = function
        self.calls = {id(call) for call in calls}
        self.structures = structures
        self.void = _is_void(function)
        self.parameters = list(function.parameters or [])
        # (sequence, index, call, drops_following_return) for a tail call in a statement list, and
        # (None, statement, call, False) for a ``return`` of a tail call or a branch that is a lone tail call.
        self.sites: list = []
        self.rebound: list[int] = []

    # Planning --------------------------------------------------------------------------------------------------
    def plan(self) -> bool:
        statements = _statements(self.function.body)
        if statements is None or not self._sequence(statements, tail=True):
            return False
        found = {id(site[2]) for site in self.sites}
        if found != self.calls:
            return False
        rebound = set()
        for _, _, call, _ in self.sites:
            arguments = list(call.arguments or [])
            if len(arguments) != len(self.parameters):
                return False
            for position, (parameter, argument) in enumerate(
                zip(self.parameters, arguments)
            ):
                if not getattr(parameter, "name", None):
                    return False
                if not _same_parameter(argument, parameter.name):
                    rebound.add(position)
        self.rebound = sorted(rebound)
        if not all(self._rebindable(self.parameters[i]) for i in self.rebound):
            return False
        if not self.void and _nested_return(self.function.body, False):
            return False
        names = {self.parameters[i].name for i in self.rebound}
        return not any(
            isinstance(node, (A.VariableNode, A.ParameterNode)) and node.name in names
            for node in _walk(self.function.body)
        )

    def _sequence(self, statements, tail):
        last = len(statements) - 1
        for position, statement in enumerate(statements):
            call = _statement_call(statement)
            if call is not None and id(call) in self.calls:
                following = statements[position + 1] if position < last else None
                returns = (
                    isinstance(following, A.ReturnNode) and following.value is None
                )
                if not self.void or not (returns or (tail and position == last)):
                    return False
                if any(self._contains(argument) for argument in call.arguments or []):
                    return False
                self.sites.append((statements, position, call, returns))
                continue
            if not self._statement(statement, tail and position == last):
                return False
        return True

    def _statement(self, statement, tail):
        if isinstance(statement, A.BlockNode):
            return self._sequence(statement.statements, tail)
        if isinstance(statement, A.IfNode):
            if self._contains(statement.condition):
                return False
            for branch in (statement.then_branch, statement.else_branch):
                if branch is None:
                    continue
                statements = _statements(branch)
                call = _statement_call(branch)
                if statements is not None:
                    if not self._sequence(statements, tail):
                        return False
                elif call is not None and id(call) in self.calls:
                    if not (self.void and tail) or any(
                        self._contains(argument) for argument in call.arguments or []
                    ):
                        return False
                    self.sites.append((None, branch, call, False))
                elif not self._statement(branch, tail):
                    return False
            return True
        if isinstance(statement, A.ReturnNode):
            value = statement.value
            if isinstance(value, A.FunctionCallNode) and id(value) in self.calls:
                if self.void or any(
                    self._contains(argument) for argument in value.arguments or []
                ):
                    return False
                self.sites.append((None, statement, value, False))
                return True
        return not self._contains(statement)

    def _contains(self, node):
        if node is None:
            return False
        return any(
            isinstance(child, A.FunctionCallNode) and id(child) in self.calls
            for child in _walk(node)
        )

    def _rebindable(self, parameter):
        qualifiers = {
            str(q).lower() for q in getattr(parameter, "qualifiers", []) or []
        }
        if qualifiers - _VALUE_QUALIFIERS or getattr(parameter, "default_value", None):
            return False
        declared = parameter.param_type
        if isinstance(declared, (A.ReferenceType, A.PointerType, A.ArrayType)):
            return False
        if isinstance(declared, A.MatrixType) or type_from_node(declared) is not None:
            return True
        return (
            isinstance(declared, A.NamedType)
            and not declared.generic_args
            and declared.name in self.structures
        )

    # Rewriting -------------------------------------------------------------------------------------------------
    def apply(self):
        function = self.function
        used = {
            node.name
            for node in _walk(function.body)
            if isinstance(node, (A.IdentifierNode, A.VariableNode))
        }
        used.update(parameter.name for parameter in self.parameters)
        originals = {
            position: self.parameters[position].name for position in self.rebound
        }
        temporaries = {
            position: _fresh(f"{name}_tail", used)
            for position, name in originals.items()
        }
        entries = []
        for position, original in originals.items():
            parameter = self.parameters[position]
            parameter.name = _fresh(f"{original}_entry", used)
            entries.append(
                A.VariableNode(
                    original,
                    parameter.param_type,
                    A.IdentifierNode(parameter.name),
                    source_location=getattr(parameter, "source_location", None),
                )
            )
        # Replace from the last site backwards so earlier indices stay valid.
        for statements, where, call, drops_return in reversed(self.sites):
            block = self._rebinding(call, originals, temporaries)
            if statements is None:
                _replace_statement(function.body, where, block)
            else:
                if drops_return:
                    del statements[where + 1]
                statements[where] = block
        location = getattr(function, "source_location", None)
        body = list(_statements(function.body))
        if self.void:
            if not (body and isinstance(body[-1], A.ReturnNode)):
                body.append(A.ReturnNode())
            loop = A.LoopNode(A.BlockNode(body), source_location=location)
            function.body = A.BlockNode([*entries, loop])
            return
        result = _fresh("tail_result", used)
        loop = A.LoopNode(A.BlockNode(body), source_location=location)
        returns = [node for node in _walk(loop) if isinstance(node, A.ReturnNode)]
        for statement in returns:
            exit_block = A.BlockNode(
                [
                    _assign(result, statement.value, statement.source_location),
                    A.BreakNode(source_location=statement.source_location),
                ],
                source_location=statement.source_location,
            )
            _replace_statement(loop, statement, exit_block)
        function.body = A.BlockNode(
            [
                *entries,
                A.VariableNode(result, function.return_type, source_location=location),
                loop,
                A.ReturnNode(A.IdentifierNode(result), source_location=location),
            ]
        )

    def _rebinding(self, call, originals, temporaries):
        location = getattr(call, "source_location", None)
        arguments = list(call.arguments or [])
        statements = []
        if len(self.rebound) == 1:
            position = self.rebound[0]
            statements.append(
                _assign(originals[position], arguments[position], location)
            )
        else:
            for position in self.rebound:
                statements.append(
                    A.VariableNode(
                        temporaries[position],
                        self.parameters[position].param_type,
                        arguments[position],
                        source_location=location,
                    )
                )
            for position in self.rebound:
                statements.append(
                    _assign(
                        originals[position],
                        A.IdentifierNode(temporaries[position]),
                        location,
                    )
                )
        statements.append(A.ContinueNode(source_location=location))
        return A.BlockNode(statements, source_location=location)


_NESTING = (A.ForNode, A.WhileNode, A.DoWhileNode, A.LoopNode, A.SwitchNode)


def _nested_return(node, nested):
    """Whether a ``return`` under ``node`` lies inside a loop or ``switch`` (``nested``: ``node`` already does)."""
    if isinstance(node, A.ReturnNode) and nested:
        return True
    nested = nested or isinstance(node, _NESTING)
    return any(
        _nested_return(get_child(node, field, key), nested)
        for field, key in list(child_slots(node))
    )


def _assign(name, value, location):
    return A.ExpressionStatementNode(
        A.AssignmentNode(A.IdentifierNode(name), value, source_location=location),
        source_location=location,
    )


def _fresh(base, used):
    name, suffix = base, 1
    while name in used:
        suffix += 1
        name = f"{base}_{suffix}"
    used.add(name)
    return name


def _replace_statement(node, target, replacement):
    """Replace the statement ``target`` under ``node`` by ``replacement``; branches that are not blocks included."""
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        if child is target:
            if key is None:
                for name, value in list(vars(node).items()):
                    if value is target and name not in ("parent", "annotations"):
                        setattr(node, name, replacement)
            else:
                getattr(node, field)[key] = replacement
            return True
        if _replace_statement(child, target, replacement):
            return True
    return False
