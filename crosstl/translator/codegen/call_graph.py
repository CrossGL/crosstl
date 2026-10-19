"""Static call graph of a CrossGL program, and the recursion refusal for targets that forbid recursion.

Vulkan SPIR-V (VUID-StandaloneSpirv-None-04634) and WGSL (section 5) forbid static recursion. Metal allows recursive
calls only in kernel functions from Metal 2.4, and only to the call-stack depth the host sets on the pipeline
(``maxCallStackDepth``, default 1); a translated artifact cannot carry that setting, and a deeper recursion hangs the
GPU (conformance case ``u_recursion``). These targets therefore refuse any cycle in the static call graph with a
:class:`RecursionUnsupportedError` before code generation, instead of emitting the cycle unchanged.

The graph has one node per function declaration: program functions, stage-local functions and stage entry points,
in declaration order. A call by name has an edge to every declaration of that name whose arity accepts the call;
among several, overload resolution keeps those whose parameter types equal the argument types the hub type scope
infers, then those with the same lane counts. Among those, arguments without a value type (matrices, cooperative
matrices, arrays, structures) keep the overloads whose parameters declare the argument's declared type, when any
do. Overloads that cannot be told apart all receive an edge.
Calls through member access, function values and built-ins have no user-declared callee and no edge.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from ..semantics.hub_types import TypeScope, call_name, type_from_node
from ..semantics.rewrite import child_slots, get_child

__all__ = [
    "RECURSION_FREE_TARGETS",
    "CallGraph",
    "RecursionUnsupportedError",
    "validate_recursion_free",
]

RECURSION_FREE_TARGETS = {
    "vulkan": (
        "Vulkan forbids static recursion: the static function-call graph for an entry point must not contain "
        "cycles (VUID-StandaloneSpirv-None-04634)"
    ),
    "wgsl": (
        "WGSL forbids recursion: functions must not be recursive, either directly or indirectly (WGSL section 5)"
    ),
    "metal": (
        "Metal allows recursive calls only in kernel functions (MSL 2.4 and later) and only to the call-stack depth "
        "the host sets on the pipeline (maxCallStackDepth, default 1); the translated artifact cannot carry that "
        "depth, and exceeding it hangs the GPU"
    ),
}


class RecursionUnsupportedError(TranslationDiagnostic):
    """The program's static call graph has a cycle, and the target cannot execute recursion."""

    def __init__(self, target, cycle, *, source_location=None):
        self.cycle = tuple(cycle)
        self.reason_code = "recursive-call-cycle"
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.function-recursion",)
        cycle_text = " -> ".join(self.cycle)
        super().__init__(
            f"recursive call cycle '{cycle_text}': {RECURSION_FREE_TARGETS[target]}",
            span=_span(source_location),
            target=target,
            code=f"{CODE_PREFIX}{target}-recursion-unsupported",
        )


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        value = location.get
    else:

        def value(key, default=None):
            return getattr(location, key, default)

    line = value("line")
    if not isinstance(line, int):
        return None
    column = value("column")
    return SourceSpan(
        value("file"), line, column if isinstance(column, int) and column > 0 else 1
    )


def _stage_functions(program):
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        yield stage, list(getattr(stage, "local_functions", None) or [])
        entry = getattr(stage, "entry_point", None)
        if isinstance(entry, A.FunctionNode):
            yield stage, [entry]


def _parameter_type(parameter):
    declared = getattr(parameter, "param_type", None)
    return declared if declared is not None else getattr(parameter, "var_type", None)


def _declared_key(node):
    """A structural key of a declared type, ignoring references and fragment annotations; ``None`` if unknown."""
    if isinstance(node, A.ReferenceType):
        return _declared_key(node.referenced_type)
    value = type_from_node(node)
    if value is not None:
        return ("value", value.kind, value.bits, value.lanes)
    if isinstance(node, A.CooperativeMatrixType):
        element = _declared_key(node.element_type)
        return ("cooperative", element, repr(node.rows), repr(node.cols), node.scope)
    if isinstance(node, A.MatrixType):
        element = _declared_key(node.element_type)
        return ("matrix", element, repr(node.rows), repr(node.cols))
    if isinstance(node, A.ArrayType):
        return ("array", _declared_key(node.element_type), repr(node.size))
    if isinstance(node, A.PointerType):
        return ("pointer", _declared_key(node.pointee_type))
    if isinstance(node, (A.NamedType, A.PrimitiveType)):
        generic = tuple(
            _declared_key(argument)
            for argument in getattr(node, "generic_args", None) or []
        )
        if None in generic:
            return None
        return ("named", node.name, generic)
    return None


def _arity(function):
    parameters = list(getattr(function, "parameters", None) or [])
    required = sum(
        1
        for parameter in parameters
        if getattr(parameter, "default_value", None) is None
    )
    return required, len(parameters)


class CallGraph:
    """Functions in declaration order, and for each one the sorted indices of the functions it may call."""

    def __init__(self, program):
        self.functions: list = []
        self._scopes: list = []
        for function in getattr(program, "functions", None) or []:
            if isinstance(function, A.FunctionNode):
                self._add(function, None)
        for stage, functions in _stage_functions(program):
            for function in functions:
                self._add(function, stage)
        self.by_name: dict[str, list[int]] = {}
        for index, function in enumerate(self.functions):
            self.by_name.setdefault(function.name, []).append(index)
        self.scope = TypeScope(program)
        self._declare_globals(program)
        self.edges: list[dict[int, object]] = [{} for _ in self.functions]
        for index, function in enumerate(self.functions):
            self._collect(index, function)

    def _add(self, function, stage):
        if function is None or not getattr(function, "name", None):
            return
        if any(function is known for known in self.functions):
            return
        self.functions.append(function)
        self._scopes.append(stage)

    def _declare_globals(self, program):
        scope = self.scope
        for constant in getattr(program, "constants", None) or []:
            scope.declare(constant.name, constant.const_type)
        for variable in getattr(program, "global_variables", None) or []:
            scope.declare(variable.name, variable.var_type)
            scope.declare_block_members(variable)
        for cbuffer in getattr(program, "cbuffers", None) or []:
            scope.declare_cbuffer(cbuffer)

    # Collection -----------------------------------------------------------------------------------------------
    def _collect(self, index, function):
        if getattr(function, "body", None) is None:
            return
        scope = self.scope
        scope.push()
        stage = self._scopes[index]
        if stage is not None:
            for variable in getattr(stage, "local_variables", None) or []:
                scope.declare(variable.name, variable.var_type)
                scope.declare_block_members(variable)
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
        scope.push()
        for parameter in function.parameters or []:
            scope.declare(getattr(parameter, "name", None), _parameter_type(parameter))
        self._walk(function.body, self.edges[index])
        scope.pop()
        scope.pop()

    def _walk(self, node, edges):
        if isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, (A.BlockNode, A.ForNode))
        if scoped:
            self.scope.push()
        for field, key in list(child_slots(node)):
            self._walk(get_child(node, field, key), edges)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        elif isinstance(node, A.FunctionCallNode):
            for callee in self._callees(node):
                edges.setdefault(callee, node)
        if scoped:
            self.scope.pop()

    def _callees(self, call):
        if not isinstance(call.function, (A.IdentifierNode, str)):
            return []
        candidates = self.by_name.get(call_name(call), [])
        if not candidates:
            return []
        arguments = list(call.arguments or [])
        candidates = [
            index
            for index in candidates
            if _arity(self.functions[index])[0]
            <= len(arguments)
            <= _arity(self.functions[index])[1]
        ]
        if len(candidates) < 2:
            return candidates
        argument_types = [self.scope.infer(argument) for argument in arguments]
        argument_nodes = [self.scope.node_type(argument) for argument in arguments]
        for exact in (True, False):
            matching = [
                index
                for index in candidates
                if self._accepts(self.functions[index], argument_types, exact)
            ]
            if matching:
                declared = [
                    index
                    for index in matching
                    if self._declared_accepts(
                        self.functions[index], argument_types, argument_nodes
                    )
                ]
                return declared or matching
        return candidates

    @staticmethod
    def _declared_accepts(function, argument_types, argument_nodes):
        """Whether no named argument's declared type contradicts a parameter where value types cannot tell.

        Matrices, cooperative matrices, arrays and structures have no value type, so ``_accepts`` lets them match
        any parameter. Their declared types still separate overloads: a ``CooperativeMatrix<float, 8, 8>`` local
        does not bind to a ``vec2`` reference. Positions where both sides have value types are left to ``_accepts``.
        """
        for parameter, argument_type, node in zip(
            function.parameters or [], argument_types, argument_nodes
        ):
            declared = _parameter_type(parameter)
            if node is None or declared is None:
                continue
            if argument_type is not None and type_from_node(declared) is not None:
                continue
            expected, actual = _declared_key(declared), _declared_key(node)
            if expected is not None and actual is not None and expected != actual:
                return False
        return True

    @staticmethod
    def _accepts(function, argument_types, exact):
        """Whether no inferred argument type contradicts a parameter: same type if ``exact``, else same lanes."""
        for parameter, argument_type in zip(function.parameters or [], argument_types):
            parameter_type = type_from_node(_parameter_type(parameter))
            if parameter_type is None or argument_type is None:
                continue
            if exact and parameter_type != argument_type:
                return False
            if parameter_type.lanes != argument_type.lanes:
                return False
        return True

    # Cycles ---------------------------------------------------------------------------------------------------
    def find_cycle(self):
        """The first cycle in a depth-first search in declaration order: ``(function indices, closing call)``."""
        state = [0] * len(self.functions)  # 0 unvisited, 1 on the stack, 2 done
        stack: list[int] = []
        for root in range(len(self.functions)):
            if state[root]:
                continue
            found = self._search(root, state, stack)
            if found is not None:
                return found
        return None

    def _search(self, root, state, stack):
        state[root] = 1
        stack.append(root)
        pending = [iter(sorted(self.edges[root]))]
        while pending:
            source = stack[-1]
            target = next(pending[-1], None)
            if target is None:
                pending.pop()
                state[stack.pop()] = 2
                continue
            if state[target] == 1:
                cycle = stack[stack.index(target) :] + [target]
                return cycle, self.edges[source][target]
            if state[target] == 0:
                state[target] = 1
                stack.append(target)
                pending.append(iter(sorted(self.edges[target])))
        return None


def validate_recursion_free(program, target):
    """Raise :class:`RecursionUnsupportedError` when ``target`` forbids recursion and ``program`` recurses."""
    if target not in RECURSION_FREE_TARGETS or not isinstance(program, A.ShaderNode):
        return
    graph = CallGraph(program)
    found = graph.find_cycle()
    if found is None:
        return
    indices, call = found
    raise RecursionUnsupportedError(
        target,
        [graph.functions[index].name for index in indices],
        source_location=getattr(call, "source_location", None),
    )
