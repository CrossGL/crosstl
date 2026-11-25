"""CrossGL's ``workgroupUniformLoad(x)``: a read of workgroup memory that every invocation of the workgroup sees.

WGSL's ``workgroupUniformLoad(p)`` (WGSL, "Synchronization Built-in Functions") takes a pointer to workgroup memory,
executes a control barrier that synchronizes workgroup memory and atomic operations, and returns the value ``p``
points to to every invocation of the workgroup, as one uniform value. CrossGL's ``workgroupUniformLoad(x)`` takes the
memory itself: ``x`` is a ``shared`` (``groupshared``) variable declared at program or stage scope, or an element or
structure member of one, reached through array indices and member names. The result has the type of ``x``.

WGSL output keeps the call (the WGSL generator writes ``workgroupUniformLoad(&x)``), and so does CrossGL output.
No other target declares the function, so for them :func:`lower_workgroup_uniform_loads` rewrites each call into a
call of a CrossGL helper that does what WGSL's polyfills do (Tint's builtin polyfill writes the same body for its
HLSL, MSL and GLSL output): a ``workgroupBarrier()``, the read, then another ``workgroupBarrier()``. CrossGL's
``workgroupBarrier`` is WGSL's: an execution barrier of the workgroup that orders its shared memory (HLSL
``GroupMemoryBarrierWithGroupSync``, MSL ``threadgroup_barrier(mem_flags::mem_threadgroup)``, GLSL ``barrier()``,
CUDA ``__syncthreads()``, SPIR-V ``OpControlBarrier`` with ``WorkgroupMemory`` semantics). The second barrier makes
the read complete in every invocation before any of them writes ``x`` again, so all of them return the same value.
The helper reads ``x`` by name and takes the array indices of ``x`` as its parameters, so the call evaluates them
where the original call did. The read is an ordinary load, also of a variable the program updates with atomic
operations: the first barrier orders it after every access before the call and the second before every write after
it, so nothing races with it, and it returns what WGSL's ``atomic<T>`` overload returns.

The helper is added to the function list of the function that makes the call (its stage's functions, or the
program's), ahead of the existing functions, so it sees the same declarations; calls with the same variable, path
and index types share one. A call whose operand is not such a variable path (a local or function-scope ``shared``
variable, a vector component, an expression), whose type cannot be determined, or that takes another number of
operands is refused with :class:`WorkgroupUniformLoadError`, as is a program that defines a function with the
helpers' reserved prefix. A call of a function the program defines itself under this name is left alone.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .helpers import HELPER_MARK, is_helper
from .hub_types import call_name, iter_functions, type_from_node
from .rewrite import call
from .scoped_walk import ScopedWalker

__all__ = [
    "FUNCTION",
    "PREFIX",
    "WorkgroupUniformLoadError",
    "lower_workgroup_uniform_loads",
]

FUNCTION = "workgroupUniformLoad"
PREFIX = "cgl_rule_wul_"
# Targets that keep the call: WGSL declares it, and CrossGL output writes it as it stands.
_KEEPING_TARGETS = frozenset({"", "cgl", "crossgl", "wgsl"})
_MARK = "workgroup_uniform_load"
_SHARED_QUALIFIERS = frozenset({"shared", "groupshared"})
_BARRIER = "workgroupBarrier"
_VALUE = PREFIX + "value"
_INDEX = PREFIX + "index"
_INTEGER_SPELLINGS = {
    ("i", 32): "int",
    ("u", 32): "uint",
    ("i", 64): "int64_t",
    ("u", 64): "uint64_t",
}


class WorkgroupUniformLoadError(TranslationDiagnostic):
    """A ``workgroupUniformLoad`` call that CrossGL cannot carry to the target."""

    code = CODE_PREFIX + "workgroup-uniform-load-unsupported"


def _shared(declaration) -> bool:
    qualifiers = {
        str(value) for value in getattr(declaration, "qualifiers", None) or []
    }
    return bool(qualifiers & _SHARED_QUALIFIERS)


def _path(argument):
    """``(root, steps)`` of a variable path: ``steps`` holds ``("index", expr)`` and ``("member", name)`` in order.

    ``None`` when ``argument`` is not a name followed by array indices and member names.
    """
    steps = []
    node = argument
    if isinstance(node, A.UnaryOpNode) and node.op == "&" and not node.is_postfix:
        # A WGSL-style pointer operand: the memory is the same.
        node = node.operand
    while True:
        if isinstance(node, A.ArrayAccessNode):
            steps.append(("index", node.index_expr))
            node = node.array_expr
        elif isinstance(node, A.MemberAccessNode):
            steps.append(("member", node.member))
            node = node.object_expr
        elif isinstance(node, A.IdentifierNode):
            return node.name, steps[::-1]
        else:
            return None


class _Calls(ScopedWalker):
    """Rewrites the calls in place and records the helpers they need, per function list."""

    def __init__(self, program, target, user_defined):
        super().__init__(program)
        self.target = target
        self.user_defined = user_defined
        self.stage = None
        self.helpers = {}
        self.added = []
        self.count = 0

    def run(self, function, stage):
        self.stage = stage
        super().run(function, stage)

    def _error(self, message):
        return WorkgroupUniformLoadError(
            f"workgroupUniformLoad: {message} for {self.target}", target=self.target
        )

    def _root_declaration(self, name):
        """The program- or stage-scope ``shared`` declaration ``name`` names here, or ``None``."""
        declarations = []
        if self.stage is not None:
            declarations.extend(getattr(self.stage, "local_variables", None) or [])
        declarations.extend(getattr(self.program, "global_variables", None) or [])
        for declaration in declarations:
            if getattr(declaration, "name", None) != name:
                continue
            # A parameter or local of the same name hides the declaration.
            if self.scope.lookup(name) is not declaration.var_type:
                return None
            return declaration if _shared(declaration) else None
        return None

    def _index_type(self, index):
        value = self.scope.infer(index)
        if value is None or value.lanes != 1:
            return None
        return _INTEGER_SPELLINGS.get((value.kind, value.bits))

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode) or call_name(node) != FUNCTION:
            return
        if FUNCTION in self.user_defined:
            return
        arguments = list(node.arguments or [])
        if len(arguments) != 1:
            raise self._error(f"takes 1 operand, not {len(arguments)},")
        found = _path(arguments[0])
        if found is None:
            raise self._error(
                "the operand must be a shared variable, or an element or member of one,"
            )
        root, steps = found
        if root.startswith(PREFIX):
            raise self._error(f"{root} uses a name reserved for its helpers")
        declaration = self._root_declaration(root)
        if declaration is None:
            raise self._error(
                f"'{root}' is not a shared variable declared at program or stage scope"
            )
        operand = arguments[0]
        if isinstance(operand, A.UnaryOpNode):
            operand = operand.operand
        value_type = self.scope.node_type(operand)
        if value_type is None:
            raise self._error(
                f"the type of the operand of '{root}' cannot be determined"
            )
        index_types = []
        for kind, item in steps:
            if kind != "index":
                continue
            spelling = self._index_type(item)
            if spelling is None:
                raise self._error(
                    f"an array index of '{root}' is not an integer scalar of known type"
                )
            index_types.append(spelling)
        container = self._container()
        shape = tuple(item if kind == "member" else None for kind, item in steps)
        key = (id(container), root, shape, tuple(index_types))
        name = self.helpers.get(key)
        if name is None:
            name = f"{PREFIX}{len(self.helpers)}_{root}"
            self.helpers[key] = name
            helper = _helper(name, root, steps, value_type, index_types)
            self.added.append((container, helper))
        indices = [item for kind, item in steps if kind == "index"]
        node.function = node.name = A.IdentifierNode(name)
        node.arguments = node.args = indices
        carried = type_from_node(value_type)
        if carried is not None:
            # Later rule passes infer the call's type as they would the read's.
            node.annotations["cgl_rule_type"] = carried
        self.count += 1

    def _container(self):
        if self.stage is None:
            functions = getattr(self.program, "functions", None)
        else:
            functions = getattr(self.stage, "local_functions", None)
            if functions is None:
                functions = self.stage.local_functions = []
        if not isinstance(functions, list):
            raise self._error("the program has no function list to receive the helper")
        return functions


def _helper(name, root, steps, value_type, index_types):
    """``T name(I0 i0, ...) { workgroupBarrier(); T value = root...; workgroupBarrier(); return value; }``."""
    parameters = []
    read = A.IdentifierNode(root)
    for kind, item in steps:
        if kind == "member":
            read = A.MemberAccessNode(read, item)
            continue
        parameter = f"{_INDEX}{len(parameters)}"
        parameters.append(
            A.ParameterNode(parameter, A.PrimitiveType(index_types[len(parameters)]))
        )
        read = A.ArrayAccessNode(read, A.IdentifierNode(parameter))
    body = A.BlockNode(
        [
            A.ExpressionStatementNode(call(_BARRIER, [])),
            A.VariableNode(_VALUE, value_type, read),
            A.ExpressionStatementNode(call(_BARRIER, [])),
            A.ReturnNode(A.IdentifierNode(_VALUE)),
        ]
    )
    function = A.FunctionNode(name, value_type, parameters, body)
    function.annotations[HELPER_MARK] = _MARK
    return function


def lower_workgroup_uniform_loads(program, target) -> int:
    """Rewrite ``workgroupUniformLoad`` calls for ``target`` into helper calls; returns their count."""
    target = str(target or "").lower()
    if target in _KEEPING_TARGETS:
        return 0
    functions = list(iter_functions(program))
    for function in functions:
        name = getattr(function, "name", None) or ""
        if (
            isinstance(name, str)
            and name.startswith(PREFIX)
            and not is_helper(function)
        ):
            raise WorkgroupUniformLoadError(
                f"the program defines {name}, a name reserved for the workgroupUniformLoad helpers",
                target=target,
            )
    user_defined = {
        getattr(function, "name", None)
        for function in functions
        if getattr(function, "name", None) == FUNCTION
    }
    walker = _Calls(program, target, user_defined)
    walker.run_program()
    added = {}
    for container, helper in walker.added:
        added.setdefault(id(container), (container, []))[1].append(helper)
    for container, helpers in added.values():
        container[0:0] = helpers
    return walker.count
