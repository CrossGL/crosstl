"""Workgroup collectives as CrossGL helpers that combine the invocations' values through workgroup memory.

:mod:`crosstl.translator.workgroup_operations` defines CrossGL's reductions, scans, votes and broadcast over a whole
workgroup (``workgroupAdd``, ``workgroupInclusiveMin``, ``workgroupAll``, ``workgroupBroadcast``, ...). No target
but OpenCL C and Mojo has them as built-ins, so for every target that does not register them as native
(:func:`register_native_workgroup_collectives`) :func:`lower_workgroup_collectives` rewrites each call into a call of
a CrossGL helper, one per operation, operand type and workgroup size, built from operations every compute target
has:

1. a ``workgroupBarrier()``, so that every invocation has finished reading what an earlier collective wrote;
2. each invocation stores its value at its ``gl_LocalInvocationIndex`` in a ``shared`` array with one element per
   invocation of the workgroup (a broadcast's source invocation stores it at element 0);
3. a ``workgroupBarrier()``, after which every store is visible to every invocation;
4. each invocation combines the stored values in index order: all of them for a reduction or a vote, those at or
   below its own index for an inclusive scan, and those below it for an exclusive scan, which starts from the
   operator's identity as SPIR-V's group operations define it (:mod:`.partitioned_waves`). A broadcast reads
   element 0.

Every invocation combines the same values in the same order, so all of them receive the same result, also for
floating-point operands. A ``bool`` is stored as a ``uint``. The helpers of one scope share one array per stored
type: each helper starts with a barrier, so no invocation stores into the array while another still reads it.

The workgroup size is the literal or program-constant size of the calling stage (``layout(local_size_x = ...)``,
``@numthreads``) or of the ``@compute`` function entry. A helper is added to the scope of the call: the stage of a
stage function, or, for a shader-scope function, the one workgroup stage whose entry point reaches it, which
:mod:`.stage_bound_functions` then gives its own copy of the function; a shader-scope function that only shader-scope
``@compute`` entries of one size reach, or that no entry reaches, keeps its helpers at shader scope. A call is
refused with :class:`WorkgroupCollectiveError` when its operand type cannot be inferred or has no helper, when it
is reached from a stage without workgroups, from entries of different sizes or of a size the host chooses
(:mod:`crosstl.translator.host_workgroup_size`) or sets as a specialization constant, when it is outside every
function body, or when the program defines a name with the helpers' reserved prefix. A call of a function the
program defines itself under a collective's name is left alone.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..host_workgroup_size import (
    HOST_WORKGROUP_SIZE_KEY,
    function_entry_execution_config,
    is_compute_function_entry,
)
from ..refusals import RegistryRefusal
from ..workgroup_operations import WORKGROUP_COLLECTIVES, WORKGROUP_VOTES
from .helpers import HELPER_MARK, is_helper
from .hub_types import BOOL, call_name, iter_functions
from .lane_reads import parse_helper_function, type_spelling, wave_operand_type
from .partitioned_waves import _COMBINE, _COMBINE_BOOL, _IDENTITIES
from .scoped_walk import ScopedWalker, program_stages

__all__ = [
    "PREFIX",
    "WorkgroupCollectiveError",
    "lower_workgroup_collectives",
    "native_workgroup_collectives",
    "register_native_workgroup_collectives",
]

PREFIX = "cgl_rule_wgc_"
_MARK = "workgroup_collective"
# Targets whose output keeps the calls: CrossGL writes them as they stand.
_NATIVE: dict[str, frozenset[str]] = {
    name: frozenset(WORKGROUP_COLLECTIVES) for name in ("", "cgl", "crossgl")
}
_WORKGROUP_STAGES = frozenset({"compute", "mesh", "task"})
_SIZE_KEYS = ("local_size_x", "local_size_y", "local_size_z")
_SIZE_LISTS = ("numthreads", "workgroup_size", "local_size")
_INDEX_TYPES = {("i", 32): "int", ("u", 32): "uint"}


class WorkgroupCollectiveError(TranslationDiagnostic):
    """A workgroup collective that CrossGL cannot compute for the target."""

    code = CODE_PREFIX + "workgroup-collective-unsupported"


def register_native_workgroup_collectives(target, operations):
    """Declare that ``target``'s generator writes ``operations`` itself, so their calls are not lowered."""
    unknown = sorted(set(operations) - set(WORKGROUP_COLLECTIVES))
    if unknown:
        raise RegistryRefusal(f"not workgroup collectives: {', '.join(unknown)}")
    target = str(target).lower()
    _NATIVE[target] = _NATIVE.get(target, frozenset()) | frozenset(operations)


def native_workgroup_collectives(target) -> frozenset[str]:
    return _NATIVE.get(str(target or "").lower(), frozenset())


def _stage_kind(stage) -> str:
    kind = getattr(stage, "stage", None)
    return str(getattr(kind, "value", kind)).lower()


def _integer_constants(program):
    constants = {}
    for constant in getattr(program, "constants", None) or []:
        value = getattr(constant, "value", None)
        if isinstance(value, A.LiteralNode) and not isinstance(value.value, bool):
            text = str(value.value).rstrip("uU")
            if text.isdigit():
                constants[constant.name] = int(text)
    return constants


class _Owner:
    """Where the helpers of a call go and the workgroup size they combine over."""

    def __init__(self, functions, variables, size):
        self.functions = functions
        self.variables = variables
        self.size = size


class _Collectives(ScopedWalker):
    def __init__(self, program, target, native, user_defined):
        super().__init__(program)
        self.target = target
        self.native = native
        self.user_defined = user_defined
        self.constants = _integer_constants(program)
        self.stage = None
        self.function = None
        self.helpers = {}
        self.scratch = {}
        self.added = []
        self.count = 0
        self._owners = {}
        self._reached = None

    def error(self, message):
        return WorkgroupCollectiveError(message, target=self.target)

    def run(self, function, stage):
        self.stage, self.function = stage, function
        super().run(function, stage)

    # Workgroup sizes and helper scopes.

    def size(self, config, where):
        config = dict(config or {})
        record = config.get(HOST_WORKGROUP_SIZE_KEY)
        if record:
            reason = record.get("reason", "") if isinstance(record, dict) else ""
            raise self.error(
                f"a workgroup collective in {where}, whose workgroup size the host chooses ({reason}), needs a "
                "workgroup size the shader declares"
            )
        if any(str(key).endswith("_id") for key in config):
            raise self.error(
                f"a workgroup collective in {where} needs a workgroup size that is not a specialization constant"
            )
        for key in _SIZE_LISTS:
            listed = config.get(key)
            if isinstance(listed, (list, tuple)) and not any(
                k in config for k in _SIZE_KEYS
            ):
                config.update(zip(_SIZE_KEYS, listed))
        total = 1
        for key in _SIZE_KEYS:
            text = str(config.get(key, "1")).strip().rstrip("uU")
            value = int(text) if text.isdigit() else self.constants.get(text)
            if value is None or value < 1:
                raise self.error(
                    f"a workgroup collective in {where} needs a workgroup size of integer constants, not "
                    f"{key} = {text}"
                )
            total *= value
        return total

    def stage_owner(self, stage):
        key = id(stage)
        if key not in self._owners:
            kind = _stage_kind(stage)
            if kind not in _WORKGROUP_STAGES:
                raise self.error(
                    f"a {kind} stage has no workgroup, so it cannot call a workgroup collective"
                )
            if getattr(stage, "local_functions", None) is None:
                stage.local_functions = []
            if getattr(stage, "local_variables", None) is None:
                stage.local_variables = []
            self._owners[key] = _Owner(
                stage.local_functions,
                stage.local_variables,
                self.size(stage.execution_config, f"the {kind} stage"),
            )
        return self._owners[key]

    def shader_owner(self, size):
        if getattr(self.program, "functions", None) is None:
            self.program.functions = []
        if getattr(self.program, "global_variables", None) is None:
            self.program.global_variables = []
        return _Owner(self.program.functions, self.program.global_variables, size)

    def entries(self):
        """``(stage or None, entry, roots)`` of every entry point: a stage's, or a shader-scope ``@compute`` one."""
        found = []
        for stage in program_stages(self.program):
            entry = getattr(stage, "entry_point", None)
            if isinstance(entry, A.FunctionNode):
                roots = [entry, *(getattr(stage, "local_functions", None) or [])]
                found.append((stage, entry, roots))
        for function in getattr(self.program, "functions", None) or []:
            if isinstance(function, A.FunctionNode) and is_compute_function_entry(
                function
            ):
                found.append((None, function, [function]))
        return found

    def reached(self):
        """The ids of the shader-scope functions each entry reaches, by entry."""
        if self._reached is None:
            by_name = {}
            for function in getattr(self.program, "functions", None) or []:
                if isinstance(function, A.FunctionNode):
                    by_name.setdefault(function.name, []).append(function)
            self._reached = []
            for stage, entry, roots in self.entries():
                seen, stack = set(), list(roots)
                while stack:
                    for node in stack.pop().walk():
                        if isinstance(node, A.FunctionCallNode):
                            for callee in by_name.get(call_name(node), ()):
                                if id(callee) not in seen:
                                    seen.add(id(callee))
                                    stack.append(callee)
                self._reached.append((stage, entry, seen))
        return self._reached

    def owner(self):
        if self.stage is not None:
            return self.stage_owner(self.stage)
        function = self.function
        if is_compute_function_entry(function):
            config = function_entry_execution_config(function)
            return self.shader_owner(
                self.size(config, f"entry point '{function.name}'")
            )
        callers = [
            (stage, entry)
            for stage, entry, seen in self.reached()
            if id(function) in seen and entry is not function
        ]
        if not callers:
            # No entry reaches the function; it combines over the program's one workgroup entry, if it has one.
            callers = [
                (stage, entry)
                for stage, entry, _ in self.reached()
                if stage is None or _stage_kind(stage) in _WORKGROUP_STAGES
            ]
            if len(callers) != 1:
                raise self.error(
                    f"function '{function.name}' calls a workgroup collective, and no single compute, mesh or "
                    "task entry point reaches it"
                )
            stage, entry = callers[0]
            if stage is None:
                where = f"entry point '{entry.name}'"
                config = function_entry_execution_config(entry)
                return self.shader_owner(self.size(config, where))
            return self.shader_owner(self.stage_owner(stage).size)
        stages = {id(stage): stage for stage, _ in callers if stage is not None}
        if len(stages) == 1 and all(stage is not None for stage, _ in callers):
            return self.stage_owner(next(iter(stages.values())))
        if stages:
            raise self.error(
                f"function '{function.name}' calls a workgroup collective and is reached from several stages or "
                "entry points"
            )
        sizes = {
            self.size(
                function_entry_execution_config(entry), f"entry point '{entry.name}'"
            )
            for _, entry in callers
        }
        if len(sizes) != 1:
            raise self.error(
                f"function '{function.name}' calls a workgroup collective and is reached from entry points of "
                f"different workgroup sizes ({', '.join(map(str, sorted(sizes)))})"
            )
        return self.shader_owner(sizes.pop())

    # Calls.

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        operation = call_name(node)
        if (
            operation not in WORKGROUP_COLLECTIVES
            or operation in self.native
            or operation in self.user_defined
        ):
            return
        operator, scope = WORKGROUP_COLLECTIVES[operation]
        arguments = list(node.arguments or [])
        wanted = 2 if scope == "broadcast" else 1
        if len(arguments) != wanted:
            raise self.error(
                f"{operation} takes {wanted} operand{'s' if wanted > 1 else ''}, not {len(arguments)}"
            )
        value = wave_operand_type(self.scope, arguments[0])
        spelling = type_spelling(value) if value is not None else None
        if spelling is None:
            raise self.error(
                f"{operation}: the operand type cannot be inferred or has no workgroup helper"
            )
        self.check_operand(operation, operator, scope, value)
        index = ""
        if scope == "broadcast":
            source = self.scope.infer(arguments[1])
            index = _INDEX_TYPES.get((source.kind, source.bits)) if source else None
            if index is None or source.lanes != 1:
                raise self.error(
                    f"{operation}: the invocation index is not an int or uint scalar of known type"
                )
        owner = self.owner()
        name = self.helper(owner, operation, operator, scope, value, spelling, index)
        node.function = node.name = A.IdentifierNode(name)
        node.arguments = node.args = arguments
        # Later rule passes infer the call's type as they would the collective's.
        node.annotations["cgl_rule_type"] = value
        self.count += 1

    def check_operand(self, operation, operator, scope, value):
        if operation in WORKGROUP_VOTES:
            valid = value == BOOL
        elif scope == "broadcast":
            valid = value.kind != "b" or value.lanes == 1
        elif value.kind == "b":
            valid = value.lanes == 1 and operator in ("and", "or", "xor")
        else:
            valid = (operator, value.kind) in _IDENTITIES
        if not valid:
            raise self.error(
                f"{operation} has no workgroup helper for a {value.kind}{value.bits}x{value.lanes} operand"
            )

    def helper(self, owner, operation, operator, scope, value, spelling, index):
        key = (id(owner.functions), operation, spelling, index, owner.size)
        if key in self.helpers:
            return self.helpers[key]
        carrier = "uint" if value.kind == "b" else spelling
        scratch_key = (id(owner.variables), carrier)
        if scratch_key not in self.scratch:
            self.scratch[scratch_key] = [f"{PREFIX}{carrier}", owner.variables, 0]
        scratch = self.scratch[scratch_key]
        scratch[2] = max(scratch[2], owner.size)
        name = f"{PREFIX}{operation}_{spelling}"
        taken = set(self.helpers.values())
        suffix = 1
        while name in taken:
            suffix += 1
            name = f"{PREFIX}{operation}_{spelling}_{suffix}"
        self.helpers[key] = name
        source = _helper_source(
            name, operator, scope, value, spelling, index, scratch[0], owner.size
        )
        function = parse_helper_function(source)
        function.annotations[HELPER_MARK] = _MARK
        self.added.append((owner.functions, function))
        return name

    def finish(self):
        for node in self.program.walk():
            if (
                isinstance(node, A.FunctionCallNode)
                and call_name(node) in WORKGROUP_COLLECTIVES
                and call_name(node) not in self.native
                and call_name(node) not in self.user_defined
            ):
                raise self.error(
                    f"{call_name(node)} is outside every function body the lowering reaches"
                )
        added = {}
        for functions, helper in self.added:
            added.setdefault(id(functions), (functions, []))[1].append(helper)
        for functions, helpers in added.values():
            functions[0:0] = helpers
        for name, variables, size in self.scratch.values():
            variables[0:0] = [_scratch_declaration(name, size)]


def _store(value, operand):
    return f"{operand} ? 1u : 0u" if value.kind == "b" else operand


def _load(value, scratch, lane):
    read = f"{scratch}[{lane}]"
    return f"({read} != 0u)" if value.kind == "b" else read


def _identity(operator, value, spelling):
    identity = _IDENTITIES[(operator, value.kind)]
    if value.kind == "f" and value.bits != 32:
        identity = f"{type_spelling(value.scalar)}({identity})"
    if value.lanes > 1:
        identity = f"{spelling}({identity})"
    return identity


def _helper_source(name, operator, scope, value, spelling, index, scratch, size):
    store = f"{scratch}[{{0}}] = {_store(value, 'value')};"
    if scope == "broadcast":
        return (
            f"{spelling} {name}({spelling} value, {index} source) {{\n"
            "    workgroupBarrier();\n"
            "    if (gl_LocalInvocationIndex == uint(source)) {\n"
            f"        {store.format('0u')}\n"
            "    }\n"
            "    workgroupBarrier();\n"
            f"    return {_load(value, scratch, '0u')};\n"
            "}\n"
        )
    combine = (_COMBINE_BOOL if value.kind == "b" else _COMBINE)[operator]
    if scope == "exclusive":
        start, first, bound = _identity(operator, value, spelling), "0u", "lane < index"
    else:
        start = _load(value, scratch, "0u")
        first = "1u"
        bound = "lane <= index" if scope == "inclusive" else f"lane < {size}u"
    return (
        f"{spelling} {name}({spelling} value) {{\n"
        "    uint index = gl_LocalInvocationIndex;\n"
        "    workgroupBarrier();\n"
        f"    {store.format('index')}\n"
        "    workgroupBarrier();\n"
        f"    {spelling} result = {start};\n"
        f"    for (uint lane = {first}; {bound}; lane++) {{\n"
        f"        {spelling} other = {_load(value, scratch, 'lane')};\n"
        f"        result = {combine};\n"
        "    }\n"
        "    return result;\n"
        "}\n"
    )


def _scratch_declaration(name, size):
    """The ``shared T name[size];`` declaration of a helper array, as the CrossGL parser reads one."""
    carrier = name[len(PREFIX) :]
    function = parse_helper_function(
        f"void {PREFIX}declaration() {{\n    {carrier} {name}[{size}];\n}}\n"
    )
    declaration = function.body.statements[0]
    declaration.qualifiers = ["shared"]
    declaration.annotations[HELPER_MARK] = _MARK
    return declaration


def lower_workgroup_collectives(program, target) -> int:
    """Rewrite ``program``'s workgroup collectives for ``target`` into helper calls; returns their count."""
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "").lower()
    native = native_workgroup_collectives(target)
    if native >= frozenset(WORKGROUP_COLLECTIVES):
        return 0
    if not any(
        isinstance(node, A.FunctionCallNode)
        and call_name(node) in WORKGROUP_COLLECTIVES
        and call_name(node) not in native
        for node in program.walk()
    ):
        return 0
    functions = list(iter_functions(program))
    declared = functions + list(getattr(program, "global_variables", None) or [])
    for stage in program_stages(program):
        declared += list(getattr(stage, "local_variables", None) or [])
    for declaration in declared:
        name = getattr(declaration, "name", None)
        if isinstance(name, str) and name.startswith(PREFIX):
            if not is_helper(declaration):
                raise WorkgroupCollectiveError(
                    f"the program declares {name}, a name reserved for the workgroup collective helpers",
                    target=target,
                )
    user_defined = {
        function.name
        for function in functions
        if getattr(function, "name", None) in WORKGROUP_COLLECTIVES
    }
    walker = _Collectives(program, target, native, user_defined)
    walker.run_program()
    walker.finish()
    return walker.count
