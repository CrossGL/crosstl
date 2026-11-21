"""CrossGL work-graph nodes: the ``node`` stage of a D3D12 work graph, its rules, and the targets that declare it.

A ``node { ... }`` stage block holds one node shader of a work graph (DirectX-Specs, "D3D12 Work Graphs"; HLSL
``[Shader("node")]`` compiled with ``-T lib_6_8``). CrossGL spells the node's object types, their methods and its
intrinsics as HLSL does, and its attributes in CrossGL's lower-case form:

* entry attributes: ``@node_launch(broadcasting | coalescing | thread)``, broadcasting when it is absent;
  ``@node_dispatch_grid(x, y, z)`` or ``@node_max_dispatch_grid(x, y, z)``; ``@node_max_recursion_depth(n)``;
  ``@node_id("name")`` or ``@node_id("name", index)``, the function name and index 0 when it is absent;
  ``@node_share_input_of("name"[, index])``; ``@node_is_program_entry``; ``@node_local_root_arguments_table_index(n)``;
  and the thread-group size ``@numthreads(x, y, z)``, with ``@wave_size`` as for compute.
* input record parameters, at most one: ``DispatchNodeInputRecord<R>`` and ``RWDispatchNodeInputRecord<R>``
  (broadcasting), ``GroupNodeInputRecords<R>``, ``RWGroupNodeInputRecords<R>`` and ``EmptyNodeInput`` (coalescing,
  ``@max_records(n)`` gives the batch size), ``ThreadNodeInputRecord<R>`` and ``RWThreadNodeInputRecord<R>`` (thread).
* output parameters: ``NodeOutput<R>``, ``NodeOutputArray<R>``, ``EmptyNodeOutput`` and ``EmptyNodeOutputArray``, with
  ``@max_records(n)`` or ``@max_records_shared_with(parameter)``, ``@node_id(...)``, ``@allow_sparse_nodes``, and on
  the arrays only ``@node_array_size(n)`` and ``@unbounded_sparse_nodes``.
* ``@globallycoherent`` on a read-write input record, and ``@node_track_rw_input_sharing`` on a record structure;
  a record member ``@ SV_DispatchGrid`` (``uint``, ``uvec2`` or ``uvec3``, or their 16-bit forms) carries the grid
  of a ``@node_max_dispatch_grid`` node.
* system values as entry parameters, in CrossGL's ``gl_`` spelling or HLSL's ``SV_`` one: ``gl_WorkGroupID``,
  ``gl_LocalInvocationID``, ``gl_GlobalInvocationID`` and ``gl_LocalInvocationIndex``.
* methods: ``Get()``, ``Get(i)``, ``[i]`` and ``Count()`` on input records, ``FinishedCrossGroupSharing()`` on
  ``RWDispatchNodeInputRecord``; ``GetThreadNodeOutputRecords(n)`` and ``GetGroupNodeOutputRecords(n)`` on outputs,
  which return ``ThreadNodeOutputRecords<R>`` and ``GroupNodeOutputRecords<R>`` with ``Get``, ``[i]`` and
  ``OutputComplete()``; ``IsValid()``, ``ThreadIncrementOutputCount(n)`` and ``GroupIncrementOutputCount(n)``;
  ``GetRemainingRecursionLevels()``; and ``Barrier(memory_or_object, flags)`` with the flags in
  :data:`BARRIER_FLAGS`.

:func:`check_work_graph_nodes` refuses, with :class:`WorkGraphNodeError`, each rule DXC 1.9.2602.24 enforces for
``lib_6_8``: an unknown launch mode; a broadcasting node without exactly one dispatch grid, or a grid on another
launch; a maximum grid whose input record has no ``SV_DispatchGrid`` member; a thread launch with a thread group
other than (1, 1, 1); more than one input, or an input the launch does not take; a system value the launch does not
provide (broadcasting provides all four, coalescing only the thread's position in its group, thread launch none);
``@globallycoherent`` on a read-only record; ``FinishedCrossGroupSharing()`` on a record whose structure is not
marked ``@node_track_rw_input_sharing``; and ``@node_array_size`` or ``@unbounded_sparse_nodes`` on a single output.

Only targets registered with :func:`register_work_graph_target` declare node shaders. For any other target
:func:`refuse_work_graph_nodes` raises :class:`~..codegen.stage_refusals.StageUnsupportedError` for a node stage and
:class:`WorkGraphUnsupportedError` for a work-graph type, intrinsic, barrier flag or ``SV_DispatchGrid`` member in a
program without one, because no other API has work graphs. CrossGL output keeps every construct.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.stage_refusals import StageUnsupportedError
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name

__all__ = [
    "BARRIER_FLAGS",
    "EMPTY_NODE_TYPES",
    "INPUT_RECORD_TYPES",
    "LAUNCH_MODES",
    "NODE_FUNCTIONS",
    "OUTPUT_TYPES",
    "WORK_GRAPH_TYPES",
    "WorkGraphNodeError",
    "WorkGraphUnsupportedError",
    "check_work_graph_nodes",
    "refuse_work_graph_nodes",
    "register_work_graph_target",
    "work_graph_targets",
]

LAUNCH_MODES = ("broadcasting", "coalescing", "thread")
# The input record types each launch mode takes (D3D12 Work Graphs, "Node input declaration").
INPUT_RECORD_TYPES = {
    "broadcasting": ("DispatchNodeInputRecord", "RWDispatchNodeInputRecord"),
    "coalescing": (
        "GroupNodeInputRecords",
        "RWGroupNodeInputRecords",
        "EmptyNodeInput",
    ),
    "thread": ("ThreadNodeInputRecord", "RWThreadNodeInputRecord"),
}
_INPUTS = frozenset(name for names in INPUT_RECORD_TYPES.values() for name in names)
_READ_WRITE_INPUTS = frozenset(name for name in _INPUTS if name.startswith("RW"))
OUTPUT_TYPES = (
    "NodeOutput",
    "NodeOutputArray",
    "EmptyNodeOutput",
    "EmptyNodeOutputArray",
)
_OUTPUT_ARRAYS = frozenset({"NodeOutputArray", "EmptyNodeOutputArray"})
# The records a node requests from an output, which it writes and then completes.
OUTPUT_RECORD_TYPES = ("ThreadNodeOutputRecords", "GroupNodeOutputRecords")
# The work-graph types written without a record type argument.
EMPTY_NODE_TYPES = frozenset(
    {"EmptyNodeInput", "EmptyNodeOutput", "EmptyNodeOutputArray"}
)
WORK_GRAPH_TYPES = _INPUTS | frozenset(OUTPUT_TYPES) | frozenset(OUTPUT_RECORD_TYPES)
NODE_FUNCTIONS = frozenset({"GetRemainingRecursionLevels"})
# The memory types and semantic flags of Shader Model 6.8's Barrier (HLSL "Barrier", the MEMORY_TYPE_FLAG and
# BARRIER_SEMANTIC_FLAG enumerations); the node record memory types exist only in work graphs.
BARRIER_FLAGS = frozenset(
    {
        "UAV_MEMORY",
        "GROUP_SHARED_MEMORY",
        "NODE_INPUT_MEMORY",
        "NODE_OUTPUT_MEMORY",
        "ALL_MEMORY",
        "GROUP_SYNC",
        "GROUP_SCOPE",
        "DEVICE_SCOPE",
    }
)
DISPATCH_GRID = "SV_DispatchGrid"
TRACK_RW_INPUT_SHARING = "node_track_rw_input_sharing"
_GRID_ATTRIBUTES = ("node_dispatch_grid", "node_max_dispatch_grid")
_THREAD_GROUP_ATTRIBUTES = ("numthreads", "workgroup_size", "local_size")
_ARRAY_ONLY_ATTRIBUTES = ("node_array_size", "unbounded_sparse_nodes")
# The system values each launch mode provides, by the GLSL name CrossGL gives them (D3D12 Work Graphs, "Node shader
# system values"); DXC rejects any other, as "Invalid system value semantic 'SV_GroupID' for launchtype 'Coalescing'".
_SYSTEM_VALUES = {
    "SV_GroupID": "gl_WorkGroupID",
    "SV_GroupThreadID": "gl_LocalInvocationID",
    "SV_DispatchThreadID": "gl_GlobalInvocationID",
    "SV_GroupIndex": "gl_LocalInvocationIndex",
}
# Each node system value under its HLSL name and its CrossGL name.
_SYSTEM_VALUE_NAMES = {**_SYSTEM_VALUES, **{value: value for value in _SYSTEM_VALUES.values()}}
_LAUNCH_SYSTEM_VALUES = {
    "broadcasting": frozenset(_SYSTEM_VALUES.values()),
    "coalescing": frozenset({"gl_LocalInvocationID", "gl_LocalInvocationIndex"}),
    "thread": frozenset(),
}
_GRID_SCALARS = frozenset({"uint", "uint16_t", "u16", "ushort", "min16uint"})

# Target names whose generators declare node shaders; CrossGL output keeps them as written.
_TARGETS: set[str] = {"cgl", "crossgl"}


def register_work_graph_target(*targets: str) -> None:
    """Declare that the generators of ``targets`` write node shaders, so programs with them reach those generators."""
    _TARGETS.update(str(target).lower() for target in targets)


def work_graph_targets() -> frozenset[str]:
    return frozenset(_TARGETS)


class WorkGraphNodeError(TranslationDiagnostic):
    """A node shader that breaks a rule of D3D12 work graphs."""

    def __init__(self, target, rule, message, *, source_location=None):
        self.source_location = source_location
        self.rule = rule
        self.missing_capabilities = (f"crossgl.work-graph.{rule}",)
        super().__init__(
            f"work-graph node: {message}",
            target=target,
            code=f"{CODE_PREFIX}work-graph-{rule}",
        )


class WorkGraphUnsupportedError(TranslationDiagnostic):
    """A work-graph construct for a target without work graphs, or outside a node stage."""

    def __init__(self, target, construct, reason, *, source_location=None):
        self.source_location = source_location
        self.construct = construct
        self.missing_capabilities = (f"{target}.work-graph",)
        super().__init__(
            f"{construct} cannot be carried to {target}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-work-graph-unsupported",
        )


def _node_stages(program):
    stages = getattr(program, "stages", None) or {}
    return [
        stage
        for key, stage in stages.items()
        if getattr(stage, "stage", key) == A.ShaderStage.NODE
        or key == A.ShaderStage.NODE
    ]


def _type_name(node) -> str:
    return str(getattr(node, "name", "") or "") if isinstance(node, A.NamedType) else ""


def _attribute_names(node):
    return [
        str(getattr(a, "name", "") or "")
        for a in getattr(node, "attributes", None) or []
    ]


def _construct(node) -> str:
    """The work-graph construct ``node`` names, or an empty string."""
    if isinstance(node, A.NamedType) and node.name in WORK_GRAPH_TYPES:
        return f"work-graph type {node.name}"
    if isinstance(node, A.FunctionCallNode) and call_name(node) in NODE_FUNCTIONS:
        return f"work-graph intrinsic {call_name(node)}"
    if isinstance(node, A.IdentifierNode) and node.name in BARRIER_FLAGS:
        return f"barrier flag {node.name}"
    if isinstance(node, A.AttributeNode) and node.name == DISPATCH_GRID:
        return f"record member semantic {DISPATCH_GRID}"
    if isinstance(node, A.AttributeNode) and node.name == TRACK_RW_INPUT_SHARING:
        return f"record attribute @{TRACK_RW_INPUT_SHARING}"
    return ""


def _walk_with_attributes(root):
    """``root.walk()`` plus the attributes of the declarations it reaches, which ``walk`` leaves out."""
    for node in root.walk():
        yield node
        for attribute in getattr(node, "attributes", None) or []:
            if isinstance(attribute, A.AttributeNode):
                yield attribute


def _first_construct(roots):
    for root in roots:
        for node in _walk_with_attributes(root):
            found = _construct(node)
            if found:
                return found, node
    return "", None


def refuse_work_graph_nodes(program, target) -> None:
    """Refuse every work-graph construct for a ``target`` that has not registered node shaders, and any outside a
    program with a node stage for every target."""
    if not isinstance(program, A.ShaderNode):
        return
    target = str(target or "").lower()
    nodes = _node_stages(program)
    if target in _TARGETS:
        others = [
            stage for stage in (program.stages or {}).values() if stage not in nodes
        ]
        roots = others if nodes else [program]
        construct, node = _first_construct(roots)
        if construct:
            raise WorkGraphUnsupportedError(
                target,
                construct,
                "work-graph objects, intrinsics and record semantics exist only in a node stage",
                source_location=getattr(node, "source_location", None),
            )
        return
    reason = "D3D12 work graphs exist only in HLSL, and this generator writes no node shaders"
    if nodes:
        raise StageUnsupportedError(target, target, ["node"], reason)
    construct, node = _first_construct([program])
    if construct:
        raise WorkGraphUnsupportedError(
            target,
            construct,
            reason,
            source_location=getattr(node, "source_location", None),
        )


def _literal_ints(attribute):
    values = []
    for argument in getattr(attribute, "arguments", None) or []:
        value = getattr(argument, "value", None)
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return None
    return tuple(values)


def _argument_word(argument) -> str:
    if isinstance(argument, A.IdentifierNode):
        return str(argument.name)
    return str(getattr(argument, "value", "") or "").strip("\"'")


class _NodeChecker:
    def __init__(self, program, stage, target):
        self.program = program
        self.stage = stage
        self.target = target
        self.entry = getattr(stage, "entry_point", None)

    def refuse(self, rule, message, node=None):
        raise WorkGraphNodeError(
            self.target,
            rule,
            f"'{getattr(self.entry, 'name', 'main')}' {message}",
            source_location=getattr(node or self.entry, "source_location", None),
        )

    def attribute(self, name):
        for attribute in getattr(self.entry, "attributes", None) or []:
            if getattr(attribute, "name", None) == name:
                return attribute
        return None

    def struct(self, name):
        structs = list(getattr(self.program, "structs", None) or [])
        structs += list(getattr(self.stage, "local_structs", None) or [])
        for struct in structs:
            if getattr(struct, "name", None) == name:
                return struct
        return None

    def record_struct(self, type_node):
        arguments = getattr(type_node, "generic_args", None) or []
        return self.struct(_type_name(arguments[0])) if arguments else None

    def launch(self):
        attribute = self.attribute("node_launch")
        if attribute is None:
            return "broadcasting"
        arguments = getattr(attribute, "arguments", None) or []
        mode = _argument_word(arguments[0]).lower() if len(arguments) == 1 else ""
        if mode not in LAUNCH_MODES:
            self.refuse(
                "launch-mode",
                f"declares @node_launch with {len(arguments)} argument(s) "
                f"'{mode}'; the launch mode is one of {', '.join(LAUNCH_MODES)}",
                attribute,
            )
        return mode

    def check(self):
        if self.entry is None:
            return
        launch = self.launch()
        inputs = self.check_parameters(launch)
        self.check_grid(launch, inputs)
        self.check_thread_group(launch)
        self.check_cross_group_sharing()

    def check_parameters(self, launch):
        inputs = []
        for parameter in getattr(self.entry, "parameters", None) or []:
            kind = _type_name(getattr(parameter, "param_type", None))
            names = _attribute_names(parameter)
            if kind in _INPUTS:
                inputs.append(parameter)
                if kind not in INPUT_RECORD_TYPES[launch]:
                    self.refuse(
                        "input-kind",
                        f"is a {launch} node, which takes "
                        f"{' or '.join(INPUT_RECORD_TYPES[launch])} as its input, not {kind}",
                        parameter,
                    )
            if "globallycoherent" in names and kind not in _READ_WRITE_INPUTS:
                self.refuse(
                    "globallycoherent",
                    f"marks parameter '{parameter.name}' of type {kind or 'non-record'} @globallycoherent; "
                    "only a read-write input record can be",
                    parameter,
                )
            for name in _ARRAY_ONLY_ATTRIBUTES:
                if name in names and kind not in _OUTPUT_ARRAYS:
                    self.refuse(
                        "output-array-attribute",
                        f"gives parameter '{parameter.name}' @{name}, which only a NodeOutputArray or "
                        "EmptyNodeOutputArray takes",
                        parameter,
                    )
            self.check_system_value(launch, parameter, names)
        if len(inputs) > 1:
            self.refuse(
                "input-count",
                f"takes {len(inputs)} input records; a node takes at most one",
                inputs[1],
            )
        return inputs

    def check_system_value(self, launch, parameter, names):
        for name in names:
            value = _SYSTEM_VALUE_NAMES.get(name)
            if value is not None and value not in _LAUNCH_SYSTEM_VALUES[launch]:
                provided = sorted(_LAUNCH_SYSTEM_VALUES[launch]) or ["none"]
                self.refuse(
                    "system-value",
                    f"reads @{name} in parameter '{parameter.name}', which a {launch} node does not have; "
                    f"it has {', '.join(provided)}",
                    parameter,
                )

    def check_grid(self, launch, inputs):
        grids = [name for name in _GRID_ATTRIBUTES if self.attribute(name) is not None]
        if launch != "broadcasting":
            if grids:
                self.refuse(
                    "dispatch-grid",
                    f"is a {launch} node and declares @{grids[0]}, which only a broadcasting node takes",
                    self.attribute(grids[0]),
                )
            return
        if len(grids) != 1:
            self.refuse(
                "dispatch-grid",
                "is a broadcasting node and must declare exactly one of @node_dispatch_grid and "
                f"@node_max_dispatch_grid, not {len(grids)}",
            )
        size = _literal_ints(self.attribute(grids[0]))
        if size is None or len(size) != 3 or min(size) < 1:
            self.refuse(
                "dispatch-grid",
                f"declares @{grids[0]} without three positive integer dimensions",
                self.attribute(grids[0]),
            )
        if grids[0] == "node_max_dispatch_grid":
            record = self.record_struct(inputs[0].param_type) if inputs else None
            members = getattr(record, "members", None) or []
            grid = [m for m in members if DISPATCH_GRID in _attribute_names(m)]
            if not grid:
                self.refuse(
                    "dispatch-grid",
                    "declares @node_max_dispatch_grid, so its input record needs a member with "
                    f"@{DISPATCH_GRID}",
                    self.attribute(grids[0]),
                )
            self.check_grid_member(grid[0])

    def check_grid_member(self, member):
        member_type = getattr(member, "member_type", None)
        size = 1
        if isinstance(member_type, A.VectorType):
            size = int(getattr(member_type, "size", 0) or 0)
            member_type = member_type.element_type
        scalar = str(getattr(member_type, "name", "") or "")
        if scalar not in _GRID_SCALARS or not 1 <= size <= 3:
            self.refuse(
                "dispatch-grid",
                f"reads its grid from record member '{member.name}', which must be a uint, uvec2 or uvec3 "
                "(or a 16-bit form)",
                member,
            )

    def check_thread_group(self, launch):
        if launch != "thread":
            return
        for name in _THREAD_GROUP_ATTRIBUTES:
            attribute = self.attribute(name)
            if attribute is not None and _literal_ints(attribute) not in (
                (1, 1, 1),
                (1,),
            ):
                self.refuse(
                    "thread-group",
                    "is a thread launch node, whose thread group must be (1, 1, 1)",
                    attribute,
                )

    def check_cross_group_sharing(self):
        records = {
            parameter.name: parameter.param_type
            for parameter in getattr(self.entry, "parameters", None) or []
            if _type_name(getattr(parameter, "param_type", None))
            == "RWDispatchNodeInputRecord"
        }
        if not records:
            return
        for node in self.entry.walk():
            if not isinstance(node, A.FunctionCallNode):
                continue
            callee = getattr(node, "function", None) or getattr(node, "name", None)
            if not isinstance(callee, A.MemberAccessNode):
                continue
            if getattr(callee, "member", None) != "FinishedCrossGroupSharing":
                continue
            receiver = getattr(callee, "object", None)
            record_type = records.get(getattr(receiver, "name", None))
            record = (
                self.record_struct(record_type) if record_type is not None else None
            )
            if record is not None and TRACK_RW_INPUT_SHARING not in _attribute_names(
                record
            ):
                self.refuse(
                    "cross-group-sharing",
                    f"calls FinishedCrossGroupSharing() on '{receiver.name}', whose record structure "
                    f"'{record.name}' is not marked @{TRACK_RW_INPUT_SHARING}",
                    node,
                )


def check_work_graph_nodes(program, target) -> None:
    """Raise :class:`WorkGraphNodeError` for the first rule of D3D12 work graphs a node stage of ``program`` breaks."""
    if not isinstance(program, A.ShaderNode):
        return
    for stage in _node_stages(program):
        _NodeChecker(program, stage, str(target or "")).check()
