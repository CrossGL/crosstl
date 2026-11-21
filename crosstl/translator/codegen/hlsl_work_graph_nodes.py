"""HLSL for CrossGL work-graph nodes: ``[Shader("node")]`` entries of a shader model 6.8 library.

A CrossGL ``node { ... }`` stage (:mod:`crosstl.translator.semantics.work_graph_nodes`) is one node shader of a
D3D12 work graph. HLSL spells its object types, their methods, ``GetRemainingRecursionLevels`` and ``Barrier`` as
CrossGL does, so this module only writes the attributes CrossGL spells in its lower-case form (DirectX-Specs,
"D3D12 Work Graphs", "Shader target" and "Node input declaration"):

* entry attributes ``[NodeLaunch("mode")]``, ``[NodeDispatchGrid(x, y, z)]``, ``[NodeMaxDispatchGrid(x, y, z)]``,
  ``[NodeMaxRecursionDepth(n)]``, ``[NodeID("name", i)]``, ``[NodeShareInputOf("name", i)]``,
  ``[NodeIsProgramEntry]``, ``[NodeLocalRootArgumentsTableIndex(n)]`` and ``[NumThreads(x, y, z)]``;
* record parameter attributes ``[MaxRecords(n)]``, ``[MaxRecordsSharedWith(parameter)]``, ``[NodeID(...)]``,
  ``[NodeArraySize(n)]``, ``[UnboundedSparseNodes]`` and ``[AllowSparseNodes]``, and the ``globallycoherent``
  storage class on a read-write input record;
* ``[NodeTrackRWInputSharing]`` on a record structure.

DXC compiles such a library with ``-T lib_6_8`` or later. The checks DXC applies to the node itself run before
generation, in :func:`~crosstl.translator.semantics.work_graph_nodes.check_work_graph_nodes`; an attribute this
module cannot write, such as a node identifier that is not a string, is refused with
:class:`HlslWorkGraphNodeError` rather than dropped.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.work_graph_nodes import (
    OUTPUT_TYPES,
    TRACK_RW_INPUT_SHARING,
    WORK_GRAPH_TYPES,
    register_work_graph_target,
)

__all__ = [
    "HLSL_NODE_STAGE",
    "HlslWorkGraphNodeError",
    "HlslWorkGraphNodeMixin",
    "is_hlsl_work_graph_attribute",
    "pin_hlsl_node_names",
]

# The generator's target names. Registering them lets node stages through the shared passes to this generator.
register_work_graph_target("directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12")

HLSL_NODE_STAGE = "node"
# CrossGL entry attributes and their HLSL spelling, in the order HLSL writes them. A ``str`` argument is quoted.
_ENTRY_ATTRIBUTES = {
    "node_launch": "NodeLaunch",
    "node_is_program_entry": "NodeIsProgramEntry",
    "node_id": "NodeID",
    "node_local_root_arguments_table_index": "NodeLocalRootArgumentsTableIndex",
    "node_share_input_of": "NodeShareInputOf",
    "node_dispatch_grid": "NodeDispatchGrid",
    "node_max_dispatch_grid": "NodeMaxDispatchGrid",
    "node_max_recursion_depth": "NodeMaxRecursionDepth",
}
_PARAMETER_ATTRIBUTES = {
    "max_records": "MaxRecords",
    "max_records_shared_with": "MaxRecordsSharedWith",
    "node_id": "NodeID",
    "node_array_size": "NodeArraySize",
    "unbounded_sparse_nodes": "UnboundedSparseNodes",
    "allow_sparse_nodes": "AllowSparseNodes",
}
_GLOBALLY_COHERENT = "globallycoherent"
# Attributes whose first argument is a node name, which HLSL writes as a string literal.
_NAMED_ATTRIBUTES = frozenset({"node_id", "node_share_input_of"})
# Attributes whose argument names another parameter of the entry.
_PARAMETER_REFERENCE_ATTRIBUTES = frozenset({"max_records_shared_with"})
_LAUNCH_MODES = frozenset({"broadcasting", "coalescing", "thread"})
# The thread-position system values a node body may read by their CrossGL names, the parameter each becomes, and
# the launch modes that provide it: broadcasting nodes have all four, coalescing nodes only the thread's place in
# its group and thread launch nodes none (D3D12 Work Graphs, "Node shader system values"; DXC rejects any other
# with "Invalid system value semantic ... for launchtype").
_NODE_SYSTEM_VALUES = (
    ("gl_GlobalInvocationID", "dispatchThreadID", "uint3", "SV_DispatchThreadID"),
    ("gl_LocalInvocationID", "groupThreadID", "uint3", "SV_GroupThreadID"),
    ("gl_WorkGroupID", "groupID", "uint3", "SV_GroupID"),
    ("gl_LocalInvocationIndex", "groupIndex", "uint", "SV_GroupIndex"),
)
_LAUNCH_SYSTEM_VALUES = {
    "broadcasting": frozenset(value[0] for value in _NODE_SYSTEM_VALUES),
    "coalescing": frozenset({"gl_LocalInvocationID", "gl_LocalInvocationIndex"}),
    "thread": frozenset(),
}
_DEFAULT_LAUNCH = "broadcasting"
_OUTPUT_TYPES = frozenset(OUTPUT_TYPES)
_NODE_NAME_ANNOTATION = "hlsl_node_name"
# Every CrossGL attribute this module writes, so that none of them is read as an HLSL semantic.
_WORK_GRAPH_ATTRIBUTES = frozenset(
    {
        *_ENTRY_ATTRIBUTES,
        *_PARAMETER_ATTRIBUTES,
        _GLOBALLY_COHERENT,
        TRACK_RW_INPUT_SHARING,
    }
)


class HlslWorkGraphNodeError(TranslationDiagnostic):
    """A work-graph node attribute HLSL cannot spell as written, or a system value the node's launch lacks."""

    def __init__(
        self, message, *, rule="attribute", target="directx", source_location=None
    ):
        self.source_location = source_location
        self.rule = rule
        self.missing_capabilities = (f"hlsl.work-graph.{rule}",)
        super().__init__(
            f"HLSL work-graph node: {message}",
            target=target,
            code=f"{CODE_PREFIX}directx-work-graph-{rule}",
        )


def _name(attribute) -> str:
    return str(getattr(attribute, "name", "") or "").lower()


def is_hlsl_work_graph_attribute(attribute) -> bool:
    """Whether ``attribute`` is a work-graph attribute, which HLSL writes as an attribute and never as a semantic."""
    return _name(attribute) in _WORK_GRAPH_ATTRIBUTES


def _written(argument) -> str:
    """An attribute argument as a diagnostic shows it: its literal value or name, else its node kind."""
    for field in ("value", "name"):
        text = getattr(argument, field, None)
        if isinstance(text, (str, int, float)) and not isinstance(text, bool):
            return repr(text) if isinstance(text, str) else str(text)
    return (
        repr(argument)
        if isinstance(argument, (str, int, float))
        else type(argument).__name__
    )


def _arguments(attribute):
    return list(
        getattr(attribute, "arguments", None) or getattr(attribute, "args", None) or []
    )


def _type_base_name(type_node) -> str:
    name = getattr(type_node, "name", None)
    if isinstance(name, str):
        return name
    return str(type_node or "").split("<", 1)[0].strip()


def is_work_graph_type(type_node) -> bool:
    return _type_base_name(type_node) in WORK_GRAPH_TYPES


def _pin(node) -> None:
    if (
        hasattr(node, "add_annotation")
        and node.get_annotation(_NODE_NAME_ANNOTATION) is None
    ):
        node.add_annotation(_NODE_NAME_ANNOTATION, getattr(node, "name", None))


def _pinned_name(node):
    get_annotation = getattr(node, "get_annotation", None)
    pinned = get_annotation(_NODE_NAME_ANNOTATION) if get_annotation else None
    return pinned or getattr(node, "name", None)


def pin_hlsl_node_names(program) -> None:
    """Record the CrossGL names of node entries and their parameters before the generator renames any.

    A node is named after its function and an output after its parameter unless ``@node_id`` names them, and
    ``@max_records_shared_with`` names a parameter. The generator renames a declaration whose name HLSL reserves;
    the recorded names keep the node IDs and the parameter references as CrossGL wrote them.
    """
    stages = getattr(program, "stages", None)
    if not isinstance(stages, dict):
        return
    for stage_type, stage in stages.items():
        if str(getattr(stage_type, "value", stage_type)).lower() != HLSL_NODE_STAGE:
            continue
        entry = getattr(stage, "entry_point", None)
        if entry is None:
            continue
        _pin(entry)
        for parameter in getattr(entry, "parameters", None) or []:
            _pin(parameter)


class HlslWorkGraphNodeMixin:
    """Write the attributes of node entries, their record parameters and record structures."""

    def hlsl_node_attribute_word(self, argument, attribute, *, quoted, siblings=None):
        """One attribute argument as HLSL text: a quoted node name, a parameter name or an integer constant."""
        if isinstance(argument, str):
            text = argument
        else:
            text = getattr(argument, "value", None)
            if not isinstance(text, (str, int)) or isinstance(text, bool):
                text = getattr(argument, "name", None)
        if quoted:
            if not isinstance(text, str):
                raise HlslWorkGraphNodeError(
                    f"'@{_name(attribute)}' names a node with {_written(argument)}, not a string"
                )
            name = text.strip('"')
            if not name or '"' in name or "\\" in name:
                raise HlslWorkGraphNodeError(
                    f"'@{_name(attribute)}' names a node '{name}' that HLSL cannot write as a string"
                )
            return f'"{name}"'
        if _name(attribute) in _PARAMETER_REFERENCE_ATTRIBUTES:
            referenced = str(text or "").strip('"')
            for sibling in siblings or ():
                if referenced in (
                    _pinned_name(sibling),
                    getattr(sibling, "name", None),
                ):
                    return self.hlsl_declaration_identifier_name(sibling.name)
            raise HlslWorkGraphNodeError(
                f"'@{_name(attribute)}({referenced})' names no parameter of the node"
            )
        value = self.hlsl_constant_int_value(argument)
        if value is None:
            raise HlslWorkGraphNodeError(
                f"'@{_name(attribute)}' takes integer constants, and {_written(argument)} is not one"
            )
        return str(value)

    def hlsl_node_attribute(self, attribute, spelling, siblings=None):
        name = _name(attribute)
        arguments = _arguments(attribute)
        words = [
            self.hlsl_node_attribute_word(
                argument,
                attribute,
                quoted=index == 0 and name in _NAMED_ATTRIBUTES,
                siblings=siblings,
            )
            for index, argument in enumerate(arguments)
        ]
        return f"[{spelling}({', '.join(words)})]" if words else f"[{spelling}]"

    def hlsl_node_launch(self, func) -> str:
        for attribute in getattr(func, "attributes", None) or []:
            if _name(attribute) == "node_launch":
                arguments = _arguments(attribute)
                if len(arguments) != 1:
                    break
                argument = arguments[0]
                mode = getattr(argument, "value", None) or getattr(
                    argument, "name", argument
                )
                mode = str(mode).strip('"').lower()
                if mode in _LAUNCH_MODES:
                    return mode
                break
        else:
            return _DEFAULT_LAUNCH
        raise HlslWorkGraphNodeError(
            "'@node_launch' takes one of broadcasting, coalescing and thread"
        )

    def generate_hlsl_entry_attributes(
        self, func, stage, stage_node, execution_config, entry_name
    ):
        """The stage attribute lines of entry ``func``: a node's (below), or any other stage's."""
        if stage != HLSL_NODE_STAGE:
            return self.generate_hlsl_stage_attributes(
                func, stage, stage_node=stage_node
            )
        return self.generate_hlsl_node_attributes(
            func,
            execution_config,
            self.hlsl_function_declaration_name(func, entry_name),
        )

    def generate_hlsl_node_attributes(self, func, execution_config, declared_name):
        """The attribute lines of node entry ``func``: launch first, then the node's identity, grid and group.

        A node without ``@node_id`` is named after its function. When HLSL declares the function under another
        name, ``[NodeID]`` keeps the CrossGL name, which the other nodes' outputs address.
        """
        lines = [f'[NodeLaunch("{self.hlsl_node_launch(func)}")]']
        attributes = {
            _name(attribute): attribute
            for attribute in getattr(func, "attributes", None) or []
        }
        node_name = _pinned_name(func)
        if "node_id" not in attributes and node_name and declared_name != node_name:
            lines.append(f'[NodeID("{node_name}")]')
        for name, spelling in _ENTRY_ATTRIBUTES.items():
            if name == "node_launch" or name not in attributes:
                continue
            lines.append(self.hlsl_node_attribute(attributes[name], spelling))
        lines.append(self.hlsl_node_numthreads(func, execution_config))
        return "".join(f"{line}\n" for line in lines)

    def hlsl_node_numthreads(self, func, execution_config):
        arguments = self.hlsl_stage_attribute_arguments(func, "numthreads")
        if arguments:
            if len(arguments) != 3:
                raise HlslWorkGraphNodeError(
                    "'@numthreads' on a node takes three dimensions"
                )
            values = [self.hlsl_constant_int_value(argument) for argument in arguments]
            if any(value is None for value in values):
                raise HlslWorkGraphNodeError(
                    "'@numthreads' on a node takes integer constants"
                )
            return f"[NumThreads({values[0]}, {values[1]}, {values[2]})]"
        numthreads = self.generate_compute_numthreads(execution_config).strip()
        return numthreads.replace("[numthreads(", "[NumThreads(", 1)

    def append_hlsl_node_builtin_parameters(self, func, params_str, param_names):
        """``params_str`` with a parameter for each thread-position value node entry ``func`` reads."""
        for (
            builtin,
            name,
            param_type,
            semantic,
        ) in self.required_hlsl_node_builtin_parameters(func):
            self.current_identifier_aliases[builtin] = name
            params_str = self.append_hlsl_parameter_declaration(
                params_str, f"{param_type} {name} : {semantic}"
            )
            param_names.add(name)
            self.local_variable_types[name] = param_type
        return params_str

    def required_hlsl_node_builtin_parameters(self, func):
        """``(builtin, parameter, type, semantic)`` for each thread-position value node entry ``func`` reads.

        A value the node's launch mode does not provide is refused: DXC rejects its semantic on the entry.
        """
        used = self.used_hlsl_compute_builtin_names(getattr(func, "body", []))
        launch = self.hlsl_node_launch(func)
        reserved = set(self.local_variable_types)
        reserved.update(self.global_variable_types)
        reserved.update(self.current_identifier_reserved_names)
        reserved.update(self.current_identifier_aliases.values())
        parameters = []
        for builtin, base_name, value_type, semantic in _NODE_SYSTEM_VALUES:
            if builtin not in used:
                continue
            if builtin not in _LAUNCH_SYSTEM_VALUES[launch]:
                raise HlslWorkGraphNodeError(
                    f"node '{getattr(func, 'name', 'main')}' reads {builtin}, which a {launch} node does not have "
                    f"(HLSL {semantic})",
                    rule="system-value",
                )
            name = self.hlsl_unique_local_identifier(base_name, reserved)
            reserved.add(name)
            parameters.append((builtin, name, value_type, semantic))
        return parameters

    def hlsl_work_graph_parameter_declaration(
        self, parameter, param_type, stage, siblings=()
    ):
        """The declaration of a node record or output parameter, or ``None`` for any other parameter.

        Only a node entry's parameters carry the record attributes; a helper function that takes a record object
        receives it as a plain parameter. ``siblings`` are the entry's parameters, which
        ``@max_records_shared_with`` names.
        """
        if not is_work_graph_type(getattr(parameter, "param_type", None)):
            return None
        declared_name = self.hlsl_declaration_identifier_name(parameter.name)
        declaration = f"{param_type} {declared_name}"
        attributes = [
            attribute
            for attribute in getattr(parameter, "attributes", None) or []
            if is_hlsl_work_graph_attribute(attribute)
        ]
        if stage != HLSL_NODE_STAGE:
            if attributes:
                raise HlslWorkGraphNodeError(
                    f"parameter '{parameter.name}' has '@{_name(attributes[0])}', which only a node entry's "
                    "parameters take"
                )
            return declaration
        prefix = []
        output_name = _pinned_name(parameter)
        if (
            _type_base_name(parameter.param_type) in _OUTPUT_TYPES
            and not any(_name(attribute) == "node_id" for attribute in attributes)
            and output_name != declared_name
        ):
            # An output without @node_id addresses the node its parameter is named after.
            prefix.append(f'[NodeID("{output_name}")]')
        coherent = False
        for attribute in attributes:
            name = _name(attribute)
            if name == _GLOBALLY_COHERENT:
                coherent = True
                continue
            spelling = _PARAMETER_ATTRIBUTES.get(name)
            if spelling is None:
                raise HlslWorkGraphNodeError(
                    f"parameter '{parameter.name}' has '@{name}', which is not a record parameter attribute"
                )
            prefix.append(self.hlsl_node_attribute(attribute, spelling, siblings))
        if coherent:
            declaration = f"{_GLOBALLY_COHERENT} {declaration}"
        return " ".join([*prefix, declaration])

    def hlsl_work_graph_record_prefix(self, struct):
        """``[NodeTrackRWInputSharing] `` for a record structure marked to track it, and otherwise nothing."""
        for attribute in getattr(struct, "attributes", None) or []:
            if _name(attribute) == TRACK_RW_INPUT_SHARING:
                return "[NodeTrackRWInputSharing] "
        return ""
