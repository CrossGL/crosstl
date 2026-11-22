"""Import HLSL work-graph node shaders as CrossGL ``node`` stages.

A function marked ``[Shader("node")]`` is one node of a D3D12 work graph (DirectX-Specs, "D3D12 Work Graphs").
CrossGL spells the node's object types, methods and intrinsics as HLSL does
(:mod:`crosstl.translator.semantics.work_graph_nodes`), so the importer keeps those and renames only the
attributes, which CrossGL writes in lower case with underscores: ``[NodeLaunch("coalescing")]`` becomes
``@node_launch("coalescing")``, ``[MaxRecords(4)]`` on a parameter becomes ``@max_records(4)``, the
``globallycoherent`` storage class of a read-write input record becomes ``@globallycoherent``, and
``[NodeTrackRWInputSharing]`` on a record structure becomes ``@node_track_rw_input_sharing``. HLSL attribute names
are case-insensitive, so ``[numthreads]`` and ``[NumThreads]`` both become ``@numthreads``.
"""

from __future__ import annotations

from .DirectxAst import AttributeNode

__all__ = [
    "NODE_STAGE",
    "crossgl_node_entry_attributes",
    "crossgl_work_graph_parameter_attributes",
    "crossgl_work_graph_record_attribute",
]

NODE_STAGE = "node"
# HLSL node entry attributes, by lower-case name, and their CrossGL names.
_ENTRY_ATTRIBUTES = {
    "nodelaunch": "node_launch",
    "nodedispatchgrid": "node_dispatch_grid",
    "nodemaxdispatchgrid": "node_max_dispatch_grid",
    "nodemaxrecursiondepth": "node_max_recursion_depth",
    "nodeid": "node_id",
    "nodeshareinputof": "node_share_input_of",
    "nodeisprogramentry": "node_is_program_entry",
    "nodelocalrootargumentstableindex": "node_local_root_arguments_table_index",
    "numthreads": "numthreads",
}
# HLSL attributes of node input and output parameters.
_PARAMETER_ATTRIBUTES = {
    "maxrecords": "max_records",
    "maxrecordssharedwith": "max_records_shared_with",
    "nodeid": "node_id",
    "nodearraysize": "node_array_size",
    "unboundedsparsenodes": "unbounded_sparse_nodes",
    "allowsparsenodes": "allow_sparse_nodes",
}
_RECORD_ATTRIBUTE = ("nodetrackrwinputsharing", "node_track_rw_input_sharing")
_GLOBALLY_COHERENT = "globallycoherent"
# The node record and output object types (the HLSL spelling CrossGL keeps).
_NODE_OBJECT_TYPES = frozenset(
    {
        "DispatchNodeInputRecord",
        "RWDispatchNodeInputRecord",
        "GroupNodeInputRecords",
        "RWGroupNodeInputRecords",
        "ThreadNodeInputRecord",
        "RWThreadNodeInputRecord",
        "EmptyNodeInput",
        "NodeOutput",
        "NodeOutputArray",
        "EmptyNodeOutput",
        "EmptyNodeOutputArray",
    }
)


def _lower(attribute) -> str:
    return str(getattr(attribute, "name", "") or "").lower()


def _arguments(attribute):
    return list(getattr(attribute, "args", None) or [])


def crossgl_node_entry_attributes(attributes):
    """The attributes of a node entry with the work-graph ones renamed to CrossGL; the others are kept as they are."""
    renamed = []
    for attribute in attributes or []:
        name = _ENTRY_ATTRIBUTES.get(_lower(attribute))
        renamed.append(
            AttributeNode(name, _arguments(attribute))
            if name is not None
            else attribute
        )
    return renamed


def _render(codegen, name, arguments):
    if not arguments:
        return f"@{name}"
    rendered = ", ".join(
        codegen.generate_expression(argument) for argument in arguments
    )
    return f"@{name}({rendered})"


def _type_base_name(vtype) -> str:
    return str(vtype or "").split("<", 1)[0].strip()


def crossgl_work_graph_parameter_attributes(codegen, parameter):
    """The CrossGL attributes of a node record or output parameter, written before it, or ``""``."""
    if _type_base_name(getattr(parameter, "vtype", None)) not in _NODE_OBJECT_TYPES:
        return ""
    rendered = []
    for attribute in getattr(parameter, "attributes", None) or []:
        name = _PARAMETER_ATTRIBUTES.get(_lower(attribute))
        if name is not None:
            rendered.append(_render(codegen, name, _arguments(attribute)))
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", None) or []}
    if _GLOBALLY_COHERENT in qualifiers:
        rendered.append(f"@{_GLOBALLY_COHERENT}")
    return " ".join(rendered)


def crossgl_work_graph_record_attribute(struct, indent="    "):
    """``@node_track_rw_input_sharing`` before a record structure HLSL marks ``[NodeTrackRWInputSharing]``."""
    for attribute in getattr(struct, "attributes", None) or []:
        if _lower(attribute) == _RECORD_ATTRIBUTE[0]:
            return f"{indent}@{_RECORD_ATTRIBUTE[1]}\n"
    return ""
