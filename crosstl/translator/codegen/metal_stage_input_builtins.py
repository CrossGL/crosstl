"""Built-in members of Metal stage-input structures, passed as entry-point parameters.

HLSL, WGSL and CrossGL let a vertex shader's input structure carry system values next to its vertex attributes
(``uint id : SV_VertexID;``, ``@builtin(vertex_index) id: u32``, ``uint id @ gl_VertexID;``). Metal accepts
``[[vertex_id]]``, ``[[instance_id]]``, ``[[base_vertex]]``, ``[[base_instance]]``, ``[[amplification_id]]`` and
``[[amplification_count]]`` only on entry-point parameters (Metal Shading Language Specification, section 5.2.3.1,
"Vertex Function Input Attributes", and section 5.2.3.4 for fragment functions), and a vertex function's
``[[stage_in]]`` structure may hold only ``[[attribute(n)]]`` members. The emitter wrote the member's attribute into
the structure, which the Metal compiler rejects ("'vertex_id' attribute only applies to parameters").

:func:`split_metal_stage_input_builtins` rewrites each vertex or fragment entry point that takes an input structure
with such members:

- the other members are copied, attributes included, into a new structure ``<Struct>_StageIn`` that the entry point
  takes in the parameter's place; when no member is left, there is no structure parameter;
- each parameter-only built-in member becomes an entry-point parameter of the member's type and attribute;
- the body starts by declaring a local of the original structure under the parameter's name and filling every
  member from those parameters, so the body, and the helpers it passes the structure to, are unchanged.

Every structure member loses a parameter-only built-in attribute, which Metal rejects on any structure member; the
structure itself stays, since the body and helpers use it as a plain value. The input AST is not modified.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    BlockNode,
    ExpressionStatementNode,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    StructNode,
    VariableNode,
)
from .stage_utils import collect_stage_entry_records

__all__ = ["PARAMETER_ONLY_BUILTINS", "split_metal_stage_input_builtins"]

# Metal attributes, as ``canonical_metal_semantic`` spells them, that the stage's functions take only as parameters.
PARAMETER_ONLY_BUILTINS = {
    "vertex": frozenset(
        {
            "vertex_id",
            "instance_id",
            "base_vertex",
            "base_instance",
            "amplification_id",
            "amplification_count",
        }
    ),
    "fragment": frozenset({"amplification_id", "amplification_count"}),
}
_ANY_STAGE = frozenset().union(*PARAMETER_ONLY_BUILTINS.values())
_OUTPUT_QUALIFIERS = frozenset({"out", "inout"})


def _iter_nodes(root):
    """Yield every AST node reachable from ``root`` once, in a deterministic order."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if not hasattr(node, "__dict__") or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(
            value
            for key, value in reversed(list(vars(node).items()))
            if key not in {"parent", "annotations"}
        )


def _struct_containers(ast):
    """``(list, struct)`` for every structure declared at program or stage scope, in declaration order."""
    containers = [getattr(ast, "structs", None)]
    for stage in (getattr(ast, "stages", None) or {}).values():
        containers.append(getattr(stage, "local_structs", None))
    for container in containers:
        for struct in container or []:
            if isinstance(struct, StructNode) and getattr(struct, "name", None):
                yield container, struct


def _input_struct(parameter, structs):
    """The structure a stage-input parameter takes by value, else None."""
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", None) or []}
    if qualifiers & _OUTPUT_QUALIFIERS:
        return None
    param_type = getattr(parameter, "param_type", None)
    if not isinstance(param_type, NamedType) or param_type.generic_args:
        return None
    return structs.get(param_type.name)


class _Splitter:
    def __init__(self, ast, codegen):
        self.ast = ast
        self.semantic_of = codegen.semantic_from_node
        self.canonical = codegen.canonical_metal_semantic
        self.entry_stage_types = set(codegen.stage_entry_types())
        self.structs = {}
        self.containers = {}
        for container, struct in _struct_containers(ast):
            self.structs.setdefault(struct.name, struct)
            self.containers.setdefault(struct.name, container)
        self.stage_in_structs = {}
        self.program_names = None

    def builtin(self, member):
        semantic = self.semantic_of(member)
        if semantic is None:
            return None
        return self.canonical(semantic)

    def split_members(self, struct, stage):
        parameter_only = PARAMETER_ONLY_BUILTINS[stage]
        return [
            member
            for member in getattr(struct, "members", None) or []
            if self.builtin(member) in parameter_only
        ]

    def entries(self):
        for _id, stage, function in collect_stage_entry_records(
            self.ast, None, set(PARAMETER_ONLY_BUILTINS)
        ):
            if stage in PARAMETER_ONLY_BUILTINS:
                yield stage, function

    def needs_rewrite(self):
        if any(
            self.builtin(member) in _ANY_STAGE
            for _container, struct in _struct_containers(self.ast)
            for member in getattr(struct, "members", None) or []
        ):
            return True
        return any(
            (struct := _input_struct(parameter, self.structs)) is not None
            and self.split_members(struct, stage)
            for stage, function in self.entries()
            for parameter in getattr(function, "parameters", None) or []
        )

    def unique(self, name, used):
        candidate, suffix = name, 1
        while candidate in used:
            suffix += 1
            candidate = f"{name}{suffix}"
        used.add(candidate)
        return candidate

    def stage_in_struct(self, struct, kept):
        """The ``[[stage_in]]`` copy of ``struct`` holding the ``kept`` members, created once per structure."""
        name = self.stage_in_structs.get(struct.name)
        if name is not None:
            return name
        if self.program_names is None:
            self.program_names = {
                node.name
                for node in _iter_nodes(self.ast)
                if isinstance(getattr(node, "name", None), str)
            }
        name = self.unique(f"{struct.name}_StageIn", self.program_names)
        copy = StructNode(
            name,
            [deepcopy(member) for member in kept],
            attributes=deepcopy(getattr(struct, "attributes", None) or []),
        )
        container = self.containers[struct.name]
        container.insert(container.index(struct) + 1, copy)
        self.stage_in_structs[struct.name] = name
        return name

    def split_parameter(self, function, index, struct, split):
        parameter = function.parameters[index]
        used = {
            node.name
            for node in _iter_nodes(function)
            if isinstance(getattr(node, "name", None), str)
        }
        used.update(
            str(node.identifier)
            for node in _iter_nodes(function)
            if isinstance(node, IdentifierNode)
        )
        split_ids = {id(member) for member in split}
        kept = [member for member in struct.members if id(member) not in split_ids]
        replacements = []
        sources = {}
        if kept:
            stage_in_name = self.unique(f"{parameter.name}_stage_in", used)
            replacements.append(
                ParameterNode(
                    stage_in_name,
                    NamedType(self.stage_in_struct(struct, kept)),
                    attributes=deepcopy(parameter.attributes),
                    qualifiers=list(parameter.qualifiers),
                )
            )
            for member in kept:
                sources[id(member)] = MemberAccessNode(
                    IdentifierNode(stage_in_name), member.name
                )
        for member in split:
            builtin_name = self.unique(f"{parameter.name}_{member.name}", used)
            replacements.append(
                ParameterNode(
                    builtin_name,
                    deepcopy(member.member_type),
                    attributes=deepcopy(member.attributes),
                )
            )
            sources[id(member)] = IdentifierNode(builtin_name)
        function.parameters[index : index + 1] = replacements
        prologue = [VariableNode(parameter.name, NamedType(struct.name))]
        for member in struct.members:
            prologue.append(
                ExpressionStatementNode(
                    AssignmentNode(
                        MemberAccessNode(IdentifierNode(parameter.name), member.name),
                        sources[id(member)],
                    )
                )
            )
        body = function.body
        if isinstance(body, BlockNode):
            body.statements[0:0] = prologue
        elif isinstance(body, list):
            body[0:0] = prologue
        else:
            function.body = BlockNode(prologue)
        return len(replacements)

    def split_function(self, stage, function):
        index = 0
        while index < len(function.parameters or []):
            struct = _input_struct(function.parameters[index], self.structs)
            split = self.split_members(struct, stage) if struct is not None else []
            if not split:
                index += 1
                continue
            index += self.split_parameter(function, index, struct, split)

    def interface_struct_names(self):
        """Structures an entry point still takes or returns once the parameters are split."""
        names = set()
        for _id, _stage, function in collect_stage_entry_records(
            self.ast, None, self.entry_stage_types
        ):
            types = [getattr(function, "return_type", None)]
            types.extend(
                getattr(parameter, "param_type", None)
                for parameter in getattr(function, "parameters", None) or []
            )
            names.update(t.name for t in types if isinstance(t, NamedType))
        return names

    def remove_semantic(self, member, semantic):
        attributes = list(getattr(member, "attributes", None) or [])
        for position, attribute in enumerate(attributes):
            if getattr(attribute, "name", None) == semantic:
                del attributes[position]
                member.attributes = attributes
                return True
        return False

    def strip_member_attributes(self):
        """Drop parameter-only built-ins from every member, and every semantic from a split structure.

        A split structure that no entry point takes or returns any more is a plain value, whose members' stage
        attributes Metal rejects or ignores (``[[attribute(n)]]`` on an array member does not compile).
        """
        plain = set(self.stage_in_structs) - self.interface_struct_names()
        for _container, struct in _struct_containers(self.ast):
            for member in getattr(struct, "members", None) or []:
                while (semantic := self.semantic_of(member)) is not None:
                    if (
                        struct.name not in plain
                        and self.canonical(semantic) not in _ANY_STAGE
                    ):
                        break
                    if not self.remove_semantic(member, semantic):
                        break


def split_metal_stage_input_builtins(ast, codegen):
    """Return ``ast`` with parameter-only built-in members of stage-input structures passed as parameters.

    ``codegen`` is the Metal emitter, whose ``semantic_from_node``, ``canonical_metal_semantic`` and
    ``stage_entry_types`` name members' attributes and entry points. Returns ``ast`` itself when nothing changes.
    """
    if not _Splitter(ast, codegen).needs_rewrite():
        return ast
    ast = deepcopy(ast)
    splitter = _Splitter(ast, codegen)
    seen = set()
    for stage, function in list(splitter.entries()):
        if id(function) in seen:
            continue
        seen.add(id(function))
        splitter.split_function(stage, function)
    splitter.strip_member_attributes()
    return ast
