"""GLSL-style mesh and task stages rewritten into the entry-parameter form for HLSL and Metal targets.

The GLSL importer, and CrossGL written the GLSL way, declares a mesh stage's outputs as module-scope arrays and
its launch shape as stage layouts (GL_EXT_mesh_shader; GLSL 4.60 with that extension, sections 4.3.4, 4.3.6 and
7.1.8)::

    Payload payload @ taskPayloadSharedEXT;
    out vec3 color @ location(0)[];
    perprimitive out vec4 tint @ location(1)[];
    mesh {
        layout(local_size_x = 32) in;
        layout(triangles, max_vertices = 64, max_primitives = 126) out;
        void main() {
            SetMeshOutputCounts(3, 1);
            gl_MeshVerticesEXT[i].gl_Position = p; color[i] = c; tint[0] = t;
            gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
        }
    }

HLSL (shader model 6.5 mesh shaders) and MSL (``mesh<V, P, NV, NP, topology>``) have no module-scope outputs. The
CrossGL the HLSL and Metal importers write, which the DirectX and Metal emitters lower, passes the outputs as entry
parameters instead::

    mesh {
        @outputtopology("triangle") @numthreads(32, 1, 1)
        void main(@mesh_payload in Payload payload, @vertices out MeshVertex meshVertices[64],
                  @primitives out MeshPrimitive meshPrimitives[126], @indices out uvec3 meshIndices[126]) { ... }
    }

:func:`raise_glsl_mesh_stages` rewrites a GLSL-style mesh stage into that form:

* ``layout(points|lines|triangles, max_vertices = V, max_primitives = P) out`` becomes ``@outputtopology`` and the
  lengths of the output arrays, and ``layout(local_size_x/y/z) in`` becomes ``@numthreads``, each size 1 by default;
* ``gl_MeshVerticesEXT[i].gl_Position``, each per-vertex ``out`` array and each member of a per-vertex ``out`` block
  become members of a ``MeshVertex`` structure; ``name[i]`` and ``block[i].member`` read and write
  ``meshVertices[i].member``. A plain output keeps its location; a block member takes the block's location plus the
  locations of the members before it, unless it carries its own (GLSL 4.60 section 4.4.2.1). Interpolation
  qualifiers are kept;
* ``perprimitiveEXT`` outputs and ``gl_MeshPrimitivesEXT[i].gl_PrimitiveID``, ``gl_Layer`` and
  ``gl_ViewportIndex`` become members of a ``MeshPrimitive`` structure. HLSL and MSL declare those three built-ins
  ``uint`` and GLSL ``int``, so a write converts its value and a read converts back;
* ``gl_PrimitivePointIndicesEXT``, ``gl_PrimitiveLineIndicesEXT`` or ``gl_PrimitiveTriangleIndicesEXT`` becomes the
  ``@indices`` array of the topology's index type;
* the ``taskPayloadSharedEXT`` variable becomes the ``@mesh_payload in`` parameter, read-only in both languages.

A task stage's ``taskPayloadSharedEXT`` variable becomes ``groupshared`` storage that each ``DispatchMesh(x, y, z)``
passes as its payload, which is how HLSL amplification shaders and the Metal emitter's object functions take it.

Built-ins that the HLSL or Metal output does not carry yet (``gl_PointSize``, clip and cull distances,
``gl_CullPrimitiveEXT``, ``gl_PrimitiveShadingRateEXT``), other uses of the output arrays, helpers that use the
mesh interface, a payload shared by a task and a mesh stage, and layouts with no HLSL or Metal form are refused
with :class:`GlslMeshStageUnsupportedError`.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .rewrite import set_child
from .stage_varyings import (
    INTERPOLATION,
    FreshNames,
    explicit_location,
    int_literal,
    location_span,
    slots,
)
from .stage_varyings import stage_name as _stage_name
from .stage_varyings import stages as _stages

__all__ = ["GlslMeshStageUnsupportedError", "raise_glsl_mesh_stages"]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_TARGETS = _HLSL_TARGETS | {"metal"}
_TASK_STAGES = frozenset({"task", "amplification", "object"})
_MESH_ROLES = frozenset({"vertices", "indices", "primitives"})
_TASK_PAYLOAD = frozenset({"taskpayloadsharedext", "taskpayloadsharednv"})
_PER_PRIMITIVE = frozenset({"perprimitive", "perprimitiveext", "perprimitivenv"})
_TOPOLOGIES = {
    "points": ("point", "gl_PrimitivePointIndicesEXT", 1),
    "lines": ("line", "gl_PrimitiveLineIndicesEXT", 2),
    "triangles": ("triangle", "gl_PrimitiveTriangleIndicesEXT", 3),
}
_INDEX_ARRAYS = frozenset(name for _t, name, _n in _TOPOLOGIES.values())
_VERTICES, _PRIMITIVES = "gl_MeshVerticesEXT", "gl_MeshPrimitivesEXT"
# Built-in members carried: member name, type and whether HLSL/MSL declare it uint where GLSL declares it int.
_VERTEX_BUILTINS = {"gl_Position": ("position", "vec4", False)}
_PRIMITIVE_BUILTINS = {
    "gl_PrimitiveID": ("primitiveID", "uint", True),
    "gl_Layer": ("layer", "uint", True),
    "gl_ViewportIndex": ("viewportIndex", "uint", True),
}
_BUILTIN_BLOCKS = frozenset({"gl_MeshPerVertexEXT", "gl_MeshPerPrimitiveEXT"})
_BUILTIN_ARRAYS = _INDEX_ARRAYS | {_VERTICES, _PRIMITIVES}
# GLSL qualifies an output ``out``; the SPIR-V importer marks it ``@output``.
_OUTPUT = frozenset({"out", "output"})
_DROPPED_ATTRIBUTES = frozenset(
    {"input", "output", "in", "out", "glsl_interface_block"}
)
_LOCAL_SIZES = ("local_size_x", "local_size_y", "local_size_z")
# Compute built-in inputs a mesh or task entry receives as parameters: preferred name and type. HLSL has no system
# value for the workgroup count.
_COMPUTE_BUILTINS = {
    "gl_GlobalInvocationID": ("globalInvocationID", 3),
    "gl_LocalInvocationID": ("localInvocationID", 3),
    "gl_WorkGroupID": ("workGroupID", 3),
    "gl_LocalInvocationIndex": ("localInvocationIndex", 1),
    "gl_NumWorkGroups": ("numWorkGroups", 3),
}
_WORKGROUP_SIZE = "gl_WorkGroupSize"


class GlslMeshStageUnsupportedError(TranslationDiagnostic):
    """A GLSL-style mesh or task stage construct the HLSL and Metal entry-parameter form cannot carry yet."""

    def __init__(self, target, detail, *, reason):
        self.reason_code = reason
        self.missing_capabilities = (f"{target}.glsl-mesh-stage.{reason}",)
        super().__init__(
            f"{detail}; {_label(target)} mesh and task shaders take their outputs "
            "and payload as entry parameters",
            target=target,
            code=f"{CODE_PREFIX}glsl-mesh-stage-unsupported",
        )


def raise_glsl_mesh_stages(program, target) -> int:
    """Rewrite the GLSL-style mesh and task stages of ``program`` for HLSL or Metal; returns how many changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in _TARGETS:
        return 0
    payload_users = _payload_users(program)
    changed = 0
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if entry is None or not isinstance(getattr(entry, "body", None), A.BlockNode):
            continue
        name = _stage_name(stage)
        if name == "mesh" and not any(
            _names(parameter) & _MESH_ROLES for parameter in entry.parameters or []
        ):
            _MeshRaising(program, stage, target, payload_users).run()
            changed += 1
        elif name in _TASK_STAGES and _raise_task_payload(
            program, stage, target, payload_users
        ):
            changed += 1
        if name == "mesh" or name in _TASK_STAGES:
            changed += _pass_compute_builtins(program, stage, target)
    return changed


def _pass_compute_builtins(program, stage, target):
    """Pass the compute built-in inputs a mesh or task entry reads by name as parameters carrying them."""
    entry = stage.entry_point
    body = entry.body
    declared = {
        node.name
        for _p, _f, _k, node in slots(body)
        if isinstance(node, A.VariableNode)
    } | {parameter.name for parameter in entry.parameters or []}
    used = (_identifier_names(body) - declared) & (
        set(_COMPUTE_BUILTINS) | {_WORKGROUP_SIZE}
    )
    if not used:
        return 0
    names = {_WORKGROUP_SIZE} | set(_COMPUTE_BUILTINS)
    for function in _stage_functions(program, stage)[1:]:
        found = sorted(_identifier_names(getattr(function, "body", None)) & names)
        if found:
            _refuse(
                target,
                f"helper '{function.name}' reads {found[0]}, which the {_stage_name(stage)} entry receives as "
                "a parameter",
                "helper-builtin",
            )
    if "gl_NumWorkGroups" in used and target in _HLSL_TARGETS:
        _refuse(
            target,
            f"the {_stage_name(stage)} stage reads gl_NumWorkGroups, for which HLSL has no system value",
            "num-work-groups",
        )
    fresh = FreshNames(program)
    replacements = {}
    for builtin in sorted(used & set(_COMPUTE_BUILTINS)):
        existing = next(
            (
                parameter.name
                for parameter in entry.parameters or []
                if builtin in {str(a.name) for a in parameter.attributes or []}
            ),
            None,
        )
        if existing is None:
            preferred, components = _COMPUTE_BUILTINS[builtin]
            existing = fresh(preferred)
            entry.parameters = list(entry.parameters or []) + [
                A.ParameterNode(
                    existing,
                    _index_type(components),
                    attributes=[A.AttributeNode(builtin)],
                )
            ]
        replacements[builtin] = existing
    sizes = _attribute_arguments(entry, "numthreads") or _layout_sizes(stage, target)
    for parent, field, key, node in list(slots(body)):
        if not isinstance(node, A.IdentifierNode) or node.name not in used:
            continue
        if node.name == _WORKGROUP_SIZE:
            set_child(
                parent,
                field,
                key,
                A.FunctionCallNode(
                    A.IdentifierNode("uvec3"), [copy.deepcopy(size) for size in sizes]
                ),
            )
        else:
            set_child(parent, field, key, A.IdentifierNode(replacements[node.name]))
    return 1


def _layout_sizes(stage, target):
    sizes = {}
    for layout in getattr(stage, "layout_qualifiers", None) or []:
        if str(getattr(layout, "direction", "") or "").lower() != "in":
            continue
        for entry in getattr(layout, "entries", None) or []:
            name = str(getattr(entry, "name", ""))
            if name in _LOCAL_SIZES:
                sizes[name] = _layout_value(target, name, entry)
    return [sizes.get(name, int_literal(1)) for name in _LOCAL_SIZES]


def _label(target):
    return "Metal" if target == "metal" else "HLSL"


def _refuse(target, detail, reason):
    raise GlslMeshStageUnsupportedError(target, detail, reason=reason)


def _names(node):
    """Lower-case names of a declaration's attributes and qualifiers."""
    names = {
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    }
    names.update(
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", None) or []
    )
    return names


def _identifier_names(root):
    return {
        node.name
        for _p, _f, _k, node in slots(root)
        if isinstance(node, A.IdentifierNode)
    }


def _stage_functions(program, stage):
    """The entry point of ``stage`` and every helper it could call."""
    functions = [stage.entry_point]
    for function in [
        *(getattr(stage, "local_functions", None) or []),
        *(getattr(program, "functions", None) or []),
    ]:
        if isinstance(function, A.FunctionNode) and function is not stage.entry_point:
            functions.append(function)
    return functions


def _payload_variables(program, stage):
    return [
        variable
        for variable in [
            *(getattr(program, "global_variables", None) or []),
            *(getattr(stage, "local_variables", None) or []),
        ]
        if isinstance(variable, A.VariableNode) and _names(variable) & _TASK_PAYLOAD
    ]


def _payload_users(program):
    """``id(payload variable) -> names of the stages whose functions use it``."""
    users = {}
    for stage in _stages(program):
        if getattr(stage, "entry_point", None) is None:
            continue
        used = set()
        for function in _stage_functions(program, stage):
            if getattr(function, "body", None) is not None:
                used |= _identifier_names(function.body)
        for variable in _payload_variables(program, stage):
            if variable.name in used:
                users.setdefault(id(variable), set()).add(_stage_name(stage))
    return users


def _single_payload(program, stage, target, payload_users):
    variables = _payload_variables(program, stage)
    if len(variables) > 1:
        _refuse(
            target,
            f"the {_stage_name(stage)} stage declares {len(variables)} taskPayloadSharedEXT variables",
            "payload-count",
        )
    if not variables:
        return None
    payload = variables[0]
    users = payload_users.get(id(payload), set())
    if len(users) > 1:
        _refuse(
            target,
            f"taskPayloadSharedEXT variable '{payload.name}' is used by the {' and '.join(sorted(users))} stages",
            "shared-payload",
        )
    return payload


def _raise_task_payload(program, stage, target, payload_users):
    payload = _single_payload(program, stage, target, payload_users)
    if payload is None:
        return False
    payload.attributes = [
        attribute
        for attribute in payload.attributes or []
        if str(getattr(attribute, "name", "")).lower() not in _TASK_PAYLOAD
    ]
    payload.qualifiers = [
        qualifier
        for qualifier in payload.qualifiers or []
        if str(qualifier).lower() not in _TASK_PAYLOAD
    ] + ["groupshared"]
    for function in _stage_functions(program, stage):
        for _p, _f, _k, node in slots(getattr(function, "body", None)):
            if (
                isinstance(node, A.MeshOpNode)
                and node.operation == "DispatchMesh"
                and len(node.arguments) == 3
            ):
                node.arguments.append(A.IdentifierNode(payload.name))
    return True


def _layout_value(target, name, layout_entry):
    arguments = list(getattr(layout_entry, "arguments", None) or [])
    if len(arguments) != 1:
        _refuse(target, f"mesh layout '{name}' needs one value", "layout-value")
    return arguments[0]


def _attribute_arguments(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) == name:
            return list(getattr(attribute, "arguments", None) or [])
    return None


def _member(name, type_node, attributes):
    return A.StructMemberNode(name, type_node, attributes=attributes)


def _carried_attributes(node, location):
    """``node``'s location and interpolation attributes, with ``location`` when it has none of its own."""
    attributes = []
    for attribute in getattr(node, "attributes", None) or []:
        lowered = str(getattr(attribute, "name", "")).lower()
        if lowered in _DROPPED_ATTRIBUTES or lowered in _PER_PRIMITIVE:
            continue
        if lowered.startswith("glsl_interface"):
            continue
        attributes.append(
            A.AttributeNode(attribute.name, list(attribute.arguments or []))
        )
    present = {str(attribute.name).lower() for attribute in attributes}
    attributes += [
        A.AttributeNode(str(qualifier).lower())
        for qualifier in getattr(node, "qualifiers", None) or []
        if str(qualifier).lower() in INTERPOLATION
        and str(qualifier).lower() not in present
    ]
    if location is not None and "location" not in present:
        attributes.insert(0, A.AttributeNode("location", [int_literal(location)]))
    return attributes


def _named_type(type_node):
    """The name of a named (struct) type, or ``None``."""
    return (
        getattr(type_node, "name", None) if isinstance(type_node, A.NamedType) else None
    )


def _builtin_type(type_name):
    if type_name == "vec4":
        return A.VectorType(A.PrimitiveType("float"), 4)
    return A.PrimitiveType(type_name)


def _index_type(components):
    if components == 1:
        return A.PrimitiveType("uint")
    return A.VectorType(A.PrimitiveType("uint"), components)


def _convert(type_name, value):
    return A.FunctionCallNode(A.IdentifierNode(type_name), [value])


def _builtin_members(body):
    """``{built-in output array: names of the members the body uses}`` for ``gl_MeshVerticesEXT[i].member``."""
    used = {}
    for _p, _f, _k, node in slots(body):
        if (
            isinstance(node, A.MemberAccessNode)
            and isinstance(node.object_expr, A.ArrayAccessNode)
            and isinstance(node.object_expr.array_expr, A.IdentifierNode)
            and node.object_expr.array_expr.name in (_VERTICES, _PRIMITIVES)
        ):
            used.setdefault(node.object_expr.array_expr.name, set()).add(node.member)
    return used


def _element(name, index, member=None):
    access = A.ArrayAccessNode(A.IdentifierNode(name), index)
    return access if member is None else A.MemberAccessNode(access, member)


class _MeshRaising:
    def __init__(self, program, stage, target, payload_users):
        self.program = program
        self.stage = stage
        self.entry = stage.entry_point
        self.target = target
        self.payload_users = payload_users
        self.fresh = FreshNames(program)

    def refuse(self, detail, reason):
        _refuse(self.target, detail, reason)

    def run(self):
        topology, components, max_vertices, max_primitives, sizes = self.layouts()
        payload = _single_payload(
            self.program, self.stage, self.target, self.payload_users
        )
        vertex_outputs, primitive_outputs, blocks, builtin_outputs = self.interface()
        used = _builtin_members(self.entry.body)
        vertex_struct, primitive_struct = self.fresh("MeshVertex"), self.fresh(
            "MeshPrimitive"
        )
        vertices, primitives = self.fresh("meshVertices"), self.fresh("meshPrimitives")
        indices = self.fresh("meshIndices")
        vertex_members, vertex_nodes = self.structure(
            vertex_outputs,
            [block for block in blocks if not _names(block) & _PER_PRIMITIVE],
            _VERTEX_BUILTINS,
            used,
            _VERTICES,
        )
        primitive_members, primitive_nodes = self.structure(
            primitive_outputs,
            [block for block in blocks if _names(block) & _PER_PRIMITIVE],
            _PRIMITIVE_BUILTINS,
            used,
            _PRIMITIVES,
        )
        index_name = _TOPOLOGIES[topology][1]
        self.refuse_helpers(
            {
                _VERTICES,
                _PRIMITIVES,
                *_INDEX_ARRAYS,
                *vertex_members,
                *primitive_members,
                *([payload.name] if payload is not None else []),
            }
        )
        rewriter = _BodyRewriter(
            self,
            {
                **{
                    name: (vertices, members)
                    for name, members in vertex_members.items()
                },
                **{
                    name: (primitives, members)
                    for name, members in primitive_members.items()
                },
            },
            index_name,
            indices,
        )
        rewriter.rewrite(self.entry.body)

        parameters = []
        if payload is not None:
            parameters.append(
                A.ParameterNode(
                    payload.name,
                    copy.deepcopy(payload.var_type),
                    qualifiers=["in"],
                    attributes=[A.AttributeNode("mesh_payload")],
                )
            )
        parameters.append(
            self.output_parameter(vertices, vertex_struct, max_vertices, "vertices")
        )
        if primitive_nodes:
            parameters.append(
                self.output_parameter(
                    primitives, primitive_struct, max_primitives, "primitives"
                )
            )
        parameters.append(
            A.ParameterNode(
                indices,
                A.ArrayType(_index_type(components), copy.deepcopy(max_primitives)),
                qualifiers=["out"],
                attributes=[A.AttributeNode("indices")],
            )
        )
        self.entry.parameters = list(self.entry.parameters or []) + parameters
        self.entry.attributes = list(getattr(self.entry, "attributes", None) or []) + [
            A.AttributeNode(
                "outputtopology",
                [A.LiteralNode(_TOPOLOGIES[topology][0], A.PrimitiveType("string"))],
            ),
            A.AttributeNode("numthreads", sizes),
        ]
        self.program.structs = list(getattr(self.program, "structs", None) or []) + [
            A.StructNode(vertex_struct, vertex_nodes)
        ]
        if primitive_nodes:
            self.program.structs.append(A.StructNode(primitive_struct, primitive_nodes))
        self.stage.layout_qualifiers = []
        self.drop_declarations(
            [
                *vertex_outputs,
                *primitive_outputs,
                *builtin_outputs,
                *([payload] if payload else []),
            ],
            blocks,
        )

    def layouts(self):
        topology = max_vertices = max_primitives = None
        sizes = {}
        for layout in getattr(self.stage, "layout_qualifiers", None) or []:
            direction = str(getattr(layout, "direction", "") or "").lower()
            for entry in getattr(layout, "entries", None) or []:
                name = str(getattr(entry, "name", ""))
                if direction == "in" and name in _LOCAL_SIZES:
                    sizes[name] = _layout_value(self.target, name, entry)
                elif direction == "out" and name in _TOPOLOGIES:
                    topology = name
                elif direction == "out" and name == "max_vertices":
                    max_vertices = _layout_value(self.target, name, entry)
                elif direction == "out" and name == "max_primitives":
                    max_primitives = _layout_value(self.target, name, entry)
                else:
                    self.refuse(
                        f"mesh layout '{name}' on the {direction or 'stage'} interface has no "
                        f"{_label(self.target)} form",
                        "layout-entry",
                    )
        if topology is None or max_vertices is None or max_primitives is None:
            self.refuse(
                "the mesh stage declares no output primitive, max_vertices and max_primitives layout",
                "output-layout",
            )
        numthreads = [
            copy.deepcopy(sizes[name]) if name in sizes else int_literal(1)
            for name in _LOCAL_SIZES
        ]
        return (
            topology,
            _TOPOLOGIES[topology][2],
            max_vertices,
            max_primitives,
            numthreads,
        )

    def interface(self):
        """The stage's per-vertex and per-primitive output variables and its ``out`` blocks."""
        vertex_outputs, primitive_outputs, builtin_outputs = [], [], []
        for variable in [
            *(getattr(self.program, "global_variables", None) or []),
            *(getattr(self.stage, "local_variables", None) or []),
        ]:
            if not isinstance(variable, A.VariableNode) or not (
                _names(variable) & _OUTPUT
            ):
                continue
            if variable.name in _BUILTIN_ARRAYS:
                # The SPIR-V importer declares the index array it writes; it is the @indices parameter.
                builtin_outputs.append(variable)
                continue
            if not isinstance(variable.var_type, A.ArrayType):
                self.refuse(
                    f"mesh output '{variable.name}' is not an array", "output-array"
                )
            if _names(variable) & _PER_PRIMITIVE:
                primitive_outputs.append(variable)
            else:
                vertex_outputs.append(variable)
        blocks = []
        for struct in getattr(self.program, "structs", None) or []:
            arguments = _attribute_arguments(struct, "glsl_interface_block")
            if arguments is None:
                continue
            if struct.name in _BUILTIN_BLOCKS:
                blocks.append(struct)
                continue
            direction = {str(getattr(a, "name", a)).lower() for a in arguments}
            instance = _attribute_arguments(struct, "glsl_interface_instance")
            if "out" not in direction:
                continue
            if (
                not instance
                or _attribute_arguments(struct, "glsl_interface_array") is None
            ):
                self.refuse(
                    f"mesh output block '{struct.name}' is not an arrayed block with an instance name",
                    "output-block",
                )
            blocks.append(struct)
        return vertex_outputs, primitive_outputs, blocks, builtin_outputs

    def structure(self, outputs, blocks, builtins, used, builtin_array):
        """``{interface name: member map}`` and the struct members for one of the two output structures.

        A plain output maps to its member name; a block instance or the built-in array maps each of its members.
        """
        # The built-in array is always part of the interface, so a use of a member it does not carry is refused.
        members, nodes, taken = {builtin_array: {}}, [], set()
        for builtin, (preferred, type_name, _uint) in builtins.items():
            # Every mesh vertex has a position (gl_MeshPerVertexEXT), which HLSL and MSL require be declared.
            if builtin in used.get(builtin_array, ()) or builtin == "gl_Position":
                member = self.taken(preferred, taken)
                members[builtin_array][builtin] = member
                nodes.append(
                    _member(
                        member,
                        _builtin_type(type_name),
                        [A.AttributeNode(builtin)],
                    )
                )
        structs = {
            struct.name: struct
            for struct in getattr(self.program, "structs", None) or []
        }
        flattened = []
        for variable in outputs:
            struct = structs.get(_named_type(variable.var_type.element_type))
            if struct is not None:
                # A SPIR-V Block output array, which GLSL spells as an arrayed ``out`` block.
                flattened.append((variable.name, struct, variable))
                continue
            member = self.taken(variable.name, taken)
            members[variable.name] = member
            nodes.append(
                _member(
                    member,
                    copy.deepcopy(variable.var_type.element_type),
                    _carried_attributes(variable, None),
                )
            )
        for block in blocks:
            if block.name in _BUILTIN_BLOCKS:
                continue
            instance = str(
                getattr(
                    _attribute_arguments(block, "glsl_interface_instance")[0],
                    "name",
                    "",
                )
            )
            flattened.append((instance, block, block))
        for instance, struct, located in flattened:
            members[instance] = self.flatten(struct, located, nodes, taken)
        return members, nodes

    def flatten(self, struct, located, nodes, taken):
        """Members of an output block or struct-typed output array, as ``{member: interface member}``.

        Each member takes its own location, else the next one after the previous member's, starting from the
        location of ``located`` (the block or the output variable).
        """
        block_location = explicit_location(located)
        mapped = {}
        for block_member in struct.members or []:
            if (
                _names(block_member) & _PER_PRIMITIVE
                and not _names(located) & _PER_PRIMITIVE
            ):
                self.refuse(
                    f"mesh output '{struct.name}' mixes per-vertex and per-primitive members",
                    "output-block",
                )
            own = explicit_location(block_member)
            location = own if own is not None else block_location
            member = self.taken(block_member.name, taken)
            mapped[block_member.name] = member
            nodes.append(
                _member(
                    member,
                    copy.deepcopy(block_member.member_type),
                    _carried_attributes(block_member, location),
                )
            )
            if location is not None:
                block_location = location + location_span(block_member.member_type)
        return mapped

    def taken(self, preferred, taken):
        name, index = preferred, 1
        while name in taken:
            index += 1
            name = f"{preferred}{index}"
        taken.add(name)
        return name

    def refuse_helpers(self, names):
        for function in _stage_functions(self.program, self.stage)[1:]:
            body = getattr(function, "body", None)
            if body is None:
                continue
            found = sorted(_identifier_names(body) & names)
            if found:
                self.refuse(
                    f"helper '{function.name}' uses the mesh interface variable '{found[0]}'",
                    "helper-interface",
                )

    def output_parameter(self, name, struct_name, count, role):
        return A.ParameterNode(
            name,
            A.ArrayType(A.NamedType(struct_name), copy.deepcopy(count)),
            qualifiers=["out"],
            attributes=[A.AttributeNode(role)],
        )

    def drop_declarations(self, variables, blocks):
        dropped = {id(variable) for variable in variables}
        self.program.global_variables = [
            variable
            for variable in getattr(self.program, "global_variables", None) or []
            if id(variable) not in dropped
        ]
        self.stage.local_variables = [
            variable
            for variable in getattr(self.stage, "local_variables", None) or []
            if id(variable) not in dropped
        ]
        block_ids = {id(block) for block in blocks}
        self.program.structs = [
            struct for struct in self.program.structs if id(struct) not in block_ids
        ]


class _BodyRewriter:
    """Rewrites the mesh entry's uses of its output variables into element accesses of the output parameters."""

    def __init__(self, raising, interface, index_name, indices):
        self.raising = raising
        self.interface = interface
        self.index_name = index_name
        self.indices = indices
        # (parameter, member) of the built-ins HLSL and MSL declare uint where GLSL declares int.
        parameter, members = interface.get(_PRIMITIVES, (None, {}))
        self.uint_members = {
            (parameter, members[builtin])
            for builtin, (_preferred, _type, is_uint) in _PRIMITIVE_BUILTINS.items()
            if is_uint and builtin in members
        }
        self.conversions = []

    def refuse(self, detail, reason):
        self.raising.refuse(detail, reason)

    def rewrite(self, body):
        found = list(slots(body))
        parents = {id(node): (parent, field, key) for parent, field, key, node in found}
        names = set(self.interface) | _INDEX_ARRAYS
        for parent, _field, _key, node in reversed(found):
            if not isinstance(node, A.IdentifierNode) or node.name not in names:
                continue
            if not (
                isinstance(parent, A.ArrayAccessNode) and parent.array_expr is node
            ):
                self.refuse(
                    f"the mesh stage uses '{node.name}' other than by indexing it",
                    "output-array-use",
                )
            self.element(node.name, parent, parents)
        for assignment in self.conversions:
            assignment.value = _convert("uint", assignment.value)

    def element(self, name, access, parents):
        index = access.index_expr
        if name in _INDEX_ARRAYS:
            if name != self.index_name:
                self.refuse(
                    f"the mesh stage writes {name}, which its output topology does not use",
                    "index-array",
                )
            self.written(name, access, parents)
            set_child(*parents[id(access)], _element(self.indices, index))
            return
        parameter, members = self.interface[name]
        if not isinstance(members, dict):
            self.written(name, access, parents)
            set_child(*parents[id(access)], _element(parameter, index, members))
            return
        outer = parents[id(access)][0]
        if not (isinstance(outer, A.MemberAccessNode) and outer.object_expr is access):
            self.refuse(
                f"the mesh stage uses '{name}[i]' other than through one of its members",
                "output-block-use",
            )
        member = members.get(outer.member)
        if member is None:
            self.refuse(
                f"the mesh stage uses {name}[].{outer.member}, which the "
                f"{_label(self.raising.target)} output does not carry yet",
                "output-builtin",
            )
        assignment = self.written(f"{name}[].{outer.member}", outer, parents)
        if (parameter, member) in self.uint_members:
            if assignment.target is not outer:
                self.refuse(
                    f"the mesh stage writes part of {name}[].{outer.member}",
                    "builtin-update",
                )
            self.conversions.append(assignment)
        set_child(*parents[id(outer)], _element(parameter, index, member))

    def written(self, name, node, parents):
        """The assignment that writes ``node`` as a whole or through a member, element or swizzle."""
        top = node
        while True:
            parent = parents[id(top)][0]
            if (
                (isinstance(parent, A.MemberAccessNode) and parent.object_expr is top)
                or (isinstance(parent, A.ArrayAccessNode) and parent.array_expr is top)
                or (isinstance(parent, A.SwizzleNode) and parent.vector_expr is top)
            ):
                top = parent
                continue
            break
        if not (
            isinstance(parent, A.AssignmentNode)
            and parent.target is top
            and parent.operator == "="
        ):
            # HLSL and MSL mesh outputs are write-only: DXC rejects a read and MSL's mesh object has no getter.
            self.refuse(f"the mesh stage reads its output {name}", "output-read")
        if top is not node and self.raising.target == "metal":
            self.refuse(
                f"the mesh stage writes part of its output {name}; Metal's mesh object sets a vertex or "
                "primitive member as a whole",
                "output-component-write",
            )
        return parent
