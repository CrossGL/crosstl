"""GLSL-style geometry stages rewritten into HLSL's stream form for HLSL targets.

The GLSL importer, and CrossGL written the GLSL way, declares a geometry stage's primitives as stage layouts and its
interface as per-vertex input arrays and stage outputs, and emits a vertex by writing the outputs and calling
``EmitVertex()`` (GLSL 4.60 sections 4.3.4, 4.3.6, 4.4.1.2, 4.4.2.2 and 8.16)::

    in vec2 uv @location(0)[];
    out vec2 outUv @location(0);
    geometry {
        layout(triangles, invocations = 2) in;
        layout(triangle_strip, max_vertices = 3) out;
        void main() { gl_Position = gl_in[0].gl_Position; outUv = uv[0]; EmitVertex(); EndPrimitive(); }
    }

HLSL has no stage-level interface variables. A geometry shader receives its primitive as a topology-qualified array
parameter and appends output vertices to a stream object (HLSL shader model 4 geometry shader object,
``[maxvertexcount(N)]``, ``[instance(N)]``). The DirectX emitter lowers that form, which the HLSL importer writes; it
printed the GLSL form unchanged, with undeclared ``gl_Position`` and ``EmitVertex``. :func:`raise_glsl_geometry_streams`
rewrites a GLSL-style geometry stage into it:

* ``layout(points|lines|lines_adjacency|triangles|triangles_adjacency) in`` becomes the input array's
  ``point|line|lineadj|triangle|triangleadj`` qualifier and length, ``invocations = N`` becomes ``@instance(N)``,
  and ``layout(points|line_strip|triangle_strip, max_vertices = N) out`` becomes a ``PointStream``, ``LineStream``
  or ``TriangleStream`` parameter and ``@maxvertexcount(N)``;
* each per-vertex input array becomes a member of a ``GeometryInput`` structure, and ``gl_in[i].gl_Position`` its
  ``gl_Position`` member: ``name[i]`` reads ``input[i].name``, ``name.length()`` is the vertex count, and any other
  use of the array reads a local copy;
* each output, and each of ``gl_Position``, ``gl_Layer``, ``gl_ViewportIndex`` and ``gl_PrimitiveID`` the stage
  writes, becomes a member of a ``GeometryOutput`` structure held in a local ``output``. ``EmitVertex()`` appends
  it to the stream and ``EndPrimitive()`` restarts the strip. GLSL leaves the outputs undefined after
  ``EmitVertex()``, so keeping their values is a valid refinement. HLSL declares the three integer built-ins
  ``uint``, so a write converts its ``int`` value and a read converts back;
* ``gl_PrimitiveIDIn`` and ``gl_InvocationID`` read ``uint`` entry parameters carrying ``SV_PrimitiveID`` and
  ``SV_GSInstanceID``, converted to GLSL's ``int``;
* helpers that use the interface, directly or through the helpers they call, take the input array, the output
  structure and the stream as extra parameters.

Members keep their location, built-in and interpolation attributes, so a varying takes the semantic the neighbouring
stages give the same location. Several output streams (``EmitStreamVertex``, ``layout(stream = N)``),
``gl_PointSize``, clip and cull distances, interface blocks and helpers that another stage also calls are refused
with :class:`HlslGeometryStreamUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name
from .rewrite import get_child, set_child
from .stage_varyings import (
    FreshNames,
    function_local_names,
    int_literal,
    literal_int,
    rewrite_array_reads,
    slots,
)
from .stage_varyings import stage_name as _stage_name
from .stage_varyings import stages as _stages

__all__ = ["HlslGeometryStreamUnsupportedError", "raise_glsl_geometry_streams"]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_INPUT_TOPOLOGIES = {
    "points": ("point", 1),
    "lines": ("line", 2),
    "lines_adjacency": ("lineadj", 4),
    "triangles": ("triangle", 3),
    "triangles_adjacency": ("triangleadj", 6),
}
_OUTPUT_TOPOLOGIES = {
    "points": "PointStream",
    "line_strip": "LineStream",
    "triangle_strip": "TriangleStream",
}
# Built-in outputs: member name and HLSL type. HLSL declares the integer ones uint (SV_RenderTargetArrayIndex,
# SV_ViewportArrayIndex and SV_PrimitiveID), GLSL int.
_OUTPUT_BUILTINS = {
    "gl_Position": ("position", "vec4"),
    "gl_Layer": ("layer", "uint"),
    "gl_ViewportIndex": ("viewportIndex", "uint"),
    "gl_PrimitiveID": ("primitiveID", "uint"),
}
# Built-in inputs read from entry parameters: the CrossGL semantic of the parameter and its preferred name.
_PARAMETER_BUILTINS = {
    "gl_PrimitiveIDIn": ("gl_PrimitiveID", "primitiveID"),
    "gl_InvocationID": ("gl_InvocationID", "invocationID"),
}
_GL_IN_MEMBERS = {"gl_Position": "position"}
_REFUSED_BUILTINS = {
    "gl_PointSize": "point-size",
    "gl_ClipDistance": "clip-distance",
    "gl_CullDistance": "cull-distance",
}
_STREAM_METHODS = {"EmitVertex": "Append", "EndPrimitive": "RestartStrip"}
_MULTI_STREAM_CALLS = frozenset({"EmitStreamVertex", "EndStreamPrimitive"})
_INTERPOLATION = frozenset({"flat", "noperspective", "centroid", "sample", "smooth"})
_DROPPED_ATTRIBUTES = frozenset({"input", "output", "in", "out"})
_PER_VERTEX_BLOCK_MEMBERS = frozenset(
    {"gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_CullDistance"}
)


class HlslGeometryStreamUnsupportedError(TranslationDiagnostic):
    """A GLSL-style geometry stage construct HLSL's stream form cannot carry yet."""

    def __init__(self, target, detail, *, reason):
        self.reason_code = reason
        self.missing_capabilities = (f"{target}.geometry-stream.{reason}",)
        super().__init__(
            f"{detail}; HLSL geometry shaders take their primitive as an entry parameter and append vertices "
            "to a stream object",
            target=target,
            code=f"{CODE_PREFIX}hlsl-geometry-stream-unsupported",
        )


def raise_glsl_geometry_streams(program, target) -> int:
    """Rewrite GLSL-style geometry stages of ``program`` for HLSL; returns how many stages changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in _HLSL_TARGETS:
        return 0
    changed = 0
    for stage in _stages(program):
        if _stage_name(stage) != "geometry":
            continue
        entry = getattr(stage, "entry_point", None)
        if entry is None or getattr(entry, "parameters", None):
            continue
        if not isinstance(getattr(entry, "body", None), A.BlockNode):
            continue
        _GeometryRaising(program, stage, target).run()
        changed += 1
    return changed


class _GeometryRaising:
    def __init__(self, program, stage, target):
        self.program = program
        self.stage = stage
        self.entry = stage.entry_point
        self.target = target
        self.fresh = FreshNames(program)
        functions = list(getattr(program, "functions", None) or []) + list(
            getattr(stage, "local_functions", None) or []
        )
        self.functions = {
            function.name: function
            for function in functions
            if isinstance(function, A.FunctionNode)
        }

    def refuse(self, detail, reason):
        raise HlslGeometryStreamUnsupportedError(self.target, detail, reason=reason)

    def run(self):
        entry = self.entry
        if not _is_void(getattr(entry, "return_type", None)):
            self.refuse(
                f"geometry entry point '{entry.name}' returns a value", "entry-return"
            )
        topology, count, invocations, stream_kind, max_vertices = self.layouts()
        reachable = self.reachable([entry])
        helpers = [function for function in reachable if function is not entry]
        inputs, outputs = self.interface(reachable)
        used = set()
        for function in reachable:
            used |= _identifier_names(function.body)
        self.refuse_unsupported(reachable, used)

        input_struct = self.fresh("GeometryInput")
        output_struct = self.fresh("GeometryOutput")
        input_name, stream_name = self.fresh("input"), self.fresh("stream")
        output_name = self.fresh("output")
        input_members, input_struct_node = self.input_structure(
            input_struct, inputs, used
        )
        output_members, output_struct_node = self.output_structure(
            output_struct, outputs, used
        )
        self.program.structs = list(getattr(self.program, "structs", None) or []) + [
            input_struct_node,
            output_struct_node,
        ]

        builtin_parameters = {}
        for builtin in sorted(set(_PARAMETER_BUILTINS) & used):
            semantic, preferred = _PARAMETER_BUILTINS[builtin]
            builtin_parameters[builtin] = (self.fresh(preferred), semantic)

        input_type = A.ArrayType(A.NamedType(input_struct), int_literal(count))
        stream_type = A.NamedType(stream_kind, [A.NamedType(output_struct)])
        tokens = [
            (input_name, input_type, [], "input"),
            (output_name, A.NamedType(output_struct), ["inout"], "output"),
            (stream_name, stream_type, ["inout"], "stream"),
        ] + [
            (name, A.PrimitiveType("uint"), [], builtin)
            for builtin, (name, _semantic) in builtin_parameters.items()
        ]
        needs = self.needs(reachable, inputs, output_members, builtin_parameters)
        self.refuse_shared_helpers([helper for helper in helpers if needs[helper.name]])

        for function in reachable:
            self.rewrite_function(
                function,
                count,
                inputs,
                input_name,
                input_members,
                output_name,
                output_members,
                stream_name,
                builtin_parameters,
            )
        for helper in helpers:
            helper.parameters = list(helper.parameters or []) + [
                A.ParameterNode(name, type_node, qualifiers=list(qualifiers))
                for name, type_node, qualifiers, key in tokens
                if key in needs[helper.name]
            ]
        self.thread_arguments(reachable, needs, tokens)

        entry.parameters = [
            A.ParameterNode(input_name, input_type, qualifiers=[topology]),
            *(
                A.ParameterNode(
                    name,
                    A.PrimitiveType("uint"),
                    attributes=[A.AttributeNode(semantic)],
                )
                for name, semantic in builtin_parameters.values()
            ),
            A.ParameterNode(stream_name, stream_type, qualifiers=["inout"]),
        ]
        entry.attributes = list(getattr(entry, "attributes", None) or [])
        if invocations is not None:
            entry.attributes.append(
                A.AttributeNode("instance", [int_literal(invocations)])
            )
        entry.attributes.append(
            A.AttributeNode("maxvertexcount", [int_literal(max_vertices)])
        )
        entry.body.statements[0:0] = [
            A.VariableNode(name=output_name, var_type=A.NamedType(output_struct))
        ]
        self.stage.layout_qualifiers = []
        self.drop_interface_declarations(inputs, outputs)

    def layouts(self):
        topology = stream_kind = max_vertices = invocations = None
        count = None
        for layout in getattr(self.stage, "layout_qualifiers", None) or []:
            direction = str(getattr(layout, "direction", "") or "").lower()
            for entry in getattr(layout, "entries", None) or []:
                name = str(getattr(entry, "name", ""))
                arguments = list(getattr(entry, "arguments", None) or [])
                if direction == "in" and name in _INPUT_TOPOLOGIES:
                    topology, count = _INPUT_TOPOLOGIES[name]
                elif direction == "in" and name == "invocations":
                    invocations = self.layout_int(name, arguments)
                elif direction == "out" and name in _OUTPUT_TOPOLOGIES:
                    stream_kind = _OUTPUT_TOPOLOGIES[name]
                elif direction == "out" and name == "max_vertices":
                    max_vertices = self.layout_int(name, arguments)
                elif direction == "out" and name == "stream":
                    if self.layout_int(name, arguments) != 0:
                        self.refuse(
                            "the geometry stage writes vertex stream "
                            f"{self.layout_int(name, arguments)}",
                            "multiple-streams",
                        )
                else:
                    self.refuse(
                        f"geometry layout '{name}' on the {direction or 'stage'} interface has no HLSL form",
                        "layout-entry",
                    )
        if topology is None:
            self.refuse(
                "the geometry stage declares no input primitive layout",
                "input-primitive",
            )
        if stream_kind is None or max_vertices is None:
            self.refuse(
                "the geometry stage declares no output primitive and max_vertices layout",
                "output-primitive",
            )
        return topology, count, invocations, stream_kind, max_vertices

    def layout_int(self, name, arguments):
        value = literal_int(arguments[0]) if len(arguments) == 1 else None
        if value is None or value < 0:
            self.refuse(
                f"geometry layout '{name}' needs a constant integer", "layout-value"
            )
        return value

    def reachable(self, roots):
        found, pending = [], list(roots)
        while pending:
            function = pending.pop(0)
            if any(function is seen for seen in found):
                continue
            found.append(function)
            for _parent, _field, _key, node in slots(function.body):
                if isinstance(node, A.FunctionCallNode):
                    callee = self.functions.get(call_name(node))
                    if callee is not None and callee is not self.entry:
                        pending.append(callee)
        return found

    def other_stage_functions(self):
        roots = [
            stage.entry_point
            for stage in _stages(self.program)
            if stage is not self.stage
            and getattr(stage, "entry_point", None) is not None
        ]
        return self.reachable(roots) if roots else []

    def interface(self, reachable):
        """The stage's per-vertex inputs and outputs, as ``{name: variable}``, in declaration order."""
        used = set()
        for function in reachable:
            used |= _identifier_names(function.body)
        others = set()
        for function in self.other_stage_functions():
            others |= _identifier_names(function.body)
        inputs, outputs = {}, {}
        local = list(getattr(self.stage, "local_variables", None) or [])
        shared = list(getattr(self.program, "global_variables", None) or [])
        single_stage = len(_stages(self.program)) == 1
        for variable in local + shared:
            direction = _direction(variable)
            if direction is None:
                continue
            is_shared = any(variable is other for other in shared)
            # An unused varying still occupies its slot of the signature the neighbouring stages link with, so
            # every one the stage declares is kept; a shader-level one belongs to this stage when no other
            # stage could own it.
            if variable.name not in used and is_shared and not single_stage:
                continue
            if is_shared and variable.name in others:
                self.refuse(
                    f"interface variable '{variable.name}' is used by the geometry stage and another stage",
                    "shared-interface-variable",
                )
            (inputs if direction == "input" else outputs)[variable.name] = variable
        for function in reachable:
            shadowed = sorted(
                (function_local_names(function) | _parameter_names(function))
                & (set(inputs) | set(outputs))
            )
            if shadowed:
                self.refuse(
                    f"function '{function.name}' declares '{shadowed[0]}', which is also a geometry "
                    "interface variable",
                    "shadowed-interface-variable",
                )
        for struct in getattr(self.program, "structs", None) or []:
            names = _interface_block_names(struct)
            if names is not None and names & used:
                self.refuse(
                    f"interface block '{struct.name}' is used by the geometry stage",
                    "interface-block",
                )
        return inputs, outputs

    def refuse_unsupported(self, reachable, used):
        for builtin, reason in _REFUSED_BUILTINS.items():
            if builtin in used:
                self.refuse(
                    f"the geometry stage uses {builtin}, which HLSL output does not carry yet",
                    reason,
                )
        for function in reachable:
            for parent, _field, _key, node in slots(function.body):
                if isinstance(node, A.FunctionCallNode):
                    name = call_name(node)
                    if name in _MULTI_STREAM_CALLS:
                        self.refuse(
                            f"'{function.name}' calls {name}, which writes a numbered vertex stream",
                            "multiple-streams",
                        )
                if isinstance(node, A.MemberAccessNode) and _is_gl_in(node.object_expr):
                    if node.member not in _GL_IN_MEMBERS:
                        reason = _REFUSED_BUILTINS.get(node.member, "input-builtin")
                        self.refuse(
                            f"the geometry stage reads gl_in[].{node.member}", reason
                        )
        for variable in getattr(self.program, "global_variables", None) or []:
            for attribute in getattr(variable, "attributes", None) or []:
                if str(getattr(attribute, "name", "")).lower() == "stream":
                    self.refuse(
                        f"output '{variable.name}' is declared on a numbered vertex stream",
                        "multiple-streams",
                    )

    def input_structure(self, name, inputs, used):
        members, nodes, taken = {}, [], set()
        for variable in inputs.values():
            var_type = getattr(variable, "var_type", None)
            if not isinstance(var_type, A.ArrayType):
                self.refuse(
                    f"geometry input '{variable.name}' is not a per-vertex array",
                    "input-array",
                )
            member = _unique(variable.name, taken)
            members[variable.name] = member
            nodes.append(
                _member(member, var_type.element_type, variable, interpolation=False)
            )
        if "gl_in" in used:
            member = _unique(_GL_IN_MEMBERS["gl_Position"], taken)
            members["gl_in.gl_Position"] = member
            nodes.append(
                A.StructMemberNode(
                    member,
                    A.VectorType(A.PrimitiveType("float"), 4),
                    attributes=[A.AttributeNode("gl_Position")],
                )
            )
        return members, A.StructNode(name, nodes)

    def output_structure(self, name, outputs, used):
        members, nodes, taken = {}, [], set()
        for variable in outputs.values():
            member = _unique(variable.name, taken)
            members[variable.name] = (member, None)
            nodes.append(
                _member(member, variable.var_type, variable, interpolation=True)
            )
        for builtin, (preferred, type_name) in _OUTPUT_BUILTINS.items():
            if builtin not in used:
                continue
            member = _unique(preferred, taken)
            members[builtin] = (member, type_name)
            type_node = (
                A.VectorType(A.PrimitiveType("float"), 4)
                if type_name == "vec4"
                else A.PrimitiveType(type_name)
            )
            nodes.append(
                A.StructMemberNode(
                    member, type_node, attributes=[A.AttributeNode(builtin)]
                )
            )
        return members, A.StructNode(name, nodes)

    def needs(self, reachable, inputs, output_members, builtin_parameters):
        """Which of the input, output, stream and built-in parameters each function needs."""
        needs = {}
        for function in reachable:
            names = _identifier_names(function.body)
            calls = {
                call_name(node)
                for _p, _f, _k, node in slots(function.body)
                if isinstance(node, A.FunctionCallNode)
            }
            direct = set()
            if names & (set(inputs) | {"gl_in"}):
                direct.add("input")
            if names & set(output_members) or "EmitVertex" in calls:
                direct.add("output")
            if calls & set(_STREAM_METHODS):
                direct.add("stream")
            direct |= names & set(builtin_parameters)
            needs[function.name] = direct
        changed = True
        while changed:
            changed = False
            for function in reachable:
                for _p, _f, _k, node in slots(function.body):
                    if not isinstance(node, A.FunctionCallNode):
                        continue
                    inherited = needs.get(call_name(node), set())
                    if not inherited <= needs[function.name]:
                        needs[function.name] |= inherited
                        changed = True
        return needs

    def refuse_shared_helpers(self, helpers):
        others = self.other_stage_functions()
        for helper in helpers:
            if any(helper is other for other in others):
                self.refuse(
                    f"helper '{helper.name}' uses the geometry interface and is also called by another stage",
                    "shared-helper",
                )

    def rewrite_function(
        self,
        function,
        count,
        inputs,
        input_name,
        input_members,
        output_name,
        output_members,
        stream_name,
        builtin_parameters,
    ):
        body = function.body
        for parent, field, key, node in list(slots(body)):
            if _is_length_call(node, set(inputs) | {"gl_in"}):
                set_child(parent, field, key, int_literal(count))
        if "gl_in" in _identifier_names(body):
            source = _input_member(input_name, input_members["gl_in.gl_Position"])
            if rewrite_array_reads(
                function, "gl_in", None, count, {"gl_Position": source}, self.refuse
            ):
                self.refuse(
                    f"'{function.name}' uses gl_in other than through gl_in[i].gl_Position",
                    "input-builtin",
                )
        prologue = []
        for name, variable in inputs.items():
            if name in _identifier_names(body):
                prologue += rewrite_array_reads(
                    function,
                    name,
                    variable.var_type.element_type,
                    count,
                    {None: _input_member(input_name, input_members[name])},
                    self.refuse,
                )
        for parent, field, key, node in list(slots(body)):
            if get_child(parent, field, key) is not node:
                continue  # replaced together with its parent
            replacement = self.replacement(
                parent,
                node,
                output_name,
                output_members,
                stream_name,
                builtin_parameters,
            )
            if replacement is not None:
                set_child(parent, field, key, replacement)
        body.statements[0:0] = prologue

    def replacement(
        self, parent, node, output_name, output_members, stream_name, builtins
    ):
        if isinstance(node, A.FunctionCallNode) and call_name(node) in _STREAM_METHODS:
            if node.arguments:
                self.refuse(
                    f"{call_name(node)} takes no arguments", "stream-call-arguments"
                )
            method = _STREAM_METHODS[call_name(node)]
            arguments = [A.IdentifierNode(output_name)] if method == "Append" else []
            return A.FunctionCallNode(
                A.MemberAccessNode(A.IdentifierNode(stream_name), method), arguments
            )
        if isinstance(node, A.AssignmentNode) and isinstance(
            node.target, A.IdentifierNode
        ):
            found = output_members.get(node.target.name)
            if found is not None and found[1] == "uint":
                if node.operator != "=":
                    self.refuse(
                        f"compound assignment to {node.target.name}", "builtin-update"
                    )
                node.target = _output_member(output_name, found[0])
                value = self.replacement(
                    node,
                    node.value,
                    output_name,
                    output_members,
                    stream_name,
                    builtins,
                )
                # The value's own slot is stale once wrapped, so its root is rewritten here; the nodes below
                # it keep their parents and are rewritten by the caller's walk.
                node.value = _convert("uint", node.value if value is None else value)
                return None
        if not isinstance(node, A.IdentifierNode):
            return None
        if node.name in builtins:
            return _convert("int", A.IdentifierNode(builtins[node.name][0]))
        found = output_members.get(node.name)
        if found is None:
            return None
        if found[1] == "uint":
            if isinstance(parent, A.UnaryOpNode) and parent.operator in ("++", "--"):
                self.refuse(f"increment of {node.name}", "builtin-update")
            return _convert("int", _output_member(output_name, found[0]))
        return _output_member(output_name, found[0])

    def thread_arguments(self, reachable, needs, tokens):
        for function in reachable:
            for _p, _f, _k, node in slots(function.body):
                if not isinstance(node, A.FunctionCallNode):
                    continue
                callee = self.functions.get(call_name(node))
                if callee is None or callee is self.entry:
                    continue
                needed = needs.get(callee.name) or set()
                extra = [
                    A.IdentifierNode(name)
                    for name, _type, _qualifiers, key in tokens
                    if key in needed
                ]
                if extra:
                    # ``args`` aliases ``arguments`` on call nodes.
                    node.arguments = node.args = list(node.arguments or []) + extra

    def drop_interface_declarations(self, inputs, outputs):
        gathered = {id(variable) for variable in [*inputs.values(), *outputs.values()]}
        self.stage.local_variables = [
            variable
            for variable in getattr(self.stage, "local_variables", None) or []
            if id(variable) not in gathered
        ]
        self.program.global_variables = [
            variable
            for variable in getattr(self.program, "global_variables", None) or []
            if id(variable) not in gathered
        ]
        self.program.structs = [
            struct
            for struct in self.program.structs
            if not _is_per_vertex_block(struct)
        ]


def _is_void(type_node):
    if type_node is None:
        return True
    return str(getattr(type_node, "name", type_node)) == "void"


def _direction(variable):
    for attribute in getattr(variable, "attributes", None) or []:
        name = str(getattr(attribute, "name", "")).lower()
        if name in ("input", "output"):
            return name
    qualifiers = {str(q).lower() for q in getattr(variable, "qualifiers", None) or []}
    if "in" in qualifiers:
        return "input"
    if "out" in qualifiers:
        return "output"
    return None


def _identifier_names(body):
    return {
        node.name
        for _p, _f, _k, node in slots(body)
        if isinstance(node, A.IdentifierNode)
    }


def _parameter_names(function):
    return {parameter.name for parameter in getattr(function, "parameters", None) or []}


def _attribute_names(node):
    return {
        str(getattr(attribute, "name", ""))
        for attribute in getattr(node, "attributes", None) or []
    }


def _attribute_argument_names(node, attribute_name):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) == attribute_name:
            return {
                str(getattr(argument, "name", getattr(argument, "value", argument)))
                for argument in getattr(attribute, "arguments", None) or []
            }
    return set()


def _interface_block_names(struct):
    """Names a GLSL interface block is reached through: its instance, else its members; ``None`` for other structs."""
    if "glsl_interface_block" not in _attribute_names(struct) or _is_per_vertex_block(
        struct
    ):
        return None
    instance = _attribute_argument_names(struct, "glsl_interface_instance")
    if instance:
        return instance
    return {member.name for member in getattr(struct, "members", None) or []}


def _is_per_vertex_block(struct):
    """A ``gl_PerVertex`` redeclaration, whose members are the built-ins the rewrite carries or refuses."""
    if getattr(struct, "name", None) != "gl_PerVertex":
        return False
    if "glsl_interface_block" not in _attribute_names(struct):
        return False
    names = {member.name for member in getattr(struct, "members", None) or []}
    return names <= _PER_VERTEX_BLOCK_MEMBERS


def _is_gl_in(node):
    return (
        isinstance(node, A.ArrayAccessNode)
        and isinstance(node.array_expr, A.IdentifierNode)
        and node.array_expr.name == "gl_in"
    )


def _is_length_call(node, names):
    if not isinstance(node, A.FunctionCallNode) or node.arguments:
        return False
    function = getattr(node, "function", None)
    return (
        isinstance(function, A.MemberAccessNode)
        and function.member == "length"
        and isinstance(function.object_expr, A.IdentifierNode)
        and function.object_expr.name in names
    )


def _unique(preferred, taken):
    name, index = preferred, 1
    while name in taken:
        index += 1
        name = f"{preferred}{index}"
    taken.add(name)
    return name


def _member(name, type_node, variable, *, interpolation):
    attributes = []
    for attribute in getattr(variable, "attributes", None) or []:
        attribute_name = str(getattr(attribute, "name", "")).lower()
        if attribute_name in _DROPPED_ATTRIBUTES:
            continue
        if attribute_name in _INTERPOLATION and not interpolation:
            continue
        attributes.append(
            A.AttributeNode(attribute.name, list(attribute.arguments or []))
        )
    if interpolation:
        present = {str(attribute.name).lower() for attribute in attributes}
        attributes += [
            A.AttributeNode(str(qualifier).lower())
            for qualifier in getattr(variable, "qualifiers", None) or []
            if str(qualifier).lower() in _INTERPOLATION
            and str(qualifier).lower() not in present
        ]
    member = A.StructMemberNode(name, type_node, attributes=attributes)
    member.source_location = getattr(variable, "source_location", None)
    return member


def _input_member(input_name, member):
    def source(index):
        return A.MemberAccessNode(
            A.ArrayAccessNode(A.IdentifierNode(input_name), index), member
        )

    return source


def _output_member(output_name, member):
    return A.MemberAccessNode(A.IdentifierNode(output_name), member)


def _convert(type_name, value):
    return A.FunctionCallNode(A.IdentifierNode(type_name), [value])
