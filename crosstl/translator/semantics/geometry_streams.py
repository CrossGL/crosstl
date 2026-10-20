"""HLSL-style geometry stages rewritten into GLSL's geometry interface for GLSL and SPIR-V targets.

An HLSL geometry shader, and the CrossGL the HLSL importer writes for it, receives its primitive as an entry-point
array parameter qualified with its topology and writes vertices into a stream object::

    geometry {
        @instance(2) @maxvertexcount(3)
        void main(triangle VSOutput input[3], uint id @ gl_InvocationID, inout TriangleStream<GSOutput> stream) {
            ...; stream.Append(vertex); stream.RestartStrip();
        }
    }

GLSL (4.60 sections 4.4.1.2, 4.4.2.2 and 8.16) and SPIR-V have no stream objects. A geometry shader declares its
input and output primitives and vertex count as stage layouts, reads each input vertex from per-vertex input arrays
and ``gl_in[]``, and emits a vertex by writing the stage outputs and calling ``EmitVertex()``. The GLSL and SPIR-V
emitters lower that form. :func:`lower_geometry_streams` rewrites an HLSL-style geometry stage into it:

* the topology qualifier becomes ``layout(points|lines|lines_adjacency|triangles|triangles_adjacency) in``, with
  ``invocations = N`` from ``@instance(N)``; the stream kind becomes ``layout(points|line_strip|triangle_strip,
  max_vertices = N) out`` from ``@maxvertexcount(N)``;
* each user member of the input vertex structure becomes a per-vertex input array, and ``gl_Position`` and
  ``gl_PointSize`` members are read from ``gl_in[i]``. Reads ``input[i].member`` read those directly; any other use of
  the input array reads a local copy of it;
* ``SV_GSInstanceID`` and ``SV_PrimitiveID`` parameters become locals initialised from ``gl_InvocationID`` and
  ``gl_PrimitiveIDIn``;
* each user member of the stream's vertex structure becomes a stage output; ``stream.Append(v)`` writes every output
  member, built-in ones (``gl_Position``, ``gl_PointSize``, ``gl_Layer``, ``gl_ViewportIndex``, ``gl_PrimitiveID``)
  included, and calls ``EmitVertex()``; ``stream.RestartStrip()`` becomes ``EndPrimitive()``. Helpers that take the
  stream lose that parameter and move into the geometry stage.

A user varying takes the location the neighbouring stage gives the same structure member
(:mod:`..codegen.stage_io_semantic_locations`): its explicit ``location``, else for GLSL the location of its semantic,
else for SPIR-V its semantic's fixed location or its running position in the structure. A GLSL varying whose
semantic has no location would link by name, and a geometry stage cannot declare an input and an output of the same
name, so it is refused. Several output streams (``EmitStreamVertex``), clip and cull distances and other system
values in the vertex structures are refused with :class:`GeometryStreamUnsupportedError` until they are lowered.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name
from .stage_varyings import GLSL_TARGETS as _GLSL_TARGETS
from .stage_varyings import SPIRV_TARGETS as _SPIRV_TARGETS
from .stage_varyings import FreshNames
from .stage_varyings import builtin as _builtin
from .stage_varyings import call_statement as _call_statement
from .stage_varyings import indexed as _indexed
from .stage_varyings import int_literal as _int
from .stage_varyings import is_system_value as _is_system_value
from .stage_varyings import literal_int as _literal_int
from .stage_varyings import location_span as _location_span
from .stage_varyings import member_type as _member_type
from .stage_varyings import (
    neighbour_location,
    per_vertex_builtin,
    replace_statement,
    rewrite_array_reads,
)
from .stage_varyings import semantic as _semantic
from .stage_varyings import slots as _slots
from .stage_varyings import stage_name as _stage_name
from .stage_varyings import stages as _stages
from .stage_varyings import type_name as _type_name
from .stage_varyings import varying_declaration

__all__ = ["GeometryStreamUnsupportedError", "lower_geometry_streams"]

_INPUT_TOPOLOGIES = {
    "point": ("points", 1),
    "line": ("lines", 2),
    "lineadj": ("lines_adjacency", 4),
    "triangle": ("triangles", 3),
    "triangleadj": ("triangles_adjacency", 6),
}
_OUTPUT_TOPOLOGIES = {
    "PointStream": "points",
    "LineStream": "line_strip",
    "TriangleStream": "triangle_strip",
}
# Built-in members of the input vertex, read from gl_in[i].
_INPUT_BUILTINS = frozenset({"gl_Position", "gl_PointSize"})
# Built-in members of the output vertex, with the type GLSL declares them with.
_OUTPUT_BUILTINS = {
    "gl_Position": "vec4",
    "gl_PointSize": "float",
    "gl_Layer": "int",
    "gl_ViewportIndex": "int",
    "gl_PrimitiveID": "int",
}
# Entry-point system-value parameters and the GLSL geometry built-in that carries each.
_PARAMETER_BUILTINS = {
    "gl_InvocationID": "gl_InvocationID",
    "gl_PrimitiveID": "gl_PrimitiveIDIn",
}


class GeometryStreamUnsupportedError(TranslationDiagnostic):
    """An HLSL-style geometry stage construct the GLSL geometry interface cannot carry yet."""

    def __init__(self, target, detail, *, reason):
        self.reason_code = reason
        self.missing_capabilities = (f"{target}.geometry-stream.{reason}",)
        super().__init__(
            f"{detail}; {target} geometry shaders write vertices through stage outputs and EmitVertex()",
            target=target,
            code=f"{CODE_PREFIX}{target}-geometry-stream-unsupported",
        )


def lower_geometry_streams(program, target) -> int:
    """Rewrite HLSL-style geometry stages of ``program`` for GLSL or SPIR-V; returns how many stages changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in (
        _GLSL_TARGETS | _SPIRV_TARGETS
    ):
        return 0
    changed = 0
    for stage in _stages(program):
        if _stage_name(stage) != "geometry":
            continue
        entry = getattr(stage, "entry_point", None)
        if entry is None or not getattr(entry, "parameters", None):
            continue
        _GeometryLowering(program, stage, target).run()
        changed += 1
    return changed


class _GeometryLowering:
    def __init__(self, program, stage, target):
        self.program = program
        self.stage = stage
        self.entry = stage.entry_point
        self.target = target
        self.fresh = FreshNames(program)
        self.structs = {
            struct.name: struct
            for struct in list(getattr(program, "structs", None) or []) + list(
                getattr(stage, "local_structs", None) or []
            )
            if isinstance(struct, A.StructNode)
        }

    def refuse(self, detail, reason):
        raise GeometryStreamUnsupportedError(self.target, detail, reason=reason)

    def run(self):
        entry = self.entry
        primitive, stream, builtins = self.classify_parameters(entry)
        if getattr(self.stage, "layout_qualifiers", None):
            self.refuse(
                f"geometry entry point '{entry.name}' takes HLSL stream parameters and also declares GLSL stage layouts",
                "mixed-geometry-interface",
            )
        input_topology, vertex_count = self.input_topology(primitive)
        element_name = self.stream_element(stream)
        max_vertices = self.take_attribute(entry, "maxvertexcount", required=True)
        invocations = self.take_attribute(entry, "instance", required=False)
        in_entries = [A.AttributeNode(input_topology)]
        if invocations is not None:
            in_entries.append(A.AttributeNode("invocations", [_int(invocations)]))
        out_entries = [
            A.AttributeNode(_OUTPUT_TOPOLOGIES[_type_name(stream.param_type)]),
            A.AttributeNode("max_vertices", [_int(max_vertices)]),
        ]
        self.stage.layout_qualifiers = [
            A.LayoutQualifierNode(in_entries, "in"),
            A.LayoutQualifierNode(out_entries, "out"),
        ]

        declarations = []
        sources = self.input_sources(primitive, vertex_count, declarations)
        outputs = self.output_targets(element_name, declarations)
        self.stage.local_variables = declarations + list(
            getattr(self.stage, "local_variables", None) or []
        )

        prologue = []
        for parameter, builtin in builtins:
            prologue.append(
                A.VariableNode(
                    name=parameter.name,
                    var_type=parameter.param_type,
                    initial_value=A.FunctionCallNode(
                        A.IdentifierNode(_type_name(parameter.param_type)),
                        [A.IdentifierNode(builtin)],
                    ),
                )
            )
        prologue.extend(self.rewrite_input_reads(primitive, vertex_count, sources))
        entry.parameters = []
        entry.body.statements[0:0] = prologue

        helpers = self.stream_helpers(element_name)
        self.rewrite_streams(entry, {stream.name}, element_name, outputs, helpers)
        for helper, (index, parameter) in helpers.items():
            helper.parameters = [
                p for i, p in enumerate(helper.parameters) if i != index
            ]
            self.rewrite_streams(
                helper, {parameter.name}, element_name, outputs, helpers
            )
            self.move_into_stage(helper)

    # Parameters and stage layouts.

    def classify_parameters(self, entry):
        primitive = stream = None
        builtins = []
        for parameter in entry.parameters:
            type_name = _type_name(parameter.param_type)
            qualifiers = {str(q).lower() for q in parameter.qualifiers or []}
            if type_name in _OUTPUT_TOPOLOGIES:
                if stream is not None:
                    self.refuse(
                        f"geometry entry point '{entry.name}' writes several output streams; GLSL needs "
                        "EmitStreamVertex() and layout(stream = N) outputs, which are not lowered yet",
                        "multiple-streams",
                    )
                stream = parameter
            elif qualifiers & set(_INPUT_TOPOLOGIES):
                if primitive is not None:
                    self.refuse(
                        f"geometry entry point '{entry.name}' takes two input primitives",
                        "multiple-primitives",
                    )
                primitive = parameter
            else:
                builtin = _builtin(parameter)
                if builtin not in _PARAMETER_BUILTINS:
                    self.refuse(
                        f"geometry entry point parameter '{parameter.name}' ({builtin or 'no system value'}) "
                        "is neither the input primitive, the output stream, SV_GSInstanceID nor SV_PrimitiveID",
                        "parameter",
                    )
                if _type_name(parameter.param_type) not in {"int", "uint"}:
                    self.refuse(
                        f"geometry system-value parameter '{parameter.name}' is not an int or uint",
                        "parameter-type",
                    )
                builtins.append((parameter, _PARAMETER_BUILTINS[builtin]))
        if primitive is None or stream is None:
            self.refuse(
                f"geometry entry point '{entry.name}' needs one topology-qualified input array and one "
                "PointStream, LineStream or TriangleStream parameter",
                "entry-signature",
            )
        return primitive, stream, builtins

    def input_topology(self, primitive):
        qualifiers = {str(q).lower() for q in primitive.qualifiers or []}
        topologies = sorted(qualifiers & set(_INPUT_TOPOLOGIES))
        if len(topologies) != 1:
            self.refuse(
                f"input '{primitive.name}' has several topology qualifiers", "topology"
            )
        topology = topologies[0]
        layout, vertex_count = _INPUT_TOPOLOGIES[topology]
        param_type = primitive.param_type
        size = _literal_int(getattr(param_type, "size", None))
        if not isinstance(param_type, A.ArrayType) or size != vertex_count:
            self.refuse(
                f"{topology} input '{primitive.name}' must be an array of {vertex_count} vertices",
                "primitive-size",
            )
        return layout, vertex_count

    def stream_element(self, stream):
        args = getattr(stream.param_type, "generic_args", None) or []
        name = _type_name(args[0]) if len(args) == 1 else None
        if name not in self.structs:
            self.refuse(
                f"output stream '{stream.name}' must carry a vertex structure",
                "stream-element",
            )
        return name

    def take_attribute(self, function, name, *, required):
        found = [
            attr
            for attr in function.attributes or []
            if str(getattr(attr, "name", "")).lower() == name
        ]
        if not found:
            if required:
                self.refuse(
                    f"geometry entry point '{function.name}' has no [{name}(N)] attribute",
                    "missing-" + name,
                )
            return None
        value = _literal_int((found[0].arguments or [None])[0])
        if len(found) > 1 or value is None or value < 1:
            self.refuse(
                f"geometry entry point '{function.name}' needs one positive [{name}(N)]",
                "attribute-" + name,
            )
        function.attributes = [
            attr for attr in function.attributes if attr not in found
        ]
        return value

    # Stage interface variables.

    def varying_location(self, struct_name, member, running):
        return neighbour_location(
            self.target, struct_name, member, running, self.refuse
        )

    def varying(self, prefix, member, var_type, location, direction):
        name = self.fresh(f"{prefix}_{member.name}")
        return varying_declaration(name, member, var_type, location, [direction])

    def input_sources(self, primitive, vertex_count, declarations):
        """``(member name or None) -> function(index expression) -> source expression`` of each input value."""
        element = primitive.param_type.element_type
        struct = self.structs.get(_type_name(element))
        members = struct.members if struct is not None else [primitive]
        sources = {}
        running = 0
        for member in members:
            member_type = _member_type(member)
            key = member.name if struct is not None else None
            builtin = _builtin(member)
            if builtin in _INPUT_BUILTINS:
                sources[key] = per_vertex_builtin("gl_in", builtin)
            elif builtin is not None or _is_system_value(member):
                self.refuse(
                    f"input vertex member '{member.name}' carries {builtin or _semantic(member)}, which the "
                    "geometry lowering does not read yet",
                    "input-system-value",
                )
            elif (
                isinstance(member_type, A.ArrayType)
                or _type_name(member_type) in self.structs
            ):
                self.refuse(
                    f"input vertex member '{member.name}' is an array or structure",
                    "input-aggregate",
                )
            else:
                location = self.varying_location(_type_name(element), member, running)
                variable = self.varying(
                    "in",
                    member,
                    A.ArrayType(member_type, _int(vertex_count)),
                    location,
                    "in",
                )
                declarations.append(variable)
                sources[key] = _indexed(variable.name)
            running += _location_span(member_type)
        return sources

    def output_targets(self, element_name, declarations):
        """``member name -> (target name, GLSL type or None)`` for each member of the emitted vertex."""
        outputs = {}
        running = 0
        for member in self.structs[element_name].members:
            member_type = _member_type(member)
            builtin = _builtin(member)
            if builtin in _OUTPUT_BUILTINS:
                outputs[member.name] = (builtin, _OUTPUT_BUILTINS[builtin])
            elif builtin is not None or _is_system_value(member):
                self.refuse(
                    f"output vertex member '{member.name}' carries {builtin or _semantic(member)}, which the "
                    "geometry lowering does not write yet",
                    "output-system-value",
                )
            elif (
                isinstance(member_type, A.ArrayType)
                or _type_name(member_type) in self.structs
            ):
                self.refuse(
                    f"output vertex member '{member.name}' is an array or structure",
                    "output-aggregate",
                )
            else:
                location = self.varying_location(element_name, member, running)
                variable = self.varying("out", member, member_type, location, "out")
                declarations.append(variable)
                outputs[member.name] = (variable.name, None)
            running += _location_span(member_type)
        return outputs

    # Body rewrites.

    def rewrite_input_reads(self, primitive, vertex_count, sources):
        """Rewrite ``input[i].m`` reads in place; returns the statements that build a local copy when one is needed."""
        return rewrite_array_reads(
            self.entry,
            primitive.name,
            primitive.param_type.element_type,
            vertex_count,
            sources,
            self.refuse,
        )

    def stream_helpers(self, element_name):
        """Functions other than the entry point that take the stream: ``function -> (index, parameter)``."""
        helpers = {}
        for function in self.candidate_functions():
            if function is self.entry:
                continue
            found = [
                (index, parameter)
                for index, parameter in enumerate(function.parameters or [])
                if _type_name(parameter.param_type) in _OUTPUT_TOPOLOGIES
            ]
            if not found:
                continue
            args = getattr(found[0][1].param_type, "generic_args", None) or []
            if len(found) > 1 or len(args) != 1 or _type_name(args[0]) != element_name:
                self.refuse(
                    f"helper '{function.name}' takes a stream other than the geometry stage's one output stream",
                    "helper-stream",
                )
            helpers[function] = found[0]
        return helpers

    def candidate_functions(self):
        return list(getattr(self.program, "functions", None) or []) + list(
            getattr(self.stage, "local_functions", None) or []
        )

    def rewrite_streams(self, function, stream_names, element_name, outputs, helpers):
        by_name = {helper.name: index for helper, (index, _) in helpers.items()}
        if len(by_name) != len(helpers):
            self.refuse("two stream helpers share a name", "helper-overload")
        splices = []
        for parent, field, key, node in _slots(function.body):
            if isinstance(node, A.FunctionCallNode) and call_name(node) in by_name:
                index = by_name[call_name(node)]
                argument = (node.arguments or [None] * (index + 1))[index]
                if not (
                    isinstance(argument, A.IdentifierNode)
                    and argument.name in stream_names
                ):
                    self.refuse(
                        f"call to '{call_name(node)}' does not pass the output stream",
                        "helper-argument",
                    )
                node.arguments.pop(index)
                if getattr(node, "args", None) is not node.arguments:
                    node.args = node.arguments
                continue
            method = _stream_method(node, stream_names)
            if method is None:
                continue
            if not isinstance(parent, A.ExpressionStatementNode):
                self.refuse(
                    f"stream method {method[0]}() is used as a value", "stream-value"
                )
            splices.append((parent, method))
        for statement, (method, arguments) in splices:
            replacement = self.stream_statements(
                method, arguments, element_name, outputs
            )
            replace_statement(function.body, statement, replacement)
        for parent, field, key, node in _slots(function.body):
            if isinstance(node, A.IdentifierNode) and node.name in stream_names:
                self.refuse(
                    f"output stream '{node.name}' is used other than through Append() and RestartStrip()",
                    "stream-use",
                )

    def stream_statements(self, method, arguments, element_name, outputs):
        if method == "RestartStrip" and not arguments:
            return [_call_statement("EndPrimitive")]
        if method != "Append" or len(arguments) != 1:
            self.refuse(
                f"stream method {method}() is not Append(v) or RestartStrip()",
                "stream-method",
            )
        value = arguments[0]
        statements = []
        if not isinstance(value, A.IdentifierNode):
            name = self.fresh("gs_vertex")
            statements.append(
                A.VariableNode(
                    name=name, var_type=A.NamedType(element_name), initial_value=value
                )
            )
            value = A.IdentifierNode(name)
        for member_name, (target_name, glsl_type) in outputs.items():
            source = A.MemberAccessNode(A.IdentifierNode(value.name), member_name)
            member = next(
                m for m in self.structs[element_name].members if m.name == member_name
            )
            if glsl_type == "int" and _type_name(_member_type(member)) != "int":
                # gl_Layer, gl_ViewportIndex and gl_PrimitiveID are int; HLSL declares them uint.
                source = A.FunctionCallNode(A.IdentifierNode(glsl_type), [source])
            statements.append(A.AssignmentNode(A.IdentifierNode(target_name), source))
        statements.append(_call_statement("EmitVertex"))
        return statements

    def move_into_stage(self, helper):
        functions = getattr(self.program, "functions", None) or []
        if helper in functions:
            functions.remove(helper)
            local = list(getattr(self.stage, "local_functions", None) or [])
            self.stage.local_functions = [helper] + local


# Helpers.


def _stream_method(node, stream_names):
    if not isinstance(node, A.FunctionCallNode):
        return None
    callee = node.function
    if (
        isinstance(callee, A.MemberAccessNode)
        and isinstance(callee.object_expr, A.IdentifierNode)
        and callee.object_expr.name in stream_names
    ):
        return callee.member, list(node.arguments or [])
    return None
