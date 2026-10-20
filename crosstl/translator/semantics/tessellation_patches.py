"""HLSL-style tessellation stages lowered to GLSL's tessellation interface for the GLSL and SPIR-V targets.

An HLSL hull shader, and the CrossGL the HLSL importer writes for it, returns one output control point per
invocation and names a patch-constant function that the pipeline runs once per patch, after every control point::

    tessellation_control {
        @domain("tri") @partitioning("fractional_odd") @outputtopology("triangle_cw") @outputcontrolpoints(3)
        @patchconstantfunc("PatchFunc") @maxtessfactor(16.0)
        HSOut HSMain(InputPatch<VSOut, 3> patch, uint id @ gl_InvocationID) { ... }
    }
    PatchOut PatchFunc(InputPatch<VSOut, 3> patch, OutputPatch<HSOut, 3> points) { ... }   // gl_TessLevel* members

and a domain shader receives the patch constants and the output patch as entry-point parameters::

    tessellation_evaluation {
        @domain("tri")
        DSOut DSMain(PatchOut pc, vec3 uvw @ gl_TessCoord, const OutputPatch<HSOut, 3> patch) { ... }
    }

GLSL (4.60 sections 4.3.4, 4.3.6, 4.4.1.2, 7.1.3 and 8.16) and SPIR-V have neither patch objects nor
patch-constant functions: a tessellation control shader reads per-vertex input arrays and ``gl_in[]``, writes its
invocation's element of per-vertex output arrays and ``gl_out[]``, and writes ``patch out`` variables and
``gl_TessLevelOuter``/``gl_TessLevelInner``; an evaluation shader reads those back and writes ordinary stage outputs.
:func:`lower_tessellation_patches` rewrites both HLSL-style stages into that form:

* the hull entry point becomes a function ``<entry>_control_point`` without its patch parameter, called by a new
  parameterless entry of the same name. That entry writes the returned control point to the outputs at
  ``gl_InvocationID``, waits on ``barrier()`` and, in invocation 0, calls the patch-constant function and writes its
  result to the tessellation levels and ``patch out`` variables, which is when and how often HLSL runs it.
  ``@outputcontrolpoints(N)`` becomes ``layout(vertices = N) out``; ``@maxtessfactor`` is dropped, as DXC's SPIR-V
  backend drops it: it is an optimisation hint, and GLSL clamps levels to ``gl_MaxTessGenLevel`` itself;
* the domain entry point becomes ``<entry>_domain_point``, called by a new entry that rebuilds the patch-constant
  structure from ``gl_TessLevel*`` and the ``patch in`` variables and writes the returned vertex to the stage
  outputs. GLSL declares the spacing and winding on the evaluation shader, HLSL on the hull, so the domain stage
  takes ``@partitioning`` and ``@outputtopology`` from the program's hull stage. A GLSL domain stage translated
  without its hull is refused; SPIR-V leaves those execution modes to the control stage, as DXC does;
* ``patch[i].member`` reads of an ``InputPatch`` or ``OutputPatch`` read the arrayed variables directly; any other use
  of the patch reads a local array copy of it, and helpers that take a patch take that array.

Locations: the hull's inputs take the locations the vertex stage gives the same members
(:func:`.stage_varyings.neighbour_location`) and the domain's outputs the ones the next stage expects. Between hull
and domain, control-point members take their running location in the control-point structure from 0 and
patch-constant members continue after them, in declaration order. When the program holds both stages the domain
stage links each member by semantic, as HLSL does, to the hull's location for it; translated separately, both
stages must declare the same members in the same order. System values other than those GLSL tessellation shaders
read or write (clip and cull distances, viewport and layer) are refused with
:class:`TessellationPatchUnsupportedError`, as are entry points of other shapes.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name
from .stage_varyings import (
    GLSL_TARGETS,
    SPIRV_TARGETS,
    FreshNames,
    builtin,
    call_statement,
    explicit_location,
    indexed,
    int_literal,
    is_system_value,
    literal_int,
    location_span,
    member_type,
    neighbour_location,
    per_vertex_builtin,
    rewrite_array_reads,
    semantic,
    slots,
    stage_name,
    stages,
    type_name,
    varying_declaration,
)

__all__ = ["TessellationPatchUnsupportedError", "lower_tessellation_patches"]

_PATCH_TYPES = frozenset({"InputPatch", "OutputPatch"})
# Built-in members of a control point, read from gl_in[i] and written to gl_out[gl_InvocationID].
_POINT_BUILTINS = frozenset({"gl_Position", "gl_PointSize"})
# Tessellation levels and the number of elements GLSL declares each with.
_LEVELS = {"gl_TessLevelOuter": 4, "gl_TessLevelInner": 2}
# Entry-point system values and the GLSL built-in that carries each.
_HULL_PARAMETERS = {
    "gl_InvocationID": "gl_InvocationID",
    "gl_PrimitiveID": "gl_PrimitiveID",
}
_PATCH_FUNCTION_PARAMETERS = {"gl_PrimitiveID": "gl_PrimitiveID"}
_DOMAIN_PARAMETERS = {"gl_PrimitiveID": "gl_PrimitiveID"}
_HULL_ONLY_ATTRIBUTES = frozenset({"patchconstantfunc", "maxtessfactor"})
_DOMAIN_FROM_HULL = ("partitioning", "outputtopology")


class TessellationPatchUnsupportedError(TranslationDiagnostic):
    """An HLSL-style tessellation construct the GLSL tessellation interface cannot carry yet."""

    def __init__(self, target, detail, *, reason):
        self.reason_code = reason
        self.missing_capabilities = (f"{target}.tessellation-patch.{reason}",)
        super().__init__(
            f"{detail}; {target} tessellation shaders exchange control points through per-vertex arrays, "
            "patch variables and gl_TessLevelOuter/gl_TessLevelInner",
            target=target,
            code=f"{CODE_PREFIX}{target}-tessellation-patch-unsupported",
        )


def lower_tessellation_patches(program, target) -> int:
    """Rewrite HLSL-style tessellation stages of ``program`` for GLSL or SPIR-V; returns how many stages changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in (
        GLSL_TARGETS | SPIRV_TARGETS
    ):
        return 0
    found = {}
    for stage in stages(program):
        name = stage_name(stage)
        if name in ("tessellation_control", "tessellation_evaluation"):
            entry = getattr(stage, "entry_point", None)
            if entry is not None and _returns_struct(program, stage, entry):
                found[name] = stage
    if not found:
        return 0
    lowering = _TessellationLowering(program, target)
    hull = None
    if "tessellation_control" in found:
        hull = lowering.lower_hull(found["tessellation_control"])
    if "tessellation_evaluation" in found:
        lowering.lower_domain(found["tessellation_evaluation"], hull)
    for stage in found.values():
        lowering.patch_helpers(stage)
    return len(found)


class _Interface:
    """Locations of the user members a hull passes its domain, keyed by semantic."""

    def __init__(self, refuse):
        self.refuse = refuse
        self.points = {}
        self.patch = {}
        self.end = 0

    def place(self, table, struct, members, start):
        running, used = start, []
        for member in members:
            span = location_span(member_type(member))
            location = explicit_location(member)
            location = running if location is None else location
            for other, other_span, other_name in used:
                if location < other + other_span and other < location + span:
                    self.refuse(
                        f"members '{struct.name}.{other_name}' and '{struct.name}.{member.name}' share a location",
                        "location-overlap",
                    )
            used.append((location, span, member.name))
            table[_link_key(member)] = (location, _type_key(member_type(member)))
            running = location + span
            self.end = max(self.end, running)
        return table


class _TessellationLowering:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.fresh = FreshNames(program)
        self.stage = None

    def refuse(self, detail, reason):
        raise TessellationPatchUnsupportedError(self.target, detail, reason=reason)

    def structs(self, stage):
        return {
            struct.name: struct
            for struct in list(getattr(self.program, "structs", None) or []) + list(
                getattr(stage, "local_structs", None) or []
            )
            if isinstance(struct, A.StructNode)
        }

    # Hull stage.

    def lower_hull(self, stage):
        self.stage = stage
        entry = stage.entry_point
        identity = _EntryIdentity(entry)
        structs = self.structs(stage)
        if getattr(stage, "layout_qualifiers", None):
            self.refuse(
                f"hull entry point '{entry.name}' returns a control point and also declares GLSL stage layouts",
                "mixed-tessellation-interface",
            )
        points = _attribute_int(entry, "outputcontrolpoints")
        if points is None or points < 1:
            self.refuse(
                f"hull entry point '{entry.name}' needs one positive [outputcontrolpoints(N)]",
                "outputcontrolpoints",
            )
        patch_function = self.patch_function(entry)
        point_struct = structs[type_name(entry.return_type)]
        patch_struct = structs.get(type_name(patch_function.return_type))
        if patch_struct is None:
            self.refuse(
                f"patch-constant function '{patch_function.name}' does not return a structure",
                "patch-constant-return",
            )
        for name in (entry.name, patch_function.name):
            self.refuse_calls(name)

        declarations = []
        interface = self.hull_interface(point_struct, patch_struct)
        input_reads = {}
        point_writes, point_reads = self.point_outputs(
            point_struct, interface, points, declarations
        )
        level_writes, patch_writes = self.patch_outputs(
            patch_struct, interface, declarations
        )

        control_point = self.detach(
            entry,
            f"{entry.name}_control_point",
            _HULL_PARAMETERS,
            {
                "InputPatch": lambda p: self.input_patch(
                    p, structs, input_reads, declarations
                )
            },
        )
        patch_call = self.detach(
            patch_function,
            None,
            _PATCH_FUNCTION_PARAMETERS,
            {
                "InputPatch": lambda p: self.input_patch(
                    p, structs, input_reads, declarations
                ),
                "OutputPatch": lambda p: self.output_patch(
                    p, point_struct, points, point_reads
                ),
            },
        )
        stage.local_variables = declarations + self.without_patch_declarations(
            stage, patch_struct
        )
        stage.layout_qualifiers = [
            A.LayoutQualifierNode(
                [A.AttributeNode("vertices", [int_literal(points)])], "out"
            )
        ]

        point = self.fresh("hs_point")
        patch = self.fresh("hs_patch")
        invocation = A.IdentifierNode("gl_InvocationID")
        body = [
            A.VariableNode(
                name=point,
                var_type=copy.deepcopy(entry.return_type),
                initial_value=control_point,
            )
        ]
        for member_name, target in point_writes:
            body.append(
                A.AssignmentNode(
                    target(invocation),
                    A.MemberAccessNode(A.IdentifierNode(point), member_name),
                )
            )
        body.append(call_statement("barrier"))
        first = [
            A.VariableNode(
                name=patch,
                var_type=copy.deepcopy(patch_function.return_type),
                initial_value=patch_call,
            )
        ]
        first += [write(A.IdentifierNode(patch)) for write in level_writes]
        first += [write(A.IdentifierNode(patch)) for write in patch_writes]
        body.append(
            A.IfNode(
                A.BinaryOpNode(
                    A.IdentifierNode("gl_InvocationID"), "==", int_literal(0)
                ),
                A.BlockNode(first),
            )
        )
        attributes = [
            attr
            for attr in identity.attributes
            if _attr_name(attr) not in _HULL_ONLY_ATTRIBUTES | {"outputcontrolpoints"}
        ]
        self.replace_entry(stage, entry, identity, body, attributes)
        self.move_into_stage(patch_function)
        return _Hull(attributes, interface, points)

    def without_patch_declarations(self, stage, patch_struct):
        """The stage's variables less the HLSL importer's ``out T name @ SEMANTIC`` copies of patch constants.

        The importer declares each user patch constant as a stage output so that cross-stage validation finds the
        domain stage's input; the patch-constant function's result is what the stage writes, through the
        ``patch out`` variables above, so the unused copies are dropped.
        """
        keys = {_link_key(m) for m in self.user_patch_members(patch_struct)}
        used = {
            node.name
            for function in self.functions()
            for _, _, _, node in slots(function.body)
            if isinstance(node, A.IdentifierNode)
        }
        return [
            variable
            for variable in stage.local_variables or []
            if not (
                isinstance(variable, A.VariableNode)
                and "out" in {str(q).lower() for q in variable.qualifiers or []}
                and semantic(variable) is not None
                and _link_key(variable) in keys
                and variable.name not in used
            )
        ]

    def patch_function(self, entry):
        found = [
            attr
            for attr in entry.attributes or []
            if _attr_name(attr) == "patchconstantfunc"
        ]
        arguments = found[0].arguments if len(found) == 1 else None
        if not arguments or len(arguments) != 1:
            self.refuse(
                f"hull entry point '{entry.name}' needs one [patchconstantfunc(\"name\")]",
                "patchconstantfunc",
            )
        name = str(getattr(arguments[0], "value", arguments[0])).strip().strip("\"'")
        matches = [
            function
            for function in self.functions()
            if function.name == name and function is not entry
        ]
        if len(matches) != 1:
            self.refuse(
                f"patch-constant function '{name}' is not one function of the program",
                "patchconstantfunc-target",
            )
        return matches[0]

    def hull_interface(self, point_struct, patch_struct):
        interface = _Interface(self.refuse)
        interface.place(
            interface.points, point_struct, self.user_point_members(point_struct), 0
        )
        interface.place(
            interface.patch,
            patch_struct,
            self.user_patch_members(patch_struct),
            interface.end,
        )
        return interface

    def point_outputs(self, struct, interface, points, declarations):
        """Writes of a returned control point, and the sources an ``OutputPatch`` of it reads."""
        writes, reads = [], {}
        for member in struct.members:
            name = builtin(member)
            if name in _POINT_BUILTINS:
                target = per_vertex_builtin("gl_out", name)
            else:
                location, _ = interface.points[_link_key(member)]
                variable = varying_declaration(
                    self.fresh(f"tc_out_{member.name}"),
                    member,
                    A.ArrayType(copy.deepcopy(member_type(member)), None),
                    location,
                    ["out"],
                )
                declarations.append(variable)
                target = indexed(variable.name)
            writes.append((member.name, target))
            reads[member.name] = target
        return writes, reads

    def patch_outputs(self, struct, interface, declarations):
        level_writes, patch_writes = [], []
        for member in struct.members:
            name = builtin(member)
            if name in _LEVELS:
                level_writes += self.level_copies(struct, member, name, write=True)
                continue
            location, _ = interface.patch[_link_key(member)]
            variable = varying_declaration(
                self.fresh(f"tc_patch_{member.name}"),
                member,
                copy.deepcopy(member_type(member)),
                location,
                ["patch", "out"],
            )
            declarations.append(variable)
            patch_writes.append(_member_copy(variable.name, member.name))
        if not any(builtin(m) == "gl_TessLevelOuter" for m in struct.members):
            self.refuse(
                f"patch-constant structure '{struct.name}' has no SV_TessFactor member",
                "missing-tess-factor",
            )
        return level_writes, patch_writes

    def input_patch(self, parameter, structs, cache, declarations):
        """The sources of an ``InputPatch``: per-vertex inputs located for the vertex stage, and ``gl_in[]``."""
        element, count = self.patch_shape(parameter, structs)
        if element.name not in cache:
            if cache:
                self.refuse(
                    "the hull entry point and its patch-constant function read input patches of different "
                    "control points",
                    "input-patch-type",
                )
            sources, running = {}, 0
            for member in element.members:
                name = builtin(member)
                if name in _POINT_BUILTINS:
                    sources[member.name] = per_vertex_builtin("gl_in", name)
                else:
                    self.refuse_value_member(element, member)
                    location = neighbour_location(
                        self.target, element.name, member, running, self.refuse
                    )
                    variable = varying_declaration(
                        self.fresh(f"tc_in_{member.name}"),
                        member,
                        A.ArrayType(copy.deepcopy(member_type(member)), None),
                        location,
                        ["in"],
                    )
                    declarations.append(variable)
                    sources[member.name] = indexed(variable.name)
                running += location_span(member_type(member))
            cache[element.name] = sources
        return element, count, cache[element.name]

    def output_patch(self, parameter, point_struct, points, reads):
        element, count = self.patch_shape(parameter, {point_struct.name: point_struct})
        if count != points:
            self.refuse(
                f"output patch '{parameter.name}' has {count} control points, the hull writes {points}",
                "output-patch-size",
            )
        return element, count, reads

    # Domain stage.

    def lower_domain(self, stage, hull):
        self.stage = stage
        entry = stage.entry_point
        identity = _EntryIdentity(entry)
        structs = self.structs(stage)
        if getattr(stage, "layout_qualifiers", None):
            self.refuse(
                f"domain entry point '{entry.name}' returns a vertex and also declares GLSL stage layouts",
                "mixed-tessellation-interface",
            )
        domain = _attribute_text(entry, "domain")
        if domain not in {"tri", "quad", "isoline"}:
            self.refuse(
                f'domain entry point \'{entry.name}\' needs [domain("tri"|"quad"|"isoline")]',
                "domain",
            )
        self.refuse_calls(entry.name)
        patches = [
            p for p in entry.parameters if type_name(p.param_type) in _PATCH_TYPES
        ]
        constants = [
            p
            for p in entry.parameters
            if type_name(p.param_type) in structs and semantic(p) is None
        ]
        if len(patches) > 1 or len(constants) > 1:
            self.refuse(
                f"domain entry point '{entry.name}' takes several patches or patch-constant structures",
                "domain-signature",
            )
        if patches and type_name(patches[0].param_type) != "OutputPatch":
            self.refuse(
                f"domain entry point '{entry.name}' reads an InputPatch; it reads the hull's OutputPatch",
                "domain-patch",
            )
        point_struct = None
        if patches:
            point_struct, count = self.patch_shape(patches[0], structs)
            if hull is not None and count != hull.points:
                self.refuse(
                    f"output patch '{patches[0].name}' has {count} control points, the hull writes "
                    f"{hull.points}",
                    "output-patch-size",
                )
        patch_struct = (
            structs[type_name(constants[0].param_type)] if constants else None
        )
        interface = self.domain_interface(hull, point_struct, patch_struct)

        declarations = []
        prologue, constant_name = [], None
        if patch_struct is not None:
            constant_name = self.fresh("ds_patch")
            prologue = self.patch_inputs(
                patch_struct, interface, constant_name, declarations
            )
        point_reads = (
            self.point_inputs(point_struct, interface, declarations)
            if point_struct is not None
            else {}
        )
        out_struct = structs[type_name(entry.return_type)]
        output_writes = self.domain_outputs(out_struct, declarations)

        coordinate = A.IdentifierNode("gl_TessCoord")
        parameters = dict(_DOMAIN_PARAMETERS)
        parameters["gl_TessCoord"] = "gl_TessCoord"
        domain_point = self.detach(
            entry,
            f"{entry.name}_domain_point",
            parameters,
            {"OutputPatch": lambda p: self.patch_shape(p, structs) + (point_reads,)},
            constants=(
                {id(constants[0]): A.IdentifierNode(constant_name)} if constants else {}
            ),
            coordinate=coordinate,
        )
        stage.local_variables = declarations + list(stage.local_variables or [])

        vertex = self.fresh("ds_vertex")
        body = prologue + [
            A.VariableNode(
                name=vertex,
                var_type=copy.deepcopy(entry.return_type),
                initial_value=domain_point,
            )
        ]
        body += [write(A.IdentifierNode(vertex)) for write in output_writes]
        attributes = list(identity.attributes)
        present = {_attr_name(attr) for attr in attributes}
        hull_attributes = list(hull.attributes or []) if hull is not None else []
        for name in _DOMAIN_FROM_HULL:
            if name in present:
                continue
            taken = [a for a in hull_attributes if _attr_name(a) == name]
            if taken:
                attributes.append(copy.deepcopy(taken[0]))
            elif self.target in GLSL_TARGETS:
                self.refuse(
                    f"domain entry point '{identity.name}' has no hull stage in the program to take "
                    f"[{name}] from; GLSL declares spacing and winding on the evaluation shader",
                    "domain-without-hull",
                )
        self.replace_entry(stage, entry, identity, body, attributes)

    def domain_interface(self, hull, point_struct, patch_struct):
        if hull is not None:
            return hull.interface
        interface = _Interface(self.refuse)
        if point_struct is not None:
            interface.place(
                interface.points, point_struct, self.user_point_members(point_struct), 0
            )
        elif patch_struct is not None and self.user_patch_members(patch_struct):
            self.refuse(
                "the domain stage reads patch constants but no control points, so it cannot place them after "
                "the hull's control-point outputs; translate the hull and domain stages together",
                "patch-locations",
            )
        if patch_struct is not None:
            interface.place(
                interface.patch,
                patch_struct,
                self.user_patch_members(patch_struct),
                interface.end,
            )
        return interface

    def linked_location(self, table, struct, member, stage):
        key = _link_key(member)
        if key not in table:
            self.refuse(
                f"domain input '{struct.name}.{member.name}' ({semantic(member) or 'no semantic'}) is not "
                f"written by the hull stage's {stage}",
                "unlinked-domain-input",
            )
        location, type_key = table[key]
        if type_key != _type_key(member_type(member)):
            self.refuse(
                f"domain input '{struct.name}.{member.name}' does not have the type of the hull's {stage} "
                "of the same semantic",
                "domain-input-type",
            )
        return location

    def point_inputs(self, struct, interface, declarations):
        reads = {}
        for member in struct.members:
            name = builtin(member)
            if name in _POINT_BUILTINS:
                reads[member.name] = per_vertex_builtin("gl_in", name)
                continue
            self.refuse_value_member(struct, member)
            location = self.linked_location(
                interface.points, struct, member, "control points"
            )
            variable = varying_declaration(
                self.fresh(f"te_in_{member.name}"),
                member,
                A.ArrayType(copy.deepcopy(member_type(member)), None),
                location,
                ["in"],
            )
            declarations.append(variable)
            reads[member.name] = indexed(variable.name)
        return reads

    def patch_inputs(self, struct, interface, name, declarations):
        statements = [A.VariableNode(name=name, var_type=A.NamedType(struct.name))]
        target = A.IdentifierNode(name)
        for member in struct.members:
            level = builtin(member)
            if level in _LEVELS:
                statements += [
                    copy_(target)
                    for copy_ in self.level_copies(struct, member, level, write=False)
                ]
                continue
            location = self.linked_location(
                interface.patch, struct, member, "patch constants"
            )
            variable = varying_declaration(
                self.fresh(f"te_patch_{member.name}"),
                member,
                copy.deepcopy(member_type(member)),
                location,
                ["patch", "in"],
            )
            declarations.append(variable)
            statements.append(
                A.AssignmentNode(
                    A.MemberAccessNode(A.IdentifierNode(name), member.name),
                    A.IdentifierNode(variable.name),
                )
            )
        return statements

    def domain_outputs(self, struct, declarations):
        writes, running = [], 0
        for member in struct.members:
            name = builtin(member)
            if name in _POINT_BUILTINS:
                writes.append(_member_copy(name, member.name))
            else:
                self.refuse_value_member(struct, member)
                location = neighbour_location(
                    self.target, struct.name, member, running, self.refuse
                )
                variable = varying_declaration(
                    self.fresh(f"te_out_{member.name}"),
                    member,
                    copy.deepcopy(member_type(member)),
                    location,
                    ["out"],
                )
                declarations.append(variable)
                writes.append(_member_copy(variable.name, member.name))
            running += location_span(member_type(member))
        return writes

    # Shared.

    def user_point_members(self, struct):
        members = []
        for member in struct.members:
            if builtin(member) in _POINT_BUILTINS:
                continue
            self.refuse_value_member(struct, member)
            members.append(member)
        return members

    def user_patch_members(self, struct):
        members = []
        for member in struct.members:
            if builtin(member) in _LEVELS:
                continue
            self.refuse_value_member(struct, member, patch=True)
            members.append(member)
        return members

    def refuse_value_member(self, struct, member, *, patch=False):
        found = builtin(member) or (
            semantic(member) if is_system_value(member) else None
        )
        if found is not None:
            self.refuse(
                f"member '{struct.name}.{member.name}' carries {found}, which the tessellation lowering does "
                "not carry yet",
                "system-value",
            )
        member_kind = member_type(member)
        if (
            patch
            and isinstance(member_kind, A.ArrayType)
            and literal_int(member_kind.size) is not None
        ):
            # A patch variable is not arrayed per vertex, so a sized array of values is one varying.
            member_kind = member_kind.element_type
        if isinstance(member_kind, A.ArrayType) or type_name(
            member_kind
        ) in self.structs(self.stage):
            # A per-vertex array member needs an array of arrays, which the stage-array sizing and the SPIR-V
            # emitter do not build yet.
            self.refuse(
                f"member '{struct.name}.{member.name}' is an array or structure",
                "aggregate-member",
            )

    def level_copies(self, struct, member, level, *, write):
        """``patch -> statement`` copies between a level member and ``gl_TessLevelOuter``/``gl_TessLevelInner``."""
        kind = member_type(member)
        count = 1
        if isinstance(kind, A.ArrayType):
            count = literal_int(kind.size)
            kind = kind.element_type
        if count is None or not 1 <= count <= _LEVELS[level]:
            self.refuse(
                f"tessellation level member '{struct.name}.{member.name}' has more elements than {level}",
                "level-size",
            )
        element = type_name(kind)
        if element not in {"float", "half"}:
            self.refuse(
                f"tessellation level member '{struct.name}.{member.name}' is not a float",
                "level-type",
            )
        array = isinstance(member_type(member), A.ArrayType)
        copies = []
        for index in range(count):

            def statement(patch, index=index):
                value = A.MemberAccessNode(patch, member.name)
                if array:
                    value = A.ArrayAccessNode(value, int_literal(index))
                level_element = A.ArrayAccessNode(
                    A.IdentifierNode(level), int_literal(index)
                )
                if write:
                    if element != "float":
                        value = A.FunctionCallNode(A.IdentifierNode("float"), [value])
                    return A.AssignmentNode(level_element, value)
                source = level_element
                if element != "float":
                    source = A.FunctionCallNode(A.IdentifierNode(element), [source])
                return A.AssignmentNode(value, source)

            copies.append(statement)
        return copies

    def patch_shape(self, parameter, structs):
        args = getattr(parameter.param_type, "generic_args", None) or []
        element = structs.get(type_name(args[0])) if len(args) == 2 else None
        count = literal_int(args[1]) if len(args) == 2 else None
        if element is None or count is None or count < 1:
            self.refuse(
                f"patch parameter '{parameter.name}' must be {type_name(parameter.param_type)}<structure, N>",
                "patch-shape",
            )
        return element, count

    def detach(
        self, function, rename, system_values, patches, constants=None, coordinate=None
    ):
        """Turn ``function`` into an ordinary stage function and return the call the new entry point makes.

        Patch parameters are removed and their reads rewritten (``patches`` maps the patch kind to a function
        returning ``(element struct, count, sources)``); system-value parameters stay, without their semantic,
        and the call passes the GLSL built-in.
        """
        constants = constants or {}
        kept, arguments, prologue = [], [], []
        for parameter in function.parameters or []:
            kind = type_name(parameter.param_type)
            if kind in _PATCH_TYPES:
                if kind not in patches:
                    self.refuse(
                        f"'{function.name}' cannot take {kind} parameter '{parameter.name}'",
                        "patch-parameter",
                    )
                element, count, sources = patches[kind](parameter)
                prologue += rewrite_array_reads(
                    function,
                    parameter.name,
                    A.NamedType(element.name),
                    count,
                    sources,
                    self.refuse,
                )
                continue
            if id(parameter) in constants:
                arguments.append(constants[id(parameter)])
            else:
                arguments.append(
                    self.system_value_argument(
                        function, parameter, system_values, coordinate
                    )
                )
            kept.append(
                A.ParameterNode(
                    name=parameter.name,
                    param_type=parameter.param_type,
                    qualifiers=[
                        q for q in parameter.qualifiers or [] if str(q) != "const"
                    ],
                )
            )
        function.parameters = kept
        function.body.statements[0:0] = prologue
        if rename is not None:
            # The entry point's attributes move to the new entry point.
            function.attributes = []
            function.name = self.fresh(rename)
            function.preserve_stage_entry_name = False
        return A.FunctionCallNode(A.IdentifierNode(function.name), arguments)

    def system_value_argument(self, function, parameter, system_values, coordinate):
        found = builtin(parameter)
        kind = parameter.param_type
        if found == "gl_TessCoord" and "gl_TessCoord" in system_values:
            size = (
                getattr(kind, "size", None) if isinstance(kind, A.VectorType) else None
            )
            if size == 3:
                return coordinate
            if size == 2:
                return A.MemberAccessNode(coordinate, "xy")
            self.refuse(
                f"domain location parameter '{parameter.name}' is not a float2 or float3",
                "domain-location",
            )
        if found not in system_values:
            self.refuse(
                f"'{function.name}' parameter '{parameter.name}' ({found or semantic(parameter) or 'no system value'}) "
                "is not a patch or a system value the GLSL tessellation stages provide",
                "parameter",
            )
        if type_name(kind) not in {"int", "uint"}:
            self.refuse(
                f"system-value parameter '{parameter.name}' is not an int or uint",
                "parameter-type",
            )
        value = A.IdentifierNode(system_values[found])
        if type_name(kind) == "uint":
            value = A.FunctionCallNode(A.IdentifierNode("uint"), [value])
        return value

    def replace_entry(self, stage, entry, identity, body, attributes):
        """Make ``entry`` (already detached) a stage function and give the stage a parameterless entry point."""
        wrapper = A.FunctionNode(
            name=identity.name,
            return_type=A.PrimitiveType("void"),
            parameters=[],
            body=A.BlockNode(body),
            attributes=attributes,
            qualifiers=identity.qualifiers,
        )
        if identity.preserve_name:
            wrapper.preserve_stage_entry_name = True
        stage.entry_point = wrapper
        stage.local_functions = [entry] + list(stage.local_functions or [])

    def functions(self):
        found = list(getattr(self.program, "functions", None) or [])
        for stage in stages(self.program):
            found += list(getattr(stage, "local_functions", None) or [])
            entry = getattr(stage, "entry_point", None)
            if entry is not None:
                found.append(entry)
        return found

    def refuse_calls(self, name):
        for function in self.functions():
            for _, _, _, node in slots(function.body):
                if isinstance(node, A.FunctionCallNode) and call_name(node) == name:
                    self.refuse(
                        f"'{function.name}' calls '{name}', which becomes part of the tessellation entry point",
                        "called-stage-function",
                    )

    def patch_helpers(self, stage):
        """Helpers that take a patch take the array of control points the lowered stage copies it into."""
        self.stage = stage
        structs = self.structs(stage)
        entries = {id(getattr(s, "entry_point", None)) for s in stages(self.program)}
        named = {
            _attribute_text(entry, "patchconstantfunc")
            for entry in (getattr(s, "entry_point", None) for s in stages(self.program))
            if entry is not None
        }
        for function in list(getattr(self.program, "functions", None) or []) + list(
            stage.local_functions or []
        ):
            if id(function) in entries or str(function.name).lower() in named:
                continue
            for parameter in function.parameters or []:
                if type_name(parameter.param_type) in _PATCH_TYPES:
                    element, count = self.patch_shape(parameter, structs)
                    parameter.param_type = A.ArrayType(
                        A.NamedType(element.name), int_literal(count)
                    )

    def move_into_stage(self, function):
        functions = getattr(self.program, "functions", None) or []
        if function in functions:
            functions.remove(function)
            self.stage.local_functions = list(self.stage.local_functions or []) + [
                function
            ]


# Helpers.


class _EntryIdentity:
    """What the new entry point takes over from the HLSL entry point it replaces."""

    def __init__(self, entry):
        self.name = entry.name
        self.attributes = list(entry.attributes or [])
        self.qualifiers = list(entry.qualifiers or [])
        self.preserve_name = bool(getattr(entry, "preserve_stage_entry_name", False))


class _Hull:
    """What the domain stage needs from the program's lowered hull stage."""

    def __init__(self, attributes, interface, points):
        self.attributes = attributes
        self.interface = interface
        self.points = points


def _returns_struct(program, stage, entry):
    name = type_name(getattr(entry, "return_type", None))
    structs = list(getattr(program, "structs", None) or []) + list(
        getattr(stage, "local_structs", None) or []
    )
    return any(isinstance(s, A.StructNode) and s.name == name for s in structs)


def _attr_name(attr):
    return str(getattr(attr, "name", "")).lower()


def _attribute_args(function, name):
    found = [attr for attr in function.attributes or [] if _attr_name(attr) == name]
    return (found[0].arguments or []) if len(found) == 1 else None


def _attribute_int(function, name):
    args = _attribute_args(function, name)
    return literal_int(args[0]) if args and len(args) == 1 else None


def _attribute_text(function, name):
    args = _attribute_args(function, name)
    if not args or len(args) != 1:
        return None
    return str(getattr(args[0], "value", args[0])).strip().strip("\"'").lower()


def _link_key(member):
    """HLSL links stages by semantic, which is case-insensitive; a member without one links by name."""
    found = semantic(member)
    return ("semantic", found.upper()) if found is not None else ("member", member.name)


def _type_key(type_node):
    return repr(type_node)


def _member_copy(target_name, member_name):
    """``value -> target = value.member``."""

    def statement(value):
        return A.AssignmentNode(
            A.IdentifierNode(target_name), A.MemberAccessNode(value, member_name)
        )

    return statement
