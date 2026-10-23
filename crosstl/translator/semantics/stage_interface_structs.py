"""Stage-level interface variables gathered into the entry point's input and output structures.

The SPIR-V importer declares a vertex or fragment stage's interface the way SPIR-V does: one stage-level variable
per input or output (``vec2 uv @input @location(0);``), read and written by a ``void main()``. MSL and HLSL have
no stage-level interface variables. An entry point receives its inputs as a structure parameter (``[[stage_in]]``
or semantics) and returns its outputs as a structure, which is the form the GLSL importer produces and both
emitters lower. The Metal emitter declared the stage-level variables as uninitialized locals of a ``void``
entry point, which compiles and reads and writes nothing; the HLSL emitter left fragment ones undeclared.

:func:`gather_stage_interface_variables` rewrites a vertex or fragment stage for those targets:

- user inputs become members of a ``VertexInput``/``FragmentInput`` structure that the entry point takes as
  ``input``; reads anywhere in the stage's functions become ``input.<name>``;
- outputs, built-in ones included, become members of a ``VertexOutput``/``FragmentOutput`` structure. The entry
  point declares ``output``, returns it from every ``return``, and passes it ``inout`` to every helper that writes
  or reads an output, directly or through the helpers it calls;
- built-in inputs (``gl_FragCoord``, ``gl_VertexID``, ...) become entry-point parameters carrying the built-in,
  passed on to the helpers that read them; built-in outputs that no function uses, such as the members of a
  SPIR-V ``gl_PerVertex`` block, are dropped.

Members keep their location, interpolation and built-in attributes; a fragment output's location names its colour
attachment, ``gl_FragData[N]``, which both emitters map to ``SV_TargetN`` or ``[[color(N)]]``. Built-in inputs that
GLSL types ``int`` but HLSL and MSL deliver as ``uint`` (vertex, instance, primitive and sample indices) arrive in a
``uint`` parameter and are converted once at the start of the entry point. An entry point that already takes
parameters keeps them when the stage-level variables are only outputs and built-ins. A stage whose entry point
returns a value besides stage-level variables, or takes parameters besides stage-level user inputs, or whose
functions declare a name that is also an interface variable, is refused with :class:`StageInterfaceUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_values import glsl_builtin_name
from .hub_types import call_name
from .rewrite import child_slots, get_child, set_child

__all__ = ["StageInterfaceUnsupportedError", "gather_stage_interface_variables"]

_TARGETS = frozenset({"metal", "directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_STRUCT_NAMES = {
    "vertex": ("VertexInput", "VertexOutput"),
    "fragment": ("FragmentInput", "FragmentOutput"),
}
_DIRECTION_ATTRIBUTES = frozenset({"input", "output"})
# Built-in inputs HLSL and MSL deliver only as uint (HLSL SV_VertexID, SV_InstanceID, SV_PrimitiveID and
# SV_SampleIndex; MSL [[vertex_id]], [[instance_id]], [[primitive_id]] and [[sample_id]]).
_UNSIGNED_BUILTINS = frozenset(
    {
        "gl_VertexID",
        "gl_VertexIndex",
        "gl_InstanceID",
        "gl_InstanceIndex",
        "gl_PrimitiveID",
        "gl_SampleID",
    }
)
_BUILTIN_MEMBER_NAMES = {
    "gl_Position": "position",
    "gl_PointSize": "pointSize",
    "gl_ClipDistance": "clipDistance",
    "gl_CullDistance": "cullDistance",
    "gl_FragDepth": "depth",
    "gl_SampleMask": "sampleMask",
}
_BUILTIN_BLOCKS = frozenset({"gl_PerVertex"})
_BUILTIN_BLOCK_MEMBERS = frozenset(
    {"gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_CullDistance"}
)


class StageInterfaceUnsupportedError(TranslationDiagnostic):
    """Stage-level interface variables the target's entry-point structures cannot carry."""

    def __init__(self, message, *, target=None, reason):
        self.reason_code = reason
        super().__init__(
            message, target=target, code=f"{CODE_PREFIX}stage-interface-unsupported"
        )


def gather_stage_interface_variables(program, target) -> int:
    """Rewrite stage-level interface variables into entry-point structures; returns how many stages changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in _TARGETS:
        return 0
    stages = _stages(program)
    names = _declared_names(program)
    changed = 0
    for stage in stages:
        stage_name = _stage_name(stage)
        variables = [
            variable
            for variable in getattr(stage, "local_variables", None) or []
            if _direction(variable) is not None
        ]
        if stage_name not in _STRUCT_NAMES or not variables:
            continue
        _Gatherer(program, stage, stage_name, variables, names, target).run()
        changed += 1
    return changed


class _Gatherer:
    def __init__(self, program, stage, stage_name, variables, names, target):
        self.program = program
        self.stage = stage
        self.stage_name = stage_name
        self.variables = variables
        self.names = names
        self.target = target
        self.entry = stage.entry_point
        self.helpers = [
            function
            for function in list(getattr(program, "functions", None) or []) + list(
                getattr(stage, "local_functions", None) or []
            )
            if function is not self.entry
        ]

    def run(self):
        entry = self.entry
        if entry is None or not isinstance(getattr(entry, "body", None), A.BlockNode):
            return
        # An entry point that takes its inputs as parameters keeps them when the stage-level variables are only
        # outputs and built-ins (a GLSL fragment shader with several outputs takes its varyings as a structure).
        reads_stage_inputs = any(
            _direction(variable) == "input" and _builtin_name(variable) is None
            for variable in self.variables
        )
        if not _is_void(entry.return_type) or (entry.parameters and reads_stage_inputs):
            self.refuse(
                f"the {self.stage_name} entry point '{entry.name}' takes parameters or returns a value as well as "
                "reading and writing stage-level interface variables",
                "mixed-entry-interface",
            )
        interface = {variable.name: variable for variable in self.variables}
        for function in [entry, *self.helpers]:
            shadowed = sorted(_function_declared_names(function) & set(interface))
            if shadowed:
                self.refuse(
                    f"function '{function.name}' declares '{shadowed[0]}', which is also a {self.stage_name} "
                    "interface variable",
                    "shadowed-interface-variable",
                )
        used = set()
        for function in [entry, *self.helpers]:
            used |= _referenced_names(function) & set(interface)

        inputs, builtin_inputs, outputs = [], [], []
        for variable in self.variables:
            builtin = _builtin_name(variable)
            if _direction(variable) == "input":
                (builtin_inputs if builtin else inputs).append(variable)
            elif builtin is None or variable.name in used:
                outputs.append(variable)
        self.stage.local_variables = [
            variable
            for variable in self.stage.local_variables
            if _direction(variable) is None
        ]

        input_struct_name, output_struct_name = _STRUCT_NAMES[self.stage_name]
        renames = {}
        parameters = []
        if inputs:
            input_name = self._local_name("input")
            struct_name = self._struct_name(input_struct_name)
            members = _member_names(inputs)
            self.program.structs.append(_struct(struct_name, inputs, members))
            parameters.append(A.ParameterNode(input_name, A.NamedType(struct_name)))
            for variable in inputs:
                renames[variable.name] = _member(input_name, members[variable.name])
        tokens = {}
        conversions = []
        for variable in builtin_inputs:
            if variable.name not in used:
                continue
            name = self._parameter_name(variable.name)
            builtin = _builtin_name(variable)
            if builtin in _UNSIGNED_BUILTINS and _type_name(variable.var_type) == "int":
                raw = self._parameter_name(f"{variable.name}_bits")
                parameters.append(
                    A.ParameterNode(
                        raw,
                        A.PrimitiveType("uint"),
                        attributes=[A.AttributeNode(builtin)],
                    )
                )
                conversions.append(
                    A.VariableNode(
                        name=name,
                        var_type=variable.var_type,
                        initial_value=A.FunctionCallNode(
                            A.IdentifierNode("int"), [A.IdentifierNode(raw)]
                        ),
                    )
                )
            else:
                parameters.append(
                    A.ParameterNode(
                        name, variable.var_type, attributes=[A.AttributeNode(builtin)]
                    )
                )
            tokens[variable.name] = (name, variable.var_type, None)
            renames[variable.name] = A.IdentifierNode(name)
        entry.parameters = list(entry.parameters or []) + parameters
        entry.body.statements[0:0] = conversions
        if outputs:
            output_name = self._local_name("output")
            struct_name = self._struct_name(output_struct_name)
            members = _member_names(outputs)
            self.program.structs.append(
                _struct(
                    struct_name,
                    outputs,
                    members,
                    colour_attachments=self.stage_name == "fragment",
                )
            )
            entry.return_type = A.NamedType(struct_name)
            for variable in outputs:
                renames[variable.name] = _member(output_name, members[variable.name])
                tokens[variable.name] = (output_name, A.NamedType(struct_name), "inout")
            statements = entry.body.statements
            statements.insert(
                len(conversions),
                A.VariableNode(name=output_name, var_type=A.NamedType(struct_name)),
            )
            _return_value(entry.body, output_name)
            if not statements or not isinstance(statements[-1], A.ReturnNode):
                statements.append(A.ReturnNode(A.IdentifierNode(output_name)))
        self._thread(tokens)
        for function in [entry, *self.helpers]:
            _rename(function.body, renames)
        _drop_unused_builtin_blocks(self.program)

    def _thread(self, tokens):
        """Pass built-in inputs and the output structure to the helpers that use them, directly or not.

        ``tokens`` maps an interface variable to ``(parameter, type, qualifier)``; every output maps to the one
        ``inout`` output structure.
        """
        if not tokens:
            return
        needs = {}
        for helper in self.helpers:
            direct = {
                tokens[source][0] for source in _referenced_names(helper) & set(tokens)
            }
            needs.setdefault(helper.name, set()).update(direct)
        changed = True
        while changed:
            changed = False
            for helper in self.helpers:
                inherited = set()
                for callee in _called_names(helper):
                    inherited |= needs.get(callee, set())
                if not inherited <= needs[helper.name]:
                    needs[helper.name] |= inherited
                    changed = True
        order = []
        for name, type_node, qualifier in tokens.values():
            if all(name != existing for existing, _type, _qualifier in order):
                order.append((name, type_node, qualifier))
        for helper in self.helpers:
            for name, type_node, qualifier in order:
                if name in needs[helper.name]:
                    helper.parameters = list(helper.parameters or []) + [
                        A.ParameterNode(
                            name, type_node, qualifiers=[qualifier] if qualifier else []
                        )
                    ]
        for function in [self.entry, *self.helpers]:
            for call in list(_walk(function.body)):
                if not isinstance(call, A.FunctionCallNode):
                    continue
                needed = needs.get(call_name(call))
                if needed:
                    arguments = list(call.arguments or []) + [
                        A.IdentifierNode(name)
                        for name, _type, _qualifier in order
                        if name in needed
                    ]
                    # ``args`` aliases ``arguments`` on call nodes.
                    call.arguments = call.args = arguments

    def _parameter_name(self, source):
        stem = source[3:] if source.startswith("gl_") else source
        preferred = f"cgl_{stem}"
        name, index = preferred, 0
        while name in self.names:
            index += 1
            name = f"{preferred}_{index}"
        self.names.add(name)
        return name

    def _local_name(self, preferred):
        if preferred in self.names:
            self.refuse(
                f"the {self.stage_name} stage needs '{preferred}' for its interface structure, and the program "
                "already declares that name",
                "interface-name-taken",
            )
        self.names.add(preferred)
        return preferred

    def _struct_name(self, preferred):
        name, index = preferred, 0
        while name in self.names:
            index += 1
            name = f"{preferred}{index}"
        self.names.add(name)
        return name

    def refuse(self, detail, reason):
        raise StageInterfaceUnsupportedError(
            f"{detail}; {self.target} entry points receive inputs and return outputs as structures",
            target=self.target,
            reason=reason,
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower()


def _direction(variable):
    for attribute in getattr(variable, "attributes", None) or []:
        name = str(getattr(attribute, "name", "")).lower()
        if name in _DIRECTION_ATTRIBUTES:
            return name
    return None


def _builtin_name(variable):
    """The GLSL name of the built-in ``variable`` carries, whichever spelling (``@gl_*``, ``@SV_*``, ``@builtin``)."""
    for attribute in getattr(variable, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            return name
    return None


def _is_void(type_node):
    if type_node is None:
        return True
    name = getattr(type_node, "name", type_node)
    return str(name) == "void"


def _type_name(type_node):
    return str(getattr(type_node, "name", type_node))


def _member_names(variables):
    """Structure member names for interface variables, unique within one structure.

    Built-in outputs take the names the HLSL and MSL vertex contracts use (``position``, ``pointSize``, ...); user
    variables drop the leading underscores SPIR-V producers add (``_ua_position``), keeping them when that would
    collide. Members are reached through the structure, so the names cannot clash with the program's own.
    """
    names, used = {}, set()
    for variable in variables:
        preferred = _BUILTIN_MEMBER_NAMES.get(_builtin_name(variable))
        if preferred is None:
            preferred = variable.name.lstrip("_") or variable.name
        if preferred in used or preferred.startswith("gl_"):
            preferred = variable.name
        name, index = preferred, 1
        while name in used:
            index += 1
            name = f"{preferred}{index}"
        used.add(name)
        names[variable.name] = name
    return names


def _drop_unused_builtin_blocks(program):
    """Drop SPIR-V ``gl_PerVertex``-style blocks once their members became structure members.

    The SPIR-V importer keeps the block's declaration next to the stage-level built-ins it split it into; HLSL and
    MSL have no such block, and a declaration nothing references would be emitted as a stray structure.
    """
    structs = list(getattr(program, "structs", None) or [])
    candidates = {
        struct.name
        for struct in structs
        if struct.name in _BUILTIN_BLOCKS
        and {member.name for member in struct.members or []} <= _BUILTIN_BLOCK_MEMBERS
    }
    if not candidates:
        return
    referenced = {
        node.name
        for node in _walk(program)
        if isinstance(node, A.NamedType) and node.name in candidates
    }
    unused = candidates - referenced
    if unused:
        program.structs = [struct for struct in structs if struct.name not in unused]


def _struct(name, variables, member_names, colour_attachments=False):
    members = []
    for variable in variables:
        attributes = []
        for attribute in getattr(variable, "attributes", None) or []:
            attribute_name = str(getattr(attribute, "name", "")).lower()
            if attribute_name in _DIRECTION_ATTRIBUTES:
                continue
            if colour_attachments and attribute_name == "location":
                attribute = A.AttributeNode(
                    "gl_FragData", list(attribute.arguments or [])
                )
            attributes.append(attribute)
        member = A.StructMemberNode(
            member_names[variable.name], variable.var_type, attributes=attributes
        )
        member.source_location = getattr(variable, "source_location", None)
        members.append(member)
    return A.StructNode(name, members)


def _member(local, name):
    return A.MemberAccessNode(A.IdentifierNode(local), name)


def _walk(node):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if id(current) in seen or not isinstance(current, A.ASTNode):
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )


def _referenced_names(function):
    return {
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.IdentifierNode)
    }


def _called_names(function):
    return {
        call_name(node)
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.FunctionCallNode)
    }


def _function_declared_names(function):
    names = {
        parameter.name for parameter in getattr(function, "parameters", None) or []
    }
    names.update(
        node.name
        for node in _walk(getattr(function, "body", None))
        if isinstance(node, A.VariableNode)
    )
    return names


def _rename(body, renames):
    if not renames:
        return
    for node in list(_walk(body)):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode) and child.name in renames:
                replacement = _copy(renames[child.name])
                replacement.source_location = getattr(child, "source_location", None)
                set_child(node, field, key, replacement)


def _copy(node):
    if isinstance(node, A.MemberAccessNode):
        return A.MemberAccessNode(A.IdentifierNode(node.object_expr.name), node.member)
    return A.IdentifierNode(node.name)


def _return_value(body, output_name):
    for node in _walk(body):
        if isinstance(node, A.ReturnNode) and node.value is None:
            node.value = A.IdentifierNode(output_name)


def _declared_names(program):
    names = set()
    for node in _walk(program):
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names
