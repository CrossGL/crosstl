"""GLSL-form tessellation evaluation stages rewritten into the patch form a Metal post-tessellation function takes.

The GLSL and SPIR-V importers write a tessellation evaluation stage as GLSL declares one (GLSL 4.60 sections 4.3.4,
4.4.1.2 and 7.1.3): the domain, spacing and winding in a stage ``layout(...) in``, each control-point value in a
per-vertex input array (``in vec4 position @location(0)[];``, or ``float4 position[32] @input @location(0);``), the
per-patch values in ``patch in`` variables, and the position in the patch, the patch index and the outputs in built-in
variables of a ``void main()``::

    in vec4 position @location(0)[];
    patch in vec4 extra @location(1);
    tessellation_evaluation {
        layout(triangles, equal_spacing, ccw) in;
        void main() { gl_Position = position[0] * gl_TessCoord.x + extra * gl_TessCoord.y; }
    }

:func:`rewrite_glsl_evaluation_entry` rewrites such an entry into the form the HLSL importer writes for a domain
shader, which :mod:`.metal_post_tessellation` writes as a ``[[patch]]`` vertex function (Metal Shading Language
Specification 4.1, section 5.1.5)::

    struct main_TessellationPoint { vec4 position @location(0); }
    struct main_TessellationPatch { vec4 extra @location(1); }
    struct main_TessellationOutput { vec4 gl_Position @gl_Position; }
    tessellation_evaluation {
        @domain("tri")
        main_TessellationOutput main(main_TessellationPatch _crossglConstants, vec3 _crossglTessCoord @gl_TessCoord,
                                     OutputPatch<main_TessellationPoint, 32> _crossglPoints) { ... }
    }

* ``triangles`` and ``quads`` become ``@domain("tri")`` and ``@domain("quad")``; Metal has no isoline patches and its
  tessellator writes no points, so ``isolines`` and ``point_mode`` are refused. The spacing and the winding are
  dropped, as ``@partitioning`` and ``@outputtopology`` are: Metal sets them in the render pipeline descriptor;
* each per-vertex input array becomes a member of the control-point structure, keeping its location, and
  ``input[i]`` reads ``_crossglPoints[i].input``; ``gl_in[i].gl_Position`` reads a ``gl_Position`` member too;
* each ``patch in`` variable becomes a member of the patch-constant structure, keeping its location;
* ``gl_TessCoord`` reads the ``[[position_in_patch]]`` parameter, which is a ``float2`` for a quad patch, so a quad
  entry reads ``vec3(coord, 0.0)``, as GLSL defines its third component; ``gl_PrimitiveID`` reads the
  ``[[patch_id]]`` parameter as an ``int``;
* the outputs become members of an output structure that the entry declares, writes and returns from every
  ``return``. Built-in outputs the stage never names, such as the rest of a SPIR-V ``gl_PerVertex`` block, are
  dropped.

GLSL fixes the number of control points in the tessellation control stage, ``layout(vertices = N) out``, not in the
evaluation stage. When the program has that stage, the patch has N points; otherwise the patch's size is the
draw's, so the function's ``[[patch]]`` attribute gives no count and the entry copies at most
``gl_MaxPatchVertices`` (32, as in Metal) points. ``gl_PatchVerticesIn`` is N, and is refused without it.

Reads of ``gl_TessLevelOuter`` and ``gl_TessLevelInner`` are refused: Metal gives the tessellation factors to the
tessellator in a buffer the host binds, not to the post-tessellation function. So are a per-vertex input used other
than through a subscript, a helper function that reads or writes a stage variable, and stage variables of other
forms. The entry and the program are rewritten in place: the caller passes a copy.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AttributeNode,
    BlockNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    PrimitiveType,
    ReturnNode,
    StructMemberNode,
    StructNode,
    VariableNode,
    VectorType,
)
from ..semantics.rewrite import child_slots, get_child, set_child
from ..stage_utils import normalize_stage_name
from .metal_post_tessellation import (
    OPEN_PATCH_ANNOTATION,
    MetalPostTessellationError,
)
from .stage_utils import stage_layout_entries, stage_layout_entry_value

__all__ = ["MAX_PATCH_POINTS", "rewrite_glsl_evaluation_entry"]

# gl_MaxPatchVertices; Metal's patches hold at most 32 control points too.
MAX_PATCH_POINTS = 32
_DOMAINS = {"triangles": "tri", "quads": "quad", "isolines": "isoline"}
_PIPELINE_ENTRIES = frozenset(
    {
        "triangles",
        "quads",
        "isolines",
        "equal_spacing",
        "fractional_even_spacing",
        "fractional_odd_spacing",
        "cw",
        "ccw",
    }
)
_LEVELS = frozenset({"gl_TessLevelOuter", "gl_TessLevelInner"})
# Built-in outputs a post-tessellation vertex function can write, with the CrossGL type GLSL gives them.
_BUILTIN_OUTPUTS = {"gl_Position": ("float", 4), "gl_PointSize": ("float", 1)}
# The gl_in[] members that a control point can carry as a vertex attribute.
_POINT_BUILTINS = {"gl_Position": ("float", 4), "gl_PointSize": ("float", 1)}
_STAGE_BUILTIN_INPUTS = frozenset(
    {"gl_TessCoord", "gl_PrimitiveID", "gl_PatchVerticesIn", "gl_in", *_LEVELS}
)
# Qualifiers of a GLSL-form output that are not interpolation attributes of the output structure's member.
_PLAIN_QUALIFIERS = frozenset({"out", "highp", "mediump", "lowp"})


def _float_type(size):
    scalar = PrimitiveType("float")
    return scalar if size == 1 else VectorType(scalar, size)


def _attribute_names(node):
    return [str(getattr(attribute, "name", "")) for attribute in node.attributes or []]


def _location(node):
    for attribute in node.attributes or []:
        if str(getattr(attribute, "name", "")).lower() == "location":
            return attribute
    return None


class _StageVariable:
    """One interface variable of the stage: a ``patch``/``in``/``out`` global or an ``@input``/``@output`` local."""

    def __init__(self, node, direction, patch, builtin):
        self.node = node
        self.name = node.name
        self.direction = direction
        self.patch = patch
        self.builtin = builtin


def _classify(node, *, stage_level):
    if stage_level:
        names = _attribute_names(node)
        direction = "in" if "input" in names else "out" if "output" in names else None
        builtins = [name for name in names if name.startswith("gl_")]
        patch = "patch" in names
    else:
        qualifiers = [str(qualifier) for qualifier in node.qualifiers or []]
        direction = (
            "in" if "in" in qualifiers else "out" if "out" in qualifiers else None
        )
        builtins = [node.name] if node.name.startswith("gl_") else []
        patch = "patch" in qualifiers
    if direction is None:
        return None
    return _StageVariable(node, direction, patch, builtins[0] if builtins else None)


def _declared_names(ast):
    names = set()
    for node in ast.walk():
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _control_point_count(ast):
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        if normalize_stage_name(stage_type) != "tessellation_control":
            continue
        value = stage_layout_entry_value(stage, "vertices", "out")
        value = getattr(value, "value", value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit() and int(value) > 0:
            return int(value)
    return None


class _GlslEvaluationRewrite:
    def __init__(self, ast, stage, function):
        self.ast = ast
        self.stage = stage
        self.function = function
        self.entry = getattr(function, "name", None) or "main"
        self.taken = _declared_names(ast)
        self.points = {}
        self.constants = {}
        self.outputs = {}
        self.builtin_inputs = {}
        self.used = set()
        self.returns = []
        self.named_outputs = set()

    def refuse(self, reason):
        raise MetalPostTessellationError(self.entry, reason)

    def fresh(self, base):
        candidate, index = base, 1
        while candidate in self.taken:
            candidate = f"{base}_{index}"
            index += 1
        self.taken.add(candidate)
        return candidate

    def domain(self):
        domain = None
        for entry in stage_layout_entries(self.stage, "in"):
            name = str(getattr(entry, "name", "")).lower()
            if name == "point_mode":
                self.refuse(
                    "it asks for point_mode, and Metal's tessellator writes only triangles"
                )
            if name in _DOMAINS:
                domain = _DOMAINS[name]
        if domain == "isoline":
            self.refuse(
                "Metal tessellates only triangle and quad patches, not isolines"
            )
        for layout in getattr(self.stage, "layout_qualifiers", None) or []:
            layout.entries = [
                entry
                for entry in layout.entries or []
                if str(getattr(entry, "name", "")).lower() not in _PIPELINE_ENTRIES
            ]
        self.stage.layout_qualifiers = [
            layout for layout in self.stage.layout_qualifiers or [] if layout.entries
        ]
        return domain

    def gather(self):
        """Take the stage's interface variables out of the program and the stage."""
        program_scope, kept_globals = [], []
        for node in getattr(self.ast, "global_variables", None) or []:
            variable = (
                _classify(node, stage_level=False)
                if isinstance(node, VariableNode)
                else None
            )
            if variable is None:
                kept_globals.append(node)
            else:
                program_scope.append(variable)
        stage_level, kept_locals = [], []
        for node in getattr(self.stage, "local_variables", None) or []:
            variable = (
                _classify(node, stage_level=True)
                if isinstance(node, VariableNode)
                else None
            )
            if variable is None:
                kept_locals.append(node)
            else:
                stage_level.append(variable)
        if program_scope and self.other_stages():
            self.refuse(
                "the program declares program-scope stage inputs or outputs and has other stages, so they do "
                "not name this stage's interface"
            )
        self.ast.global_variables = kept_globals
        self.stage.local_variables = kept_locals
        for variable in program_scope + stage_level:
            self.add(variable)

    def other_stages(self):
        return [
            stage_type
            for stage_type, stage in (getattr(self.ast, "stages", None) or {}).items()
            if stage is not self.stage
            and normalize_stage_name(stage_type) != "tessellation_control"
        ]

    def add(self, variable):
        node = variable.node
        if variable.direction == "out":
            if variable.patch:
                self.refuse(
                    f"it writes the patch output '{variable.name}', and an evaluation stage has none"
                )
            self.outputs[variable.name] = (node.var_type, variable.builtin, node)
            return
        if variable.builtin is not None:
            if variable.builtin not in _STAGE_BUILTIN_INPUTS:
                self.refuse(
                    f"it reads '{variable.builtin}', which a Metal post-tessellation function does not receive"
                )
            self.builtin_inputs[variable.name] = (variable.builtin, node.var_type)
            return
        if variable.patch:
            if isinstance(node.var_type, ArrayType):
                self.refuse(
                    f"the patch input '{variable.name}' is an array, and a Metal patch attribute is a scalar "
                    "or a vector"
                )
            self.constants[variable.name] = StructMemberNode(
                variable.name,
                node.var_type,
                attributes=[_location(node)] if _location(node) else [],
            )
            return
        if not isinstance(node.var_type, ArrayType):
            self.refuse(
                f"the per-vertex input '{variable.name}' is not an array of the patch's control points"
            )
        location = _location(node)
        self.points[variable.name] = StructMemberNode(
            variable.name,
            node.var_type.element_type,
            attributes=[location] if location else [],
        )

    def check_helpers(self):
        names = (
            set(self.points)
            | set(self.constants)
            | set(self.outputs)
            | set(self.builtin_inputs)
            | _STAGE_BUILTIN_INPUTS
            | set(_BUILTIN_OUTPUTS)
        )
        functions = list(getattr(self.ast, "functions", None) or [])
        for stage in (getattr(self.ast, "stages", None) or {}).values():
            functions.extend(getattr(stage, "local_functions", None) or [])
        for function in functions:
            if function is self.function or function.body is None:
                continue
            for node in function.body.walk():
                if isinstance(node, IdentifierNode) and node.name in names:
                    self.refuse(
                        f"the helper '{function.name}' reads or writes the stage variable '{node.name}'; pass "
                        "it as a parameter"
                    )

    def run(self):
        domain = self.domain()
        if domain is None:
            self.refuse("its layout gives no triangles or quads domain")
        self.gather()
        self.check_helpers()
        count = _control_point_count(self.ast)
        self.count = count
        self.names = {
            "points": self.fresh("_crossglPoints"),
            "constants": self.fresh("_crossglConstants"),
            "output": self.fresh("_crossglOutput"),
            "coord": self.fresh("_crossglTessCoord"),
            "patch_id": self.fresh("_crossglPatchId"),
        }
        self.point_builtins = {}
        self.quad = domain == "quad"
        body = self.function.body or BlockNode([])
        self.function.body = self.rewrite(body)
        self.function.body = self.with_interface(self.function.body)
        self.function.attributes = list(self.function.attributes or []) + [
            AttributeNode("domain", [IdentifierNode(domain)])
        ]
        if count is None and self.points_struct_members():
            self.function.add_annotation(OPEN_PATCH_ANNOTATION, True)
        return self.declare_structs()

    # Body rewriting.

    def builtin_input(self, name):
        if name in self.builtin_inputs:
            return self.builtin_inputs[name][0]
        return name if name in _STAGE_BUILTIN_INPUTS else None

    def output_name(self, name):
        if name not in self.outputs and name in _BUILTIN_OUTPUTS:
            self.outputs[name] = (_float_type(_BUILTIN_OUTPUTS[name][1]), name, None)
        if name not in self.outputs:
            return None
        self.named_outputs.add(name)
        return name

    def kept_outputs(self):
        """The outputs the structure holds: every user output, and the built-in ones the entry names."""
        return [
            (name, value)
            for name, value in self.outputs.items()
            if value[1] is None or name in self.named_outputs
        ]

    def rewrite(self, node):
        replacement = self.replace(node)
        if replacement is not None:
            return replacement
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            new = self.rewrite(child)
            if new is not child:
                set_child(node, field, key, new)
        return node

    def point_read(self, index, member):
        return MemberAccessNode(
            ArrayAccessNode(IdentifierNode(self.names["points"]), self.rewrite(index)),
            member,
        )

    def replace(self, node):
        if isinstance(node, VariableNode) and node.name in (
            set(self.points) | set(self.constants) | set(self.outputs)
        ):
            self.refuse(
                f"it declares a local '{node.name}' that hides a stage variable"
            )
        if isinstance(node, ReturnNode):
            if node.value is not None:
                self.refuse("its main function returns a value")
            returned = ReturnNode(IdentifierNode(self.names["output"]))
            self.returns.append(returned)
            return returned
        if isinstance(node, MemberAccessNode):
            array = node.object_expr
            if (
                isinstance(array, ArrayAccessNode)
                and isinstance(array.array_expr, IdentifierNode)
                and self.builtin_input(array.array_expr.name) == "gl_in"
            ):
                if node.member not in _POINT_BUILTINS:
                    self.refuse(
                        f"it reads gl_in[].{node.member}, which a Metal control point does not carry"
                    )
                self.point_builtins[node.member] = _float_type(
                    _POINT_BUILTINS[node.member][1]
                )
                return self.point_read(array.index_expr, node.member)
        if isinstance(node, ArrayAccessNode) and isinstance(
            node.array_expr, IdentifierNode
        ):
            if node.array_expr.name in self.points:
                return self.point_read(node.index_expr, node.array_expr.name)
        if not isinstance(node, IdentifierNode):
            return None
        name = node.name
        if name in self.points:
            self.refuse(
                f"it uses the per-vertex input '{name}' other than through a subscript"
            )
        if name in self.constants:
            self.used.add("constants")
            return MemberAccessNode(IdentifierNode(self.names["constants"]), name)
        output = self.output_name(name)
        if output is not None:
            self.used.add("output")
            return MemberAccessNode(IdentifierNode(self.names["output"]), output)
        builtin = self.builtin_input(name)
        if builtin is None:
            return None
        if builtin in _LEVELS:
            self.refuse(
                f"it reads {builtin}, which Metal passes to the tessellator in a host-bound buffer, not to the "
                "post-tessellation function"
            )
        if builtin == "gl_in":
            self.refuse("it uses gl_in other than as gl_in[i].gl_Position")
        if builtin == "gl_PatchVerticesIn":
            if self.count is None:
                self.refuse(
                    "it reads gl_PatchVerticesIn, and the program has no tessellation control stage that "
                    "fixes the patch size"
                )
            return LiteralNode(self.count, PrimitiveType("int"))
        if builtin == "gl_TessCoord":
            self.used.add("coord")
            return IdentifierNode(self.coord_value())
        self.used.add("patch_id")
        return IdentifierNode(self.patch_id_value())

    def coord_value(self):
        if not self.quad:
            return self.names["coord"]
        if "coord3" not in self.names:
            self.names["coord3"] = self.fresh("_crossglTessCoord3")
        return self.names["coord3"]

    def patch_id_value(self):
        if "patch_id_int" not in self.names:
            self.names["patch_id_int"] = self.fresh("_crossglPrimitiveId")
        return self.names["patch_id_int"]

    # The entry's new interface.

    def points_struct_members(self):
        members = list(self.points.values())
        members.extend(
            StructMemberNode(name, type_node)
            for name, type_node in sorted(self.point_builtins.items())
        )
        return members

    def with_interface(self, body):
        statements = []
        parameters = list(self.function.parameters or [])
        if parameters:
            self.refuse("its main function takes parameters")
        output_type = None
        if self.kept_outputs():
            output_type = NamedType(self.fresh(f"{self.entry}_TessellationOutput"))
            statements.append(VariableNode(self.names["output"], output_type))
        if self.constants and "constants" in self.used:
            self.patch_type = NamedType(self.fresh(f"{self.entry}_TessellationPatch"))
            parameters.append(ParameterNode(self.names["constants"], self.patch_type))
        if "coord" in self.used:
            parameters.append(
                ParameterNode(
                    self.names["coord"],
                    _float_type(2 if self.quad else 3),
                    attributes=[AttributeNode("gl_TessCoord")],
                )
            )
            if self.quad:
                statements.append(
                    VariableNode(
                        self.names["coord3"],
                        _float_type(3),
                        FunctionCallNode(
                            IdentifierNode("vec3"),
                            [
                                IdentifierNode(self.names["coord"]),
                                LiteralNode(0.0, PrimitiveType("float")),
                            ],
                        ),
                    )
                )
        point_members = self.points_struct_members()
        if point_members:
            self.point_type = NamedType(self.fresh(f"{self.entry}_TessellationPoint"))
            size = self.count if self.count is not None else MAX_PATCH_POINTS
            parameters.append(
                ParameterNode(
                    self.names["points"],
                    NamedType(
                        "OutputPatch",
                        [self.point_type, LiteralNode(size, PrimitiveType("int"))],
                    ),
                )
            )
        if "patch_id" in self.used:
            parameters.append(
                ParameterNode(
                    self.names["patch_id"],
                    PrimitiveType("uint"),
                    attributes=[AttributeNode("gl_PrimitiveID")],
                )
            )
            statements.append(
                VariableNode(
                    self.names["patch_id_int"],
                    PrimitiveType("int"),
                    FunctionCallNode(
                        IdentifierNode("int"), [IdentifierNode(self.names["patch_id"])]
                    ),
                )
            )
        statements.extend(body.statements or [])
        if output_type is not None:
            if not statements or not isinstance(statements[-1], ReturnNode):
                statements.append(ReturnNode(IdentifierNode(self.names["output"])))
            self.function.return_type = output_type
        else:
            for returned in self.returns:
                returned.value = None
        self.output_type = output_type
        self.function.parameters = parameters
        body.statements = statements
        return body

    def declare_structs(self):
        structs = []
        point_members = self.points_struct_members()
        if point_members:
            structs.append(StructNode(self.point_type.name, point_members))
        if self.constants and "constants" in self.used:
            structs.append(
                StructNode(self.patch_type.name, list(self.constants.values()))
            )
        if self.output_type is not None:
            members = []
            for name, (type_node, builtin, node) in self.kept_outputs():
                attributes = []
                if builtin is not None:
                    attributes.append(AttributeNode(builtin))
                elif node is not None:
                    # A GLSL-form output's interpolation is a qualifier, a member's an attribute.
                    attributes.extend(
                        AttributeNode(str(qualifier))
                        for qualifier in node.qualifiers or []
                        if str(qualifier) not in _PLAIN_QUALIFIERS
                    )
                    attributes.extend(
                        attribute
                        for attribute in node.attributes or []
                        if str(getattr(attribute, "name", "")) != "output"
                    )
                members.append(StructMemberNode(name, type_node, attributes=attributes))
            structs.append(StructNode(self.output_type.name, members))
        return structs


def rewrite_glsl_evaluation_entry(ast, stage, function):
    """Rewrite the GLSL-form evaluation ``function`` of ``stage`` into a domain entry; returns the new structures.

    The structures are appended to ``ast.structs`` too. Raises :class:`MetalPostTessellationError` for a stage Metal
    cannot write.
    """
    structs = _GlslEvaluationRewrite(ast, stage, function).run()
    ast.structs = list(getattr(ast, "structs", None) or []) + structs
    return structs
