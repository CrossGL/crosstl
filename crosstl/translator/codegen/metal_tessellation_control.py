"""Metal compute kernels for CrossGL tessellation control (hull) stages.

Metal has no hull stage. A tessellated draw reads its per-patch tessellation factors from a buffer the host binds with
``setTessellationFactorBuffer``, as ``MTLTriangleTessellationFactorsHalf`` or ``MTLQuadTessellationFactorsHalf``
records (Metal Shading Language Specification 4.1, section 2.18, "Per-Patch Tessellation Factors Structure"), and
the post-tessellation vertex function reads the control points and the other per-patch values as ``[[stage_in]]``
attributes (section 5.1.5). Whatever computes those buffers runs before the draw, which is a compute kernel's job.
A CrossGL tessellation control entry in the HLSL form, as the HLSL importer writes a hull shader::

    tessellation_control {
        @domain("tri") @partitioning("integer") @outputtopology("triangle_ccw") @outputcontrolpoints(3)
        @patchconstantfunc("PatchFunc")
        HSOut HSMain(InputPatch<VSOut, 3> patch, uint id @ gl_InvocationID, uint pid @ gl_PrimitiveID) { ... }
    }
    PatchOut PatchFunc(InputPatch<VSOut, 3> patch, OutputPatch<HSOut, 3> points, uint pid @ gl_PrimitiveID) { ... }

becomes a compute stage whose entry keeps the hull's name and runs one thread per patch, the patch index being
``gl_GlobalInvocationID.x``. :func:`lower_metal_tessellation_control` writes it as CrossGL, which the generator then
emits as any other kernel:

* the hull entry becomes the helper ``<entry>_control_point`` without its system-value semantics, and helpers that
  take a patch take an array of its control points;
* the kernel reads the patch's input control points from ``<entry>_input_points`` (``patch * N + i``, N being the
  ``InputPatch`` size), calls the helper for each output control point id 0 .. M-1 (``@outputcontrolpoints(M)``)
  and writes each result to ``<entry>_output_points`` (``patch * M + id``), as HLSL runs the control-point phase
  once per output control point;
* it then calls the patch-constant function, as HLSL does once per patch after every control point, with the input
  points, the output points and the patch index its parameters ask for. The ``gl_TessLevelOuter``
  (``SV_TessFactor``) and ``gl_TessLevelInner`` (``SV_InsideTessFactor``) members of the result go to the
  ``edgeTessellationFactor`` and ``insideTessellationFactor`` members of ``<entry>_tessellation_factors[patch]``,
  in the same order: Metal numbers the edges of a triangle and of a quad patch as Direct3D and GLSL do. The other
  members go, in declaration order, to the ``<entry>_PatchData`` record ``<entry>_patch_data[patch]``.

The host binds the output points as the post-tessellation draw's per-patch-control-point vertex buffer, the patch
data as its per-patch buffer and the factors as its tessellation factor buffer; :mod:`.metal_post_tessellation`
numbers the control-point members ``[[attribute(0)]]``, ``[[attribute(1)]]``, ... and the patch-constant members
after them. A vertex stage that feeds the hull has no place in a Metal tessellation pipeline, whose only vertex
function is the post-tessellation one, so its results reach the kernel through the input buffer the host fills.

``@partitioning`` and a triangle ``@outputtopology`` winding are Metal render pipeline state
(``tessellationPartitionMode``, ``tessellationOutputWindingOrder``) and ``@maxtessfactor`` is an optimisation hint the
pipeline's ``maxTessellationFactor`` carries, so the kernel drops them. Metal tessellates only triangle and quad
patches into triangles, so an isoline domain and a point or line output topology are refused with
:class:`MetalTessellationControlError`, as are entries and patch-constant functions of other shapes. A tessellation
control stage in the GLSL form, whose invocations write ``gl_out`` and synchronise with ``barrier()``, has no
patch-constant function and stays refused (:mod:`.metal_stage_refusals`). The input AST is not modified.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LayoutQualifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import normalize_stage_name
from .metal_library_structs import METAL_LIBRARY_STRUCTS, is_metal_library_struct
from .metal_post_tessellation import (
    _interface_struct_names,
    _literal_int,
    _patch_object,
    _patch_parameters_as_arrays,
    _strip_plain_semantics,
    _structs,
)

__all__ = ["MetalTessellationControlError", "lower_metal_tessellation_control"]

_STAGE = "tessellation_control"
_PATCH_TYPES = {"tri": "triangle", "triangle": "triangle", "quad": "quad"}
_FACTOR_STRUCTS = {
    "triangle": "MTLTriangleTessellationFactorsHalf",
    "quad": "MTLQuadTessellationFactorsHalf",
}
# The tessellation factors of each patch type: (edge count, inside count or None for a scalar).
_FACTOR_COUNTS = {"triangle": (3, None), "quad": (4, 2)}
_OUTER = frozenset({"gl_tesslevelouter", "sv_tessfactor"})
_INNER = frozenset({"gl_tesslevelinner", "sv_insidetessfactor"})
_POINT_IDS = frozenset({"gl_invocationid", "sv_outputcontrolpointid"})
_PATCH_IDS = frozenset({"gl_primitiveid", "sv_primitiveid"})
_PARTITIONINGS = frozenset({"integer", "fractional_odd", "fractional_even", "pow2"})
_WINDINGS = frozenset({"triangle_cw", "triangle_ccw"})
_FACTOR_ELEMENTS = frozenset({"float", "half"})
# gl_MaxPatchVertices and Direct3D's 32 control points; Metal's patches hold at most 32 too.
_MAX_PATCH_POINTS = 32


class MetalTessellationControlError(TranslationDiagnostic):
    """A tessellation control entry that has no Metal compute kernel."""

    code = CODE_PREFIX + "metal-tessellation-control-unsupported"

    def __init__(self, entry: str, reason: str):
        self.entry = entry
        super().__init__(
            f"tessellation control entry '{entry}': {reason} (Metal Shading Language Specification 2.18 and "
            "5.1.5)",
            target="metal",
        )


def _attribute_name(attribute):
    return str(getattr(attribute, "name", attribute) or "").lower()


def _argument(attribute):
    """The single argument of ``attribute`` as text without quotes, else None."""
    arguments = getattr(attribute, "arguments", None) or []
    if len(arguments) != 1:
        return None
    argument = arguments[0]
    text = getattr(argument, "name", None)
    if text is None:
        text = getattr(argument, "value", None)
    return None if text is None else str(text).strip().strip("\"'")


def _uint(value):
    return LiteralNode(value, PrimitiveType("uint"))


def _store(target, value):
    return ExpressionStatementNode(AssignmentNode(target, value))


def _index(array, index):
    return ArrayAccessNode(IdentifierNode(array), index)


def _loop(counter, count, statements):
    return ForNode(
        VariableNode(counter, PrimitiveType("uint"), _uint(0)),
        BinaryOpNode(IdentifierNode(counter), "<", _uint(count)),
        UnaryOpNode("++", IdentifierNode(counter), is_postfix=True),
        BlockNode(statements),
    )


def _element(patch, count, counter):
    """``patch * count + counter``, the buffer element of control point ``counter`` of patch ``patch``."""
    return BinaryOpNode(
        BinaryOpNode(IdentifierNode(patch), "*", _uint(count)),
        "+",
        IdentifierNode(counter),
    )


def _all_names(ast):
    names = set()
    walk = getattr(ast, "walk", None)
    for node in walk() if callable(walk) else ():
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _library_struct(name):
    members = []
    for element, member, length in METAL_LIBRARY_STRUCTS[name]:
        member_type = PrimitiveType(element)
        if length is not None:
            member_type = ArrayType(
                member_type, LiteralNode(length, PrimitiveType("int"))
            )
        members.append(StructMemberNode(member, member_type))
    return StructNode(name, members)


class _HullLowering:
    """Rewrites one HLSL-form tessellation control stage into a compute stage."""

    def __init__(self, ast, stage, codegen):
        self.ast = ast
        self.stage = stage
        self.entry = stage.entry_point
        self.codegen = codegen
        self.name = getattr(self.entry, "name", None) or "main"
        self.structs = _structs(ast)
        self.taken = _all_names(ast)

    def refuse(self, reason):
        raise MetalTessellationControlError(self.name, reason)

    def fresh(self, base):
        name, number = base, 1
        while name in self.taken:
            number += 1
            name = f"{base}_{number}"
        self.taken.add(name)
        return name

    def semantic(self, node):
        semantic = self.codegen.semantic_from_node(node)
        return str(semantic).lower() if semantic else None

    def struct(self, type_node, role):
        if not isinstance(type_node, NamedType) or type_node.name not in self.structs:
            self.refuse(f"its {role} is not a structure")
        return self.structs[type_node.name]

    def attributes(self):
        """``(patch type, output control point count, patch-constant function name)``; drops pipeline state."""
        found = {}
        for attribute in getattr(self.entry, "attributes", None) or []:
            name = _attribute_name(attribute)
            if name in found:
                self.refuse(f"it states @{name} twice")
            found[name] = _argument(attribute)
        domain = (found.pop("domain", None) or "").lower()
        patch_type = _PATCH_TYPES.get(domain)
        if patch_type is None:
            self.refuse(
                f"Metal tessellates only triangle and quad patches, not the '{domain or 'unstated'}' domain"
            )
        partitioning = (found.pop("partitioning", None) or "integer").lower()
        if partitioning not in _PARTITIONINGS:
            self.refuse(f"Metal has no '{partitioning}' tessellation partition mode")
        topology = (found.pop("outputtopology", None) or "triangle_cw").lower()
        if topology not in _WINDINGS:
            self.refuse(
                f"Metal tessellates a patch into triangles only, not the '{topology}' output topology"
            )
        found.pop("maxtessfactor", None)
        count = _literal_int(found.pop("outputcontrolpoints", None))
        if count is None or not 0 < count <= _MAX_PATCH_POINTS:
            self.refuse("@outputcontrolpoints does not give 1 to 32 control points")
        function = found.pop("patchconstantfunc", None)
        if not function:
            self.refuse("it names no patch-constant function")
        found.pop("stage_entry", None)
        if found:
            self.refuse(
                f"Metal has no counterpart of its @{sorted(found)[0]} attribute"
            )
        return patch_type, count, function

    def entry_parameters(self):
        """``(input point type, input count, [(parameter kind, ...)])`` of the hull entry's parameters."""
        patch, kinds = None, []
        for parameter in getattr(self.entry, "parameters", None) or []:
            patch_object = _patch_object(getattr(parameter, "param_type", None))
            semantic = self.semantic(parameter)
            if patch_object is not None:
                if patch is not None or parameter.param_type.name != "InputPatch":
                    self.refuse("a hull entry takes one InputPatch and no OutputPatch")
                patch = patch_object
                kinds.append("inputs")
            elif semantic in _POINT_IDS:
                kinds.append("point")
            elif semantic in _PATCH_IDS:
                kinds.append("patch")
            else:
                self.refuse(
                    f"parameter '{parameter.name}' is neither the input patch, the output control point id "
                    "nor the patch id, the values a Metal compute kernel can supply"
                )
        if patch is None:
            self.refuse("it takes no InputPatch")
        point_type, count_node = patch
        count = _literal_int(count_node)
        if count is None or not 0 < count <= _MAX_PATCH_POINTS:
            self.refuse("its InputPatch does not hold 1 to 32 control points")
        self.struct(point_type, "input control point")
        return point_type, count, kinds

    def find_function(self, name):
        for function in [
            *(getattr(self.ast, "functions", None) or []),
            *(getattr(self.stage, "local_functions", None) or []),
        ]:
            if getattr(function, "name", None) == name:
                return function
        self.refuse(f"its patch-constant function '{name}' is not declared")

    def constant_arguments(self, function, inputs, outputs):
        """The kinds of the patch-constant function's arguments, checking its patches against the hull's."""
        kinds = []
        for parameter in getattr(function, "parameters", None) or []:
            parameter_type = getattr(parameter, "param_type", None)
            patch_object = _patch_object(parameter_type)
            if patch_object is not None:
                kind = "inputs" if parameter_type.name == "InputPatch" else "outputs"
                expected = inputs if kind == "inputs" else outputs
                if (
                    getattr(patch_object[0], "name", None) != expected[0].name
                    or _literal_int(patch_object[1]) != expected[1]
                ):
                    self.refuse(
                        f"the {parameter_type.name} of patch-constant function '{function.name}' is not the "
                        "hull's"
                    )
                kinds.append(kind)
            elif self.semantic(parameter) in _PATCH_IDS:
                kinds.append("patch")
            else:
                self.refuse(
                    f"parameter '{parameter.name}' of patch-constant function '{function.name}' is neither a "
                    "patch nor the patch id"
                )
            parameter.attributes = []
            if hasattr(parameter, "semantic"):
                parameter.semantic = None
        return kinds

    def factor_member(self, member, count, role):
        member_type = getattr(member, "member_type", None)
        element, length = member_type, None
        if isinstance(member_type, (ArrayType, VectorType)):
            element, length = member_type.element_type, _literal_int(member_type.size)
        if getattr(element, "name", None) not in _FACTOR_ELEMENTS or (
            length != count and not (count is None and length in (None, 1))
        ):
            shape = "a float" if count is None else f"an array of {count} floats"
            self.refuse(
                f"its {role} tessellation factor '{member.name}' is not {shape}"
            )
        return length

    def constants(self, struct, patch_type):
        """``(outer member, inner member, inner length, the other members)`` of the patch-constant structure."""
        outer = inner = None
        data = []
        for member in getattr(struct, "members", None) or []:
            semantic = self.semantic(member)
            if semantic in _OUTER or semantic in _INNER:
                if (outer if semantic in _OUTER else inner) is not None:
                    self.refuse(f"its patch constants carry '{semantic}' twice")
                if semantic in _OUTER:
                    outer = member
                else:
                    inner = member
            else:
                data.append(member)
        if outer is None or inner is None:
            self.refuse(
                "its patch constants lack the edge or the inside tessellation factors Metal's tessellator reads"
            )
        edges, insides = _FACTOR_COUNTS[patch_type]
        self.factor_member(outer, edges, "edge")
        inner_length = self.factor_member(inner, insides, "inside")
        return outer, inner, inner_length, data

    def factor_stores(self, outer, inner, inner_length, patch_type):
        """Store the edge and inside factors of the patch constants as halves in the patch's factor record."""
        names = self.names
        factors, patch, constants = (
            names["tessellation_factors"],
            names["patch"],
            names["constants"],
        )
        record = ArrayAccessNode(IdentifierNode(factors), IdentifierNode(patch))
        edges, insides = _FACTOR_COUNTS[patch_type]
        stores = []

        def half(member, index):
            value = MemberAccessNode(IdentifierNode(constants), member.name)
            if index is not None:
                value = ArrayAccessNode(value, LiteralNode(index, PrimitiveType("int")))
            return FunctionCallNode(IdentifierNode("half"), [value])

        for index in range(edges):
            target = ArrayAccessNode(
                MemberAccessNode(record, "edgeTessellationFactor"),
                LiteralNode(index, PrimitiveType("int")),
            )
            stores.append(_store(target, half(outer, index)))
        if insides is None:
            target = MemberAccessNode(record, "insideTessellationFactor")
            stores.append(_store(target, half(inner, 0 if inner_length == 1 else None)))
        else:
            for index in range(insides):
                target = ArrayAccessNode(
                    MemberAccessNode(record, "insideTessellationFactor"),
                    LiteralNode(index, PrimitiveType("int")),
                )
                stores.append(_store(target, half(inner, index)))
        return stores

    def stage_locals(self, data):
        """The stage's variables the kernel keeps; drops the ``out`` copies of patch constants the importer writes."""
        constants = {member.name for member in data}
        kept = []
        for variable in getattr(self.stage, "local_variables", None) or []:
            qualifiers = {
                str(q).lower() for q in getattr(variable, "qualifiers", None) or []
            }
            if "out" in qualifiers and getattr(variable, "name", None) in constants:
                continue
            if qualifiers & {"in", "out", "inout", "patch"}:
                self.refuse(
                    f"stage variable '{variable.name}' is a stage input or output, which a compute kernel has not"
                )
            kept.append(variable)
        return kept

    def helper(self):
        """Turn the hull entry into the control-point helper; returns its name."""
        helper = self.entry
        helper.name = self.fresh(f"{self.name}_control_point")
        helper.attributes = []
        helper.preserve_stage_entry_name = False
        helper.qualifiers = []
        if hasattr(helper, "qualifier"):
            helper.qualifier = None
        for parameter in getattr(helper, "parameters", None) or []:
            parameter.attributes = []
            if hasattr(parameter, "semantic"):
                parameter.semantic = None
        return helper

    def point_phase(self, inputs, outputs, helper, kinds):
        """Copy the patch's input points, then call ``helper`` for each output point and store its result."""
        names = self.names
        point = IdentifierNode(names["point"])
        (input_type, input_count), (output_type, output_count) = inputs, outputs
        return [
            VariableNode(
                names["patch"],
                PrimitiveType("uint"),
                MemberAccessNode(IdentifierNode("gl_GlobalInvocationID"), "x"),
            ),
            VariableNode(
                names["inputs"],
                ArrayType(
                    NamedType(input_type.name),
                    LiteralNode(input_count, PrimitiveType("int")),
                ),
            ),
            _loop(
                names["point"],
                input_count,
                [
                    _store(
                        _index(names["inputs"], deepcopy(point)),
                        _index(
                            names["input_points"],
                            _element(names["patch"], input_count, names["point"]),
                        ),
                    )
                ],
            ),
            VariableNode(
                names["outputs"],
                ArrayType(
                    NamedType(output_type.name),
                    LiteralNode(output_count, PrimitiveType("int")),
                ),
            ),
            _loop(
                names["point"],
                output_count,
                [
                    _store(
                        _index(names["outputs"], deepcopy(point)),
                        FunctionCallNode(IdentifierNode(helper), self.arguments(kinds)),
                    ),
                    _store(
                        _index(
                            names["output_points"],
                            _element(names["patch"], output_count, names["point"]),
                        ),
                        _index(names["outputs"], deepcopy(point)),
                    ),
                ],
            ),
        ]

    def arguments(self, kinds):
        """The kernel's values for parameters of ``kinds``: the input points, output points, patch or point id."""
        return [IdentifierNode(self.names[kind]) for kind in kinds]

    def patch_data(self, data):
        """The ``<entry>_PatchData`` structure of the patch constants ``data`` and the stores that fill it."""
        names = self.names
        struct = StructNode(
            self.fresh(f"{self.name}_PatchData"),
            [
                StructMemberNode(member.name, deepcopy(member.member_type))
                for member in data
            ],
        )
        record = ArrayAccessNode(
            IdentifierNode(names["patch_data"]), IdentifierNode(names["patch"])
        )
        stores = [
            _store(
                MemberAccessNode(deepcopy(record), member.name),
                MemberAccessNode(IdentifierNode(names["constants"]), member.name),
            )
            for member in data
        ]
        return struct, stores

    def factor_structs(self, factor_struct):
        """The Metal library's factor structure to declare, unless the program declares it as the library does."""
        declared = self.structs.get(factor_struct)
        if declared is None:
            return [_library_struct(factor_struct)]
        if not is_metal_library_struct(declared):
            self.refuse(
                f"the program declares a '{factor_struct}' other than the one the Metal library declares"
            )
        return []

    def compute_stage(self, body, buffers, kept_locals):
        kernel = FunctionNode(self.name, PrimitiveType("void"), [], BlockNode(body))
        kernel.preserve_stage_entry_name = True
        variables = [
            VariableNode(name, PointerType(NamedType(type_name)), qualifiers=["buffer"])
            for name, type_name in buffers
        ]
        size = [
            AttributeNode(f"local_size_{axis}", [LiteralNode(1, PrimitiveType("int"))])
            for axis in "xyz"
        ]
        return StageNode(
            ShaderStage.COMPUTE,
            kernel,
            local_variables=variables + kept_locals,
            local_functions=[],
            local_structs=[],
            execution_config={f"local_size_{axis}": "1" for axis in "xyz"},
            layout_qualifiers=[LayoutQualifierNode(size, direction="in")],
            local_cbuffers=list(getattr(self.stage, "local_cbuffers", None) or []),
        )

    def run(self):
        """Return the compute stage that replaces the hull stage and the structures whose semantics go."""
        patch_type, output_count, constant_name = self.attributes()
        point_type, input_count, entry_kinds = self.entry_parameters()
        output_type = getattr(self.entry, "return_type", None)
        self.struct(output_type, "output control point")
        constant_function = self.find_function(constant_name)
        constants_struct = self.struct(
            getattr(constant_function, "return_type", None),
            "patch-constant function's result",
        )
        constant_kinds = self.constant_arguments(
            constant_function, (point_type, input_count), (output_type, output_count)
        )
        outer, inner, inner_length, data = self.constants(constants_struct, patch_type)
        kept_locals = self.stage_locals(data)
        helper = self.helper()
        self.names = {
            key: self.fresh(f"{self.name}_{key}")
            for key in (
                "input_points",
                "output_points",
                "patch_data",
                "tessellation_factors",
                "patch",
                "inputs",
                "outputs",
                "point",
                "constants",
            )
        }
        body = self.point_phase(
            (point_type, input_count),
            (output_type, output_count),
            helper.name,
            entry_kinds,
        )
        body.append(
            VariableNode(
                self.names["constants"],
                NamedType(constants_struct.name),
                FunctionCallNode(
                    IdentifierNode(constant_function.name),
                    self.arguments(constant_kinds),
                ),
            )
        )
        body += self.factor_stores(outer, inner, inner_length, patch_type)
        buffers = [
            (self.names["input_points"], point_type.name),
            (self.names["output_points"], output_type.name),
        ]
        new_structs = []
        if data:
            struct, stores = self.patch_data(data)
            new_structs.append(struct)
            buffers.append((self.names["patch_data"], struct.name))
            body += stores
        factor_struct = _FACTOR_STRUCTS[patch_type]
        buffers.append((self.names["tessellation_factors"], factor_struct))
        new_structs += self.factor_structs(factor_struct)
        self.ast.functions = [
            *(getattr(self.ast, "functions", None) or []),
            *(getattr(self.stage, "local_functions", None) or []),
            helper,
        ]
        self.ast.structs = [
            *(getattr(self.ast, "structs", None) or []),
            *(getattr(self.stage, "local_structs", None) or []),
            *new_structs,
        ]
        compute = self.compute_stage(body, buffers, kept_locals)
        return compute, {point_type.name, output_type.name, constants_struct.name}


def lower_metal_tessellation_control(ast, codegen, target_stage=None):
    """Return ``(ast, target_stage)`` with each HLSL-form tessellation control stage written as a compute stage.

    ``codegen`` is the Metal emitter, whose ``semantic_from_node`` reads semantics. A requested
    ``tessellation_control`` stage becomes ``compute``. Returns ``ast`` itself when it has no such stage; raises
    :class:`MetalTessellationControlError` for an entry Metal cannot write. A GLSL-form stage, which names no
    patch-constant function, is left for :func:`.metal_stage_refusals.refuse_metal_stages`.
    """
    stages = getattr(ast, "stages", None) or {}
    keys = [
        key
        for key, stage in stages.items()
        if normalize_stage_name(key) == _STAGE
        and getattr(stage, "entry_point", None) is not None
        and any(
            _attribute_name(attribute) == "patchconstantfunc"
            for attribute in getattr(stage.entry_point, "attributes", None) or []
        )
    ]
    if not keys or target_stage not in (None, _STAGE):
        return ast, target_stage
    if any(normalize_stage_name(key) == "compute" for key in stages):
        raise MetalTessellationControlError(
            stages[keys[0]].entry_point.name,
            "the program has a compute stage too, and the hull becomes a Metal compute kernel",
        )
    ast = deepcopy(ast)
    rebuilt, plain = {}, set()
    for key, stage in ast.stages.items():
        if key not in keys:
            rebuilt[key] = stage
            continue
        compute, names = _HullLowering(ast, stage, codegen).run()
        rebuilt[ShaderStage.COMPUTE] = compute
        plain |= names
    ast.stages = rebuilt
    _patch_parameters_as_arrays(ast, set())
    _strip_plain_semantics(
        _structs(ast), plain - _interface_struct_names(ast, set()), codegen
    )
    return ast, ("compute" if target_stage == _STAGE else target_stage)
