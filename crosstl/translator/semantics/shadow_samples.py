"""Depth comparisons written with GLSL's sampling functions on shadow samplers.

CrossGL follows GLSL for its sampling functions (GLSL 4.60 section 8.9.2, "Texel Lookup Functions"). On a shadow
sampler they compare instead of filtering texels: ``texture(sampler2DShadow s, vec3 P)`` tests the reference
``P.z`` against the depth texels at ``P.xy`` and returns the filtered result of the comparison. The reference is the
coordinate's last component (``P.z`` for 1D, 1D-array, 2D and rectangle shadows, ``P.w`` for cube and 2D-array
shadows); ``samplerCubeArrayShadow`` takes it as a separate ``compare`` operand. The projective forms divide the
coordinate and the reference by ``q`` (GLSL ``textureProj``: the reference is ``P.z / P.w``), and ``textureGather``
on a shadow sampler takes a ``refZ`` operand and gathers comparison results. A separate texture sampled through a
comparison sampler (``texture(sampler2DShadow(t, s), P)`` in Vulkan GLSL, imported as ``texture(t, s, P)``) is a
comparison too.

CrossGL also spells comparisons explicitly, with the reference as its own operand: ``textureCompare(s, P.xy, P.z)``,
and the ``textureCompareOffset``, ``Lod``, ``LodOffset``, ``Grad``, ``GradOffset``, ``Proj*`` and
``textureGatherCompare*`` forms. Only the GLSL emitters print the GLSL spelling unchanged, except for gathers, whose
``refZ`` they took for a component index. The HLSL, WGSL, Slang and SPIR-V emitters lowered it as an ordinary sample,
which reads the depth texel instead of comparing (HLSL ``Sample`` with a truncated coordinate) or is rejected by the
target compiler, and the Metal emitter lowered only some forms. :func:`legalize_shadow_samples` therefore rewrites
each such call into its compare form, splitting the reference off the coordinate; for GLSL and GLSL ES it rewrites
only the gathers and the explicit-level ``textureLod`` forms, which the GLSL emitter lowers on shadow samplers without
a core ``textureLod`` overload (:mod:`..codegen.glsl_shadow_lod`):

* ``texture``, ``textureOffset``, ``textureLod``, ``textureLodOffset``, ``textureGrad`` and ``textureGradOffset``
  become ``textureCompare``, ``textureCompareOffset``, ``textureCompareLod``, ``textureCompareLodOffset``,
  ``textureCompareGrad`` and ``textureCompareGradOffset``;
* ``textureProj`` and its ``Offset``, ``Lod``, ``LodOffset``, ``Grad`` and ``GradOffset`` forms become the matching
  ``textureCompareProj`` forms, whose coordinate keeps ``q`` as its last component (``P.xyw`` for a 2D shadow);
* ``textureGather``, ``textureGatherOffset`` and ``textureGatherOffsets`` become the ``textureGatherCompare`` forms.

The coordinate is split by swizzling when evaluating it twice is equivalent to evaluating it once, and by taking a
vector constructor's operands apart when they meet at the split. Any other coordinate (one that calls a function, for
example) is refused with :class:`ShadowSampleUnsupportedError` rather than evaluated twice.

A bias operand has no compare form in CrossGL. The Metal emitter lowers ``texture`` and ``textureOffset`` with a bias
to ``sample_compare`` with ``bias()`` (MSL 2.3 and later), so those calls are left to it; every other biased shadow
sample is refused, as are operand counts and sampler kinds for which GLSL defines no overload.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.call_graph import _stage_functions
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from .hub_types import TypeScope, call_name, element_type_node, iter_structs
from .rewrite import child_slots, clone, get_child, is_pure

__all__ = [
    "GLSL_SHADOW_TARGETS",
    "ShadowSampleUnsupportedError",
    "legalize_shadow_samples",
]

# Targets whose emitters print GLSL's sampling functions. Only their gathers and explicit-level samples are rewritten:
# the GLSL emitter reads a gather's trailing operand as a component and emits the refZ overload from
# textureGatherCompare, and it lowers textureCompareLod on shadow samplers that have no core textureLod overload.
GLSL_SHADOW_TARGETS = frozenset({"opengl", "glsl", "webgl", "gles", "essl"})
_GLSL_REWRITTEN_FORMS = frozenset({"textureLod", "textureLodOffset"})
# CrossGL itself: the program is left unchanged.
_UNCHANGED_TARGETS = frozenset({"", "cgl", "crossgl"})


class _Shape:
    """Where a sampler kind keeps the reference: the coordinate and reference swizzles, plain and projective."""

    __slots__ = ("lanes", "coordinate", "reference", "projective", "proj_reference")

    def __init__(
        self, lanes, coordinate, reference, projective=None, proj_reference=None
    ):
        self.lanes = lanes
        self.coordinate = coordinate
        self.reference = reference
        self.projective = projective
        self.proj_reference = proj_reference


_SHAPES_1D = _Shape(3, "x", "z", "xw", "z")
_SHAPES_1D_ARRAY = _Shape(3, "xy", "z")
_SHAPES_2D = _Shape(3, "xy", "z", "xyw", "z")
_SHAPES_LAYERED = _Shape(4, "xyz", "w")
# samplerCubeArrayShadow: the coordinate is all of P and the reference is a separate operand.
_SHAPES_CUBE_ARRAY = _Shape(4, None, None)

# Combined shadow samplers (GLSL 4.60 section 4.1.7.1).
_SHADOW_SAMPLERS = {
    "sampler1DShadow": _SHAPES_1D,
    "sampler1DArrayShadow": _SHAPES_1D_ARRAY,
    "sampler2DShadow": _SHAPES_2D,
    "sampler2DRectShadow": _SHAPES_2D,
    "samplerCubeShadow": _SHAPES_LAYERED,
    "sampler2DArrayShadow": _SHAPES_LAYERED,
    "samplerCubeArrayShadow": _SHAPES_CUBE_ARRAY,
}
# Separate textures, which compare when sampled through a comparison sampler (GLSL 4.60 section 4.1.7.2).
_SEPARATE_TEXTURES = {
    "texture1D": _SHAPES_1D,
    "texture1DArray": _SHAPES_1D_ARRAY,
    "texture2D": _SHAPES_2D,
    "texture2DRect": _SHAPES_2D,
    "textureCube": _SHAPES_LAYERED,
    "texture2DArray": _SHAPES_LAYERED,
    "textureCubeArray": _SHAPES_CUBE_ARRAY,
}
_COMPARISON_SAMPLERS = frozenset(
    {"samplerShadow", "comparison_sampler", "SamplerComparisonState"}
)
_SEPARATE_SAMPLERS = _COMPARISON_SAMPLERS | {"sampler", "SamplerState"}


class _Form:
    """A GLSL sampling function: its compare form, how many operands follow the coordinate, and its kind."""

    __slots__ = ("name", "compare", "operands", "kind", "accepts_bias")

    def __init__(self, name, compare, operands, kind="sample", accepts_bias=False):
        self.name = name
        self.compare = compare
        self.operands = operands
        self.kind = kind
        self.accepts_bias = accepts_bias


_FORMS = {
    form.name: form
    for form in (
        _Form("texture", "textureCompare", 0, accepts_bias=True),
        _Form("textureOffset", "textureCompareOffset", 1, accepts_bias=True),
        _Form("textureLod", "textureCompareLod", 1),
        _Form("textureLodOffset", "textureCompareLodOffset", 2),
        _Form("textureGrad", "textureCompareGrad", 2),
        _Form("textureGradOffset", "textureCompareGradOffset", 3),
        _Form("textureProj", "textureCompareProj", 0, "proj", True),
        _Form("textureProjOffset", "textureCompareProjOffset", 1, "proj", True),
        _Form("textureProjLod", "textureCompareProjLod", 1, "proj"),
        _Form("textureProjLodOffset", "textureCompareProjLodOffset", 2, "proj"),
        _Form("textureProjGrad", "textureCompareProjGrad", 2, "proj"),
        _Form("textureProjGradOffset", "textureCompareProjGradOffset", 3, "proj"),
        _Form("textureGather", "textureGatherCompare", 1, "gather"),
        _Form("textureGatherOffset", "textureGatherCompareOffset", 2, "gather"),
        _Form("textureGatherOffsets", "textureGatherCompareOffsets", 2, "gather"),
    )
}
# samplerCubeArrayShadow overloads: the compare operand follows P (GLSL texture, GL_EXT_texture_shadow_lod
# textureLod) and textureGather's refZ is that operand too.
_CUBE_ARRAY_FORMS = frozenset({"texture", "textureLod", "textureGather"})
# Biased forms the Metal emitter lowers to sample_compare with bias() (MSL 2.3 section 6.12.9).
_METAL_BIASED_FORMS = frozenset({"texture", "textureOffset"})
_VECTOR_CONSTRUCTOR_LANES = {
    f"{prefix}{lanes}": lanes
    for prefix in ("vec", "float", "half")
    for lanes in (2, 3, 4)
}

_BIAS_REASON = (
    "CrossGL has no compare form with a level-of-detail bias for this target (HLSL SampleCmpBias needs Shader "
    "Model 6.8, WGSL has no biased comparison, and the SPIR-V and Slang emitters do not lower one); sample with "
    "textureCompare, or with textureCompareLod at an explicit level"
)
_IMPURE_REASON = (
    "the coordinate would be evaluated once for the coordinate and once for the reference, and it is not known to "
    "be free of side effects; assign it to a local variable first"
)


class ShadowSampleUnsupportedError(TranslationDiagnostic):
    """A sampling function on a shadow sampler that has no compare form for the target."""

    def __init__(
        self, target, operation, sampler_type, reason, *, source_location=None
    ):
        self.operation = operation
        self.sampler_type = sampler_type
        self.reason_code = "shadow-sample-unsupported"
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.shadow-{operation}",)
        super().__init__(
            f"operation '{operation}' on {sampler_type} cannot be lowered to a depth comparison: {reason}",
            span=_span(source_location),
            target=target,
            code=f"{CODE_PREFIX}{target}-shadow-sample-unsupported",
        )


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        line, column, file = (location.get(key) for key in ("line", "column", "file"))
    else:
        line, column, file = (
            getattr(location, key, None) for key in ("line", "column", "file")
        )
    if not isinstance(line, int):
        return None
    return SourceSpan(
        file, line, column if isinstance(column, int) and column > 0 else 1
    )


def _type_name(type_node):
    while True:
        element = element_type_node(type_node)
        if element is None:
            break
        type_node = element
    if isinstance(type_node, str):
        return type_node
    return getattr(type_node, "name", None)


def _declared_type_names(program):
    """Every type spelling a declaration of ``program`` names, for the cheap early exit."""
    names = set()

    def add(type_node):
        name = _type_name(type_node)
        if isinstance(name, str):
            names.add(name)

    for variable in getattr(program, "global_variables", None) or []:
        add(getattr(variable, "var_type", None))
    for struct in iter_structs(program):
        for member in getattr(struct, "members", None) or []:
            add(
                getattr(member, "member_type", None)
                or getattr(member, "var_type", None)
            )
    functions = list(getattr(program, "functions", None) or [])
    for stage, stage_functions in _stage_functions(program):
        for variable in getattr(stage, "local_variables", None) or []:
            add(getattr(variable, "var_type", None))
        functions.extend(stage_functions)
    for function in functions:
        for parameter in getattr(function, "parameters", None) or []:
            add(
                getattr(parameter, "param_type", None)
                or getattr(parameter, "var_type", None)
            )
    return names


def _swizzle(expression, components):
    return A.MemberAccessNode(clone(expression), components)


def _rename(node, name, arguments):
    node.function = A.IdentifierNode(name)
    node.name = node.function
    node.arguments = arguments
    node.args = arguments


class _ShadowSampleRewriter:
    """Rewrites the shadow samples of one function at a time, tracking declarations in a :class:`TypeScope`."""

    def __init__(self, program, target):
        self.target = target
        self.gathers_only = target in GLSL_SHADOW_TARGETS  # and the textureLod forms
        self.scope = TypeScope(program)
        for constant in getattr(program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
        for variable in getattr(program, "global_variables", None) or []:
            self.scope.declare(variable.name, variable.var_type)
            self.scope.declare_block_members(variable)
        for cbuffer in getattr(program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)

    def run(self, function, stage):
        if getattr(function, "body", None) is None:
            return
        scope = self.scope
        scope.push()
        if stage is not None:
            for variable in getattr(stage, "local_variables", None) or []:
                scope.declare(variable.name, variable.var_type)
                scope.declare_block_members(variable)
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
        scope.push()
        for parameter in function.parameters or []:
            declared = getattr(parameter, "param_type", None)
            if declared is None:
                declared = getattr(parameter, "var_type", None)
            scope.declare(getattr(parameter, "name", None), declared)
        self._walk(function.body)
        scope.pop()
        scope.pop()

    def _walk(self, node):
        if node is None or isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, (A.BlockNode, A.ForNode))
        if scoped:
            self.scope.push()
        for field, key in list(child_slots(node)):
            self._walk(get_child(node, field, key))
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        elif isinstance(node, A.FunctionCallNode):
            self._visit_call(node)
        if scoped:
            self.scope.pop()

    def _sampler_type(self, expression):
        if isinstance(expression, A.FunctionCallNode):
            return "sampler" if call_name(expression) == "sampler" else None
        return _type_name(self.scope.node_type(expression))

    def _comparison(self, arguments):
        """``(coordinate index, sampler type, shape)`` when the call samples a shadow sampler, else ``None``."""
        texture_type = _type_name(self.scope.node_type(arguments[0]))
        if len(arguments) > 2:
            sampler_type = self._sampler_type(arguments[1])
            if sampler_type in _SEPARATE_SAMPLERS:
                if (
                    sampler_type in _COMPARISON_SAMPLERS
                    and texture_type in _SEPARATE_TEXTURES
                ):
                    return (
                        2,
                        f"{texture_type} with {sampler_type}",
                        _SEPARATE_TEXTURES[texture_type],
                    )
                return None
        if texture_type in _SHADOW_SAMPLERS:
            return 1, texture_type, _SHADOW_SAMPLERS[texture_type]
        return None

    def _refuse(self, node, operation, sampler_type, reason):
        raise ShadowSampleUnsupportedError(
            self.target,
            operation,
            sampler_type,
            reason,
            source_location=getattr(node, "source_location", None),
        )

    def _visit_call(self, node):
        operation = call_name(node)
        form = _FORMS.get(operation)
        arguments = list(node.arguments or [])
        if form is None or len(arguments) < 2:
            return
        if (
            self.gathers_only
            and form.kind != "gather"
            and operation not in _GLSL_REWRITTEN_FORMS
        ):
            return
        comparison = self._comparison(arguments)
        if comparison is None:
            return
        index, sampler_type, shape = comparison
        head = arguments[: index + 1]
        coordinate = arguments[index]
        operands = arguments[index + 1 :]
        if shape is _SHAPES_CUBE_ARRAY:
            self._rewrite_cube_array(node, form, sampler_type, head, operands)
            return
        if form.kind == "gather":
            if len(operands) != form.operands:
                self._refuse(
                    node,
                    operation,
                    sampler_type,
                    self._arity_reason(form, len(operands)),
                )
            _rename(node, form.compare, arguments)
            return
        if len(operands) == form.operands + 1 and form.accepts_bias:
            if (
                self.target == "metal"
                and form.kind == "sample"
                and operation in _METAL_BIASED_FORMS
            ):
                return
            self._refuse(node, operation, sampler_type, _BIAS_REASON)
        if len(operands) != form.operands:
            self._refuse(
                node, operation, sampler_type, self._arity_reason(form, len(operands))
            )
        if form.kind == "proj":
            if shape.projective is None:
                self._refuse(
                    node,
                    operation,
                    sampler_type,
                    "GLSL defines projective shadow lookups only for 1D, 2D and rectangle shadow samplers",
                )
            split = self._split(coordinate, 4, shape.projective, shape.proj_reference)
        else:
            split = self._split(
                coordinate, shape.lanes, shape.coordinate, shape.reference
            )
        if split is None:
            self._refuse(node, operation, sampler_type, _IMPURE_REASON)
        sampled, reference = split
        _rename(node, form.compare, [*head[:-1], sampled, reference, *operands])

    def _rewrite_cube_array(self, node, form, sampler_type, head, operands):
        operation = form.name
        if operation not in _CUBE_ARRAY_FORMS:
            self._refuse(
                node,
                operation,
                sampler_type,
                "GLSL defines only texture, textureLod (GL_EXT_texture_shadow_lod) and textureGather for "
                "cube-array shadow samplers",
            )
        expected = 2 if operation == "textureLod" else 1
        if len(operands) != expected:
            self._refuse(
                node,
                operation,
                sampler_type,
                f"a cube-array shadow {operation} takes {expected} operand(s) after the coordinate, got "
                f"{len(operands)}",
            )
        _rename(node, form.compare, [*head, *operands])

    @staticmethod
    def _arity_reason(form, count):
        return (
            f"GLSL defines no {form.name} overload for shadow samplers with {count} operand(s) after the "
            "coordinate"
        )

    def _split(self, coordinate, lanes, sampled, reference):
        """The coordinate and reference operands taken from ``coordinate``, or ``None`` when it is impure."""
        inferred = self.scope.infer(coordinate)
        if inferred is not None and inferred.lanes != lanes:
            return (
                None
                if not is_pure(coordinate)
                else self._swizzle_split(coordinate, sampled, reference)
            )
        split = self._constructor_split(coordinate, lanes, sampled, reference)
        if split is not None:
            return split
        if is_pure(coordinate):
            return self._swizzle_split(coordinate, sampled, reference)
        return None

    @staticmethod
    def _swizzle_split(coordinate, sampled, reference):
        return _swizzle(coordinate, sampled), _swizzle(coordinate, reference)

    def _constructor_split(self, coordinate, lanes, sampled, reference):
        """Split ``vecN(a, ..., ref)`` into ``(a, ...)`` and ``ref`` when the reference is its own last operand."""
        if not isinstance(coordinate, A.FunctionCallNode):
            return None
        if _VECTOR_CONSTRUCTOR_LANES.get(call_name(coordinate)) != lanes:
            return None
        # Only the trailing-reference shapes split structurally; the reference must be the last lane.
        if "xyzw"[lanes - 1] != reference or sampled != "xyzw"[: lanes - 1]:
            return None
        parts = list(coordinate.arguments or [])
        if len(parts) < 2:
            return None
        counts = [self.scope.infer(part) for part in parts]
        if any(count is None for count in counts) or counts[-1].lanes != 1:
            return None
        if sum(count.lanes for count in counts) != lanes:
            return None
        leading = parts[:-1]
        if len(leading) == 1:
            sampled_operand = leading[0]
        else:
            sampled_operand = A.FunctionCallNode(
                A.IdentifierNode(f"{call_name(coordinate)[:-1]}{lanes - 1}"), leading
            )
        return sampled_operand, parts[-1]


def legalize_shadow_samples(program, target) -> None:
    """Rewrite GLSL-style sampling of shadow samplers into CrossGL's compare forms for ``target``.

    The AST is rewritten in place. Raises :class:`ShadowSampleUnsupportedError` for a shadow sample that has no
    compare form, or whose coordinate cannot be split without evaluating it twice.
    """
    target = str(target or "").lower()
    if target in _UNCHANGED_TARGETS or not isinstance(program, A.ShaderNode):
        return
    declared = _declared_type_names(program)
    if not declared & (set(_SHADOW_SAMPLERS) | _COMPARISON_SAMPLERS):
        return
    rewriter = _ShadowSampleRewriter(program, target)
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode):
            rewriter.run(function, None)
    for stage, functions in _stage_functions(program):
        for function in functions:
            rewriter.run(function, stage)
