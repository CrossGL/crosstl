"""Implicit level-of-detail texture operations outside fragment stages.

CrossGL follows GLSL (GLSL 4.60 section 8.9, "Texture Functions"): a sample without an explicit level of detail
uses the implicit level in a fragment shader, and the base level in any other shader. Bias operands, the derivative
functions and ``textureQueryLod`` exist only where implicit derivatives do, in fragment shaders.

The other targets restrict implicit-LOD sampling to fragment shaders instead of defining it elsewhere. SPIR-V allows
``ImplicitLod`` instructions only in the Fragment execution model (and compute-like models with derivative groups);
DXC rejects ``Sample``, ``SampleBias``, ``SampleCmp`` and ``CalculateLevelOfDetail`` outside pixel shaders (and
compute-like shaders of Shader Model 6.6); WGSL allows ``textureSample`` and ``textureSampleCompare`` only in
fragment shaders (WGSL section 17.7). :func:`legalize_implicit_lod` therefore rewrites, for those targets, each
implicit-LOD sample reached from a vertex, geometry, tessellation or ray-tracing stage into the explicit-LOD form at
level 0, which is the base level GLSL defines there:

* ``texture``, ``textureOffset``, ``textureProj``, ``textureProjOffset`` become their ``Lod`` forms;
* ``textureCompare``, ``textureCompareOffset``, ``textureCompareProj``, ``textureCompareProjOffset`` become theirs.

A helper reached both from such a stage and from a fragment (or compute-like) stage is cloned, and the clone is
rewritten and called from the base-level paths, so the fragment path keeps its implicit level of detail.

Every target except CrossGL itself refuses, in those stages, what has no meaning without derivatives: a bias
operand, ``textureQueryLod``, the derivative functions and the implicit forms of the minimum-LOD clamp family (whose
GLSL functions, from GL_ARB_sparse_texture_clamp, exist only in fragment shaders). An explicit-LOD sample carries no
clamp either: SPIR-V's ``MinLod`` operand is valid only with implicit and gradient sampling, HLSL's ``SampleLevel``
has no clamp overload, and an explicit level of ``max(0, clamp)`` would apply the sampler's LOD bias after the clamp
instead of before it.

Compute, mesh and task stages are left unchanged: their implicit-LOD semantics depend on derivative groups, which
the emitters decide on their own.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.call_graph import CallGraph
from ..codegen.texture_lod_clamp import TEXTURE_LOD_CLAMP_FORMS
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .hub_types import call_name, element_type_node
from .rewrite import child_slots, clone, get_child

__all__ = [
    "BASE_LEVEL_STAGES",
    "EXPLICIT_LOD_TARGETS",
    "ImplicitLodStageUnsupportedError",
    "legalize_implicit_lod",
]

# Stages whose implicit level of detail is the base level and which have no derivatives in any target.
BASE_LEVEL_STAGES = frozenset(
    {
        "vertex",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "ray_generation",
        "ray_intersection",
        "ray_closest_hit",
        "ray_miss",
        "ray_any_hit",
        "ray_callable",
    }
)

# Targets whose implicit-LOD sampling is fragment-only, so base-level samples must be written with an explicit LOD.
EXPLICIT_LOD_TARGETS = frozenset(
    {"directx", "hlsl", "vulkan", "spirv", "wgsl", "slang"}
)

# Targets that print CrossGL itself and keep the program unchanged.
_CROSSGL_TARGETS = frozenset({"", "cgl", "crossgl"})


class _ImplicitForm:
    """An implicit-LOD operation: its explicit-LOD form and the operands around the inserted level of detail."""

    __slots__ = ("name", "explicit", "before", "after", "accepts_bias")

    def __init__(self, name, explicit, before, after, accepts_bias):
        self.name = name
        self.explicit = explicit
        self.before = before
        self.after = after
        self.accepts_bias = accepts_bias


_IMPLICIT_FORMS = {
    form.name: form
    for form in (
        _ImplicitForm("texture", "textureLod", 0, 0, True),
        _ImplicitForm("textureOffset", "textureLodOffset", 0, 1, True),
        _ImplicitForm("textureProj", "textureProjLod", 0, 0, True),
        _ImplicitForm("textureProjOffset", "textureProjLodOffset", 0, 1, True),
        _ImplicitForm("textureCompare", "textureCompareLod", 1, 0, False),
        _ImplicitForm("textureCompareOffset", "textureCompareLodOffset", 1, 1, False),
        _ImplicitForm("textureCompareProj", "textureCompareProjLod", 1, 0, False),
        _ImplicitForm(
            "textureCompareProjOffset", "textureCompareProjLodOffset", 1, 1, False
        ),
    )
}
_IMPLICIT_CLAMP_FORMS = frozenset(
    name for name, form in TEXTURE_LOD_CLAMP_FORMS.items() if not form.gradient
)
_DERIVATIVE_FUNCTIONS = frozenset(
    {
        "dFdx",
        "dFdy",
        "fwidth",
        "dFdxFine",
        "dFdyFine",
        "fwidthFine",
        "dFdxCoarse",
        "dFdyCoarse",
        "fwidthCoarse",
        "ddx",
        "ddy",
        "ddx_fine",
        "ddy_fine",
        "ddx_coarse",
        "ddy_coarse",
        "fwidth_fine",
        "fwidth_coarse",
    }
)
_SEPARATE_SAMPLER_TYPES = frozenset(
    {
        "sampler",
        "samplerShadow",
        "comparison_sampler",
        "SamplerState",
        "SamplerComparisonState",
    }
)
_FRAGMENT_ONLY_OPERATIONS = (
    frozenset(_IMPLICIT_FORMS)
    | _IMPLICIT_CLAMP_FORMS
    | _DERIVATIVE_FUNCTIONS
    | {"textureQueryLod"}
)

_BIAS_REASON = (
    "a level-of-detail bias needs implicit derivatives, which only fragment shaders have; GLSL accepts the bias "
    "parameter only in fragment shaders, and SPIR-V Bias, HLSL SampleBias and Metal bias() are fragment sampling"
)
_QUERY_LOD_REASON = (
    "textureQueryLod computes the implicit level of detail from derivatives, which only fragment shaders have "
    "(GLSL defines it only for fragment shaders; SPIR-V OpImageQueryLod and HLSL CalculateLevelOfDetail are "
    "fragment operations)"
)
_DERIVATIVE_REASON = (
    "derivatives exist only in fragment shaders (GLSL dFdx/dFdy/fwidth, HLSL ddx/ddy, SPIR-V OpDPdx, Metal dfdx "
    "and WGSL dpdx are fragment-only)"
)
_CLAMP_REASON = (
    "the implicit-LOD clamp functions of GL_ARB_sparse_texture_clamp (textureClampARB, textureOffsetClampARB and "
    "their shadow forms) exist only in fragment shaders, and an explicit-LOD sample carries no minimum-LOD clamp "
    "(SPIR-V MinLod is valid only with implicit and gradient sampling; HLSL SampleLevel has no clamp overload); an "
    "explicit level of max(0, clamp) would apply the sampler's LOD bias after the clamp instead of before it"
)


class ImplicitLodStageUnsupportedError(TranslationDiagnostic):
    """A derivative-dependent operation reached from a stage that has no derivatives."""

    def __init__(self, target, operation, stage, reason, *, source_location=None):
        self.operation = operation
        self.stage = stage
        self.reason_code = "implicit-lod-outside-fragment-stage"
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.{stage}-stage-derivatives",)
        super().__init__(
            f"operation '{operation}' is reached from the {stage} stage: {reason}",
            span=_span(source_location),
            target=target,
            code=f"{CODE_PREFIX}{target}-implicit-lod-stage-unsupported",
        )


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        value = location.get
    else:

        def value(key, default=None):
            return getattr(location, key, default)

    line = value("line")
    if not isinstance(line, int):
        return None
    column = value("column")
    return SourceSpan(
        value("file"), line, column if isinstance(column, int) and column > 0 else 1
    )


def _function_stage_qualifier(function):
    qualifiers = getattr(function, "qualifiers", None)
    candidates = list(qualifiers) if qualifiers else []
    candidates.append(getattr(function, "qualifier", None))
    candidates.extend(
        getattr(attribute, "name", attribute)
        for attribute in getattr(function, "attributes", None) or []
    )
    for candidate in candidates:
        if candidate is None:
            continue
        stage = normalize_stage_name(candidate)
        if stage in STAGE_QUALIFIER_NAMES:
            return stage
    return None


def _calls(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, A.FunctionCallNode):
            yield current
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
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


class _StageReach:
    """For each function of the call graph, the stages whose entry points may reach it."""

    def __init__(self, program, graph):
        self.graph = graph
        self.stages = [set() for _ in graph.functions]
        pending = []
        for index, function in enumerate(graph.functions):
            stage_node = graph.stage_of(index)
            if stage_node is not None:
                stage = normalize_stage_name(getattr(stage_node, "stage", None))
            else:
                stage = _function_stage_qualifier(function)
            if stage:
                self.stages[index].add(stage)
                pending.append(index)
        while pending:
            index = pending.pop()
            for callee in sorted(graph.edges[index]):
                missing = self.stages[index] - self.stages[callee]
                if missing:
                    self.stages[callee] |= missing
                    pending.append(callee)


def _fragment_only_closure(graph):
    """Which functions perform, directly or through calls, an operation that depends on implicit derivatives."""
    needs = [
        (
            any(
                call_name(node) in _FRAGMENT_ONLY_OPERATIONS
                for node in _calls(function.body)
            )
            if getattr(function, "body", None) is not None
            else False
        )
        for function in graph.functions
    ]
    changed = True
    while changed:
        changed = False
        for index, edges in enumerate(graph.edges):
            if not needs[index] and any(needs[callee] for callee in edges):
                needs[index] = True
                changed = True
    return needs


def _program_names(program):
    """Every name the program declares or refers to, so a clone's name shadows and collides with none."""
    names = set()
    stack = [program]
    while stack:
        node = stack.pop()
        for attribute in ("name", "identifier"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                names.add(value)
        stack.extend(get_child(node, field, key) for field, key in child_slots(node))
    return names


class _BaseLevelRewriter:
    """Checks, and for explicit-LOD targets rewrites, the operations of functions run by base-level stages."""

    def __init__(self, graph, target, rewrite, clone_names):
        self.graph = graph
        self.scope = graph.scope
        self.target = target
        self.rewrite = rewrite
        self.clone_names = clone_names
        self.stage = None

    def run(self, function, stage_node, stage):
        self.stage = stage
        scope = self.scope
        scope.push()
        if stage_node is not None:
            for variable in getattr(stage_node, "local_variables", None) or []:
                scope.declare(variable.name, variable.var_type)
                scope.declare_block_members(variable)
            for cbuffer in getattr(stage_node, "local_cbuffers", None) or []:
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

    def _refuse(self, node, operation, reason):
        raise ImplicitLodStageUnsupportedError(
            self.target,
            operation,
            self.stage,
            reason,
            source_location=getattr(node, "source_location", None),
        )

    def _is_separate_sampler(self, expression):
        if isinstance(expression, A.FunctionCallNode):
            return call_name(expression) == "sampler"
        return _type_name(self.scope.node_type(expression)) in _SEPARATE_SAMPLER_TYPES

    def _visit_call(self, node):
        operation = call_name(node)
        if operation in self.clone_names:
            if self.rewrite:
                _rename(node, self.clone_names[operation])
            return
        if operation in _DERIVATIVE_FUNCTIONS:
            self._refuse(node, operation, _DERIVATIVE_REASON)
        if operation == "textureQueryLod":
            self._refuse(node, operation, _QUERY_LOD_REASON)
        if operation in _IMPLICIT_CLAMP_FORMS:
            self._refuse(node, operation, _CLAMP_REASON)
        form = _IMPLICIT_FORMS.get(operation)
        if form is None:
            return
        arguments = node.arguments or []
        coordinate = (
            2 if len(arguments) > 2 and self._is_separate_sampler(arguments[1]) else 1
        )
        extra = len(arguments) - coordinate - 1
        if form.accepts_bias and extra == form.before + form.after + 1:
            self._refuse(node, operation, _BIAS_REASON)
        if extra != form.before + form.after or not self.rewrite:
            return
        node.arguments.insert(
            coordinate + 1 + form.before,
            A.LiteralNode(0.0, A.PrimitiveType("float")),
        )
        _rename(node, form.explicit)


def _rename(node, name):
    node.function = A.IdentifierNode(name)
    node.name = node.function
    node.args = node.arguments


def _unique_name(base, taken):
    candidate = f"{base}_base_lod"
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_base_lod{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _insert_after(container, original, duplicate):
    for position, item in enumerate(container):
        if item is original:
            container.insert(position + 1, duplicate)
            return True
    return False


def legalize_implicit_lod(program, target) -> None:
    """Refuse derivative-dependent operations in base-level stages, and lower their samples for ``target``.

    The AST is rewritten in place. Raises :class:`ImplicitLodStageUnsupportedError` for an operation reached from a
    base-level stage that ``target`` cannot express there.
    """
    target = str(target or "").lower()
    if target in _CROSSGL_TARGETS or not isinstance(program, A.ShaderNode):
        return
    graph = CallGraph(program)
    if not graph.functions:
        return
    needs = _fragment_only_closure(graph)
    if not any(needs):
        return
    reach = _StageReach(program, graph)
    rewrite = target in EXPLICIT_LOD_TARGETS

    clone_groups = set()
    if rewrite:
        for index, stages in enumerate(reach.stages):
            if (
                needs[index]
                and stages & BASE_LEVEL_STAGES
                and stages - BASE_LEVEL_STAGES
            ):
                clone_groups.add(graph.functions[index].name)
    # A base-level call to a cloned name calls the clone, so every overload base-level stages reach is cloned.
    base_level = [
        index
        for index, stages in enumerate(reach.stages)
        if stages & BASE_LEVEL_STAGES
        and (needs[index] or graph.functions[index].name in clone_groups)
    ]

    clone_names = {}
    if clone_groups:
        taken = _program_names(program)
        clone_names = {name: _unique_name(name, taken) for name in sorted(clone_groups)}
    rewriter = _BaseLevelRewriter(graph, target, rewrite, clone_names)
    for index in base_level:
        function = graph.functions[index]
        stage_node = graph.stage_of(index)
        stage = sorted(reach.stages[index] & BASE_LEVEL_STAGES)[0]
        if function.name in clone_names:
            if not reach.stages[index] - BASE_LEVEL_STAGES:
                # Only base-level stages run this overload: it moves to the clone name instead of being copied.
                function.name = clone_names[function.name]
                rewriter.run(function, stage_node, stage)
                continue
            duplicate = clone(function)
            duplicate.name = clone_names[function.name]
            container = (
                getattr(stage_node, "local_functions", None)
                if stage_node is not None
                else program.functions
            )
            if container is None or not _insert_after(container, function, duplicate):
                program.functions.append(duplicate)
            function = duplicate
        rewriter.run(function, stage_node, stage)
