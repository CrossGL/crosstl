"""Texture operations whose operands only some generators lower: compare bias, compare-gather component, sample
position.

CrossGL follows GLSL for its sampling functions, and adds the operands other languages give them:

* **A level-of-detail bias on a depth comparison.** ``textureCompare(t[, s], P, ref[, bias])``,
  ``textureCompareOffset(t[, s], P, ref, offset[, bias])``, ``textureCompareProj(t[, s], P, ref[, bias])``,
  ``textureCompareProjOffset(t[, s], P, ref, offset[, bias])``, ``textureCompareClamp(t[, s], P, ref, clamp[, bias])``
  and ``textureCompareOffsetClamp(t[, s], P, ref, offset, clamp[, bias])``: the bias is the last operand, as it is
  on ``texture`` and its offset, projective and clamp forms. HLSL writes it as ``SampleCmpBias`` (Shader Model 6.8),
  GLSL as the ``bias`` operand of ``texture`` and ``textureProj`` on 1D, 2D and cube shadow samplers, SPIR-V as the
  ``Bias`` image operand of ``OpImageSampleDref*ImplicitLod`` and MSL as the ``bias`` option of ``sample_compare``.
  WGSL's ``textureSampleCompare`` takes no bias.
* **The component a depth-compare gather compares.** ``textureGatherCompare(t[, s], P, ref[, comp])``,
  ``textureGatherCompareOffset(t[, s], P, ref, offset[, comp])`` and ``textureGatherCompareOffsets(t[, s], P, ref,
  offsets[, comp])``: ``comp`` is an integer constant from 0 to 3, as on ``textureGather``, and 0 (the default)
  compares the first component. HLSL writes 1, 2 and 3 as ``GatherCmpGreen``, ``GatherCmpBlue`` and
  ``GatherCmpAlpha`` (DXIL ``TextureGatherCmp`` with a channel operand). GLSL's ``textureGather`` with ``refZ``,
  SPIR-V's ``OpImageDrefGather``, MSL's ``gather_compare`` and WGSL's ``textureGatherCompare`` compare only the
  first component.
* **The position of a multisample texture's sample.** ``vec2 textureSamplePosition(t, int sample)`` on a
  ``texture2DMS``, ``texture2DMSArray``, ``image2DMS`` or ``image2DMSArray`` returns the position of sample
  ``sample`` relative to the pixel centre (HLSL ``Texture2DMS.GetSamplePosition``, DXIL
  ``Texture2DMSGetSamplePosition``). GLSL, SPIR-V and WGSL have no per-texture query: ``gl_SamplePosition`` is the
  current fragment's sample.

:func:`check_texture_operands` runs before generation. It drops a compare-gather component that is the constant 0,
refuses a component that is not an integer constant from 0 to 3 with :class:`TextureOperandError`, and refuses each
of these operations, for a target whose generator has not declared that it lowers it with
:func:`register_texture_operand_target`, with :class:`TextureOperandUnsupportedError`. The generators used to drop the
operands or print the call verbatim. CrossGL output keeps every operation as written.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.call_graph import _stage_functions
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from ..refusals import RegistryRefusal
from .hub_types import TypeScope, call_name, element_type_node
from .rewrite import child_slots, get_child

__all__ = [
    "COMPARE_BIAS",
    "COMPARE_BIAS_FORMS",
    "GATHER_COMPARE_COMPONENT",
    "GATHER_COMPARE_FORMS",
    "SAMPLE_POSITION",
    "SAMPLE_POSITION_OPERATION",
    "TEXTURE_OPERAND_FEATURES",
    "TextureOperandError",
    "TextureOperandUnsupportedError",
    "check_texture_operands",
    "register_texture_operand_target",
    "texture_operand_targets",
]

COMPARE_BIAS = "compare-bias"
GATHER_COMPARE_COMPONENT = "gather-compare-component"
SAMPLE_POSITION = "sample-position"
TEXTURE_OPERAND_FEATURES = (COMPARE_BIAS, GATHER_COMPARE_COMPONENT, SAMPLE_POSITION)
SAMPLE_POSITION_OPERATION = "textureSamplePosition"

# The operands each compare form takes after its coordinate without a bias (the reference first).
COMPARE_BIAS_FORMS = {
    "textureCompare": 1,
    "textureCompareOffset": 2,
    "textureCompareProj": 1,
    "textureCompareProjOffset": 2,
    "textureCompareClamp": 2,
    "textureCompareOffsetClamp": 3,
}
# The operand counts each compare gather takes after its coordinate without a component: the reference, then
# its offset, or one offset array or four offsets.
GATHER_COMPARE_FORMS = {
    "textureGatherCompare": (1,),
    "textureGatherCompareOffset": (2,),
    "textureGatherCompareOffsets": (2, 5),
}
_SEPARATE_SAMPLER_TYPES = frozenset(
    {
        "sampler",
        "samplerShadow",
        "comparison_sampler",
        "SamplerState",
        "SamplerComparisonState",
    }
)
_GATHER_COMPONENTS = {1: "Green", 2: "Blue", 3: "Alpha"}
_UNCHANGED_TARGETS = frozenset({"", "cgl", "crossgl"})
_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_GLSL_TARGETS = frozenset({"opengl", "glsl", "webgl", "gles", "essl"})
_SPIRV_TARGETS = frozenset({"vulkan", "spirv"})

# Target names whose generators lower each feature. CrossGL output keeps the operations as written, and the HLSL
# generator writes textureSamplePosition as GetSamplePosition.
_TARGETS: dict[str, set[str]] = {
    COMPARE_BIAS: set(_UNCHANGED_TARGETS),
    GATHER_COMPARE_COMPONENT: set(_UNCHANGED_TARGETS),
    SAMPLE_POSITION: set(_UNCHANGED_TARGETS) | set(_HLSL_TARGETS),
}


def register_texture_operand_target(feature: str, *targets: str) -> None:
    """Declare that the generators of ``targets`` lower ``feature``, so programs that use it reach them."""
    if feature not in _TARGETS:
        raise RegistryRefusal(
            f"unknown texture operand feature {feature!r}; expected one of {', '.join(TEXTURE_OPERAND_FEATURES)}"
        )
    _TARGETS[feature].update(str(target).lower() for target in targets)


def texture_operand_targets(feature: str) -> frozenset[str]:
    """The targets whose generators lower ``feature``."""
    return frozenset(_TARGETS[feature])


def _family(target):
    if target in _HLSL_TARGETS:
        return "hlsl"
    if target in _GLSL_TARGETS:
        return "glsl"
    if target in _SPIRV_TARGETS:
        return "spirv"
    return target


# Reasons a target cannot express a feature at all; any other unregistered target lacks a lowering.
_INEXPRESSIBLE = {
    COMPARE_BIAS: {
        "wgsl": "WGSL's textureSampleCompare takes no bias (WGSL section 17.7)",
    },
    GATHER_COMPARE_COMPONENT: {
        "glsl": (
            "GLSL's textureGather with a refZ operand compares only the first component (GLSL 4.60 section 8.9.4)"
        ),
        "spirv": (
            "SPIR-V's OpImageDrefGather has no component operand and compares the first component"
        ),
        "metal": "MSL's gather_compare compares only the first component",
        "wgsl": (
            "WGSL's textureGatherCompare compares only the first component (WGSL section 17.7)"
        ),
    },
    SAMPLE_POSITION: {
        "glsl": (
            "GLSL has no per-texture sample-position query; gl_SamplePosition is the current fragment's sample"
        ),
        "spirv": "SPIR-V has no image instruction that returns a sample's position",
        "wgsl": "WGSL has no sample-position query (WGSL section 17.7)",
    },
}
_DESCRIPTIONS = {
    COMPARE_BIAS: "a depth comparison with a level-of-detail bias",
    GATHER_COMPARE_COMPONENT: (
        "a depth-compare gather of another component than the first"
    ),
    SAMPLE_POSITION: "a multisample texture's sample-position query",
}


class TextureOperandUnsupportedError(TranslationDiagnostic):
    """A texture operation or operand that the target's generator does not lower."""

    def __init__(self, target, feature, operation, reason, *, source_location=None):
        self.feature = feature
        self.operation = operation
        self.reason_code = "texture-operand-unsupported"
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.{feature}",)
        super().__init__(
            f"'{operation}' is {_DESCRIPTIONS[feature]}, which {target} output cannot carry: {reason}",
            span=_span(source_location),
            target=target,
            code=f"{CODE_PREFIX}{target}-texture-operand-unsupported",
        )


class TextureOperandError(TranslationDiagnostic):
    """A texture operand that no target accepts, such as a non-constant compare-gather component."""

    def __init__(self, target, operation, reason, *, source_location=None):
        self.operation = operation
        self.reason_code = "texture-operand-invalid"
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"'{operation}': {reason}",
            span=_span(source_location),
            target=target or "cgl",
            code=f"{CODE_PREFIX}{target or 'cgl'}-texture-operand-invalid",
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
    element = element_type_node(type_node) if type_node is not None else None
    node = element if element is not None else type_node
    if isinstance(node, str):
        return node
    return getattr(node, "name", None)


def _component(argument):
    """The integer value of a constant component operand, or ``None``."""
    if isinstance(argument, A.LiteralNode):
        value = argument.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.rstrip("uU")
            return int(text, 0) if text.isdigit() else None
    return None


class _Checker:
    """Checks the texture operations of one function at a time, tracking declarations in a :class:`TypeScope`."""

    def __init__(self, program, target):
        self.target = target
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

    def _coordinate_index(self, arguments):
        if len(arguments) > 2:
            sampler = arguments[1]
            if isinstance(sampler, A.FunctionCallNode):
                if call_name(sampler) == "sampler":
                    return 2
            elif _type_name(self.scope.node_type(sampler)) in _SEPARATE_SAMPLER_TYPES:
                return 2
        return 1

    def _refuse(self, node, feature, operation):
        family = _family(self.target)
        reason = _INEXPRESSIBLE[feature].get(family) or (
            f"the {self.target} generator does not lower it"
        )
        raise TextureOperandUnsupportedError(
            self.target,
            feature,
            operation,
            reason,
            source_location=getattr(node, "source_location", None),
        )

    def _supported(self, feature):
        return self.target in _TARGETS[feature]

    def _visit_call(self, node):
        operation = call_name(node)
        if operation == SAMPLE_POSITION_OPERATION:
            if not self._supported(SAMPLE_POSITION):
                self._refuse(node, SAMPLE_POSITION, operation)
            return
        arguments = node.arguments or []
        if operation in COMPARE_BIAS_FORMS:
            extra = len(arguments) - self._coordinate_index(arguments) - 1
            if extra == COMPARE_BIAS_FORMS[operation] + 1 and not self._supported(
                COMPARE_BIAS
            ):
                self._refuse(node, COMPARE_BIAS, operation)
            return
        counts = GATHER_COMPARE_FORMS.get(operation)
        if counts is None:
            return
        extra = len(arguments) - self._coordinate_index(arguments) - 1
        if extra not in tuple(count + 1 for count in counts):
            return
        component = _component(arguments[-1])
        if component is None or not 0 <= component <= 3:
            raise TextureOperandError(
                self.target,
                operation,
                "the component operand must be an integer constant from 0 to 3, as on textureGather",
                source_location=getattr(node, "source_location", None),
            )
        if self.target in _UNCHANGED_TARGETS:
            return
        if component == 0:
            # The first component is the default: the call without the operand compares the same texels.
            node.arguments = list(arguments[:-1])
            node.args = node.arguments
            return
        if not self._supported(GATHER_COMPARE_COMPONENT):
            self._refuse(node, GATHER_COMPARE_COMPONENT, operation)


def _functions(program):
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode):
            yield function, None
    for stage, functions in _stage_functions(program):
        for function in functions:
            yield function, stage


def check_texture_operands(program, target) -> None:
    """Normalize compare-gather components and refuse the texture operands ``target`` does not lower."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return
    checker = _Checker(program, target)
    for function, stage in _functions(program):
        checker.run(function, stage)
