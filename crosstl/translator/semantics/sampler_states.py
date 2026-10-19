"""Fixed sampler state in CrossGL: ``sampler(...)`` constructors, validated and carried or refused per target.

CrossGL spells fixed (shader-declared) sampler state as a ``sampler`` constructor whose arguments use the Metal
sampler vocabulary, which covers every portable sampler parameter (MSL 3.2 §2.10, table 2.7)::

    const sampler linearClamp = sampler(address::clamp_to_edge, filter::linear, mip_filter::linear);

A ``const sampler`` declared without an initializer is the default state (normalized coordinates, clamp to edge,
nearest filtering, no mipmapping, no comparison). :func:`decode_sampler_state` turns a constructor into a
:class:`SamplerState`, refusing any argument outside the vocabulary with :class:`SamplerStateArgumentError`.

Only Metal declares such state in shader source. The other targets fix sampler state outside the shader, so
:func:`legalize_sampler_states` refuses it there with :class:`FixedSamplerStateUnsupportedError` rather than
sampling with whatever sampler the host happens to bind:

* GLSL and GLSL ES have no shader-declared sampler state; filtering, addressing, LOD clamping and comparison are
  sampler-object state set through the API (GLSL 4.60 §4.1.7, OpenGL 4.6 §8.2).
* SPIR-V's ``OpConstantSampler`` needs the ``LiteralSampler`` capability, which only the OpenCL ``Kernel``
  execution environment allows; Vulkan fixes sampler state through immutable samplers in the descriptor set
  layout (``VkDescriptorSetLayoutBinding::pImmutableSamplers``).
* Direct3D 12 fixes sampler state through root-signature static samplers (``D3D12_STATIC_SAMPLER_DESC``), and
  CrossTL does not emit root signatures yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .storage_blocks import _walk

ADDRESS_MODES = frozenset(
    {"repeat", "mirrored_repeat", "clamp_to_edge", "clamp_to_zero", "clamp_to_border"}
)
FILTERS = frozenset({"nearest", "linear"})
MIP_FILTERS = frozenset({"none", "nearest", "linear"})
COMPARE_FUNCTIONS = frozenset(
    {
        "never",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
        "always",
    }
)
SAMPLER_STATE_ENUMERATIONS = {
    "coord": frozenset({"normalized", "pixel"}),
    "address": ADDRESS_MODES,
    "s_address": ADDRESS_MODES,
    "t_address": ADDRESS_MODES,
    "r_address": ADDRESS_MODES,
    "border_color": frozenset({"transparent_black", "opaque_black", "opaque_white"}),
    "filter": FILTERS,
    "mag_filter": FILTERS,
    "min_filter": FILTERS,
    "mip_filter": MIP_FILTERS,
    "compare_func": COMPARE_FUNCTIONS,
}
# Arguments written as calls: ``max_anisotropy(n)`` and ``lod_clamp(min, max)``.
SAMPLER_STATE_CALL_ARGUMENTS = {"max_anisotropy": 1, "lod_clamp": 2}
# Targets that declare fixed sampler state in source, or keep CrossGL itself.
_CARRYING_TARGETS = frozenset({"metal", "cgl", "crossgl"})
_TARGET_REASONS = {
    "opengl": (
        "GLSL has no shader-declared sampler state: filtering, addressing, LOD clamp and comparison are "
        "sampler-object state set through the API (GLSL 4.60 section 4.1.7, OpenGL 4.6 section 8.2)"
    ),
    "vulkan": (
        "SPIR-V OpConstantSampler requires the LiteralSampler capability, which Vulkan does not allow; "
        "Vulkan fixes sampler state through immutable samplers in the descriptor set layout"
    ),
    "directx": (
        "Direct3D 12 fixes sampler state through root-signature static samplers "
        "(D3D12_STATIC_SAMPLER_DESC), which CrossTL does not emit yet"
    ),
}
# Every target spelling a reason applies to, so the lookup needs no alias step.
_TARGET_FAMILIES = {
    "opengl": ("opengl", "glsl", "webgl", "essl"),
    "vulkan": ("vulkan", "spirv"),
    "directx": ("directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"),
}
_TARGET_REASONS = {
    spelling: _TARGET_REASONS[family]
    for family, spellings in _TARGET_FAMILIES.items()
    for spelling in spellings
}


def canonical_sampler_argument(name: str) -> str:
    """``metal::filter::linear`` -> ``filter::linear``: the namespace adds nothing to the CrossGL spelling."""
    text = str(name).strip()
    while text.startswith("metal::"):
        text = text[len("metal::") :]
    return text


@dataclass(frozen=True)
class SamplerState:
    """One fixed sampler state, with MSL's defaults for every parameter the constructor leaves out."""

    coord: str = "normalized"
    s_address: str = "clamp_to_edge"
    t_address: str = "clamp_to_edge"
    r_address: str = "clamp_to_edge"
    border_color: str = "transparent_black"
    mag_filter: str = "nearest"
    min_filter: str = "nearest"
    mip_filter: str = "none"
    compare_func: str = "never"
    max_anisotropy: int = 1
    lod_clamp: tuple | None = None


class SamplerStateArgumentError(TranslationDiagnostic):
    """A ``sampler(...)`` constructor argument outside the CrossGL sampler-state vocabulary."""

    def __init__(self, argument, detail, *, target=None):
        self.missing_capabilities = ("sampler-state-argument",)
        super().__init__(
            f"sampler state argument '{argument}' {detail}",
            target=target,
            code=f"{CODE_PREFIX}sampler-state-argument-invalid",
        )


class FixedSamplerStateUnsupportedError(TranslationDiagnostic):
    """Fixed sampler state the target cannot declare in shader source."""

    def __init__(self, target, name):
        self.missing_capabilities = (f"{target}.fixed-sampler-state",)
        reason = _TARGET_REASONS.get(
            target, f"{target} has no shader-declared sampler state"
        )
        super().__init__(
            f"fixed sampler state {name} cannot be expressed for {target}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-fixed-sampler-state-unsupported",
        )


def _identifier(node):
    if isinstance(node, A.IdentifierNode):
        return node.name
    if isinstance(node, str):
        return node
    return None


def _number(node, argument):
    value = getattr(node, "value", None) if isinstance(node, A.LiteralNode) else None
    if isinstance(value, bool) or value is None:
        raise SamplerStateArgumentError(argument, "takes numeric literal operands")
    try:
        return float(str(value).rstrip("fFhHuU"))
    except ValueError as error:
        raise SamplerStateArgumentError(
            argument, "takes numeric literal operands"
        ) from error


def _call_name(node):
    function = getattr(node, "function", None)
    name = _identifier(function) if function is not None else None
    return name if name is not None else _identifier(getattr(node, "name", None))


def is_sampler_constructor(node) -> bool:
    return isinstance(node, A.FunctionCallNode) and _call_name(node) == "sampler"


def decode_sampler_state(call) -> SamplerState:
    """The :class:`SamplerState` a ``sampler(...)`` call constructs; later arguments override earlier ones."""
    fields = {}
    for argument in (
        getattr(call, "arguments", None) or getattr(call, "args", None) or []
    ):
        if isinstance(argument, A.FunctionCallNode):
            name = canonical_sampler_argument(_call_name(argument) or "")
            arity = SAMPLER_STATE_CALL_ARGUMENTS.get(name)
            operands = getattr(argument, "arguments", None) or []
            if arity is None or len(operands) != arity:
                raise SamplerStateArgumentError(
                    name or "<call>",
                    (
                        "is not a sampler state parameter"
                        if arity is None
                        else f"takes {arity} operand(s)"
                    ),
                )
            values = tuple(_number(operand, name) for operand in operands)
            if name == "max_anisotropy":
                if not values[0].is_integer() or not 1 <= values[0] <= 16:
                    raise SamplerStateArgumentError(
                        name, "must be an integer from 1 to 16"
                    )
                fields[name] = int(values[0])
            else:
                if not 0 <= values[0] <= values[1]:
                    raise SamplerStateArgumentError(name, "needs 0 <= min <= max")
                fields[name] = values
            continue
        text = _identifier(argument)
        if text is None or "::" not in text:
            raise SamplerStateArgumentError(
                text or type(argument).__name__, "is not a sampler state enumerant"
            )
        parameter, _, value = canonical_sampler_argument(text).rpartition("::")
        if value not in SAMPLER_STATE_ENUMERATIONS.get(parameter, ()):
            raise SamplerStateArgumentError(text, "is not a sampler state enumerant")
        if parameter == "address":
            fields.update(s_address=value, t_address=value, r_address=value)
        elif parameter == "filter":
            fields.update(mag_filter=value, min_filter=value)
        else:
            fields[parameter] = value
    return SamplerState(**fields)


def _is_fixed_sampler_declaration(node) -> bool:
    """A ``const sampler`` without an initializer: the default fixed state, not a bound resource."""
    if (
        not isinstance(node, A.VariableNode)
        or getattr(node, "initial_value", None) is not None
    ):
        return False
    var_type = getattr(node, "var_type", None)
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    return getattr(var_type, "name", None) == "sampler" and bool(
        qualifiers & {"const", "constexpr"}
    )


def legalize_sampler_states(program, target) -> int:
    """Validate every fixed sampler state and refuse it for targets that cannot declare one; returns the count."""
    target = str(target or "").lower()
    count = 0
    for node in _walk(program):
        if is_sampler_constructor(node):
            decode_sampler_state(node)
            name = "sampler(...)"
        elif _is_fixed_sampler_declaration(node):
            name = f"'{node.name}'"
        else:
            continue
        count += 1
        if target and target not in _CARRYING_TARGETS:
            raise FixedSamplerStateUnsupportedError(target, name)
    return count
