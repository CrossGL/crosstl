"""HLSL output for CrossGL's compare bias and compare-gather component (:mod:`..semantics.texture_operands`).

* A depth comparison with a level-of-detail bias, ``textureCompare[Offset|Proj|ProjOffset](t, s, P, ref[, offset],
  bias)``, is ``SampleCmpBias(S, Location, CompareValue, Bias[, Offset])`` (Shader Model 6.8). The clamped forms
  ``textureCompare[Offset]Clamp`` reach this module through :mod:`.texture_lod_clamp`, which lowers the biased
  counterpart here and then appends the ``Clamp`` operand.
* A compare gather of component 1, 2 or 3, ``textureGatherCompare[Offset](t, s, P, ref[, offset], comp)``, is
  ``GatherCmpGreen``, ``GatherCmpBlue`` or ``GatherCmpAlpha`` (DXIL ``TextureGatherCmp`` with a channel operand).
  :func:`~..semantics.texture_operands.check_texture_operands` has already dropped a component of 0 and refused
  one that is not an integer constant from 0 to 3.

The generator lowers the call without the operand through its ordinary path, so the sampler, coordinate, offset and
resource checks stay those of the plain operation, and this module rewrites the method it produced. When that call
is not the method the operand attaches to, the operation is refused rather than emitted without the operand.
"""

from __future__ import annotations

from ..semantics.texture_operands import (
    COMPARE_BIAS,
    COMPARE_BIAS_FORMS,
    GATHER_COMPARE_COMPONENT,
    GATHER_COMPARE_FORMS,
    register_texture_operand_target,
)
from .texture_lod_clamp import join_call_text, split_call_text

__all__ = [
    "HLSL_GATHER_COMPARE_METHODS",
    "HLSL_TEXTURE_OPERAND_TARGETS",
    "biased_sample_cmp_text",
    "component_gather_cmp_text",
    "split_compare_bias",
    "split_gather_compare_component",
]

HLSL_TEXTURE_OPERAND_TARGETS = ("directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12")
# The HLSL method that gathers each component of a depth comparison; component 0 is GatherCmp (GatherCmpRed).
HLSL_GATHER_COMPARE_METHODS = {
    0: "GatherCmp",
    1: "GatherCmpGreen",
    2: "GatherCmpBlue",
    3: "GatherCmpAlpha",
}

for _feature in (COMPARE_BIAS, GATHER_COMPARE_COMPONENT):
    register_texture_operand_target(_feature, *HLSL_TEXTURE_OPERAND_TARGETS)


def split_compare_bias(operation, extra_args):
    """``(operands, bias)`` for a compare form's operands after its coordinate; ``bias`` is ``None`` without one."""
    count = COMPARE_BIAS_FORMS.get(operation)
    if count is None or len(extra_args) != count + 1:
        return list(extra_args), None
    return list(extra_args[:-1]), extra_args[-1]


def _component_value(argument):
    value = getattr(argument, "value", None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.rstrip("uU")
        return int(text, 0) if text.isdigit() else None
    return None


def split_gather_compare_component(operation, extra_args):
    """``(operands, component)`` for a compare gather's operands after its coordinate.

    ``component`` is ``None`` without a component operand, and otherwise its integer value (``-1`` when it is not
    an integer constant, which no HLSL method gathers).
    """
    counts = GATHER_COMPARE_FORMS.get(operation)
    if counts is None or len(extra_args) not in tuple(count + 1 for count in counts):
        return list(extra_args), None
    value = _component_value(extra_args[-1])
    return list(extra_args[:-1]), -1 if value is None else value


def _rewritten(base_text, expected, method, insert_at, operand):
    """``base_text``, a call of the HLSL method ``expected``, as a call of ``method``.

    ``operand`` (when given) goes in at index ``insert_at`` of the call's arguments. ``None`` when ``base_text`` is
    not a call of ``expected`` with at least ``insert_at`` arguments.
    """
    parts = split_call_text(base_text)
    if parts is None:
        return None
    callee, arguments = parts
    receiver, dot, name = callee.rpartition(".")
    if not dot or name != expected or len(arguments) < insert_at:
        return None
    if operand is not None:
        arguments.insert(insert_at, operand)
    return join_call_text(f"{receiver}.{method}", arguments)


def biased_sample_cmp_text(base_text, bias_text):
    """``SampleCmpBias`` for ``base_text``, a ``SampleCmp(S, Location, CompareValue[, Offset])`` call, or ``None``."""
    return _rewritten(base_text, "SampleCmp", "SampleCmpBias", 3, bias_text)


def component_gather_cmp_text(base_text, component):
    """The ``GatherCmp<Channel>`` call for ``base_text``, a ``GatherCmp`` call, or ``None``."""
    method = HLSL_GATHER_COMPARE_METHODS.get(component)
    if method is None:
        return None
    return _rewritten(base_text, "GatherCmp", method, 0, None)
