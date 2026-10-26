"""Sampler feedback textures, refused for the targets whose generators cannot declare them.

``FeedbackTexture2D<KIND>`` and ``FeedbackTexture2DArray<KIND>`` record which mip levels or regions of a paired
texture its samples touched (D3D12 sampler feedback, HLSL Shader Model 6.5). ``KIND`` is a value,
``SAMPLER_FEEDBACK_MIN_MIP`` or ``SAMPLER_FEEDBACK_MIP_REGION_USED``, and the HLSL importer spells the types
``feedbackTexture2D`` and ``feedbackTexture2DArray``. The DirectX generator declares them. No other generator lowers
them: each copied the type into the target verbatim, which no other target compiler accepts.
:func:`refuse_feedback_textures` raises :class:`FeedbackTextureUnsupportedError` before generation for a feedback
texture on any other target, and for one whose argument is not a feedback kind.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "FEEDBACK_TEXTURE_TARGETS",
    "FEEDBACK_TEXTURE_TYPES",
    "SAMPLER_FEEDBACK_KINDS",
    "FeedbackTextureUnsupportedError",
    "is_sampler_feedback_kind",
    "refuse_feedback_textures",
]

FEEDBACK_TEXTURE_TYPES = frozenset(
    {
        "FeedbackTexture2D",
        "FeedbackTexture2DArray",
        "feedbackTexture2D",
        "feedbackTexture2DArray",
    }
)
SAMPLER_FEEDBACK_KINDS = frozenset(
    {"SAMPLER_FEEDBACK_MIN_MIP", "SAMPLER_FEEDBACK_MIP_REGION_USED"}
)
# Targets that declare feedback textures: the Direct3D generators and CrossGL itself.
FEEDBACK_TEXTURE_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "cgl", "crossgl"}
)


class FeedbackTextureUnsupportedError(TranslationDiagnostic):
    """A sampler feedback texture for a target that cannot declare it, or with an argument that is not a kind."""

    def __init__(self, target, type_name, reason, *, source_location=None):
        self.source_location = source_location
        # Direct3D declares feedback textures, so a refusal there is a malformed argument, not a missing capability.
        self.missing_capabilities = (
            ()
            if target in FEEDBACK_TEXTURE_TARGETS
            else (f"{target}.sampler-feedback",)
        )
        super().__init__(
            f"sampler feedback texture '{type_name}' cannot be carried to {target}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-sampler-feedback-unsupported",
        )


def is_sampler_feedback_kind(argument) -> bool:
    """Whether a type argument names one of the two sampler feedback kinds."""
    return (
        isinstance(argument, A.NamedType)
        and not argument.generic_args
        and argument.name in SAMPLER_FEEDBACK_KINDS
    )


def _refusal_reason(node, target):
    if target not in FEEDBACK_TEXTURE_TARGETS:
        return "only the DirectX generator declares sampler feedback textures"
    arguments = node.generic_args
    if arguments and (
        len(arguments) != 1 or not is_sampler_feedback_kind(arguments[0])
    ):
        return "its template argument must be SAMPLER_FEEDBACK_MIN_MIP or SAMPLER_FEEDBACK_MIP_REGION_USED"
    return None


def refuse_feedback_textures(program, target) -> None:
    """Raise :class:`FeedbackTextureUnsupportedError` for the first feedback texture ``target`` cannot declare."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return
    for node in program.walk():
        if not isinstance(node, A.NamedType) or node.name not in FEEDBACK_TEXTURE_TYPES:
            continue
        reason = _refusal_reason(node, target)
        if reason:
            raise FeedbackTextureUnsupportedError(
                target,
                node.name,
                reason,
                source_location=getattr(node, "source_location", None),
            )
