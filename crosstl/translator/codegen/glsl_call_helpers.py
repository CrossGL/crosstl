"""GLSL helper functions that expression generation declares for the calls it rewrites.

Each helper family records its definitions while the stage code is generated, keyed by the GLSL value type the
calls need, and the program writes them after that code, in the order the families are listed here.
"""

from __future__ import annotations

from .exponent_builtins import emit_glsl_exponent_helpers
from .glsl_half_conversions import emit_glsl_half_conversion_helpers
from .glsl_quad_lane_reads import emit_glsl_quad_lane_read_helpers

__all__ = ["emit_glsl_call_helpers", "reset_glsl_call_helpers"]


def emit_glsl_call_helpers(codegen) -> str:
    """The ``frexp``/``modf``/``ldexp``, ``crossglQuadReadLaneAt`` and half conversion helpers the calls use."""
    return (
        emit_glsl_exponent_helpers(codegen)
        + emit_glsl_quad_lane_read_helpers(codegen)
        + emit_glsl_half_conversion_helpers(codegen)
    )


def reset_glsl_call_helpers(codegen) -> None:
    """Forget the helpers a previous rendering of the program recorded."""
    codegen.glsl_quad_lane_read_helpers = {}
    codegen.glsl_half_conversion_helpers = {}
