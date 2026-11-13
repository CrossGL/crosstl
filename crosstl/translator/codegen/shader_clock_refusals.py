"""Typed refusal of the GLSL shader clock built-ins by emitters that have no counter to read.

``clockARB``, ``clock2x32ARB``, ``clockRealtimeEXT`` and ``clockRealtime2x32EXT`` (``GL_ARB_shader_clock`` and
``GL_EXT_shader_realtime_clock``) read a hardware counter. CUDA, HIP, GLSL and SPIR-V have an instruction for it; the
Slang, Rust and Mojo emitters have no counter they could name, and used to copy the call through unchanged, so the
output called a function no compiler of that language defines. They refuse it instead.
"""

from __future__ import annotations

from .lowering_refusals import refuse_unsupported

__all__ = ["SHADER_CLOCK_BUILTINS", "refuse_shader_clock_call"]

SHADER_CLOCK_BUILTINS = frozenset(
    {"clockARB", "clock2x32ARB", "clockRealtimeEXT", "clockRealtime2x32EXT"}
)


def refuse_shader_clock_call(target, function_name, placeholder="0"):
    """Record the refusal of the shader clock built-in ``function_name`` for ``target``."""
    return refuse_unsupported(
        target,
        "shader clock builtin",
        f"'{function_name}' reads a hardware counter that {target} has no "
        "built-in to read",
        placeholder,
    )
