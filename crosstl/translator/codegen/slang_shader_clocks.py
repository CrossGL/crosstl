"""Slang spellings of the CrossGL shader clock built-ins.

Slang v2026.20's core module declares the ``GL_ARB_shader_clock`` reads ``clockARB()`` (``uint64_t``) and
``clock2x32ARB()`` (``uint2``), which read the subgroup-scope clock (``OpReadClockKHR`` with ``Subgroup`` scope), and
``getRealtimeClock()`` (``uint2``), which reads the device-scope clock as ``GL_EXT_shader_realtime_clock``'s
``clockRealtime2x32EXT`` does (``OpReadClockKHR`` with ``Device`` scope; GLSL output calls
``clockRealtime2x32EXT()``). Slang declares no 64-bit device-scope read, so ``clockRealtimeEXT()`` is a helper that
joins the two words of one ``getRealtimeClock()`` read: the first component holds the 32 least significant bits, as
the ``SPV_KHR_shader_clock`` and ``GL_EXT_shader_realtime_clock`` specifications define.
"""

from __future__ import annotations

from .lowering_refusals import refuse_unsupported

__all__ = ["SLANG_SHADER_CLOCK_CALLS", "generate_slang_shader_clock_call"]

# CrossGL built-in -> the Slang call that reads the same clock and returns the same type, or None for the
# 64-bit device-scope read, which needs a helper.
SLANG_SHADER_CLOCK_CALLS = {
    "clockARB": "clockARB()",
    "clock2x32ARB": "clock2x32ARB()",
    "clockRealtime2x32EXT": "getRealtimeClock()",
    "clockRealtimeEXT": None,
}

_REALTIME_CLOCK_HELPER = "crosstl_clock_realtime"


def _realtime_clock_helper(name):
    return (
        f"uint64_t {name}()\n"
        "{\n"
        "    uint2 words = getRealtimeClock();\n"
        "    return (uint64_t(words.y) << 32) | uint64_t(words.x);\n"
        "}"
    )


def generate_slang_shader_clock_call(codegen, function_name, arguments):
    """The Slang expression for a call of the CrossGL shader clock built-in ``function_name``, or None when
    ``function_name`` is not one. A call that passes arguments, which no clock built-in takes, is refused.
    """
    if function_name not in SLANG_SHADER_CLOCK_CALLS:
        return None
    if arguments:
        return refuse_unsupported(
            "Slang",
            "shader clock builtin",
            f"'{function_name}' takes no arguments, got {len(arguments)}",
            "0",
        )
    call = SLANG_SHADER_CLOCK_CALLS[function_name]
    if call is not None:
        return call
    helper = codegen.helper_function_name(_REALTIME_CLOCK_HELPER)
    codegen.register_helper_function(helper, _realtime_clock_helper(helper))
    return f"{helper}()"
