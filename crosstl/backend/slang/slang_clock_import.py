"""CrossGL spellings of the Slang core module's device-scope clock reads.

Slang v2026.20 declares ``getRealtimeClock()``, which returns the two 32-bit words of the device-scope clock
(``OpReadClockKHR`` with ``Device`` scope, written as ``clockRealtime2x32EXT()`` for GLSL), and
``getRealtimeClockLow()``, its low word. CrossGL has ``GL_EXT_shader_realtime_clock``'s ``clockRealtime2x32EXT()``,
whose first component is the low word. ``clockARB`` and ``clock2x32ARB`` keep their names: CrossGL declares both.
"""

from __future__ import annotations

__all__ = ["SLANG_CLOCK_IMPORTS"]

SLANG_CLOCK_IMPORTS = {
    "getRealtimeClock": "clockRealtime2x32EXT()",
    "getRealtimeClockLow": "clockRealtime2x32EXT().x",
}
