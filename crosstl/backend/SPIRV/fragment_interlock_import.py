"""Fragment shader interlock in SPIR-V input.

SPV_EXT_fragment_shader_interlock brackets a fragment entry point's critical section with
``OpBeginInvocationInterlockEXT`` and ``OpEndInvocationInterlockEXT``, and one of four execution modes says whether
the section orders the fragments of a pixel or of a sample, by primitive order or in any order. CrossGL spells them
as GLSL does (GL_ARB_fragment_shader_interlock): the calls ``beginInvocationInterlockARB()`` and
``endInvocationInterlockARB()``, and a stage layout such as ``layout(pixel_interlock_ordered) in;``. The SPIR-V
generator maps them back with the same tables (:mod:`crosstl.translator.codegen.spirv_fragment_interlock`), so a
round trip keeps the instructions, the mode and its capability.
"""

from __future__ import annotations

from ...translator.codegen.spirv_fragment_interlock import (
    INTERLOCK_CALLS,
    INTERLOCK_MODES,
)

__all__ = ["SPIRV_INTERLOCK_FUNCTIONS", "SPIRV_INTERLOCK_LAYOUTS"]

# OpBeginInvocationInterlockEXT -> beginInvocationInterlockARB, OpEndInvocationInterlockEXT -> its end call.
SPIRV_INTERLOCK_FUNCTIONS = {
    opcode: function for function, opcode in sorted(INTERLOCK_CALLS.items())
}
# PixelInterlockOrderedEXT -> pixel_interlock_ordered, and the other three modes.
SPIRV_INTERLOCK_LAYOUTS = {
    mode: layout for layout, (mode, _capability) in sorted(INTERLOCK_MODES.items())
}
