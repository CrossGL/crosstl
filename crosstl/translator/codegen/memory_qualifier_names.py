"""The name a memory qualifier of a CrossGL resource goes by in the SPIR-V and HLSL generators.

The Metal importer keeps Metal's scoped coherence, ``coherent(device)`` (MSL 3.2, ``coherent`` buffers and
textures), as a ``ResourceMemoryQualifierNode`` whose text is ``coherent(device)``. The SPIR-V and HLSL generators
matched the qualifier text against ``coherent`` alone, so a Metal buffer declared ``coherent(device)`` lost its
``Coherent`` decoration in SPIR-V and its ``globallycoherent`` in HLSL.

:func:`memory_qualifier_name` names device- and threadgroup-scoped coherence ``coherent``. SPIR-V's ``Coherent``
under the GLSL450 memory model, and HLSL's ``globallycoherent`` (coherence across the device rather than the
thread group), are exact for device scope and stronger than threadgroup scope requires. System scope keeps its
scoped name: neither target has a coherence wider than the device.
"""

from __future__ import annotations

import re

__all__ = ["memory_qualifier_name"]

_SCOPED_COHERENCE = re.compile(r"coherent\((?:device|threadgroup)\)")


def memory_qualifier_name(qualifier):
    """The lower-case name of ``qualifier``: a string or a ``ResourceMemoryQualifierNode``."""
    text = str(qualifier).strip().lower()
    return "coherent" if _SCOPED_COHERENCE.fullmatch(text) else text
