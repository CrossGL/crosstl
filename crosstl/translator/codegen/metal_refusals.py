"""Typed refusal for constructs the Metal emitter cannot lower.

The Metal emitter used to write ``/* unsupported Metal ...: reason */`` next to a zero value, an empty statement
or a stub helper and return the program. Every such site now records its reason through
:meth:`MetalCodeGen.refuse_lowering`, and generation raises :class:`MetalLoweringRefused` listing all of them.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefusals, LoweringRefused

__all__ = ["MetalLoweringRefused", "metal_refusals", "split_shared_fallback"]


class MetalLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful Metal lowering."""

    target_name = "metal"
    label = "Metal"
    code = f"{CODE_PREFIX}metal-lowering-unsupported"


def metal_refusals():
    return LoweringRefusals(MetalLoweringRefused)


_SHARED_FALLBACK = re.compile(r"/\*\s*unsupported Metal (?P<reason>.*?)\s*\*/", re.S)


def split_shared_fallback(text):
    """Split a fallback built by the backend-neutral helpers into its reasons and the remaining expression.

    The image-access helpers shared by several emitters return ``/* unsupported Metal <reason> */ <value>``. The
    Metal emitter refuses each reason and keeps only ``<value>`` as its placeholder.
    """
    text = str(text)
    reasons = [
        " ".join(match.group("reason").split())
        for match in _SHARED_FALLBACK.finditer(text)
    ]
    remainder = " ".join(_SHARED_FALLBACK.sub(" ", text).split())
    return reasons, remainder
