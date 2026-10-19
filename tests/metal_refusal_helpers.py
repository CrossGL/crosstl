"""Helpers for tests of Metal constructs that must be refused rather than lowered to a placeholder."""

from __future__ import annotations

import pytest

from crosstl.translator.codegen.lowering_refusals import LoweringRefusals
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

METAL_REFUSAL_CODE = "project.translate.metal-lowering-unsupported"


def refused_metal(generate, *args, **kwargs):
    """Run a Metal generation that must be refused and return the refusal message."""
    with pytest.raises(MetalLoweringRefused) as refusal:
        generate(*args, **kwargs)
    assert refusal.value.code == METAL_REFUSAL_CODE
    return str(refusal.value)


def metal_partial_output(generate, *args, **kwargs):
    """Return the Metal rendered around refused constructs, and the refusal message.

    The public API returns no program while a refusal is pending, which ``refused_metal`` checks first; the
    second run keeps the rendered text so that tests can still check the constructs that were lowered.
    """
    message = refused_metal(generate, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LoweringRefusals, "raise_pending", lambda self, cause=None: None)
        output = generate(*args, **kwargs)
    return output, message


def recorded_refusals(codegen):
    """The refusal reasons a direct call to an emitter helper recorded, joined for substring checks."""
    return "; ".join(codegen.refusals.reasons)
