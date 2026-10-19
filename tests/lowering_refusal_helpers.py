"""Helpers for tests of constructs an emitter must refuse rather than lower to a placeholder."""

from __future__ import annotations

import pytest

from crosstl.backend.import_refusals import DirectXImportRefused, MetalImportRefused
from crosstl.translator.codegen.glsl_refusals import (
    OpenGLLoweringRefused,
    WebGLLoweringRefused,
)
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.lowering_refusals import LoweringRefusals

OPENGL_REFUSAL_CODE = "project.translate.opengl-lowering-unsupported"
WEBGL_REFUSAL_CODE = "project.translate.webgl-lowering-unsupported"
DIRECTX_REFUSAL_CODE = "project.translate.directx-lowering-unsupported"
DIRECTX_IMPORT_REFUSAL_CODE = "project.translate.directx-import-unsupported"
METAL_IMPORT_REFUSAL_CODE = "project.translate.metal-import-unsupported"


def refused(error_type, generate, *args, **kwargs):
    """Run a generation that must raise ``error_type`` and return the refusal message."""
    with pytest.raises(error_type) as refusal:
        generate(*args, **kwargs)
    assert type(refusal.value) is error_type
    assert refusal.value.code == error_type.code
    return str(refusal.value)


def partial_output(error_type, generate, *args, **kwargs):
    """Return the program rendered around refused constructs, and the refusal message.

    The public API returns no program while a refusal is pending, which ``refused`` checks first; the second run
    keeps the rendered text so that tests can still check the constructs that were lowered.
    """
    message = refused(error_type, generate, *args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LoweringRefusals, "raise_pending", lambda self, cause=None: None)
        output = generate(*args, **kwargs)
    return output, message


def refused_glsl(generate, *args, **kwargs):
    return refused(OpenGLLoweringRefused, generate, *args, **kwargs)


def glsl_partial_output(generate, *args, **kwargs):
    return partial_output(OpenGLLoweringRefused, generate, *args, **kwargs)


def refused_webgl(generate, *args, **kwargs):
    return refused(WebGLLoweringRefused, generate, *args, **kwargs)


def refused_hlsl(generate, *args, **kwargs):
    return refused(DirectXLoweringRefused, generate, *args, **kwargs)


def hlsl_partial_output(generate, *args, **kwargs):
    return partial_output(DirectXLoweringRefused, generate, *args, **kwargs)


def emitter_refusal(codegen, helper, *args, **kwargs):
    """Call an emitter helper that must refuse; return its placeholder and the reasons it recorded.

    The helper runs with the emitter's own sink active, as during a generation, so that refusals recorded through
    ``refuse_lowering`` and through the shared helpers both land in it.
    """
    sink = codegen.refusals
    sink.reset()
    with sink.active():
        placeholder = helper(*args, **kwargs)
    reasons = "; ".join(sink.reasons)
    sink.reset()
    assert reasons, f"{getattr(helper, '__name__', helper)} recorded no refusal"
    return placeholder, reasons


def shared_refusal(helper, *args, **kwargs):
    """Call a helper that must refuse inside a generation; return its placeholder and the reasons it recorded."""
    sink = LoweringRefusals(OpenGLLoweringRefused)
    with sink.active():
        placeholder = helper(*args, **kwargs)
    assert sink.reasons, f"{getattr(helper, '__name__', helper)} recorded no refusal"
    return placeholder, "; ".join(sink.reasons)


def refused_hlsl_import(generate, *args, **kwargs):
    return refused(DirectXImportRefused, generate, *args, **kwargs)


def hlsl_import_partial_output(generate, *args, **kwargs):
    return partial_output(DirectXImportRefused, generate, *args, **kwargs)


def refused_metal_import(generate, *args, **kwargs):
    return refused(MetalImportRefused, generate, *args, **kwargs)


def metal_import_partial_output(generate, *args, **kwargs):
    return partial_output(MetalImportRefused, generate, *args, **kwargs)
