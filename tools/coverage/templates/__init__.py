"""Probe templates for the language-level features of :mod:`.constructs`, in each target language.

A kernel template gives the statements of a compute kernel (``body``), what precedes them (``declarations``; for
GLSL extensions and WGSL ``enable`` directives, ``prelude``) and extra entry-point parameters (``params``). The kernel
around them is the language's :class:`~tools.coverage.dialect.Dialect` kernel: invocation index ``i``, input buffers
``fin`` (float) and ``iin`` (int), output buffer ``outp`` (float), at bindings 0, 1 and 2. A stage template is a whole
``program``. Templates may write ``$vec4``, ``$mat2`` and similar for the language's spelling of a type.

:data:`ABSENT` marks a feature the language does not have; a feature missing from a language's table has no template
yet.
"""

from __future__ import annotations

from .base import ABSENT, C_FAMILY, Template
from .compute import CUDA, HIP, MOJO, OPENCL, RUST
from .graphics import GLSL, HLSL, METAL, SLANG, WGSL
from .webgl import WEBGL

__all__ = ["ABSENT", "C_FAMILY", "TEMPLATES", "Template"]

TEMPLATES = {
    "cuda": CUDA,
    "hip": HIP,
    "mojo": MOJO,
    "opencl": OPENCL,
    "rust": RUST,
    "glsl": GLSL,
    "hlsl": HLSL,
    "metal": METAL,
    "slang": SLANG,
    "webgl": WEBGL,
    "wgsl": WGSL,
    "spirv": {},
}
