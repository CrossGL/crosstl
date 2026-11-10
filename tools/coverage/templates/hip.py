"""Probe templates for HIP."""

from __future__ import annotations

from . import (
    formats,
    images,
    qualifiers,
    samplers,
    storage,
    subgroups,
    textures,
)
from .base import Template, scalar
from .cuda import CUDA

# HIP keeps CUDA's kernel language and renames the runtime types.
HIP = {
    **CUDA,
    **subgroups.HIP,
    **qualifiers.HIP,
    **textures.HIP,
    **images.HIP,
    **formats.HIP,
    **storage.HIP,
    **samplers.KERNELS,
    "half": scalar("__half", "fin", "#include <hip/hip_fp16.h>"),
    "sampled-texture": Template(
        CUDA["sampled-texture"].body, params="hipTextureObject_t tex"
    ),
    "storage-image": Template(
        CUDA["storage-image"].body, params="hipSurfaceObject_t img"
    ),
}
