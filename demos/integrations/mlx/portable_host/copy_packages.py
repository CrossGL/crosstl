"""Translate unchanged ``copy.metal`` casts and general copies on first use.

``v_copy`` converts a contiguous vector between any two MLX dtypes, including
``complex64``, and ``ggn2_dynamic_copy`` copies one dtype's storage between
strided layouts, as ``copy_gpu_inplace`` does for an unknown rank. Both are
instantiated for every GPU dtype, so each entry is translated once from the pinned source and cached
by ``lazy_kernel_cache``. The pinned binary32 profiles and the OpenGL offset
bounds are the ones the prebuilt copy families use.
"""

from demos.integrations.mlx.portable_host.element_storage import STORAGE_DTYPES
from demos.integrations.mlx.portable_host.lazy_kernel_cache import (
    KernelRecipe,
    LazyKernelCache,
    index_range_assertions,
)
from demos.integrations.mlx.portable_host.packages import (
    COPY_SOURCE,
    OPENGL_COPY_INDEX_ASSERTIONS,
)

CACHE = "copies"
VECTOR, GENERAL = "v_copy", "ggn2_dynamic_copy"
ENTRY_PREFIXES = (VECTOR, GENERAL)
ENTRIES = {
    **{
        f"{VECTOR}{source}{destination}": (VECTOR, source, destination)
        for source in STORAGE_DTYPES
        for destination in STORAGE_DTYPES
        if source != destination
    },
    **{f"{GENERAL}{dtype}{dtype}": (GENERAL, dtype, dtype) for dtype in STORAGE_DTYPES},
}
PROFILES = (
    'binary32_fma_profile = "rne-flush"\n'
    'binary32_sqrt_profile = "flush-subnormals"\n'
    'binary32_rsqrt_profile = "flush-subnormals"\n'
)


def signature(entry):
    """``(kernel, source dtype, destination dtype)`` of a copy entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown copy entry") from error


def options(target, entry):
    kernel, _, _ = signature(entry)
    if target == "opengl" and kernel == GENERAL:
        return PROFILES + index_range_assertions(
            COPY_SOURCE, OPENGL_COPY_INDEX_ASSERTIONS
        )
    return PROFILES


RECIPE = KernelRecipe("Copy", COPY_SOURCE, signature, options, __file__)


def CopyPackageCache(root, directory, target):  # noqa: N802
    """The lazy ``copy.metal`` cache of one target."""
    return LazyKernelCache(root, directory, target, RECIPE)
