"""Translate the unchanged ``arange.metal`` entries no installed family covers.

The pinned source instantiates ``arange<type>`` for every real dtype but
Boolean. The base family installs float32 and the 32- and 64-bit integers;
the 8- and 16-bit integers, float16 and bfloat16 are translated once from the
pinned source on first use and cached by ``lazy_kernel_cache``, with the same
float profiles as the base family.
"""

from demos.integrations.mlx.portable_host.lazy_kernel_cache import (
    KernelRecipe,
    LazyKernelCache,
)
from demos.integrations.mlx.portable_host.packages import ARANGE_ENTRIES, SOURCE

CACHE = "aranges"
PREFIX = "arange"
# instantiate_arange in the pinned arange.metal, in source order.
INSTANTIATED = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
)
ENTRIES = {
    f"{PREFIX}{dtype}": dtype
    for dtype in INSTANTIATED
    if f"{PREFIX}{dtype}" not in ARANGE_ENTRIES
}
# The base family's project options.
PROFILES = (
    ("binary32_fma_profile", "rne-flush"),
    ("binary32_sqrt_profile", "flush-subnormals"),
    ("binary32_rsqrt_profile", "flush-subnormals"),
)


def signature(entry):
    """The element type of a lazily translated ``arange`` entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown lazy arange entry") from error


def options(target, entry):
    """The project source options of ``entry``; every target shares them."""
    signature(entry)
    return "".join(f'{option} = "{value}"\n' for option, value in PROFILES)


RECIPE = KernelRecipe("Arange", SOURCE, signature, options, __file__)


def cache(root, directory, target):
    """One target's lazy ``arange`` packages under ``directory``."""
    return LazyKernelCache(root, directory, target, RECIPE)
