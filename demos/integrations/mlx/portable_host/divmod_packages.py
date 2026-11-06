"""Translate unchanged ``binary_two.metal`` DivMod entries on first use.

``DivMod`` is MLX's only two-output elementwise primitive. Its host port picks
the same kernel families as ``binary_op_gpu`` (scalar, vector and general
strided layouts), so each kernel/dtype pair is translated once from the pinned
source and cached by ``lazy_kernel_cache`` under a content identity that covers
the translator, this recipe and the kernel source.
"""

from demos.integrations.mlx.portable_host.lazy_kernel_cache import (
    KernelRecipe,
    LazyKernelCache,
    index_range_assertions,
    project_config,
)

SOURCE = "mlx/backend/metal/kernels/binary_two.metal"
CACHE = "divmod"
# binary_two.metal instantiates these N=1 and int-indexed kernels for each type.
KERNELS = ("ss", "sv", "vs", "vv", "g1", "g2", "g3", "gn2")
DTYPES = (
    "float16",
    "float32",
    "bfloat16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
)
ENTRIES = {
    f"{kernel}_DivMod{dtype}": (kernel, dtype) for kernel in KERNELS for dtype in DTYPES
}
# The pinned per-type profiles reproduce Metal's binary32 division, comparison,
# remainder and additive results, and its binary16 remainder quotient.
BINARY32_PROFILES = (
    ("binary32_division_profile", "rne-flush"),
    ("binary32_comparison_profile", "flush-subnormals"),
    ("binary32_remainder_profile", "flush-arithmetic-subnormals"),
    ("binary32_additive_profile", "rne-flush"),
)
PROFILES = {
    "float32": BINARY32_PROFILES,
    "bfloat16": BINARY32_PROFILES,
    "float16": (("binary16_remainder_profile", "binary32-quotient"),),
}
# OpenGL lowers the kernels' element indices to signed ints. divmod_dispatch
# bounds every output element and input span below 2**31, so these hold.
OPENGL_INDEX_EXPRESSIONS = (
    "offset + i",
    "a_idx",
    "b_idx",
    "out_idx",
    "out_idx++",
    "idx.x",
    "idx.y",
)


def signature(entry):
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown DivMod entry") from error


def options(target, entry):
    """The pinned float profiles and OpenGL index bounds of ``entry``."""
    _, dtype = signature(entry)
    profiles = "".join(
        f'{option} = "{value}"\n' for option, value in PROFILES.get(dtype, ())
    )
    if target == "opengl":
        profiles += index_range_assertions(SOURCE, OPENGL_INDEX_EXPRESSIONS)
    return profiles


def _config(output, target, entry):
    return project_config(output, target, SOURCE, entry, options(target, entry))


RECIPE = KernelRecipe("DivMod", SOURCE, signature, options, __file__)


def DivModPackageCache(root, directory, target):  # noqa: N802
    """The lazy ``binary_two`` DivMod cache of one target."""
    return LazyKernelCache(root, directory, target, RECIPE)
