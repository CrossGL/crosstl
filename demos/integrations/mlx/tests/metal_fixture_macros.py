"""Macros a synthetic MLX Metal fixture defines itself.

A real MLX kernel source includes ``mlx/backend/metal/kernels/defines.h``, which
defines ``instantiate_kernel``. A fixture that calls the macro without that
header has to define it in front of the first call, the same way, so the Metal
front end sees the explicit instantiation the header expands to (#4670).
"""

INSTANTIATE_KERNEL_DEFINE = (
    "#define instantiate_kernel(name, func, ...) template [[host_name(name)]] "
    "[[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;"
)
