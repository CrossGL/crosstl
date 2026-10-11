"""Reference-only overloads for Metal compilers predating shared float atomics."""

SHARED_FLOAT_REFERENCE = """#include <metal_stdlib>
using namespace metal;
#if __METAL_VERSION__ < 410
inline float atomic_load_explicit(threadgroup atomic_float* object, memory_order) {
    return as_type<float>(metal::atomic_load_explicit(
        reinterpret_cast<threadgroup atomic_uint*>(object), memory_order_relaxed));
}
inline void atomic_store_explicit(threadgroup atomic_float* object, float desired, memory_order) {
    metal::atomic_store_explicit(reinterpret_cast<threadgroup atomic_uint*>(object),
        as_type<uint>(desired), memory_order_relaxed);
}
inline bool atomic_compare_exchange_weak_explicit(threadgroup atomic_float* object,
    thread float* expected, float desired, memory_order, memory_order) {
    uint observed = as_type<uint>(*expected);
    bool success = metal::atomic_compare_exchange_weak_explicit(
        reinterpret_cast<threadgroup atomic_uint*>(object), &observed,
        as_type<uint>(desired), memory_order_relaxed, memory_order_relaxed);
    if (!success) *expected = as_type<float>(observed);
    return success;
}
#endif
"""


def shared_float_reference_flags(work):
    # Only the owned reduced fixtures use this relaxed-order reference. The
    # translated source and unchanged upstream MLX headers never include it.
    header = work / "shared_float_reference.h"
    header.write_text(SHARED_FLOAT_REFERENCE, encoding="utf-8")
    return ("-std=metal4.0", "-include", str(header))
