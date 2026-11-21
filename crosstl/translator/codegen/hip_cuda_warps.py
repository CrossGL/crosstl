"""HIP output of the CUDA warp intrinsics a CrossGL program calls.

CrossGL keeps CUDA's warp intrinsics (``__shfl_sync`` ... ``__syncwarp``) under their CUDA names
(:mod:`crosstl.translator.semantics.warp`), and CUDA defines them over a warp of 32 lanes. A HIP wavefront has
``warpSize`` lanes, 32 or 64 depending on the device, and hipcc 5.7 declares none of the ``_sync`` forms; ROCm 6.2
added them behind ``HIP_ENABLE_WARP_SYNC_BUILTINS``, with 64-bit masks over the whole wavefront, which is not a
CUDA warp either. HIP output used to print these calls verbatim, which hipcc 5.7 rejects, and ``__shfl_down_sync``
through a helper that returned its own operand unless ``HIP_ENABLE_WARP_SYNC_BUILTINS`` was defined.

AMD hardware forms wavefronts from consecutive linear thread indices, as CUDA forms warps, so wavefront lane ``l``
(``__lane_id()``) is lane ``l & 31`` of the CUDA warp that starts at wavefront lane ``l & ~31``. HIP output computes
each intrinsic over that 32-lane segment, with helpers built on HIP's own wavefront functions
(``amd_warp_functions.h``):

* the shuffles call ``__shfl``, ``__shfl_up``, ``__shfl_down`` and ``__shfl_xor`` with the CUDA ``width`` (default
  32). For a power-of-two width up to 32 these pick the source lane inside the caller's ``width``-lane segment, or
  return the caller's own value, by the same rules as PTX ``shfl.sync`` (PTX ISA, "shfl.sync"), whose lane operand
  is read only in its low five bits, so the helpers pass ``lane & 31``;
* ``__ballot_sync`` is the caller's 32 bits of ``__ballot`` and the mask; ``__any_sync``, ``__all_sync`` and
  ``__uni_sync`` test those bits against the mask's active lanes (CUDA C++ Programming Guide, "Warp Vote
  Functions": the vote is over the non-exited threads of the mask); ``__activemask`` is the caller's 32 bits of
  ``__ballot(1)``;
* ``__reduce_*_sync`` and ``__match_*_sync`` read each of the 32 lanes with ``__shfl`` and fold or compare the
  values of the mask's active lanes (the integer add wraps, as ``redux.sync`` does; the matches compare bit
  patterns, as ``match.sync`` does);
* ``__syncwarp`` orders the memory of the wavefront's lanes with a release and an acquire fence at wavefront scope
  around ``__builtin_amdgcn_wave_barrier``; the lanes of a wavefront execute together, so no lane waits for another.

The lanes of a wavefront execute in lockstep, so the lanes a mask names execute the intrinsic together wherever the
program is convergent, which is what CUDA requires for a defined result ("all non-exited threads named in mask must
execute the same intrinsic with the same mask"); a shuffle's source lane outside the mask is undefined in CUDA.
"""

from __future__ import annotations

from ..refusals import HipRefusal
from .cuda_pointer_operands import pointer_operand_arguments

__all__ = ["CUDA_WARP_INTRINSIC_ARITIES", "hip_cuda_warp_call"]

# Argument counts of the CUDA warp intrinsics (CUDA C++ Programming Guide, "Warp Vote Functions", "Warp Match
# Functions", "Warp Reduce Functions", "Warp Shuffle Functions" and "__syncwarp").
CUDA_WARP_INTRINSIC_ARITIES = {
    "__shfl_sync": (3, 4),
    "__shfl_up_sync": (3, 4),
    "__shfl_down_sync": (3, 4),
    "__shfl_xor_sync": (3, 4),
    "__ballot_sync": (2,),
    "__any_sync": (2,),
    "__all_sync": (2,),
    "__uni_sync": (2,),
    "__activemask": (0,),
    "__syncwarp": (0, 1),
    "__match_any_sync": (2,),
    "__match_all_sync": (3,),
    "__reduce_add_sync": (2,),
    "__reduce_min_sync": (2,),
    "__reduce_max_sync": (2,),
    "__reduce_and_sync": (2,),
    "__reduce_or_sync": (2,),
    "__reduce_xor_sync": (2,),
}

_WARP_BITS = (
    "// CUDA warps are 32 lanes: wavefront lane l is lane l & 31 of the warp at wavefront lane l & ~31.\n"
    "__device__ inline unsigned int cgl_cuda_warp_bits(unsigned long long wave_bits) {\n"
    "    return static_cast<unsigned int>(wave_bits >> (__lane_id() & ~31u));\n"
    "}"
)
_BALLOT = (
    "__device__ inline unsigned int cgl_cuda_ballot_sync(unsigned int mask, int predicate) {\n"
    "    return cgl_cuda_warp_bits(__ballot(predicate)) & mask;\n"
    "}"
)


def _shuffle(name, hip_function, lane_type, lane_mask):
    return (
        "template <typename T>\n"
        f"__device__ inline T cgl_cuda_{name}(unsigned int mask, T value, {lane_type} lane, int width = 32) {{\n"
        "    (void)mask;\n"
        f"    return {hip_function}(value, lane & {lane_mask}, width);\n"
        "}"
    )


def _reduction(name, initial, fold):
    return (
        f"__device__ inline unsigned int cgl_cuda_{name}(unsigned int mask, unsigned int value) {{\n"
        "    unsigned int lanes = cgl_cuda_ballot_sync(mask, 1);\n"
        f"    unsigned int result = {initial};\n"
        "    for (int lane = 0; lane < 32; ++lane) {\n"
        "        unsigned int other = __shfl(value, lane, 32);\n"
        "        if ((lanes >> lane) & 1u) {\n"
        f"            result {fold}= other;\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}"
    )


def _extremum(name, comparison):
    return (
        "template <typename T>\n"
        f"__device__ inline T cgl_cuda_{name}(unsigned int mask, T value) {{\n"
        "    unsigned int lanes = cgl_cuda_ballot_sync(mask, 1);\n"
        "    T result = value;\n"
        "    for (int lane = 0; lane < 32; ++lane) {\n"
        "        T other = __shfl(value, lane, 32);\n"
        f"        if (((lanes >> lane) & 1u) && other {comparison} result) {{\n"
        "            result = other;\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}"
    )


# Each helper: (its HIP text, the helpers it calls).
_HELPERS = {
    "cgl_cuda_warp_bits": (_WARP_BITS, ()),
    "cgl_cuda_ballot_sync": (_BALLOT, ("cgl_cuda_warp_bits",)),
    "cgl_cuda_shfl_sync": (_shuffle("shfl_sync", "__shfl", "int", "31"), ()),
    "cgl_cuda_shfl_up_sync": (
        _shuffle("shfl_up_sync", "__shfl_up", "unsigned int", "31u"),
        (),
    ),
    "cgl_cuda_shfl_down_sync": (
        _shuffle("shfl_down_sync", "__shfl_down", "unsigned int", "31u"),
        (),
    ),
    "cgl_cuda_shfl_xor_sync": (
        _shuffle("shfl_xor_sync", "__shfl_xor", "int", "31"),
        (),
    ),
    "cgl_cuda_any_sync": (
        "__device__ inline int cgl_cuda_any_sync(unsigned int mask, int predicate) {\n"
        "    return cgl_cuda_ballot_sync(mask, predicate) != 0u;\n"
        "}",
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_all_sync": (
        "__device__ inline int cgl_cuda_all_sync(unsigned int mask, int predicate) {\n"
        "    return cgl_cuda_ballot_sync(mask, predicate) == cgl_cuda_ballot_sync(mask, 1);\n"
        "}",
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_uni_sync": (
        "__device__ inline int cgl_cuda_uni_sync(unsigned int mask, int predicate) {\n"
        "    unsigned int set = cgl_cuda_ballot_sync(mask, predicate);\n"
        "    return set == 0u || set == cgl_cuda_ballot_sync(mask, 1);\n"
        "}",
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_activemask": (
        "__device__ inline unsigned int cgl_cuda_activemask() {\n"
        "    return cgl_cuda_warp_bits(__ballot(1));\n"
        "}",
        ("cgl_cuda_warp_bits",),
    ),
    "cgl_cuda_syncwarp": (
        "__device__ inline void cgl_cuda_syncwarp(unsigned int mask = 0xffffffffu) {\n"
        "    (void)mask;\n"
        '    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");\n'
        "    __builtin_amdgcn_wave_barrier();\n"
        '    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");\n'
        "}",
        (),
    ),
    "cgl_cuda_warp_value_bits": (
        "template <typename T>\n"
        "__device__ inline unsigned long long cgl_cuda_warp_value_bits(T value) {\n"
        '    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "CUDA warp matches compare 32-bit or 64-bit values");\n'
        "    unsigned long long bits = 0ull;\n"
        "    __builtin_memcpy(&bits, &value, sizeof(T));\n"
        "    return bits;\n"
        "}",
        (),
    ),
    "cgl_cuda_match_any_sync": (
        "template <typename T>\n"
        "__device__ inline unsigned int cgl_cuda_match_any_sync(unsigned int mask, T value) {\n"
        "    unsigned int lanes = cgl_cuda_ballot_sync(mask, 1);\n"
        "    unsigned long long bits = cgl_cuda_warp_value_bits(value);\n"
        "    unsigned int matches = 0u;\n"
        "    for (int lane = 0; lane < 32; ++lane) {\n"
        "        unsigned long long other = __shfl(bits, lane, 32);\n"
        "        if (((lanes >> lane) & 1u) && other == bits) {\n"
        "            matches |= 1u << lane;\n"
        "        }\n"
        "    }\n"
        "    return matches;\n"
        "}",
        ("cgl_cuda_ballot_sync", "cgl_cuda_warp_value_bits"),
    ),
    "cgl_cuda_match_all_sync": (
        "template <typename T>\n"
        "__device__ inline unsigned int cgl_cuda_match_all_sync(unsigned int mask, T value, int* pred) {\n"
        "    int all = cgl_cuda_match_any_sync(mask, value) == cgl_cuda_ballot_sync(mask, 1);\n"
        "    *pred = all;\n"
        "    return all ? mask : 0u;\n"
        "}",
        ("cgl_cuda_ballot_sync", "cgl_cuda_match_any_sync"),
    ),
    "cgl_cuda_reduce_add_sync": (
        "template <typename T>\n"
        "__device__ inline T cgl_cuda_reduce_add_sync(unsigned int mask, T value) {\n"
        "    unsigned int lanes = cgl_cuda_ballot_sync(mask, 1);\n"
        "    unsigned int sum = 0u;\n"
        "    for (int lane = 0; lane < 32; ++lane) {\n"
        "        unsigned int other = __shfl(static_cast<unsigned int>(value), lane, 32);\n"
        "        if ((lanes >> lane) & 1u) {\n"
        "            sum += other;\n"
        "        }\n"
        "    }\n"
        "    return static_cast<T>(sum);\n"
        "}",
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_reduce_min_sync": (
        _extremum("reduce_min_sync", "<"),
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_reduce_max_sync": (
        _extremum("reduce_max_sync", ">"),
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_reduce_and_sync": (
        _reduction("reduce_and_sync", "~0u", "&"),
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_reduce_or_sync": (
        _reduction("reduce_or_sync", "0u", "|"),
        ("cgl_cuda_ballot_sync",),
    ),
    "cgl_cuda_reduce_xor_sync": (
        _reduction("reduce_xor_sync", "0u", "^"),
        ("cgl_cuda_ballot_sync",),
    ),
}


def _require(codegen, helper_name):
    """Add ``helper_name`` and, before it, every helper it calls to the HIP output's helper functions."""
    if helper_name in codegen.helper_functions:
        return
    text, dependencies = _HELPERS[helper_name]
    for dependency in dependencies:
        _require(codegen, dependency)
    codegen.helper_functions[helper_name] = text


def hip_cuda_warp_call(codegen, func_name, raw_args, args):
    """The HIP expression for a call of the CUDA warp intrinsic ``func_name``, or ``None`` for any other call."""
    expected_counts = CUDA_WARP_INTRINSIC_ARITIES.get(func_name)
    if expected_counts is None:
        return None
    if len(raw_args) not in expected_counts:
        expected = " or ".join(str(count) for count in expected_counts)
        raise HipRefusal(
            f"CUDA warp intrinsic {func_name} takes {expected} arguments, got {len(raw_args)}"
        )
    if func_name == "__match_all_sync":
        # CrossGL may pass the predicate as an out operand; the helper, like CUDA, takes its address.
        args = pointer_operand_arguments(
            codegen, func_name, raw_args, args, HipRefusal, "HIP"
        )
    helper_name = "cgl_cuda_" + func_name[2:]
    _require(codegen, helper_name)
    return f"{helper_name}({', '.join(args)})"
