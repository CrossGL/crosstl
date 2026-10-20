"""HIP device functions from the headers hipcc compiles with.

clang's ``__clang_hip_math.h`` defines the HIP math functions (``sinf``, ``erf``) and their intrinsics (``__fadd_rn``,
``__expf``). ROCm's ``amd_hip_atomic.h`` defines the atomic functions, and ``amd_device_functions.h`` the integer
intrinsics, conversions, warp votes and barriers. All three are pinned in :mod:`..upstream` at the versions the
validating compiler ships, so the list names exactly what it defines. A name defined more than once, for several
types or architectures, is probed with its first definition.
"""

from __future__ import annotations

import re

from .. import crossgl
from ..model import Feature, Signature, Type
from .cuda import crossgl_name, cuda_type, parameter

# Definitions in the headers that are not HIP functions a kernel calls: the headers' own helpers, the C library and
# printf, and AMDGPU builtin wrappers the HIP API does not document.
LEFT_OUT = frozenset("""
    __suppress_unused_warning __static_assert_equal_size __make_mantissa __make_mantissa_base8
    __make_mantissa_base10 __make_mantissa_base16 hip_cas_expander hip_cas_extrema_expander printf abort memcpy
    memset __lastbit_u32_u64 __hip_hc_add8pk __hip_hc_sub8pk __hip_hc_mul8pk __hip_hc_memcpy __hip_hc_memset
    __local_to_generic __get_dynamicgroupbaseptr __amdgcn_get_dynamicgroupbaseptr __work_group_barrier __barrier
    __named_sync
    """.split())
_DEFINITION = re.compile(
    r"^(?:(?:__device__|__DEVICE__|__host__|static|inline)\s+)+([^;{}()#<>]*?)\b(\w+)\s*\(([^)]*)\)\s*\{",
    re.M,
)
_ATTRIBUTE = re.compile(r"__attribute__\s*\(\([^)]*\)\)")
# ``__RETURN_TYPE`` is the math header's macro for the predicates' result, ``bool`` in C++.
_SPELLINGS = {"int32_t": "int", "uint32_t": "unsigned int", "__RETURN_TYPE": "bool"}
_TYPE_WORDS = frozenset(
    "const unsigned signed long short int char float double bool void".split()
)


def _plain(text: str) -> str:
    """A declaration's type without its parameter name, with fixed-width and macro spellings resolved."""
    text = re.sub(
        r"\b(?:u?int32_t|__RETURN_TYPE)\b", lambda m: _SPELLINGS[m.group(0)], text
    ).strip()
    words = re.match(r"(.*?[\w*&])\s*\b(\w+)$", text)
    if words and words.group(2) not in _TYPE_WORDS:
        return words.group(1).strip()
    return text


def definitions(text: str, group: str) -> list[tuple[str, str, Signature]]:
    """``(name, group, signature)`` for every function the header defines, first definition of a name only."""
    found: dict[str, tuple[str, str, Signature]] = {}
    for result, name, params in _DEFINITION.findall(_ATTRIBUTE.sub("", text)):
        if name in LEFT_OUT or name in found:
            continue
        atomic = group == "atomic"
        parts = [p for p in params.split(",") if p.strip() and p.strip() != "void"]
        found[name] = (
            name,
            group,
            Signature(
                cuda_type(_plain(result)),
                tuple(
                    parameter(_plain(p), atomic, i == 0) for i, p in enumerate(parts)
                ),
            ),
        )
    return list(found.values())


def parse(
    math: str,
    atomic: str,
    device: str,
    glsl_names: set[str] | frozenset[str] = frozenset(),
) -> list[Feature]:
    entries = (
        definitions(math, "math")
        + definitions(atomic, "atomic")
        + definitions(device, "device")
    )
    double_names = {
        name for name, _, signature in entries if signature.result == Type("double")
    }
    seen: set[str] = set()
    features = []
    for name, group, signature in sorted(entries, key=lambda e: (e[1], e[0])):
        if name in seen:
            continue
        seen.add(name)
        features.append(
            Feature(
                "hip",
                "function",
                name,
                signature,
                group=group,
                crossgl=crossgl.spelling(
                    "hip", crossgl_name(name, signature, double_names), glsl_names
                ),
            )
        )
    return features
