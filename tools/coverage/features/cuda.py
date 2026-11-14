"""CUDA device functions from LLVM's CUDA wrapper headers.

Clang compiles CUDA without NVIDIA's math headers: ``__clang_cuda_math.h`` defines the CUDA Math API's standard
functions (``sinf``, ``erf``, ``fmaf``), ``__clang_cuda_device_functions.h`` its intrinsics (``__saturatef``,
``__popc``, ``__fadd_rn``) and the typed helpers behind the atomic functions, and ``__clang_cuda_intrinsics.h`` the
warp shuffles, votes, matches and reductions, the cache-hinted loads and stores and the funnel shifts. All three are
machine-readable and pinned in :mod:`..upstream`. Each typed helper such as ``__iAtomicAdd`` or ``__fAtomicAdd_block``
becomes the public function it implements (``atomicAdd``, ``atomicAdd_block``), probed with its ``int`` overload where
there is one.

The intrinsics header defines its functions in blocks guarded by the target architecture and CUDA version, and
through macros that stamp out one overload per type, so it is read as a compile with :data:`COMPILE` sees it
(:mod:`.preprocess`). Each of its functions is probed with its first overload whose types CrossGL has.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import STRING, VOID, Feature, Param, Signature, Type
from . import preprocess

# Definitions in the headers that are not CUDA device functions a kernel calls: the pre-Volta warp votes and
# shuffles, which CUDA 9 removed for compute capability 7.0 and later, a helper of those votes, the debugger trap and
# profiler counters, and C library functions. ``__isfinited``, ``__signbitd``, ``powi``, ``powif`` and
# ``__nvvm_get_smem_pointer`` wrap libdevice and NVVM builtins under names that only Clang's headers give them: nvcc on
# Linux, where the probes are validated, rejects a kernel that calls them as using an undefined identifier. The
# ``__nv_memcpy_async_*`` and ``__nv_associate_access_property`` helpers implement libcu++'s ``cuda::memcpy_async``
# and ``cuda::associate_access_property``, which a kernel calls instead.
LEFT_OUT = frozenset(
    "__all __any __ballot __shfl __shfl_up __shfl_down __shfl_xor __bool2mask __brkpt __trap __pm0 __pm1 __pm2"
    " __pm3 memcpy memset __isfinited __signbitd powi powif __nvvm_get_smem_pointer __nv_memcpy_async_shared_global_4"
    " __nv_memcpy_async_shared_global_8 __nv_memcpy_async_shared_global_16 __nv_associate_access_property".split()
)
# The macros nvcc 13.0 defines when it compiles a probe for ``validators.CUDA_ARCH`` (sm_80) as C++17, which decide
# the conditional blocks of ``__clang_cuda_intrinsics.h`` the probe sees. The sm_90 cluster functions are not among
# them.
COMPILE = {
    "__CUDA__": 1,
    "__CUDA_ARCH__": 800,
    "CUDA_VERSION": 13000,
    "__cplusplus": 201703,
}
# The public CUDA functions that the intrinsics header implements under Clang's own names, which nvcc does not define:
# the ``__match_any_sync`` and ``__match_all_sync`` overloads of NVIDIA's ``sm_70_rt.hpp``, and the address space
# conversions ``__cvta_generic_to_global`` and the rest.
_PUBLIC = (
    (re.compile(r"__match(?:32|64)_(any|all)_sync"), r"__match_\1_sync"),
    (re.compile(r"__nv_(cvta_\w+)_impl"), r"__\1"),
)
# An address space conversion ``__cvta_generic_to_<space>`` takes a generic pointer into the space it names and
# returns that space's address (CUDA C++ Programming Guide, "Address Space Conversion Functions"). A probe passes it a
# pointer of the kind its space holds: the input buffer for global memory, or the address of a local, a __shared__ or
# a __constant__ variable.
_GENERIC_TO = re.compile(r"__cvta_generic_to_(global|shared|constant|local)")
_SPACES = {
    "global": "buffer",
    "shared": "workgroup address",
    "constant": "constant address",
    "local": "address",
}
_INLINE = re.compile(
    r"(?:inline\s+__device__|__device__\s+inline)\s+([^;{}()#]*?)\b(\w+)\s*\(([^)]*)\)\s*\{"
)
_TYPE_WORDS = frozenset(
    "const unsigned signed long short int char float double bool void".split()
)
# ``long``, and so ``size_t``, is 64 bits wide in device code on Linux, where the headers are read and nvcc validates
# the probes.
_SCALARS = {
    "size_t": "uint64",
    "bool": "bool",
    "int": "int",
    "unsigned int": "uint",
    "unsigned": "uint",
    "float": "float",
    "double": "double",
    "long": "int64",
    "long int": "int64",
    "long long": "int64",
    "long long int": "int64",
    "unsigned long": "uint64",
    "unsigned long int": "uint64",
    "unsigned long long": "uint64",
    "unsigned long long int": "uint64",
}
_DEFINITION = re.compile(
    r"^__DEVICE(?:_VOID)?__\s+([^;{(]*?)\b(\w+)\s*\(([^)]*)\)", re.M
)
_ATOMIC = re.compile(r"__(i|u|ll|ull|ill|us|f|d)Atomic(\w+?)(_block|_system)?")
# Which typed helper stands for the public atomic function: the first of these the header has.
_ATOMIC_PREFERENCE = ("i", "u", "f", "ll", "ull", "ill", "d", "us")
_ATTRIBUTE = re.compile(r"__attribute__\s*\(\([^)]*\)\)")
# Block comments, such as the ``/* clock_t= */`` before clock's result type.
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
# Declarations only an MSVC build sees: the ``long double`` predicates ``__finitel``, ``__isinfl`` and ``__isnanl``
# that Microsoft's C library headers call. The probes are validated on Linux, where Clang's headers do not declare
# them.
_MSVC_ONLY = re.compile(r"^#ifdef _MSC_VER\n.*?^#endif[^\n]*\n", re.M | re.S)


def cuda_type(spelling: str) -> Type:
    """A CUDA value type; ``long double``, ``short``, ``char`` and the like stay opaque."""
    text = " ".join(spelling.replace("const ", "").split())
    if text == "void":
        return VOID
    if text in _SCALARS:
        return Type(_SCALARS[text])
    return Type.other(text)


def parameter(text: str, atomic: bool, first: bool) -> Param:
    text = re.sub(r"\b__\w+$", "", text.strip()).strip()
    if not text.endswith("*"):
        return Param(cuda_type(text))
    pointee = text.rstrip("*").strip()
    if text.count("*") > 1:
        return Param(Type.other(text))
    if " ".join(pointee.split()) == "const char":
        # A C string, such as the tag of nan and nanf, takes a string constant.
        return Param(STRING, "const")
    if pointee.startswith("const "):
        # A pointer to constant values, such as the vector norm reads, takes the address of a local.
        kind = cuda_type(pointee)
        return Param(kind, "address") if not kind.opaque else Param(Type.other(text))
    if atomic and first:
        return Param(cuda_type(pointee), "inout")
    return Param(cuda_type(pointee), "out")


def definitions(text: str, group: str) -> list[tuple[str, str, Signature, str]]:
    """``(name, group, signature, helper)`` for every function the header defines, first definition of a name only.
    ``helper`` is the typed atomic helper's own name, or empty."""
    text = _ATTRIBUTE.sub("", _COMMENT.sub("", _MSVC_ONLY.sub("", text)))
    found: dict[str, tuple[str, str, Signature, str]] = {}
    for result, name, params in _DEFINITION.findall(text):
        if name in LEFT_OUT:
            continue
        atomic = _ATOMIC.fullmatch(name)
        public = f"atomic{atomic.group(2)}{atomic.group(3) or ''}" if atomic else name
        parts = [p for p in params.split(",") if p.strip() and p.strip() != "void"]
        signature = Signature(
            cuda_type(result),
            tuple(parameter(p, bool(atomic), i == 0) for i, p in enumerate(parts)),
        )
        entry = (
            public,
            "atomic" if atomic else group,
            signature,
            name if atomic else "",
        )
        if public not in found or (atomic and _rank(name) < _rank(found[public][3])):
            found[public] = entry
    return list(found.values())


def _rank(helper: str) -> int:
    match = _ATOMIC.fullmatch(helper)
    return (
        _ATOMIC_PREFERENCE.index(match.group(1)) if match else len(_ATOMIC_PREFERENCE)
    )


def declared(text: str) -> str:
    """A parameter's declared type, without its default argument and its name."""
    text = text.split("=")[0].strip()
    words = re.match(r"(.*?[\w*&])\s*\b(\w+)$", text)
    if words and words.group(2) not in _TYPE_WORDS:
        return words.group(1).strip()
    return text


def _opaque(signature: Signature) -> bool:
    kinds = (signature.result, *(p.type for p in signature.params))
    return any(kind.opaque and kind != VOID for kind in kinds)


def intrinsics(text: str) -> list[tuple[str, str, Signature, str]]:
    """``(name, group, signature, "")`` for every public function ``__clang_cuda_intrinsics.h`` defines for a
    :data:`COMPILE` compile, with its first overload whose types CrossGL has, or its first overload.
    """
    source = preprocess.expand(preprocess.keep(_ATTRIBUTE.sub("", text), COMPILE))
    found: dict[str, tuple[str, str, Signature, str]] = {}
    for result, name, params in _INLINE.findall(source):
        if name in LEFT_OUT:
            continue
        public = name
        for pattern, replacement in _PUBLIC:
            public = pattern.sub(replacement, public)
        parts = [
            declared(p) for p in params.split(",") if p.strip() not in ("", "void")
        ]
        signature = Signature(
            cuda_type(result),
            tuple(parameter(p, False, i == 0) for i, p in enumerate(parts)),
        )
        space = _GENERIC_TO.fullmatch(public)
        if space:
            pointer = Param(Type("float"), _SPACES[space.group(1)])
            signature = Signature(signature.result, (pointer,))
        if public not in found or (
            _opaque(found[public][2]) and not _opaque(signature)
        ):
            found[public] = (public, "intrinsics", signature, "")
    return list(found.values())


def crossgl_name(name: str, signature: Signature, double_names: set[str]) -> str:
    """CrossGL overloads by type, so a single-precision C function is spelled as its double-precision twin."""
    base = name[:-1] if name.endswith("f") and name[:-1] in double_names else name
    if signature.result == Type("float") and base != name:
        return base
    return name


def parse(
    math: str,
    device: str,
    glsl_names: set[str] | frozenset[str] = frozenset(),
    intrinsic: str = "",
) -> list[Feature]:
    entries = definitions(math, "math") + definitions(device, "intrinsics")
    defined = {name for name, *_ in entries}
    entries += [entry for entry in intrinsics(intrinsic) if entry[0] not in defined]
    double_names = {
        name for name, _, signature, _ in entries if signature.result == Type("double")
    }
    features = []
    for name, group, signature, _ in sorted(entries, key=lambda e: (e[1], e[0])):
        features.append(
            Feature(
                "cuda",
                "function",
                name,
                signature,
                group=group,
                crossgl=spellings.spelling(
                    "cuda", crossgl_name(name, signature, double_names), glsl_names
                ),
            )
        )
    return features
