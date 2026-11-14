"""CUDA device functions from LLVM's CUDA wrapper headers.

Clang compiles CUDA without NVIDIA's math headers: ``__clang_cuda_math.h`` defines the CUDA Math API's standard
functions (``sinf``, ``erf``, ``fmaf``) and ``__clang_cuda_device_functions.h`` its intrinsics (``__saturatef``,
``__popc``, ``__fadd_rn``) and the typed helpers behind the atomic functions. Both are machine-readable and pinned in
:mod:`..upstream`. Each typed helper such as ``__iAtomicAdd`` or ``__fAtomicAdd_block`` becomes the public function it
implements (``atomicAdd``, ``atomicAdd_block``), probed with its ``int`` overload where there is one.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import STRING, VOID, Feature, Param, Signature, Type

# Definitions in the headers that are not CUDA device functions a kernel calls: the pre-Volta warp votes CUDA 9
# removed, a helper of those votes, the debugger trap and profiler counters, and C library functions. The last four
# wrap libdevice functions under names that only Clang's headers give them: nvcc on Linux, where the probes are
# validated, rejects a kernel that calls them as using an undefined identifier.
LEFT_OUT = frozenset(
    "__all __any __ballot __bool2mask __brkpt __trap __pm0 __pm1 __pm2 __pm3 memcpy memset"
    " __isfinited __signbitd powi powif".split()
)
# ``long`` is 64 bits wide in device code on Linux, where the headers are read and nvcc validates the probes.
_SCALARS = {
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


def crossgl_name(name: str, signature: Signature, double_names: set[str]) -> str:
    """CrossGL overloads by type, so a single-precision C function is spelled as its double-precision twin."""
    base = name[:-1] if name.endswith("f") and name[:-1] in double_names else name
    if signature.result == Type("float") and base != name:
        return base
    return name


def parse(
    math: str, device: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    entries = definitions(math, "math") + definitions(device, "intrinsics")
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
