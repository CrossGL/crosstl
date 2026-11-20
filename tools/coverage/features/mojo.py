"""Mojo functions: the public API of ``std.math``, ``std.bit`` and ``std.atomic``, and of MAX's GPU warp, block,
index and synchronization modules.

A package's public API is what its ``__init__.mojo`` imports from its modules, so the list is the lower-case names
those files export (the capitalised ones are traits and types). ``std.math`` also exports the floating-point predicates
of ``std.utils.numerics``. ``max.gpu.primitives.warp`` and ``max.gpu.primitives.block`` are modules no package
re-exports, so their public API is every module-level definition whose name does not start with an underscore;
``max.gpu.sync`` exports its barriers from its ``sync`` module. ``max.gpu.primitives.id`` defines ``warp_id`` and
``sm_id`` and re-exports ``lane_id`` and the index values (``thread_idx``, ``global_idx`` and the rest) from
``std._gpu``; those values are not functions, and the compute construct probes read them. ``std.atomic`` exports
``fence`` and the type ``Atomic``. A probe calls an Atomic method through the type: a static method on a pointer to
the output buffer's element, and a method only an Atomic value has on that element viewed as an Atomic. Each name is
probed with its first definition, a static one where both exist. Mojo functions are generic over a ``DType`` and a
SIMD width; a probe calls them on scalars, ``Float32``
unless the definition requires an integral type, as all of ``std.bit`` does. A compile-time parameter no argument
fixes is written into the call, as in ``nan[dtype=DType.float32]()``.

MAX's cluster and programmatic dependent launch functions (``max.gpu.primitives.cluster`` and
``max.gpu.primitives.grid_controls``) and its tensor memory accelerator copies (the ``cp_async_bulk_tensor`` functions
of ``max.gpu.memory``) are not read. They exist only on NVIDIA GPUs from sm_90, and the CUDA row does not measure
CUDA's thread block clusters, dependent launches or tensor memory accelerator either.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .. import spellings
from ..execution import THREADS
from ..model import REDUCE_CLOSURE, VOID, Feature, Param, Signature, Type
from .common import prototype
from .mojo_source import (
    Definition,
    argument_default,
    compile_parameters,
    definitions,
    exports,
    methods,
    public,
    split_commas,
)

# The DType names a probe can spell, and the scalar type of :mod:`..model` each one is.
_DTYPES = {
    "float32": "float",
    "float64": "double",
    "int32": "int",
    "uint32": "uint",
    "bool": "bool",
}
_DTYPE_OF = {scalar: dtype for dtype, scalar in _DTYPES.items()}
# ``Int`` is the machine word, 64 bits wide on the 64-bit hosts and GPUs Mojo targets.
_NAMED = {
    "Float32": "float",
    "Float64": "double",
    "Int32": "int",
    "UInt32": "uint",
    "UInt64": "uint64",
    "Bool": "bool",
    "Int": "int64",
}
_SPELLED = {scalar: name for name, scalar in _NAMED.items()}
# Argument conventions a probe can pass a local to; ``mut`` makes the argument read-modify-write.
_CONVENTIONS = frozenset({"mut", "read", "var", "ref", "owned"})
# Compile-time parameter kinds a probe can fill in with a constant, and the constant it uses.
_LITERALS = {"Int": "1", "Int32": "1", "SIMDLength": "1", "Bool": "True"}
# Operations CrossGL has no spelling for, with the reason.
UNSPELLED = {
    "divmod": "no CrossGL function returns a quotient and a remainder together",
    "warp.match_all": (
        "CrossGL's lane match over the whole warp is CUDA's __match_all_sync, which also returns whether the lanes "
        "agree through a pointer"
    ),
    "warp.reduce": "no CrossGL warp reduction applies a function the kernel passes",
    "warp.lane_group_reduce": (
        "no CrossGL warp reduction applies a function the kernel passes"
    ),
    "Atomic.compare_exchange": (
        "CrossGL's compare-and-swap returns the value it found, and compare_exchange returns whether it stored and "
        "writes the value it found to its expected argument"
    ),
}
# MAX's GPU modules, as a probe imports them (:meth:`..kernels.Mojo.enable`). A warp or block function is called
# through its module, as ``warp.sum``, because their sum, max and min have the names of built-in and std.math
# functions; a synchronization or index function is imported by name, and an Atomic method through its type.
WARP = "max.gpu.primitives.warp"
BLOCK = "max.gpu.primitives.block"
SYNC = "max.gpu.sync"
GPU = "max.gpu"
ATOMIC = "std.atomic"
# A lane-group reduction is probed over the whole warp, where it is the warp reduction CrossGL has. Over fewer lanes
# it leaves the lanes past num_lanes unchanged (warp.mojo, lane_group_sum), which no one CrossGL operation does. A
# generic reduction exchanges values with shuffle_xor, which leaves the result in every lane, as warp.mojo's own
# broadcasting reductions do (_lane_group_broadcast_reduce).
_WARP_LITERALS = {"num_lanes": "warp.WARP_SIZE", "shuffle": "warp.shuffle_xor"}
# A block reduction is probed over a block of the invocations a subgroup probe runs as one workgroup, a whole number
# of warps, as block.mojo asserts the block size is.
_BLOCK_LITERALS = {"block_size": str(THREADS)}
# A named barrier waits for the threads of a whole number of warps (PTX ISA, ``bar.sync``), here those of that block.
_SYNC_LITERALS = {"num_threads": str(THREADS)}
# Why a module-qualified function has no CrossGL spelling, by group.
_UNSPELLED_GROUPS = {
    "block": (
        "CrossGL reduces, scans and broadcasts over a subgroup only, not over a workgroup"
    )
}
# Atomic methods are probed with the relaxed ordering, the one CrossGL's atomics have, as GLSL's do (glslang writes
# atomicAdd with the memory semantics None). MAX's default is sequentially consistent except on Apple GPUs
# (atomic.mojo, _DEFAULT_ARITHMETIC_ORDERING).
_RELAXED = {
    parameter: "Ordering.RELAXED"
    for parameter in ("ordering", "success_ordering", "failure_ordering")
}
# The receiver of a method only an Atomic value has, which a probe passes the output buffer's element as.
_ATOMIC_VALUE = re.compile(r"Atomic\[(.*)\]")
# A pointer argument, which a probe passes the output buffer's element as.
_POINTER = re.compile(r"(?:Mut|Imm|Unsafe)?Pointer\[(.*)\]")
# The address space of a pointer that a probe passes a workgroup variable as: shared memory, or an address space the
# call infers, which MAX's memory barriers accept shared memory for (sync.mojo, ``_mbarrier_impl``).
_SHARED = re.compile(r"address_space=(?:AddressSpace)?\.SHARED")


def mojo_type(
    spelling: str,
    element: str,
    generics: set[str],
    arguments: dict[str, Type],
    resolved: dict[str, str] | None = None,
) -> Type:
    """The probe type of a Mojo type spelling; ``element`` is the scalar a generic ``DType`` or type stands for, and
    ``resolved`` the scalar of each ``DType`` parameter the definition pins to one (:func:`_pinned`).
    """
    spelling = re.sub(r"\s*([\[\]])\s*", r"\1", re.sub(r"\s+", " ", spelling)).strip()
    referenced = re.fullmatch(r"type_of\((\w+)\)", spelling)
    if referenced and referenced.group(1) in arguments:
        return arguments[referenced.group(1)]
    if spelling in _NAMED:
        return Type(_NAMED[spelling])
    if spelling in generics:
        return Type(element)
    pair = _pair(spelling, element, generics, arguments)
    if pair:
        return pair
    simd = re.fullmatch(r"(?:SIMD|Scalar)(?:\[(.*)\])?", spelling)
    if simd:
        dtype = split_commas(simd.group(1) or "")[:1]
        # ``Scalar[val.dtype]``: the element type of the argument ``val``.
        argument = re.fullmatch(r"(\w+)\.dtype", dtype[0]) if dtype else None
        if argument and argument.group(1) in arguments:
            return Type(arguments[argument.group(1)].scalar)
        if dtype and dtype[0] in (resolved or {}):
            return Type(resolved[dtype[0]])
        if not dtype or dtype[0] in generics:
            return Type(element)
        bare = dtype[0].removeprefix("DType").lstrip(".")
        if bare in _DTYPES:
            return Type(_DTYPES[bare])
    return Type.other(spelling)


def _pair(
    spelling: str, element: str, generics: set[str], arguments: dict[str, Type]
) -> Type | None:
    """Two values of one scalar type, as ``frexp``, ``modf`` and ``divmod`` return: the opaque type spelled with the
    scalar, as ``Tuple[Float32, Float32]`` or ``StaticTuple[Float32, 2]``, which a probe reads the first value of.
    """
    match = re.fullmatch(r"(Tuple|StaticTuple)\[(.*)\]", spelling)
    if not match:
        return None
    parts = split_commas(match.group(2))
    if match.group(1) == "StaticTuple":
        parts = [parts[0]] * 2 if len(parts) == 2 and parts[1] == "2" else []
    kinds = {mojo_type(part, element, generics, arguments) for part in parts}
    kind = kinds.pop() if len(parts) == 2 and len(kinds) == 1 else None
    if kind is None or kind.opaque or kind.shape:
        return None
    name = _SPELLED[kind.scalar]
    second = "2" if match.group(1) == "StaticTuple" else name
    return Type.other(f"{match.group(1)}[{name}, {second}]")


_ASSERT = re.compile(r"comptime assert\s*(\([^\"]*?\)|[^,\n]*)")


def _constraints(body: str) -> str:
    """A definition's ``where`` clause and ``comptime assert`` conditions; a ``comptime if`` on a type only selects
    an implementation, so it does not count."""
    return " ".join(
        [body.split(":\n", 1)[0], *(m.group(1) for m in _ASSERT.finditer(body))]
    )


def _element(body: str, default: str) -> str:
    """The scalar a definition's constraints require."""
    constraints = _constraints(body)
    if re.search(r"\bis_floating_point\(\)", constraints):
        return "float"
    if re.search(r"\bis_integral\(\)", constraints):
        return "int"
    return default


_DTYPE_NAMED = re.compile(rf"(?:\bDType)?\.({'|'.join(_DTYPES)})\b")


def _pinned(parameter: str, default: str, body: str) -> str | None:
    """The scalar of a ``DType`` parameter that only the result names, when the definition pins it: the first DType
    its default names, as ``DType.uint32 if WARP_SIZE <= 32 else DType.uint64``, or a constraint compares it with,
    as ``ret_type == .uint32``."""
    compared = re.findall(rf"\b{parameter}\s*(?:==|in)\s*([^,]*)", _constraints(body))
    for text in (default, *compared):
        named = _DTYPE_NAMED.search(text)
        if named:
            return _DTYPES[named.group(1)]
    return None


def signature(
    name: str,
    definition: Definition,
    default: str,
    literals: dict[str, str] | None = None,
    named: dict[str, str] | None = None,
) -> tuple[Signature | None, str, str]:
    """``(signature, spelling, note)`` of one definition; ``spelling`` is set when the call names parameters.

    ``literals`` gives the value of a compile-time parameter by name, where the probe passes another value than
    :data:`_LITERALS` gives for the parameter's kind, and ``named`` the parameters the call names even though they
    have a default. A pointer argument takes the output buffer's element, the operand of an atomic operation, an
    Atomic receiver that element viewed as an Atomic, and a ``mut`` argument a local the probe declares.
    """
    parameters, arguments, result, body = definition
    element = _element(body, default)
    declared = compile_parameters(parameters)
    # A generic constrained to a reduction function is the type of the function the probe passes.
    closures = {n for n, kind, _, _ in declared if kind == REDUCE_CLOSURE}
    generics = {
        n for n, kind, _, _ in declared if kind not in _LITERALS and n not in closures
    }
    params: list[Param] = []
    by_name: dict[str, Type] = {}
    passed: list[str] = []
    result_type = None
    for part in split_commas(arguments):
        if part == "*":
            break  # keyword-only arguments follow, and the probe passes none
        if part == "/" or part.startswith("*"):
            continue  # the positional-only marker, or variadic arguments the probe leaves empty
        declaration, default_value = argument_default(part)
        words, _, kind = declaration.partition(":")
        convention, _, local = words.strip().rpartition(" ")
        spelled = re.sub(r"\s+", " ", kind.strip())
        pointer = _POINTER.fullmatch(spelled)
        receiver = local == "self" and _ATOMIC_VALUE.fullmatch(spelled)
        shared = False
        if pointer or receiver:
            # ``Pointer[mut=True, type, _, address_space=.SHARED]`` names its mutability before the pointee, and
            # ``Atomic[Scalar[dtype], scope=_]`` its scope after the value.
            held = (pointer or receiver).group(1)
            pointee = [p for p in split_commas(held) if "=" not in p]
            mapped = mojo_type(pointee[0], element, generics, by_name)
            space = re.search(r"address_space=(\w+)\b", held)
            shared = bool(_SHARED.search(held)) or bool(
                space and space.group(1) in generics
            )
        elif spelled in closures:
            mapped = Type.other(REDUCE_CLOSURE)
        else:
            mapped = mojo_type(kind, element, generics, by_name)
        by_name[local] = mapped
        if convention == "out":
            result_type = mapped
        elif convention and convention not in _CONVENTIONS:
            return None, "", f"no probe passes a {convention} argument"
        elif not default_value.strip():
            if receiver:
                direction = "atomic"
            elif shared:
                direction = "workgroup address"
            elif pointer:
                direction = "inout"
            else:
                direction = "address" if convention == "mut" else "in"
            params.append(Param(mapped, direction))
            passed.append(kind)
    # A DType parameter no passed argument's type mentions and the definition pins to one type, which the call names.
    pinned = {
        parameter: scalar
        for parameter, kind, _, value in declared
        if kind == "DType" and not re.search(rf"\b{parameter}\b", " ".join(passed))
        for scalar in [_pinned(parameter, value, body)]
        if scalar
    }
    if result_type is None:
        result_type = (
            mojo_type(result, element, generics, by_name, pinned) if result else VOID
        )
    # A parameter no passed argument's type mentions cannot be inferred, so the call names it.
    fixed = []
    for parameter, kind, required, _ in declared:
        if parameter in pinned:
            fixed.append(f"{parameter}=DType.{_DTYPE_OF[pinned[parameter]]}")
            continue
        if parameter in (named or {}):
            fixed.append(f"{parameter}={named[parameter]}")
            continue
        if not required or re.search(rf"\b{parameter}\b", " ".join(passed)):
            continue
        if kind == "DType":
            fixed.append(f"{parameter}=DType.{_DTYPE_OF[element]}")
        elif kind in _LITERALS:
            fixed.append(
                f"{parameter}={(literals or {}).get(parameter, _LITERALS[kind])}"
            )
        elif kind.startswith("def") and parameter in (literals or {}):
            fixed.append(f"{parameter}={literals[parameter]}")
        elif kind.startswith("def"):
            return None, "", f"no probe passes a function as the parameter {parameter}"
        else:
            return None, "", f"no probe value for the parameter {parameter}: {kind}"
    spelling = f"{name}[{', '.join(fixed)}]" if fixed else ""
    return Signature(result_type, tuple(params)), spelling, ""


def parse(
    math_init: str,
    math: str,
    numerics: str,
    bit_init: str,
    bit: str,
    glsl_names: set[str] | frozenset[str] = frozenset(),
    *,
    warp: str = "",
    block: str = "",
    sync_init: str = "",
    sync: str = "",
    ids: str = "",
    lanes: str = "",
    atomic_init: str = "",
    atomic: str = "",
) -> list[Feature]:
    """The feature list. The keyword arguments are MAX's ``max/gpu/primitives/warp.mojo`` and ``block.mojo``, its
    ``max/gpu/sync/__init__.mojo`` and ``sync.mojo`` and its ``max/gpu/primitives/id.mojo``, the
    ``std/_gpu/primitives/id.mojo`` that defines ``lane_id``, and ``std/atomic/__init__.mojo`` and ``atomic.mojo``.
    """
    static, instance = {}, {}
    if "Atomic" in exports(atomic_init, ".atomic", functions=False):
        static, instance = methods(atomic, "Atomic")
    lane = definitions(lanes)
    atomic_imports = f"{ATOMIC}.Atomic {ATOMIC}.Ordering"
    # (group, definitions, exported names, default element, how a probe calls and imports a name)
    modules = [
        ("math", definitions(math), exports(math_init, ".math"), "float", _plain),
        (
            "math",
            definitions(numerics),
            exports(math_init, "std.utils.numerics"),
            "float",
            _plain,
        ),
        ("bit", definitions(bit), exports(bit_init, ".bit"), "int", _plain),
        (
            "warp",
            definitions(warp),
            public(warp),
            "float",
            _through(WARP, _WARP_LITERALS),
        ),
        (
            "block",
            definitions(block),
            public(block),
            "float",
            _through(BLOCK, _BLOCK_LITERALS),
        ),
        (
            "sync",
            definitions(sync),
            exports(sync_init, ".sync"),
            "float",
            _imported(SYNC, _SYNC_LITERALS),
        ),
        (
            "id",
            {**lane, **definitions(ids)},
            [n for n in exports(ids, "std._gpu.primitives.id") if n in lane]
            + public(ids),
            "int",
            _imported(GPU),
        ),
        (
            "atomic",
            {**instance, **static},
            [*static, *instance],
            "int",
            lambda n: (f"Atomic.{n}", atomic_imports, None, _RELAXED),
        ),
        (
            "atomic",
            definitions(atomic),
            exports(atomic_init, ".atomic"),
            "int",
            _imported(ATOMIC),
        ),
    ]
    features = []
    for group, defined, names, default, naming in modules:
        for exported in names:
            name, extension, literals, named = naming(exported)
            found, spelling = None, ""
            note = "no definition in the module it is exported from"
            if exported in defined:
                found, spelling, note = signature(
                    name, defined[exported], default, literals, named
                )
            crossgl, counterpart = spellings.spelling("mojo", name, glsl_names), None
            if name in spellings.COUNTERPARTS["mojo"]:
                called, counterpart = prototype(spellings.COUNTERPARTS["mojo"][name])
                # An expression spelling takes the counterpart's arguments in place of Mojo's.
                if not spellings.is_expression(crossgl):
                    crossgl = called
            if name in UNSPELLED:
                crossgl, note = "", UNSPELLED[name]
            elif crossgl == name and "." in name:
                # A module-qualified name is no CrossGL spelling.
                crossgl = ""
                note = note or _UNSPELLED_GROUPS.get(
                    group, "no CrossGL operation is listed for it"
                )
            features.append(
                Feature(
                    "mojo",
                    "function",
                    name,
                    found,
                    group=group,
                    crossgl=crossgl,
                    note=note,
                    spelling=spelling,
                    counterpart=counterpart,
                    extension=extension,
                )
            )
    return sorted(features, key=lambda f: (f.group, f.name))


# A feature's name, what its probe imports, and the literals and named parameters of its call.
Naming = tuple[str, str, dict[str, str] | None, dict[str, str] | None]


def _plain(name: str) -> Naming:
    """A function of the standard library, which every probe imports."""
    return name, "", None, None


def _through(module: str, literals: dict[str, str]) -> Callable[[str], Naming]:
    """A function a probe calls through its ``module``, as ``warp.sum``."""
    short = module.rpartition(".")[2]
    return lambda name: (f"{short}.{name}", module, literals, None)


def _imported(
    package: str, literals: dict[str, str] | None = None
) -> Callable[[str], Naming]:
    """A function a probe imports by name from ``package``."""
    return lambda name: (name, f"{package}.{name}", literals, None)
