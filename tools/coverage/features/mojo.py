"""Mojo functions: the public API of ``std.math``, ``std.bit`` and ``std.atomic``, and of MAX's GPU warp, block,
index and synchronization modules.

A package's public API is what its ``__init__.mojo`` imports from its modules, so the list is the lower-case names
those files export (the capitalised ones are traits and types). ``std.math`` also exports the floating-point predicates
of ``std.utils.numerics``. ``max.gpu.primitives.warp`` and ``max.gpu.primitives.block`` are modules no package
re-exports, so their public API is every module-level definition whose name does not start with an underscore;
``max.gpu.sync`` exports its barriers from its ``sync`` module. ``max.gpu.primitives.id`` defines ``warp_id`` and
``sm_id`` and re-exports ``lane_id`` and the index values (``thread_idx``, ``global_idx`` and the rest) from
``std._gpu``; those values are not functions, and the compute construct probes read them. ``std.atomic`` exports
``fence`` and the type ``Atomic``, whose static methods operate on a pointer. Each name is probed with its first
definition. Mojo functions are generic over a ``DType`` and a SIMD width; a probe calls them on scalars, ``Float32``
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
from ..model import VOID, Feature, Param, Signature, Type
from .common import prototype

_EXPORTS = re.compile(r"^from\s+(\S+)\s+import\s+(\([^)]*\)|[^\n]+)", re.M)
_DEFINITION = re.compile(r"^def\s+(\w+)\s*", re.M)
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
_LITERALS = {"Int": "1", "SIMDLength": "1", "Bool": "True"}
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
# it leaves the lanes past num_lanes unchanged (warp.mojo, lane_group_sum), which no one CrossGL operation does.
_WARP_LITERALS = {"num_lanes": "warp.WARP_SIZE"}
# A block reduction is probed over a block of the invocations a subgroup probe runs as one workgroup, a whole number
# of warps, as block.mojo asserts the block size is.
_BLOCK_LITERALS = {"block_size": str(THREADS)}
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
# Only an Atomic value has these methods; a probe calls Atomic's static methods on a buffer element instead.
_INSTANCE = "a method of an Atomic value only, and the probes call Atomic's static methods on a buffer element"
# A pointer argument, which a probe passes the output buffer's element as.
_POINTER = re.compile(r"(?:Mut|Imm|Unsafe)?Pointer\[(.*)\]")

Definition = tuple[str, str, str, str]


def exports(init: str, module: str, functions: bool = True) -> list[str]:
    """The function names a package's ``__init__.mojo`` imports from ``module``, in file order; every name it imports
    unless ``functions``."""
    names = []
    for source, imported in _EXPORTS.findall(init):
        if source != module:
            continue
        for name in re.split(r"[\s,()]+", imported):
            if name and (name[0].islower() or not functions) and name not in names:
                names.append(name)
    return names


def _split(text: str) -> list[str]:
    """``text`` split at its top-level commas."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _group(text: str, start: int, closing: str) -> tuple[str, int]:
    """The text inside the bracket group that opens at ``text[start]``, and the index just past it."""
    opening, depth = text[start], 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError(f"unbalanced {opening}{closing} at offset {start}")


def definitions(text: str) -> dict[str, Definition]:
    """``name -> (parameters, arguments, result, body)`` for the first module-level definition of each name.

    ``body`` starts with the header's ``where`` clause, so a constraint written there or asserted in the body is
    found the same way.
    """
    found: dict[str, Definition] = {}
    for match in _DEFINITION.finditer(text):
        name, position = match.group(1), match.end()
        if name in found:
            continue
        parameters = ""
        if text[position] == "[":
            parameters, position = _group(text, position, "]")
        arguments, position = _group(text, position, ")")
        header_end = text.index(":\n", position)
        tail = text[position:header_end]
        result = re.sub(r"\swhere\s.*", "", tail, flags=re.S).strip()
        following = re.search(r"^(?=[^\s#])", text[header_end + 2 :], re.M)
        body_end = header_end + 2 + (following.start() if following else len(text))
        found[name] = (
            parameters,
            arguments,
            result.removeprefix("->").strip(),
            tail + text[header_end:body_end],
        )
    return found


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
        dtype = _split(simd.group(1) or "")[:1]
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
    parts = _split(match.group(2))
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


def _parameters(text: str) -> list[tuple[str, str, bool, str]]:
    """``(name, kind, required, default)`` per compile-time parameter.

    Parameters before ``//`` are inferred from the arguments, and a parameter with a default needs no value, so
    neither is required.
    """
    parts = _split(re.sub(r"#[^\n]*", "", text))
    inferred = "//" in parts
    result = []
    for part in parts:
        if part == "//":
            inferred = False
            continue
        if part == "*":
            continue
        name, _, rest = part.partition(":")
        kind, _, default = rest.partition("=")
        result.append(
            (
                name.strip(),
                kind.strip(),
                not inferred and not default.strip(),
                default.strip(),
            )
        )
    return result


def _default(part: str) -> tuple[str, str]:
    """An argument's declaration and its default value, split at the first ``=`` outside brackets, as in
    ``ptr: MutPointer[Scalar[dtype], _, address_space=_]``, which has none."""
    depth = 0
    for index, char in enumerate(part):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            return part[:index], part[index + 1 :]
    return part, ""


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
    have a default. A pointer argument takes the output buffer's element, the operand of an atomic operation, and a
    ``mut`` argument a local the probe declares.
    """
    parameters, arguments, result, body = definition
    element = _element(body, default)
    declared = _parameters(parameters)
    generics = {n for n, kind, _, _ in declared if kind not in _LITERALS}
    params: list[Param] = []
    by_name: dict[str, Type] = {}
    passed: list[str] = []
    result_type = None
    for part in _split(arguments):
        if part == "*":
            break  # keyword-only arguments follow, and the probe passes none
        if part == "/" or part.startswith("*"):
            continue  # the positional-only marker, or variadic arguments the probe leaves empty
        declaration, default_value = _default(part)
        words, _, kind = declaration.partition(":")
        convention, _, local = words.strip().rpartition(" ")
        pointer = _POINTER.fullmatch(re.sub(r"\s+", " ", kind.strip()))
        if pointer:
            mapped = mojo_type(_split(pointer.group(1))[0], element, generics, by_name)
        else:
            mapped = mojo_type(kind, element, generics, by_name)
        by_name[local] = mapped
        if convention == "out":
            result_type = mapped
        elif convention and convention not in _CONVENTIONS:
            return None, "", f"no probe passes a {convention} argument"
        elif not default_value.strip():
            direction = (
                "inout" if pointer else "address" if convention == "mut" else "in"
            )
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
    static, instance = {}, []
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
            _imported(SYNC),
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
            static,
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
    absent = {f"Atomic.{name}": _INSTANCE for name in instance}
    features = []
    for group, defined, names, default, naming in modules:
        for exported in names:
            name, extension, literals, named = naming(exported)
            found, spelling = None, ""
            note = absent.get(name, "no definition in the module it is exported from")
            if exported in defined:
                found, spelling, note = signature(
                    name, defined[exported], default, literals, named
                )
            crossgl, counterpart = spellings.spelling("mojo", name, glsl_names), None
            if name in spellings.COUNTERPARTS["mojo"]:
                crossgl, counterpart = prototype(spellings.COUNTERPARTS["mojo"][name])
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


def _imported(package: str) -> Callable[[str], Naming]:
    """A function a probe imports by name from ``package``."""
    return lambda name: (name, f"{package}.{name}", None, None)


_METHOD = re.compile(r"^((?:@[^\n]*\n)*)def\s+(\w+)", re.M)


def methods(text: str, struct: str) -> tuple[dict[str, Definition], list[str]]:
    """The definitions of the static methods of the struct ``struct`` by name, and the names of the public methods
    only its values have, in file order."""
    found = re.search(rf"^struct\s+{struct}\b", text, re.M)
    if not found:
        return {}, []
    position = found.end()
    if text[position] == "[":
        _, position = _group(text, position, "]")
    header_end = text.index(":\n", position) + 2
    following = re.search(r"^(?=[^\s#])", text[header_end:], re.M)
    end = header_end + (following.start() if following else len(text))
    body = re.sub(r"^    ", "", text[header_end:end], flags=re.M)
    static: dict[str, Definition] = {}
    instance: list[str] = []
    for match in _METHOD.finditer(body):
        decorators, name = match.groups()
        if name.startswith("_"):
            continue
        if "@staticmethod" in decorators:
            static.setdefault(name, definitions(body[match.start(2) - 4 :])[name])
        elif name not in instance:
            instance.append(name)
    return static, [name for name in instance if name not in static]


def public(text: str) -> list[str]:
    """The names of a module's module-level definitions that do not start with an underscore, in file order."""
    names = []
    for name in _DEFINITION.findall(text):
        if not name.startswith("_") and name not in names:
            names.append(name)
    return names
