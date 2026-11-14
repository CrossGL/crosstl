"""Mojo functions: the public API of ``std.math`` and ``std.bit``, and of MAX's GPU warp and synchronization modules.

A package's public API is what its ``__init__.mojo`` imports from its modules, so the list is the lower-case names
those files export (the capitalised ones are traits and types). ``std.math`` also exports the floating-point predicates
of ``std.utils.numerics``. ``max.gpu.primitives.warp`` is a module no package re-exports, so its public API is every
module-level definition whose name does not start with an underscore; ``max.gpu.sync`` exports its barriers from its
``sync`` module. Each name is probed with its first module-level definition. Mojo functions are generic over a
``DType`` and a SIMD width; a probe calls them on scalars, ``Float32`` unless the definition requires an integral
type, as all of ``std.bit`` does. A compile-time parameter no argument fixes is written into the call, as in
``nan[dtype=DType.float32]()``.
"""

from __future__ import annotations

import re

from .. import spellings
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
}
# MAX's GPU modules, as a probe imports them (:meth:`..kernels.Mojo.enable`). A warp function is called through its
# module, as ``warp.sum``, because warp's sum, max and min have the names of built-in and std.math functions; a
# synchronization function is imported by name.
WARP = "max.gpu.primitives.warp"
SYNC = "max.gpu.sync"
# A lane-group reduction is probed over the whole warp, where it is the warp reduction CrossGL has. Over fewer lanes
# it leaves the lanes past num_lanes unchanged (warp.mojo, lane_group_sum), which no one CrossGL operation does.
_WARP_LITERALS = {"num_lanes": "warp.WARP_SIZE"}

Definition = tuple[str, str, str, str]


def exports(init: str, module: str) -> list[str]:
    """The function names a package's ``__init__.mojo`` imports from ``module``, in file order."""
    names = []
    for source, imported in _EXPORTS.findall(init):
        if source != module:
            continue
        for name in re.split(r"[\s,()]+", imported):
            if name and name[0].islower() and name not in names:
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


def signature(
    name: str,
    definition: Definition,
    default: str,
    literals: dict[str, str] | None = None,
) -> tuple[Signature | None, str, str]:
    """``(signature, spelling, note)`` of one definition; ``spelling`` is set when the call names parameters.

    ``literals`` gives the value of a compile-time parameter by name, where the probe passes another value than
    :data:`_LITERALS` gives for the parameter's kind.
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
        declaration, _, default_value = part.partition("=")
        words, _, kind = declaration.partition(":")
        convention, _, local = words.strip().rpartition(" ")
        mapped = mojo_type(kind, element, generics, by_name)
        by_name[local] = mapped
        if convention == "out":
            result_type = mapped
        elif convention and convention not in _CONVENTIONS:
            return None, "", f"no probe passes a {convention} argument"
        elif not default_value.strip():
            params.append(Param(mapped, "inout" if convention == "mut" else "in"))
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
    sync_init: str = "",
    sync: str = "",
) -> list[Feature]:
    """The feature list; ``warp``, ``sync_init`` and ``sync`` are MAX's ``max/gpu/primitives/warp.mojo``,
    ``max/gpu/sync/__init__.mojo`` and ``max/gpu/sync/sync.mojo``."""
    modules = [
        ("math", math, exports(math_init, ".math"), "float"),
        ("math", numerics, exports(math_init, "std.utils.numerics"), "float"),
        ("bit", bit, exports(bit_init, ".bit"), "int"),
        ("warp", warp, public(warp), "float"),
        ("sync", sync, exports(sync_init, ".sync"), "float"),
    ]
    features = []
    for group, text, names, default in modules:
        defined = definitions(text)
        for exported in names:
            name, extension, literals = exported, "", None
            if group == "warp":
                name, extension, literals = f"warp.{exported}", WARP, _WARP_LITERALS
            elif group == "sync":
                extension = f"{SYNC}.{exported}"
            found, spelling, note = (
                None,
                "",
                "no definition in the module it is exported from",
            )
            if exported in defined:
                found, spelling, note = signature(
                    name, defined[exported], default, literals
                )
            crossgl, counterpart = spellings.spelling("mojo", name, glsl_names), None
            if name in spellings.COUNTERPARTS["mojo"]:
                crossgl, counterpart = prototype(spellings.COUNTERPARTS["mojo"][name])
            if name in UNSPELLED:
                crossgl, note = "", UNSPELLED[name]
            elif crossgl == name and "." in name:
                # A module-qualified name is no CrossGL spelling.
                crossgl, note = "", note or "no CrossGL operation is listed for it"
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


def public(text: str) -> list[str]:
    """The names of a module's module-level definitions that do not start with an underscore, in file order."""
    names = []
    for name in _DEFINITION.findall(text):
        if not name.startswith("_") and name not in names:
            names.append(name)
    return names
