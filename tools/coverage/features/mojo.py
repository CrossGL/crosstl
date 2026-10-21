"""Mojo standard library functions: the public API of ``std.math`` and ``std.bit``.

A package's public API is what its ``__init__.mojo`` imports from its modules, so the list is the lower-case names
those files export (the capitalised ones are traits). ``std.math`` also exports the floating-point predicates of
``std.utils.numerics``. Each name is probed with its first module-level definition. Mojo functions are generic over a
``DType`` and a SIMD width; a probe calls them on scalars, ``Float32`` unless the definition requires an integral
type, as all of ``std.bit`` does. A compile-time parameter no argument fixes is written into the call, as in
``nan[dtype=DType.float32]()``.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import VOID, Feature, Param, Signature, Type

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
_NAMED = {
    "Float32": "float",
    "Float64": "double",
    "Int32": "int",
    "UInt32": "uint",
    "Bool": "bool",
}
# Argument conventions a probe can pass a local to; ``mut`` makes the argument read-modify-write.
_CONVENTIONS = frozenset({"mut", "read", "var", "ref", "owned"})
# Compile-time parameter kinds a probe can fill in with a constant, and the constant it uses.
_LITERALS = {"Int": "1", "SIMDLength": "1", "Bool": "True"}

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
    spelling: str, element: str, generics: set[str], arguments: dict[str, Type]
) -> Type:
    """The probe type of a Mojo type spelling; ``element`` is the scalar a generic ``DType`` or type stands for."""
    spelling = re.sub(r"\s*([\[\]])\s*", r"\1", re.sub(r"\s+", " ", spelling)).strip()
    referenced = re.fullmatch(r"type_of\((\w+)\)", spelling)
    if referenced and referenced.group(1) in arguments:
        return arguments[referenced.group(1)]
    if spelling in _NAMED:
        return Type(_NAMED[spelling])
    if spelling in generics:
        return Type(element)
    simd = re.fullmatch(r"(?:SIMD|Scalar)(?:\[(.*)\])?", spelling)
    if simd:
        dtype = _split(simd.group(1) or "")[:1]
        if not dtype or dtype[0] in generics:
            return Type(element)
        bare = dtype[0].removeprefix("DType").lstrip(".")
        if bare in _DTYPES:
            return Type(_DTYPES[bare])
    return Type.other(spelling)


_ASSERT = re.compile(r"comptime assert\s*(\([^\"]*?\)|[^,\n]*)")


def _element(body: str, default: str) -> str:
    """The scalar a definition's ``where`` clause or ``comptime assert`` requires; a ``comptime if`` on the type
    only selects an implementation, so it does not count."""
    constraints = " ".join(
        [body.split(":\n", 1)[0], *(m.group(1) for m in _ASSERT.finditer(body))]
    )
    if re.search(r"\bis_floating_point\(\)", constraints):
        return "float"
    if re.search(r"\bis_integral\(\)", constraints):
        return "int"
    return default


def _parameters(text: str) -> list[tuple[str, str, bool]]:
    """``(name, kind, required)`` per compile-time parameter.

    Parameters before ``//`` are inferred from the arguments, and a parameter with a default needs no value, so
    neither is required.
    """
    parts = _split(text)
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
            (name.strip(), kind.strip(), not inferred and not default.strip())
        )
    return result


def signature(
    name: str, definition: Definition, default: str
) -> tuple[Signature | None, str, str]:
    """``(signature, spelling, note)`` of one definition; ``spelling`` is set when the call names parameters."""
    parameters, arguments, result, body = definition
    element = _element(body, default)
    declared = _parameters(parameters)
    generics = {n for n, kind, _ in declared if kind not in _LITERALS}
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
    if result_type is None:
        result_type = mojo_type(result, element, generics, by_name) if result else VOID
    # A parameter no passed argument's type mentions cannot be inferred, so the call names it.
    fixed = []
    for parameter, kind, required in declared:
        if not required or re.search(rf"\b{parameter}\b", " ".join(passed)):
            continue
        if kind == "DType":
            fixed.append(f"{parameter}=DType.{_DTYPE_OF[element]}")
        elif kind in _LITERALS:
            fixed.append(f"{parameter}={_LITERALS[kind]}")
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
) -> list[Feature]:
    modules = [
        ("math", math, exports(math_init, ".math"), "float"),
        ("math", numerics, exports(math_init, "std.utils.numerics"), "float"),
        ("bit", bit, exports(bit_init, ".bit"), "int"),
    ]
    features = []
    for group, text, names, default in modules:
        defined = definitions(text)
        for name in names:
            found, spelling, note = (
                None,
                "",
                "no definition in the module it is exported from",
            )
            if name in defined:
                found, spelling, note = signature(name, defined[name], default)
            features.append(
                Feature(
                    "mojo",
                    "function",
                    name,
                    found,
                    group=group,
                    crossgl=spellings.spelling("mojo", name, glsl_names),
                    note=note,
                    spelling=spelling,
                )
            )
    return sorted(features, key=lambda f: (f.group, f.name))
