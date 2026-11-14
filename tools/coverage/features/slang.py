"""Slang built-in functions from the core module sources slangc is built from.

``hlsl.meta.slang`` and ``core.meta.slang`` define Slang's standard library. :mod:`.slang_meta` reads the global
functions they make public, and this module turns each into a feature with a signature to probe.

Generic parameters are made concrete as in the HLSL list: a floating-point or arithmetic ``T`` is ``float``, an
integer, atomic or unconstrained one ``int``, a logical one ``uint``, and a ``let`` size is 2, or 4 for the length of a
cooperative vector (``CoopVec<T, N>``), which keeps its element type and length. A value parameter no argument fixes,
such as the alignment of ``loadAligned``, is spelled out as 4. A string argument is the empty literal, a pack of
parameters (``expand each T``) is empty, a type pack (``each T``) is one ``int``, so that ``makeTuple`` makes a
``Tuple<int>``, and a pointer is a uniform that holds a device address. A function whose every
``[require(...)]`` names a stage-only capability is probed in that stage. Probes compile for SPIR-V. A function whose
``require`` names no SPIR-V target but allows HLSL compiles for HLSL instead, which DXC then checks, in the stage
Direct3D allows it in; any other function stays untested.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import (
    COOP_COLUMNS,
    COOP_VECTOR_LENGTH,
    DESCRIPTOR,
    DESCRIPTOR_HANDLE,
    OTHER_WIDTHS,
    STRING,
    VOID,
    Feature,
    Param,
    Signature,
    Type,
    coop_element,
    coop_vector,
    raytracing_stage,
    tuple_of,
)
from .common import pick, split
from .hlsl import stage_of as direct3d_stage
from .slang_meta import HEAD, Declaration, declarations

# The scalar each generic constraint is probed with, and the type for a descriptor.
CONSTRAINTS = {
    "__BuiltinFloatingPointType": "float",
    "__BuiltinRealType": "float",
    "__BuiltinArithmeticType": "float",
    "__BuiltinSignedArithmeticType": "float",
    "__BuiltinType": "float",
    "IFloat": "float",
    "IFloatingPointCoopElement": "float",
    "IArithmetic": "float",
    "IArithmeticAtomicable": "int",
    "IAtomicable": "int",
    "__BuiltinIntegerType": "int",
    "__BuiltinInt32Type": "int",
    "IInteger": "int",
    "__BuiltinLogicalType": "uint",
    "ILogical": "uint",
    "IOpaqueDescriptor": DESCRIPTOR,
}
# ``[require(...)]`` capabilities only one kind of stage has, matched as prefixes of the capability atoms.
STAGE_CAPABILITIES = (
    ("fragment", "fragment"),
    ("getattributeatvertex", "fragment"),
    ("helper_lane", "fragment"),
    ("raytracing", "raytracing"),
    ("ser_", "raytracing"),
    ("meshshading", "mesh"),
)
_SCALARS = {
    "bool": "bool",
    "int": "int",
    "int32_t": "int",
    "uint": "uint",
    "uint32_t": "uint",
    "float": "float",
    "float32_t": "float",
    "double": "double",
    "float64_t": "double",
    "half": "half",
    "float16_t": "half",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "int8_t4_packed": "int8x4_packed",
    "uint8_t4_packed": "uint8x4_packed",
}
# Slang's own spelling of each scalar, for a generic argument.
_SCALAR_SPELLINGS = {scalar: name for name, scalar in reversed(_SCALARS.items())}
_SHORTHAND = re.compile(
    r"(bool|int|uint|float|double|half|int32_t|uint32_t|float32_t|float64_t|float16_t|int16_t|uint16_t|int64_t"
    r"|uint64_t)([1-4])(?:x([1-4]))?"
)
# The targets a ``require`` capability atom such as ``cuda_glsl_hlsl_spirv`` is made of.
TARGETS = frozenset("cpp cuda glsl hlsl metal spirv wgsl llvm".split())
# Capability atoms that imply one target without naming it (source/slang/slang-capabilities.capdef).
_ATOM_TARGETS = {"optix_coopvec": "cuda", "optix_multilevel_traversal": "cuda"}
_GENERIC_TYPE = re.compile(r"(vector|matrix|CoopVec|Tuple)\s*<(.*)>")
_DESCRIPTOR_HANDLE = re.compile(r"DescriptorHandle\s*<\s*(\w+)\s*>")
_QUALIFIERS = {
    "in": "in",
    "out": "out",
    "inout": "inout",
    "__ref": "inout",
    "__constref": "in",
    "const": "in",
    "__constexpr": "const",
    "constexpr": "const",
    "no_diff": "in",
    "uniform": "in",
}
# String parameters take a literal, which slangc folds at compile time (core.meta.slang, getStringHash).
_STRINGS = frozenset({"String", "NativeString"})
_VALUE_GENERIC = re.compile(r"(?:int|uint)\s+(\w+)")
# A value generic parameter no argument fixes, such as loadAligned's alignment: a power of two, and the size in bytes
# of the 32-bit scalars the probes use.
_FREE_VALUE = 4


def _generics(text: str) -> dict[str, str]:
    """Generic parameter name -> the scalar it is probed with, or ``"size"`` for a ``let`` size."""
    bound = {}
    for part in split(text):
        part = part.split("=")[0].strip()
        if not part:
            continue
        if part.startswith("let "):
            bound[part[4:].split(":")[0].strip()] = "size"
            continue
        value = _VALUE_GENERIC.fullmatch(part)
        if value:
            # The older spelling of a value parameter, ``int alignment`` for ``let alignment : int``.
            bound[value.group(1)] = "size"
            continue
        name, _, constraint = part.partition(":")
        constraint = constraint.strip()
        # A type pack, ``each T``, holds one element.
        bound[name.strip().removeprefix("each ").strip()] = (
            CONSTRAINTS.get(constraint, f"?{constraint}") if constraint else "int"
        )
    return bound


def slang_type(spelling: str, generics: dict[str, str]) -> Type:
    spelling = " ".join(spelling.split())
    if spelling == "void":
        return VOID
    if spelling in generics:
        scalar = generics[spelling]
        return (
            Type(scalar)
            if scalar in _SCALARS.values()
            else Type.other(scalar.lstrip("?") or spelling)
        )
    if spelling in _SCALARS:
        return Type(_SCALARS[spelling])
    handle = _DESCRIPTOR_HANDLE.fullmatch(spelling)
    if handle and generics.get(handle.group(1)) == DESCRIPTOR:
        return Type.other(DESCRIPTOR_HANDLE)
    shorthand = _SHORTHAND.fullmatch(spelling)
    if shorthand:
        scalar, rows, columns = shorthand.groups()
        if columns:
            return Type(_SCALARS[scalar], (int(columns), int(rows)))
        return (
            Type(_SCALARS[scalar])
            if rows == "1"
            else Type(_SCALARS[scalar], (int(rows),))
        )
    generic = _GENERIC_TYPE.fullmatch(spelling)
    if generic:
        parts = split(generic.group(2))
        element = slang_type(parts[0], generics)
        if element.opaque or element.shape:
            return Type.other(spelling)

        def size(text: str) -> int:
            text = text.strip()
            return int(text) if text.isdigit() else 2

        if generic.group(1) == "Tuple":
            same = all(slang_type(p, generics) == element for p in parts)
            return (
                tuple_of(element.scalar, len(parts)) if same else Type.other(spelling)
            )
        sizes = [size(p) for p in parts[1:]]
        if generic.group(1) == "CoopVec" and len(parts) == 2:
            length = parts[1].strip()
            length = int(length) if length.isdigit() else COOP_VECTOR_LENGTH
            return (
                coop_vector(element.scalar, length)
                if 2 <= length <= 4
                else Type.other(spelling)
            )
        if generic.group(1) == "vector" and len(sizes) == 1 and 2 <= sizes[0] <= 4:
            return Type(element.scalar, (sizes[0],))
        if generic.group(1) == "matrix" and len(sizes) >= 2:
            rows, columns = sizes[:2]
            if 2 <= rows <= 4 and 2 <= columns <= 4:
                return Type(element.scalar, (columns, rows))
    return Type.other(spelling)


def _parameter(text: str, generics: dict[str, str]) -> Param:
    text = text.split("=")[0]
    # The length of an array parameter, ``T[M] data``.
    array = re.search(r"\[\s*(\w*)\s*\]\s*\w+\s*$", text)
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    words = text.split()
    direction = "in"
    while words and words[0] in _QUALIFIERS:
        direction = _QUALIFIERS[words.pop(0)] if direction == "in" else direction
    if words[:1] == ["groupshared"] and direction == "in" and array:
        # A workgroup array the function reads, such as the one coopVecLoadGroupshared loads from.
        element = slang_type(" ".join(w for w in words[1:-1] if w != "const"), generics)
        if not element.opaque and not element.shape:
            size = array.group(1)
            count = int(size) if size.isdigit() else COOP_VECTOR_LENGTH
            return Param(Type(element.scalar, count=count), "workgroup")
    spelling = " ".join(words[:-1]) if len(words) > 1 else " ".join(words)
    if direction == "const" and words[-2:] == ["int", "k"]:
        # The column count of a packed cooperative-vector multiply.
        return Param(COOP_COLUMNS, direction)
    if direction == "const" and words[-2:] == ["bool", "condition"]:
        # What static_assert checks while compiling.
        return Param(Type("bool"), "condition")
    if spelling in _STRINGS:
        return Param(STRING, "const")
    if spelling.endswith("*") and spelling.count("*") == 1:
        # A pointer to device memory, such as loadAligned reads through.
        pointee = slang_type(spelling[:-1], generics)
        if not pointee.opaque:
            return Param(pointee, "pointer")
    return Param(slang_type(spelling, generics), direction)


def signature(declaration: Declaration) -> Signature | None:
    match = HEAD.fullmatch(declaration.head)
    if match is None:
        return None
    generics = _generics(declaration.generics)
    result = match.group("arrow") or match.group("result")
    params = [p for p in split(match.group("params")) if p.strip()]
    if any("..." in p for p in params):
        return None
    # A parameter pack (``expand each T``) is probed with no elements.
    params = [p for p in params if p.split()[0] != "expand"]
    return Signature(
        slang_type(result, generics), tuple(_parameter(p, generics) for p in params)
    )


def stage(declaration: Declaration) -> str:
    """The stage a probe needs. ``require`` attributes are alternatives, so a stage is needed only when each of them
    names a stage-only capability."""
    stages = []
    for clause in re.findall(r"require\s*\(([^)]*)\)", declaration.requires):
        found = [
            name
            for atom in re.findall(r"\w+", clause)
            for prefix, name in STAGE_CAPABILITIES
            if atom.startswith(prefix)
        ]
        if not found:
            return "compute"
        stages.append(found[0])
    return stages[0] if stages else "compute"


def _stage(name: str, declaration: Declaration) -> str:
    """:func:`stage`, with a ray-tracing function probed in the ray-tracing stage that has it, and DispatchMesh, which
    launches mesh workgroups, in the task stage. A declaration with no ``require`` attribute, such as the
    ``ObjectToWorld`` alias of ``ObjectToWorld3x4``, takes the stage of the Direct3D intrinsic of its name.
    """
    found = stage(declaration)
    if not re.search(r"require\s*\(", declaration.requires):
        found = direct3d_stage(name) or found
    if found == "mesh" and name == "DispatchMesh":
        return "task"
    return raytracing_stage(name) if found == "raytracing" else found


def _preference(signature: Signature) -> tuple[bool, bool, int]:
    """Sort key among overloads: value-returning before ``void``, 32-bit before 64-bit or 16-bit, and vectors
    before scalars before matrices, so a generic function is probed with two-component vectors as in the HLSL list.
    """
    kinds = [signature.result, *(p.type for p in signature.params)]
    shapes = {len(k.shape) for k in kinds if not k.opaque}
    rank = 2 if 2 in shapes else 0 if 1 in shapes else 1
    return (
        signature.result == VOID
        and any(p.direction == "out" for p in signature.params),
        any(k.scalar in OTHER_WIDTHS for k in kinds),
        rank,
    )


def allows(declaration: Declaration, target: str) -> bool:
    """Whether some ``require`` alternative allows ``target``; one that names no target allows every target."""
    clauses = re.findall(r"require\s*\(([^)]*)\)", declaration.requires)
    for clause in clauses or [""]:
        atoms = [_ATOM_TARGETS.get(atom, atom) for atom in re.findall(r"\w+", clause)]
        targets = {
            part
            for atom in atoms
            if set(atom.split("_")) <= TARGETS
            for part in atom.split("_")
        }
        if not targets or target in targets:
            return True
    return False


def _spell(kind: Type) -> str:
    if len(kind.shape) == 2:
        columns, rows = kind.shape
        return f"{kind.scalar}{rows}x{columns}"
    return f"{kind.scalar}{kind.shape[0]}" if kind.shape else kind.scalar


def _coop_call(declaration: Declaration) -> str:
    """How the native probe calls a function that returns a cooperative vector: slangc infers no generic argument
    from the result, so the probe spells each one before the first that has a default, as
    ``coopVecLoad<4, float>``. An empty string when one of them has no probe value."""
    arguments = []
    for part in split(declaration.generics):
        if "=" in part:
            break
        bound = _generics(part)
        value = next(iter(bound.values()), "?")
        if value == "size":
            arguments.append(str(COOP_VECTOR_LENGTH))
        elif value in _SCALARS.values():
            arguments.append(_SCALAR_SPELLINGS[value])
        else:
            return ""
    return f"{declaration.name}<{', '.join(arguments)}>"


def _value_call(declaration: Declaration, generics: dict[str, str], params: str) -> str:
    """How the native probe calls a function with a value generic parameter no argument fixes, such as
    ``loadAligned<4>``: slangc infers it from nothing, so the probe spells each generic argument up to the last such
    one. An empty string when the arguments fix every value parameter, or one of the leading ones has no probe value.
    """
    names = list(generics)
    free = [
        index
        for index, name in enumerate(names)
        if generics[name] == "size" and not re.search(rf"\b{re.escape(name)}\b", params)
    ]
    if not free:
        return ""
    arguments = []
    for name in names[: free[-1] + 1]:
        value = generics[name]
        if value == "size":
            arguments.append(str(_FREE_VALUE))
        elif value in _SCALARS.values():
            arguments.append(_SCALAR_SPELLINGS[value])
        else:
            return ""
    return f"{declaration.name}<{', '.join(arguments)}>"


def explicit(declaration: Declaration, chosen: Signature) -> str:
    """How the native probe calls a function whose result type is a generic parameter no argument fixes, as
    ``bit_cast<int2>``, or that has a value generic parameter no argument fixes, as ``loadAligned<4>``; an empty
    string when the arguments fix every generic parameter.
    """
    match = HEAD.fullmatch(declaration.head)
    generics = _generics(declaration.generics)
    if match and coop_element(chosen.result.opaque):
        return _coop_call(declaration)
    if not match:
        return ""
    params = match.group("params")
    value_call = _value_call(declaration, generics, params)
    free = [
        name
        for name, bound in generics.items()
        if bound != "size" and not re.search(rf"\b{re.escape(name)}\b", params)
    ]
    if free and list(generics) == free and set(generics.values()) == {DESCRIPTOR}:
        # slangc infers no descriptor type from a result or from nothing, as for
        # getDescriptorFromExplicitHeapIndex<T>(uint2) and usesNativeSpirvDescriptorHandle<T>().
        return f"{declaration.name}<{DESCRIPTOR}>"
    if value_call or chosen.result.opaque:
        return value_call
    result = match.group("arrow") or match.group("result")
    if free and re.search(rf"\b{re.escape(free[0])}\b", result):
        return f"{declaration.name}<{_spell(chosen.result)}>"
    return ""


def unprobed_generic(declaration: Declaration) -> str:
    """Why no probe can call a function with a type generic parameter that neither its parameters nor its result
    mention and whose constraint has no probe type in :data:`CONSTRAINTS`: slangc cannot infer the argument, and the
    probe has no type to spell for it. An empty string when there is no such parameter.
    """
    match = HEAD.fullmatch(declaration.head)
    if not match:
        return ""
    mentioned = (
        f"{match.group('params')} {match.group('arrow') or match.group('result')}"
    )
    for name, bound in _generics(declaration.generics).items():
        if bound.startswith("?") and not re.search(
            rf"\b{re.escape(name)}\b", mentioned
        ):
            return f"no probe type for the generic parameter {name} : {bound[1:]}"
    return ""


def parse(
    core: str, hlsl: str, glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    overloads: dict[str, list[tuple[str, Declaration]]] = {}
    for group, text in (("core", core), ("hlsl", hlsl)):
        for declaration in declarations(text):
            overloads.setdefault(declaration.name, []).append((group, declaration))
    features = []
    for name, declared in sorted(overloads.items()):
        candidates = [(s, d) for s, d in ((signature(d), d) for _, d in declared) if s]
        note = "" if candidates else "variadic"
        target, stage = "", _stage(name, declared[0][1])
        if not any(allows(d, "spirv") for _, d in declared):
            candidates = [(s, d) for s, d in candidates if allows(d, "hlsl")]
            target, stage = "hlsl", direct3d_stage(name) or stage
            if not any(allows(d, "hlsl") for _, d in declared):
                note = "needs a target other than SPIR-V and HLSL"
        chosen = None
        if candidates:
            chosen = pick(sorted((s for s, _ in candidates), key=_preference))
        source = next((d for s, d in candidates if s == chosen), declared[0][1])
        if chosen and (unprobed := unprobed_generic(source)):
            chosen, note = None, unprobed
        spelled = explicit(source, chosen) if chosen else ""
        crossgl = spellings.spelling("slang", name, glsl_names)
        features.append(
            Feature(
                "slang",
                "function",
                name,
                chosen,
                group=declared[0][0],
                # CrossGL calls a function it has no name of its own for by Slang's name, with the same generic
                # arguments, which neither language infers from the result.
                crossgl=spelled if spelled and crossgl == name else crossgl,
                stage=stage,
                note=note,
                spelling=spelled,
                target=target if chosen else "",
            )
        )
    return features
