"""Rust GPU functions: ``spirv_std::arch``, the ``num_traits::Float`` methods and glam's ``Vec3`` methods.

rust-gpu shaders call the GPU intrinsics of ``spirv_std::arch``, do scalar math through the ``num_traits::Float``
trait ``spirv_std`` re-exports, and vector math through glam's methods, ``Vec3`` among them. The lists are read from
the sources of the versions rust-gpu's lock file resolves. ``spirv_std::arch`` defines some functions with
``macro_rules!`` macros, which are expanded here. A function generic over a numeric trait is probed with the scalar
type :data:`_BOUNDS` names for the trait; constant generic parameters take the values of :data:`_CONSTANTS`.
"""

from __future__ import annotations

import re

from .. import spellings
from ..model import VOID, Feature, Param, Signature, Type
from .common import prototype

# The scalar type a probe uses for a parameter bounded by each spirv_std trait; atomics and subgroup integer
# operations on unsigned values are the ones every target has.
_BOUNDS = {
    "Float": "float",
    "SignedInteger": "int",
    "UnsignedInteger": "uint",
    "Integer": "uint",
    "Number": "uint",
    "ScalarComposite": "float",
    "ScalarOrVector": "float",
    "Scalar": "float",
}
_SCALARS = {
    "f32": "float",
    "f64": "double",
    "i32": "int",
    "u32": "uint",
    "i16": "int16",
    "u16": "uint16",
    "i64": "int64",
    "u64": "uint64",
    "usize": "usize",
    "bool": "bool",
}
_GLAM = {"": "float", "D": "double", "I": "int", "U": "uint", "B": "bool"}
# The cargo features spirv-std 0.11 enables on its glam 0.34 dependency (crates/spirv-std/Cargo.toml). glam puts its
# other vector widths behind features, so a method such as ``Vec3::as_i16vec3`` is absent from the glam a rust-gpu
# shader links, and is not a rust-gpu feature.
_GLAM_FEATURES = frozenset({"f64", "i32", "libm", "u32"})
_GATED = re.compile(
    r'#\[cfg\(feature = "([\w-]+)"\)\]\s*(?:#\[[^\]]*\]\s*)*pub (?:const )?(?:unsafe )?fn (\w+)'
)
# Values for constant generic parameters: memory scopes and semantics as ``spirv_std::memory`` spells them. Subgroup
# scope is one every operation that takes a scope accepts, the shader clock included.
_SCOPE = "{ Scope::Subgroup as u32 }"
_CONSTANTS = {
    "SCOPE": _SCOPE,
    "EXECUTION": _SCOPE,
    "MEMORY": _SCOPE,
    "SEMANTICS": "{ Semantics::NONE.bits() }",
    "EQUAL": "{ Semantics::NONE.bits() }",
    "UNEQUAL": "{ Semantics::NONE.bits() }",
    "ID": "0",
    "CLUSTER_SIZE": "4",
    "DIRECTION": "0",
    "STREAM": "0",
}
# A memory barrier has to order memory, so its semantics name an ordering and a storage class.
_BARRIER = "{ Semantics::ACQUIRE_RELEASE.bits() | Semantics::UNIFORM_MEMORY.bits() }"
_FUNCTION_CONSTANTS = {"memory_barrier": {"SEMANTICS": _BARRIER}}
# The stage each spirv_std::arch module's functions need; the others run in compute kernels. A ray-tracing function
# is probed in a closest-hit shader, which allows most of them.
_STAGES = {
    "demote_to_helper_invocation_ext": "fragment",
    "derivative": "fragment",
    "mesh_shading": "mesh",
    "primitive": "geometry",
    "ray_tracing": "closest-hit",
}
# Functions of a module that only another stage has (SPIR-V specification, OpKill, OpIgnoreIntersectionKHR,
# OpTerminateRayKHR, OpReportIntersectionKHR and OpEmitMeshTasksEXT).
_FUNCTION_STAGES = {
    "kill": "fragment",
    "ignore_intersection": "any-hit",
    "terminate_ray": "any-hit",
    "report_intersection": "intersection",
    "emit_mesh_tasks_ext": "task",
    "emit_mesh_tasks_ext_payload": "task",
    # OpExecuteCallableKHR passes on callable data, which a callable shader receives as a parameter.
    "execute_callable": "callable",
}
# Functions whose bare name is a CrossGL function that computes something else, so they have no CrossGL spelling.
_DIFFERENT = {
    "Float::fract": "Rust's fract is x - trunc(x); CrossGL's is x - floor(x)",
    "Vec3::fract": "Rust's fract is x - trunc(x); CrossGL's is x - floor(x)",
    "Float::log": "Float::log takes a base; CrossGL's log is the natural logarithm",
}
# Functions whose result no CrossGL function computes, with the reason.
_UNSPELLED = {
    "Float::integer_decode": (
        "no CrossGL function returns a float's mantissa, exponent and sign as integers"
    ),
    "Vec3::any_orthonormal_pair": (
        "no CrossGL function returns two vectors orthonormal to a vector"
    ),
    "Vec3::normalize_and_length": (
        "no CrossGL function returns a vector's direction and length together"
    ),
    "Vec3::try_normalize": "no CrossGL function returns an optional value",
    "Vec3::max_position": (
        "no CrossGL function returns the index of a vector's largest element"
    ),
    "Vec3::min_position": (
        "no CrossGL function returns the index of a vector's smallest element"
    ),
    "Float::classify": "no CrossGL function returns a float's category",
    "Vec3::to_array": "no CrossGL function converts a vector to an array",
    "Vec3::write_to_slice": (
        "no CrossGL function writes a vector's elements to an array"
    ),
}
# CrossGL builds vectors with constructors, and reads or replaces an element with the subscript operator, rather than
# with functions.
_CONSTRUCTORS = {
    "Vec3::new": "vec3",
    "Vec3::splat": "vec3",
    "vector_extract_dynamic": spellings.SUBSCRIPT,
    "vector_insert_dynamic": spellings.ELEMENT,
}
# Parameters a probe passes as literals. The probes target Vulkan 1.1 (SPIR-V 1.3), where a subgroup broadcast's lane
# id and a quad broadcast's index must be constants (SPIR-V specification, OpGroupNonUniformBroadcast and
# OpGroupNonUniformQuadBroadcast).
_IMMEDIATE = {"subgroup_broadcast": (1,), "subgroup_quad_broadcast": (1,)}

_STRING = re.compile(r'(?<!\w)r#*"(?:.|\n)*?"#*|"(?:\\.|[^"\\])*"')
_FUNCTION = re.compile(r"^pub (?:const )?(unsafe )?fn (\w+)", re.M)
_TRAIT_METHOD = re.compile(r"^    (unsafe )?fn (\w+)", re.M)
_IMPL_METHOD = re.compile(r"^    pub (?:const )?(unsafe )?fn (\w+)", re.M)
_MACRO = re.compile(r"^macro_rules! (\w+) \{", re.M)
_TEMPLATE = re.compile(r"pub (?:const )?(?:unsafe )?fn \$name")
# A slice parameter, ``&[T]`` or ``&mut [T]``. The probe passes its input buffer for one the function reads.
_SLICE = re.compile(r"&\s*(mut\s+)?\[([^;\]]+)\]")


def _nesting(text: str, index: int) -> int:
    """How the bracket at ``text[index]`` changes the nesting depth; ``->`` is not a closing bracket."""
    char = text[index]
    if char in "([{<":
        return 1
    if char in ")]}" or (char == ">" and text[index - 1] != "-"):
        return -1
    return 0


def _split(text: str, separators: str = ",") -> list[str]:
    """``text`` split at its top-level separators."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += _nesting(text, index)
        if char in separators and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _group(text: str, start: int) -> tuple[str, int]:
    """The text inside the bracket group that opens at ``text[start]``, and the index just past it."""
    closing = {"(": ")", "<": ">", "{": "}", "[": "]"}[text[start]]
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == text[start]:
            depth += 1
        elif char == closing and not (char == ">" and text[index - 1] == "-"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError(f"unbalanced {text[start]} at offset {start}")


Header = tuple[str, str, str]


def header(text: str, position: int) -> Header:
    """``(generics, parameters, result)`` of the signature whose name ends at ``position``."""
    generics = ""
    while text[position] == " ":
        position += 1
    if text[position] == "<":
        generics, position = _group(text, position)
    parameters, position = _group(text, position)
    depth, end = 0, position
    while depth or text[end] not in "{;":
        depth += _nesting(text, end)
        end += 1
    result, *where = re.split(r"\bwhere\b", text[position:end], maxsplit=1)
    result = result.strip().removeprefix("->").strip()
    return _bound(generics, where[0] if where else ""), parameters, result


def _bound(generics: str, where: str) -> str:
    """``generics`` with the bounds a ``where`` clause gives its unbounded type parameters."""
    bounds = {}
    for predicate in _split(where):
        name, _, bound = predicate.partition(":")
        if re.fullmatch(r"\w+", name.strip()) and bound.strip():
            bounds[name.strip()] = bound.strip()
    return ", ".join(
        f"{part}: {bounds[part]}" if part in bounds else part
        for part in _split(generics)
    )


def clean(text: str) -> str:
    """``text`` without string literals and comments, whose brackets need not balance."""
    return re.sub(r"//[^\n]*", "", _STRING.sub('""', text))


def _generic_types(generics: str) -> tuple[dict[str, Type], list[tuple[str, bool]]]:
    """The probe type of each type parameter, and every parameter in order as ``(name, is_constant)``."""
    types: dict[str, Type] = {}
    order = []
    for part in _split(generics):
        if part.startswith("const "):
            order.append((part.split()[1].rstrip(":"), True))
            continue
        name, _, bounds = part.partition(":")
        name = name.strip()
        order.append((name, False))
        if not bounds.strip():
            # A parameter of any type takes a struct of the probe's own: the payload a task or hit shader declares.
            types[name] = Type.other("udt")
            continue
        if re.search(r"\bFn(?:Mut|Once)?\s*\(", bounds):
            # A closure has no probe; its signature's scalars are not the parameter's type.
            continue
        scalar = re.search(r"Scalar\s*=\s*(?:impl\s+)?(\w+)", bounds)
        if re.search(r"\bVector<\s*bool\b", bounds):
            types[name] = Type("bool", (3,))
            continue
        # A vector of an earlier type parameter's scalar, as in ``V: Vector<T, N>``.
        element = re.search(r"\bVector<\s*(\w+)\s*,", bounds)
        if element and element.group(1) in types:
            types[name] = Type(types[element.group(1)].scalar, (3,))
            continue
        for word in [scalar.group(1) if scalar else "", *re.findall(r"\w+", bounds)]:
            if word in _SCALARS:
                types[name] = Type(_SCALARS[word])
                break
            if word in _BOUNDS:
                types[name] = Type(_BOUNDS[word])
                break
    return types, order


def rust_type(
    spelling: str,
    generics: dict[str, Type],
    self_type: Type,
    aliases: dict[str, str] | None = None,
) -> Type:
    """The probe type of a Rust type spelling; ``aliases`` maps type aliases to what they name."""
    spelling = spelling.strip().removeprefix("crate::")
    spelling = (aliases or {}).get(spelling, spelling).removeprefix("crate::glam::")
    if spelling == "Self":
        return self_type
    if spelling in _SCALARS:
        return Type(_SCALARS[spelling])
    if spelling in generics:
        return generics[spelling]
    glam = re.fullmatch(r"([DIUB]?)Vec([234])", spelling)
    if glam:
        return Type(_GLAM[glam.group(1)], (int(glam.group(2)),))
    vector = re.fullmatch(r"impl\s+Vector<\s*(\w+)\s*,\s*\w+\s*>", spelling)
    if vector:
        element = rust_type(vector.group(1), generics, self_type)
        if not element.opaque and not element.shape:
            return Type(element.scalar, (3,))
    array = re.fullmatch(r"\[(.+);\s*(\d+)\]", spelling)
    if array:
        element = rust_type(array.group(1), generics, self_type, aliases)
        if not element.opaque:
            return Type(element.scalar, element.shape, count=int(array.group(2)))
    # A tuple or an option stays opaque, spelled with the types it holds so that a probe can read its first value.
    held = re.fullmatch(r"\((.+)\)|Option<(.+)>", spelling)
    if held:
        parts = [
            _spelling(rust_type(part, generics, self_type, aliases))
            for part in _split(held.group(1) or held.group(2))
        ]
        if held.group(1):
            return Type.other(f"({', '.join(parts)})")
        return Type.other(f"Option<{parts[0]}>")
    return Type.other(spelling)


def _spelling(kind: Type) -> str:
    """The Rust spelling of a type :func:`rust_type` returns: a scalar, a glam vector or an opaque type."""
    if kind.opaque:
        return kind.opaque
    if kind.shape:
        prefixes = {scalar: prefix for prefix, scalar in _GLAM.items()}
        return f"{prefixes.get(kind.scalar, kind.scalar)}Vec{kind.shape[0]}"
    return {scalar: name for name, scalar in _SCALARS.items()}[kind.scalar]


def signature(
    name: str,
    parsed: Header,
    self_type: Type = VOID,
    aliases: dict[str, str] | None = None,
) -> tuple[Signature | None, str, str]:
    """``(signature, call spelling, note)`` of one function. The spelling names an associated function by its
    type, a method with a leading dot, and constant generic parameters with a turbofish.
    """
    generics, parameters, result = parsed
    types, order = _generic_types(generics)
    params, method = [], False
    for part in _split(parameters):
        if re.fullmatch(r"(?:mut\s+)?&?(?:mut\s+)?self", part):
            method = True
            params.append(Param(self_type))
            continue
        kind = part.partition(":")[2].strip()
        direction = "in"
        sliced = _SLICE.fullmatch(kind)
        element = rust_type(sliced.group(2), types, self_type) if sliced else None
        if element and not element.opaque and not sliced.group(1):
            params.append(Param(element, "buffer"))
            continue
        if element and not element.opaque and len(self_type.shape) == 1:
            # A vector's mutable slice holds one value per component, as glam requires, so the probe passes a local
            # array of that length.
            array = Type(element.scalar, element.shape, count=self_type.shape[0])
            params.append(Param(array, "out"))
            continue
        if kind.startswith("&"):
            direction, kind = "inout", re.sub(r"^&\s*(?:mut\s+)?", "", kind)
        params.append(Param(rust_type(kind, types, self_type, aliases), direction))
    # A function that never returns (``-> !``) is called as a statement, like one that returns nothing.
    returned = (
        rust_type(result, types, self_type, aliases)
        if result not in ("", "!")
        else VOID
    )
    constants = [name for name, constant in order if constant]
    # A constant the arguments determine, such as a vector's length, needs no value.
    missing = [c for c in constants if c not in _CONSTANTS and c != "N"]
    if missing:
        return None, "", f"no probe value for the constant parameter {missing[0]}"
    values = {**_CONSTANTS, **_FUNCTION_CONSTANTS.get(name, {})}
    spelling = ""
    if any(c in values for c in constants):
        arguments = [values.get(n, "_") if c else "_" for n, c in order]
        spelling = f"{name}::<{', '.join(arguments)}>"
    if method:
        spelling = "." + (spelling or name)
    return Signature(returned, tuple(params)), spelling, ""


def _expand(text: str) -> str:
    """``text`` followed by the functions its ``macro_rules!`` macros define. Every lower-case identifier argument of
    an invocation names a function, except the first argument where the macro uses it as ``$scalar``.
    """
    templates: dict[str, str] = {}
    for match in _MACRO.finditer(text):
        body, _ = _group(text, match.end() - 1)
        found = _TEMPLATE.search(body)
        if found:
            end = body.index("{", body.index(")", found.end()))
            templates[match.group(1)] = body[found.start() : end] + "{}"
    expanded = []
    for macro, template in templates.items():
        for match in re.finditer(rf"^{macro}!\(", text, re.M):
            arguments, _ = _group(text, match.end() - 1)
            parts = _split(arguments, ",;")
            candidates = parts[1:] if "$scalar" in template else parts
            names = [p for p in candidates if re.fullmatch(r"[a-z_][a-z0-9_]*", p)]
            for name in names:
                expanded.append(
                    template.replace("$name", name).replace("$scalar", parts[0])
                )
    return text + "\n" + "\n".join(f"{line}\n" for line in expanded)


def arch(
    modules: dict[str, str], glsl_names: set[str] | frozenset[str]
) -> list[Feature]:
    """The public functions of ``spirv_std::arch``: each module's top-level ``pub fn`` items, macro-made ones
    included, and the methods of the ``Derivative`` trait."""
    features = []
    texts = {module: _expand(clean(text)) for module, text in sorted(modules.items())}
    aliases = {
        alias: target.strip()
        for text in texts.values()
        for alias, target in re.findall(r"^pub type (\w+) = ([^;]+);", text, re.M)
    }
    for module, text in texts.items():
        stage = _STAGES.get(module, "compute")
        for match in _FUNCTION.finditer(text):
            name = match.group(2)
            if any(f.name == name for f in features):
                continue
            found, spelling, note = signature(
                name, header(text, match.end()), aliases=aliases
            )
            if match.group(1):
                spelling = "unsafe " + (spelling or name)
            features.append(
                _feature(
                    name,
                    "arch",
                    found,
                    spelling,
                    note,
                    _FUNCTION_STAGES.get(name, stage),
                    glsl_names,
                )
            )
        trait = re.search(r"^pub unsafe trait Derivative\b[^{]*\{", text, re.M)
        if trait:
            body, _ = _group(text, trait.end() - 1)
            for match in _TRAIT_METHOD.finditer(body):
                name = match.group(2)
                found, spelling, note = signature(
                    name, header(body, match.end()), Type("float"), aliases
                )
                features.append(
                    _feature(
                        f"Derivative::{name}",
                        "arch",
                        found,
                        spelling,
                        note,
                        "fragment",
                        glsl_names,
                    )
                )
    return features


def methods(
    text: str,
    opening: str,
    prefix: str,
    self_type: Type,
    group: str,
    glsl_names: set[str] | frozenset[str],
    enabled: frozenset[str] | None = None,
) -> list[Feature]:
    """The functions of the ``impl`` or ``trait`` block that ``opening`` matches, named ``prefix::name``. An
    associated function's call names the type ``self_type`` is (``f32`` for a trait implemented by it). Given the
    crate's ``enabled`` cargo features, a function behind any other feature is left out.
    """
    gated = {
        function
        for feature, function in _GATED.findall(text)
        if enabled is not None and feature not in enabled
    }
    text = clean(text)
    block = re.search(opening, text, re.M)
    if block is None:
        raise ValueError(f"no block matching {opening!r}")
    body, _ = _group(text, block.end() - 1)
    owner = "f32" if prefix == "Float" else prefix
    features: list[Feature] = []
    pattern = _TRAIT_METHOD if opening.startswith("^pub trait") else _IMPL_METHOD
    for match in pattern.finditer(body):
        name = f"{prefix}::{match.group(2)}"
        if match.group(2) in gated or any(f.name == name for f in features):
            continue
        found, spelling, note = signature(
            match.group(2), header(body, match.end()), self_type
        )
        if found is not None and not spelling.startswith("."):
            spelling = f"{owner}::{spelling or match.group(2)}"
        features.append(
            _feature(name, group, found, spelling, note, "compute", glsl_names)
        )
    return features


def _feature(
    name: str,
    group: str,
    found: Signature | None,
    spelling: str,
    note: str,
    stage: str,
    glsl_names: set[str] | frozenset[str],
) -> Feature:
    bare = name.rpartition("::")[2]
    positions = _IMMEDIATE.get(name, ())
    if found is not None and positions:
        params = tuple(
            Param(p.type, "const") if index in positions else p
            for index, p in enumerate(found.params)
        )
        found = Signature(found.result, params)
    counterpart = None
    if name in _CONSTRUCTORS:
        spelled = _CONSTRUCTORS[name]
    elif name in _DIFFERENT:
        spelled, note = "", note or _DIFFERENT[name]
    elif name in _UNSPELLED:
        spelled, note = "", note or _UNSPELLED[name]
    elif name in spellings.COUNTERPARTS["rust"]:
        spelled, counterpart = prototype(spellings.COUNTERPARTS["rust"][name])
    else:
        renamed = spellings.spelling("rust", name, glsl_names)
        spelled = renamed if renamed != name else bare
    return Feature(
        "rust",
        "function",
        name,
        found,
        group=group,
        crossgl=spelled,
        stage=stage,
        note=note,
        spelling=spelling,
        counterpart=counterpart,
    )


def parse(
    modules: dict[str, str],
    num_traits_float: str,
    glam_vec3: str,
    glsl_names: set[str] | frozenset[str] = frozenset(),
) -> list[Feature]:
    """``modules`` maps each ``spirv_std::arch`` module's name (``arch`` for the module itself) to its source."""
    features = arch(modules, glsl_names)
    features += methods(
        num_traits_float,
        r"^pub trait Float\b[^{]*\{",
        "Float",
        Type("float"),
        "float",
        glsl_names,
    )
    features += methods(
        glam_vec3,
        r"^impl Vec3 \{",
        "Vec3",
        Type("float", (3,)),
        "vec3",
        glsl_names,
        _GLAM_FEATURES,
    )
    return sorted(features, key=lambda f: (f.group, f.name))
