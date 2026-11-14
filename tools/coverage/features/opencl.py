"""OpenCL C built-in functions from Clang's ``opencl-c.h``, ``opencl-c-base.h``, ``Builtins.def`` and
``TokenKinds.def``.

Clang declares each OpenCL C built-in function that is not a compiler built-in in ``opencl-c.h``, once per overload,
under headings that name the OpenCL C specification's sections. The files are pinned in :mod:`..upstream` at the
release of the clang that validates the probes. Every name is one feature in the group of its section, or of the
Khronos extension that declares it, probed with the overload a probe can spell most plainly.

The list follows the specification rather than the header's expansion of it:

- The specification describes a conversion or a vector load or store once for every vector width ``n``
  (``convert_intn``, ``vloadn``); the header declares each width. A width's name joins the scalar function where
  there is one (``convert_int4`` joins ``convert_int``), and otherwise stands as the ``n`` form (``vloadn``), probed
  at width 4.
- Declarations only C++ for OpenCL sees, and vendor extensions (``cl_intel_*``, ``cl_amd_*``, ``cl_arm_*``), which
  are neither part of the specification nor Khronos extensions, are left out.

The rest of the built-ins come from where Clang defines them (:func:`compiler`):

- ``opencl-c-base.h`` defines the ``as_type`` and ``as_typen`` operators as macros, one per type and width, and
  declares ``printf``. Each type is one feature, as each conversion is.
- ``Builtins.def`` lists the pipe, device-side enqueue and generic address space functions, which Clang checks
  itself (``OCL_PIPE``, ``OCL_DSE`` and ``OCL_GAS``), under the headings of their sections.
- ``TokenKinds.def`` lists ``vec_step``, the unary operator an OpenCL C keyword adds.

None of these three files gives a signature a probe can spell: the compiler built-ins are variadic there, and the
operators are macros or keywords. A probe calls each with the declaration the specification gives it, in
:data:`SPECIFIED`, and reinterprets a value of the other kind for an ``as_type`` operator. A name with no declaration
there is listed without a signature, so it reports as untested rather than dropping out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..model import OPENCL_ATOMICS as ATOMIC_TYPES
from ..model import OPENCL_ENUMERATIONS as ENUMERATIONS
from ..model import STRING, VOID, Feature, Param, Signature, Type
from ..objects import UNIFORMS

# Vendor extensions, which the specification and the Khronos registry leave to their vendors.
_VENDOR = re.compile(r"\bcl_(?:intel|amd|arm)_\w+")
# Khronos extensions that add functions; their functions take the extension's name as their group.
_EXTENSION = re.compile(
    r"\b(cl_khr_(?:subgroup_\w+|extended_bit_ops|mipmap_image\w*)|__opencl_c_integer_dot_product\w*|"
    r"__opencl_subgroup_builtins)"
)
# The groups of the extensions the header names by a feature macro.
_EXTENSION_GROUPS = {
    "__opencl_subgroup_builtins": "cl_khr_subgroups",
    "__opencl_c_integer_dot_product_input_4x8bit": "cl_khr_integer_dot_product",
    "__opencl_c_integer_dot_product_input_4x8bit_packed": "cl_khr_integer_dot_product",
}
# A section of the specification. Extension headings, such as the mipmap one inside the image functions, are left to
# the extension's own guard.
_SECTION = re.compile(r"^// OpenCL v[\w./ ,]+? - (.+)$")
# Sections whose headings slug to a long or ambiguous group.
_SECTION_GROUPS = {
    "atomics": "c11-atomic",
    "async-copies-from-global-to-local-memory-local-to-global-memory-and-prefetch": (
        "async-copy"
    ),
    # Builtins.def's heading for the functions opencl-c.h lists under "Enqueue Kernels".
    "enqueue-kernel": "enqueue-kernels",
}
_DECLARATION = re.compile(
    r"([A-Za-z_][\w \t*]*?)\s+__ovld\b((?:\s+__\w+)*)\s+(\w+)\s*\(([^)]*)\)\s*;"
)
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ADDRESS_SPACES = ("__global", "__local", "__private", "__constant", "__generic")
_SCALARS = {
    "bool": "bool",
    "int": "int",
    "uint": "uint",
    "unsigned int": "uint",
    "float": "float",
    "double": "double",
    "half": "half",
    "char": "int8",
    "uchar": "uint8",
    "unsigned char": "uint8",
    "short": "int16",
    "ushort": "uint16",
    "unsigned short": "uint16",
    "long": "int64",
    "ulong": "uint64",
    "unsigned long": "uint64",
    "size_t": "uint64",
    "ptrdiff_t": "int64",
    "intptr_t": "int64",
    "uintptr_t": "uint64",
}
_TYPE_WORDS = frozenset(
    "void bool char uchar signed unsigned short ushort int uint long ulong float double half const volatile".split()
) | frozenset(_ADDRESS_SPACES)
# A vector width the specification writes as ``n``, as the name of one width spells it.
_WIDTH = re.compile(
    r"(convert_[a-z]+|vloada?_half|vstorea?_half|vload|vstore)(2|3|4|8|16)((?:_sat)?(?:_rt[enpz])?)"
)
# The width a function the specification writes with ``n`` is probed at.
PROBED_WIDTH = "4"
# The rank of each scalar type when one overload is chosen: the plainest types first. An atomic function is probed on
# int, which every atomic function takes; atomic float arithmetic is an extension (cl_ext_float_atomics).
_PREFERENCE = ("float", "int", "uint", "bool", "int64", "uint64", "double")
_ATOMIC_PREFERENCE = ("int", "uint", "float", "int64", "uint64", "double")


@dataclass(frozen=True)
class _Declaration:
    name: str
    group: str
    signature: Signature
    # The address space of each parameter's pointer: "" for a generic pointer, "-" for a parameter that is not one.
    spaces: tuple[str, ...]


def opencl_type(spelling: str) -> Type:
    """An OpenCL C value type. Vectors wider than four, ``char`` and the opaque types stay opaque."""
    text = " ".join(re.sub(r"\b(?:const|volatile)\b", " ", spelling).split())
    if text == "void":
        return VOID
    if text in ENUMERATIONS:
        return Type("uint", opaque=text)
    if text in _SCALARS:
        return Type(_SCALARS[text])
    vector = re.fullmatch(r"([a-z]+?)(2|3|4|8|16)", text)
    if vector and vector.group(1) in _SCALARS and vector.group(2) in "234":
        return Type(_SCALARS[vector.group(1)], (int(vector.group(2)),))
    return Type.other(text)


def _type_word(word: str) -> bool:
    return bool(
        word in _TYPE_WORDS
        or word in ENUMERATIONS
        or word.endswith("_t")
        or word.startswith("atomic_")
        or re.fullmatch(r"u?(?:char|short|int|long)\d*|(?:float|double|half)\d*", word)
    )


def _unnamed(text: str) -> str:
    """``text`` without the parameter's name, which some declarations give."""
    match = re.fullmatch(r"(.*?[\w*])\s*\b([A-Za-z_]\w*)", text.strip())
    if not match:
        return text.strip()
    head, last = match.groups()
    if head.endswith("*") or not _type_word(last):
        return head.strip()
    return text.strip()


def parameter(text: str, atomic: bool, first: bool) -> tuple[Param, str]:
    """The parameter ``text`` declares, and the address space of its pointer (``-`` when it is not a pointer)."""
    text = _unnamed(text)
    if "[" in text or "(" in text:
        return Param(Type.other(" ".join(text.split()))), "-"
    if "*" not in text:
        kind = opencl_type(text)
        return Param(kind, "const" if kind.opaque in ENUMERATIONS else "in"), "-"
    words = text.replace("*", " ").split()
    space = next((w for w in words if w in _ADDRESS_SPACES), "")
    pointee = " ".join(w for w in words if w not in _ADDRESS_SPACES)
    if text.count("*") > 1:
        return Param(Type.other(" ".join(text.split()))), space
    kind = opencl_type(pointee)
    if kind.opaque in ATOMIC_TYPES and first:
        return (
            Param(Type(ATOMIC_TYPES[kind.opaque], opaque=kind.opaque), "inout"),
            space,
        )
    if kind.opaque:
        return Param(Type.other(" ".join(text.split()))), space
    if atomic and first:
        return Param(kind, "inout"), space
    if space in ("__global", "__local", "__constant"):
        # A probe passes the address of a private local, which converts to a generic or private pointer only.
        return Param(Type.other(" ".join(text.split()))), space
    if re.search(r"\bconst\b", pointee):
        return Param(kind, "address"), space
    return Param(kind, "out"), space


def _pointee(kind: Type) -> Type:
    """The type an opaque pointer ``kind`` points at, or ``VOID`` when ``kind`` is no such pointer."""
    text = kind.opaque
    if not text.endswith("*") or text.count("*") > 1:
        return VOID
    return opencl_type(
        " ".join(w for w in text[:-1].split() if w not in _ADDRESS_SPACES)
    )


def _rank(declaration: _Declaration) -> tuple:
    """Lower is plainer: spellable types, the preferred scalars, scalars before vectors, then the address space a
    probe can pass (a local's generic address, or a global buffer element for an atomic).
    """
    signature = declaration.signature
    kinds = [signature.result, *(p.type for p in signature.params)]
    spelled = [k for k in kinds if k != VOID and k.opaque not in ENUMERATIONS]
    opaque = sum(1 for k in spelled if not k.scalar and k.opaque not in _SPELLED)
    # A pointer into a named address space stays opaque; the type it points at ranks the overload.
    values = [k for k in [*spelled, *map(_pointee, spelled)] if k.scalar]
    preference = (
        _ATOMIC_PREFERENCE
        if declaration.name.startswith(("atomic_", "atom_"))
        else _PREFERENCE
    )
    scalars = sum(
        preference.index(k.scalar) if k.scalar in preference else len(preference)
        for k in values
    )
    shapes = sum(len(k.shape) for k in values)
    spaces = sum(
        space != ("__global" if param.direction == "inout" else "")
        for param, space in zip(signature.params, declaration.spaces)
        if space != "-"
    )
    return opaque, spaces, scalars, shapes


# The opaque types a probe passes: the images and the sampler it declares.
_SPELLED = frozenset(UNIFORMS["opencl"])


def _kept(stack: list[tuple[str, bool]]) -> bool:
    """Whether a declaration under the open conditionals ``stack`` is part of the list."""
    for condition, otherwise in stack:
        if otherwise:
            continue
        if _VENDOR.search(condition):
            return False
        if re.fullmatch(r"#ifdef\s+__OPENCL_CPP_VERSION__", condition):
            return False
    return True


def _conditional(stack: list[tuple[str, bool]], directive: str) -> None:
    """Track the preprocessor conditional ``directive`` opens, switches or closes on ``stack``."""
    directive = re.sub(r"\s*//.*$", "", directive)
    if directive.startswith("#if"):
        stack.append((directive, False))
    elif directive.startswith(("#else", "#elif")) and stack:
        stack[-1] = (stack[-1][0], True)
    elif directive.startswith("#endif") and stack:
        stack.pop()


def _slug(title: str) -> str:
    title = re.sub(r"\s+functions?\b.*$", "", title.strip(), flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return _SECTION_GROUPS.get(slug, slug)


def declarations(header: str) -> list[_Declaration]:
    """Every declaration the list keeps, in the header's order."""
    found: list[_Declaration] = []
    stack: list[tuple[str, bool]] = []
    section = ""
    pending = ""
    for line in _COMMENT.sub("", header).splitlines():
        stripped = line.strip()
        heading = _SECTION.match(stripped)
        if heading:
            section = _slug(heading.group(1))
            continue
        if stripped.startswith("#"):
            _conditional(stack, stripped)
            continue
        stripped = re.sub(r"//.*$", "", stripped)
        pending = f"{pending} {stripped}".strip()
        if "__ovld" not in pending or ";" not in pending:
            if ";" in pending or "{" in pending or "}" in pending:
                pending = ""
            continue
        match = _DECLARATION.search(pending)
        pending = ""
        if not match or not _kept(stack):
            continue
        result, _, name, params = match.groups()
        if name.startswith("__"):
            continue
        extension = next(
            (
                m.group(1)
                for condition, _ in reversed(stack)
                for m in [_EXTENSION.search(condition)]
                if m
            ),
            "",
        )
        extension = _EXTENSION_GROUPS.get(extension, extension)
        atomic = name.startswith(("atomic_", "atom_"))
        parts = [p for p in params.split(",") if p.strip() and p.strip() != "void"]
        parsed = [parameter(p, atomic, i == 0) for i, p in enumerate(parts)]
        if section == "work-item":
            # Kernels name a dimension with a constant, which an importer maps to a component of a built-in.
            parsed = [(Param(p.type, "const"), space) for p, space in parsed]
        found.append(
            _Declaration(
                name,
                extension or section,
                Signature(opencl_type(result), tuple(p for p, _ in parsed)),
                tuple(space for _, space in parsed),
            )
        )
    return found


def _folded(name: str, names: set[str]) -> tuple[str, bool]:
    """The feature ``name`` belongs to, empty when it joins a scalar function, and whether its declarations are the
    ones the feature is probed with."""
    match = _WIDTH.fullmatch(name)
    if not match:
        return name, True
    base, width, modifiers = match.groups()
    if base.startswith("convert_") and base + modifiers in names:
        return "", False
    return f"{base}n{modifiers}", width == PROBED_WIDTH


def parse(header: str) -> list[Feature]:
    found = declarations(header)
    names = {d.name for d in found}
    candidates: dict[str, list[tuple[bool, _Declaration]]] = {}
    for declaration in found:
        feature, probed = _folded(declaration.name, names)
        if feature:
            candidates.setdefault(feature, []).append((not probed, declaration))
    features = []
    for feature, entries in candidates.items():
        # The first of the plainest overloads, of the probed width where the feature has one.
        _, chosen = min(entries, key=lambda entry: (entry[0], _rank(entry[1])))
        features.append(
            Feature(
                "opencl",
                "function",
                feature,
                chosen.signature,
                group=chosen.group,
                spelling="" if chosen.name == feature else chosen.name,
            )
        )
    return sorted(features, key=lambda f: (f.group, f.name))


# The as_type operators, one macro per type and width: ``#define as_uint4(x) __builtin_astype((x), uint4)``.
_AS_TYPE = re.compile(r"^#define (as_\w+)\(x\) __builtin_astype\(\(x\), (\w+)\)$")
_AS_WIDTH = re.compile(r"(as_[a-z_]+?)(?:2|3|4|8|16)?")
# A function the base header declares without __ovld, as printf, whose variable arguments end its parameters.
_PLAIN = re.compile(
    r"^(?!typedef\b)([A-Za-z_][\w \t*]*?)\s+(\w+)\s*\(([^()]*)\)\s*(?:__attribute__\(\(.*\)\)\s*)?;$"
)
# A built-in Clang checks itself, in the OpenCL C languages that have pipes, device-side enqueue or the generic
# address space.
_LANGBUILTIN = re.compile(
    r'^LANGBUILTIN\((\w+), "[^"]*", "[^"]*", (?:OCL_PIPE|OCL_DSE|OCL_GAS)\)'
)
# A unary operator a keyword adds, as sizeof, and the languages that have the keyword.
_OPERATOR = re.compile(r"^UNARY_EXPR_OR_TYPE_TRAIT\((\w+), \w+, ([^)]*)\)")
_OPERATOR_GROUP = "operators"
_AS_TYPE_GROUP = "as-type"
# The declarations a probe calls the compiler built-ins and the keyword operator with, as the OpenCL C 3.0
# specification declares them, with int for a pipe's packet type and for the type a generic pointer points at.
# Builtins.def declares every one variadic, and the specification overloads read_pipe and write_pipe with a
# reservation; a probe calls the plainer overload.
SPECIFIED = {
    "read_pipe": "int read_pipe(read_only pipe int p, int *ptr)",
    "write_pipe": "int write_pipe(write_only pipe int p, const int *ptr)",
    "reserve_read_pipe": (
        "reserve_id_t reserve_read_pipe(read_only pipe int p, uint num_packets)"
    ),
    "reserve_write_pipe": (
        "reserve_id_t reserve_write_pipe(write_only pipe int p, uint num_packets)"
    ),
    "commit_read_pipe": (
        "void commit_read_pipe(read_only pipe int p, reserve_id_t reserve_id)"
    ),
    "commit_write_pipe": (
        "void commit_write_pipe(write_only pipe int p, reserve_id_t reserve_id)"
    ),
    "work_group_reserve_read_pipe": (
        "reserve_id_t work_group_reserve_read_pipe(read_only pipe int p, uint num_packets)"
    ),
    "work_group_reserve_write_pipe": (
        "reserve_id_t work_group_reserve_write_pipe(write_only pipe int p, uint num_packets)"
    ),
    "work_group_commit_read_pipe": (
        "void work_group_commit_read_pipe(read_only pipe int p, reserve_id_t reserve_id)"
    ),
    "work_group_commit_write_pipe": (
        "void work_group_commit_write_pipe(write_only pipe int p, reserve_id_t reserve_id)"
    ),
    "sub_group_reserve_read_pipe": (
        "reserve_id_t sub_group_reserve_read_pipe(read_only pipe int p, uint num_packets)"
    ),
    "sub_group_reserve_write_pipe": (
        "reserve_id_t sub_group_reserve_write_pipe(write_only pipe int p, uint num_packets)"
    ),
    "sub_group_commit_read_pipe": (
        "void sub_group_commit_read_pipe(read_only pipe int p, reserve_id_t reserve_id)"
    ),
    "sub_group_commit_write_pipe": (
        "void sub_group_commit_write_pipe(write_only pipe int p, reserve_id_t reserve_id)"
    ),
    "get_pipe_num_packets": "uint get_pipe_num_packets(read_only pipe int p)",
    "get_pipe_max_packets": "uint get_pipe_max_packets(read_only pipe int p)",
    "enqueue_kernel": (
        "int enqueue_kernel(queue_t queue, kernel_enqueue_flags_t flags, const ndrange_t ndrange, "
        "void (^block)(void))"
    ),
    "get_kernel_work_group_size": (
        "uint get_kernel_work_group_size(void (^block)(void))"
    ),
    "get_kernel_preferred_work_group_size_multiple": (
        "uint get_kernel_preferred_work_group_size_multiple(void (^block)(void))"
    ),
    "get_kernel_sub_group_count_for_ndrange": (
        "uint get_kernel_sub_group_count_for_ndrange(const ndrange_t ndrange, void (^block)(void))"
    ),
    "get_kernel_max_sub_group_size_for_ndrange": (
        "uint get_kernel_max_sub_group_size_for_ndrange(const ndrange_t ndrange, void (^block)(void))"
    ),
    "to_global": "__global int *to_global(int *ptr)",
    "to_local": "__local int *to_local(int *ptr)",
    "to_private": "__private int *to_private(int *ptr)",
    "vec_step": "int vec_step(float4 a)",
}
_UNSPECIFIED = (
    "SPECIFIED in tools/coverage/features/opencl.py has no declaration for it yet"
)
_SPECIFIED_DECLARATION = re.compile(r"(.+?[\s*])(\w+)\((.*)\)")
# The byte size of each scalar type an as_type operator reinterprets.
_SIZES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "half": 2,
    "int": 4,
    "uint": 4,
    "float": 4,
    "int64": 8,
    "uint64": 8,
    "double": 8,
}
_FLOATS = frozenset({"half", "float", "double"})


def _signature(result: str, params: str) -> Signature:
    """The signature of a declaration's ``result`` and comma-separated ``params``; ``...`` takes one float."""
    parts = [p.strip() for p in params.split(",") if p.strip() not in ("", "void")]
    kinds = []
    for index, part in enumerate(parts):
        if part == "...":
            # The variable arguments of printf: one float, which the probe's format string converts.
            kinds.append(Param(Type("float")))
        elif re.fullmatch(r"__constant const char\s*\*\s*\w*", part):
            kinds.append(Param(STRING, "const"))
        else:
            kinds.append(parameter(part, False, index == 0)[0])
    return Signature(opencl_type(result.strip()), tuple(kinds))


def specified(name: str) -> Signature | None:
    """The signature :data:`SPECIFIED` gives ``name``, or None when it gives none."""
    text = SPECIFIED.get(name)
    if text is None:
        return None
    match = _SPECIFIED_DECLARATION.fullmatch(text)
    if not match or match.group(2) != name:
        raise ValueError(f"cannot read the declaration of {name}: {text}")
    return _signature(match.group(1), match.group(3))


def reinterpreted(result: str) -> str:
    """The scalar an ``as_type`` probe reinterprets as ``result``: a floating-point type of the same size for an
    integer and a signed integer for a floating-point type, so that a value conversion computes something else. A
    byte has no floating-point type of its size, so a char is read from a uchar and a uchar from a char.
    """
    same = [s for s, size in _SIZES.items() if size == _SIZES[result] and s != result]
    if result in _FLOATS:
        return next(s for s in same if s.startswith("int"))
    floating = [s for s in same if s in _FLOATS]
    return (floating or same)[0]


def _as_types(base: str) -> list[Feature]:
    """The ``as_type`` operators, one per type with its widths folded in, probed on the scalar type."""
    types: dict[str, str] = {}
    for line in base.splitlines():
        match = _AS_TYPE.match(line.strip())
        if not match:
            continue
        name, spelled = match.groups()
        feature = _AS_WIDTH.fullmatch(name).group(1)
        if feature == name:
            types[feature] = spelled
        else:
            types.setdefault(feature, "")
    features = []
    for name, spelled in types.items():
        kind = opencl_type(spelled)
        if kind.scalar in _SIZES:
            signature = Signature(kind, (Param(Type(reinterpreted(kind.scalar))),))
            feature = Feature("opencl", "function", name, signature, _AS_TYPE_GROUP)
        else:
            note = f"opencl-c-base.h defines no scalar {name} to probe"
            feature = Feature(
                "opencl", "function", name, None, _AS_TYPE_GROUP, note=note
            )
        features.append(feature)
    return features


def _declared(base: str) -> list[Feature]:
    """The functions the base header declares without ``__ovld``, under their section's heading."""
    features = []
    stack: list[tuple[str, bool]] = []
    section = ""
    for line in _COMMENT.sub("", base).splitlines():
        stripped = line.strip()
        heading = _SECTION.match(stripped)
        if heading:
            section = _slug(heading.group(1))
            continue
        if stripped.startswith("#"):
            _conditional(stack, stripped)
            continue
        match = _PLAIN.match(stripped)
        if not match or not _kept(stack) or match.group(2).startswith("__"):
            continue
        result, name, params = match.groups()
        features.append(
            Feature("opencl", "function", name, _signature(result, params), section)
        )
    return features


def _listed(
    text: str, entry: re.Pattern, default: str = "", keep=lambda _: True
) -> list[Feature]:
    """The names ``entry`` finds in ``text``, in the group of the heading above each or else ``default``, with
    :data:`SPECIFIED`'s signature."""
    features = []
    section = ""
    for line in text.splitlines():
        heading = _SECTION.match(line.strip())
        if heading:
            section = _slug(heading.group(1))
            continue
        match = entry.match(line.strip())
        if not match or not keep(match) or match.group(1).startswith("__"):
            continue
        name = match.group(1)
        signature = specified(name)
        features.append(
            Feature(
                "opencl",
                "function",
                name,
                signature,
                section or default,
                note="" if signature else _UNSPECIFIED,
            )
        )
    return features


def compiler(base: str, builtins: str, keywords: str) -> list[Feature]:
    """The OpenCL C built-ins ``opencl-c.h`` does not declare: the ``as_type`` operators and ``printf`` of
    ``opencl-c-base.h``, the OpenCL C built-ins of ``Builtins.def`` and the keyword operators of ``TokenKinds.def``.
    """
    features = _as_types(base) + _declared(base)
    features += _listed(builtins, _LANGBUILTIN)
    features += _listed(
        keywords,
        _OPERATOR,
        _OPERATOR_GROUP,
        keep=lambda match: "KEYOPENCLC" in match.group(2),
    )
    for feature in features:
        if not feature.group:
            raise ValueError(f"{feature.name} has no section heading above it")
    return sorted(features, key=lambda f: (f.group, f.name))
