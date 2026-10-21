"""The CrossGL buffer layout that reproduces the bytes of a CUDA or HIP kernel buffer element.

A CUDA buffer element is laid out by C++ rules with the sizes and alignments of CUDA's built-in vector types (CUDA C++
Programming Guide, "Built-in Vector Types"): a two-component vector is aligned to its size, a four-component one to
its size but at most 16 bytes, and a three-component vector to its scalar, so ``float3`` takes 12 bytes at four-byte
alignment. HIP's vector types keep these sizes and alignments. The other targets lay out a CrossGL ``vec3`` with 16
bytes of stride (GLSL 4.60 section 7.6.2.2 for std430, MSL 3.2 table 2.4), so a ``const float3 *`` kernel argument was
read at other offsets than the ones the host wrote, and ``struct { float a; float2 b; }`` took 12 bytes in HLSL
structured buffers (D3D11.3 functional specification section 5.3.7) against CUDA's 16.

:class:`KernelBufferLayouts` computes the CUDA layout of each kernel buffer's element type from the CrossGL type
the importer emits, compares it with the HLSL structured-buffer, std430 and natural MSL layouts of that type, and
names the ``@buffer_layout`` the parameter needs:

* an element that all four rule sets lay out alike keeps CrossGL's default (``None``), and so does a lone ``bool``
  scalar or vector, whose one-byte storage the host binding contract reports;
* an element whose CUDA layout is the HLSL structured-buffer layout is marked ``hlsl_structured``;
* an element whose CUDA layout is the natural MSL layout of its CrossGL type is marked ``metal``; where that needs
  the three-component vector members of its structs stored as packed vectors, those members are spelled
  ``@ packed`` (:mod:`crosstl.translator.codegen.buffer_member_layout`), which MSL places at their scalar
  alignment like CUDA;
* any other element is refused with :class:`KernelBufferLayoutError`, and so is an element whose layout depends on
  the alignment of a four-component 64-bit vector, which is 16 bytes for CUDA's ``double4`` and 32 for CUDA 13's
  ``double4_32a``, and an element holding a struct whose declared alignment (``__align__``, ``alignas``) CrossGL
  does not keep.

Every target then reproduces the marked layout (:mod:`crosstl.translator.semantics.structured_layouts` and the
Metal generator's packed storage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "HLSL_STRUCTURED",
    "METAL",
    "EmittedBufferLayouts",
    "KernelBufferLayoutError",
    "KernelBufferLayouts",
]

HLSL_STRUCTURED = "hlsl_structured"
METAL = "metal"

_SCALAR_BYTES = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}
_VECTOR = re.compile(r"^vec([234])<(\w+)>$")
_ARRAY = re.compile(r"^array<(.+),(\d+)>$")
_ALIGNMENT = re.compile(
    r"^(?:__align__|alignas|__attribute__)\s*\(+\s*(?:aligned\s*\(\s*)?(\d+)\s*\)+$"
)


class KernelBufferLayoutError(TranslationDiagnostic):
    """A CUDA or HIP kernel buffer whose element layout no CrossGL buffer layout reproduces."""

    code = f"{CODE_PREFIX}kernel-buffer-layout-unsupported"


class _Unknown(Exception):
    """The element type has a part this module does not model."""


@dataclass(frozen=True)
class _Type:
    form: str  # scalar, vector, array or struct
    size: int = 0  # bytes per scalar
    lanes: int = 1
    boolean: bool = False
    element: _Type | None = None
    count: int = 0
    name: str = ""
    members: tuple = ()
    alignment: int = 0  # a struct's declared alignment, 0 when none


@dataclass
class _Struct:
    members: tuple  # ((CrossGL type text, name), ...)
    alignment: int = 0
    unmodelled: str = ""  # why this module cannot place the struct, empty when it can


def _round(value, alignment):
    return -(-value // alignment) * alignment


def _vector(size, lanes, rules, packed):
    """``(size, alignment)`` of a vector of ``lanes`` scalars of ``size`` bytes under ``rules``."""
    if lanes == 1 or rules == "hlsl":
        return size * lanes, size
    if rules in ("cuda", "cuda32"):
        if lanes == 3:
            return size * 3, size
        extent = size * lanes
        return extent, min(extent, 32 if rules == "cuda32" else 16)
    if rules == "metal":
        if packed:
            return size * lanes, size
        extent = size * (4 if lanes == 3 else lanes)
        return extent, extent
    return size * lanes, size * (2 if lanes == 2 else 4)


def _packable(node):
    """Whether a struct member of type ``node`` has a packed MSL form CUDA's layout needs."""
    if node.form == "array":
        node = node.element
    return node.form == "vector" and node.lanes == 3


def _place(node, rules, packed_structs, packed=False):
    """``(size, alignment, leaves)`` of ``node`` under ``rules``; ``leaves`` are ``(offset, width)`` of its scalars."""
    scalar = 4 if node.boolean and rules in ("hlsl", "std430") else node.size
    if node.form == "scalar":
        return scalar, scalar, ((0, scalar),)
    if node.form == "vector":
        size, alignment = _vector(scalar, node.lanes, rules, packed)
        return (
            size,
            alignment,
            tuple((scalar * lane, scalar) for lane in range(node.lanes)),
        )
    if node.form == "array":
        size, alignment, leaves = _place(node.element, rules, packed_structs, packed)
        stride = _round(size, alignment)
        return (
            stride * node.count,
            alignment,
            tuple(
                (index * stride + offset, width)
                for index in range(node.count)
                for offset, width in leaves
            ),
        )
    offset, alignment, leaves = 0, 1, []
    pack = rules == "metal" and node.name in packed_structs
    for member in node.members:
        size, member_alignment, member_leaves = _place(
            member, rules, packed_structs, pack and _packable(member)
        )
        offset = _round(offset, member_alignment)
        leaves.extend((offset + start, width) for start, width in member_leaves)
        offset += size
        alignment = max(alignment, member_alignment)
    if rules in ("cuda", "cuda32"):
        alignment = max(alignment, node.alignment)
    return _round(offset, alignment), alignment, tuple(leaves)


def _layout(node, rules, packed_structs=frozenset()):
    size, alignment, leaves = _place(node, rules, packed_structs)
    return _round(size, alignment), leaves


def _struct_names(node):
    if node.form == "array":
        yield from _struct_names(node.element)
    elif node.form == "struct":
        yield node.name
        for member in node.members:
            yield from _struct_names(member)


@dataclass
class KernelBufferLayouts:
    """The ``@buffer_layout`` of each kernel buffer element and the struct members CrossGL stores packed.

    ``structs`` maps CrossGL struct names to ``(members, attributes, union)``, members being ``(type text, name)``
    pairs, attributes the source's struct attributes and ``union`` whether the record is a union, whose overlapping
    storage CrossGL does not keep; ``aliases`` maps type alias names to CrossGL type text; ``label`` names the source
    language in diagnostics.
    """

    structs: dict
    aliases: dict = field(default_factory=dict)
    label: str = "CUDA"
    packed_structs: set = field(default_factory=set)
    layouts: dict = field(default_factory=dict)

    def __post_init__(self):
        self.structs = {
            name: self._struct(*entry) for name, entry in self.structs.items()
        }

    @staticmethod
    def _struct(members, attributes, union=False):
        alignment, unmodelled = 0, "it is a union" if union else ""
        for attribute in attributes or ():
            text = str(attribute).strip()
            lowered = text.lower()
            if "align" not in lowered and "packed" not in lowered:
                continue
            match = _ALIGNMENT.match(text)
            if match is None:
                unmodelled = (
                    unmodelled
                    or f"its layout attribute {text} has no CrossGL buffer layout"
                )
                continue
            alignment = max(alignment, int(match.group(1)))
        return _Struct(tuple(members), alignment, unmodelled)

    def plan(self, elements):
        """Decide the layout of every ``(context, element type text)`` in ``elements``.

        Raises :class:`KernelBufferLayoutError` for an element no CrossGL buffer layout reproduces.
        """
        nodes = []
        for context, text in elements:
            key = self._key(text)
            try:
                node = self._type(key, ())
            except _Unknown:
                continue
            nodes.append((context, key, node))
            if key not in self.layouts:
                self.layouts[key] = self._decide(context, node)
        # Packing a struct's members for one buffer changes the MSL layout of every buffer that holds it, and a
        # struct that one buffer holds in HLSL's layout and another in MSL's has one declaration on Metal. Both
        # only pack more structs and move buffers from the HLSL layout to MSL's, so the loop ends.
        while True:
            state = (frozenset(self.packed_structs), tuple(self.layouts.items()))
            for context, key, node in nodes:
                self.layouts[key] = self._confirm(context, node, self.layouts[key])
            self._separate(nodes)
            if state == (frozenset(self.packed_structs), tuple(self.layouts.items())):
                return

    def _separate(self, nodes):
        """Give the MSL layout to every HLSL-layout buffer that shares a struct with an MSL-layout buffer.

        The Metal generator declares the structs of an ``hlsl_structured`` buffer with packed storage at scalar
        alignment, which would move them inside the ``metal`` buffers that hold them too.
        """
        metal_structs = {
            name
            for _context, key, node in nodes
            if self.layouts[key] == METAL
            for name in _struct_names(node)
        }
        for context, key, node in nodes:
            names = set(_struct_names(node))
            if self.layouts[key] != HLSL_STRUCTURED or not names & metal_structs:
                continue
            packed = self.packed_structs | names
            if _layout(node, "cuda") != _layout(node, "metal", packed):
                shared = sorted(names & metal_structs)[0]
                self._refuse(
                    context,
                    f"its element holds struct {shared}, which another buffer needs in MSL's layout, and the "
                    f"{self.label} layout of its element is not MSL's",
                )
            self.packed_structs.update(names)
            self.layouts[key] = METAL

    def layout(self, text):
        """The ``@buffer_layout`` value for a buffer of element type ``text``, or ``None``."""
        return self.layouts.get(self._key(text))

    def packed_member(self, struct_name, member_text):
        """Whether a member of ``struct_name`` with CrossGL type ``member_text`` is spelled ``@ packed``."""
        if struct_name not in self.packed_structs:
            return False
        try:
            return _packable(self._type(self._key(member_text), ()))
        except _Unknown:
            return False

    @staticmethod
    def _key(text):
        return re.sub(r"\s+", "", str(text or ""))

    def _type(self, text, resolving):
        if text in _SCALAR_BYTES:
            return _Type("scalar", _SCALAR_BYTES[text], boolean=text == "bool")
        match = _VECTOR.match(text)
        if match is not None:
            scalar = match.group(2)
            if scalar not in _SCALAR_BYTES:
                raise _Unknown(text)
            return _Type(
                "vector",
                _SCALAR_BYTES[scalar],
                int(match.group(1)),
                boolean=scalar == "bool",
            )
        match = _ARRAY.match(text)
        if match is not None:
            element = self._type(match.group(1), resolving)
            return _Type("array", element=element, count=int(match.group(2)))
        if text in resolving:
            raise _Unknown(text)
        if text in self.aliases:
            return self._type(self._key(self.aliases[text]), (*resolving, text))
        struct = self.structs.get(text)
        if struct is None:
            raise _Unknown(text)
        members = tuple(
            self._type(self._key(member), (*resolving, text))
            for member, _name in struct.members
        )
        if not members:
            raise _Unknown(text)
        return _Type("struct", name=text, members=members, alignment=struct.alignment)

    def _refuse(self, context, reason):
        raise KernelBufferLayoutError(f"{context}: {reason}", target="cgl")

    def _check_modelled(self, context, node):
        for name in _struct_names(node):
            reason = self.structs[name].unmodelled
            if reason:
                self._refuse(context, f"its element holds struct {name}, and {reason}")
        cuda = _layout(node, "cuda")
        if cuda != _layout(node, "cuda32"):
            self._refuse(
                context,
                f"the {self.label} layout of its element depends on the alignment of a four-component 64-bit "
                "vector, which is 16 bytes for double4 and 32 bytes for double4_32a",
            )
        return cuda

    def _decide(self, context, node):
        cuda = self._check_modelled(context, node)
        if node.form in ("scalar", "vector") and node.boolean:
            return None
        hlsl, std430 = _layout(node, "hlsl"), _layout(node, "std430")
        metal = _layout(node, "metal", self.packed_structs)
        if cuda == hlsl == std430 == metal:
            return None
        if cuda == hlsl:
            return HLSL_STRUCTURED
        if cuda == metal:
            return METAL
        packed = self.packed_structs | set(_struct_names(node))
        if cuda == _layout(node, "metal", packed):
            self.packed_structs.update(packed)
            return METAL
        reason = f"the {self.label} layout of its element matches neither the HLSL structured-buffer layout nor the "
        reason += "natural MSL layout of its CrossGL type"
        if any(self.structs[name].alignment for name in _struct_names(node)):
            reason += ", which does not keep the declared alignment of its structs"
        self._refuse(context, reason)

    def _confirm(self, context, node, layout):
        """``layout`` once every struct whose members are packed is known; refuses an element it no longer fits."""
        if layout == HLSL_STRUCTURED:
            return layout
        if not any(name in self.packed_structs for name in _struct_names(node)):
            return layout
        cuda = _layout(node, "cuda")
        if cuda == _layout(node, "metal", self.packed_structs):
            return METAL
        if cuda == _layout(node, "hlsl"):
            return HLSL_STRUCTURED
        self._refuse(
            context,
            f"the {self.label} layout of its element differs from the MSL layout its packed struct members give it",
        )


class EmittedBufferLayouts:
    """The struct members and kernel buffer parameters an importer emitted, marked once the program is rendered.

    The importer reports each struct (:meth:`struct`), each struct member line (:meth:`member`) and each kernel
    buffer parameter line (:meth:`buffer`) with its index in the output; :meth:`apply` then plans the layouts with
    :class:`KernelBufferLayouts` from the CrossGL types that were emitted, so the plan sees exactly the program the
    importer wrote, and marks those lines.
    """

    def __init__(self, label):
        self.label = label
        self.structs = {}
        self.members = []  # (struct name, member type text, output index)
        self.buffers = []  # (context, element type text, output index)
        self.aliases = {}

    def struct(self, name, attributes, union=False):
        self.structs.setdefault(name, ([], attributes, union))

    def member(self, struct_name, member_type, member_name, index):
        entry = self.structs.get(struct_name)
        if entry is not None:
            entry[0].append((member_type, member_name))
        self.members.append((struct_name, member_type, index))

    def alias(self, name, crossgl_type):
        self.aliases.setdefault(name, crossgl_type)

    def buffer(self, context, element_type, index):
        self.buffers.append((context, element_type, index))

    def apply(self, output):
        """Mark the emitted lines of ``output`` in place; raises :class:`KernelBufferLayoutError` for a buffer
        whose element layout no CrossGL buffer layout reproduces."""
        if not self.buffers:
            return
        planner = KernelBufferLayouts(
            {
                name: (tuple(members), attributes, union)
                for name, (members, attributes, union) in self.structs.items()
            },
            self.aliases,
            self.label,
        )
        planner.plan((context, element) for context, element, _index in self.buffers)
        for _context, element, index in self.buffers:
            layout = planner.layout(element)
            if layout:
                output[index] = output[index].replace(
                    "var<storage", f"@buffer_layout({layout}) var<storage", 1
                )
        for struct_name, member_type, index in self.members:
            if planner.packed_member(struct_name, member_type):
                line = output[index]
                body = line.lstrip()
                output[index] = f"{line[: len(line) - len(body)]}@packed {body}"
