"""Memory layouts of rust-gpu buffer bindings, checked against the layout the CrossGL program declares.

A CrossGL storage buffer (``buffer T* data;``) is laid out with std430 and a uniform value (``uniform T u;``) with
std140, as the GLSL and SPIR-V output declare them (GLSL 4.60 section 7.6.2.2). The rust-gpu output binds the same
memory as a slice ``&[T]`` or a reference ``&T``, and rustc_codegen_spirv decorates each member with the offset of
its ``#[repr(C)]`` Rust layout. glam (0.34, the version spirv-std pins) gives its vector and matrix types the
alignment of their component, except ``Vec4``, ``Mat2`` and ``Mat4``, which are 16-byte aligned, so a Rust structure
often places a member somewhere else than std430 does: ``struct { vec3 p; vec3 v; }`` has ``v`` at offset 12 in
Rust and at 16 in std430. The host fills the buffer with the CrossGL layout, so such a binding would read the wrong
bytes; spirv-val rejects some of these layouts and accepts others.

:func:`layout_mismatch` compares the two layouts member by member, nested structures and arrays included, and
names the first member whose offset, or the element stride, differs. The generator refuses that binding.
"""

from __future__ import annotations

import re

from .glsl_buffer_layout import align_to

__all__ = ["layout_mismatch"]

STD430 = "std430"
STD140 = "std140"

# Component sizes of the Rust scalar types the generator spells (``RustCodeGen.map_type``).
_COMPONENT_SIZES = {
    "f32": 4,
    "i32": 4,
    "u32": 4,
    "f64": 8,
    "i64": 8,
    "u64": 8,
    "f16": 2,
    "i16": 2,
    "u16": 2,
    "i8": 1,
    "u8": 1,
}
# glam types whose alignment is 16 bytes on the SPIR-V target (``repr(align(16))`` without ``scalar-math``).
_RUST_ALIGNED_16 = {("vector", "f32", 4), ("matrix", "f32", 2), ("matrix", "f32", 4)}

_VECTOR = re.compile(r"Vec(?P<n>[234])<(?P<component>\w+)>")
_MATRIX = re.compile(r"Mat(?P<c>[234])(?:x(?P<r>[234]))?<(?P<component>\w+)>")
_RUST_ARRAY = re.compile(r"\[(?P<element>.+);\s*(?P<size>\d+)\s*\]")
_WGSL_ARRAY = re.compile(r"array<(?P<element>.+?)(?:,\s*(?P<size>\d+))?>")
_CROSSGL_ARRAY = re.compile(r"(?P<element>.+?)\s*\[\s*(?P<size>\d*)\s*\]")


class _Layout:
    """Size, alignment and member offsets ``(path, offset)`` of a type under one rule."""

    __slots__ = ("size", "align", "offsets")

    def __init__(self, size, align, offsets=()):
        self.size = size
        self.align = align
        self.offsets = list(offsets)


class _Unlaid(Exception):
    """A type with no buffer layout under one of the rules; its message is the refusal reason."""


def _value_kind(rust_type):
    """``(kind, component, count)`` of a Rust scalar, vector or matrix spelling, else ``None``."""
    if rust_type in _COMPONENT_SIZES:
        return "scalar", rust_type, 1
    if rust_type == "bool":
        return "bool", rust_type, 0
    match = _VECTOR.fullmatch(rust_type)
    if match is not None:
        component = match.group("component")
        if component == "bool":
            return "bool", rust_type, 0
        if component in _COMPONENT_SIZES:
            return "vector", component, int(match.group("n"))
        return None
    match = _MATRIX.fullmatch(rust_type)
    if match is not None and match.group("component") in _COMPONENT_SIZES:
        columns = int(match.group("c"))
        if match.group("r") is not None and int(match.group("r")) != columns:
            return "nonsquare", match.group("component"), columns
        return "matrix", match.group("component"), columns
    return None


def _std_vector(component, count, rule):
    size = _COMPONENT_SIZES[component]
    align = size * (2 if count == 2 else 4)
    return _Layout(size * count, align)


def _value_layout(kind, component, count, rule, rust):
    width = _COMPONENT_SIZES[component]
    if kind == "scalar":
        return _Layout(width, width)
    if kind == "vector":
        if rust:
            align = 16 if (kind, component, count) in _RUST_ALIGNED_16 else width
            return _Layout(width * count, align)
        return _std_vector(component, count, rule)
    # A square matrix is ``count`` column vectors of ``count`` rows, in both layouts.
    if rust:
        align = 16 if (kind, component, count) in _RUST_ALIGNED_16 else width
        column = width * count
        return _Layout(
            column * count, align, [(f" column {c}", c * column) for c in range(count)]
        )
    column = _std_vector(component, count, rule)
    column_align = max(16, column.align) if rule == STD140 else column.align
    stride = align_to(column.size, column_align)
    return _Layout(
        stride * count,
        column_align,
        [(f" column {c}", c * stride) for c in range(count)],
    )


class _Calculator:
    def __init__(self, generator, rule, rust):
        self.generator = generator
        self.rule = rule
        self.rust = rust

    def layout(self, type_name, path):
        text = str(type_name).strip()
        members = self.generator.rust_stage_io_members(text)
        if members is not None:
            return self.struct_layout(members, path)
        for pattern in (_CROSSGL_ARRAY, _WGSL_ARRAY):
            array = pattern.fullmatch(text)
            if array is not None:
                return self.array_layout(
                    array.group("element"), array.group("size"), path
                )
        rust_type = str(self.generator.map_type(text)).strip()
        array = _RUST_ARRAY.fullmatch(rust_type)
        if array is not None:
            return self.array_layout(array.group("element"), array.group("size"), path)
        if (
            rust_type != text
            and self.generator.rust_stage_io_members(rust_type) is not None
        ):
            return self.struct_layout(
                self.generator.rust_stage_io_members(rust_type), path
            )
        value = _value_kind(rust_type)
        if value is None:
            raise _Unlaid(
                f"'{path}' has type {text}, which has no rust-gpu buffer layout"
            )
        kind, component, count = value
        if kind == "bool":
            raise _Unlaid(
                f"'{path}' is a {text}; a rust-gpu buffer cannot hold a Boolean"
            )
        if kind == "nonsquare":
            raise _Unlaid(f"'{path}' is a {text}; glam has no non-square matrix")
        laid = _value_layout(kind, component, count, self.rule, self.rust)
        laid.offsets = [(f"{path}{name}", offset) for name, offset in laid.offsets]
        return laid

    def array_layout(self, element, size, path):
        if not size:
            raise _Unlaid(
                f"'{path}' is a runtime-sized array inside the buffer element"
            )
        inner = self.layout(element, f"{path}[]")
        if self.rust:
            stride, align = inner.size, inner.align
        else:
            align = max(16, inner.align) if self.rule == STD140 else inner.align
            stride = align_to(inner.size, align)
        offsets = [(f"{path}[]", 0), (f"{path} stride", stride)]
        offsets += [(name, offset) for name, offset in inner.offsets]
        return _Layout(stride * int(size), align, offsets)

    def struct_layout(self, members, path):
        offset = 0
        align = 1
        offsets = []
        for member_name, member_type, _node in members:
            member_path = f"{path}.{member_name}"
            inner = self.layout(member_type, member_path)
            offset = align_to(offset, inner.align)
            offsets.append((member_path, offset))
            offsets += [
                (name, offset + inner_offset) for name, inner_offset in inner.offsets
            ]
            offset += inner.size
            align = max(align, inner.align)
        if self.rule == STD140 and not self.rust:
            align = max(16, align)
        return _Layout(align_to(offset, align), align, offsets)


def _first_difference(rule, crossgl, rust):
    rust_offsets = dict(rust.offsets)
    for name, offset in crossgl.offsets:
        other = rust_offsets.get(name)
        if other != offset:
            if name.endswith(" stride"):
                member = name[: -len(" stride")]
                return f"'{member}' has the {rule} array stride {offset}, rust-gpu lays it out with stride {other}"
            return f"'{name}' is at {rule} offset {offset}, rust-gpu lays it out at offset {other}"
    return None


def layout_mismatch(generator, type_name, rule, element_of_slice):
    """Why a rust-gpu binding of ``type_name`` would not read the CrossGL ``rule`` layout, else ``None``.

    ``element_of_slice`` compares the array stride of the elements too, as a storage slice ``&[T]`` needs.
    """
    try:
        crossgl = _Calculator(generator, rule, rust=False).layout(
            type_name, str(type_name)
        )
        rust = _Calculator(generator, rule, rust=True).layout(type_name, str(type_name))
    except _Unlaid as error:
        return str(error)
    difference = _first_difference(rule, crossgl, rust)
    if difference is not None:
        return difference
    if element_of_slice:
        stride = align_to(crossgl.size, crossgl.align)
        if stride != rust.size:
            return (
                f"its {type_name} elements have the {rule} array stride {stride}, rust-gpu lays them out with "
                f"stride {rust.size}"
            )
    return None
