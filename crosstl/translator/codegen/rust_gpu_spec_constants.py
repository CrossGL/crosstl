"""CrossGL specialization constants as rust-gpu ``#[spirv(spec_constant)]`` entry-point parameters.

rust-gpu declares a specialization constant as an entry-point parameter, ``#[spirv(spec_constant(id = N, default =
D))] name: u32``, and lowers it to ``OpSpecConstant`` decorated with ``SpecId N``. Its type must be ``u32`` (or an
array of them; rustc_codegen_spirv/src/codegen_cx/entry.rs: "unsupported `#[spirv(spec_constant)]` type ... (expected
`u32` or `[u32; N]`)"), so a CrossGL ``int``, ``float`` or ``bool`` constant is the same 32 bits read back into its
type: ``name as i32``, ``f32::from_bits(name)`` or ``name != 0``. The pipeline writes the 32-bit value of the
CrossGL type, which those conversions read back unchanged.

The entry point converts the parameter into a local of the constant's name, and helpers receive the converted value
as a trailing parameter, like a uniform value (:mod:`.rust_gpu_interface`). A constant whose initializer reads a
specialization constant, a specialization constant without a literal initializer and one of another type are
refused: rust-gpu evaluates neither ``const`` items nor array sizes from a specialization constant.
"""

from __future__ import annotations

import struct

from ..ast import ConstantNode, LiteralNode, PrimitiveType, UnaryOpNode, VariableNode
from .specialization_constants import (
    declares_specialization_id,
    identifier_names,
    specialization_constant_ids,
)

SPEC_CONSTANT_KIND = "spec_constant"

# The CrossGL scalar types a 32-bit specialization constant can carry, and how the entry point reads the u32 back.
_CONVERSIONS = {
    "uint": "{name}",
    "int": "({name} as i32)",
    "float": "f32::from_bits({name})",
    "bool": "({name} != 0)",
}


def _type_name(node) -> str:
    value = getattr(node, "const_type", None) or getattr(node, "var_type", None)
    value = value or getattr(node, "vtype", None)
    return str(getattr(value, "name", value))


def _literal_value(expr):
    """The Python value of a literal initializer (``3``, ``-2``, ``1.5``, ``true``), else ``None``."""
    if isinstance(expr, LiteralNode):
        return expr.value
    if isinstance(expr, UnaryOpNode) and str(getattr(expr, "op", "")) == "-":
        inner = _literal_value(getattr(expr, "operand", None))
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
    return None


def _default_bits(value, type_name: str) -> int | None:
    if isinstance(value, str):
        text = value.strip().lower().rstrip("uf")
        if text in {"true", "false"}:
            value = text == "true"
        else:
            try:
                value = float(text) if type_name == "float" else int(text, 0)
            except ValueError:
                return None
    if type_name == "bool":
        return int(bool(value)) if isinstance(value, (bool, int)) else None
    if type_name == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if type_name == "int" and not -(2**31) <= value < 2**31:
        return None
    if type_name == "uint" and not 0 <= value < 2**32:
        return None
    return value & 0xFFFFFFFF


def spec_constant_infos(generator, ast) -> dict:
    """An entry-parameter record, keyed by name, for every specialization constant of ``ast``; refuses the rest."""
    nodes = [
        node
        for node in (
            *(getattr(ast, "constants", None) or []),
            *(getattr(ast, "global_variables", None) or []),
        )
        if isinstance(node, (ConstantNode, VariableNode))
        and declares_specialization_id(node)
    ]
    if not nodes:
        return {}
    ids = specialization_constant_ids(nodes, "Rust", 2**32 - 1)
    infos = {}
    for node in nodes:
        name = node.name
        type_name = _type_name(node)
        if type_name not in _CONVERSIONS:
            generator.refuse_lowering(
                f"specialization constant: '{name}' has type {type_name}; a rust-gpu specialization constant is "
                "32 bits, so CrossGL int, uint, float and bool constants are lowered"
            )
            continue
        initializer = getattr(node, "value", getattr(node, "initial_value", None))
        bits = (
            0
            if initializer is None
            else _default_bits(_literal_value(initializer), type_name)
        )
        if bits is None:
            generator.refuse_lowering(
                f"specialization constant: '{name}' has no literal default value, which rust-gpu's "
                "spec_constant(default = ...) needs"
            )
            continue
        infos[name] = {
            "node": VariableNode(name, PrimitiveType(type_name)),
            "type_name": type_name,
            "kind": SPEC_CONSTANT_KIND,
            "spec_id": ids[name],
            "default_bits": bits,
        }
    _refuse_dependent_constants(generator, ast, set(infos))
    return infos


def _refuse_dependent_constants(generator, ast, names: set[str]) -> None:
    dependent = set(names)
    for node in (
        *(getattr(ast, "constants", None) or []),
        *(getattr(ast, "global_variables", None) or []),
    ):
        name = getattr(node, "name", None)
        if not isinstance(name, str) or name in names:
            continue
        initializer = getattr(node, "value", getattr(node, "initial_value", None))
        read = identifier_names(initializer) & dependent
        if read:
            dependent.add(name)
            generator.refuse_lowering(
                f"specialization constant: '{name}' is computed from '{sorted(read)[0]}'; rust-gpu reads a "
                "specialization constant only as an entry-point parameter, so a constant cannot follow it"
            )


def spec_constant_parameter(name: str, info: dict) -> str:
    return (
        f"#[spirv(spec_constant(id = {info['spec_id']}, default = {info['default_bits']}))] "
        f"{name}: u32"
    )


def spec_constant_local(name: str, info: dict, rust_type: str) -> str | None:
    """The entry-point local that reads the u32 parameter back as the constant's type, or ``None`` for ``uint``."""
    if info["type_name"] == "uint":
        return None
    return f"let {name}: {rust_type} = {_CONVERSIONS[info['type_name']].format(name=name)};"
