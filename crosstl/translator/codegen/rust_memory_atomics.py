"""Atomic operations on storage-buffer and workgroup memory for the Rust generator, as spirv_std atomics.

CrossGL's atomic built-ins act on one memory location and return the value it held before the operation (GLSL 4.60
section 8.11): ``atomicAdd(mem, data)``, ``atomicMin``, ``atomicMax``, ``atomicAnd``, ``atomicOr``, ``atomicXor``,
``atomicExchange`` and ``atomicCompSwap(mem, compare, data)``. ``atomicLoad(mem)`` and ``atomicStore(mem, data)``,
and their Metal spellings ``atomic_load`` and ``atomic_store``, are relaxed loads and stores
(GL_KHR_memory_scope_semantics; MSL 4.0 section 6.15.2). HLSL-style calls, which return the previous value through a
trailing argument, are rewritten into these forms before generation (:mod:`..semantics.atomic_forms`).

``spirv_std::arch`` defines each operation as an ``unsafe`` function of a reference to the location, with the
memory scope and semantics as constant generic parameters (crates/spirv-std/src/arch/atomics.rs). The generator
passes the scopes the SPIR-V generator uses, ``Device`` for a storage buffer and ``Workgroup`` for workgroup memory,
and relaxed (``None``) semantics, which GLSL's atomic functions have:

=================================  ============================================================
CrossGL                            spirv_std::arch
=================================  ============================================================
``atomicAdd(m, v)``                ``atomic_i_add``; ``atomic_f_add`` for ``float``
``atomicMin(m, v)``, ``atomicMax``  ``atomic_s_min``/``atomic_u_min``/``atomic_f_min`` by the location's type
``atomicAnd``, ``atomicOr``,        ``atomic_and``, ``atomic_or``, ``atomic_xor``
``atomicXor``
``atomicExchange(m, v)``           ``atomic_exchange``
``atomicCompSwap(m, c, v)``        ``atomic_compare_exchange(m, v, c)``, relaxed whether or not it stores
``atomicLoad(m)``                  ``atomic_load``
``atomicStore(m, v)``              ``atomic_store``
=================================  ============================================================

A value of another scalar type than the location's is converted with ``as``, since the functions take the
location's type. The result has the location's type.

Refused, through the generation's refusal sink:

- a call with an argument count other than its form's: a byte-address buffer atomic ``atomicAdd(buffer, offset,
  value)``, or an HLSL-style call the canonical rewrite left alone, has a different meaning;
- a location that is not an element, member or variable of a storage buffer or of workgroup memory: an atomic on a
  local or a parameter has no shared memory to act on;
- any operation but a load on a read-only storage buffer;
- a location of another type than a 32- or 64-bit integer, or than ``float`` for the operations SPIR-V defines on
  floats (add, min, max, exchange, load and store).
"""

from __future__ import annotations

from .. import ast as A
from .rust_gpu_interface import STORAGE_SLICE_KIND, WORKGROUP_KIND

__all__ = [
    "RUST_MEMORY_ATOMICS",
    "generate_rust_memory_atomic_call",
    "rust_memory_atomic_location_roots",
    "rust_memory_atomic_result_type",
]

# CrossGL name -> (operation, number of operands).
RUST_MEMORY_ATOMICS = {
    "atomicAdd": ("add", 2),
    "atomicMin": ("min", 2),
    "atomicMax": ("max", 2),
    "atomicAnd": ("and", 2),
    "atomicOr": ("or", 2),
    "atomicXor": ("xor", 2),
    "atomicExchange": ("exchange", 2),
    "atomicCompSwap": ("compare_exchange", 3),
    "atomicLoad": ("load", 1),
    "atomic_load": ("load", 1),
    "atomicStore": ("store", 2),
    "atomic_store": ("store", 2),
}

_INTEGERS = frozenset({"i32", "u32", "i64", "u64"})
_FLOAT_OPERATIONS = frozenset({"add", "min", "max", "exchange", "load", "store"})
_SCOPES = {STORAGE_SLICE_KIND: "Device", WORKGROUP_KIND: "Workgroup"}
_RELAXED = "{ spirv_std::memory::Semantics::NONE.bits() }"


def _location_root(expression):
    """The variable a location expression is an element, member or single component of, or ``None``."""
    while True:
        if isinstance(expression, (A.IdentifierNode, A.VariableNode)):
            return getattr(expression, "name", None)
        if isinstance(expression, A.MemberAccessNode):
            expression = expression.object
        elif isinstance(expression, A.ArrayAccessNode):
            expression = expression.array
        elif (
            isinstance(expression, A.SwizzleNode)
            and len(str(expression.components or "")) == 1
        ):
            expression = expression.vector_expr
        else:
            return None


def _function(operation, scalar):
    """The spirv_std::arch function of ``operation`` on a location of Rust type ``scalar``, or ``None``."""
    if scalar == "f32":
        if operation not in _FLOAT_OPERATIONS:
            return None
        return (
            f"atomic_f_{operation}"
            if operation in {"add", "min", "max"}
            else f"atomic_{operation}"
        )
    if scalar not in _INTEGERS:
        return None
    if operation in {"min", "max"}:
        return f"atomic_{'s' if scalar.startswith('i') else 'u'}_{operation}"
    if operation == "add":
        return "atomic_i_add"
    return f"atomic_{operation}"


def _location_kind(generator, root):
    """``(kind, writable)`` of the storage buffer or workgroup variable ``root`` names, else ``None``."""
    info = (getattr(generator, "rust_entry_resource_infos", None) or {}).get(root)
    if not info or info.get("kind") not in _SCOPES:
        return None
    writable = info.get("kind") == WORKGROUP_KIND or str(
        info.get("type_name", "")
    ).startswith("&mut")
    return info["kind"], writable


def generate_rust_memory_atomic_call(generator, func_name, args):
    """The rust-gpu call for the CrossGL atomic ``func_name``, or ``None`` when ``func_name`` is not one."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    if entry is None or generator.is_user_defined_function(func_name):
        return None
    operation, arity = entry
    args = list(args or [])
    if len(args) != arity:
        return generator.refuse_lowering(
            f"atomic: {func_name} with {len(args)} arguments has no spirv_std form; it takes the memory location "
            f"and {arity - 1} value{'' if arity == 2 else 's'}",
            "()",
        )
    root = _location_root(args[0])
    location = _location_kind(generator, root) if root else None
    if location is None:
        return generator.refuse_lowering(
            f"atomic: {func_name}'s location is not in a storage buffer or workgroup memory, so no other "
            "invocation shares it",
            "()",
        )
    kind, writable = location
    if operation != "load" and not writable:
        return generator.refuse_lowering(
            f"atomic: {func_name} writes to the read-only storage buffer '{root}'", "()"
        )
    scalar = generator.normalize_scalar_type(generator.expression_result_type(args[0]))
    function = _function(operation, scalar)
    if function is None:
        return generator.refuse_lowering(
            f"atomic: {func_name} on a {scalar or 'non-scalar'} location has no spirv_std atomic",
            "()",
        )
    place = generator.generate_expression(args[0])
    borrow = f"&{place}" if operation == "load" else f"&mut {place}"
    values = [_converted(generator, arg, scalar) for arg in args[1:]]
    if operation == "compare_exchange":
        # spirv_std takes the value to store before the comparator; CrossGL's atomicCompSwap takes it last.
        compare, value = values
        values = [value, compare]
        semantics = f"{_RELAXED}, {_RELAXED}"
    else:
        semantics = _RELAXED
    scope = f"{{ spirv_std::memory::Scope::{_SCOPES[kind]} as u32 }}"
    operands = ", ".join([borrow, *values])
    return f"unsafe {{ spirv_std::arch::{function}::<_, {scope}, {semantics}>({operands}) }}"


def _converted(generator, arg, scalar):
    value = generator.generate_expression(arg)
    if generator.normalize_scalar_type(generator.expression_result_type(arg)) == scalar:
        return value
    return f"(({value}) as {scalar})"


def rust_memory_atomic_result_type(generator, func_name, arguments):
    """The CrossGL type of the value an atomic call returns: the type of its location, or ``None``."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    arguments = list(arguments or [])
    if entry is None or len(arguments) != entry[1]:
        return None
    if entry[0] == "store":
        return "void"
    return generator.expression_result_type(arguments[0])


def rust_memory_atomic_location_roots(generator, func_name, args):
    """The bindings an atomic call borrows with ``&mut``, which must therefore be declared ``let mut``."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    args = list(args or [])
    if (
        entry is None
        or entry[0] == "load"
        or not args
        or generator.is_user_defined_function(func_name)
    ):
        return set()
    root = generator.assignment_target_root_name(args[0])
    return {root} if root else set()
