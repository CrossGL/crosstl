"""Atomic operations on buffer and workgroup memory for the Rust generator.

CrossGL's atomic built-ins act on a memory location and return the value it held before the operation (GLSL 4.60
section 8.11): ``atomicAdd(mem, data)``, ``atomicMin``, ``atomicMax``, ``atomicAnd``, ``atomicOr``, ``atomicXor``,
``atomicExchange`` and ``atomicCompSwap(mem, compare, data)``. HLSL-style calls, which return the previous value
through a trailing argument, are rewritten into these forms before generation (:mod:`..semantics.atomic_forms`).

Rust has no free functions of those names. The generator wrote the calls as they stood, which named nothing the
generated crate defines. The Rust runtime's ``gpu`` module, imported by every generated program, spells each
operation as a function that takes the location as a ``&mut`` place and returns its previous value, as rust-gpu's
``spirv_std::arch`` atomics do (``atomic_i_add(ptr: &mut I, value: I) -> I``):

========================  ===========================================
CrossGL                   Rust
========================  ===========================================
``atomicAdd(m, v)``       ``atomic_add(&mut m, v)``
``atomicMin(m, v)``       ``atomic_min(&mut m, v)``
``atomicMax(m, v)``       ``atomic_max(&mut m, v)``
``atomicAnd(m, v)``       ``atomic_and(&mut m, v)``
``atomicOr(m, v)``        ``atomic_or(&mut m, v)``
``atomicXor(m, v)``       ``atomic_xor(&mut m, v)``
``atomicExchange(m, v)``  ``atomic_exchange(&mut m, v)``
``atomicCompSwap(m, c, v)``  ``atomic_comp_swap(&mut m, c, v)``
========================  ===========================================

The result has the type of the location. A program function or local of the same name would shadow the glob
import ``use gpu::*``, so the call is then written with the module path, ``gpu::atomic_add``.

Refused, through the generation's refusal sink:

- a call with an argument count other than its form's: a byte-address buffer atomic ``atomicAdd(buffer, offset,
  value)``, or an HLSL-style call the canonical rewrite left alone, has a different meaning;
- a first argument that is not a place expression (a variable, or a member, element or single-component swizzle of
  one), because an atomic needs one memory location and a borrow of a temporary would update nothing.
"""

from __future__ import annotations

from .. import ast as A

__all__ = [
    "RUST_MEMORY_ATOMICS",
    "generate_rust_memory_atomic_call",
    "rust_memory_atomic_location_roots",
    "rust_memory_atomic_result_type",
]

# CrossGL atomic built-in -> (gpu runtime function, argument count).
RUST_MEMORY_ATOMICS = {
    "atomicAdd": ("atomic_add", 2),
    "atomicMin": ("atomic_min", 2),
    "atomicMax": ("atomic_max", 2),
    "atomicAnd": ("atomic_and", 2),
    "atomicOr": ("atomic_or", 2),
    "atomicXor": ("atomic_xor", 2),
    "atomicExchange": ("atomic_exchange", 2),
    "atomicCompSwap": ("atomic_comp_swap", 3),
}

_RUNTIME_MODULE = "gpu"


def _is_place(expression):
    """Whether ``expression`` names a memory location Rust can borrow mutably."""
    while True:
        if isinstance(expression, (A.IdentifierNode, A.VariableNode)):
            return True
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
            return False


def generate_rust_memory_atomic_call(generator, func_name, args):
    """The Rust call for the CrossGL atomic ``func_name``, or ``None`` when ``func_name`` is not one."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    if entry is None or generator.is_user_defined_function(func_name):
        return None
    runtime_name, arity = entry
    args = list(args or [])
    if len(args) != arity:
        return generator.refuse_lowering(
            f"call to {func_name} with {len(args)} arguments: the Rust runtime's {runtime_name} takes the memory "
            f"location and {arity - 1} value{'s' if arity > 2 else ''}, and this form has no Rust spelling"
        )
    if not _is_place(args[0]):
        return generator.refuse_lowering(
            f"call to {func_name}: its first argument is not a memory location, so there is nothing to update "
            "atomically"
        )
    callee = runtime_name
    if generator.rust_function_name_is_shadowed(runtime_name):
        callee = f"{_RUNTIME_MODULE}::{runtime_name}"
    location = generator.generate_expression(args[0])
    operands = ", ".join(generator.generate_expression(arg) for arg in args[1:])
    return f"{callee}(&mut {location}, {operands})"


def rust_memory_atomic_result_type(generator, func_name, arguments):
    """The CrossGL type of the value an atomic call returns: the type of its location, or ``None``."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    arguments = list(arguments or [])
    if entry is None or len(arguments) != entry[1]:
        return None
    return generator.expression_result_type(arguments[0])


def rust_memory_atomic_location_roots(generator, func_name, args):
    """The bindings an atomic call borrows with ``&mut``, which must therefore be declared ``let mut``."""
    entry = RUST_MEMORY_ATOMICS.get(func_name)
    args = list(args or [])
    if entry is None or not args or generator.is_user_defined_function(func_name):
        return set()
    root = generator.assignment_target_root_name(args[0])
    return {root} if root else set()
