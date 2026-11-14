"""Pointer parameters and locals of functions, which the Slang generator has no lowering for.

A Slang pointer ``T*`` is ``Ptr<T, Access.ReadWrite, AddressSpace.Device>``: the address of device memory, as a
buffer device address or a pointer-typed shader parameter supplies it (Slang User's Guide, "Pointers (limited)").
The guide's limitations rule out the other memories a CrossGL pointer can point into:

- "It is invalid to define a pointer to a local variable." slangc accepts ``T* p = &local`` only where its
  optimizer removes the pointer, and otherwise emits SPIR-V that spirv-val rejects (a ``PhysicalStorageBuffer``
  access chain into a ``Function`` variable).
- A bound buffer has no device address, so ``void f(float* p)`` cannot receive one; slangc rejects ``f(buffer)``
  with "expected an expression of type 'Ptr<float, ...>', got 'RWStructuredBuffer<float>'".
- ``&`` of a ``groupshared`` element has type ``Ptr<T, ..., AddressSpace.Device>``, so no pointer into workgroup
  memory can be formed with it either.

The Slang generator wrote such a pointer as ``T*``, which slangc rejected or miscompiled.
:func:`refuse_slang_pointer_declarations` refuses each function parameter and local declared with a pointer type,
naming the memory it points into. Storage buffers declared as ``buffer T* name`` at module or stage scope are buffers,
not pointers, and are not affected.
"""

from __future__ import annotations

from .. import ast as A
from .generic_pointer_spaces import (
    function_body_nodes,
    is_generic_pointer_type,
    program_function_nodes,
)
from .stage_utils import STAGE_QUALIFIER_NAMES, function_stage_name

__all__ = ["refuse_slang_pointer_declarations"]

# Why Slang has no pointer into each memory, by the address space a CrossGL pointer type names.
_WHY_NOT = {
    "thread": "Slang defines no pointer to a local variable",
    "device": "a bound Slang buffer has no device address",
    "constant": "a bound Slang buffer has no device address",
    "threadgroup": "Slang's & of groupshared memory gives a device pointer",
}
_THREAD_SPACES = frozenset({None, "", "thread", "private", "function", "local"})


def _pointed_memory(type_node):
    """The memory a declared type points into and why Slang cannot point there, or ``None`` for a non-pointer."""
    if isinstance(type_node, A.PointerType):
        space = getattr(type_node, "address_space", None)
        space = "thread" if space in _THREAD_SPACES else str(space)
        why = _WHY_NOT.get(
            space, "a Slang pointer addresses only device memory by address"
        )
        return f"{space} memory", why
    if is_generic_pointer_type(type_node):
        return (
            "memory of an unknown address space",
            "nothing determines the pointer's address space",
        )
    return None


def _entry_ids(ast, functions):
    entries = {
        id(f) for f in functions if function_stage_name(f) in STAGE_QUALIFIER_NAMES
    }
    stages = getattr(ast, "stages", None) or {}
    stage_list = (
        [s for key in stages.keys() for s in stages.all_for_stage(key)]
        if isinstance(stages, A.StageMap)
        else list(stages.values()) if isinstance(stages, dict) else []
    )
    entries.update(id(getattr(stage, "entry_point", None)) for stage in stage_list)
    return entries


def refuse_slang_pointer_declarations(ast, refuse):
    """Refuse, through ``refuse(reason)``, every pointer local and helper pointer parameter of ``ast``'s functions.

    The pointer parameters of an entry point bind buffers, which the generator declares as buffer resources.
    """
    functions = program_function_nodes(ast)
    entries = _entry_ids(ast, functions)
    for function in functions:
        declarations = [
            (parameter, getattr(parameter, "param_type", None))
            for parameter in getattr(function, "parameters", None) or []
            if id(function) not in entries
        ]
        declarations.extend(
            (node, node.var_type)
            for node in function_body_nodes(function)
            if isinstance(node, A.VariableNode)
        )
        for declaration, declared in declarations:
            pointed = _pointed_memory(declared)
            if pointed is None:
                continue
            memory, why = pointed
            refuse(
                f"pointer declaration: '{function.name}.{declaration.name}' points into {memory}, and {why} "
                '(Slang User\'s Guide, "Pointers (limited)")'
            )
