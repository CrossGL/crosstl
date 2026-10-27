"""File-scope resource declarations that CUDA and HIP device code reads.

nvcc and hipcc reject device code that names an ordinary namespace-scope variable, because such a variable lives
in host memory ("identifier is undefined in device code", "reference to __host__ variable in __global__ function").
CrossGL declares storage buffers, textures, samplers and images at shader or stage scope, so the CUDA and HIP
generators emit them at file scope. Each one, and each companion declaration the generators add for it (a buffer
length, an append counter or texture query metadata), therefore needs an execution-space qualifier.

``__device__`` gives the declaration a device-memory slot that the host binds by name before a launch
(``cudaMemcpyToSymbol``, ``cuModuleGetGlobal``, ``hipMemcpyToSymbol`` or ``hipModuleGetGlobal``). That is the
binding the ``// CrossGL resource metadata`` comment above each resource describes, and it leaves the signatures of
kernels and helper functions unchanged. Uniforms are ``__constant__`` and workgroup variables ``__shared__``
already, and keep the space they name.

Constants follow the CUDA rule for const-qualified namespace-scope variables: device code may read a const scalar
with a constant initializer directly, but not a const array, vector or structure. Constant arrays and vectors are
therefore ``__device__`` variables too. A device variable must be initialized statically, and ``make_float2`` and
its relatives are not constexpr, so a vector constructor in such an initializer becomes the equivalent brace list:
the built-in vector types are aggregates of their components in CUDA and have a constexpr component constructor in
HIP.
"""

from __future__ import annotations

import re

EXECUTION_SPACES = ("__device__", "__constant__", "__shared__", "__managed__")
# Qualifiers that make a CrossGL declaration a storage resource whatever its type: `buffer float* values;`.
STORAGE_QUALIFIERS = frozenset({"buffer", "storage"})
# The constructors of CUDA's and HIP's built-in vector types. half2 and bfloat162 are classes, not aggregates.
VECTOR_CONSTRUCTOR = re.compile(
    r"\bmake_(?:u?char|u?short|u?int|u?long|u?longlong|float|double)[1-4]\("
)


def names_execution_space(declaration: str) -> bool:
    """Whether a C++ declaration already names a CUDA or HIP execution space."""
    return any(word in EXECUTION_SPACES for word in declaration.split())


def in_device_memory(declaration: str) -> str:
    """Place a file-scope declaration in device memory unless it names an execution space already."""
    if not declaration or names_execution_space(declaration):
        return declaration
    return f"__device__ {declaration}"


def is_storage_declaration(node) -> bool:
    """Whether a CrossGL variable is a storage buffer by qualifier (``buffer float* values;``)."""
    return any(
        str(qualifier).lower() in STORAGE_QUALIFIERS
        for qualifier in getattr(node, "qualifiers", None) or []
    )


def _matching_parenthesis(text: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                return position
    return -1


def braced_vector_constructors(initializer: str) -> str:
    """Write the built-in vector constructors an initializer consists of as brace lists of their components.

    Only a constructor that is a whole initializer or a whole element of a brace list is rewritten; when any other
    appears (``make_float2(a, b).x``), the initializer is returned unchanged.
    """
    result = []
    position = 0
    for match in VECTOR_CONSTRUCTOR.finditer(initializer):
        closing = _matching_parenthesis(initializer, match.end() - 1)
        before = initializer[: match.start()].rstrip()
        after = initializer[closing + 1 :].lstrip() if closing >= 0 else ""
        if (
            closing < 0
            or (before and before[-1] not in "{,")
            or (after and after[0] not in ",}")
        ):
            return initializer
        result.append(initializer[position : match.start()])
        result.append("{")
        result.append(initializer[match.end() : closing])
        result.append("}")
        position = closing + 1
    if not result:
        return initializer
    result.append(initializer[position:])
    return "".join(result)


def constant_in_device_memory(declaration: str) -> str:
    """Place a file-scope constant array or vector in device memory; a constant scalar stays a host constant."""
    declarator, separator, initializer = declaration.partition(" = ")
    braced = braced_vector_constructors(initializer) if separator else initializer
    if "[" in declarator or braced != initializer:
        return in_device_memory(f"{declarator}{separator}{braced}")
    return declaration


def is_constant_declaration(node) -> bool:
    return any(
        str(qualifier).lower() == "const"
        for qualifier in getattr(node, "qualifiers", None) or []
    )


def _unchanged(declaration: str) -> str:
    return declaration


def at_file_scope(codegen) -> bool:
    """Whether the generator is emitting file-scope declarations rather than a function body."""
    return getattr(codegen, "current_function_name", None) is None


def file_scope_placement(codegen, node, metadata_comment: str = ""):
    """The function that places the declarations the generator emits for ``node`` where device code reads them.

    ``metadata_comment`` is the resource metadata comment the generator wrote for the node; it is empty for
    anything that is not a resource. Declarations inside a function body are automatic variables and stay as they
    are.
    """
    if not at_file_scope(codegen):
        return _unchanged
    if metadata_comment or is_storage_declaration(node):
        return in_device_memory
    if is_constant_declaration(node):
        return constant_in_device_memory
    return _unchanged
