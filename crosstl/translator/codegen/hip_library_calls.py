"""HIP spelling of device library calls whose CrossGL form names one overload set.

CrossGL writes some CUDA and HIP device functions under the name of their overload set, while HIP declares them
with argument types that a call of another type cannot reach unambiguously:

- ``__ffsll``: HIP declares it for ``long long`` and for ``unsigned long long``, so an ``int64_t``, ``uint64_t``
  (``long`` on LP64) or 32-bit argument converts equally well to both and the call is ambiguous. The lowest set bit
  of a two's complement value does not depend on its signedness, so the argument is converted to
  ``unsigned long long``.
- ``norm``/``rnorm`` of an array: CrossGL spells both precisions without the ``f`` suffix. HIP's ``norm`` and
  ``rnorm`` take ``const double*`` and ``normf`` and ``rnormf`` take ``const float*``, so the name is chosen from
  the element type of the array, and an array of any other element type is refused.
"""

from __future__ import annotations

from .hip_refusals import HipLoweringRefused

__all__ = [
    "FFS64_ARGUMENT_TYPES",
    "ffs64_argument",
    "generate_hip_library_call",
]

# The argument types of HIP's two ``__ffsll`` overloads; any other argument type needs a conversion.
FFS64_ARGUMENT_TYPES = frozenset(
    {
        "long long",
        "long long int",
        "unsigned long long",
        "unsigned long long int",
    }
)

_EUCLIDEAN_NORM_FUNCTIONS = frozenset({"norm", "rnorm"})


def ffs64_argument(argument: str, hip_type: str | None) -> str:
    """Return ``argument`` as an operand that selects exactly one HIP ``__ffsll`` overload."""
    if hip_type in FFS64_ARGUMENT_TYPES:
        return argument
    return f"static_cast<unsigned long long>({argument})"


def generate_hip_library_call(codegen, func_name, raw_args, args):
    """Return the HIP call for ``func_name``, or ``None`` when the generic call is already correct.

    ``codegen`` is the HIP generator; ``raw_args`` are the argument nodes and ``args`` their generated code.
    """
    if func_name == "__ffsll" and len(args) == 1:
        return f"__ffsll({ffs64_argument(args[0], _hip_type(codegen, raw_args[0]))})"
    if func_name in _EUCLIDEAN_NORM_FUNCTIONS and len(args) == 2:
        name = _euclidean_norm_name(codegen, func_name, raw_args[1])
        return f"{name}({', '.join(args)})"
    return None


def _hip_type(codegen, node) -> str | None:
    type_name = codegen.type_name_string(codegen.expression_result_type(node))
    if not type_name:
        return None
    return codegen.map_type(type_name)


def _euclidean_norm_name(codegen, func_name, array_node) -> str:
    element_type = _array_element_type(codegen, array_node)
    if element_type == "float":
        return f"{func_name}f"
    if element_type == "double":
        return func_name
    elements = f"{element_type} elements" if element_type else "unresolved elements"
    raise HipLoweringRefused(
        [
            f"Euclidean norm: {func_name} of {elements}: HIP has normf and rnormf for "
            "float arrays and norm and rnorm for double arrays"
        ]
    )


def _array_element_type(codegen, array_node) -> str | None:
    """The HIP element type the pointer operand of ``norm``/``rnorm`` points at."""
    lvalue = codegen.strip_address_of_expression(array_node)
    if lvalue is not array_node:
        # ``&x`` points at ``x`` itself.
        return _hip_type(codegen, lvalue)

    type_name = codegen.type_name_string(codegen.expression_result_type(array_node))
    if not type_name:
        return None
    if type_name.endswith("]") and type_name.count("[") == 1:
        # A one-dimensional array decays to a pointer to its first element.
        return codegen.map_type(type_name[: type_name.index("[")].strip())
    pointer = codegen.hip_indirect_type_info(type_name)
    if pointer is not None and pointer["kind"] == "pointer":
        return codegen.map_type(pointer["pointee_type"])
    return None
